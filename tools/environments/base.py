"""Base class for all Hermes execution environment backends."""

from abc import ABC, abstractmethod
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from hermes_cli.config import get_hermes_home

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_sandbox_dir() -> Path:
    """Return the host-side root for all sandbox storage (Docker workspaces,
    Singularity overlays/SIF cache, etc.).

    Configurable via TERMINAL_SANDBOX_DIR. Defaults to {HERMES_HOME}/sandboxes/.
    """
    custom = os.getenv("TERMINAL_SANDBOX_DIR")
    if custom:
        p = Path(custom)
    else:
        p = get_hermes_home() / "sandboxes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_passthrough_env_vars(
    extra_names: Iterable[str] | None = None,
) -> dict[str, str]:
    """Resolve active passthrough env vars from the shell or Hermes .env."""
    try:
        from tools.env_passthrough import get_all_passthrough

        names = set(get_all_passthrough())
    except Exception:
        names = set()

    for item in extra_names or ():
        if isinstance(item, str):
            names.add(item)

    if not names:
        return {}

    try:
        from hermes_cli.config import load_env

        env_snapshot = load_env() or {}
    except Exception:
        env_snapshot = {}

    resolved: dict[str, str] = {}
    for raw_name in sorted(names):
        name = raw_name.strip()
        if not name or not _ENV_VAR_NAME_RE.match(name):
            continue

        value = os.getenv(name)
        if value is None:
            value = env_snapshot.get(name)
        if value is not None:
            resolved[name] = value

    return resolved


def _prepend_env_exports(command: str, env_vars: dict[str, str]) -> str:
    """Prefix shell-safe export statements ahead of *command*."""
    if not env_vars:
        return command

    exports = "; ".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(env_vars.items())
    )
    return f"{exports}; {command}"


class BaseEnvironment(ABC):
    """Common interface for all Hermes execution backends.

    Subclasses implement execute() and cleanup(). Shared helpers eliminate
    duplicated subprocess boilerplate across backends.
    """

    def __init__(self, cwd: str, timeout: int, env: dict = None):
        self.cwd = cwd
        self.timeout = timeout
        self.env = env or {}

    @abstractmethod
    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        """Execute a command, return {"output": str, "returncode": int}."""
        ...

    @abstractmethod
    def cleanup(self):
        """Release backend resources (container, instance, connection)."""
        ...

    def stop(self):
        """Alias for cleanup (compat with older callers)."""
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shared helpers (eliminate duplication across backends)
    # ------------------------------------------------------------------

    def _prepare_command(self, command: str) -> tuple[str, str | None]:
        """Transform sudo commands if SUDO_PASSWORD is available.

        Returns:
            (transformed_command, sudo_stdin) — see _transform_sudo_command
            for the full contract.  Callers that drive a subprocess directly
            should prepend sudo_stdin (when not None) to any stdin_data they
            pass to Popen.  Callers that embed stdin via heredoc (modal,
            daytona) handle sudo_stdin in their own execute() method.
        """
        from tools.terminal_tool import _transform_sudo_command
        return _transform_sudo_command(command)

    def _resolve_passthrough_env(
        self, extra_names: Iterable[str] | None = None
    ) -> dict[str, str]:
        """Resolve passthrough env vars visible to this backend."""
        return _resolve_passthrough_env_vars(extra_names)

    def _prepend_env_exports(self, command: str, env_vars: dict[str, str]) -> str:
        """Prefix export statements for backends that run through a shell."""
        return _prepend_env_exports(command, env_vars)

    def _build_run_kwargs(self, timeout: int | None,
                          stdin_data: str | None = None) -> dict:
        """Build common subprocess.run kwargs for non-interactive execution."""
        kw = {
            "text": True,
            "timeout": timeout or self.timeout,
            "encoding": "utf-8",
            "errors": "replace",
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if stdin_data is not None:
            kw["input"] = stdin_data
        else:
            kw["stdin"] = subprocess.DEVNULL
        return kw

    def _timeout_result(self, timeout: int | None) -> dict:
        """Standard return dict when a command times out."""
        return {
            "output": f"Command timed out after {timeout or self.timeout}s",
            "returncode": 124,
        }
