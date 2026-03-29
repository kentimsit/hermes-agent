"""Base class for all Hermes execution environment backends."""

from abc import ABC, abstractmethod
import os
import shlex
import subprocess
from pathlib import Path

from hermes_cli.config import get_hermes_home, load_env


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


def _get_passthrough_exports() -> str:
    """Build a shell snippet that exports active passthrough env vars.

    Values come from the live process environment first, then fall back to
    ``HERMES_HOME/.env`` so remote-backed shells can see skill-declared vars
    even when Hermes intentionally filters them out of child process envs.
    """
    try:
        from tools.env_passthrough import get_all_passthrough
    except Exception:
        return ""

    passthrough_names = sorted(get_all_passthrough())
    if not passthrough_names:
        return ""

    env_snapshot: dict[str, str] | None = None
    export_parts: list[str] = []
    for name in passthrough_names:
        value = os.getenv(name)
        if value is None:
            if env_snapshot is None:
                env_snapshot = load_env()
            value = env_snapshot.get(name)
        if value is None:
            continue
        export_parts.append(f"export {name}={shlex.quote(value)}")

    return "; ".join(export_parts)


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
        transformed_command, sudo_stdin = _transform_sudo_command(command)
        passthrough_exports = _get_passthrough_exports()
        if not passthrough_exports:
            return transformed_command, sudo_stdin

        # Wrap in a subshell so exported vars apply to the whole command
        # without polluting long-lived persistent shells between invocations.
        #
        # Multiline commands need the closing parenthesis on its own line so
        # shell heredoc terminators remain bare terminator lines.
        if "\n" in transformed_command:
            wrapped_command = (
                "(\n"
                f"{passthrough_exports}\n"
                f"{transformed_command}\n"
                ")"
            )
        else:
            wrapped_command = f"({passthrough_exports}; {transformed_command})"
        return wrapped_command, sudo_stdin

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
