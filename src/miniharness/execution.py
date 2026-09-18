import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


class ExecutionError(Exception):
    """Raised when a command could not be executed to completion."""


class ExecutionBackend(Protocol):
    def execute(
        self,
        argv: list[str],
        *,
        cwd: Path,
        timeout: float,
    ) -> CommandResult:
        ...


class LocalExecutionBackend:
    """Execute trusted commands with host privileges; this is not a sandbox."""

    def execute(
        self,
        argv: list[str],
        *,
        cwd: Path,
        timeout: float,
    ) -> CommandResult:
        if not argv:
            raise ExecutionError(
                "Command argv must not be empty."
            )

        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                f"Command timed out after {timeout} seconds: {argv[0]}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ExecutionError(
                f"Could not execute command {argv[0]!r}: {exc}"
            ) from exc

        return CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
