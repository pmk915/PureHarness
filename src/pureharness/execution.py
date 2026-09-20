import os
import subprocess
import uuid

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


DEFAULT_DOCKER_IMAGE = "python:3.12-bookworm"
_CONTAINER_WORKSPACE = PurePosixPath("/workspace")
_DEFAULT_DOCKER_MEMORY_LIMIT = "512m"
_DEFAULT_DOCKER_CPU_LIMIT = 1.0
_DEFAULT_DOCKER_PIDS_LIMIT = 128
_DOCKER_TMPFS = "/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777"
_DOCKER_CLEANUP_TIMEOUT_SECONDS = 5.0
_DOCKER_INFRASTRUCTURE_EXIT_CODES = {125, 126, 127}


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


class ExecutionError(Exception):
    """Raised when a command could not be executed to completion."""


class ExecutionTimeoutError(ExecutionError):
    """Raised when command execution exceeds its configured timeout."""


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
            raise ExecutionTimeoutError(
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


class DockerExecutionBackend:
    """Run commands in an ephemeral, constrained Docker container."""

    def __init__(
        self,
        workspace: str | Path,
        image: str = DEFAULT_DOCKER_IMAGE,
        *,
        memory_limit: str = _DEFAULT_DOCKER_MEMORY_LIMIT,
        cpu_limit: float = _DEFAULT_DOCKER_CPU_LIMIT,
        pids_limit: int = _DEFAULT_DOCKER_PIDS_LIMIT,
        user_id: int | None = None,
        group_id: int | None = None,
        docker_executable: str = "docker",
    ) -> None:
        workspace_path = Path(workspace).resolve()

        if not workspace_path.is_dir():
            raise ValueError(
                f"Docker workspace is not a directory: {workspace}"
            )

        if "," in str(workspace_path):
            raise ValueError(
                "Docker workspace path must not contain a comma."
            )

        if user_id is not None and (
            not isinstance(user_id, int)
            or isinstance(user_id, bool)
            or user_id <= 0
        ):
            raise ValueError(
                "Docker user ID must be a positive integer."
            )

        if group_id is not None and (
            not isinstance(group_id, int)
            or isinstance(group_id, bool)
            or group_id <= 0
        ):
            raise ValueError(
                "Docker group ID must be a positive integer."
            )

        host_user_id = (
            os.getuid()
            if hasattr(os, "getuid")
            else 65534
        )
        host_group_id = (
            os.getgid()
            if hasattr(os, "getgid")
            else 65534
        )
        resolved_user_id = (
            host_user_id
            if user_id is None
            else user_id
        )
        resolved_group_id = (
            host_group_id
            if group_id is None
            else group_id
        )

        if resolved_user_id <= 0:
            resolved_user_id = 65534

        if resolved_group_id <= 0:
            resolved_group_id = 65534

        if not image:
            raise ValueError("Docker image must not be empty.")

        if not memory_limit:
            raise ValueError(
                "Docker memory limit must not be empty."
            )

        if cpu_limit <= 0:
            raise ValueError(
                "Docker CPU limit must be positive."
            )

        if pids_limit <= 0:
            raise ValueError(
                "Docker PID limit must be positive."
            )

        if not docker_executable:
            raise ValueError(
                "Docker executable must not be empty."
            )

        self.workspace = workspace_path
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.user_id = resolved_user_id
        self.group_id = resolved_group_id
        self.docker_executable = docker_executable

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

        container_cwd = self._container_cwd(cwd)
        container_name = f"pureharness-{uuid.uuid4().hex}"
        docker_argv = self._docker_argv(
            argv,
            container_name=container_name,
            container_cwd=container_cwd,
        )

        try:
            completed = subprocess.run(
                docker_argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            cleanup_error = self._remove_container(
                container_name
            )
            detail = (
                f" Cleanup failed: {cleanup_error}"
                if cleanup_error
                else ""
            )
            raise ExecutionTimeoutError(
                "Docker command timed out after "
                f"{timeout} seconds.{detail}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ExecutionError(
                f"Could not start Docker execution: {exc}"
            ) from exc

        if (
            completed.returncode
            in _DOCKER_INFRASTRUCTURE_EXIT_CODES
            or (
                completed.returncode != 0
                and _looks_like_docker_infrastructure_failure(
                    completed.stderr
                )
            )
        ):
            detail = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "unknown Docker error"
            )
            raise ExecutionError(
                "Docker execution infrastructure failed "
                f"with exit code {completed.returncode}: {detail}"
            )

        return CommandResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def _container_cwd(
        self,
        cwd: Path,
    ) -> PurePosixPath:
        try:
            resolved_cwd = Path(cwd).resolve(strict=True)
        except OSError as exc:
            raise ExecutionError(
                f"Docker cwd cannot be resolved: {cwd}"
            ) from exc

        if not resolved_cwd.is_dir():
            raise ExecutionError(
                f"Docker cwd is not a directory: {cwd}"
            )

        if (
            resolved_cwd != self.workspace
            and self.workspace not in resolved_cwd.parents
        ):
            raise ExecutionError(
                f"Docker cwd escapes workspace: {cwd}"
            )

        relative = resolved_cwd.relative_to(
            self.workspace
        )

        if relative == Path("."):
            return _CONTAINER_WORKSPACE

        return _CONTAINER_WORKSPACE / PurePosixPath(
            relative.as_posix()
        )

    def _docker_argv(
        self,
        argv: list[str],
        *,
        container_name: str,
        container_cwd: PurePosixPath,
    ) -> list[str]:
        return [
            self.docker_executable,
            "run",
            "--rm",
            "--name",
            container_name,
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpu_limit),
            "--pids-limit",
            str(self.pids_limit),
            "--user",
            f"{self.user_id}:{self.group_id}",
            "--tmpfs",
            _DOCKER_TMPFS,
            "--mount",
            (
                f"type=bind,source={self.workspace},"
                f"target={_CONTAINER_WORKSPACE}"
            ),
            "--workdir",
            str(container_cwd),
            "--env",
            "HOME=/tmp",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "LANG=C.UTF-8",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            *argv,
        ]

    def _remove_container(
        self,
        container_name: str,
    ) -> str | None:
        try:
            completed = subprocess.run(
                [
                    self.docker_executable,
                    "rm",
                    "--force",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return str(exc)

        if completed.returncode == 0:
            return None

        detail = completed.stderr.strip()

        if "No such container" in detail:
            return None

        return detail or (
            "docker rm failed with exit code "
            f"{completed.returncode}"
        )


def _looks_like_docker_infrastructure_failure(
    stderr: str,
) -> bool:
    normalized = stderr.lower()
    markers = (
        "cannot connect to the docker daemon",
        "is the docker daemon running",
        "error response from daemon",
        "permission denied while trying to connect",
        "the command 'docker' could not be found",
    )

    return any(
        marker in normalized
        for marker in markers
    )
