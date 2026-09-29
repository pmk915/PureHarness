import os
import signal
import subprocess
import tempfile
import uuid

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pureharness.workspace_discipline import resolve_workspace_path


_DEFAULT_MAX_ACTIVE_JOBS = 4
_PROCESS_STREAM_LIMIT_BYTES = 20_000
_PROCESS_STREAM_HEAD_BYTES = 10_000
_PROCESS_STREAM_TAIL_BYTES = 10_000
_TERMINATE_GRACE_SECONDS = 0.5


class ProcessError(Exception):
    """Base error for bounded background-process operations."""


class ProcessCapabilityUnavailableError(ProcessError):
    """The configured execution backend has no persistent-process support."""


class ProcessLimitError(ProcessError):
    """The Run already has the maximum number of active jobs."""


class UnknownProcessError(ProcessError):
    """A process operation referenced an unknown Run-scoped job ID."""


@dataclass(frozen=True)
class ProcessObservation:
    job_id: str
    status: Literal["running", "exited"]
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""


class ProcessManager(Protocol):
    def start(self, argv: list[str], *, cwd: Path) -> ProcessObservation:
        ...

    def poll(self, job_id: str) -> ProcessObservation:
        ...

    def stop(self, job_id: str) -> ProcessObservation:
        ...

    def reset_run_state(self) -> None:
        ...

    def cleanup_run_state(self) -> None:
        ...


@dataclass
class _LocalJob:
    job_id: str
    process: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path
    stdout_offset: int = 0
    stderr_offset: int = 0


class LocalProcessManager:
    """Run-scoped local background jobs without shell or PTY semantics."""

    def __init__(
        self,
        workspace: Path,
        *,
        max_active_jobs: int = _DEFAULT_MAX_ACTIVE_JOBS,
        job_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if (
            not isinstance(max_active_jobs, int)
            or isinstance(max_active_jobs, bool)
            or max_active_jobs <= 0
        ):
            raise ValueError("max_active_jobs must be a positive integer")
        self.workspace = workspace.resolve()
        self.max_active_jobs = max_active_jobs
        self._job_id_factory = job_id_factory or (
            lambda: str(uuid.uuid4())
        )
        self._jobs: dict[str, _LocalJob] = {}
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    @property
    def active_job_count(self) -> int:
        return sum(job.process.poll() is None for job in self._jobs.values())

    def start(self, argv: list[str], *, cwd: Path) -> ProcessObservation:
        if not isinstance(argv, list) or not argv:
            raise ProcessError("Process argv must be a non-empty list.")
        if any(not isinstance(value, str) for value in argv):
            raise ProcessError("Process argv must contain only strings.")
        if self.active_job_count >= self.max_active_jobs:
            raise ProcessLimitError(
                "Maximum active background jobs reached: "
                f"{self.max_active_jobs}."
            )
        process_cwd = self._resolve_cwd(cwd)
        job_id = self._job_id_factory()
        if not isinstance(job_id, str) or not job_id or job_id in self._jobs:
            raise ProcessError("Process job ID factory returned an invalid ID.")
        directory = self._ensure_temporary_directory()
        stdout_path = directory / f"{job_id}.stdout"
        stderr_path = directory / f"{job_id}.stderr"
        try:
            with stdout_path.open("wb") as stdout_file, stderr_path.open(
                "wb"
            ) as stderr_file:
                process = subprocess.Popen(
                    argv,
                    cwd=process_cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
        except (OSError, ValueError) as exc:
            raise ProcessError(
                f"Could not start background process {argv[0]!r}: {exc}"
            ) from exc
        job = _LocalJob(
            job_id=job_id,
            process=process,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        self._jobs[job_id] = job
        exit_code = process.poll()
        return ProcessObservation(
            job_id=job_id,
            status="running" if exit_code is None else "exited",
            exit_code=exit_code,
        )

    def poll(self, job_id: str) -> ProcessObservation:
        return self._observe(self._require_job(job_id))

    def stop(self, job_id: str) -> ProcessObservation:
        job = self._require_job(job_id)
        if job.process.poll() is None:
            self._terminate(job.process)
        return self._observe(job)

    def reset_run_state(self) -> None:
        self.cleanup_run_state()

    def cleanup_run_state(self) -> None:
        for job in self._jobs.values():
            if job.process.poll() is None:
                self._terminate(job.process)
        self._jobs.clear()
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def _resolve_cwd(self, cwd: Path) -> Path:
        resolved = resolve_workspace_path(self.workspace, str(cwd))
        if not resolved.is_dir():
            raise ProcessError(
                f"Process cwd is not a directory inside workspace: {cwd}"
            )
        return resolved

    def _ensure_temporary_directory(self) -> Path:
        if self._temporary_directory is None:
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="pureharness-processes-"
            )
        return Path(self._temporary_directory.name)

    def _require_job(self, job_id: str) -> _LocalJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise UnknownProcessError(
                f"Unknown background job ID: {job_id!r}."
            )
        return job

    def _observe(self, job: _LocalJob) -> ProcessObservation:
        exit_code = job.process.poll()
        stdout, job.stdout_offset = _read_incremental_output(
            job.stdout_path,
            job.stdout_offset,
            "stdout",
        )
        stderr, job.stderr_offset = _read_incremental_output(
            job.stderr_path,
            job.stderr_offset,
            "stderr",
        )
        return ProcessObservation(
            job_id=job.job_id,
            status="running" if exit_code is None else "exited",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

    def _terminate(self, process: subprocess.Popen[bytes]) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


class UnavailableProcessManager:
    """Explicit process capability failure for unsupported backends."""

    def start(self, argv: list[str], *, cwd: Path) -> ProcessObservation:
        del argv, cwd
        self._raise()

    def poll(self, job_id: str) -> ProcessObservation:
        del job_id
        self._raise()

    def stop(self, job_id: str) -> ProcessObservation:
        del job_id
        self._raise()

    def reset_run_state(self) -> None:
        return None

    def cleanup_run_state(self) -> None:
        return None

    def _raise(self) -> None:
        raise ProcessCapabilityUnavailableError(
            "Background process capability is unavailable for the configured "
            "execution backend."
        )


def _read_incremental_output(
    path: Path,
    offset: int,
    stream_name: str,
) -> tuple[str, int]:
    end = path.stat().st_size
    available = max(0, end - offset)
    with path.open("rb") as handle:
        handle.seek(offset)
        if available <= _PROCESS_STREAM_LIMIT_BYTES:
            data = handle.read(available)
            return data.decode("utf-8", errors="replace"), end

        head = handle.read(_PROCESS_STREAM_HEAD_BYTES)
        handle.seek(end - _PROCESS_STREAM_TAIL_BYTES)
        tail = handle.read(_PROCESS_STREAM_TAIL_BYTES)
    omitted = available - (
        _PROCESS_STREAM_HEAD_BYTES + _PROCESS_STREAM_TAIL_BYTES
    )
    notice = (
        f"\n... {stream_name} truncated; new byte count: {available}; "
        f"omitted: {omitted} ...\n"
    )
    return (
        head.decode("utf-8", errors="replace")
        + notice
        + tail.decode("utf-8", errors="replace"),
        end,
    )
