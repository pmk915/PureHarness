import sys

import pytest

from miniharness.execution import (
    ExecutionError,
    ExecutionTimeoutError,
    LocalExecutionBackend,
)


def test_local_execution_backend_captures_output(
    tmp_path,
):
    backend = LocalExecutionBackend()

    result = backend.execute(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('standard output'); "
                "print('standard error', file=sys.stderr)"
            ),
        ],
        cwd=tmp_path,
        timeout=5.0,
    )

    assert result.exit_code == 0
    assert result.stdout == "standard output\n"
    assert result.stderr == "standard error\n"


def test_local_execution_backend_returns_nonzero_exit(
    tmp_path,
):
    backend = LocalExecutionBackend()

    result = backend.execute(
        [sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
        timeout=5.0,
    )

    assert result.exit_code == 7
    assert result.stdout == ""


def test_local_execution_backend_uses_supplied_cwd(
    tmp_path,
):
    backend = LocalExecutionBackend()

    result = backend.execute(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; print(Path.cwd())",
        ],
        cwd=tmp_path,
        timeout=5.0,
    )

    assert result.stdout.strip() == str(tmp_path.resolve())


def test_local_execution_backend_timeout_is_distinct_execution_error(
    tmp_path,
):
    backend = LocalExecutionBackend()

    with pytest.raises(
        ExecutionTimeoutError,
        match="Command timed out",
    ):
        backend.execute(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(5)",
            ],
            cwd=tmp_path,
            timeout=0.01,
        )


def test_local_execution_backend_missing_executable_is_execution_error(
    tmp_path,
):
    backend = LocalExecutionBackend()

    with pytest.raises(
        ExecutionError,
        match="Could not execute command",
    ):
        backend.execute(
            ["miniharness-definitely-missing-executable"],
            cwd=tmp_path,
            timeout=5.0,
        )
