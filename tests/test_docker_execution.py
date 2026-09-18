import json
import subprocess
import uuid

from pathlib import Path

import pytest

import miniharness.execution as execution_module
from miniharness.coding_tools import create_coding_tools
from miniharness.execution import (
    DEFAULT_DOCKER_IMAGE,
    DockerExecutionBackend,
    ExecutionError,
)


def _backend(workspace: Path) -> DockerExecutionBackend:
    return DockerExecutionBackend(
        workspace,
        user_id=1000,
        group_id=1000,
    )


def test_docker_backend_builds_constrained_invocation(
    tmp_path,
    monkeypatch,
):
    nested = tmp_path / "nested"
    nested.mkdir()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="application output",
            stderr="application error",
        )

    monkeypatch.setenv(
        "MINIHARNESS_TEST_SECRET",
        "do-not-expose",
    )
    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        fake_run,
    )
    backend = DockerExecutionBackend(
        tmp_path,
        memory_limit="768m",
        cpu_limit=1.5,
        pids_limit=64,
        user_id=1234,
        group_id=5678,
    )

    result = backend.execute(
        ["python", "-c", "print('hello')"],
        cwd=nested,
        timeout=7.0,
    )

    assert result.exit_code == 0
    assert result.stdout == "application output"
    assert result.stderr == "application error"
    assert len(calls) == 1

    docker_argv, kwargs = calls[0]
    assert docker_argv[:3] == ["docker", "run", "--rm"]
    assert "--pull" in docker_argv
    assert docker_argv[docker_argv.index("--pull") + 1] == "never"
    assert docker_argv[docker_argv.index("--network") + 1] == "none"
    assert "--read-only" in docker_argv
    assert docker_argv[docker_argv.index("--cap-drop") + 1] == "ALL"
    assert (
        docker_argv[docker_argv.index("--security-opt") + 1]
        == "no-new-privileges:true"
    )
    assert docker_argv[docker_argv.index("--memory") + 1] == "768m"
    assert docker_argv[docker_argv.index("--cpus") + 1] == "1.5"
    assert docker_argv[docker_argv.index("--pids-limit") + 1] == "64"
    assert docker_argv[docker_argv.index("--user") + 1] == "1234:5678"
    assert (
        docker_argv[docker_argv.index("--tmpfs") + 1]
        == "/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777"
    )
    assert docker_argv.count("--mount") == 1
    assert docker_argv[docker_argv.index("--mount") + 1] == (
        f"type=bind,source={tmp_path.resolve()},target=/workspace"
    )
    assert (
        docker_argv[docker_argv.index("--workdir") + 1]
        == "/workspace/nested"
    )
    assert "HOME=/tmp" in docker_argv
    assert "TMPDIR=/tmp" in docker_argv
    assert "LANG=C.UTF-8" in docker_argv
    assert "PYTHONDONTWRITEBYTECODE=1" in docker_argv
    assert "do-not-expose" not in docker_argv
    assert "/var/run/docker.sock" not in " ".join(docker_argv)
    assert "--privileged" not in docker_argv
    assert "--network=host" not in docker_argv
    assert docker_argv[-4:] == [
        DEFAULT_DOCKER_IMAGE,
        "python",
        "-c",
        "print('hello')",
    ]
    assert kwargs == {
        "capture_output": True,
        "text": True,
        "timeout": 7.0,
        "check": False,
    }


def test_docker_backend_maps_workspace_root(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        fake_run,
    )

    _backend(tmp_path).execute(
        ["python", "--version"],
        cwd=tmp_path,
        timeout=5.0,
    )

    docker_argv = calls[0]
    assert (
        docker_argv[docker_argv.index("--workdir") + 1]
        == "/workspace"
    )


def test_docker_backend_resolves_workspace_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(
        workspace,
        target_is_directory=True,
    )

    backend = _backend(workspace_link)

    assert backend.workspace == workspace.resolve()


def test_docker_backend_never_uses_root_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(execution_module.os, "getuid", lambda: 0)
    monkeypatch.setattr(execution_module.os, "getgid", lambda: 0)

    backend = DockerExecutionBackend(tmp_path)

    assert backend.user_id == 65534
    assert backend.group_id == 65534


@pytest.mark.parametrize("escape_kind", ["outside", "symlink"])
def test_docker_backend_rejects_cwd_escape(
    tmp_path,
    escape_kind,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    if escape_kind == "symlink":
        cwd = workspace / "escape"
        cwd.symlink_to(
            outside,
            target_is_directory=True,
        )
    else:
        cwd = outside

    with pytest.raises(
        ExecutionError,
        match="Docker cwd escapes workspace",
    ):
        _backend(workspace).execute(
            ["python", "--version"],
            cwd=cwd,
            timeout=5.0,
        )


def test_docker_backend_returns_normal_nonzero_result(
    tmp_path,
    monkeypatch,
):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="test output",
            stderr="test failure",
        )

    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        fake_run,
    )

    result = _backend(tmp_path).execute(
        ["python", "-m", "pytest"],
        cwd=tmp_path,
        timeout=5.0,
    )

    assert result.exit_code == 1
    assert result.stdout == "test output"
    assert result.stderr == "test failure"


def test_docker_backend_preserves_successful_application_stderr(
    tmp_path,
    monkeypatch,
):
    application_stderr = "Error response from daemon is application text"

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="",
            stderr=application_stderr,
        )

    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        fake_run,
    )

    result = _backend(tmp_path).execute(
        ["python", "application.py"],
        cwd=tmp_path,
        timeout=5.0,
    )

    assert result.exit_code == 0
    assert result.stderr == application_stderr


@pytest.mark.parametrize(
    "return_code, stderr",
    [
        (125, "docker: Error response from daemon: image missing"),
        (1, "Cannot connect to the Docker daemon"),
    ],
)
def test_docker_backend_reports_infrastructure_failure(
    tmp_path,
    monkeypatch,
    return_code,
    stderr,
):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            return_code,
            stdout="",
            stderr=stderr,
        )

    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        ExecutionError,
        match="Docker execution infrastructure failed",
    ):
        _backend(tmp_path).execute(
            ["python", "--version"],
            cwd=tmp_path,
            timeout=5.0,
        )


def test_docker_backend_reports_missing_cli(
    tmp_path,
    monkeypatch,
):
    def missing_cli(argv, **kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        missing_cli,
    )

    with pytest.raises(
        ExecutionError,
        match="Could not start Docker execution",
    ):
        _backend(tmp_path).execute(
            ["python", "--version"],
            cwd=tmp_path,
            timeout=5.0,
        )


def test_docker_backend_timeout_forces_container_cleanup(
    tmp_path,
    monkeypatch,
):
    calls = []

    class FixedUuid:
        hex = "fixed-container-id"

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))

        if argv[1] == "run":
            raise subprocess.TimeoutExpired(
                argv,
                kwargs["timeout"],
            )

        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(
        execution_module.uuid,
        "uuid4",
        lambda: FixedUuid(),
    )
    monkeypatch.setattr(
        execution_module.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        ExecutionError,
        match="Docker command timed out",
    ):
        _backend(tmp_path).execute(
            ["python", "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            timeout=0.01,
        )

    assert len(calls) == 2
    assert calls[1][0] == [
        "docker",
        "rm",
        "--force",
        "miniharness-fixed-container-id",
    ]


def _docker_integration_skip_reason() -> str | None:
    try:
        version = subprocess.run(
            [
                "docker",
                "version",
                "--format",
                "{{.Server.Version}}",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Docker is unavailable: {exc}"

    if version.returncode != 0:
        return "Docker daemon is unavailable"

    try:
        image = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                DEFAULT_DOCKER_IMAGE,
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Docker image inspection is unavailable: {exc}"

    if image.returncode != 0:
        return (
            "Docker integration image is not installed locally: "
            f"{DEFAULT_DOCKER_IMAGE}"
        )

    return None


@pytest.fixture(scope="module")
def docker_integration_image():
    skip_reason = _docker_integration_skip_reason()

    if skip_reason is not None:
        pytest.skip(skip_reason)

    return DEFAULT_DOCKER_IMAGE


def test_docker_integration_security_and_workspace(
    tmp_path,
    monkeypatch,
    docker_integration_image,
):
    (tmp_path / "visible.txt").write_text(
        "visible",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MINIHARNESS_TEST_SECRET",
        "do-not-expose",
    )
    script = """
import json
import os
import socket
from pathlib import Path

root_write_failed = False
try:
    Path("/miniharness-root-write-test").write_text("blocked")
except OSError:
    root_write_failed = True

Path("/tmp/miniharness-test").write_text("temporary")
Path("workspace-output.txt").write_text("from container")

print(json.dumps({
    "uid": os.geteuid(),
    "secret": os.environ.get("MINIHARNESS_TEST_SECRET"),
    "interfaces": [name for _, name in socket.if_nameindex()],
    "workspace_input": Path("visible.txt").read_text(),
    "root_write_failed": root_write_failed,
    "tmp_writable": Path("/tmp/miniharness-test").read_text() == "temporary",
}))
"""
    backend = DockerExecutionBackend(
        tmp_path,
        image=docker_integration_image,
    )

    result = backend.execute(
        ["python", "-c", script],
        cwd=tmp_path,
        timeout=10.0,
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["uid"] != 0
    assert data["secret"] is None
    assert data["interfaces"] == ["lo"]
    assert data["workspace_input"] == "visible"
    assert data["root_write_failed"] is True
    assert data["tmp_writable"] is True
    assert (tmp_path / "workspace-output.txt").read_text(
        encoding="utf-8"
    ) == "from container"


def test_docker_integration_git_tools(
    tmp_path,
    docker_integration_image,
):
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
    )
    target = tmp_path / "example.py"
    target.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "example.py"],
        cwd=tmp_path,
        check=True,
    )
    target.write_text("value = 2\n", encoding="utf-8")
    backend = DockerExecutionBackend(
        tmp_path,
        image=docker_integration_image,
    )
    tools = {
        tool.name: tool
        for tool in create_coding_tools(
            tmp_path,
            execution_backend=backend,
        )
    }

    status = tools["git_status"].execute({})
    diff = tools["git_diff"].execute(
        {"path": "example.py"}
    )

    assert "example.py" in status
    assert "-value = 1" in diff
    assert "+value = 2" in diff


def test_docker_integration_timeout_cleanup(
    tmp_path,
    monkeypatch,
    docker_integration_image,
):
    container_suffix = uuid.uuid4().hex

    class FixedUuid:
        hex = container_suffix

    monkeypatch.setattr(
        execution_module.uuid,
        "uuid4",
        lambda: FixedUuid(),
    )
    backend = DockerExecutionBackend(
        tmp_path,
        image=docker_integration_image,
    )

    with pytest.raises(
        ExecutionError,
        match="Docker command timed out",
    ):
        backend.execute(
            ["python", "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            timeout=0.5,
        )

    inspect = subprocess.run(
        [
            "docker",
            "container",
            "inspect",
            f"miniharness-{container_suffix}",
        ],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )
    assert inspect.returncode != 0
