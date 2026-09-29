import os
import signal
import sys
import time

import pytest

from pureharness.agent import Agent
from pureharness.coding_tools import create_process_tools
from pureharness.messages import Message, ToolCall
from pureharness.model import ModelError
from pureharness.processes import (
    LocalProcessManager,
    ProcessCapabilityUnavailableError,
    ProcessError,
    ProcessLimitError,
    UnknownProcessError,
    UnavailableProcessManager,
)
from pureharness.tools import ToolRegistry


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, messages, tools):
        del messages, tools
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return output


def _python_process(source: str) -> list[str]:
    return [sys.executable, "-u", "-c", source]


def _poll_until_exited(
    manager: LocalProcessManager,
    job_id: str,
):
    deadline = time.monotonic() + 5
    stdout = ""
    stderr = ""
    while True:
        observation = manager.poll(job_id)
        stdout += observation.stdout
        stderr += observation.stderr
        if observation.status == "exited":
            return observation, stdout, stderr
        if time.monotonic() >= deadline:
            pytest.fail("background process did not exit")
        time.sleep(0.01)


def test_local_process_manager_polls_incremental_output_and_normal_exit(
    tmp_path,
):
    working_directory = tmp_path / "service"
    working_directory.mkdir()
    manager = LocalProcessManager(
        tmp_path,
        job_id_factory=lambda: "job-1",
    )

    started = manager.start(
        _python_process(
            "import pathlib, time\n"
            "print('first', flush=True)\n"
            "while not pathlib.Path('release').exists(): time.sleep(0.01)\n"
            "print('second', flush=True)\n"
        ),
        cwd=working_directory,
    )

    assert started.job_id == "job-1"
    assert started.status == "running"
    deadline = time.monotonic() + 5
    while True:
        running = manager.poll("job-1")
        if running.stdout:
            break
        if time.monotonic() >= deadline:
            pytest.fail("background output was not observed")
        time.sleep(0.01)
    assert running.status == "running"
    assert running.stdout == "first\n"

    (working_directory / "release").touch()
    exited, stdout, stderr = _poll_until_exited(manager, "job-1")

    assert exited.exit_code == 0
    assert stdout == "second\n"
    assert stderr == ""
    assert manager.poll("job-1").stdout == ""
    manager.cleanup_run_state()


def test_local_process_manager_stops_running_process(tmp_path):
    manager = LocalProcessManager(tmp_path)
    started = manager.start(
        _python_process("import time; time.sleep(30)"),
        cwd=tmp_path,
    )

    stopped = manager.stop(started.job_id)

    assert stopped.status == "exited"
    assert stopped.exit_code is not None
    assert manager.active_job_count == 0
    manager.cleanup_run_state()


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal behavior")
def test_local_process_manager_force_kills_process_if_needed(tmp_path):
    manager = LocalProcessManager(tmp_path)
    started = manager.start(
        _python_process(
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "print('ready', flush=True)\n"
            "time.sleep(30)\n"
        ),
        cwd=tmp_path,
    )
    deadline = time.monotonic() + 5
    while not manager.poll(started.job_id).stdout:
        if time.monotonic() >= deadline:
            pytest.fail("background process did not become ready")
        time.sleep(0.01)

    stopped = manager.stop(started.job_id)

    assert stopped.status == "exited"
    assert stopped.exit_code == -signal.SIGKILL
    manager.cleanup_run_state()


def test_local_process_manager_rejects_unknown_job_and_active_limit(tmp_path):
    manager = LocalProcessManager(tmp_path, max_active_jobs=1)

    with pytest.raises(UnknownProcessError, match="missing"):
        manager.poll("missing")

    manager.start(
        _python_process("import time; time.sleep(30)"),
        cwd=tmp_path,
    )
    with pytest.raises(ProcessLimitError, match="Maximum active"):
        manager.start(
            _python_process("import time; time.sleep(30)"),
            cwd=tmp_path,
        )
    manager.cleanup_run_state()


def test_local_process_manager_enforces_workspace_cwd(tmp_path):
    manager = LocalProcessManager(tmp_path)

    with pytest.raises(ValueError, match="escapes workspace"):
        manager.start(
            _python_process("print('no')"),
            cwd=tmp_path.parent,
        )
    with pytest.raises(ProcessError, match="not a directory"):
        manager.start(
            _python_process("print('no')"),
            cwd=tmp_path / "missing",
        )


def test_local_process_manager_bounds_each_incremental_stream(tmp_path):
    manager = LocalProcessManager(tmp_path)
    started = manager.start(
        _python_process(
            "import sys\n"
            "sys.stdout.write('a' * 25000)\n"
            "sys.stderr.write('b' * 25000)\n"
        ),
        cwd=tmp_path,
    )

    _, stdout, stderr = _poll_until_exited(manager, started.job_id)

    assert len(stdout) < 21_000
    assert "stdout truncated; new byte count: 25000; omitted: 5000" in stdout
    assert stdout.startswith("a" * 10_000)
    assert stdout.endswith("a" * 10_000)
    assert len(stderr) < 21_000
    assert "stderr truncated; new byte count: 25000; omitted: 5000" in stderr
    manager.cleanup_run_state()


def test_unavailable_process_manager_reports_capability_explicitly(tmp_path):
    manager = UnavailableProcessManager()
    tools = {
        tool.name: tool
        for tool in create_process_tools(
            tmp_path,
            process_manager=manager,
        )
    }

    with pytest.raises(ProcessCapabilityUnavailableError, match="unavailable"):
        tools["start_process"].execute({"argv": ["example"]})


@pytest.mark.parametrize(
    "terminal_output, expected_exception",
    [
        (Message(role="assistant", content="done"), None),
        (ModelError("provider failed"), ModelError),
        (KeyboardInterrupt(), KeyboardInterrupt),
    ],
)
def test_agent_cleans_background_jobs_on_every_run_exit(
    tmp_path,
    terminal_output,
    expected_exception,
):
    manager = LocalProcessManager(tmp_path)
    registry = ToolRegistry()
    for tool in create_process_tools(tmp_path, process_manager=manager):
        registry.register(tool)
    model = ScriptedModel(
        [
            [
                ToolCall(
                    name="start_process",
                    arguments={
                        "argv": _python_process(
                            "import time; time.sleep(30)"
                        )
                    },
                    call_id="start",
                )
            ],
            terminal_output,
        ]
    )
    agent = Agent(model=model, tools=registry, max_steps=2)

    if expected_exception is None:
        assert agent.run("start a service") == "done"
    else:
        with pytest.raises(expected_exception):
            agent.run("start a service")

    assert manager.active_job_count == 0
