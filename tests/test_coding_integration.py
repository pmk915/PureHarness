import subprocess
import sys
import time

import pytest

from pureharness.agent import Agent
from pureharness.coding_tools import (
    create_coding_tools,
    create_process_tools,
)
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.processes import LocalProcessManager
from pureharness.run_record import RUN_RECORD_SCHEMA_VERSION
from pureharness.runtime import ExecutionBudget, ExecutionBudgetExceeded
from pureharness.tool_executor import ToolExecutor
from pureharness.tools import ToolRegistry
from pureharness.workspace_discipline import WorkspaceDiscipline


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, messages, tools):
        del messages, tools
        return next(self.outputs)


def _registry(tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_complete_coding_capability_workflow_without_model(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    target = package / "app.py"
    target.write_text(
        'GREETING = "old"\n\nprint(GREETING)\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "package/app.py"],
        cwd=tmp_path,
        check=True,
    )
    registry = _registry(create_coding_tools(tmp_path))
    executor = ToolExecutor(
        registry,
        precondition=WorkspaceDiscipline(tmp_path),
    )
    executor.reset_run_state()

    try:
        assert executor.execute(
            "find_files",
            {"pattern": "**/*.py"},
        ) == "package/app.py"
        assert "package/app.py:1" in executor.execute(
            "search_text",
            {
                "query": r"GREETING\s*=",
                "regex": True,
                "file_glob": "**/*.py",
            },
        )
        range_result = executor.execute(
            "read_file_range",
            {"path": "package/app.py", "max_lines": 2},
        )
        assert "package/app.py lines 1-2 of 3" in range_result

        full_result = executor.execute(
            "read_file",
            {"path": "package/app.py"},
        )
        assert 'GREETING = "old"' in full_result
        assert executor.execute(
            "apply_patch",
            {
                "path": "package/app.py",
                "old_text": 'GREETING = "old"',
                "new_text": 'GREETING = "new"',
            },
        ) == "Patched package/app.py"

        command_result = executor.execute(
            "run_command",
            {
                "argv": [sys.executable, "app.py"],
                "cwd": "package",
                "timeout_seconds": 5,
            },
        )
        assert "exit_code: 0" in str(command_result)
        assert "stdout:\nnew\n" in str(command_result)

        started = executor.execute(
            "start_process",
            {
                "argv": [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; print('ready', flush=True); time.sleep(30)",
                ],
                "cwd": "package",
            },
        )
        job_id = started.splitlines()[0].removeprefix("job_id: ")
        deadline = time.monotonic() + 5
        while True:
            observation = executor.execute(
                "poll_process",
                {"job_id": job_id},
            )
            if "ready" in observation:
                break
            assert "status: running" in observation
            if time.monotonic() >= deadline:
                raise AssertionError("background process produced no output")
            time.sleep(0.01)
        stopped = executor.execute(
            "stop_process",
            {"job_id": job_id},
        )
        assert "status: exited" in stopped
        assert "exit_code:" in stopped

        diff = executor.execute(
            "git_diff",
            {"path": "package/app.py"},
        )
        assert '-GREETING = "old"' in diff
        assert '+GREETING = "new"' in diff
    finally:
        executor.cleanup_run_state()


def test_process_tools_keep_m18_budget_progress_and_run_record_contract(
    tmp_path,
):
    manager = LocalProcessManager(
        tmp_path,
        job_id_factory=lambda: "job-1",
    )
    registry = _registry(
        create_process_tools(tmp_path, process_manager=manager)
    )
    calls = [
        ToolCall(
            name="start_process",
            arguments={
                "argv": [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; time.sleep(30)",
                ]
            },
            call_id="start",
        ),
        ToolCall(
            name="poll_process",
            arguments={"job_id": "job-1"},
            call_id="poll",
        ),
        ToolCall(
            name="stop_process",
            arguments={"job_id": "job-1"},
            call_id="stop",
        ),
    ]
    agent = Agent(
        model=ScriptedModel(
            [calls, Message(role="assistant", content="done")]
        ),
        tools=registry,
        max_steps=2,
        execution_budget=ExecutionBudget(max_tool_calls=3),
    )

    assert agent.run("manage a process") == "done"

    assert agent.execution_usage.tool_calls == 3
    assert agent.progress_snapshot.tool_calls == 3
    assert agent.progress_snapshot.successful_tool_results == 3
    record = agent.last_run_record
    assert record is not None
    assert RUN_RECORD_SCHEMA_VERSION == 2
    assert record.schema_version == 2
    assert record.tool_call_count == 3
    assert record.tool_execution_count == 3
    assert record.tool_result_error_count == 0
    persisted = record.to_dict()
    assert "processes" not in persisted
    assert "active_job_count" not in persisted


def test_process_tool_batch_respects_atomic_execution_budget(tmp_path):
    manager = LocalProcessManager(
        tmp_path,
        job_id_factory=lambda: "job-1",
    )
    registry = _registry(
        create_process_tools(tmp_path, process_manager=manager)
    )
    calls = [
        ToolCall(
            name="start_process",
            arguments={
                "argv": [sys.executable, "-c", "import time; time.sleep(30)"]
            },
            call_id="start",
        ),
        ToolCall(
            name="poll_process",
            arguments={"job_id": "job-1"},
            call_id="poll",
        ),
        ToolCall(
            name="stop_process",
            arguments={"job_id": "job-1"},
            call_id="stop",
        ),
    ]
    agent = Agent(
        model=ScriptedModel([calls]),
        tools=registry,
        max_steps=1,
        execution_budget=ExecutionBudget(max_tool_calls=2),
    )

    with pytest.raises(ExecutionBudgetExceeded) as exc_info:
        agent.run("manage a process")

    assert exc_info.value.resource == "tool_calls"
    assert exc_info.value.requested == 3
    assert agent.execution_usage.tool_calls == 0
    assert manager.active_job_count == 0


def test_process_tool_failure_remains_recoverable_tool_result(tmp_path):
    manager = LocalProcessManager(tmp_path)
    registry = _registry(
        create_process_tools(tmp_path, process_manager=manager)
    )
    agent = Agent(
        model=ScriptedModel(
            [
                [
                    ToolCall(
                        name="poll_process",
                        arguments={"job_id": "missing"},
                        call_id="poll",
                    )
                ],
                Message(role="assistant", content="recovered"),
            ]
        ),
        tools=registry,
        max_steps=2,
    )

    assert agent.run("poll a missing process") == "recovered"

    result = next(
        item for item in agent.messages if isinstance(item, ToolResult)
    )
    assert result.is_error is True
    assert "UnknownProcessError" in result.content
    assert agent.progress_snapshot.failed_tool_results == 1
    record = agent.last_run_record
    assert record is not None
    assert record.end_reason == "completed"
    assert record.tool_result_error_count == 1
