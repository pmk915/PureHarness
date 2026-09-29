import json
import subprocess

import pytest

from pureharness.agent import Agent
from pureharness.approval import ApprovalDecision
from pureharness.benchmark import BENCHMARK_RESULT_SCHEMA_VERSION
from pureharness.coding_tools import create_coding_tools
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.model import AddModel
from pureharness.observability import (
    EVENT_WIRE_SCHEMA_VERSION,
    JsonlEventRenderer,
)
from pureharness.run_record import RUN_RECORD_SCHEMA_VERSION
from pureharness.runtime import ExecutionBudget
from pureharness.task_state import TaskStateReducer
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_policy import PolicyDecision
from pureharness.tools import ADD_TOOL, ToolRegistry
from pureharness.workspace_discipline import WorkspaceDiscipline


class ScriptedModel:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)

    def generate(self, messages, tools):
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class RecordingPolicy:
    def __init__(
        self,
        decision: PolicyDecision = PolicyDecision.ALLOW,
    ) -> None:
        self.decision = decision
        self.calls: list[str] = []

    def evaluate(self, tool, arguments):
        self.calls.append(tool.name)
        return self.decision


class RecordingApprovalHandler:
    def __init__(self) -> None:
        self.requests = []

    def request_approval(self, request):
        self.requests.append(request)
        return ApprovalDecision.APPROVE


def _call(name: str, arguments: dict[str, object], call_id: str) -> ToolCall:
    return ToolCall(name=name, arguments=arguments, call_id=call_id)


def _read(path: str, call_id: str = "read") -> ToolCall:
    return _call("read_file", {"path": path}, call_id)


def _write(
    path: str,
    content: str,
    call_id: str = "write",
) -> ToolCall:
    return _call(
        "write_file",
        {"path": path, "content": content},
        call_id,
    )


def _patch(
    path: str,
    old_text: str,
    new_text: str,
    call_id: str = "patch",
) -> ToolCall:
    return _call(
        "apply_patch",
        {
            "path": path,
            "old_text": old_text,
            "new_text": new_text,
        },
        call_id,
    )


def _agent(
    workspace,
    outputs,
    *,
    policy=None,
    approval_handler=None,
    execution_budget=None,
    listeners=None,
) -> tuple[Agent, WorkspaceDiscipline]:
    registry = ToolRegistry()
    for tool in create_coding_tools(workspace):
        registry.register(tool)
    discipline = WorkspaceDiscipline(workspace)
    executor = ToolExecutor(
        registry,
        policy=policy,
        approval_handler=approval_handler,
        precondition=discipline,
    )
    return (
        Agent(
            model=ScriptedModel(outputs),
            tools=registry,
            tool_executor=executor,
            max_steps=len(outputs),
            execution_budget=execution_budget,
            listeners=listeners,
            run_id_factory=lambda: "workspace-run",
        ),
        discipline,
    )


@pytest.mark.parametrize(
    "blocked_call",
    [
        _patch("a.py", "before", "after"),
        _write("a.py", "after"),
    ],
    ids=["apply_patch", "write_file"],
)
def test_existing_mutation_without_read_is_recoverable_before_authorization(
    tmp_path,
    blocked_call,
):
    target = tmp_path / "a.py"
    target.write_text("before", encoding="utf-8")
    policy = RecordingPolicy(PolicyDecision.REQUIRE_APPROVAL)
    approvals = RecordingApprovalHandler()
    agent, discipline = _agent(
        tmp_path,
        [[blocked_call], Message(role="assistant", content="recovered")],
        policy=policy,
        approval_handler=approvals,
    )

    assert agent.run("edit") == "recovered"

    assert target.read_text(encoding="utf-8") == "before"
    assert discipline.observed_paths == ()
    assert policy.calls == []
    assert approvals.requests == []
    assert agent.trace.approvals == []
    assert all(event.type != "tool_started" for event in agent.events)
    result = next(
        item for item in agent.messages if isinstance(item, ToolResult)
    )
    assert result.is_error is True
    assert "ReadBeforeEditError" in result.content
    assert "must be read successfully" in result.content
    blocked = next(
        event
        for event in agent.events
        if event.type == "workspace_precondition_failed"
    )
    assert blocked.data == {
        "step": 0,
        "name": blocked_call.name,
        "call_id": blocked_call.call_id,
        "path": "a.py",
        "reason": "read_required",
    }
    assert agent.last_run_record is not None
    assert agent.last_run_record.tool_execution_count == 0


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (_patch("a.py", "before", "after"), "after"),
        (_write("a.py", "replacement"), "replacement"),
    ],
    ids=["patch", "full_write"],
)
def test_successful_read_authorizes_existing_mutation_in_same_batch(
    tmp_path,
    mutation,
    expected,
):
    target = tmp_path / "a.py"
    target.write_text("before", encoding="utf-8")
    agent, discipline = _agent(
        tmp_path,
        [[_read("a.py"), mutation], Message(role="assistant", content="done")],
    )

    assert agent.run("edit") == "done"

    assert target.read_text(encoding="utf-8") == expected
    assert discipline.observed_paths == ("a.py",)
    results = [
        item for item in agent.messages if isinstance(item, ToolResult)
    ]
    assert [result.is_error for result in results] == [False, False]


def test_new_file_creation_becomes_known_for_later_mutations(tmp_path):
    agent, discipline = _agent(
        tmp_path,
        [
            [
                _write("new.py", "value = 1\n", "create"),
                _patch(
                    "new.py",
                    "value = 1",
                    "value = 2",
                    "patch-new",
                ),
                _write("new.py", "value = 3\n", "rewrite-new"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    assert agent.run("create") == "done"

    assert (tmp_path / "new.py").read_text(encoding="utf-8") == (
        "value = 3\n"
    )
    assert discipline.observed_paths == ("new.py",)
    assert agent.progress_snapshot.successful_tool_results == 3


def test_successful_mutation_keeps_existing_file_known(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("value = 1\n", encoding="utf-8")
    agent, discipline = _agent(
        tmp_path,
        [
            [
                _read("a.py"),
                _patch("a.py", "value = 1", "value = 2", "patch-1"),
                _patch("a.py", "value = 2", "value = 3", "patch-2"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("edit twice")

    assert target.read_text(encoding="utf-8") == "value = 3\n"
    assert discipline.observed_paths == ("a.py",)


def test_observation_resets_between_runs_despite_session_history(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("before", encoding="utf-8")
    model = ScriptedModel(
        [
            [_read("a.py", "run-1-read")],
            Message(role="assistant", content="first"),
            [_patch("a.py", "before", "after", "run-2-patch")],
            Message(role="assistant", content="second"),
        ]
    )
    registry = ToolRegistry()
    for tool in create_coding_tools(tmp_path):
        registry.register(tool)
    discipline = WorkspaceDiscipline(tmp_path)
    agent = Agent(
        model=model,
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            precondition=discipline,
        ),
        max_steps=2,
    )

    assert agent.run("read") == "first"
    assert discipline.observed_paths == ("a.py",)
    assert agent.run("edit") == "second"

    assert target.read_text(encoding="utf-8") == "before"
    assert discipline.observed_paths == ()
    result = [
        item for item in agent.messages if isinstance(item, ToolResult)
    ][-1]
    assert result.is_error is True
    assert "ReadBeforeEditError" in result.content


def test_canonical_aliases_share_identity_and_escape_stays_rejected(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "foo").mkdir()
    target = source / "a.py"
    target.write_text("before", encoding="utf-8")
    agent, discipline = _agent(
        tmp_path,
        [
            [
                _read("src/./a.py"),
                _patch("src/foo/../a.py", "before", "after"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("edit")

    assert target.read_text(encoding="utf-8") == "after"
    assert discipline.observed_paths == ("src/a.py",)
    assert discipline.is_observed("src/foo/../a.py") is True
    with pytest.raises(ValueError, match="Path escapes workspace"):
        discipline.is_observed("../outside.py")


def test_failed_read_does_not_authorize_existing_write(tmp_path):
    target = tmp_path / "invalid.py"
    target.write_bytes(b"\xff")
    agent, discipline = _agent(
        tmp_path,
        [
            [
                _read("invalid.py"),
                _write("invalid.py", "replacement"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("edit")

    assert target.read_bytes() == b"\xff"
    assert discipline.observed_paths == ()
    results = [
        item for item in agent.messages if isinstance(item, ToolResult)
    ]
    assert len(results) == 2
    assert all(result.is_error for result in results)
    assert "UnicodeDecodeError" in results[0].content
    assert "ReadBeforeEditError" in results[1].content


def test_partial_workspace_views_do_not_authorize_edit(tmp_path):
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
    )
    target = tmp_path / "a.py"
    target.write_text("needle\n", encoding="utf-8")
    agent, discipline = _agent(
        tmp_path,
        [
            [
                _call(
                    "search_text",
                    {"query": "needle", "path": "a.py"},
                    "search",
                ),
                _call("list_files", {"path": "."}, "list"),
                _call("git_status", {}, "status"),
                _call("git_diff", {"path": "a.py"}, "diff"),
                _patch("a.py", "needle", "changed"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("inspect then edit")

    assert target.read_text(encoding="utf-8") == "needle\n"
    assert discipline.observed_paths == ()
    results = [
        item for item in agent.messages if isinstance(item, ToolResult)
    ]
    assert [result.is_error for result in results] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_edit_then_read_batch_does_not_retroactively_authorize_edit(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("before", encoding="utf-8")
    agent, discipline = _agent(
        tmp_path,
        [
            [
                _patch("a.py", "before", "after"),
                _read("a.py"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("edit")

    assert target.read_text(encoding="utf-8") == "before"
    results = [
        item for item in agent.messages if isinstance(item, ToolResult)
    ]
    assert [result.is_error for result in results] == [True, False]
    assert discipline.observed_paths == ("a.py",)


def test_blocked_mutation_preserves_budget_progress_and_task_state(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("before", encoding="utf-8")
    blocked_1 = _patch("a.py", "before", "after", "blocked-1")
    blocked_2 = _patch("a.py", "before", "after", "blocked-2")
    agent, _ = _agent(
        tmp_path,
        [
            [blocked_1, blocked_2],
            Message(role="assistant", content="done"),
        ],
        execution_budget=ExecutionBudget(max_tool_calls=2),
    )

    assert agent.run("edit") == "done"

    assert target.read_text(encoding="utf-8") == "before"
    assert agent.execution_usage.tool_calls == 2
    snapshot = agent.progress_snapshot
    assert snapshot.tool_calls == 2
    assert snapshot.failed_tool_results == 2
    assert snapshot.successful_tool_results == 0
    assert snapshot.unique_tool_actions == 1
    assert snapshot.repeated_tool_actions == 1
    record = agent.last_run_record
    assert record is not None
    assert record.tool_execution_count == 0
    assert record.tool_result_error_count == 2

    state = TaskStateReducer().reduce(agent.session.items)
    assert [action.tool_name for action in state.failed_actions] == [
        "apply_patch",
        "apply_patch",
    ]
    assert state.files_modified == ()
    assert len(state.recent_errors) == 2
    assert all(
        "ReadBeforeEditError" in error.message
        for error in state.recent_errors
    )


def test_workspace_precondition_event_is_additive_jsonl_v1(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("before", encoding="utf-8")
    output: list[str] = []
    agent, _ = _agent(
        tmp_path,
        [
            [_patch("./a.py", "before", "after", "blocked")],
            Message(role="assistant", content="done"),
        ],
        listeners=[JsonlEventRenderer(output.append)],
    )

    agent.run("edit")

    wire = next(
        json.loads(line)
        for line in output
        if json.loads(line)["event"] == "workspace_precondition_failed"
    )
    assert wire == {
        "schema_version": 1,
        "event": "workspace_precondition_failed",
        "timestamp": wire["timestamp"],
        "run_id": "workspace-run",
        "step": 0,
        "payload": {
            "tool_name": "apply_patch",
            "call_id": "blocked",
            "path": "a.py",
            "reason": "read_required",
        },
    }
    assert str(tmp_path) not in json.dumps(wire)


def test_generic_agent_and_persistence_schemas_are_unchanged():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    agent = Agent(model=AddModel(), tools=registry, max_steps=2)

    assert agent.run("add") == "The result is 29"

    assert RUN_RECORD_SCHEMA_VERSION == 2
    assert BENCHMARK_RESULT_SCHEMA_VERSION == 1
    assert EVENT_WIRE_SCHEMA_VERSION == 1
    record = agent.last_run_record
    assert record is not None
    persisted = record.to_dict()
    assert "workspace_discipline" not in persisted
    assert "observed_paths" not in persisted
