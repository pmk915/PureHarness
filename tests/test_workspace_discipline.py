import json
import subprocess
import sys
from dataclasses import FrozenInstanceError

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
from pureharness.tools import ADD_TOOL, Tool, ToolRegistry
from pureharness.workspace_discipline import WorkspaceDiscipline


class ScriptedModel:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)

    def generate(self, messages, tools):
        output = self.outputs.pop(0)
        if callable(output):
            output = output()
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


class SideEffectPolicy(RecordingPolicy):
    def __init__(self, side_effect, *, require_approval=False) -> None:
        super().__init__()
        self.side_effect = side_effect
        self.require_approval = require_approval

    def evaluate(self, tool, arguments):
        self.calls.append(tool.name)
        if tool.name in {"write_file", "apply_patch"}:
            self.side_effect()
            if self.require_approval:
                return PolicyDecision.REQUIRE_APPROVAL
        return PolicyDecision.ALLOW


class SideEffectApprovalHandler(RecordingApprovalHandler):
    def __init__(self, side_effect) -> None:
        super().__init__()
        self.side_effect = side_effect

    def request_approval(self, request):
        self.requests.append(request)
        self.side_effect()
        return ApprovalDecision.APPROVE


class DenyingApprovalHandler(RecordingApprovalHandler):
    def request_approval(self, request):
        self.requests.append(request)
        return ApprovalDecision.DENY


class MutationApprovalPolicy(RecordingPolicy):
    def evaluate(self, tool, arguments):
        self.calls.append(tool.name)
        if tool.name in {"write_file", "apply_patch"}:
            return PolicyDecision.REQUIRE_APPROVAL
        return PolicyDecision.ALLOW


class MutationDenyPolicy(RecordingPolicy):
    def evaluate(self, tool, arguments):
        self.calls.append(tool.name)
        if tool.name in {"write_file", "apply_patch"}:
            return PolicyDecision.DENY
        return PolicyDecision.ALLOW


def _call(name: str, arguments: dict[str, object], call_id: str) -> ToolCall:
    return ToolCall(name=name, arguments=arguments, call_id=call_id)


def _read(path: str, call_id: str = "read") -> ToolCall:
    return _call("read_file", {"path": path}, call_id)


def _read_range(path: str, call_id: str = "read-range") -> ToolCall:
    return _call(
        "read_file_range",
        {"path": path, "start_line": 1, "max_lines": 20},
        call_id,
    )


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


def test_partial_range_read_does_not_authorize_edit_but_full_read_does(
    tmp_path,
):
    target = tmp_path / "a.py"
    target.write_text("before\n", encoding="utf-8")
    agent, discipline = _agent(
        tmp_path,
        [
            [
                _read_range("a.py"),
                _patch("a.py", "before", "after", "blocked"),
            ],
            [
                _read("a.py", "full-read"),
                _patch("a.py", "before", "after", "allowed"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    assert agent.run("inspect then edit") == "done"

    assert target.read_text(encoding="utf-8") == "after\n"
    assert discipline.observed_paths == ("a.py",)
    results = [
        item for item in agent.messages if isinstance(item, ToolResult)
    ]
    assert [result.is_error for result in results] == [
        False,
        True,
        False,
        False,
    ]
    assert "ReadBeforeEditError" in results[1].content


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
    assert agent.workspace_snapshot is None


def test_content_changed_after_read_blocks_before_authorization(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("version A\n", encoding="utf-8")
    policy = RecordingPolicy(PolicyDecision.REQUIRE_APPROVAL)
    approvals = RecordingApprovalHandler()
    output: list[str] = []

    def change_then_patch():
        target.write_text("version B\n", encoding="utf-8")
        return [_patch("a.py", "version A", "agent change")]

    agent, discipline = _agent(
        tmp_path,
        [
            [_read("a.py")],
            change_then_patch,
            Message(role="assistant", content="recovered"),
        ],
        policy=policy,
        approval_handler=approvals,
        listeners=[JsonlEventRenderer(output.append)],
    )

    assert agent.run("edit") == "recovered"
    assert target.read_text(encoding="utf-8") == "version B\n"
    assert policy.calls == ["read_file"]
    assert len(approvals.requests) == 1
    assert discipline.observed_paths == ()
    stale = next(
        event
        for event in agent.events
        if event.type == "workspace_precondition_failed"
    )
    assert stale.data["reason"] == "stale_observation"
    assert stale.data["change"] == "content_changed"
    wire = next(
        json.loads(line)
        for line in output
        if json.loads(line)["event"] == "workspace_precondition_failed"
    )
    assert wire["payload"] == {
        "tool_name": "apply_patch",
        "call_id": "patch",
        "path": "a.py",
        "reason": "stale_observation",
        "change": "content_changed",
    }
    assert not any(
        event.type == "tool_started"
        and event.data["call_id"] == "patch"
        for event in agent.events
    )


def test_observation_fingerprints_exact_successful_read_result(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("returned A", encoding="utf-8")
    registry = ToolRegistry()
    for tool in create_coding_tools(tmp_path):
        registry.register(tool)

    def read_then_external_change(path: str) -> str:
        observed = (tmp_path / path).read_text(encoding="utf-8")
        (tmp_path / path).write_text("disk B", encoding="utf-8")
        return observed

    registry.register(
        Tool(
            name="read_file",
            description="Controlled read race fixture.",
            parameters={"type": "object"},
            function=read_then_external_change,
        )
    )
    discipline = WorkspaceDiscipline(tmp_path)
    policy = RecordingPolicy()
    agent = Agent(
        model=ScriptedModel(
            [
                [
                    _read("a.py"),
                    _patch("a.py", "returned A", "agent"),
                ],
                Message(role="assistant", content="done"),
            ]
        ),
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            policy=policy,
            precondition=discipline,
        ),
        max_steps=2,
    )

    agent.run("race")

    assert target.read_text(encoding="utf-8") == "disk B"
    assert policy.calls == ["read_file"]
    stale = next(
        event
        for event in agent.events
        if event.type == "workspace_precondition_failed"
    )
    assert stale.data["change"] == "content_changed"


def test_successful_read_refreshes_after_stale_observation(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")

    def change_then_patch():
        target.write_text("B", encoding="utf-8")
        return [_patch("a.py", "A", "X", "stale")]

    agent, discipline = _agent(
        tmp_path,
        [
            [_read("a.py", "read-a")],
            change_then_patch,
            [_read("a.py", "read-b")],
            [_patch("a.py", "B", "C", "fresh")],
            Message(role="assistant", content="done"),
        ],
    )

    assert agent.run("recover") == "done"
    assert target.read_text(encoding="utf-8") == "C"
    assert discipline.observed_paths == ("a.py",)


def test_deleted_observed_target_reports_target_missing(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")

    def delete_then_patch():
        target.unlink()
        return [_patch("a.py", "A", "B")]

    agent, discipline = _agent(
        tmp_path,
        [
            [_read("a.py")],
            delete_then_patch,
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("edit")

    stale = next(
        event
        for event in agent.events
        if event.type == "workspace_precondition_failed"
    )
    assert stale.data["change"] == "target_missing"
    assert discipline.observed_paths == ()
    assert not any(
        event.type == "tool_started"
        and event.data["call_id"] == "patch"
        for event in agent.events
    )


def test_existing_write_is_blocked_when_observation_is_stale(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")

    def change_then_write():
        target.write_text("external", encoding="utf-8")
        return [_write("a.py", "agent")]

    agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py")],
            change_then_write,
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("overwrite")

    assert target.read_text(encoding="utf-8") == "external"
    stale = next(
        event
        for event in agent.events
        if event.type == "workspace_precondition_failed"
    )
    assert stale.data["change"] == "content_changed"


def test_new_target_appearing_after_prepare_is_not_overwritten(tmp_path):
    target = tmp_path / "new.py"
    policy = SideEffectPolicy(
        lambda: target.write_text("external", encoding="utf-8")
    )
    agent, _ = _agent(
        tmp_path,
        [
            [_write("new.py", "agent")],
            Message(role="assistant", content="done"),
        ],
        policy=policy,
    )

    agent.run("create")

    assert target.read_text(encoding="utf-8") == "external"
    stale = next(
        event
        for event in agent.events
        if event.type == "workspace_precondition_failed"
    )
    assert stale.data["change"] == "target_appeared"
    assert not any(event.type == "tool_started" for event in agent.events)


def test_approval_delay_is_revalidated_before_tool_started(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    approvals = SideEffectApprovalHandler(
        lambda: target.write_text("external", encoding="utf-8")
    )
    agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py")],
            [_patch("a.py", "A", "agent")],
            Message(role="assistant", content="done"),
        ],
        policy=MutationApprovalPolicy(),
        approval_handler=approvals,
    )

    agent.run("edit")

    assert target.read_text(encoding="utf-8") == "external"
    event_types = [
        event.type
        for event in agent.events
        if event.data.get("call_id") == "patch"
    ]
    assert event_types == [
        "tool_policy_evaluated",
        "approval_requested",
        "approval_granted",
        "workspace_precondition_failed",
    ]
    assert len(agent.trace.approvals) == 1


def test_allow_policy_seam_is_revalidated_before_tool_started(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    policy = SideEffectPolicy(
        lambda: target.write_text("external", encoding="utf-8")
    )
    agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py")],
            [_patch("a.py", "A", "agent")],
            Message(role="assistant", content="done"),
        ],
        policy=policy,
    )

    agent.run("edit")

    assert target.read_text(encoding="utf-8") == "external"
    assert not any(
        event.type == "tool_started"
        and event.data.get("call_id") == "patch"
        for event in agent.events
    )


def test_write_overwrite_refreshes_observation_for_later_patch(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    agent, _ = _agent(
        tmp_path,
        [
            [
                _read("a.py"),
                _write("a.py", "B", "overwrite"),
                _patch("a.py", "B", "C", "patch-after-write"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("edit")

    assert target.read_text(encoding="utf-8") == "C"


def test_run_command_side_effect_is_detected_without_command_parsing(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    command = _call(
        "run_command",
        {
            "argv": [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('a.py').write_text('external')",
            ]
        },
        "command",
    )
    agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py"), command, _patch("a.py", "A", "agent")],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("edit")

    assert target.read_text(encoding="utf-8") == "external"
    stale = next(
        event
        for event in agent.events
        if event.type == "workspace_precondition_failed"
    )
    assert stale.data["change"] == "content_changed"


def test_same_content_external_rewrite_remains_fresh(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    policy = SideEffectPolicy(
        lambda: target.write_text("A", encoding="utf-8")
    )
    agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py")],
            [_patch("a.py", "A", "B")],
            Message(role="assistant", content="done"),
        ],
        policy=policy,
    )

    agent.run("edit")

    assert target.read_text(encoding="utf-8") == "B"
    assert not any(
        event.type == "workspace_precondition_failed"
        for event in agent.events
    )


@pytest.mark.parametrize(
    ("calls", "expected_operation", "expected_tool"),
    [
        ([_write("new.py", "new")], "created", "write_file"),
        (
            [_read("a.py"), _write("a.py", "new")],
            "overwritten",
            "write_file",
        ),
        (
            [_read("a.py"), _patch("a.py", "old", "new")],
            "patched",
            "apply_patch",
        ),
    ],
    ids=["created", "overwritten", "patched"],
)
def test_successful_structured_mutation_records_operation_and_event(
    tmp_path,
    calls,
    expected_operation,
    expected_tool,
):
    (tmp_path / "a.py").write_text("old", encoding="utf-8")
    output: list[str] = []
    agent, _ = _agent(
        tmp_path,
        [calls, Message(role="assistant", content="done")],
        listeners=[JsonlEventRenderer(output.append)],
    )

    agent.run("mutate")

    snapshot = agent.workspace_snapshot
    assert snapshot is not None
    assert len(snapshot.mutations) == 1
    mutation = snapshot.mutations[0]
    assert mutation.operation == expected_operation
    assert mutation.tool_name == expected_tool
    assert snapshot.modified_paths == (mutation.path,)
    with pytest.raises(FrozenInstanceError):
        mutation.path = "changed.py"

    wire = next(
        json.loads(line)
        for line in output
        if json.loads(line)["event"] == "workspace_mutated"
    )
    assert wire["payload"] == {
        "tool_name": expected_tool,
        "call_id": calls[-1].call_id,
        "path": mutation.path,
        "operation": expected_operation,
    }


def test_repeated_mutations_have_ordered_ledger_and_unique_paths(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    agent, _ = _agent(
        tmp_path,
        [
            [
                _read("a.py"),
                _patch("a.py", "A", "B", "patch-1"),
                _patch("a.py", "B", "C", "patch-2"),
                _write("b.py", "new", "create-b"),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("mutate")

    snapshot = agent.workspace_snapshot
    assert snapshot is not None
    assert [item.operation for item in snapshot.mutations] == [
        "patched",
        "patched",
        "created",
    ]
    assert [item.path for item in snapshot.mutations] == [
        "a.py",
        "a.py",
        "b.py",
    ]
    assert snapshot.modified_paths == ("a.py", "b.py")


def test_blocked_denied_and_failed_actions_are_not_mutation_evidence(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")

    read_required_agent, _ = _agent(
        tmp_path,
        [[_patch("a.py", "A", "B")], Message(role="assistant", content="done")],
    )
    read_required_agent.run("blocked")
    assert read_required_agent.workspace_snapshot.mutations == ()

    stale_agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py")],
            lambda: (
                target.write_text("external", encoding="utf-8"),
                [_patch("a.py", "A", "B")],
            )[1],
            Message(role="assistant", content="done"),
        ],
    )
    stale_agent.run("stale")
    assert stale_agent.workspace_snapshot.mutations == ()

    target.write_text("A", encoding="utf-8")
    denied_agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py")],
            [_patch("a.py", "A", "B")],
            Message(role="assistant", content="done"),
        ],
        policy=MutationDenyPolicy(),
    )
    denied_agent.run("denied")
    assert denied_agent.workspace_snapshot.mutations == ()

    approval_agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py")],
            [_patch("a.py", "A", "B")],
            Message(role="assistant", content="done"),
        ],
        policy=MutationApprovalPolicy(),
        approval_handler=DenyingApprovalHandler(),
    )
    approval_agent.run("approval denied")
    assert approval_agent.workspace_snapshot.mutations == ()

    failed_agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py"), _patch("a.py", "missing", "B")],
            Message(role="assistant", content="done"),
        ],
    )
    failed_agent.run("tool failure")
    assert failed_agent.workspace_snapshot.mutations == ()


def test_workspace_snapshot_resets_between_runs(tmp_path):
    model = ScriptedModel(
        [
            [_write("first.py", "one")],
            Message(role="assistant", content="first"),
            [_patch("existing.py", "old", "new", "blocked-second")],
            Message(role="assistant", content="second"),
        ]
    )
    (tmp_path / "existing.py").write_text("old", encoding="utf-8")
    registry = ToolRegistry()
    for tool in create_coding_tools(tmp_path):
        registry.register(tool)
    agent = Agent(
        model=model,
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            precondition=WorkspaceDiscipline(tmp_path),
        ),
        max_steps=2,
    )

    assert agent.run("first") == "first"
    assert len(agent.workspace_snapshot.mutations) == 1
    assert agent.run("second") == "second"

    snapshot = agent.workspace_snapshot
    assert snapshot.mutations == ()
    assert snapshot.modified_paths == ()
    assert snapshot.read_required_block_count == 1
    assert snapshot.stale_block_count == 0


def test_snapshot_and_jsonl_do_not_expose_fingerprints(tmp_path):
    secret_content = "unique-model-visible-content"
    target = tmp_path / "a.py"
    target.write_text(secret_content, encoding="utf-8")
    output: list[str] = []
    agent, _ = _agent(
        tmp_path,
        [
            [_read("a.py"), _patch("a.py", secret_content, "replacement")],
            Message(role="assistant", content="done"),
        ],
        listeners=[JsonlEventRenderer(output.append)],
    )

    agent.run("edit")

    workspace_lines = [
        line
        for line in output
        if json.loads(line)["event"]
        in {"workspace_precondition_failed", "workspace_mutated"}
    ]
    public_text = repr(agent.workspace_snapshot) + "\n".join(workspace_lines)
    assert secret_content not in public_text
    assert "fingerprint" not in public_text.lower()
    assert "sha256" not in public_text.lower()


def test_full_workspace_recovery_composes_with_runtime_evidence(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    attempted = _patch("a.py", "A", "agent", "attempted")

    def external_change_then_retry():
        target.write_text("B", encoding="utf-8")
        return [_patch("a.py", "A", "agent", "stale")]

    agent, _ = _agent(
        tmp_path,
        [
            [attempted],
            [_read("a.py", "read-a")],
            external_change_then_retry,
            [_read("a.py", "read-b")],
            [_patch("a.py", "B", "C", "successful")],
            Message(role="assistant", content="done"),
        ],
        execution_budget=ExecutionBudget(max_tool_calls=5),
    )

    assert agent.run("recover") == "done"
    assert target.read_text(encoding="utf-8") == "C"

    snapshot = agent.workspace_snapshot
    assert snapshot is not None
    assert snapshot.read_required_block_count == 1
    assert snapshot.stale_block_count == 1
    assert snapshot.modified_paths == ("a.py",)
    assert [item.operation for item in snapshot.mutations] == ["patched"]

    assert agent.execution_usage.tool_calls == 5
    progress = agent.progress_snapshot
    assert progress.tool_calls == 5
    assert progress.successful_tool_results == 3
    assert progress.failed_tool_results == 2

    record = agent.last_run_record
    assert record is not None
    assert record.tool_execution_count == 3
    assert record.tool_result_error_count == 2
    persisted = record.to_dict()
    assert "workspace_snapshot" not in persisted
    assert "mutations" not in persisted
    assert "stale_block_count" not in persisted

    state = TaskStateReducer().reduce(agent.session.items)
    assert state.files_read == ("a.py",)
    assert state.files_modified == ("a.py",)
    assert "ReadBeforeEditError" in state.recent_errors[-2].message
    assert "StaleFileError" in state.recent_errors[-1].message

    precondition_events = [
        event
        for event in agent.events
        if event.type == "workspace_precondition_failed"
    ]
    assert [event.data["reason"] for event in precondition_events] == [
        "read_required",
        "stale_observation",
    ]
    assert sum(event.type == "workspace_mutated" for event in agent.events) == 1


def test_blocked_workspace_calls_keep_budget_and_progress_boundaries(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    repeated = _patch("a.py", "A", "agent", "same-action")

    def change_then_repeat():
        target.write_text("external", encoding="utf-8")
        return [_patch("a.py", "A", "agent", "same-action-2")]

    agent, _ = _agent(
        tmp_path,
        [
            [repeated],
            [_read("a.py")],
            change_then_repeat,
            Message(role="assistant", content="done"),
        ],
        execution_budget=ExecutionBudget(max_tool_calls=3),
    )

    agent.run("attempt")

    assert agent.execution_usage.tool_calls == 3
    progress = agent.progress_snapshot
    assert progress.failed_tool_results == 2
    assert progress.successful_tool_results == 1
    assert progress.repeated_tool_actions == 1
    record = agent.last_run_record
    assert record is not None
    assert record.tool_execution_count == 1
    assert record.tool_result_error_count == 2
    assert agent.workspace_snapshot.mutations == ()

    state = TaskStateReducer().reduce(agent.session.items)
    assert state.files_modified == ()
    assert len(state.failed_actions) == 2


def test_workspace_event_ordering_for_precondition_and_success(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("A", encoding="utf-8")
    agent, _ = _agent(
        tmp_path,
        [
            [_patch("a.py", "A", "B", "blocked")],
            [_read("a.py", "read")],
            [_patch("a.py", "A", "B", "success")],
            Message(role="assistant", content="done"),
        ],
    )

    agent.run("edit")

    blocked_types = [
        event.type
        for event in agent.events
        if event.data.get("call_id") == "blocked"
    ]
    assert blocked_types == ["workspace_precondition_failed"]
    success_types = [
        event.type
        for event in agent.events
        if event.data.get("call_id") == "success"
    ]
    assert success_types == [
        "tool_policy_evaluated",
        "tool_started",
        "tool_completed",
        "workspace_mutated",
    ]
