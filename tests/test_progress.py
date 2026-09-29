import json
from dataclasses import FrozenInstanceError

import pytest

from pureharness.agent import Agent
from pureharness.approval import ApprovalDecision
from pureharness.events import AgentEvent
from pureharness.messages import Message, ToolCall
from pureharness.observability import JsonlEventRenderer, event_to_wire
from pureharness.progress import ProgressSnapshot
from pureharness.run_record import RUN_RECORD_SCHEMA_VERSION
from pureharness.runtime import ExecutionBudget, ExecutionBudgetExceeded
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_policy import PolicyDecision
from pureharness.tools import Tool, ToolRegistry


class ScriptedModel:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class StaticPolicy:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision

    def evaluate(self, tool, arguments):
        return self.decision


class DenyingApprovalHandler:
    def request_approval(self, request):
        return ApprovalDecision.DENY


def _message(content: str = "done") -> Message:
    return Message(role="assistant", content=content)


def _tool(name: str, function=None) -> Tool:
    return Tool(
        name=name,
        description=f"Deterministic {name} test tool.",
        parameters={"type": "object", "properties": {}},
        function=(function if function is not None else lambda **_: "ok"),
    )


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _call(
    name: str,
    arguments: dict[str, object],
    call_id: str,
) -> ToolCall:
    return ToolCall(name=name, arguments=arguments, call_id=call_id)


def test_progress_snapshot_is_immutable_non_negative_and_resets_per_run():
    model = ScriptedModel([_message("first"), _message("second")])
    agent = Agent(model=model)

    assert agent.progress_snapshot == ProgressSnapshot()
    with pytest.raises(FrozenInstanceError):
        agent.progress_snapshot.model_attempts = 3
    with pytest.raises(ValueError):
        ProgressSnapshot(model_attempts=-1)

    assert agent.run("one") == "first"
    assert agent.progress_snapshot.logical_steps_completed == 1
    assert agent.progress_snapshot.model_attempts == 1

    assert agent.run("two") == "second"
    assert agent.progress_snapshot.logical_steps_completed == 1
    assert agent.progress_snapshot.model_attempts == 1
    assert len(agent.session.items) == 4


def test_normal_completion_exposes_terminal_snapshot_and_jsonl_v1():
    output: list[str] = []
    agent = Agent(
        model=ScriptedModel([_message()]),
        listeners=[JsonlEventRenderer(output.append)],
    )

    assert agent.run("go") == "done"

    snapshot = agent.progress_snapshot
    assert snapshot == ProgressSnapshot(
        logical_steps_completed=1,
        model_attempts=1,
    )
    assert snapshot.model_attempts == agent.execution_usage.model_attempts
    assert snapshot.tool_calls == agent.execution_usage.tool_calls
    assert [event.type for event in agent.events[-2:]] == [
        "progress_snapshot",
        "agent_completed",
    ]
    event = agent.events[-2]
    assert event.data["terminal"] is True
    assert event.data["step"] == 0

    wire = json.loads(output[-2])
    assert wire["schema_version"] == 1
    assert wire["event"] == "progress_snapshot"
    assert wire["step"] == 0
    assert wire["payload"] == {
        "logical_steps_completed": 1,
        "model_attempts": 1,
        "tool_calls": 0,
        "successful_tool_results": 0,
        "failed_tool_results": 0,
        "unique_tool_actions": 0,
        "repeated_tool_actions": 0,
        "max_identical_tool_action_count": 0,
        "model_retries": 0,
        "context_recoveries": 0,
        "context_pressure_count": 0,
        "context_window_exceeded_count": 0,
        "terminal": True,
    }


def test_action_identity_ignores_call_id_and_canonicalizes_dict_keys():
    calls = [
        _call(
            "act",
            {"path": "a", "options": {"x": 1, "y": 2}},
            "1",
        ),
        _call(
            "act",
            {"options": {"y": 2, "x": 1}, "path": "a"},
            "99",
        ),
        _call("act", {"path": "b"}, "2"),
        _call(
            "act",
            {"options": {"x": 1, "y": 2}, "path": "a"},
            "3",
        ),
        _call("act", {"items": [1, 2]}, "4"),
        _call("act", {"items": [1, 2]}, "5"),
    ]
    agent = Agent(
        model=ScriptedModel([calls, _message()]),
        tools=_registry(_tool("act")),
        max_steps=2,
    )

    assert agent.run("go") == "done"

    snapshot = agent.progress_snapshot
    assert snapshot.logical_steps_completed == 2
    assert snapshot.tool_calls == 6
    assert snapshot.successful_tool_results == 6
    assert snapshot.unique_tool_actions == 3
    assert snapshot.repeated_tool_actions == 3
    assert snapshot.max_identical_tool_action_count == 3
    nonterminal = [
        event
        for event in agent.events
        if event.type == "progress_snapshot"
        and event.data["terminal"] is False
    ]
    assert len(nonterminal) == 1
    assert nonterminal[0].data["logical_steps_completed"] == 1


def test_list_order_is_meaningful_for_action_identity():
    calls = [
        _call("act", {"items": [1, 2]}, "1"),
        _call("act", {"items": [2, 1]}, "2"),
    ]
    agent = Agent(
        model=ScriptedModel([calls, _message()]),
        tools=_registry(_tool("act")),
        max_steps=2,
    )

    agent.run("go")

    assert agent.progress_snapshot.unique_tool_actions == 2
    assert agent.progress_snapshot.repeated_tool_actions == 0


def test_tool_results_include_errors_but_only_dispatched_actions_count():
    def fail(**arguments):
        raise ValueError("failure")

    agent = Agent(
        model=ScriptedModel(
            [
                [
                    _call("ok", {}, "1"),
                    _call("fail", {}, "2"),
                    _call("hidden", {}, "3"),
                ],
                _message(),
            ]
        ),
        tools=_registry(_tool("ok"), _tool("fail", fail)),
        max_steps=2,
    )

    agent.run("go")

    snapshot = agent.progress_snapshot
    assert snapshot.tool_calls == 2
    assert snapshot.unique_tool_actions == 2
    assert snapshot.successful_tool_results == 1
    assert snapshot.failed_tool_results == 2


@pytest.mark.parametrize(
    ("decision", "approval_handler"),
    [
        (PolicyDecision.DENY, None),
        (PolicyDecision.REQUIRE_APPROVAL, DenyingApprovalHandler()),
    ],
)
def test_policy_and_approval_denials_are_failed_dispatched_actions(
    decision,
    approval_handler,
):
    registry = _registry(_tool("act"))
    executor = ToolExecutor(
        registry,
        policy=StaticPolicy(decision),
        approval_handler=approval_handler,
    )
    agent = Agent(
        model=ScriptedModel(
            [[_call("act", {"value": 1}, "1")], _message()]
        ),
        tool_executor=executor,
        max_steps=2,
    )

    agent.run("go")

    snapshot = agent.progress_snapshot
    assert snapshot.tool_calls == 1
    assert snapshot.unique_tool_actions == 1
    assert snapshot.successful_tool_results == 0
    assert snapshot.failed_tool_results == 1


def test_budget_rejected_batch_records_no_actions_or_results():
    agent = Agent(
        model=ScriptedModel(
            [[_call("act", {"value": 1}, "1"), _call(
                "act", {"value": 2}, "2"
            )]]
        ),
        tools=_registry(_tool("act")),
        execution_budget=ExecutionBudget(max_tool_calls=1),
    )

    with pytest.raises(ExecutionBudgetExceeded):
        agent.run("go")

    snapshot = agent.progress_snapshot
    assert snapshot.model_attempts == 1
    assert snapshot.tool_calls == 0
    assert snapshot.unique_tool_actions == 0
    assert snapshot.successful_tool_results == 0
    assert snapshot.failed_tool_results == 0
    assert [event.type for event in agent.events[-2:]] == [
        "progress_snapshot",
        "agent_failed",
    ]
    assert agent.events[-2].data["terminal"] is True


def test_unserializable_arguments_skip_identity_without_breaking_tool():
    agent = Agent(
        model=ScriptedModel(
            [[_call("act", {"value": {1, 2}}, "1")], _message()]
        ),
        tools=_registry(_tool("act")),
        max_steps=2,
    )

    assert agent.run("go") == "done"
    assert agent.progress_snapshot.tool_calls == 1
    assert agent.progress_snapshot.successful_tool_results == 1
    assert agent.progress_snapshot.unique_tool_actions == 0
    assert agent.progress_snapshot.repeated_tool_actions == 0


def test_repetition_is_observed_without_stopping_execution():
    model = ScriptedModel(
        [
            [_call("act", {"path": "same"}, "1")],
            [_call("act", {"path": "same"}, "2")],
            [_call("act", {"path": "same"}, "3")],
            _message("finished"),
        ]
    )
    agent = Agent(
        model=model,
        tools=_registry(_tool("act")),
        max_steps=4,
    )

    assert agent.run("go") == "finished"
    assert model.call_count == 4
    assert agent.progress_snapshot.repeated_tool_actions == 2
    assert agent.progress_snapshot.max_identical_tool_action_count == 3


def test_progress_event_wire_supports_additive_event_without_raw_identity():
    event = AgentEvent(
        type="progress_snapshot",
        run_id="run-1",
        data={
            "step": 2,
            "logical_steps_completed": 2,
            "model_attempts": 3,
            "tool_calls": 4,
            "successful_tool_results": 3,
            "failed_tool_results": 1,
            "unique_tool_actions": 2,
            "repeated_tool_actions": 2,
            "max_identical_tool_action_count": 3,
            "model_retries": 1,
            "context_recoveries": 0,
            "context_pressure_count": 1,
            "context_window_exceeded_count": 0,
            "terminal": False,
        },
    )

    wire = event_to_wire(event)

    assert wire["schema_version"] == 1
    assert wire["event"] == "progress_snapshot"
    assert wire["step"] == 2
    assert wire["payload"]["terminal"] is False
    assert "arguments" not in wire["payload"]
    assert "fingerprint" not in wire["payload"]


def test_run_record_v2_does_not_persist_progress_fields():
    agent = Agent(model=ScriptedModel([_message()]))

    agent.run("go")

    assert RUN_RECORD_SCHEMA_VERSION == 2
    record = agent.last_run_record
    assert record is not None
    persisted = record.to_dict()
    for field in (
        "progress_snapshot",
        "model_attempts",
        "model_retries",
        "context_recoveries",
        "unique_tool_actions",
        "repeated_tool_actions",
    ):
        assert field not in persisted
