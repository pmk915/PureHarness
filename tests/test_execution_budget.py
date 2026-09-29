import json

import pytest

from pureharness.agent import Agent
from pureharness.approval import ApprovalDecision
from pureharness.context import ContextBuilder
from pureharness.events import AgentEvent
from pureharness.messages import AgentItem, Message, ToolCall, ToolResult
from pureharness.model import (
    ContextWindowExceededError,
    RecoverableModelError,
)
from pureharness.observability import JsonlEventRenderer, event_to_wire
from pureharness.run_record import RunRecord
from pureharness.runtime import (
    ExecutionBudget,
    ExecutionBudgetExceeded,
    ExecutionUsage,
    FailureCategory,
    RuntimeStage,
)
from pureharness.session import Session
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_policy import PolicyDecision
from pureharness.tools import Tool, ToolRegistry
from pureharness.trajectory_compaction import IdentityTrajectoryCompactor


class ScriptedModel:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.contexts: list[list[AgentItem]] = []

    @property
    def call_count(self) -> int:
        return len(self.contexts)

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class ContentEstimator:
    def estimate(self, items) -> int:
        if not items:
            return 0
        first = items[0]
        if isinstance(first, Message) and first.role == "system":
            return 1
        return 4


def _call(name: str, call_id: str) -> ToolCall:
    return ToolCall(name=name, arguments={}, call_id=call_id)


def _registry(effects: list[str], *names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(
            Tool(
                name=name,
                description=f"Record {name}.",
                parameters={"type": "object", "properties": {}},
                function=lambda current=name: effects.append(current),
                side_effects=True,
            )
        )
    return registry


def test_execution_budget_value_objects_validate_and_default_unlimited():
    assert ExecutionBudget() == ExecutionBudget(None, None)
    assert ExecutionUsage() == ExecutionUsage(model_attempts=0, tool_calls=0)

    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            ExecutionBudget(max_model_attempts=value)
        with pytest.raises(ValueError):
            ExecutionBudget(max_tool_calls=value)


def test_no_budget_preserves_behavior_and_exposes_physical_usage():
    effects: list[str] = []
    registry = _registry(effects, "one")
    model = ScriptedModel(
        [[_call("one", "one-1")], Message(role="assistant", content="done")]
    )
    agent = Agent(model=model, tools=registry, max_steps=2)

    assert agent.run("go") == "done"
    assert effects == ["one"]
    assert agent.execution_usage == ExecutionUsage(
        model_attempts=2,
        tool_calls=1,
    )


def test_execution_usage_resets_for_each_run():
    model = ScriptedModel(
        [
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ]
    )
    agent = Agent(model=model)

    assert agent.run("one") == "first"
    assert agent.execution_usage == ExecutionUsage(model_attempts=1)
    assert agent.run("two") == "second"
    assert agent.execution_usage == ExecutionUsage(model_attempts=1)
    assert model.call_count == 2


def test_model_attempt_limit_blocks_third_physical_call():
    effects: list[str] = []
    registry = _registry(effects, "one", "two")
    model = ScriptedModel(
        [
            [_call("one", "one-1")],
            [_call("two", "two-1")],
            Message(role="assistant", content="not reached"),
        ]
    )
    agent = Agent(
        model=model,
        tools=registry,
        max_steps=3,
        execution_budget=ExecutionBudget(max_model_attempts=2),
    )

    with pytest.raises(ExecutionBudgetExceeded) as exc_info:
        agent.run("go")

    assert model.call_count == 2
    assert effects == ["one", "two"]
    assert agent.execution_usage == ExecutionUsage(2, 2)
    assert exc_info.value.resource == "model_attempts"
    assert agent.trace.end_reason == "execution_budget_exceeded"
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.stage is RuntimeStage.EXECUTION
    assert agent.last_runtime_failure.category is FailureCategory.BUDGET
    assert [event.type for event in agent.events[-3:]] == [
        "execution_budget_exhausted",
        "progress_snapshot",
        "agent_failed",
    ]
    assert all(event.type != "model_failed" for event in agent.events)

    record = agent.last_run_record
    assert record is not None
    assert record.schema_version == 2
    assert record.end_reason == "execution_budget_exceeded"
    assert record.model_call_count == len(record.model_invocations) == 3
    assert "execution_usage" not in record.to_dict()
    assert "model_attempt_count" not in record.to_dict()
    assert RunRecord.from_json(record.to_json()) == record


def test_model_retry_is_blocked_by_global_attempt_budget():
    model = ScriptedModel(
        [
            RecoverableModelError("retry locally"),
            Message(role="assistant", content="not reached"),
        ]
    )
    agent = Agent(
        model=model,
        max_model_retries=1,
        execution_budget=ExecutionBudget(max_model_attempts=1),
    )

    with pytest.raises(ExecutionBudgetExceeded):
        agent.run("go")

    assert model.call_count == 1
    assert agent.execution_usage.model_attempts == 1
    assert agent.progress_snapshot.model_retries == 0
    assert "model_retrying" in [event.type for event in agent.events]
    assert all(event.type != "model_failed" for event in agent.events)
    assert agent.trace.end_reason == "execution_budget_exceeded"


def test_context_recovery_retry_is_blocked_by_global_attempt_budget():
    session = Session(
        items=[
            Message(role="user", content="old-1"),
            Message(role="assistant", content="old-2"),
        ]
    )
    model = ScriptedModel(
        [
            ContextWindowExceededError("too large"),
            Message(role="assistant", content="not reached"),
        ]
    )
    agent = Agent(
        model=model,
        session=session,
        context_builder=ContextBuilder(
            token_estimator=ContentEstimator(),
            trajectory_compactor=IdentityTrajectoryCompactor(),
        ),
        execution_budget=ExecutionBudget(max_model_attempts=1),
    )

    with pytest.raises(ExecutionBudgetExceeded):
        agent.run("new")

    assert model.call_count == 1
    assert agent.execution_usage.model_attempts == 1
    assert agent.progress_snapshot.context_window_exceeded_count == 1
    assert agent.progress_snapshot.context_recoveries == 1
    event_types = [event.type for event in agent.events]
    assert "context_window_exceeded" in event_types
    assert "context_recovering" in event_types
    assert event_types[-3:] == [
        "execution_budget_exhausted",
        "progress_snapshot",
        "agent_failed",
    ]
    assert "context_build_failed" not in event_types
    assert agent.trace.end_reason == "execution_budget_exceeded"


def test_tool_budget_exact_fit_executes_complete_batch():
    effects: list[str] = []
    registry = _registry(effects, "one", "two", "three")
    model = ScriptedModel(
        [
            [
                _call("one", "one-1"),
                _call("two", "two-1"),
                _call("three", "three-1"),
            ],
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        model=model,
        tools=registry,
        max_steps=2,
        execution_budget=ExecutionBudget(max_tool_calls=3),
    )

    assert agent.run("go") == "done"
    assert effects == ["one", "two", "three"]
    assert agent.execution_usage.tool_calls == 3


class RecordingPolicy:
    def __init__(self, decision=PolicyDecision.ALLOW) -> None:
        self.decision = decision
        self.calls: list[str] = []

    def evaluate(self, tool, arguments):
        self.calls.append(tool.name)
        return self.decision


class RecordingApprovalHandler:
    def __init__(self, decision=ApprovalDecision.APPROVE) -> None:
        self.decision = decision
        self.calls: list[str] = []

    def request_approval(self, request):
        self.calls.append(request.tool_name)
        return self.decision


def test_oversized_tool_batch_is_rejected_before_all_side_effects():
    effects: list[str] = []
    registry = _registry(effects, "one", "two", "three", "four")
    policy = RecordingPolicy(PolicyDecision.REQUIRE_APPROVAL)
    approvals = RecordingApprovalHandler()
    model = ScriptedModel(
        [
            [_call("one", "one-1"), _call("two", "two-1")],
            [_call("three", "three-1"), _call("four", "four-1")],
        ]
    )
    agent = Agent(
        model=model,
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            policy,
            approval_handler=approvals,
        ),
        max_steps=2,
        execution_budget=ExecutionBudget(max_tool_calls=3),
    )

    with pytest.raises(ExecutionBudgetExceeded) as exc_info:
        agent.run("go")

    assert effects == ["one", "two"]
    assert policy.calls == ["one", "two"]
    assert approvals.calls == ["one", "two"]
    assert agent.execution_usage.tool_calls == 2
    assert exc_info.value.requested == 2
    assert exc_info.value.remaining == 1
    assert agent.messages == [
        Message(role="user", content="go"),
        _call("one", "one-1"),
        ToolResult("one", "None", "one-1", False),
        _call("two", "two-1"),
        ToolResult("two", "None", "two-1", False),
    ]
    event = next(
        event
        for event in agent.events
        if event.type == "execution_budget_exhausted"
    )
    assert event.data == {
        "step": 1,
        "resource": "tool_calls",
        "used": 2,
        "limit": 3,
        "requested": 2,
        "remaining": 1,
    }
    assert all(event.type != "model_failed" for event in agent.events)


def test_tool_error_policy_denial_and_approval_denial_consume_usage():
    executions: list[str] = []
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fail",
            description="Fail.",
            parameters={"type": "object", "properties": {}},
            function=lambda: (_ for _ in ()).throw(ValueError("boom")),
        )
    )
    registry.register(
        Tool(
            name="denied",
            description="Denied.",
            parameters={"type": "object", "properties": {}},
            function=lambda: executions.append("denied"),
        )
    )
    registry.register(
        Tool(
            name="approval",
            description="Approval denied.",
            parameters={"type": "object", "properties": {}},
            function=lambda: executions.append("approval"),
        )
    )

    class MixedPolicy:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def evaluate(self, tool, arguments):
            self.calls.append(tool.name)
            return {
                "fail": PolicyDecision.ALLOW,
                "denied": PolicyDecision.DENY,
                "approval": PolicyDecision.REQUIRE_APPROVAL,
            }[tool.name]

    policy = MixedPolicy()
    approvals = RecordingApprovalHandler(ApprovalDecision.DENY)
    model = ScriptedModel(
        [
            [
                _call("fail", "fail-1"),
                _call("denied", "denied-1"),
                _call("approval", "approval-1"),
            ],
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        model=model,
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            policy,
            approval_handler=approvals,
        ),
        max_steps=2,
        execution_budget=ExecutionBudget(max_tool_calls=3),
    )

    assert agent.run("go") == "done"
    assert executions == []
    assert policy.calls == ["fail", "denied", "approval"]
    assert approvals.calls == ["approval"]
    assert agent.execution_usage.tool_calls == 3
    assert all(
        isinstance(item, ToolResult) and item.is_error
        for item in agent.messages
        if isinstance(item, ToolResult)
    )


def test_unexposed_tool_call_does_not_consume_dispatch_budget():
    effects: list[str] = []
    registry = _registry(effects, "known")
    model = ScriptedModel(
        [
            [_call("hidden", "hidden-1")],
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        model=model,
        tools=registry,
        max_steps=2,
        execution_budget=ExecutionBudget(max_tool_calls=1),
    )

    assert agent.run("go") == "done"
    assert effects == []
    assert agent.execution_usage.tool_calls == 0
    result = agent.messages[2]
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert "ToolNotExposedError" in result.content


def test_budget_exhaustion_has_additive_jsonl_v1_rendering():
    event = AgentEvent(
        type="execution_budget_exhausted",
        data={
            "step": 4,
            "resource": "tool_calls",
            "used": 2,
            "limit": 3,
            "requested": 2,
            "remaining": 1,
        },
        run_id="run-budget",
    )

    assert event_to_wire(event) == {
        "schema_version": 1,
        "event": "execution_budget_exhausted",
        "timestamp": event.timestamp.isoformat().replace("+00:00", "Z"),
        "run_id": "run-budget",
        "step": 4,
        "payload": {
            "resource": "tool_calls",
            "used": 2,
            "limit": 3,
            "requested": 2,
            "remaining": 1,
        },
    }

    output: list[str] = []
    JsonlEventRenderer(output.append)(event)
    assert json.loads(output[0])["event"] == "execution_budget_exhausted"
