import json

import pytest

from pureharness.agent import Agent
from pureharness.context import (
    ContextBudgetExceeded,
    ContextBuilder,
    ContextLimits,
)
from pureharness.messages import AgentItem, Message, ToolCall, ToolResult
from pureharness.model import (
    ContextWindowExceededError,
    MalformedModelOutputError,
    ModelError,
)
from pureharness.observability import JsonlEventRenderer
from pureharness.run_record import RunRecord
from pureharness.runtime import FailureCategory, RuntimeStage
from pureharness.session import Session
from pureharness.tools import Tool, ToolRegistry
from pureharness.trajectory_compaction import IdentityTrajectoryCompactor


class DeterministicEstimator:
    def __init__(
        self,
        message_costs: dict[str, int],
        *,
        task_state_tokens: int = 2,
        tool_unit_tokens: int = 8,
    ) -> None:
        self.message_costs = message_costs
        self.task_state_tokens = task_state_tokens
        self.tool_unit_tokens = tool_unit_tokens

    def estimate(self, items: list[AgentItem]) -> int:
        if not items:
            return 0
        first = items[0]
        if isinstance(first, Message):
            if first.role == "system":
                return self.task_state_tokens
            return self.message_costs[first.content]
        return self.tool_unit_tokens


class FixedProjector:
    def project(self, result: ToolResult) -> ToolResult:
        return ToolResult(
            name=result.name,
            content="projected",
            call_id=result.call_id,
            is_error=result.is_error,
        )


class ScriptedModel:
    def __init__(self, outputs, *, session: Session | None = None) -> None:
        self.outputs = list(outputs)
        self.session = session
        self.contexts: list[list[AgentItem]] = []
        self.tool_lists: list[list[Tool]] = []
        self.session_snapshots: list[list[AgentItem]] = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        self.tool_lists.append(list(tools))
        if self.session is not None:
            self.session_snapshots.append(self.session.snapshot())
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _builder(
    costs: dict[str, int],
    *,
    project_tools: bool = False,
) -> ContextBuilder:
    return ContextBuilder(
        token_estimator=DeterministicEstimator(costs),
        tool_result_projector=(
            FixedProjector() if project_tools else None
        ),
        trajectory_compactor=IdentityTrajectoryCompactor(),
    )


def _tool() -> Tool:
    return Tool(
        name="lookup",
        description="Deterministic test tool.",
        parameters={"type": "object", "properties": {}},
        function=lambda: "unused",
    )


def _recovery_event(agent: Agent):
    return next(
        event
        for event in agent.events
        if event.type == "context_recovering"
    )


def test_provider_overflow_recovers_smaller_context_without_limits():
    session = Session(
        items=[
            Message(role="user", content="old-1"),
            Message(role="assistant", content="old-2"),
        ]
    )
    overflow = ContextWindowExceededError("provider rejected context")
    model = ScriptedModel(
        [overflow, Message(role="assistant", content="recovered")],
        session=session,
    )
    registry = ToolRegistry()
    tool = _tool()
    registry.register(tool)
    output: list[str] = []
    agent = Agent(
        model=model,
        tools=registry,
        session=session,
        context_builder=_builder(
            {"old-1": 4, "old-2": 4, "new": 4}
        ),
        listeners=[JsonlEventRenderer(output.append)],
    )

    assert agent.run("new") == "recovered"

    assert len(model.contexts) == 2
    assert model.contexts[0][0] == model.contexts[1][0]
    assert model.contexts[0][1:] == [
        Message(role="user", content="old-1"),
        Message(role="assistant", content="old-2"),
        Message(role="user", content="new"),
    ]
    assert model.contexts[1][1:] == [
        Message(role="user", content="new")
    ]
    assert model.tool_lists == [[tool], [tool]]
    expected_request_history = [
        Message(role="user", content="old-1"),
        Message(role="assistant", content="old-2"),
        Message(role="user", content="new"),
    ]
    assert model.session_snapshots == [
        expected_request_history,
        expected_request_history,
    ]
    assert session.items == [
        *expected_request_history,
        Message(role="assistant", content="recovered"),
    ]
    assert [step.index for step in agent.trace.steps] == [0]
    assert agent.progress_snapshot.logical_steps_completed == 1
    assert agent.progress_snapshot.model_attempts == 2
    assert agent.progress_snapshot.context_window_exceeded_count == 1
    assert agent.progress_snapshot.context_recoveries == 1

    failure = agent.last_runtime_failure
    assert failure is not None
    assert failure.stage is RuntimeStage.CONTEXT_PREPARATION
    assert failure.category is FailureCategory.CONTEXT
    assert failure.error_type == "ContextWindowExceededError"

    event = _recovery_event(agent)
    assert event.data["recovery_attempt"] == 1
    assert event.data["max_recoveries"] == 1
    assert event.data["previous_history_tokens"] == 12
    assert event.data["recovery_history_budget"] == 6
    assert event.data["recovered_history_tokens"] == 4
    assert event.data["recovered_estimated_request_tokens"] < event.data[
        "previous_estimated_request_tokens"
    ]
    assert len(
        [
            event
            for event in agent.events
            if event.type == "context_window_exceeded"
        ]
    ) == 1
    assert all(
        event.type != "context_build_failed" for event in agent.events
    )
    assert all(event.type != "model_failed" for event in agent.events)

    wire_events = [json.loads(line) for line in output]
    exceeded_wire = next(
        item
        for item in wire_events
        if item["event"] == "context_window_exceeded"
    )
    assert exceeded_wire["schema_version"] == 1
    assert exceeded_wire["payload"] == {
        "error_type": "ContextWindowExceededError",
        "context_recovery_available": True,
        "recovery_attempt": 1,
        "max_context_recoveries": 1,
    }
    recovery_wire = next(
        item
        for item in wire_events
        if item["event"] == "context_recovering"
    )
    assert recovery_wire["schema_version"] == 1
    assert recovery_wire["step"] == 0
    assert recovery_wire["payload"]["recovered_history_tokens"] == 4

    record = agent.last_run_record
    assert record is not None
    assert record.model_call_count == len(record.model_invocations) == 1
    assert record.model_invocations[0].estimated_history_tokens == 4
    assert record.sum_estimated_history_tokens == 4
    assert RunRecord.from_dict(record.to_dict()) == record
    assert RunRecord.from_json(record.to_json()) == record


def test_context_recovery_can_be_disabled():
    overflow = ContextWindowExceededError("too large")
    model = ScriptedModel([overflow])
    agent = Agent(
        model=model,
        context_builder=_builder({"new": 4}),
        max_context_recoveries=0,
    )

    with pytest.raises(ContextWindowExceededError) as exc_info:
        agent.run("new")

    assert exc_info.value is overflow
    assert len(model.contexts) == 1
    assert all(
        event.type != "context_recovering" for event in agent.events
    )
    assert [event.type for event in agent.events[-4:]] == [
        "context_window_exceeded",
        "progress_snapshot",
        "coding_evidence_snapshot",
        "agent_failed",
    ]
    exceeded = agent.events[-4]
    assert exceeded.data["context_recovery_available"] is False
    assert "recovery_attempt" not in exceeded.data
    assert all(
        event.type != "context_build_failed" for event in agent.events
    )
    assert all(event.type != "model_failed" for event in agent.events)
    assert agent.trace.end_reason == "context_error"
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.category is FailureCategory.CONTEXT


def test_context_recovery_exhaustion_stops_after_second_provider_call():
    first = ContextWindowExceededError("first overflow")
    second = ContextWindowExceededError("second overflow")
    model = ScriptedModel([first, second])
    agent = Agent(
        model=model,
        session=Session(
            items=[Message(role="user", content="old")]
        ),
        context_builder=_builder({"old": 4, "new": 4}),
    )

    with pytest.raises(ContextWindowExceededError) as exc_info:
        agent.run("new")

    assert exc_info.value is second
    assert len(model.contexts) == 2
    assert len(
        [
            event
            for event in agent.events
            if event.type == "context_recovering"
        ]
    ) == 1
    exceeded = [
        event
        for event in agent.events
        if event.type == "context_window_exceeded"
    ]
    assert len(exceeded) == 2
    assert agent.progress_snapshot.model_attempts == 2
    assert agent.progress_snapshot.context_window_exceeded_count == 2
    assert agent.progress_snapshot.context_recoveries == 1
    assert exceeded[0].data["context_recovery_available"] is True
    assert exceeded[1].data["context_recovery_available"] is False
    assert all(event.type != "model_failed" for event in agent.events)
    assert [event.type for event in agent.events[-4:]] == [
        "context_window_exceeded",
        "progress_snapshot",
        "coding_evidence_snapshot",
        "agent_failed",
    ]
    assert all(
        event.type != "context_build_failed" for event in agent.events
    )
    assert agent.trace.end_reason == "context_error"
    assert agent.last_run_record is not None
    assert agent.last_run_record.model_call_count == 1
    assert (
        agent.last_run_record.model_invocations[0].estimated_history_tokens
        == 4
    )


def test_unknown_model_error_does_not_trigger_context_recovery():
    error = ModelError("unrelated provider error")
    model = ScriptedModel([error])
    agent = Agent(
        model=model,
        context_builder=_builder({"new": 4}),
    )

    with pytest.raises(ModelError) as exc_info:
        agent.run("new")

    assert exc_info.value is error
    assert len(model.contexts) == 1
    assert all(
        event.type != "context_recovering" for event in agent.events
    )
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.category is FailureCategory.MODEL
    assert agent.trace.end_reason == "model_error"


def test_provider_overflow_recovers_below_proactive_context_limit():
    model = ScriptedModel(
        [
            ContextWindowExceededError("provider accounting differed"),
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        model=model,
        session=Session(
            items=[Message(role="user", content="old")]
        ),
        context_builder=_builder({"old": 4, "new": 4}),
        context_limits=ContextLimits(
            context_window_tokens=12,
            reserved_output_tokens=2,
        ),
    )

    assert agent.run("new") == "done"

    built = next(
        event for event in agent.events if event.type == "context_built"
    )
    assert built.data["context_pressure_detected"] is False
    recovery = _recovery_event(agent)
    assert recovery.data["previous_history_tokens"] == 8
    assert recovery.data["recovery_history_budget"] == 4
    assert recovery.data["recovered_history_tokens"] == 4
    assert len(model.contexts[1]) < len(model.contexts[0])


def test_atomic_recovery_failure_does_not_call_provider_again():
    overflow = ContextWindowExceededError("too large")
    model = ScriptedModel([overflow])
    session = Session()
    agent = Agent(
        model=model,
        session=session,
        context_builder=_builder({"new": 3}),
    )

    with pytest.raises(
        ContextBudgetExceeded,
        match="Newest indivisible context unit",
    ) as exc_info:
        agent.run("new")

    assert exc_info.value.__cause__ is overflow
    assert len(model.contexts) == 1
    assert session.items == [Message(role="user", content="new")]
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.error_type == "ContextBudgetExceeded"
    assert agent.progress_snapshot.context_window_exceeded_count == 1
    assert agent.progress_snapshot.context_recoveries == 0
    event = _recovery_event(agent)
    assert event.data["previous_history_tokens"] == 3
    assert event.data["recovery_history_budget"] == 1
    assert "recovered_history_tokens" not in event.data
    assert [item.type for item in agent.events[-6:]] == [
        "context_window_exceeded",
        "context_recovering",
        "context_build_failed",
        "progress_snapshot",
        "coding_evidence_snapshot",
        "agent_failed",
    ]


def test_context_recovery_preserves_raw_projected_session_history():
    raw_result = "raw provider-visible tool output"
    initial = [
        Message(role="user", content="old-1"),
        Message(role="assistant", content="old-2"),
        Message(role="user", content="old-3"),
        ToolCall(name="lookup", arguments={}, call_id="lookup-1"),
        ToolResult(
            name="lookup",
            content=raw_result,
            call_id="lookup-1",
        ),
        Message(role="assistant", content="recent"),
    ]
    session = Session(items=list(initial))
    model = ScriptedModel(
        [
            ContextWindowExceededError("too large"),
            Message(role="assistant", content="done"),
        ],
        session=session,
    )
    agent = Agent(
        model=model,
        session=session,
        context_builder=_builder(
            {
                "old-1": 6,
                "old-2": 6,
                "old-3": 6,
                "recent": 4,
                "new": 3,
            },
            project_tools=True,
        ),
    )

    assert agent.run("new") == "done"

    assert model.contexts[1][1:] == [
        ToolCall(name="lookup", arguments={}, call_id="lookup-1"),
        ToolResult(
            name="lookup",
            content="projected",
            call_id="lookup-1",
        ),
        Message(role="assistant", content="recent"),
        Message(role="user", content="new"),
    ]
    expected_request_history = [
        *initial,
        Message(role="user", content="new"),
    ]
    assert model.session_snapshots == [
        expected_request_history,
        expected_request_history,
    ]
    assert session.items == [
        *expected_request_history,
        Message(role="assistant", content="done"),
    ]
    assert session.items[4].content == raw_result


def test_context_recovery_then_model_retry_reuses_smaller_context():
    model = ScriptedModel(
        [
            ContextWindowExceededError("too large"),
            MalformedModelOutputError("malformed tool arguments"),
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        model=model,
        session=Session(
            items=[Message(role="user", content="old")]
        ),
        context_builder=_builder({"old": 4, "new": 4}),
        max_model_retries=1,
        max_context_recoveries=1,
    )

    assert agent.run("new") == "done"

    assert len(model.contexts) == 3
    assert len(model.contexts[1]) < len(model.contexts[0])
    assert model.contexts[1] == model.contexts[2]
    assert [step.index for step in agent.trace.steps] == [0]
    assert len(
        [
            event
            for event in agent.events
            if event.type == "context_recovering"
        ]
    ) == 1
    assert len(
        [event for event in agent.events if event.type == "model_retrying"]
    ) == 1
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.error_type == (
        "MalformedModelOutputError"
    )
    assert agent.last_run_record is not None
    assert agent.last_run_record.model_call_count == 1
    assert len(agent.last_run_record.model_invocations) == 1


@pytest.mark.parametrize("value", [-1, True, 1.5, 2])
def test_agent_rejects_invalid_context_recovery_budget(value):
    with pytest.raises(
        ValueError,
        match="max_context_recoveries must be 0 or 1",
    ):
        Agent(
            model=ScriptedModel([]),
            max_context_recoveries=value,
        )
