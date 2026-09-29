import json

import pytest

from pureharness.agent import Agent
from pureharness.context import (
    ContextBudgetExceeded,
    ContextBuilder,
    ContextLimits,
)
from pureharness.messages import AgentItem, Message, ToolCall, ToolResult
from pureharness.observability import JsonlEventRenderer
from pureharness.runtime import FailureCategory, RuntimeStage
from pureharness.session import Session
from pureharness.tool_selection import estimate_tool_schema_tokens
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


class RecordingModel:
    def __init__(self) -> None:
        self.contexts: list[list[AgentItem]] = []
        self.tool_lists: list[list[Tool]] = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        self.tool_lists.append(list(tools))
        return Message(role="assistant", content="done")


class FixedProjector:
    def project(self, result: ToolResult) -> ToolResult:
        return ToolResult(
            name=result.name,
            content="projected",
            call_id=result.call_id,
            is_error=result.is_error,
        )


class CountingCompactor:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = IdentityTrajectoryCompactor()

    def compact(self, units, token_estimator):
        self.calls += 1
        return self.delegate.compact(units, token_estimator)


def _builder(
    costs: dict[str, int],
    *,
    task_state_tokens: int = 2,
    tool_unit_tokens: int = 8,
    project_tools: bool = False,
    trajectory_compactor=None,
) -> ContextBuilder:
    return ContextBuilder(
        token_estimator=DeterministicEstimator(
            costs,
            task_state_tokens=task_state_tokens,
            tool_unit_tokens=tool_unit_tokens,
        ),
        tool_result_projector=(
            FixedProjector() if project_tools else None
        ),
        trajectory_compactor=(
            trajectory_compactor
            if trajectory_compactor is not None
            else IdentityTrajectoryCompactor()
        ),
    )


def _context_built(agent: Agent):
    return next(
        event
        for event in agent.events
        if event.type == "context_built"
    )


def _tool(name: str = "lookup") -> Tool:
    return Tool(
        name=name,
        description="Deterministic test tool.",
        parameters={"type": "object", "properties": {}},
        function=lambda: "unused",
    )


def test_context_limits_validate_capacity_and_reserve():
    limits = ContextLimits(
        context_window_tokens=100,
        reserved_output_tokens=25,
    )

    assert limits.usable_input_tokens == 75

    for values in ((0, 1), (10, 0), (10, 10), (10, 11)):
        with pytest.raises(ValueError):
            ContextLimits(*values)

    for values in ((True, 1), (10, True), (10.5, 1)):
        with pytest.raises(ValueError):
            ContextLimits(*values)


def test_no_context_limits_preserves_full_history_behavior():
    session = Session(items=[Message(role="user", content="old")])
    model = RecordingModel()
    agent = Agent(
        model=model,
        session=session,
        context_builder=_builder({"old": 4, "new": 3}),
    )

    assert agent.run("new") == "done"

    assert len(model.contexts) == 1
    assert model.contexts[0][1:] == [
        Message(role="user", content="old"),
        Message(role="user", content="new"),
    ]
    event = _context_built(agent)
    assert event.data["estimated_history_tokens"] == 7
    assert "context_pressure_detected" not in event.data
    assert "bounded_history_applied" not in event.data


def test_context_within_capacity_uses_normal_compiled_context():
    session = Session(items=[Message(role="user", content="old")])
    model = RecordingModel()
    agent = Agent(
        model=model,
        session=session,
        context_builder=_builder({"old": 4, "new": 3}),
        context_limits=ContextLimits(
            context_window_tokens=20,
            reserved_output_tokens=5,
        ),
    )

    assert agent.run("new") == "done"

    assert len(model.contexts) == 1
    assert model.contexts[0][1:] == [
        Message(role="user", content="old"),
        Message(role="user", content="new"),
    ]
    event = _context_built(agent)
    assert event.data["context_window_tokens"] == 20
    assert event.data["reserved_output_tokens"] == 5
    assert event.data["usable_input_tokens"] == 15
    assert event.data["available_history_tokens"] == 13
    assert event.data["estimated_request_tokens"] == 9
    assert event.data["context_pressure_detected"] is False
    assert event.data["bounded_history_applied"] is False
    assert "history_token_budget" not in event.data


def test_pressure_bounds_atomic_history_and_preserves_raw_session():
    raw_result = "raw result that must remain durable"
    initial = [
        Message(role="user", content="old"),
        ToolCall(name="lookup", arguments={}, call_id="lookup-1"),
        ToolResult(
            name="lookup",
            content=raw_result,
            call_id="lookup-1",
        ),
        Message(role="assistant", content="recent"),
    ]
    session = Session(items=list(initial))
    model = RecordingModel()
    compactor = CountingCompactor()
    agent = Agent(
        model=model,
        session=session,
        context_builder=_builder(
            {"old": 6, "recent": 4, "new": 3},
            task_state_tokens=3,
            tool_unit_tokens=8,
            project_tools=True,
            trajectory_compactor=compactor,
        ),
        context_limits=ContextLimits(
            context_window_tokens=23,
            reserved_output_tokens=5,
        ),
    )

    assert agent.run("new") == "done"

    assert agent.progress_snapshot.context_pressure_count == 1
    assert agent.progress_snapshot.context_recoveries == 0
    assert agent.progress_snapshot.context_window_exceeded_count == 0

    sent = model.contexts[0]
    assert sent[1:] == [
        ToolCall(name="lookup", arguments={}, call_id="lookup-1"),
        ToolResult(
            name="lookup",
            content="projected",
            call_id="lookup-1",
        ),
        Message(role="assistant", content="recent"),
        Message(role="user", content="new"),
    ]
    assert session.items == [
        *initial,
        Message(role="user", content="new"),
        Message(role="assistant", content="done"),
    ]
    assert session.items[2].content == raw_result
    assert compactor.calls == 2

    event = _context_built(agent)
    assert event.data["context_pressure_detected"] is True
    assert event.data["bounded_history_applied"] is True
    assert event.data["available_history_tokens"] == 15
    assert event.data["estimated_history_tokens"] == 15
    assert event.data["estimated_request_tokens"] == 18
    assert event.data["estimated_request_tokens"] <= event.data[
        "usable_input_tokens"
    ]
    assert event.data["history_token_budget"] == 15
    assert event.data["dropped_units"] == 1

    record = agent.last_run_record
    assert record is not None
    assert record.model_invocations[0].estimated_history_tokens == 15


def test_task_state_and_exposed_tool_schema_trigger_pressure():
    tool = _tool()
    schema_tokens = estimate_tool_schema_tokens([tool])
    registry = ToolRegistry()
    registry.register(tool)
    model = RecordingModel()
    session = Session(items=[Message(role="user", content="old")])
    task_state_tokens = 2
    available_history_tokens = 6
    usable_input_tokens = (
        task_state_tokens + schema_tokens + available_history_tokens
    )
    agent = Agent(
        model=model,
        tools=registry,
        session=session,
        context_builder=_builder(
            {"old": 5, "new": 5},
            task_state_tokens=task_state_tokens,
        ),
        context_limits=ContextLimits(
            context_window_tokens=usable_input_tokens + 5,
            reserved_output_tokens=5,
        ),
    )

    assert agent.run("new") == "done"

    event = _context_built(agent)
    assert 10 <= usable_input_tokens
    assert event.data["estimated_history_tokens"] == 5
    assert event.data["estimated_task_state_tokens"] == task_state_tokens
    assert event.data["available_history_tokens"] == 6
    assert event.data["context_pressure_detected"] is True
    assert event.data["estimated_request_tokens"] == (
        5 + task_state_tokens + schema_tokens
    )
    assert model.contexts[0][1:] == [
        Message(role="user", content="new")
    ]
    assert model.tool_lists == [[tool]]


def test_output_reserve_reduces_available_history_capacity():
    def run_with_reserve(reserve: int):
        model = RecordingModel()
        agent = Agent(
            model=model,
            session=Session(
                items=[Message(role="user", content="old")]
            ),
            context_builder=_builder({"old": 5, "new": 5}),
            include_task_state=False,
            context_limits=ContextLimits(
                context_window_tokens=15,
                reserved_output_tokens=reserve,
            ),
        )
        agent.run("new")
        return agent, model

    low_reserve_agent, low_reserve_model = run_with_reserve(4)
    high_reserve_agent, high_reserve_model = run_with_reserve(6)

    low_event = _context_built(low_reserve_agent)
    high_event = _context_built(high_reserve_agent)
    assert low_event.data["available_history_tokens"] == 11
    assert high_event.data["available_history_tokens"] == 9
    assert low_event.data["bounded_history_applied"] is False
    assert high_event.data["bounded_history_applied"] is True
    assert low_reserve_model.contexts[0] == [
        Message(role="user", content="old"),
        Message(role="user", content="new"),
    ]
    assert high_reserve_model.contexts[0] == [
        Message(role="user", content="new")
    ]


def test_non_history_pressure_fails_before_model_call():
    tool = _tool()
    schema_tokens = estimate_tool_schema_tokens([tool])
    registry = ToolRegistry()
    registry.register(tool)
    model = RecordingModel()
    task_state_tokens = 2
    reserve = 5
    agent = Agent(
        model=model,
        tools=registry,
        context_builder=_builder(
            {"new": 1},
            task_state_tokens=task_state_tokens,
        ),
        context_limits=ContextLimits(
            context_window_tokens=(
                reserve + task_state_tokens + schema_tokens
            ),
            reserved_output_tokens=reserve,
        ),
    )

    with pytest.raises(
        ContextBudgetExceeded,
        match="no positive history budget",
    ):
        agent.run("new")

    assert model.contexts == []
    assert agent.trace.end_reason == "context_error"
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.stage is RuntimeStage.CONTEXT_PREPARATION
    assert agent.last_runtime_failure.category is FailureCategory.CONTEXT
    assert agent.messages == [Message(role="user", content="new")]
    assert [event.type for event in agent.events] == [
        "agent_started",
        "context_build_started",
        "context_build_failed",
        "progress_snapshot",
        "agent_failed",
    ]
    assert agent.last_run_record is not None
    assert agent.last_run_record.model_call_count == 0


def test_pressure_rejects_oversized_newest_atomic_unit():
    model = RecordingModel()
    agent = Agent(
        model=model,
        context_builder=_builder({"oversized": 9}),
        include_task_state=False,
        context_limits=ContextLimits(
            context_window_tokens=10,
            reserved_output_tokens=2,
        ),
    )

    with pytest.raises(
        ContextBudgetExceeded,
        match="Newest indivisible context unit",
    ):
        agent.run("oversized")

    assert model.contexts == []
    assert agent.trace.end_reason == "context_error"


def test_context_pressure_fields_render_in_additive_jsonl_v1():
    output: list[str] = []
    agent = Agent(
        model=RecordingModel(),
        listeners=[JsonlEventRenderer(output.append)],
        context_builder=_builder({"new": 5}),
        context_limits=ContextLimits(
            context_window_tokens=10,
            reserved_output_tokens=3,
        ),
    )

    assert agent.run("new") == "done"

    events = [json.loads(line) for line in output]
    context_event = next(
        event for event in events if event["event"] == "context_built"
    )
    assert context_event["schema_version"] == 1
    assert context_event["step"] == 0
    assert context_event["payload"]["context_window_tokens"] == 10
    assert context_event["payload"]["reserved_output_tokens"] == 3
    assert context_event["payload"]["usable_input_tokens"] == 7
    assert context_event["payload"]["available_history_tokens"] == 5
    assert context_event["payload"]["estimated_request_tokens"] == 7
    assert context_event["payload"]["context_pressure_detected"] is False
    assert context_event["payload"]["bounded_history_applied"] is False
