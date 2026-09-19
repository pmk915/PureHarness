from dataclasses import FrozenInstanceError

import pytest

from miniharness.agent import Agent
from miniharness.context import (
    CompiledContext,
    ContextBudget,
    ContextBudgetExceeded,
    ContextBuilder,
    TokenBudgetContextBuilder,
)
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.model import AddModel, EchoModel, ModelError
from miniharness.replay import replay_run
from miniharness.run_record import (
    RUN_RECORD_SCHEMA_VERSION,
    RunRecord,
    RunRecordSerializationError,
)
from miniharness.session import Session
from miniharness.tool_executor import ToolExecutor
from miniharness.tool_policy import PolicyDecision
from miniharness.tool_selection import (
    StaticToolSelector,
    ToolSelectionError,
)
from miniharness.tools import ADD_TOOL, Tool, ToolRegistry


class FailingModel:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1
        raise ModelError("simulated model failure")


class InterruptingModel:
    def generate(self, messages, tools):
        raise KeyboardInterrupt


class FixedEstimator:
    def __init__(self, cost: int) -> None:
        self.cost = cost

    def estimate(self, items):
        return self.cost


class StaticPolicy:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision

    def evaluate(self, tool, arguments):
        return self.decision


class ToolThenMessageModel:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1
        if self.call_count == 1:
            return [
                ToolCall(
                    name=self.tool_name,
                    arguments={},
                    call_id="call-1",
                )
            ]
        return Message(role="assistant", content="done")


class MetricsContextBuilder(ContextBuilder):
    def estimate_tokens(self, items):
        return 7

    def compile(self, history):
        return CompiledContext(
            items=list(history),
            estimated_tokens=101,
            total_units=len(history),
            included_units=len(history),
            dropped_units=0,
            strategy="MetricsContext",
            projected_tool_results=4,
            compacted_tool_results=2,
            trajectory_compacted=True,
            compacted_source_units=3,
            compacted_tool_actions=3,
            original_trajectory_estimated_tokens=200,
            compacted_trajectory_estimated_tokens=101,
            recent_raw_units=1,
            recent_raw_estimated_tokens=10,
            trajectory_compaction_strategy="TestCompactor",
        )


def _add_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    return registry


def _completed_record():
    agent = Agent(
        model=EchoModel(),
        session_id="session-1",
        run_id_factory=lambda: "run-completed",
    )
    assert agent.run("hello") == "Echo: hello"
    assert agent.last_run_record is not None
    return agent.last_run_record


def test_completed_run_produces_one_record_without_changing_return_api():
    record = _completed_record()

    assert record.run_id == "run-completed"
    assert record.session_id == "session-1"
    assert record.end_reason == "completed"
    assert record.trace.end_reason == "completed"
    assert record.model_call_count == 1
    assert record.tool_call_count == 0
    assert record.tool_execution_count == 0
    assert record.tool_result_error_count == 0
    assert record.step_count == 1
    assert len(record.model_invocations) == 1


def test_run_record_captures_per_inference_and_aggregate_metrics():
    registry = _add_registry()
    agent = Agent(
        model=EchoModel(),
        tools=registry,
        context_builder=MetricsContextBuilder(),
        run_id_factory=lambda: "run-metrics",
    )

    agent.run("hello")
    record = agent.last_run_record
    assert record is not None
    invocation = record.model_invocations[0]

    assert invocation.step == 0
    assert invocation.context_strategy == "MetricsContext"
    assert invocation.estimated_history_tokens == 101
    assert invocation.estimated_task_state_tokens == 7
    assert invocation.registered_tool_count == 1
    assert invocation.exposed_tool_count == 1
    assert invocation.estimated_tool_schema_tokens > 0
    assert invocation.selector_strategy == "AllTools"
    assert invocation.trajectory_compacted is True
    assert invocation.compacted_source_units == 3
    assert invocation.compacted_tool_results == 2
    assert record.sum_estimated_history_tokens == 101
    assert record.sum_estimated_task_state_tokens == 7
    assert record.sum_estimated_tool_schema_tokens > 0
    assert record.trajectory_compaction_count == 1
    assert record.compacted_source_unit_count == 3
    assert record.tool_result_compaction_count == 2


def test_requested_and_executed_tool_calls_are_distinct():
    registry = _add_registry()
    agent = Agent(
        model=AddModel(),
        tools=registry,
        max_steps=2,
        run_id_factory=lambda: "run-tool",
    )

    assert agent.run("calculate") == "The result is 29"
    record = agent.last_run_record
    assert record is not None
    assert record.model_call_count == 2
    assert record.tool_call_count == 1
    assert record.tool_execution_count == 1
    assert record.tool_result_error_count == 0
    assert record.step_count == 2


@pytest.mark.parametrize("rejection", ["policy", "exposure"])
def test_rejected_tool_calls_count_as_requested_not_executed(rejection):
    side_effect_count = 0

    def effect():
        nonlocal side_effect_count
        side_effect_count += 1
        return "executed"

    tool = Tool(
        name="target",
        description="Target tool.",
        parameters={"type": "object", "properties": {}},
        function=effect,
    )
    registry = ToolRegistry()
    registry.register(tool)

    if rejection == "policy":
        selector = StaticToolSelector(["target"])
        executor = ToolExecutor(
            registry,
            StaticPolicy(PolicyDecision.DENY),
        )
    else:
        selector = StaticToolSelector([])
        executor = ToolExecutor(
            registry,
            StaticPolicy(PolicyDecision.ALLOW),
        )

    agent = Agent(
        model=ToolThenMessageModel("target"),
        tools=registry,
        tool_executor=executor,
        tool_selector=selector,
        max_steps=2,
    )
    assert agent.run("test rejection") == "done"
    record = agent.last_run_record
    assert record is not None

    assert side_effect_count == 0
    assert record.tool_call_count == 1
    assert record.tool_execution_count == 0
    assert record.tool_result_error_count == 1


def test_max_steps_failure_has_finalized_record():
    agent = Agent(
        model=AddModel(),
        tools=_add_registry(),
        max_steps=1,
        run_id_factory=lambda: "run-max",
    )

    with pytest.raises(RuntimeError, match="exceeded max steps"):
        agent.run("calculate")

    record = agent.last_run_record
    assert record is not None
    assert record.end_reason == "max_steps_exceeded"
    assert record.model_call_count == 1
    assert record.tool_call_count == 1
    assert record.tool_execution_count == 1
    assert record.step_count == 1


def test_failed_model_attempt_counts_and_serializes():
    model = FailingModel()
    agent = Agent(
        model=model,
        run_id_factory=lambda: "run-model-error",
    )

    with pytest.raises(ModelError, match="simulated model failure"):
        agent.run("hello")

    record = agent.last_run_record
    assert record is not None
    assert model.call_count == 1
    assert record.end_reason == "model_error"
    assert record.model_call_count == 1
    assert record.step_count == 0
    assert RunRecord.from_json(record.to_json()) == record


def test_interrupted_run_is_finalized_and_rolls_back_session_state():
    existing = Message(role="assistant", content="durable history")
    session = Session(items=[existing])
    agent = Agent(
        model=InterruptingModel(),
        session=session,
        session_id="session-interrupted",
        run_id_factory=lambda: "run-interrupted",
    )

    with pytest.raises(KeyboardInterrupt):
        agent.run("new request")

    record = agent.last_run_record
    assert record is not None
    assert record.run_id == "run-interrupted"
    assert record.session_id == "session-interrupted"
    assert record.end_reason == "interrupted"
    assert record.model_call_count == 1
    assert record.step_count == 0
    assert session.items == [existing]
    assert RunRecord.from_json(record.to_json()) == record
    assert agent.events[-1].type == "agent_interrupted"
    assert agent.events[-1].data["reason"] == "interrupted"


def test_interrupted_tool_is_not_left_pending_in_session():
    def uncertain_side_effect():
        raise KeyboardInterrupt

    tool = Tool(
        name="interrupting_tool",
        description="Interrupt while executing.",
        parameters={"type": "object", "properties": {}},
        function=uncertain_side_effect,
    )
    registry = ToolRegistry()
    registry.register(tool)
    agent = Agent(
        model=ToolThenMessageModel("interrupting_tool"),
        tools=registry,
        session_id="session-interrupted-tool",
        run_id_factory=lambda: "run-interrupted-tool",
    )

    with pytest.raises(KeyboardInterrupt):
        agent.run("run uncertain tool")

    record = agent.last_run_record
    assert record is not None
    assert record.end_reason == "interrupted"
    assert record.tool_call_count == 1
    assert record.tool_execution_count == 0
    assert record.step_count == 0
    assert agent.session.items == []


def test_context_failure_has_record_without_model_call():
    agent = Agent(
        model=EchoModel(),
        context_builder=TokenBudgetContextBuilder(
            ContextBudget(max_estimated_tokens=1),
            FixedEstimator(2),
        ),
        run_id_factory=lambda: "run-context-error",
    )

    with pytest.raises(ContextBudgetExceeded):
        agent.run("too large")

    record = agent.last_run_record
    assert record is not None
    assert record.end_reason == "context_error"
    assert record.model_call_count == 0
    assert record.step_count == 0
    assert record.model_invocations == ()


def test_tool_selection_failure_has_record_without_model_call():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    agent = Agent(
        model=EchoModel(),
        tools=registry,
        tool_selector=StaticToolSelector(["missing"]),
        run_id_factory=lambda: "run-selection-error",
    )

    with pytest.raises(ToolSelectionError):
        agent.run("hello")

    record = agent.last_run_record
    assert record is not None
    assert record.end_reason == "tool_selection_error"
    assert record.model_call_count == 0
    assert record.step_count == 0


@pytest.mark.parametrize(
    "record_factory",
    [
        _completed_record,
        lambda: _failed_record(),
    ],
)
def test_run_record_serialization_round_trip(record_factory):
    record = record_factory()

    assert RunRecord.from_dict(record.to_dict()) == record
    assert RunRecord.from_json(record.to_json()) == record
    assert record.to_json() == record.to_json()


def _failed_record():
    agent = Agent(
        model=FailingModel(),
        run_id_factory=lambda: "run-failed-round-trip",
    )
    with pytest.raises(ModelError):
        agent.run("hello")
    assert agent.last_run_record is not None
    return agent.last_run_record


def test_run_record_rejects_unsupported_schema_version():
    data = _completed_record().to_dict()
    data["schema_version"] = RUN_RECORD_SCHEMA_VERSION + 1

    with pytest.raises(
        RunRecordSerializationError,
        match="Unsupported RunRecord schema version",
    ):
        RunRecord.from_dict(data)


def test_serialized_record_replay_is_ordered_and_idempotent():
    agent = Agent(
        model=AddModel(),
        tools=_add_registry(),
        max_steps=2,
        run_id_factory=lambda: "run-replay",
    )
    agent.run("calculate")
    assert agent.last_run_record is not None
    expected = replay_run(agent.last_run_record)
    restored = RunRecord.from_json(agent.last_run_record.to_json())

    first = replay_run(restored)
    second = replay_run(restored)

    assert first == expected
    assert first == second
    assert [entry.kind for entry in first] == [
        "run_started",
        "model_invocation",
        "tool_call",
        "tool_result",
        "model_invocation",
        "model_message",
        "run_ended",
    ]
    assert [entry.step for entry in first] == [
        None,
        0,
        0,
        0,
        1,
        1,
        None,
    ]


def test_replay_never_invokes_model_tool_or_backend():
    counters = {"model": 0, "tool": 0, "backend": 0}

    class FakeBackend:
        def execute(self):
            counters["backend"] += 1
            return "backend result"

    backend = FakeBackend()

    def tool_function():
        counters["tool"] += 1
        return backend.execute()

    class CountingModel(ToolThenMessageModel):
        def generate(self, messages, tools):
            counters["model"] += 1
            return super().generate(messages, tools)

    tool = Tool(
        name="effect",
        description="Execute a fake backend.",
        parameters={"type": "object", "properties": {}},
        function=tool_function,
    )
    registry = ToolRegistry()
    registry.register(tool)
    agent = Agent(
        model=CountingModel("effect"),
        tools=registry,
        max_steps=2,
        run_id_factory=lambda: "run-side-effect",
    )
    agent.run("execute")
    assert agent.last_run_record is not None
    before = dict(counters)

    restored = RunRecord.from_json(agent.last_run_record.to_json())
    replay_run(restored)
    replay_run(restored)

    assert counters == before


def test_same_session_multiple_runs_have_distinct_records():
    run_ids = iter(["run-a", "run-b"])
    session = Session()
    agent = Agent(
        model=EchoModel(),
        session=session,
        session_id="shared-session",
        run_id_factory=lambda: next(run_ids),
    )

    agent.run("first")
    first = agent.last_run_record
    agent.run("second")
    second = agent.last_run_record

    assert first is not None
    assert second is not None
    assert first.run_id == "run-a"
    assert second.run_id == "run-b"
    assert first.session_id == second.session_id == "shared-session"
    assert len(session.items) == 4
    assert len(first.trace.steps) == 1
    assert len(second.trace.steps) == 1


def test_finalized_record_is_frozen_and_does_not_store_session():
    record = _completed_record()

    with pytest.raises(FrozenInstanceError):
        record.run_id = "changed"

    assert isinstance(record.model_invocations, tuple)
    assert "session" not in record.to_dict()
    assert "events" not in record.to_dict()
