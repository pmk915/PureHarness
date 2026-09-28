import pytest

from pureharness.agent import Agent
from pureharness.context import (
    ContextBudget,
    ContextBudgetExceeded,
    TokenBudgetContextBuilder,
)
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.model import (
    ContextWindowExceededError,
    RecoverableModelError,
)
from pureharness.run_record import RunRecord
from pureharness.runtime import (
    FailureCategory,
    RecoveryAction,
    RuntimeController,
    RuntimeStage,
)
from pureharness.tool_selection import (
    StaticToolSelector,
    ToolSelectionError,
)
from pureharness.tools import ADD_TOOL, Tool, ToolRegistry


class ToolThenMessageModel:
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1
        if self.call_count == 1:
            return [
                ToolCall(
                    name="add",
                    arguments={"a": 1, "b": 2},
                    call_id="add-1",
                )
            ]
        return Message(role="assistant", content="done")


class CountingFailingModel:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1
        raise self.error


class ScriptedModel:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.contexts = []

    @property
    def call_count(self) -> int:
        return len(self.contexts)

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class FixedEstimator:
    def estimate(self, items):
        return 2


@pytest.mark.parametrize(
    ("stage", "category"),
    [
        (RuntimeStage.STATE_REDUCTION, FailureCategory.CONTEXT),
        (RuntimeStage.CONTEXT_PREPARATION, FailureCategory.CONTEXT),
        (RuntimeStage.TOOL_SELECTION, FailureCategory.TOOL),
        (RuntimeStage.MODEL_REQUEST, FailureCategory.MODEL),
        (RuntimeStage.TOOL_EXECUTION, FailureCategory.TOOL),
        (RuntimeStage.POLICY_EVALUATION, FailureCategory.POLICY),
    ],
)
def test_runtime_controller_classifies_failure_stage(stage, category):
    failure = RuntimeController().classify_failure(
        step=3,
        stage=stage,
        error=ValueError("failure"),
    )

    assert failure.step == 3
    assert failure.stage is stage
    assert failure.category is category
    assert failure.error_type == "ValueError"
    assert failure.recoverable is False


def test_runtime_controller_retries_only_recognized_failure_with_budget():
    controller = RuntimeController()
    failure = controller.classify_failure(
        step=0,
        stage=RuntimeStage.MODEL_REQUEST,
        error=RecoverableModelError("temporary"),
    )

    assert failure.recoverable is True
    assert controller.recovery_action(
        failure, attempt=1, max_attempts=2
    ) is RecoveryAction.RETRY
    assert controller.recovery_action(
        failure, attempt=2, max_attempts=2
    ) is RecoveryAction.FAIL


def test_runtime_controller_rebuilds_only_context_overflow_with_budget():
    controller = RuntimeController()
    failure = controller.classify_failure(
        step=2,
        stage=RuntimeStage.MODEL_REQUEST,
        error=ContextWindowExceededError("too large"),
    )

    assert failure.stage is RuntimeStage.CONTEXT_PREPARATION
    assert failure.category is FailureCategory.CONTEXT
    assert failure.recoverable is True
    assert controller.recovery_action(
        failure,
        attempt=1,
        max_attempts=2,
        context_recovery_attempt=0,
        max_context_recoveries=1,
    ) is RecoveryAction.REBUILD_CONTEXT
    assert controller.recovery_action(
        failure,
        attempt=1,
        max_attempts=2,
        context_recovery_attempt=1,
        max_context_recoveries=1,
    ) is RecoveryAction.FAIL


def test_agent_runtime_lifecycle_still_executes_tool_then_completes():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    model = ToolThenMessageModel()
    agent = Agent(model=model, tools=registry, max_steps=2)

    assert agent.run("calculate") == "done"

    assert model.call_count == 2
    assert agent.last_runtime_failure is None
    assert agent.trace.end_reason == "completed"
    assert [type(item) for item in agent.messages] == [
        Message,
        ToolCall,
        ToolResult,
        Message,
    ]


def test_context_failure_is_classified_without_changing_outcome():
    agent = Agent(
        model=ToolThenMessageModel(),
        context_builder=TokenBudgetContextBuilder(
            ContextBudget(max_estimated_tokens=1),
            FixedEstimator(),
        ),
    )

    with pytest.raises(ContextBudgetExceeded) as exc_info:
        agent.run("oversized")

    failure = agent.last_runtime_failure
    assert failure is not None
    assert failure.step == 0
    assert failure.stage is RuntimeStage.CONTEXT_PREPARATION
    assert failure.category is FailureCategory.CONTEXT
    assert failure.error_type == type(exc_info.value).__name__
    assert failure.recoverable is False
    assert agent.trace.end_reason == "context_error"
    assert agent.last_run_record is not None
    assert agent.last_run_record.end_reason == "context_error"


def test_model_failure_is_classified_and_not_retried():
    error = ValueError("malformed provider response")
    model = CountingFailingModel(error)
    agent = Agent(model=model)

    with pytest.raises(ValueError) as exc_info:
        agent.run("hello")

    assert exc_info.value is error
    assert model.call_count == 1
    failure = agent.last_runtime_failure
    assert failure is not None
    assert failure.step == 0
    assert failure.stage is RuntimeStage.MODEL_REQUEST
    assert failure.category is FailureCategory.MODEL
    assert failure.error_type == "ValueError"
    assert failure.recoverable is False
    assert agent.trace.end_reason == "model_error"
    assert agent.last_run_record is not None
    assert agent.last_run_record.end_reason == "model_error"


def test_tool_selection_failure_is_a_runtime_tool_failure():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    agent = Agent(
        model=ToolThenMessageModel(),
        tools=registry,
        tool_selector=StaticToolSelector(["missing"]),
    )

    with pytest.raises(ToolSelectionError):
        agent.run("hello")

    failure = agent.last_runtime_failure
    assert failure is not None
    assert failure.stage is RuntimeStage.TOOL_SELECTION
    assert failure.category is FailureCategory.TOOL
    assert agent.trace.end_reason == "tool_selection_error"


def test_tool_execution_error_remains_an_observation():
    def fail() -> None:
        raise ValueError("ordinary tool failure")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="Fail for a deterministic test.",
            parameters=ADD_TOOL.parameters,
            function=lambda a, b: fail(),
        )
    )
    agent = Agent(
        model=ToolThenMessageModel(),
        tools=registry,
        max_steps=2,
    )

    assert agent.run("calculate") == "done"

    result = agent.messages[2]
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert result.content == (
        "Tool error: ValueError: ordinary tool failure"
    )
    assert agent.last_runtime_failure is None
    assert agent.trace.end_reason == "completed"


def test_recoverable_model_failure_retries_in_same_step():
    model = ScriptedModel(
        [
            RecoverableModelError("retry me"),
            Message(role="assistant", content="recovered"),
        ]
    )
    agent = Agent(model=model)

    assert agent.run("hello") == "recovered"

    assert model.call_count == 2
    assert model.contexts[0] == model.contexts[1]
    assert agent.messages == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="recovered"),
    ]
    assert len(agent.trace.steps) == 1
    assert agent.trace.steps[0].index == 0
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.step == 0
    assert agent.last_runtime_failure.recoverable is True
    assert [event.type for event in agent.events] == [
        "agent_started",
        "context_build_started",
        "context_built",
        "model_started",
        "model_retrying",
        "model_completed",
        "agent_completed",
    ]
    retrying = agent.events[4]
    assert retrying.data == {
        "step": 0,
        "attempt": 2,
        "max_attempts": 2,
        "error_type": "RecoverableModelError",
        "failure_category": "model",
    }
    assert all(
        event.type not in {"model_failed", "agent_failed"}
        for event in agent.events
    )

    record = agent.last_run_record
    assert record is not None
    assert record.step_count == 1
    assert record.model_call_count == len(record.model_invocations) == 1
    assert record.sum_estimated_history_tokens == sum(
        item.estimated_history_tokens
        for item in record.model_invocations
    )
    assert record.sum_estimated_task_state_tokens == sum(
        item.estimated_task_state_tokens
        for item in record.model_invocations
    )
    assert record.sum_estimated_tool_schema_tokens == sum(
        item.estimated_tool_schema_tokens
        for item in record.model_invocations
    )
    assert RunRecord.from_dict(record.to_dict()) == record
    assert RunRecord.from_json(record.to_json()) == record


def test_recoverable_model_failure_stops_when_budget_is_exhausted():
    first = RecoverableModelError("first")
    second = RecoverableModelError("second")
    model = ScriptedModel([first, second])
    agent = Agent(model=model, max_model_retries=1)

    with pytest.raises(RecoverableModelError) as exc_info:
        agent.run("hello")

    assert exc_info.value is second
    assert model.call_count == 2
    assert agent.messages == [Message(role="user", content="hello")]
    assert agent.trace.end_reason == "model_error"
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.error_type == (
        "RecoverableModelError"
    )
    assert [event.type for event in agent.events[-3:]] == [
        "model_retrying",
        "model_failed",
        "agent_failed",
    ]
    record = agent.last_run_record
    assert record is not None
    assert record.model_call_count == len(record.model_invocations) == 1
    assert record.step_count == 0


def test_recoverable_model_failure_is_fail_fast_when_retry_disabled():
    error = RecoverableModelError("do not retry")
    model = ScriptedModel([error])
    agent = Agent(model=model, max_model_retries=0)

    with pytest.raises(RecoverableModelError) as exc_info:
        agent.run("hello")

    assert exc_info.value is error
    assert model.call_count == 1
    assert all(event.type != "model_retrying" for event in agent.events)
    assert agent.trace.end_reason == "model_error"


def test_unknown_model_exception_does_not_use_retry_budget():
    error = ValueError("unknown")
    model = ScriptedModel([error])
    agent = Agent(model=model, max_model_retries=3)

    with pytest.raises(ValueError) as exc_info:
        agent.run("hello")

    assert exc_info.value is error
    assert model.call_count == 1
    assert agent.last_runtime_failure is not None
    assert agent.last_runtime_failure.recoverable is False
    assert all(event.type != "model_retrying" for event in agent.events)


def test_model_retry_does_not_consume_tool_loop_step():
    model = ScriptedModel(
        [
            RecoverableModelError("retry tool request"),
            [
                ToolCall(
                    name="add",
                    arguments={"a": 1, "b": 2},
                    call_id="add-retry",
                )
            ],
            Message(role="assistant", content="done"),
        ]
    )
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    agent = Agent(model=model, tools=registry, max_steps=2)

    assert agent.run("calculate") == "done"

    assert model.call_count == 3
    assert [step.index for step in agent.trace.steps] == [0, 1]
    assert agent.messages == [
        Message(role="user", content="calculate"),
        ToolCall(
            name="add",
            arguments={"a": 1, "b": 2},
            call_id="add-retry",
        ),
        ToolResult(
            name="add",
            content="3",
            call_id="add-retry",
            is_error=False,
        ),
        Message(role="assistant", content="done"),
    ]
    record = agent.last_run_record
    assert record is not None
    assert record.step_count == 2
    assert record.model_call_count == len(record.model_invocations) == 2
    assert [item.step for item in record.model_invocations] == [0, 1]


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_agent_rejects_invalid_model_retry_budget(value):
    with pytest.raises(
        ValueError,
        match="max_model_retries must be non-negative",
    ):
        Agent(model=ToolThenMessageModel(), max_model_retries=value)
