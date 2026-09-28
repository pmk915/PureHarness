import pytest

from pureharness.agent import Agent
from pureharness.context import (
    ContextBudget,
    ContextBudgetExceeded,
    TokenBudgetContextBuilder,
)
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.runtime import (
    FailureCategory,
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
