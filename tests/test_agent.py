import pytest

from miniharness.agent import Agent
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.model import AddModel, EchoModel
from miniharness.tools import ADD_TOOL, Tool, ToolRegistry


def failing_add(a: int, b: int) -> int:
    raise ValueError("simulated tool failure")


def test_agent_returns_model_response():
    agent = Agent(model=EchoModel())

    result = agent.run("hello")

    assert result == "Echo: hello"


def test_agent_keeps_conversation_history():
    agent = Agent(model=EchoModel())

    agent.run("hello")
    agent.run("world")

    assert len(agent.messages) == 4

    assert agent.messages[0].content == "hello"
    assert agent.messages[1].content == "Echo: hello"
    assert agent.messages[2].content == "world"
    assert agent.messages[3].content == "Echo: world"


def test_agent_executes_tool_loop():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    agent = Agent(
        model=AddModel(),
        tools=registry,
        max_steps=2,
    )

    result = agent.run("calculate")

    assert result == "The result is 29"

    assert len(agent.messages) == 4

    assert isinstance(agent.messages[0], Message)
    assert isinstance(agent.messages[1], ToolCall)
    assert isinstance(agent.messages[2], ToolResult)
    assert isinstance(agent.messages[3], Message)

    assert agent.messages[0].role == "user"
    assert agent.messages[0].content == "calculate"

    assert agent.messages[1].name == "add"
    assert agent.messages[1].arguments == {
        "a": 12,
        "b": 17,
    }

    assert agent.messages[2].name == "add"
    assert agent.messages[2].content == "29"
    assert agent.messages[2].is_error is False

    assert agent.messages[3].role == "assistant"
    assert agent.messages[3].content == "The result is 29"


def test_agent_stops_after_max_steps():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    agent = Agent(
        model=AddModel(),
        tools=registry,
        max_steps=1,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent exceeded max steps",
    ):
        agent.run("calculate")

    assert agent.trace.end_reason == "max_steps_exceeded"

    assert len(agent.messages) == 3

    assert isinstance(agent.messages[0], Message)
    assert isinstance(agent.messages[1], ToolCall)
    assert isinstance(agent.messages[2], ToolResult)


def test_agent_converts_tool_error_to_observation():
    registry = ToolRegistry()

    failing_tool = Tool(
        name="add",
        description="A failing add tool for testing.",
        parameters=ADD_TOOL.parameters,
        function=failing_add,
    )

    registry.register(failing_tool)

    agent = Agent(
        model=AddModel(),
        tools=registry,
        max_steps=2,
    )

    result = agent.run("calculate")

    assert result == (
        "The result is "
        "Tool error: ValueError: simulated tool failure"
    )

    assert len(agent.messages) == 4

    assert isinstance(agent.messages[2], ToolResult)
    assert agent.messages[2].is_error is True
    assert agent.messages[2].content == (
        "Tool error: ValueError: simulated tool failure"
    )


def test_agent_records_execution_trace():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    agent = Agent(
        model=AddModel(),
        tools=registry,
        max_steps=2,
    )

    agent.run("calculate")

    assert len(agent.trace.steps) == 2

    first_step = agent.trace.steps[0]
    second_step = agent.trace.steps[1]

    assert first_step.index == 0
    assert isinstance(first_step.output, ToolCall)
    assert isinstance(first_step.tool_result, ToolResult)
    assert first_step.tool_result.content == "29"
    assert first_step.tool_result.is_error is False

    assert second_step.index == 1
    assert isinstance(second_step.output, Message)
    assert second_step.output.content == "The result is 29"
    assert second_step.tool_result is None
    assert agent.trace.end_reason == "completed"


def test_agent_resets_trace_for_each_run():
    agent = Agent(model=EchoModel())

    agent.run("hello")
    agent.run("world")

    assert len(agent.trace.steps) == 1

    step = agent.trace.steps[0]

    assert step.index == 0
    assert isinstance(step.output, Message)
    assert step.output.content == "Echo: world"