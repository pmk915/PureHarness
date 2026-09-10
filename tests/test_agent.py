import pytest

from miniharness.agent import Agent
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.model import AddModel, EchoModel
from miniharness.tools import ADD_TOOL, ToolRegistry


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

    assert len(agent.messages) == 3

    assert isinstance(agent.messages[0], Message)
    assert isinstance(agent.messages[1], ToolCall)
    assert isinstance(agent.messages[2], ToolResult)