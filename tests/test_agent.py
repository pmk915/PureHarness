from miniharness.agent import Agent
from miniharness.model import AddModel, EchoModel


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


from miniharness.tools import ADD_TOOL, ToolRegistry


def test_agent_executes_tool_loop():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    agent = Agent(
        model=AddModel(),
        tools=registry,
    )

    result = agent.run("calculate")

    assert result == "The result is 29"
    assert len(agent.messages) == 3
    assert agent.messages[0].role == "user"
    assert agent.messages[1].role == "tool"
    assert agent.messages[2].role == "assistant"