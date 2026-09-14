import pytest

from miniharness.agent import Agent
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.model import AddModel, EchoModel, ModelError
from miniharness.tools import ADD_TOOL, Tool, ToolRegistry

class FailingModel:
    def generate(self, messages, tools):
        raise ModelError("simulated model failure")

class MultiToolModel:

    def generate(
        self,
        messages,
        tools,
    ):

        if len(messages) == 1:

            return [
                ToolCall(
                    name="add",
                    arguments={
                        "a":1,
                        "b":2,
                    },
                    call_id="1",
                ),
                ToolCall(
                    name="add",
                    arguments={
                        "a":3,
                        "b":4,
                    },
                    call_id="2",
                ),
            ]

        return Message(
            role="assistant",
            content="done",
        )


def failing_add(a: int, b: int) -> int:
    raise ValueError("simulated tool failure")


def test_agent_returns_model_response():
    agent = Agent(model=EchoModel())

    assert agent.trace.end_reason is None

    result = agent.run("hello")

    assert result == "Echo: hello"

def test_agent_records_lifecycle_events():
    registry = ToolRegistry()

    agent = Agent(
        model=EchoModel(),
        tools=registry,
    )

    agent.run("hello")

    assert [
        event.type
        for event in agent.events
    ] == [
        "agent_started",
        "model_completed",
        "agent_completed",
    ]

    assert agent.events[-1].data["reason"] == "completed"


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

    assert [
        event.type
        for event in agent.events
    ] == [
        "agent_started",
        "model_completed",
        "tool_started",
        "tool_completed",
        "agent_failed",
    ]

    assert (
        agent.events[-1].data["reason"]
        == "max_steps_exceeded"
    )



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
    assert isinstance(first_step.output, list)
    assert isinstance(first_step.output[0], ToolCall)

    assert isinstance(first_step.tool_result, list)
    assert isinstance(first_step.tool_result[0], ToolResult)

    assert first_step.tool_result[0].content == "29"
    assert first_step.tool_result[0].is_error is False

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


def test_agent_records_model_error_end_reason():
    agent = Agent(model=FailingModel())

    with pytest.raises(
        ModelError,
        match="simulated model failure",
    ):
        agent.run("hello")

    assert agent.trace.end_reason == "model_error"
    assert agent.trace.steps == []
    
    assert [
        event.type
        for event in agent.events
    ] == [
        "agent_started",
        "agent_failed",
    ]

    assert (
        agent.events[-1].data["reason"]
        == "model_error"
    )


def test_agent_executes_multiple_tools():
    registry = ToolRegistry()

    registry.register(ADD_TOOL)

    agent = Agent(
        model=MultiToolModel(),
        tools=registry,
    )

    result = agent.run(
        "calculate"
    )

    assert result == "done"

    tool_results = [
        item
        for item in agent.messages
        if isinstance(item, ToolResult)
    ]

    assert len(tool_results) == 2


def test_agent_records_tool_lifecycle_events():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    agent = Agent(
        model=AddModel(),
        tools=registry,
        max_steps=2,
    )

    agent.run("calculate")

    assert [
        event.type
        for event in agent.events
    ] == [
        "agent_started",
        "model_completed",
        "tool_started",
        "tool_completed",
        "model_completed",
        "agent_completed",
    ]

    tool_started = agent.events[2]
    tool_completed = agent.events[3]

    assert tool_started.data["step"] == 0
    assert tool_started.data["name"] == "add"
    assert tool_started.data["call_id"] == "1"

    assert tool_completed.data["step"] == 0
    assert tool_completed.data["name"] == "add"
    assert tool_completed.data["call_id"] == "1"
    assert tool_completed.data["is_error"] is False


def test_agent_notifies_event_listener():
    registry = ToolRegistry()

    received_events = []

    def listener(event):
        received_events.append(event)

    agent = Agent(
        model=EchoModel(),
        tools=registry,
        listeners=[listener],
    )

    agent.run("hello")

    assert [
        event.type
        for event in received_events
    ] == [
        "agent_started",
        "model_completed",
        "agent_completed",
    ]

    assert received_events == agent.events


def test_listener_error_does_not_stop_agent():
    registry = ToolRegistry()

    def broken_listener(event):
        raise ValueError("listener failed")

    agent = Agent(
        model=EchoModel(),
        tools=registry,
        listeners=[broken_listener],
    )

    result = agent.run("hello")

    assert result == "Echo: hello"

    assert [
        event.type
        for event in agent.events
    ] == [
        "agent_started",
        "model_completed",
        "agent_completed",
    ]

    assert len(agent.listener_errors) == 3

    assert all(
        isinstance(error, ValueError)
        for error in agent.listener_errors
    )


def test_listener_error_does_not_block_other_listeners():
    registry = ToolRegistry()

    received_events = []

    def broken_listener(event):
        raise ValueError("listener failed")

    def working_listener(event):
        received_events.append(event)

    agent = Agent(
        model=EchoModel(),
        tools=registry,
        listeners=[
            broken_listener,
            working_listener,
        ],
    )

    agent.run("hello")

    assert [
        event.type
        for event in received_events
    ] == [
        "agent_started",
        "model_completed",
        "agent_completed",
    ]