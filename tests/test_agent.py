import pytest

from miniharness.agent import Agent
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.model import AddModel, EchoModel, ModelError
from miniharness.tools import ADD_TOOL, Tool, ToolRegistry
from miniharness.context import ContextBuilder
from miniharness.session import Session

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


class RecordingContextBuilder(ContextBuilder):
    def __init__(self):
        self.histories = []

    def build(self, history):
        self.histories.append(list(history))

        return list(history)


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
        "context_build_started",
        "context_built",
        "model_started",
        "model_completed",
        "agent_completed",
    ]

    assert agent.events[-1].data["reason"] == "completed"
    assert agent.events[-1].data["step_count"] == 1

    context_built = agent.events[2]

    assert context_built.data == {
        "step": 0,
        "history_item_count": 1,
        "context_item_count": 1,
        "context_strategy": "ContextBuilder",
    }

    model_completed = agent.events[4]

    assert model_completed.data == {
        "step": 0,
        "output_kind": "message",
        "tool_call_count": 0,
    }


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
        "context_build_started",
        "context_built",
        "model_started",
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

    tool_completed = next(
        event
        for event in agent.events
        if event.type == "tool_completed"
    )

    assert tool_completed.data["is_error"] is True
    assert (
        tool_completed.data["result_character_count"]
        == len(agent.messages[2].content)
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
        "context_build_started",
        "context_built",
        "model_started",
        "model_failed",
        "agent_failed",
    ]

    assert agent.events[-2].data == {
        "step": 0,
        "reason": "model_error",
        "error_type": "ModelError",
    }

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
        "context_build_started",
        "context_built",
        "model_started",
        "model_completed",
        "tool_started",
        "tool_completed",
        "context_build_started",
        "context_built",
        "model_started",
        "model_completed",
        "agent_completed",
    ]

    tool_started = agent.events[5]
    tool_completed = agent.events[6]

    assert tool_started.data["step"] == 0
    assert tool_started.data["name"] == "add"
    assert tool_started.data["call_id"] == "1"
    assert tool_started.data["arguments_preview"] == {
        "a": "12",
        "b": "17",
    }

    assert tool_completed.data["step"] == 0
    assert tool_completed.data["name"] == "add"
    assert tool_completed.data["call_id"] == "1"
    assert tool_completed.data["is_error"] is False
    assert tool_completed.data["duration_seconds"] >= 0
    assert tool_completed.data["result_character_count"] == 2


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
        "context_build_started",
        "context_built",
        "model_started",
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
        "context_build_started",
        "context_built",
        "model_started",
        "model_completed",
        "agent_completed",
    ]

    assert len(agent.listener_errors) == 6

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
        "context_build_started",
        "context_built",
        "model_started",
        "model_completed",
        "agent_completed",
    ]


def test_agent_builds_model_context_from_history():
    registry = ToolRegistry()

    context_builder = RecordingContextBuilder()

    agent = Agent(
        model=EchoModel(),
        tools=registry,
        context_builder=context_builder,
    )

    agent.run("hello")

    assert len(context_builder.histories) == 1

    history = context_builder.histories[0]

    assert len(history) == 1
    assert isinstance(history[0], Message)
    assert history[0].role == "user"
    assert history[0].content == "hello"

    assert len(agent.messages) == 2


def test_agent_records_history_in_session():
    registry = ToolRegistry()

    agent = Agent(
        model=EchoModel(),
        tools=registry,
    )

    agent.run("hello")

    assert len(agent.session.items) == 2

    assert isinstance(
        agent.session.items[0],
        Message,
    )

    assert (
        agent.session.items[0].content
        == "hello"
    )

    assert isinstance(
        agent.session.items[1],
        Message,
    )

    assert (
        agent.session.items[1].content
        == "Echo: hello"
    )

    assert (
        agent.messages
        is agent.session.items
    )


def test_agent_resumes_existing_session():
    registry = ToolRegistry()

    session = Session()

    session.append(
        Message(
            role="user",
            content="first",
        )
    )

    session.append(
        Message(
            role="assistant",
            content="Echo: first",
        )
    )

    agent = Agent(
        model=EchoModel(),
        tools=registry,
        session=session,
    )

    result = agent.run("second")

    assert result == "Echo: second"

    assert agent.session is session

    assert len(agent.session.items) == 4

    assert (
        agent.session.items[0].content
        == "first"
    )

    assert (
        agent.session.items[1].content
        == "Echo: first"
    )

    assert (
        agent.session.items[2].content
        == "second"
    )
    assert (
        agent.session.items[3].content
        == "Echo: second"
    )



def test_agent_resumes_session_loaded_from_jsonl(
    tmp_path,
):
    original = Session()

    original.append(
        Message(
            role="user",
            content="first",
        )
    )

    original.append(
        Message(
            role="assistant",
            content="Echo: first",
        )
    )

    path = tmp_path / "session.jsonl"

    original.save_jsonl(path)

    loaded = Session.load_jsonl(path)

    registry = ToolRegistry()

    agent = Agent(
        model=EchoModel(),
        tools=registry,
        session=loaded,
    )

    agent.run("second")

    assert len(agent.session.items) == 4

    assert (
        agent.session.items[0].content
        == "first"
    )

    assert (
        agent.session.items[2].content
        == "second"
    )
