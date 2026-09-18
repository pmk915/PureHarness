import pytest

from miniharness.agent import Agent
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.model import AddModel, EchoModel, ModelError
from miniharness.tools import ADD_TOOL, Tool, ToolRegistry
from miniharness.context import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextBuilder,
    TokenBudgetContextBuilder,
)
from miniharness.session import Session
from miniharness.session_store import JsonlSessionStore
from miniharness.tool_executor import ToolExecutor
from miniharness.tool_policy import PolicyDecision

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


class StaticPolicy:
    def __init__(self, decision):
        self.decision = decision

    def evaluate(self, tool, arguments):
        return self.decision


class RecordingContextBuilder(ContextBuilder):
    def __init__(self):
        super().__init__()
        self.histories = []

    def compile(self, history):
        self.histories.append(list(history))

        return super().compile(history)


class RecordingModel:
    def __init__(self):
        self.contexts = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        return Message(role="assistant", content="done")


class FixedMessageEstimator:
    def __init__(self, costs):
        self.costs = costs

    def estimate(self, items):
        return self.costs[items[0].content]


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

    assert context_built.data["step"] == 0
    assert context_built.data["history_item_count"] == 1
    assert context_built.data["context_item_count"] == 1
    assert context_built.data["context_strategy"] == "FullHistory"
    assert context_built.data["estimated_history_tokens"] > 0
    assert context_built.data["total_units"] == 1
    assert context_built.data["included_units"] == 1
    assert context_built.data["dropped_units"] == 0
    assert "history_token_budget" not in context_built.data

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
        "tool_policy_evaluated",
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

    tool_events = [
        event
        for event in agent.events
        if event.type.startswith("tool_")
    ]

    assert [event.type for event in tool_events] == [
        "tool_policy_evaluated",
        "tool_started",
        "tool_completed",
    ]

    tool_completed = tool_events[2]

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
        "tool_policy_evaluated",
        "tool_started",
        "tool_completed",
        "context_build_started",
        "context_built",
        "model_started",
        "model_completed",
        "agent_completed",
    ]

    policy_evaluated = agent.events[5]
    tool_started = agent.events[6]
    tool_completed = agent.events[7]

    assert policy_evaluated.data == {
        "step": 0,
        "name": "add",
        "call_id": "1",
        "risk_level": "read",
        "decision": "allow",
    }

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


def test_agent_uses_replaceable_allow_policy():
    call_count = 0

    def add(a, b):
        nonlocal call_count
        call_count += 1
        return a + b

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="Add two values.",
            parameters=ADD_TOOL.parameters,
            function=add,
        )
    )
    executor = ToolExecutor(
        registry,
        StaticPolicy(PolicyDecision.ALLOW),
    )
    agent = Agent(
        model=AddModel(),
        tools=registry,
        tool_executor=executor,
        max_steps=2,
    )

    assert agent.run("calculate") == "The result is 29"
    assert call_count == 1
    assert agent.messages[2].is_error is False


@pytest.mark.parametrize(
    ("decision", "error_detail"),
    [
        (
            PolicyDecision.DENY,
            "Tool 'add' was denied by policy.",
        ),
        (
            PolicyDecision.REQUIRE_APPROVAL,
            "Tool 'add' requires approval and was not executed.",
        ),
    ],
)
def test_agent_observes_unauthorized_tool_as_error(
    decision,
    error_detail,
):
    call_count = 0

    def add(a, b):
        nonlocal call_count
        call_count += 1
        return a + b

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="Add two values.",
            parameters=ADD_TOOL.parameters,
            function=add,
        )
    )
    executor = ToolExecutor(
        registry,
        StaticPolicy(decision),
    )
    agent = Agent(
        model=AddModel(),
        tools=registry,
        tool_executor=executor,
        max_steps=2,
    )

    result = agent.run("calculate")

    assert result == (
        "The result is Tool error: ToolPolicyError: "
        f"{error_detail}"
    )
    assert call_count == 0
    assert isinstance(agent.messages[2], ToolResult)
    assert agent.messages[2].is_error is True
    assert agent.messages[2].content == (
        f"Tool error: ToolPolicyError: {error_detail}"
    )

    tool_events = [
        event
        for event in agent.events
        if event.type.startswith("tool_")
    ]

    assert [event.type for event in tool_events] == [
        "tool_policy_evaluated",
    ]
    assert tool_events[0].data["decision"] == decision.value


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


def test_agent_sends_compiled_context_items_and_emits_statistics():
    session = Session(
        items=[
            Message(role="user", content="old"),
            Message(role="assistant", content="recent"),
        ]
    )
    model = RecordingModel()
    builder = TokenBudgetContextBuilder(
        ContextBudget(max_estimated_tokens=5),
        FixedMessageEstimator(
            {"old": 10, "recent": 3, "new": 2}
        ),
    )
    agent = Agent(
        model=model,
        session=session,
        context_builder=builder,
    )

    assert agent.run("new") == "done"
    assert model.contexts == [
        [
            Message(role="assistant", content="recent"),
            Message(role="user", content="new"),
        ]
    ]

    context_built = next(
        event
        for event in agent.events
        if event.type == "context_built"
    )

    assert context_built.data == {
        "step": 0,
        "history_item_count": 3,
        "context_item_count": 2,
        "context_strategy": "TokenBudget",
        "estimated_history_tokens": 5,
        "total_units": 3,
        "included_units": 2,
        "dropped_units": 1,
        "history_token_budget": 5,
    }


def test_context_budget_failure_stops_before_model_call():
    model = RecordingModel()
    agent = Agent(
        model=model,
        context_builder=TokenBudgetContextBuilder(
            ContextBudget(max_estimated_tokens=1),
            FixedMessageEstimator({"oversized": 2}),
        ),
    )

    with pytest.raises(
        ContextBudgetExceeded,
        match="Newest indivisible context unit",
    ):
        agent.run("oversized")

    assert model.contexts == []
    assert agent.trace.end_reason == "context_error"
    assert isinstance(agent.messages[0], Message)
    assert agent.messages[0].content == "oversized"
    assert [event.type for event in agent.events] == [
        "agent_started",
        "context_build_started",
        "context_build_failed",
        "agent_failed",
    ]
    assert agent.events[2].data == {
        "step": 0,
        "reason": "context_error",
        "error_type": "ContextBudgetExceeded",
    }
    assert agent.events[3].data["reason"] == "context_error"


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



def test_agent_continues_session_loaded_from_store(
    tmp_path,
):
    original_agent = Agent(
        model=EchoModel(),
    )
    original_agent.run("first")

    store = JsonlSessionStore(tmp_path)
    store.save("task-001", original_agent.session)

    resumed_store = JsonlSessionStore(tmp_path)
    loaded = resumed_store.load("task-001")
    resumed_agent = Agent(
        model=EchoModel(),
        session=loaded,
    )

    result = resumed_agent.run("second")

    assert result == "Echo: second"
    assert len(resumed_agent.session.items) == 4

    assert (
        resumed_agent.session.items[0].content
        == "first"
    )

    assert (
        resumed_agent.session.items[1].content
        == "Echo: first"
    )

    assert (
        resumed_agent.session.items[2].content
        == "second"
    )

    assert (
        resumed_agent.session.items[3].content
        == "Echo: second"
    )
