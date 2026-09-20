from copy import deepcopy

import pytest

from pureharness.agent import Agent
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.model import AddModel
from pureharness.session import Session
from pureharness.task_state import TaskStateReducer
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_policy import PolicyDecision
from pureharness.tool_selection import (
    AllToolsSelector,
    StaticToolSelector,
    ToolSelectionContext,
    ToolSelectionError,
    estimate_tool_schema_tokens,
    prepare_tool_selection,
)
from pureharness.tools import ADD_TOOL, Tool, ToolRegistry


def _tool(name: str, *, description: str = "Test tool.", function=None):
    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
        },
        function=function or (lambda value=None: value or name),
    )


def _selection_context(step: int = 0) -> ToolSelectionContext:
    return ToolSelectionContext(
        step=step,
        task_state=TaskStateReducer().reduce([]),
    )


class CapturingModel:
    def __init__(self, output=None) -> None:
        self.output = output or Message(role="assistant", content="done")
        self.contexts = []
        self.tool_lists = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        self.tool_lists.append(list(tools))
        return self.output


class ToolThenMessageModel:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.tool_lists = []

    def generate(self, messages, tools):
        self.tool_lists.append(list(tools))

        if not any(isinstance(item, ToolResult) for item in messages):
            return [
                ToolCall(
                    name=self.tool_name,
                    arguments={},
                    call_id="call-1",
                )
            ]

        return Message(role="assistant", content="done")


class RecordingSelector:
    strategy = "Recording"

    def __init__(self) -> None:
        self.contexts = []

    def select(self, tools, context):
        self.contexts.append(context)
        return tuple(tools)


class ReturningSelector:
    def __init__(self, selected) -> None:
        self.selected = selected

    def select(self, tools, context):
        del tools, context
        return self.selected


class RecordingPolicy:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        self.calls = []

    def evaluate(self, tool, arguments):
        self.calls.append((tool, arguments))
        return self.decision


def test_all_tools_selector_preserves_registry_order_and_contents():
    tools = (_tool("b"), _tool("a"), _tool("c"))
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    before = registry.list_tools()

    first = AllToolsSelector().select(
        registry.list_tools(),
        _selection_context(),
    )
    second = AllToolsSelector().select(
        registry.list_tools(),
        _selection_context(),
    )

    assert tuple(first) == tools
    assert tuple(second) == tools
    assert registry.list_tools() == before


def test_static_selector_preserves_registry_order_not_config_order():
    tools = (_tool("read"), _tool("write"), _tool("diff"))
    selector = StaticToolSelector(["diff", "read"])

    selected = selector.select(tools, _selection_context())
    repeated = selector.select(tools, _selection_context())

    assert [tool.name for tool in selected] == ["read", "diff"]
    assert tuple(repeated) == tuple(selected)


def test_static_selector_rejects_unknown_and_duplicate_configuration():
    with pytest.raises(ValueError, match="not one string"):
        StaticToolSelector("read")

    with pytest.raises(ValueError, match="must not contain duplicates"):
        StaticToolSelector(["read", "read"])

    selector = StaticToolSelector(["missing"])

    with pytest.raises(ToolSelectionError, match="unknown tool"):
        selector.select([_tool("read")], _selection_context())


def test_static_selector_supports_empty_selection_without_fallback():
    tools = (_tool("read"), _tool("write"))

    selected = StaticToolSelector([]).select(
        tools,
        _selection_context(),
    )

    assert tuple(selected) == ()


def test_custom_selector_output_is_normalized_to_registry_order():
    tools = (_tool("a"), _tool("b"), _tool("c"))
    selection = prepare_tool_selection(
        tools,
        ReturningSelector([tools[2], tools[0]]),
        _selection_context(),
    )

    assert selection.tools == (tools[0], tools[2])


@pytest.mark.parametrize("invalid_kind", ["duplicate", "unregistered"])
def test_custom_selector_rejects_invalid_output(invalid_kind):
    registered = _tool("registered")

    if invalid_kind == "duplicate":
        selected = [registered, registered]
        match = "duplicate tool"
    else:
        selected = [_tool("registered")]
        match = "unregistered Tool instance"

    with pytest.raises(ToolSelectionError, match=match):
        prepare_tool_selection(
            [registered],
            ReturningSelector(selected),
            _selection_context(),
        )


def test_schema_estimation_is_deterministic_and_size_sensitive():
    small = _tool("small", description="short")
    large = _tool("large", description="large " * 100)

    small_first = estimate_tool_schema_tokens([small])
    small_second = estimate_tool_schema_tokens([small])
    large_estimate = estimate_tool_schema_tokens([large])
    both_forward = estimate_tool_schema_tokens([small, large])
    both_reverse = estimate_tool_schema_tokens([large, small])

    assert small_first == small_second
    assert large_estimate > small_first
    assert both_forward > large_estimate
    assert both_forward == both_reverse
    assert estimate_tool_schema_tokens([]) == 0


def test_selection_metrics_measure_registered_and_exposed_schemas():
    tools = (_tool("a"), _tool("b", description="b" * 200))
    selection = prepare_tool_selection(
        tools,
        StaticToolSelector(["a"]),
        _selection_context(),
    )

    assert selection.registered_tool_count == 2
    assert selection.exposed_tool_count == 1
    assert selection.estimated_tool_schema_tokens > 0
    assert selection.estimated_all_tool_schema_tokens > (
        selection.estimated_tool_schema_tokens
    )
    assert selection.estimated_tool_schema_tokens_saved == (
        selection.estimated_all_tool_schema_tokens
        - selection.estimated_tool_schema_tokens
    )
    assert selection.selector_strategy == "StaticNames"


def test_agent_exposes_only_selected_tools_and_reports_metrics():
    registry = ToolRegistry()
    tools = [_tool(name) for name in ("a", "b", "c", "d")]
    for tool in tools:
        registry.register(tool)
    model = CapturingModel()
    agent = Agent(
        model=model,
        tools=registry,
        tool_selector=StaticToolSelector(["a", "c"]),
    )

    assert agent.run("hello") == "done"
    assert [tool.name for tool in model.tool_lists[0]] == ["a", "c"]
    event = next(
        event
        for event in agent.events
        if event.type == "model_started"
    )
    assert event.data["registered_tool_count"] == 4
    assert event.data["exposed_tool_count"] == 2
    assert event.data["estimated_tool_schema_tokens"] > 0
    assert event.data["estimated_tool_schema_tokens_saved"] > 0
    assert event.data["selector_strategy"] == "StaticNames"


def test_default_all_tools_selector_preserves_existing_agent_behavior():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    model = CapturingModel()

    Agent(model=model, tools=registry).run("hello")

    assert model.tool_lists == [[ADD_TOOL]]


def test_agent_invokes_selector_for_every_inference():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    selector = RecordingSelector()
    agent = Agent(
        model=AddModel(),
        tools=registry,
        tool_selector=selector,
        max_steps=2,
    )

    assert agent.run("calculate") == "The result is 29"
    assert [context.step for context in selector.contexts] == [0, 1]
    assert selector.contexts[0].task_state.current_request == "calculate"
    assert selector.contexts[1].task_state.completed_actions


def test_hidden_tool_call_becomes_error_before_policy_or_side_effect():
    side_effects = 0

    def hidden_tool():
        nonlocal side_effects
        side_effects += 1
        return "executed"

    visible = _tool("visible")
    hidden = _tool("hidden", function=hidden_tool)
    registry = ToolRegistry()
    registry.register(visible)
    registry.register(hidden)
    policy = RecordingPolicy(PolicyDecision.ALLOW)
    model = ToolThenMessageModel("hidden")
    agent = Agent(
        model=model,
        tools=registry,
        tool_executor=ToolExecutor(registry, policy),
        tool_selector=StaticToolSelector(["visible"]),
        max_steps=2,
    )

    assert agent.run("try hidden") == "done"
    assert side_effects == 0
    assert policy.calls == []
    result = next(
        item
        for item in agent.messages
        if isinstance(item, ToolResult)
    )
    assert result.is_error is True
    assert "ToolNotExposedError" in result.content
    assert "was not exposed" in result.content
    assert not any(
        event.type == "tool_policy_evaluated"
        for event in agent.events
    )


def test_exposed_tool_call_still_requires_policy_authorization():
    side_effects = 0

    def exposed_tool():
        nonlocal side_effects
        side_effects += 1
        return "executed"

    tool = _tool("exposed", function=exposed_tool)
    registry = ToolRegistry()
    registry.register(tool)
    policy = RecordingPolicy(PolicyDecision.DENY)
    agent = Agent(
        model=ToolThenMessageModel("exposed"),
        tools=registry,
        tool_executor=ToolExecutor(registry, policy),
        tool_selector=StaticToolSelector(["exposed"]),
        max_steps=2,
    )

    assert agent.run("try exposed") == "done"
    assert side_effects == 0
    assert len(policy.calls) == 1
    result = next(
        item
        for item in agent.messages
        if isinstance(item, ToolResult)
    )
    assert result.is_error is True
    assert "ToolPolicyError" in result.content
    assert any(
        event.type == "tool_policy_evaluated"
        and event.data["decision"] == "deny"
        for event in agent.events
    )


def test_empty_selector_agent_calls_model_without_tools():
    registry = ToolRegistry()
    registry.register(_tool("registered"))
    model = CapturingModel()
    agent = Agent(
        model=model,
        tools=registry,
        tool_selector=StaticToolSelector([]),
    )

    assert agent.run("tool-free") == "done"
    assert model.tool_lists == [[]]
    event = next(
        event
        for event in agent.events
        if event.type == "model_started"
    )
    assert event.data["registered_tool_count"] == 1
    assert event.data["exposed_tool_count"] == 0
    assert event.data["estimated_tool_schema_tokens"] == 0


def test_selection_failure_prevents_model_request_and_emits_failure():
    registry = ToolRegistry()
    registry.register(_tool("registered"))
    model = CapturingModel()
    agent = Agent(
        model=model,
        tools=registry,
        tool_selector=StaticToolSelector(["missing"]),
    )

    with pytest.raises(ToolSelectionError, match="unknown tool"):
        agent.run("hello")

    assert model.contexts == []
    assert agent.trace.end_reason == "tool_selection_error"
    assert not any(
        event.type == "model_started"
        for event in agent.events
    )
    assert agent.events[-1].type == "agent_failed"
    assert agent.events[-1].data == {
        "reason": "tool_selection_error",
        "error_type": "ToolSelectionError",
        "step_count": 0,
    }


def test_selector_changes_only_tool_definitions_not_model_context():
    initial_items = [
        Message(role="user", content="earlier"),
        Message(role="assistant", content="response"),
    ]
    registry = ToolRegistry()
    registry.register(_tool("a"))
    registry.register(_tool("b"))
    all_model = CapturingModel()
    static_model = CapturingModel()
    all_session = Session(items=deepcopy(initial_items))
    static_session = Session(items=deepcopy(initial_items))

    Agent(
        model=all_model,
        tools=registry,
        session=all_session,
        tool_selector=AllToolsSelector(),
    ).run("latest")
    Agent(
        model=static_model,
        tools=registry,
        session=static_session,
        tool_selector=StaticToolSelector(["a"]),
    ).run("latest")

    assert all_model.contexts == static_model.contexts
    assert all_session.items == static_session.items
    assert [tool.name for tool in all_model.tool_lists[0]] == ["a", "b"]
    assert [tool.name for tool in static_model.tool_lists[0]] == ["a"]
