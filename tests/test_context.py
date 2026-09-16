import pytest

from miniharness.context import (
    ContextBuilder,
    RecentContextBuilder,
)
from miniharness.messages import (
    Message,
    ToolCall,
    ToolResult,
)


def test_context_builder_preserves_history():
    history = [
        Message(
            role="user",
            content="hello",
        ),
        Message(
            role="assistant",
            content="hi",
        ),
    ]

    builder = ContextBuilder()

    context = builder.build(history)

    assert context == history
    assert context is not history


def test_recent_context_keeps_recent_user_turn():
    history = [
        Message(
            role="user",
            content="first",
        ),
        Message(
            role="assistant",
            content="first answer",
        ),
        Message(
            role="user",
            content="second",
        ),
        Message(
            role="assistant",
            content="second answer",
        ),
    ]

    builder = RecentContextBuilder(
        max_items=2,
    )

    context = builder.build(history)

    assert len(context) == 2

    assert context[0].content == "second"
    assert context[1].content == "second answer"


def test_recent_context_preserves_tool_turn(): ##不要切断 Tool 过程
    history = [
        Message(
            role="user",
            content="old question",
        ),
        Message(
            role="assistant",
            content="old answer",
        ),
        Message(
            role="user",
            content="calculate",
        ),
        ToolCall(
            name="add",
            arguments={
                "a": 12,
                "b": 17,
            },
            call_id="1",
        ),
        ToolResult(
            name="add",
            content="29",
            call_id="1",
        ),
        Message(
            role="assistant",
            content="The result is 29",
        ),
    ]

    builder = RecentContextBuilder(
        max_items=2,
    )

    context = builder.build(history)

    assert len(context) == 4

    assert isinstance(
        context[0],
        Message,
    )
    assert context[0].content == "calculate"

    assert isinstance(
        context[1],
        ToolCall,
    )

    assert isinstance(
        context[2],
        ToolResult,
    )

    assert isinstance(
        context[3],
        Message,
    )


def test_recent_context_rejects_invalid_max_items():
    with pytest.raises(
        ValueError,
        match="max_items must be greater than 0",
    ):
        RecentContextBuilder(
            max_items=0,
        )