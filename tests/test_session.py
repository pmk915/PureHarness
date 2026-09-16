from miniharness.messages import (
    Message,
    ToolCall,
    ToolResult,
)
from miniharness.session import Session


def test_session_appends_items():
    session = Session()

    message = Message(
        role="user",
        content="hello",
    )

    session.append(message)

    assert session.items == [message]


def test_session_snapshot_returns_copy():
    session = Session()

    session.append(
        Message(
            role="user",
            content="hello",
        )
    )

    snapshot = session.snapshot()

    assert snapshot == session.items
    assert snapshot is not session.items


def test_session_jsonl_round_trip(
    tmp_path,
):
    session = Session()

    session.append(
        Message(
            role="user",
            content="calculate",
        )
    )

    session.append(
        ToolCall(
            name="add",
            arguments={
                "a": 12,
                "b": 17,
            },
            call_id="1",
        )
    )

    session.append(
        ToolResult(
            name="add",
            content="29",
            call_id="1",
            is_error=False,
        )
    )

    session.append(
        Message(
            role="assistant",
            content="The result is 29",
        )
    )

    path = tmp_path / "session.jsonl"

    session.save_jsonl(path)

    loaded = Session.load_jsonl(
        path
    )

    assert loaded.items == session.items



def test_session_jsonl_preserves_multiple_tool_calls(
    tmp_path,
):
    session = Session()

    session.append(
        Message(
            role="user",
            content="inspect files",
        )
    )

    session.append(
        ToolCall(
            name="read_file",
            arguments={
                "path": "agent.py",
            },
            call_id="call-1",
        )
    )

    session.append(
        ToolResult(
            name="read_file",
            content="agent content",
            call_id="call-1",
        )
    )

    session.append(
        ToolCall(
            name="read_file",
            arguments={
                "path": "tools.py",
            },
            call_id="call-2",
        )
    )

    session.append(
        ToolResult(
            name="read_file",
            content="tools content",
            call_id="call-2",
        )
    )

    path = tmp_path / "session.jsonl"

    session.save_jsonl(path)

    loaded = Session.load_jsonl(
        path
    )

    assert loaded.items == session.items

    assert isinstance(
        loaded.items[1],
        ToolCall,
    )

    assert isinstance(
        loaded.items[2],
        ToolResult,
    )

    assert isinstance(
        loaded.items[3],
        ToolCall,
    )

    assert isinstance(
        loaded.items[4],
        ToolResult,
    )



def test_session_jsonl_preserves_tool_error(
    tmp_path,
):
    session = Session()

    session.append(
        ToolResult(
            name="read_file",
            content=(
                "Tool error: "
                "FileNotFoundError"
            ),
            call_id="1",
            is_error=True,
        )
    )

    path = tmp_path / "session.jsonl"

    session.save_jsonl(path)

    loaded = Session.load_jsonl(
        path
    )

    result = loaded.items[0]

    assert isinstance(
        result,
        ToolResult,
    )

    assert result.is_error is True