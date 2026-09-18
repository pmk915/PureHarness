import json

import pytest

import miniharness.session_store as session_store_module
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.session import Session
from miniharness.session_store import (
    JsonlSessionStore,
    MemorySessionStore,
    SessionStore,
    SessionStoreError,
)


def test_memory_store_save_and_load():
    store: SessionStore = MemorySessionStore()
    session = Session(
        items=[
            Message(
                role="user",
                content="hello",
            )
        ]
    )

    store.save("task-001", session)

    assert store.load("task-001").items == session.items


def test_memory_store_keeps_session_ids_separate():
    store = MemorySessionStore()
    first = Session(
        items=[Message(role="user", content="first")]
    )
    second = Session(
        items=[Message(role="user", content="second")]
    )

    store.save("first", first)
    store.save("second", second)

    assert store.load("first").items == first.items
    assert store.load("second").items == second.items


def test_memory_store_isolates_saved_and_loaded_mutable_state():
    store = MemorySessionStore()
    session = Session(
        items=[
            ToolCall(
                name="inspect",
                arguments={
                    "options": {
                        "depth": 1,
                    }
                },
                call_id="call-1",
            )
        ]
    )

    store.save("task", session)
    session.items.clear()

    loaded = store.load("task")
    call = loaded.items[0]
    assert isinstance(call, ToolCall)
    options = call.arguments["options"]
    assert isinstance(options, dict)
    options["depth"] = 9
    loaded.items.append(
        Message(role="assistant", content="changed")
    )

    stored = store.load("task")
    stored_call = stored.items[0]
    assert isinstance(stored_call, ToolCall)
    assert stored_call.arguments == {
        "options": {
            "depth": 1,
        }
    }
    assert len(stored.items) == 1


def test_memory_store_missing_session_fails_clearly():
    store = MemorySessionStore()

    with pytest.raises(
        SessionStoreError,
        match="Session not found: missing",
    ):
        store.load("missing")


def test_memory_store_wraps_copy_failure():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise ValueError("cannot copy")

    store = MemorySessionStore()
    session = Session(
        items=[
            ToolCall(
                name="test",
                arguments={"value": Uncopyable()},
            )
        ]
    )

    with pytest.raises(
        SessionStoreError,
        match="Could not save session 'task': cannot copy",
    ):
        store.save("task", session)


def test_jsonl_store_round_trips_all_agent_item_types(
    tmp_path,
):
    store = JsonlSessionStore(tmp_path)
    session = Session(
        items=[
            Message(
                role="user",
                content="检查两个文件",
            ),
            ToolCall(
                name="read_file",
                arguments={"path": "agent.py"},
                call_id="call-1",
            ),
            ToolResult(
                name="read_file",
                content="agent 内容",
                call_id="call-1",
            ),
            ToolCall(
                name="read_file",
                arguments={"path": "tools.py"},
                call_id="call-2",
            ),
            ToolResult(
                name="read_file",
                content="Tool error: 文件不存在",
                call_id="call-2",
                is_error=True,
            ),
        ]
    )

    store.save("task-001", session)
    loaded = store.load("task-001")

    assert loaded.items == session.items
    assert isinstance(loaded.items[0], Message)
    assert isinstance(loaded.items[1], ToolCall)
    assert isinstance(loaded.items[3], ToolCall)
    assert isinstance(loaded.items[4], ToolResult)
    assert loaded.items[4].is_error is True

    text = (tmp_path / "task-001.jsonl").read_text(
        encoding="utf-8"
    )
    assert "检查两个文件" in text
    assert "\\u68c0" not in text


def test_jsonl_store_writes_version_metadata_not_session_item(
    tmp_path,
):
    store = JsonlSessionStore(tmp_path)
    session = Session(
        items=[Message(role="user", content="hello")]
    )

    store.save("task", session)

    lines = (tmp_path / "task.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert json.loads(lines[0]) == {
        "type": "session_meta",
        "schema_version": 1,
    }
    assert len(store.load("task").items) == 1


def test_jsonl_store_replaces_complete_snapshot(
    tmp_path,
):
    store = JsonlSessionStore(tmp_path)
    session = Session(
        items=[Message(role="user", content="first")]
    )
    store.save("task", session)

    session.append(
        Message(role="assistant", content="second")
    )
    store.save("task", session)

    loaded = store.load("task")
    lines = (tmp_path / "task.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert loaded.items == session.items
    assert len(lines) == 3


def test_jsonl_store_keeps_session_ids_separate(
    tmp_path,
):
    store = JsonlSessionStore(tmp_path)
    first = Session(
        items=[Message(role="user", content="first")]
    )
    second = Session(
        items=[Message(role="user", content="second")]
    )

    store.save("first", first)
    store.save("second", second)

    assert store.load("first").items == first.items
    assert store.load("second").items == second.items


def test_jsonl_store_creates_parent_directory(
    tmp_path,
):
    directory = tmp_path / "nested" / "sessions"
    store = JsonlSessionStore(directory)

    store.save("task", Session())

    assert (directory / "task.jsonl").is_file()


def test_jsonl_store_missing_session_fails_clearly(
    tmp_path,
):
    store = JsonlSessionStore(tmp_path)

    with pytest.raises(
        SessionStoreError,
        match="Could not load session 'missing'",
    ):
        store.load("missing")


def test_jsonl_store_rejects_unsupported_schema(
    tmp_path,
):
    path = tmp_path / "task.jsonl"
    path.write_text(
        '{"type": "session_meta", "schema_version": 2}\n',
        encoding="utf-8",
    )
    store = JsonlSessionStore(tmp_path)

    with pytest.raises(
        SessionStoreError,
        match="Unsupported session schema version: 2",
    ):
        store.load("task")


@pytest.mark.parametrize(
    "contents, message",
    [
        ("not-json\n", "Malformed JSON on line 1"),
        (
            '{"type": "message", "role": "user", '
            '"content": "missing metadata"}\n',
            "First session record must be session metadata",
        ),
        (
            '{"type": "session_meta", "schema_version": 1}\n'
            '{"type": "message", "role": 3, "content": "bad"}\n',
            "Session field 'role' must be a string",
        ),
    ],
)
def test_jsonl_store_rejects_malformed_data(
    tmp_path,
    contents,
    message,
):
    (tmp_path / "task.jsonl").write_text(
        contents,
        encoding="utf-8",
    )
    store = JsonlSessionStore(tmp_path)

    with pytest.raises(
        SessionStoreError,
        match=message,
    ):
        store.load("task")


def test_jsonl_store_preserves_previous_file_when_replace_fails(
    tmp_path,
    monkeypatch,
):
    store = JsonlSessionStore(tmp_path)
    original = Session(
        items=[Message(role="user", content="original")]
    )
    store.save("task", original)
    changed = Session(
        items=[Message(role="user", content="changed")]
    )

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        session_store_module.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(
        SessionStoreError,
        match="simulated replace failure",
    ):
        store.save("task", changed)

    assert store.load("task").items == original.items
    assert list(tmp_path.glob("*.tmp")) == []
