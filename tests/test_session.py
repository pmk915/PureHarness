from miniharness.messages import Message
from miniharness.session import Session


def test_session_appends_items():
    session = Session()
    message = Message(
        role="user",
        content="hello",
    )

    session.append(message)

    assert session.items == [message]


def test_session_snapshot_returns_isolated_list():
    session = Session()
    message = Message(
        role="user",
        content="hello",
    )
    session.append(message)

    snapshot = session.snapshot()
    snapshot.append(
        Message(
            role="assistant",
            content="new",
        )
    )

    assert snapshot is not session.items
    assert session.items == [message]


def test_session_does_not_own_jsonl_persistence():
    assert not hasattr(Session, "save_jsonl")
    assert not hasattr(Session, "load_jsonl")
