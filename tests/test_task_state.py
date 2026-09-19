from copy import deepcopy

import pytest

from miniharness.context import ContextBuilder
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.session import Session
from miniharness.session_store import JsonlSessionStore
from miniharness.task_state import (
    RecentTaskError,
    TaskAction,
    TaskStateError,
    TaskStateReducer,
    render_task_state,
)
from miniharness.tool_result_projection import (
    DeterministicToolResultProjector,
    IdentityToolResultProjector,
)


def _tool_interaction(
    name,
    *,
    call_id,
    arguments=None,
    content="ok",
    is_error=False,
):
    return [
        ToolCall(
            name=name,
            arguments=arguments or {},
            call_id=call_id,
        ),
        ToolResult(
            name=name,
            content=content,
            call_id=call_id,
            is_error=is_error,
        ),
    ]


def test_empty_history_produces_empty_task_state():
    state = TaskStateReducer().reduce([])

    assert state.current_request is None
    assert state.completed_actions == ()
    assert state.failed_actions == ()
    assert state.files_read == ()
    assert state.files_modified == ()
    assert state.recent_errors == ()


def test_newest_user_message_is_current_request():
    state = TaskStateReducer().reduce(
        [
            Message(role="user", content="first request"),
            Message(role="assistant", content="response"),
            Message(role="user", content="latest request"),
        ]
    )

    assert state.current_request == "latest request"


def test_one_user_message_is_preserved_as_current_request():
    state = TaskStateReducer().reduce(
        [Message(role="user", content="keep this exact text")]
    )

    assert state.current_request == "keep this exact text"


def test_empty_user_message_is_present_without_invented_text():
    state = TaskStateReducer().reduce(
        [Message(role="user", content="")]
    )

    assert state.current_request == ""
    assert "Current request:\n\n\nFiles read:" in (
        render_task_state(state).content
    )


def test_successful_read_records_action_and_file():
    state = TaskStateReducer().reduce(
        _tool_interaction(
            "read_file",
            call_id="read-1",
            arguments={"path": "./src/agent.py"},
        )
    )

    assert state.completed_actions == (
        TaskAction("read_file", "read-1"),
    )
    assert state.files_read == ("src/agent.py",)


def test_failed_read_records_failure_but_not_file():
    state = TaskStateReducer().reduce(
        _tool_interaction(
            "read_file",
            call_id="read-1",
            arguments={"path": "missing.py"},
            content="not found",
            is_error=True,
        )
    )

    assert state.completed_actions == ()
    assert state.failed_actions == (
        TaskAction("read_file", "read-1"),
    )
    assert state.files_read == ()
    assert state.recent_errors == (
        RecentTaskError("read_file", "read-1", "not found"),
    )


@pytest.mark.parametrize("tool_name", ["write_file", "apply_patch"])
def test_successful_file_mutation_records_modified_path(tool_name):
    state = TaskStateReducer().reduce(
        _tool_interaction(
            tool_name,
            call_id="write-1",
            arguments={"path": "src/../src/model.py"},
        )
    )

    assert state.files_modified == ("src/model.py",)


@pytest.mark.parametrize("tool_name", ["write_file", "apply_patch"])
def test_failed_file_mutation_does_not_record_modified_path(
    tool_name,
):
    state = TaskStateReducer().reduce(
        _tool_interaction(
            tool_name,
            call_id="write-1",
            arguments={"path": "src/model.py"},
            content="failed",
            is_error=True,
        )
    )

    assert state.files_modified == ()
    assert state.failed_actions == (
        TaskAction(tool_name, "write-1"),
    )


def test_duplicate_paths_use_stable_first_seen_order():
    history = [
        *_tool_interaction(
            "read_file",
            call_id="1",
            arguments={"path": "a.py"},
        ),
        Message(role="assistant", content="next"),
        *_tool_interaction(
            "read_file",
            call_id="2",
            arguments={"path": "b.py"},
        ),
        Message(role="assistant", content="next"),
        *_tool_interaction(
            "read_file",
            call_id="3",
            arguments={"path": "./a.py"},
        ),
    ]

    state = TaskStateReducer().reduce(history)

    assert state.files_read == ("a.py", "b.py")


def test_run_command_arguments_do_not_infer_file_state():
    history = [
        *_tool_interaction(
            "run_command",
            call_id="1",
            arguments={"argv": ["cat", "secret.py"]},
        ),
        Message(role="assistant", content="next"),
        *_tool_interaction(
            "run_command",
            call_id="2",
            arguments={"argv": ["rm", "x.py"]},
        ),
    ]

    state = TaskStateReducer().reduce(history)

    assert state.files_read == ()
    assert state.files_modified == ()
    assert len(state.completed_actions) == 2


def test_actions_and_errors_are_recent_bounded_and_deterministic():
    history = []

    for index in range(5):
        history.extend(
            _tool_interaction(
                "run_command",
                call_id=str(index),
                content=("error\n" + "x" * 200),
                is_error=True,
            )
        )
        history.append(Message(role="assistant", content="next"))

    reducer = TaskStateReducer(
        max_failed_actions=3,
        max_recent_errors=2,
        max_error_chars=80,
    )

    first = reducer.reduce(history)
    second = reducer.reduce(history)

    assert first == second
    assert [action.call_id for action in first.failed_actions] == [
        "2",
        "3",
        "4",
    ]
    assert [error.call_id for error in first.recent_errors] == [
        "3",
        "4",
    ]
    assert all(
        len(error.message) <= 80
        for error in first.recent_errors
    )
    assert "middle omitted" in first.recent_errors[0].message


def test_completed_actions_are_recent_bounded():
    history = []

    for index in range(4):
        history.extend(
            _tool_interaction(
                "read_file",
                call_id=str(index),
                arguments={"path": f"{index}.py"},
            )
        )
        history.append(Message(role="assistant", content="next"))

    state = TaskStateReducer(
        max_completed_actions=2
    ).reduce(history)

    assert [action.call_id for action in state.completed_actions] == [
        "2",
        "3",
    ]


def test_current_request_is_bounded_without_mutating_message():
    content = "start" + "x" * 500 + "finish"
    message = Message(role="user", content=content)
    state = TaskStateReducer(
        max_current_request_chars=100
    ).reduce([message])

    assert state.current_request is not None
    assert len(state.current_request) <= 100
    assert state.current_request.startswith("start")
    assert state.current_request.endswith("finish")
    assert message.content == content


def test_reduction_and_rendering_are_deterministic_and_immutable():
    session = Session(
        items=[
            Message(role="user", content="inspect"),
            *_tool_interaction(
                "read_file",
                call_id="1",
                arguments={"path": "a.py"},
            ),
        ]
    )
    before = deepcopy(session.snapshot())
    reducer = TaskStateReducer()

    first = reducer.reduce(session.snapshot())
    second = reducer.reduce(session.snapshot())

    assert first == second
    assert render_task_state(first) == render_task_state(second)
    assert render_task_state(first).role == "system"
    assert session.snapshot() == before


def test_task_state_is_independent_of_context_projection():
    raw_content = "start" + "x" * 1_000 + "finish"
    session = Session(
        items=[
            Message(role="user", content="inspect"),
            *_tool_interaction(
                "read_file",
                call_id="1",
                arguments={"path": "a.py"},
                content=raw_content,
            ),
        ]
    )
    reducer = TaskStateReducer()
    before = reducer.reduce(session.snapshot())

    ContextBuilder(
        tool_result_projector=IdentityToolResultProjector()
    ).compile(session.snapshot())
    identity_state = reducer.reduce(session.snapshot())
    ContextBuilder(
        tool_result_projector=DeterministicToolResultProjector(
            max_chars=100,
            head_chars=20,
            tail_chars=20,
        )
    ).compile(session.snapshot())
    compacted_state = reducer.reduce(session.snapshot())

    assert identity_state == before
    assert compacted_state == before


def test_task_state_rebuilds_after_jsonl_resume(tmp_path):
    session = Session(
        items=[
            Message(role="user", content="fix the file"),
            *_tool_interaction(
                "read_file",
                call_id="1",
                arguments={"path": "a.py"},
            ),
            Message(role="assistant", content="editing"),
            *_tool_interaction(
                "apply_patch",
                call_id="2",
                arguments={"path": "a.py"},
            ),
            Message(role="assistant", content="checking"),
            *_tool_interaction(
                "run_command",
                call_id="3",
                content="tests failed",
                is_error=True,
            ),
        ]
    )
    reducer = TaskStateReducer()
    expected = reducer.reduce(session.snapshot())
    store = JsonlSessionStore(tmp_path)

    store.save("task", session)
    resumed = store.load("task")

    assert reducer.reduce(resumed.snapshot()) == expected


def test_task_state_rejects_malformed_tool_history():
    history = [
        ToolResult(
            name="read_file",
            content="orphan",
            call_id="1",
        )
    ]

    with pytest.raises(
        TaskStateError,
        match="ToolResult has no matching ToolCall",
    ):
        TaskStateReducer().reduce(history)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_current_request_chars": 79},
        {"max_completed_actions": 0},
        {"max_failed_actions": True},
        {"max_recent_errors": -1},
        {"max_error_chars": 79},
    ],
)
def test_task_state_reducer_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        TaskStateReducer(**kwargs)
