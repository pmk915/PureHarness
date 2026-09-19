from datetime import datetime, timezone
from pathlib import Path

import pytest

from miniharness.agent import Agent
from miniharness.approval import ApprovalDecision
from miniharness.cli import main
from miniharness.messages import Message, ToolCall, ToolResult
from miniharness.session import Session
from miniharness.session_store import (
    DurableSession,
    JsonlDurableSessionStore,
)
from miniharness.tool_executor import ToolExecutor
from miniharness.tool_policy import PolicyDecision
from miniharness.tools import Tool, ToolRegistry


class RecordingModel:
    def __init__(self):
        self.contexts = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        user_count = sum(
            isinstance(item, Message) and item.role == "user"
            for item in messages
        )
        return Message(
            role="assistant",
            content=f"response-{user_count}",
        )


class WriteThenCompleteModel:
    def __init__(self):
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1
        if self.call_count == 1:
            return [
                ToolCall(
                    name="write_file",
                    arguments={
                        "path": "marker.txt",
                        "content": "written once",
                    },
                    call_id="write-1",
                )
            ]
        return Message(role="assistant", content="done")


class InterruptingModel:
    def __init__(self):
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1
        raise KeyboardInterrupt


def _input(values):
    iterator = iter(values)
    return lambda prompt: next(iterator)


@pytest.fixture
def durable_environment(tmp_path, monkeypatch):
    home = tmp_path / "miniharness-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MINIHARNESS_HOME", str(home))
    return home, workspace


def _store(home: Path) -> JsonlDurableSessionStore:
    return JsonlDurableSessionStore(home / "sessions")


def _create_session(home, workspace, model=None):
    selected_model = model or RecordingModel()
    output = []
    exit_code = main(
        [
            "--workspace",
            str(workspace),
            "--model",
            "fake-model",
        ],
        model_factory=lambda name: selected_model,
        input_fn=_input(["first turn", "second turn", "/exit"]),
        output_fn=output.append,
    )
    assert exit_code == 0
    summaries = _store(home).list_sessions()
    assert len(summaries) == 1
    return summaries[0].session_id, selected_model, output


def test_interactive_session_persists_two_turns_and_reloads(
    durable_environment,
):
    home, workspace = durable_environment

    session_id, _, _ = _create_session(home, workspace)
    loaded = JsonlDurableSessionStore(
        home / "sessions"
    ).load(session_id)

    assert loaded.session_id == session_id
    assert len(loaded.run_records) == 2
    assert [
        item.content
        for item in loaded.session.items
        if isinstance(item, Message)
    ] == [
        "first turn",
        "response-1",
        "second turn",
        "response-2",
    ]


def test_resume_keeps_session_id_and_creates_a_new_run(
    durable_environment,
):
    home, workspace = durable_environment
    session_id, _, _ = _create_session(home, workspace)
    before = _store(home).load(session_id)

    exit_code = main(
        ["resume", session_id],
        model_factory=lambda name: RecordingModel(),
        input_fn=_input(["resumed turn", "/exit"]),
        output_fn=lambda value: None,
    )
    after = _store(home).load(session_id)

    assert exit_code == 0
    assert after.session_id == before.session_id
    assert len(after.run_records) == 3
    assert after.run_records[-1].run_id not in {
        record.run_id for record in before.run_records
    }
    assert all(
        record.session_id == session_id
        for record in after.run_records
    )


def test_resumed_model_context_contains_prior_conversation(
    durable_environment,
):
    home, workspace = durable_environment
    session_id, _, _ = _create_session(home, workspace)
    resumed_model = RecordingModel()

    exit_code = main(
        ["resume", session_id],
        model_factory=lambda name: resumed_model,
        input_fn=_input(["continue", "/exit"]),
        output_fn=lambda value: None,
    )

    assert exit_code == 0
    assert len(resumed_model.contexts) == 1
    visible_messages = [
        (item.role, item.content)
        for item in resumed_model.contexts[0]
        if isinstance(item, Message) and item.role != "system"
    ]
    assert visible_messages == [
        ("user", "first turn"),
        ("assistant", "response-1"),
        ("user", "second turn"),
        ("assistant", "response-2"),
        ("user", "continue"),
    ]


def test_resume_is_passive_and_does_not_replay_historical_tools(
    durable_environment,
):
    home, workspace = durable_environment
    model = WriteThenCompleteModel()
    output = []
    exit_code = main(
        ["--workspace", str(workspace), "--model", "fake-model"],
        model_factory=lambda name: model,
        input_fn=_input(["write marker", "/exit"]),
        output_fn=output.append,
    )
    assert exit_code == 0
    session_id = _store(home).list_sessions()[0].session_id
    marker = workspace / "marker.txt"
    before = marker.read_text(encoding="utf-8")
    calls_before_resume = model.call_count

    resumed_output = []
    resume_exit = main(
        ["resume", session_id],
        model_factory=lambda name: pytest.fail(
            "resume without a new turn must not create a model"
        ),
        input_fn=_input(["/status", "/exit"]),
        output_fn=resumed_output.append,
    )

    assert resume_exit == 0
    assert model.call_count == calls_before_resume == 2
    assert marker.read_text(encoding="utf-8") == before
    assert "Runs in session: 1" in resumed_output


def test_resumed_slash_commands_need_no_api_key_or_model(
    durable_environment,
    monkeypatch,
):
    home, workspace = durable_environment
    session_id, _, _ = _create_session(home, workspace)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    output = []

    exit_code = main(
        ["resume", session_id],
        model_factory=lambda name: pytest.fail(
            "resumed slash commands must not create a model"
        ),
        input_fn=_input(["/status", "/help", "/exit"]),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert f"Resumed session {session_id}" in output
    assert f"Session ID: {session_id}" in output
    assert output[-1] == "Goodbye."


def test_sessions_command_handles_empty_store_and_lists_saved_sessions(
    durable_environment,
):
    home, workspace = durable_environment
    empty_output = []

    assert main(["sessions"], output_fn=empty_output.append) == 0
    assert empty_output == ["No saved sessions."]

    session_id, _, _ = _create_session(home, workspace)
    output = []
    assert main(["sessions"], output_fn=output.append) == 0
    rendered = "\n".join(output)
    assert "SESSION ID" in rendered
    assert session_id in rendered
    assert "fake-model" in rendered
    assert str(workspace) in rendered
    assert "\t2\t" in rendered


def test_resume_unknown_session_is_a_friendly_error(
    durable_environment,
):
    output = []

    exit_code = main(
        ["resume", "does-not-exist"],
        output_fn=output.append,
    )

    assert exit_code == 2
    assert output == ["Error: Session not found: does-not-exist"]


def test_resume_corrupted_session_fails_closed_without_overwrite(
    durable_environment,
):
    home, _ = durable_environment
    session_directory = home / "sessions"
    session_directory.mkdir(parents=True)
    path = session_directory / "broken.jsonl"
    original = "not-json\n"
    path.write_text(original, encoding="utf-8")
    output = []

    exit_code = main(
        ["resume", "broken"],
        output_fn=output.append,
    )

    assert exit_code == 2
    assert "broken" in output[0]
    assert str(path) in output[0]
    assert "Malformed JSON" in output[0]
    assert path.read_text(encoding="utf-8") == original


def test_resume_missing_workspace_fails_clearly(
    durable_environment,
):
    home, workspace = durable_environment
    now = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
    state = DurableSession(
        session_id="missing-workspace",
        created_at=now,
        updated_at=now,
        workspace=workspace.resolve(),
        model="fake-model",
        session=Session(),
        run_records=[],
    )
    _store(home).save(state)
    workspace.rmdir()
    output = []

    exit_code = main(
        ["resume", state.session_id],
        output_fn=output.append,
    )

    assert exit_code == 2
    assert output == [
        "Error: session workspace no longer exists: "
        f"{state.workspace}"
    ]


def test_ctrl_c_at_prompt_keeps_durable_session_usable(
    durable_environment,
):
    home, workspace = durable_environment
    calls = 0

    def interrupted_then_commands(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        if calls == 2:
            return "/status"
        return "/exit"

    output = []
    exit_code = main(
        ["--workspace", str(workspace)],
        model_factory=lambda name: pytest.fail(
            "prompt commands must not create a model"
        ),
        input_fn=interrupted_then_commands,
        output_fn=output.append,
    )
    summary = _store(home).list_sessions()[0]

    assert exit_code == 0
    assert summary.run_count == 0
    assert "Interrupted. Use /exit to leave." in output
    assert "Runs in session: 0" in output


def test_interrupted_run_is_durable_and_not_replayed_on_resume(
    durable_environment,
):
    home, workspace = durable_environment
    model = InterruptingModel()
    output = []

    exit_code = main(
        ["--workspace", str(workspace), "--model", "fake-model"],
        model_factory=lambda name: model,
        input_fn=_input(["interrupt me", "/exit"]),
        output_fn=output.append,
    )
    summary = _store(home).list_sessions()[0]
    state = _store(home).load(summary.session_id)

    assert exit_code == 0
    assert model.call_count == 1
    assert len(state.run_records) == 1
    assert state.run_records[0].end_reason == "interrupted"
    assert state.session.items == []
    assert "Run interrupted." in output
    assert "[agent] interrupted" in output
    assert not any("failed: interrupted" in line for line in output)

    resume_output = []
    resume_exit = main(
        ["resume", state.session_id],
        model_factory=lambda name: pytest.fail(
            "interrupted history must not be replayed"
        ),
        input_fn=_input(["/status", "/exit"]),
        output_fn=resume_output.append,
    )

    assert resume_exit == 0
    assert model.call_count == 1
    assert "Last end reason: interrupted" in resume_output


def test_durable_files_stay_under_configured_miniharness_home(
    durable_environment,
):
    home, workspace = durable_environment

    exit_code = main(
        ["--workspace", str(workspace)],
        model_factory=lambda name: pytest.fail(
            "exit without a turn must not create a model"
        ),
        input_fn=_input(["/exit"]),
        output_fn=lambda value: None,
    )

    assert exit_code == 0
    files = tuple(home.rglob("*"))
    assert any(path.suffix == ".jsonl" for path in files)
    assert all(path == home or home in path.parents for path in files)


@pytest.mark.parametrize(
    ("approval_decision", "expected_executions"),
    [
        (ApprovalDecision.APPROVE, 1),
        (ApprovalDecision.DENY, 0),
    ],
)
def test_resume_never_replays_historical_approval_gated_action(
    durable_environment,
    approval_decision,
    expected_executions,
):
    home, workspace = durable_environment
    execution_count = 0
    approval_requests = 0

    class RequireApprovalPolicy:
        def evaluate(self, tool, arguments):
            return PolicyDecision.REQUIRE_APPROVAL

    class RecordingHandler:
        def request_approval(self, request):
            nonlocal approval_requests
            approval_requests += 1
            return approval_decision

    class EffectModel:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return [
                    ToolCall(
                        name="effect",
                        arguments={},
                        call_id="historical-approval",
                    )
                ]
            return Message(role="assistant", content="done")

    def effect():
        nonlocal execution_count
        execution_count += 1
        return "executed"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="effect",
            description="Approval-gated historical effect.",
            parameters={"type": "object", "properties": {}},
            function=effect,
            side_effects=True,
        )
    )
    agent = Agent(
        model=EffectModel(),
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            RequireApprovalPolicy(),
            approval_handler=RecordingHandler(),
        ),
        max_steps=2,
        session_id="approval-session",
        run_id_factory=lambda: "approval-run",
    )

    assert agent.run("perform gated action") == "done"
    assert agent.last_run_record is not None
    now = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
    state = DurableSession(
        session_id="approval-session",
        created_at=now,
        updated_at=now,
        workspace=workspace.resolve(),
        model="fake-model",
        session=agent.session,
        run_records=[agent.last_run_record],
    )
    _store(home).save(state)

    loaded = _store(home).load(state.session_id)
    assert loaded.run_records[0].trace.approvals[0].decision is (
        approval_decision
    )
    result = next(
        item
        for item in loaded.session.items
        if isinstance(item, ToolResult)
    )
    assert result.is_error is (
        approval_decision is ApprovalDecision.DENY
    )

    exit_code = main(
        ["resume", state.session_id],
        model_factory=lambda name: pytest.fail(
            "passive resume must not create a model"
        ),
        input_fn=_input(["/status", "/exit"]),
        output_fn=lambda value: None,
    )

    assert exit_code == 0
    assert approval_requests == 1
    assert execution_count == expected_executions


def test_resume_never_retries_approved_tool_interrupted_in_execution(
    durable_environment,
):
    home, workspace = durable_environment
    execution_count = 0
    approval_requests = 0

    class RequireApprovalPolicy:
        def evaluate(self, tool, arguments):
            return PolicyDecision.REQUIRE_APPROVAL

    class ApproveOnce:
        def request_approval(self, request):
            nonlocal approval_requests
            approval_requests += 1
            return ApprovalDecision.APPROVE

    class EffectModel:
        def generate(self, messages, tools):
            return [
                ToolCall(
                    name="uncertain_effect",
                    arguments={},
                    call_id="approved-interrupted",
                )
            ]

    def uncertain_effect():
        nonlocal execution_count
        execution_count += 1
        raise KeyboardInterrupt

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="uncertain_effect",
            description="Interrupt after an uncertain side effect.",
            parameters={"type": "object", "properties": {}},
            function=uncertain_effect,
            side_effects=True,
        )
    )
    agent = Agent(
        model=EffectModel(),
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            RequireApprovalPolicy(),
            approval_handler=ApproveOnce(),
        ),
        session_id="approved-interrupted-session",
        run_id_factory=lambda: "approved-interrupted-run",
    )

    with pytest.raises(KeyboardInterrupt):
        agent.run("perform uncertain effect")

    assert agent.last_run_record is not None
    assert agent.last_run_record.end_reason == "interrupted"
    assert agent.last_run_record.trace.approvals[0].decision is (
        ApprovalDecision.APPROVE
    )
    assert agent.session.items == []
    now = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
    state = DurableSession(
        session_id="approved-interrupted-session",
        created_at=now,
        updated_at=now,
        workspace=workspace.resolve(),
        model="fake-model",
        session=agent.session,
        run_records=[agent.last_run_record],
    )
    _store(home).save(state)

    assert main(
        ["resume", state.session_id],
        model_factory=lambda name: pytest.fail(
            "passive resume must not recreate interrupted work"
        ),
        input_fn=_input(["/status", "/exit"]),
        output_fn=lambda value: None,
    ) == 0
    assert approval_requests == 1
    assert execution_count == 1
