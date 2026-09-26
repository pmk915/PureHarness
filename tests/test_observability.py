import json

from datetime import datetime, timedelta, timezone
from typing import get_args

import pytest

import pureharness.cli as cli_module
import pureharness.observability as observability_module
from pureharness.agent import Agent
from pureharness.approval import AutoApproveApprovalHandler
from pureharness.cli import main
from pureharness.events import AgentEvent, AgentEventType
from pureharness.messages import Message, ToolCall
from pureharness.model import EchoModel
from pureharness.observability import (
    EventSerializationError,
    JsonlEventRenderer,
    event_to_wire,
)
from pureharness.session import Session
from pureharness.session_store import (
    DurableSession,
    JsonlDurableSessionStore,
)
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_policy import PolicyDecision
from pureharness.tools import Tool, ToolRegistry


class ListThenCompleteModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return [
                ToolCall(
                    name="list_files",
                    arguments={"path": "."},
                    call_id="list-1",
                )
            ]
        return Message(role="assistant", content="done")


class InterruptingModel:
    def generate(self, messages, tools):
        raise KeyboardInterrupt


class StaticPolicy:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision

    def evaluate(self, tool, arguments):
        return self.decision


class EffectThenCompleteModel:
    def __init__(self, arguments=None) -> None:
        self.arguments = arguments or {}
        self.calls = 0

    def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return [
                ToolCall(
                    name="effect",
                    arguments=self.arguments,
                    call_id="effect-1",
                )
            ]
        return Message(role="assistant", content="done")


def _parse_jsonl(value: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in value.splitlines() if line]


def _approval_agent(handler, output, *, arguments=None):
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="effect",
            description="Approval-gated test effect.",
            parameters={"type": "object", "properties": {}},
            function=lambda **kwargs: "executed",
            side_effects=True,
        )
    )
    return Agent(
        model=EffectThenCompleteModel(arguments),
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            StaticPolicy(PolicyDecision.REQUIRE_APPROVAL),
            approval_handler=handler,
        ),
        listeners=[JsonlEventRenderer(output.append)],
        max_steps=2,
        session_id="session-wire",
        run_id_factory=lambda: "run-wire",
    )


def test_wire_serializer_has_stable_envelope_and_tool_payload():
    event = AgentEvent(
        type="tool_started",
        data={
            "step": 3,
            "name": "shell",
            "call_id": "call-1",
            "arguments_preview": {"argv": '["pytest"]'},
        },
        timestamp=datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc),
        run_id="run-1",
        session_id="session-1",
    )

    assert event_to_wire(event) == {
        "schema_version": 1,
        "event": "tool_started",
        "timestamp": "2026-09-20T10:00:00Z",
        "run_id": "run-1",
        "session_id": "session-1",
        "step": 3,
        "payload": {
            "tool_name": "shell",
            "call_id": "call-1",
            "arguments_preview": {"argv": '["pytest"]'},
        },
    }


def test_wire_mapping_explicitly_covers_every_public_event_type():
    assert observability_module.SUPPORTED_EVENT_TYPES == frozenset(
        get_args(AgentEventType)
    )


def test_wire_serializer_rejects_unknown_event_and_missing_run_id():
    with pytest.raises(EventSerializationError, match="Unsupported"):
        event_to_wire(AgentEvent(type="unknown", data={}))
    with pytest.raises(EventSerializationError, match="missing run_id"):
        event_to_wire(AgentEvent(type="agent_started", data={}))


def test_one_shot_jsonl_stdout_is_pure_and_versioned(tmp_path, capsys):
    exit_code = main(
        [
            "run",
            "hello",
            "--workspace",
            str(tmp_path),
            "--model",
            "fake-model",
            "--output",
            "jsonl",
        ],
        model_factory=lambda name: EchoModel(),
    )
    captured = capsys.readouterr()
    events = _parse_jsonl(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert events
    assert events[0]["event"] == "agent_started"
    assert events[-1]["event"] == "agent_completed"
    assert {event["schema_version"] for event in events} == {1}
    assert len({event["run_id"] for event in events}) == 1
    for event in events:
        timestamp = str(event["timestamp"])
        assert timestamp.endswith("Z")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert "PureHarness" not in captured.out
    assert "[model]" not in captured.out
    assert "Echo:" not in captured.out


def test_jsonl_tool_events_preserve_runtime_order(tmp_path, capsys):
    (tmp_path / "sample.txt").write_text("sample", encoding="utf-8")

    assert main(
        [
            "run",
            "list files",
            "--workspace",
            str(tmp_path),
            "--model",
            "fake-model",
            "--output",
            "jsonl",
        ],
        model_factory=lambda name: ListThenCompleteModel(),
    ) == 0
    events = _parse_jsonl(capsys.readouterr().out)
    names = [event["event"] for event in events]

    assert names.index("tool_started") < names.index("tool_completed")
    started = next(
        event for event in events if event["event"] == "tool_started"
    )
    assert started["step"] == 0
    assert started["payload"]["tool_name"] == "list_files"
    assert started["payload"]["call_id"] == "list-1"


def test_jsonl_approval_order_and_redaction():
    output = []
    agent = _approval_agent(
        AutoApproveApprovalHandler(),
        output,
        arguments={"api_key": "plain-secret", "target": "artifact"},
    )

    assert agent.run("perform effect") == "done"
    events = [json.loads(line) for line in output]
    names = [event["event"] for event in events]

    assert names.index("approval_requested") < names.index(
        "approval_granted"
    ) < names.index("tool_started") < names.index("tool_completed")
    assert "plain-secret" not in "\n".join(output)
    requested = next(
        event for event in events if event["event"] == "approval_requested"
    )
    assert requested["payload"]["arguments_preview"]["api_key"] == (
        "[REDACTED]"
    )


def test_jsonl_approval_denial_has_no_execution_event():
    output = []
    agent = _approval_agent(None, output)

    assert agent.run("perform effect") == "done"
    names = [json.loads(line)["event"] for line in output]

    assert "approval_requested" in names
    assert "approval_denied" in names
    assert "tool_started" not in names
    assert "tool_completed" not in names


def test_one_shot_jsonl_approval_denial_never_prompts_or_executes(
    tmp_path,
    monkeypatch,
    capsys,
):
    execution_count = 0

    def create_agent(
        workspace,
        model,
        *,
        session_id,
        output_fn,
        session=None,
        approval_handler=None,
        event_listener=None,
        max_steps=10,
    ):
        nonlocal execution_count
        assert approval_handler is None
        assert event_listener is not None

        def effect():
            nonlocal execution_count
            execution_count += 1

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="effect",
                description="Approval-gated effect.",
                parameters={"type": "object", "properties": {}},
                function=effect,
                side_effects=True,
            )
        )
        return Agent(
            model=model,
            tools=registry,
            tool_executor=ToolExecutor(
                registry,
                StaticPolicy(PolicyDecision.REQUIRE_APPROVAL),
            ),
            listeners=[event_listener],
            session_id=session_id,
            session=session,
            max_steps=max_steps,
        )

    monkeypatch.setattr(cli_module, "_create_agent", create_agent)
    assert main(
        [
            "run",
            "perform effect",
            "--workspace",
            str(tmp_path),
            "--output",
            "jsonl",
        ],
        model_factory=lambda name: EffectThenCompleteModel(),
    ) == 0
    captured = capsys.readouterr()
    names = [event["event"] for event in _parse_jsonl(captured.out)]

    assert execution_count == 0
    assert "approval_requested" in names
    assert "approval_denied" in names
    assert "tool_started" not in names
    assert "Approval required" not in captured.out + captured.err


def test_jsonl_renderer_failure_is_an_application_error(tmp_path):
    errors = []

    def broken_output(value):
        raise RuntimeError("output unavailable")

    exit_code = main(
        [
            "run",
            "hello",
            "--workspace",
            str(tmp_path),
            "--output",
            "jsonl",
        ],
        model_factory=lambda name: EchoModel(),
        output_fn=broken_output,
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert errors == [
        "Error: JSONL event output failed: "
        "RuntimeError: output unavailable"
    ]


def test_jsonl_interruption_is_structured_without_human_text(
    tmp_path,
    capsys,
):
    exit_code = main(
        [
            "run",
            "interrupt",
            "--workspace",
            str(tmp_path),
            "--model",
            "fake-model",
            "--output",
            "jsonl",
        ],
        model_factory=lambda name: InterruptingModel(),
    )
    captured = capsys.readouterr()
    events = _parse_jsonl(captured.out)

    assert exit_code == 130
    assert events[-1]["event"] == "agent_interrupted"
    assert events[-1]["payload"]["end_reason"] == "interrupted"
    assert "Interrupted." not in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_default_one_shot_output_remains_human(tmp_path, capsys):
    assert main(
        ["run", "hello", "--workspace", str(tmp_path)],
        model_factory=lambda name: EchoModel(),
    ) == 0
    captured = capsys.readouterr()

    assert "[model] request" in captured.out
    assert "Echo: hello" in captured.out
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.out.splitlines()[0])


def test_inspect_json_is_one_run_record_document(tmp_path, capsys):
    agent = Agent(
        model=EchoModel(),
        session_id="inspect-session",
        run_id_factory=lambda: "inspect-run",
    )
    agent.run("hello")
    assert agent.last_run_record is not None
    path = tmp_path / "run.json"
    path.write_text(agent.last_run_record.to_json(), encoding="utf-8")

    assert main(["inspect", str(path), "--json"]) == 0
    captured = capsys.readouterr()
    value = json.loads(captured.out)

    assert captured.err == ""
    assert value["schema_version"] == 1
    assert value["run_id"] == "inspect-run"
    assert value["session_id"] == "inspect-session"
    assert value["end_reason"] == "completed"


def test_inspect_human_output_remains_a_summary(tmp_path, capsys):
    agent = Agent(model=EchoModel(), run_id_factory=lambda: "human-run")
    agent.run("hello")
    assert agent.last_run_record is not None
    path = tmp_path / "run.json"
    path.write_text(agent.last_run_record.to_json(), encoding="utf-8")

    assert main(["inspect", str(path)]) == 0
    captured = capsys.readouterr()

    assert "Run ID: human-run" in captured.out
    assert "End reason: completed" in captured.out


def test_inspect_json_error_uses_only_stderr(tmp_path, capsys):
    missing = tmp_path / "missing.json"

    assert main(["inspect", str(missing), "--json"]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "Error: Could not read RunRecord" in captured.err
    assert "Traceback" not in captured.err


def test_sessions_json_is_newest_first_and_versioned(
    tmp_path,
    monkeypatch,
    capsys,
):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("PUREHARNESS_HOME", str(home))
    store = JsonlDurableSessionStore(home / "sessions")
    now = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
    for session_id, offset in (("older", 0), ("newer", 1)):
        timestamp = now + timedelta(hours=offset)
        store.save(
            DurableSession(
                session_id=session_id,
                created_at=timestamp,
                updated_at=timestamp,
                workspace=workspace.resolve(),
                model=f"model-{session_id}",
                session=Session(),
                run_records=[],
            )
        )

    assert main(["sessions", "--json"]) == 0
    captured = capsys.readouterr()
    value = json.loads(captured.out)

    assert captured.err == ""
    assert value["schema_version"] == 1
    assert [item["session_id"] for item in value["sessions"]] == [
        "newer",
        "older",
    ]
    assert value["sessions"][0] == {
        "session_id": "newer",
        "created_at": "2026-09-20T11:00:00Z",
        "updated_at": "2026-09-20T11:00:00Z",
        "workspace": str(workspace.resolve()),
        "model": "model-newer",
        "run_count": 0,
        "last_run_id": None,
        "last_end_reason": None,
    }


def test_empty_sessions_json_contains_no_human_prose(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("PUREHARNESS_HOME", str(tmp_path / "empty-home"))

    assert main(["sessions", "--json"]) == 0
    captured = capsys.readouterr()

    assert json.loads(captured.out) == {
        "schema_version": 1,
        "sessions": [],
    }
    assert "No saved sessions" not in captured.out
    assert captured.err == ""
