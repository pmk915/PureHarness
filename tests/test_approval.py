import pytest

from pureharness.agent import Agent
from pureharness.approval import (
    ApprovalDecision,
    ApprovalRequest,
    AutoApproveApprovalHandler,
    AutoDenyApprovalHandler,
    ToolApprovalError,
)
from pureharness.cli import main
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.run_record import RunRecord
from pureharness.replay import replay_run
from pureharness.terminal_approval import TerminalApprovalHandler
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_policy import (
    PolicyDecision,
    ToolPolicyError,
)
from pureharness.tools import Tool, ToolRegistry


class StaticPolicy:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision

    def evaluate(self, tool, arguments):
        return self.decision


class RecordingApprovalHandler:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    def request_approval(self, request):
        self.requests.append(request)
        return self.decision


class ToolThenMessageModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return [
                ToolCall(
                    name="effect",
                    arguments={"target": "artifact"},
                    call_id="approval-call",
                )
            ]
        return Message(role="assistant", content="done")


def _executor(policy_decision, approval_handler=None):
    executions = []

    def effect(target):
        executions.append(target)
        return f"changed {target}"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="effect",
            description="Perform a visible test effect.",
            parameters={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            function=effect,
            side_effects=True,
        )
    )
    return (
        registry,
        ToolExecutor(
            registry,
            StaticPolicy(policy_decision),
            approval_handler=approval_handler,
        ),
        executions,
    )


def test_allow_bypasses_approval_handler_and_executes_once():
    handler = RecordingApprovalHandler(ApprovalDecision.DENY)
    _, executor, executions = _executor(
        PolicyDecision.ALLOW,
        handler,
    )

    assert executor.execute("effect", {"target": "a"}) == "changed a"
    assert handler.requests == []
    assert executions == ["a"]


def test_policy_deny_bypasses_approval_handler_and_does_not_execute():
    handler = RecordingApprovalHandler(ApprovalDecision.APPROVE)
    _, executor, executions = _executor(
        PolicyDecision.DENY,
        handler,
    )

    with pytest.raises(ToolPolicyError, match="denied by policy"):
        executor.execute("effect", {"target": "a"})

    assert handler.requests == []
    assert executions == []


def test_required_approval_executes_once_after_explicit_approval():
    handler = RecordingApprovalHandler(ApprovalDecision.APPROVE)
    _, executor, executions = _executor(
        PolicyDecision.REQUIRE_APPROVAL,
        handler,
    )

    assert executor.execute("effect", {"target": "a"}) == "changed a"
    assert len(handler.requests) == 1
    assert handler.requests[0].tool_name == "effect"
    assert handler.requests[0].arguments == {"target": "a"}
    assert executions == ["a"]


def test_required_approval_denial_is_structured_and_not_executed():
    handler = RecordingApprovalHandler(ApprovalDecision.DENY)
    _, executor, executions = _executor(
        PolicyDecision.REQUIRE_APPROVAL,
        handler,
    )

    with pytest.raises(ToolApprovalError, match="denied during approval"):
        executor.execute("effect", {"target": "a"})

    assert len(handler.requests) == 1
    assert executions == []


def test_required_approval_without_handler_fails_closed():
    _, executor, executions = _executor(
        PolicyDecision.REQUIRE_APPROVAL
    )

    with pytest.raises(
        ToolApprovalError,
        match="no approval handler was configured",
    ):
        executor.execute("effect", {"target": "a"})

    assert executions == []


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES"])
def test_terminal_handler_accepts_only_explicit_yes_values(answer):
    prompts = []
    handler = TerminalApprovalHandler(
        input_fn=lambda prompt: prompts.append(prompt) or answer,
        output_fn=lambda value: None,
    )

    decision = handler.request_approval(
        ApprovalRequest("effect", {"target": "a"})
    )

    assert decision is ApprovalDecision.APPROVE
    assert prompts == ["Approve this action? [y/N]: "]


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", "foo", "1"])
def test_terminal_handler_defaults_every_other_input_to_deny(answer):
    handler = TerminalApprovalHandler(
        input_fn=lambda prompt: answer,
        output_fn=lambda value: None,
    )

    assert handler.request_approval(
        ApprovalRequest("effect", {})
    ) is ApprovalDecision.DENY


def test_terminal_handler_redacts_sensitive_arguments():
    output = []
    handler = TerminalApprovalHandler(
        input_fn=lambda prompt: "n",
        output_fn=output.append,
    )

    handler.request_approval(
        ApprovalRequest(
            "effect",
            {"api_key": "secret-value", "target": "a"},
        )
    )

    rendered = "\n".join(output)
    assert "api_key: [REDACTED]" in rendered
    assert "secret-value" not in rendered


def test_ctrl_c_during_terminal_approval_is_a_safe_denial():
    output = []

    def interrupt(prompt):
        raise KeyboardInterrupt

    handler = TerminalApprovalHandler(
        input_fn=interrupt,
        output_fn=output.append,
    )

    assert handler.request_approval(
        ApprovalRequest("effect", {})
    ) is ApprovalDecision.DENY
    assert output[-1] == "Approval cancelled; action denied."


@pytest.mark.parametrize(
    ("handler", "decision", "executed"),
    [
        (
            AutoApproveApprovalHandler(),
            ApprovalDecision.APPROVE,
            True,
        ),
        (AutoDenyApprovalHandler(), ApprovalDecision.DENY, False),
    ],
)
def test_agent_records_ordered_approval_events_and_evidence(
    handler,
    decision,
    executed,
    tmp_path,
):
    registry, executor, executions = _executor(
        PolicyDecision.REQUIRE_APPROVAL,
        handler,
    )
    agent = Agent(
        model=ToolThenMessageModel(),
        tools=registry,
        tool_executor=executor,
        max_steps=2,
        session_id="approval-session",
        run_id_factory=lambda: "approval-run",
    )

    assert agent.run("perform effect") == "done"
    event_types = [
        event.type
        for event in agent.events
        if event.type
        in {
            "tool_policy_evaluated",
            "approval_requested",
            "approval_granted",
            "approval_denied",
            "tool_started",
            "tool_completed",
        }
    ]
    expected = [
        "tool_policy_evaluated",
        "approval_requested",
        (
            "approval_granted"
            if decision is ApprovalDecision.APPROVE
            else "approval_denied"
        ),
    ]
    if executed:
        expected.extend(["tool_started", "tool_completed"])
    assert event_types == expected

    approval_event = next(
        event
        for event in agent.events
        if event.type == "approval_requested"
    )
    assert approval_event.data["run_id"] == "approval-run"
    assert approval_event.run_id == "approval-run"
    assert approval_event.data["call_id"] == "approval-call"
    assert executions == (["artifact"] if executed else [])

    record = agent.last_run_record
    assert record is not None
    assert record.end_reason == "completed"
    assert record.tool_call_count == 1
    assert record.tool_execution_count == int(executed)
    assert record.tool_result_error_count == int(not executed)
    assert len(record.trace.approvals) == 1
    assert record.trace.approvals[0].decision is decision
    assert RunRecord.from_json(record.to_json()) == record
    replay = replay_run(record)
    approval_replay = [
        entry for entry in replay if entry.kind == "approval_decision"
    ]
    assert len(approval_replay) == 1
    assert dict(approval_replay[0].metadata) == {
        "name": "effect",
        "call_id": "approval-call",
        "decision": decision.value,
    }
    record_path = tmp_path / "approval-run.json"
    record_path.write_text(record.to_json(), encoding="utf-8")
    inspect_output = []
    assert main(
        ["inspect", str(record_path)],
        output_fn=inspect_output.append,
    ) == 0
    assert "Approvals: 1" in inspect_output
    assert any(
        f"tool=effect" in line
        and f"decision={decision.value}" in line
        for line in inspect_output
    )

    result = next(
        item for item in agent.session.items if isinstance(item, ToolResult)
    )
    assert result.is_error is (not executed)
    if not executed:
        assert "denied during approval" in result.content
