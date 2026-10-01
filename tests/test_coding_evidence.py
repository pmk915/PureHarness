import sys

from dataclasses import FrozenInstanceError

import pytest

from pureharness.agent import Agent
from pureharness.coding_evidence import (
    CodingEvidenceSnapshot,
    CodingEvidenceTracker,
)
from pureharness.coding_tools import (
    create_coding_tools,
    create_run_command_tool,
)
from pureharness.completion import EvidenceAwareCodingCompletionPolicy
from pureharness.execution import CommandResult, ExecutionTimeoutError
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.processes import LocalProcessManager
from pureharness.run_record import RUN_RECORD_SCHEMA_VERSION
from pureharness.runtime import ExecutionBudget, ExecutionBudgetExceeded
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_policy import PolicyDecision
from pureharness.tool_selection import StaticToolSelector
from pureharness.tools import RiskLevel, Tool, ToolRegistry
from pureharness.verification import (
    CommandPurpose,
    CommandToolObservation,
    VerificationOutcome,
)
from pureharness.workspace_discipline import (
    WorkspaceDiscipline,
    WorkspaceMutation,
)


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, messages, tools):
        del messages, tools
        return next(self.outputs)


class StaticPolicy:
    def __init__(self, decision):
        self.decision = decision

    def evaluate(self, tool, arguments):
        del tool, arguments
        return self.decision


class FakeExecutionBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def execute(self, argv, *, cwd, timeout):
        self.calls.append(
            {"argv": argv, "cwd": cwd, "timeout": timeout}
        )
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _tool(name: str, category: str) -> Tool:
    return Tool(
        name=name,
        description=f"Test {name}.",
        parameters={"type": "object", "properties": {}},
        function=lambda: None,
        category=category,
        risk_level=RiskLevel.READ,
    )


def _mutation(path: str = "a.py") -> WorkspaceMutation:
    return WorkspaceMutation(
        path=path,
        tool_name="apply_patch",
        operation="patched",
    )


def _call(
    name: str,
    arguments: dict[str, object],
    call_id: str,
) -> ToolCall:
    return ToolCall(name=name, arguments=arguments, call_id=call_id)


def _agent(
    workspace,
    outputs,
    *,
    policy=None,
    tool_selector=None,
    execution_budget=None,
    process_manager=None,
    execution_backend=None,
    completion_policy=None,
):
    registry = ToolRegistry()
    for tool in create_coding_tools(
        workspace,
        execution_backend=execution_backend,
        process_manager=process_manager,
    ):
        registry.register(tool)
    executor = ToolExecutor(
        registry,
        policy=policy,
        precondition=WorkspaceDiscipline(workspace),
    )
    return Agent(
        model=ScriptedModel(outputs),
        tools=registry,
        tool_executor=executor,
        tool_selector=tool_selector,
        execution_budget=execution_budget,
        completion_policy=completion_policy,
        max_steps=max(1, len(outputs)),
        run_id_factory=lambda: "coding-evidence-run",
    )


def test_empty_coding_evidence_snapshot_is_immutable():
    snapshot = CodingEvidenceTracker().snapshot

    assert snapshot == CodingEvidenceSnapshot()
    assert snapshot.last_mutation_step is None
    assert snapshot.last_execution_step is None
    with pytest.raises(FrozenInstanceError):
        snapshot.workspace_mutations = 1


def test_generic_agent_exposes_zero_coding_evidence_snapshot():
    agent = Agent(
        model=ScriptedModel([Message(role="assistant", content="done")])
    )

    assert agent.coding_evidence_snapshot == CodingEvidenceSnapshot()
    assert agent.run("hello") == "done"
    assert agent.coding_evidence_snapshot == CodingEvidenceSnapshot()


def test_tracker_records_successful_mutation():
    tracker = CodingEvidenceTracker()

    tracker.record_workspace_mutation(_mutation(), step=3)

    assert tracker.snapshot.workspace_mutations == 1
    assert tracker.snapshot.last_mutation_step == 3
    assert tracker.snapshot.executions_since_last_mutation == 0


def test_execution_before_mutation_does_not_become_post_mutation_evidence():
    tracker = CodingEvidenceTracker()

    tracker.record_tool_started(_tool("run_command", "execution"), step=1)
    tracker.record_workspace_mutation(_mutation(), step=2)

    assert tracker.snapshot.command_executions == 1
    assert tracker.snapshot.executions_since_last_mutation == 0
    assert tracker.snapshot.last_execution_step == 1


def test_same_step_execution_after_mutation_is_ordered_and_counted():
    tracker = CodingEvidenceTracker()

    tracker.record_workspace_mutation(_mutation(), step=4)
    tracker.record_tool_started(_tool("run_command", "execution"), step=4)

    assert tracker.snapshot.executions_since_last_mutation == 1
    assert tracker.snapshot.last_mutation_step == 4
    assert tracker.snapshot.last_execution_step == 4


def test_new_mutation_resets_post_mutation_execution_count():
    tracker = CodingEvidenceTracker()
    command = _tool("run_command", "execution")

    tracker.record_workspace_mutation(_mutation("a.py"), step=1)
    tracker.record_tool_started(command, step=2)
    tracker.record_workspace_mutation(_mutation("b.py"), step=3)

    assert tracker.snapshot.workspace_mutations == 2
    assert tracker.snapshot.command_executions == 1
    assert tracker.snapshot.executions_since_last_mutation == 0
    assert tracker.snapshot.last_mutation_step == 3


def test_tracker_counts_started_command_and_process_errors_only():
    tracker = CodingEvidenceTracker()
    command = _tool("run_command", "execution")
    start = _tool("start_process", "process")
    poll = _tool("poll_process", "process")
    stop = _tool("stop_process", "process")

    for tool in (command, start, poll, stop):
        tracker.record_tool_started(tool, step=2)
    tracker.record_tool_result(command, is_error=True)
    tracker.record_tool_result(start, is_error=True)
    tracker.record_tool_result(poll, is_error=False)
    tracker.record_tool_result(stop, is_error=True)

    snapshot = tracker.snapshot
    assert snapshot.command_executions == 1
    assert snapshot.command_tool_errors == 1
    assert snapshot.process_starts == 1
    assert snapshot.process_polls == 1
    assert snapshot.process_stops == 1
    assert snapshot.process_tool_errors == 2
    assert snapshot.executions_since_last_mutation == 0
    assert snapshot.last_execution_step == 2


def test_tracker_requires_canonical_tool_name_and_category():
    tracker = CodingEvidenceTracker()

    tracker.record_tool_started(_tool("run_command", "general"), step=0)
    tracker.record_tool_started(_tool("other", "execution"), step=0)
    tracker.record_tool_started(_tool("other", "process"), step=0)

    assert tracker.snapshot == CodingEvidenceSnapshot()


def test_general_command_exit_does_not_create_verification_evidence(
    tmp_path,
):
    tool = create_run_command_tool(tmp_path)
    tracker = CodingEvidenceTracker()
    arguments = {"argv": ["check"], "purpose": "general"}
    result = CommandToolObservation(
        purpose=CommandPurpose.GENERAL,
        exit_code=1,
        content="result",
    )

    tracker.record_tool_started(tool, arguments, step=1)
    tracker.record_tool_result(
        tool,
        arguments,
        result=result,
        is_error=False,
        step=1,
    )

    assert tracker.snapshot.command_executions == 1
    assert tracker.snapshot.verification_attempts == 0
    assert tracker.snapshot.last_verification_outcome is None


@pytest.mark.parametrize(
    (
        "exit_code",
        "expected_outcome",
        "expected_zero",
        "expected_nonzero",
    ),
    [
        (0, VerificationOutcome.EXIT_ZERO, 1, 0),
        (7, VerificationOutcome.EXIT_NONZERO, 0, 1),
        (-9, VerificationOutcome.EXIT_NONZERO, 0, 1),
    ],
)
def test_verification_command_records_structured_process_outcome(
    tmp_path,
    exit_code,
    expected_outcome,
    expected_zero,
    expected_nonzero,
):
    tool = create_run_command_tool(tmp_path)
    tracker = CodingEvidenceTracker()
    arguments = {"argv": ["check"], "purpose": "verification"}
    result = CommandToolObservation(
        purpose=CommandPurpose.VERIFICATION,
        exit_code=exit_code,
        content="not parsed",
    )

    tracker.record_tool_started(tool, arguments, step=3)
    pending = tracker.snapshot
    assert pending.last_verification_step == 3
    assert pending.last_verification_outcome is None
    assert pending.last_verification_exit_code is None
    tracker.record_tool_result(
        tool,
        arguments,
        result=result,
        is_error=False,
        step=3,
    )

    snapshot = tracker.snapshot
    assert snapshot.command_executions == 1
    assert snapshot.verification_attempts == 1
    assert snapshot.verification_exit_zero == expected_zero
    assert snapshot.verification_exit_nonzero == expected_nonzero
    assert snapshot.verification_tool_errors == 0
    assert snapshot.last_verification_outcome is expected_outcome
    assert snapshot.last_verification_exit_code == exit_code
    assert snapshot.last_verification_step == 3


def test_started_verification_tool_error_records_no_exit_code(tmp_path):
    tool = create_run_command_tool(tmp_path)
    tracker = CodingEvidenceTracker()
    arguments = {"argv": ["check"], "purpose": "verification"}

    tracker.record_tool_started(tool, arguments, step=2)
    tracker.record_tool_result(
        tool,
        arguments,
        is_error=True,
        step=2,
    )

    snapshot = tracker.snapshot
    assert snapshot.command_executions == 1
    assert snapshot.command_tool_errors == 1
    assert snapshot.verification_attempts == 1
    assert snapshot.verification_tool_errors == 1
    assert snapshot.last_verification_outcome is VerificationOutcome.TOOL_ERROR
    assert snapshot.last_verification_exit_code is None
    assert snapshot.last_verification_step == 2


def test_verification_ordering_is_actual_order_and_resets_on_mutation(
    tmp_path,
):
    tool = create_run_command_tool(tmp_path)
    tracker = CodingEvidenceTracker()
    arguments = {"argv": ["check"], "purpose": "verification"}

    tracker.record_workspace_mutation(_mutation("a.py"), step=4)
    tracker.record_tool_started(tool, arguments, step=4)
    tracker.record_tool_result(
        tool,
        arguments,
        result=CommandToolObservation(
            CommandPurpose.VERIFICATION,
            0,
            "result",
        ),
        is_error=False,
        step=4,
    )
    after_verification = tracker.snapshot
    assert after_verification.verifications_since_last_mutation == 1
    assert after_verification.executions_since_last_mutation == 1

    tracker.record_workspace_mutation(_mutation("b.py"), step=4)
    final = tracker.snapshot
    assert final.verifications_since_last_mutation == 0
    assert final.verification_attempts == 1
    assert final.verification_exit_zero == 1


def test_verification_before_first_mutation_is_not_post_mutation(
    tmp_path,
):
    tool = create_run_command_tool(tmp_path)
    tracker = CodingEvidenceTracker()
    arguments = {"argv": ["check"], "purpose": "verification"}

    tracker.record_tool_started(tool, arguments, step=0)
    tracker.record_tool_result(
        tool,
        arguments,
        result=CommandToolObservation(
            CommandPurpose.VERIFICATION,
            0,
            "result",
        ),
        is_error=False,
        step=0,
    )
    tracker.record_workspace_mutation(_mutation(), step=1)

    assert tracker.snapshot.verification_attempts == 1
    assert tracker.snapshot.verifications_since_last_mutation == 0


def test_latest_verification_overwrites_latest_state_but_not_totals(
    tmp_path,
):
    tool = create_run_command_tool(tmp_path)
    tracker = CodingEvidenceTracker()
    arguments = {"argv": ["check"], "purpose": "verification"}

    for step, exit_code in enumerate((1, 0)):
        tracker.record_tool_started(tool, arguments, step=step)
        tracker.record_tool_result(
            tool,
            arguments,
            result=CommandToolObservation(
                CommandPurpose.VERIFICATION,
                exit_code,
                "result",
            ),
            is_error=False,
            step=step,
        )

    snapshot = tracker.snapshot
    assert snapshot.verification_attempts == 2
    assert snapshot.verification_exit_zero == 1
    assert snapshot.verification_exit_nonzero == 1
    assert snapshot.last_verification_outcome is VerificationOutcome.EXIT_ZERO
    assert snapshot.last_verification_exit_code == 0
    assert snapshot.last_verification_step == 1


def test_custom_run_command_result_does_not_invent_verification_outcome():
    tracker = CodingEvidenceTracker()
    custom = Tool(
        name="run_command",
        description="Synthetic command-shaped tool.",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "purpose": {
                    "type": "string",
                    "enum": ["general", "verification"],
                    "default": "general",
                },
            },
        },
        function=lambda purpose="general": {"exit_code": 0},
        category="execution",
        risk_level=RiskLevel.EXECUTE,
    )
    arguments = {"purpose": "verification"}

    tracker.record_tool_started(custom, arguments, step=0)
    tracker.record_tool_result(
        custom,
        arguments,
        result={"exit_code": 0},
        is_error=False,
        step=0,
    )

    assert tracker.snapshot.command_executions == 1
    assert tracker.snapshot.verification_attempts == 0
    assert tracker.snapshot.last_verification_exit_code is None


def test_inspection_only_agent_run_has_zero_coding_evidence(tmp_path):
    (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
    agent = _agent(
        tmp_path,
        [
            [
                _call("read_file", {"path": "a.py"}, "read"),
                _call(
                    "search_text",
                    {"query": "needle", "path": "."},
                    "search",
                ),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    assert agent.run("inspect") == "done"
    assert agent.coding_evidence_snapshot == CodingEvidenceSnapshot()
    assert agent.trace.end_reason == "completed"


def test_mutation_only_run_exposes_evidence_without_blocking_completion(
    tmp_path,
):
    (tmp_path / "a.py").write_text("before\n", encoding="utf-8")
    agent = _agent(
        tmp_path,
        [
            [
                _call("read_file", {"path": "a.py"}, "read"),
                _call(
                    "apply_patch",
                    {
                        "path": "a.py",
                        "old_text": "before",
                        "new_text": "after",
                    },
                    "patch",
                ),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    assert agent.run("edit") == "done"
    snapshot = agent.coding_evidence_snapshot
    assert snapshot.workspace_mutations == 1
    assert snapshot.executions_since_last_mutation == 0
    assert snapshot.last_mutation_step == 0
    assert snapshot.last_execution_step is None
    assert agent.trace.end_reason == "completed"


def test_same_batch_mutation_then_command_records_post_mutation_execution(
    tmp_path,
):
    (tmp_path / "a.py").write_text("before\n", encoding="utf-8")
    agent = _agent(
        tmp_path,
        [
            [
                _call("read_file", {"path": "a.py"}, "read"),
                _call(
                    "apply_patch",
                    {
                        "path": "a.py",
                        "old_text": "before",
                        "new_text": "after",
                    },
                    "patch",
                ),
                _call(
                    "run_command",
                    {"argv": [sys.executable, "-c", "print('checked')"]},
                    "run",
                ),
            ],
            Message(role="assistant", content="done"),
        ],
    )

    assert agent.run("edit and execute") == "done"
    snapshot = agent.coding_evidence_snapshot
    assert snapshot.workspace_mutations == 1
    assert snapshot.command_executions == 1
    assert snapshot.command_tool_errors == 0
    assert snapshot.executions_since_last_mutation == 1
    assert snapshot.last_mutation_step == 0
    assert snapshot.last_execution_step == 0
    evidence_events = [
        event
        for event in agent.events
        if event.type == "coding_evidence_snapshot"
    ]
    assert [event.data["terminal"] for event in evidence_events] == [
        False,
        True,
    ]
    assert evidence_events[0].data["workspace_mutations"] == 1
    assert evidence_events[0].data["command_executions"] == 1
    assert [event.type for event in agent.events[-3:]] == [
        "progress_snapshot",
        "coding_evidence_snapshot",
        "agent_completed",
    ]


def test_nonzero_command_exit_is_not_a_tool_error(tmp_path):
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "run_command",
                    {"argv": [sys.executable, "-c", "raise SystemExit(3)"]},
                    "run",
                )
            ],
            Message(role="assistant", content="done"),
        ],
    )

    assert agent.run("execute") == "done"
    assert agent.coding_evidence_snapshot.command_executions == 1
    assert agent.coding_evidence_snapshot.command_tool_errors == 0
    result = next(
        item for item in agent.messages if isinstance(item, ToolResult)
    )
    assert result.is_error is False
    assert "exit_code: 3" in result.content


def test_same_batch_mutation_then_verification_records_ordered_evidence(
    tmp_path,
):
    (tmp_path / "a.py").write_text("before\n", encoding="utf-8")
    backend = FakeExecutionBackend(
        [CommandResult(exit_code=0, stdout="ok", stderr="")]
    )
    agent = _agent(
        tmp_path,
        [
            [
                _call("read_file", {"path": "a.py"}, "read"),
                _call(
                    "apply_patch",
                    {
                        "path": "a.py",
                        "old_text": "before",
                        "new_text": "after",
                    },
                    "patch",
                ),
                _call(
                    "run_command",
                    {"argv": ["check"], "purpose": "verification"},
                    "verify",
                ),
            ],
            Message(role="assistant", content="done"),
        ],
        execution_backend=backend,
    )

    assert agent.run("edit and verify") == "done"
    snapshot = agent.coding_evidence_snapshot
    assert snapshot.workspace_mutations == 1
    assert snapshot.executions_since_last_mutation == 1
    assert snapshot.verifications_since_last_mutation == 1
    assert snapshot.verification_attempts == 1
    assert snapshot.verification_exit_zero == 1
    assert snapshot.last_verification_outcome is VerificationOutcome.EXIT_ZERO


def test_verification_tool_exception_is_observed_after_start(tmp_path):
    backend = FakeExecutionBackend(
        [ExecutionTimeoutError("controlled timeout")]
    )
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "run_command",
                    {"argv": ["slow"], "purpose": "verification"},
                    "verify",
                )
            ],
            Message(role="assistant", content="done"),
        ],
        execution_backend=backend,
    )

    assert agent.run("verify") == "done"
    snapshot = agent.coding_evidence_snapshot
    assert snapshot.verification_attempts == 1
    assert snapshot.verification_tool_errors == 1
    assert snapshot.command_tool_errors == 1
    assert snapshot.last_verification_outcome is VerificationOutcome.TOOL_ERROR
    assert snapshot.last_verification_exit_code is None


def test_started_command_exception_counts_as_command_tool_error(tmp_path):
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "run_command",
                    {"argv": ["definitely-missing-pureharness-command"]},
                    "run",
                )
            ],
            Message(role="assistant", content="recovered"),
        ],
    )

    assert agent.run("execute") == "recovered"
    assert agent.coding_evidence_snapshot.command_executions == 1
    assert agent.coding_evidence_snapshot.command_tool_errors == 1


def test_process_lifecycle_has_separate_factual_counters(tmp_path):
    manager = LocalProcessManager(
        tmp_path,
        job_id_factory=lambda: "job-1",
    )
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "start_process",
                    {
                        "argv": [
                            sys.executable,
                            "-c",
                            "import time; time.sleep(30)",
                        ]
                    },
                    "start",
                ),
                _call("poll_process", {"job_id": "job-1"}, "poll"),
                _call("stop_process", {"job_id": "job-1"}, "stop"),
            ],
            Message(role="assistant", content="done"),
        ],
        process_manager=manager,
    )

    assert agent.run("manage process") == "done"
    snapshot = agent.coding_evidence_snapshot
    assert snapshot.process_starts == 1
    assert snapshot.process_polls == 1
    assert snapshot.process_stops == 1
    assert snapshot.process_tool_errors == 0
    assert snapshot.last_execution_step == 0
    assert snapshot.executions_since_last_mutation == 0
    assert manager.active_job_count == 0


def test_blocked_mutation_does_not_count(tmp_path):
    (tmp_path / "a.py").write_text("before\n", encoding="utf-8")
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "apply_patch",
                    {
                        "path": "a.py",
                        "old_text": "before",
                        "new_text": "after",
                    },
                    "blocked",
                )
            ],
            Message(role="assistant", content="done"),
        ],
    )

    assert agent.run("edit") == "done"
    assert agent.coding_evidence_snapshot.workspace_mutations == 0


@pytest.mark.parametrize(
    "decision",
    [PolicyDecision.DENY, PolicyDecision.REQUIRE_APPROVAL],
)
def test_policy_or_approval_rejected_mutation_does_not_count(
    tmp_path,
    decision,
):
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "write_file",
                    {"path": "new.py", "content": "not written\n"},
                    "write",
                )
            ],
            Message(role="assistant", content="done"),
        ],
        policy=StaticPolicy(decision),
    )

    assert agent.run("write") == "done"
    assert agent.coding_evidence_snapshot.workspace_mutations == 0
    assert not (tmp_path / "new.py").exists()


def test_started_process_failure_counts_operation_and_error(tmp_path):
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "poll_process",
                    {"job_id": "missing"},
                    "poll",
                )
            ],
            Message(role="assistant", content="recovered"),
        ],
    )

    assert agent.run("poll") == "recovered"
    snapshot = agent.coding_evidence_snapshot
    assert snapshot.process_polls == 1
    assert snapshot.process_tool_errors == 1
    assert snapshot.last_execution_step is None


@pytest.mark.parametrize(
    "rejection",
    ["unexposed", "policy", "approval", "arguments"],
)
def test_pre_execution_rejection_does_not_count_command(tmp_path, rejection):
    arguments: dict[str, object] = {
        "argv": [sys.executable, "-c", "print('not reached')"],
        "purpose": "verification",
    }
    policy = None
    selector = None
    if rejection == "unexposed":
        selector = StaticToolSelector(["read_file"])
    elif rejection == "policy":
        policy = StaticPolicy(PolicyDecision.DENY)
    elif rejection == "approval":
        policy = StaticPolicy(PolicyDecision.REQUIRE_APPROVAL)
    else:
        arguments["unknown"] = True
    agent = _agent(
        tmp_path,
        [
            [_call("run_command", arguments, "run")],
            Message(role="assistant", content="done"),
        ],
        policy=policy,
        tool_selector=selector,
    )

    assert agent.run("execute") == "done"
    assert agent.coding_evidence_snapshot.command_executions == 0
    assert agent.coding_evidence_snapshot.command_tool_errors == 0
    assert agent.coding_evidence_snapshot.verification_attempts == 0


def test_budget_rejected_command_batch_does_not_count(tmp_path):
    calls = [
        _call(
            "run_command",
            {
                "argv": [sys.executable, "-c", "print('not reached')"],
                "purpose": "verification",
            },
            str(index),
        )
        for index in range(2)
    ]
    agent = _agent(
        tmp_path,
        [calls],
        execution_budget=ExecutionBudget(max_tool_calls=1),
    )

    with pytest.raises(ExecutionBudgetExceeded):
        agent.run("execute")

    assert agent.coding_evidence_snapshot.command_executions == 0
    assert agent.coding_evidence_snapshot.command_tool_errors == 0
    assert agent.coding_evidence_snapshot.verification_attempts == 0


def test_coding_evidence_resets_between_runs_with_persistent_session(tmp_path):
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "write_file",
                    {"path": "new.py", "content": "value = 1\n"},
                    "write",
                ),
                _call(
                    "run_command",
                    {
                        "argv": [sys.executable, "-c", "print('checked')"],
                        "purpose": "verification",
                    },
                    "run",
                ),
            ],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ],
    )

    assert agent.run("first run") == "first"
    assert agent.coding_evidence_snapshot.workspace_mutations == 1
    assert agent.coding_evidence_snapshot.command_executions == 1
    assert agent.coding_evidence_snapshot.verification_attempts == 1

    assert agent.run("second run") == "second"
    assert agent.coding_evidence_snapshot == CodingEvidenceSnapshot()
    assert len(agent.session.items) > 2


def test_nonzero_verification_after_mutation_requests_recheck(
    tmp_path,
):
    (tmp_path / "a.py").write_text("before\n", encoding="utf-8")
    backend = FakeExecutionBackend(
        [
            CommandResult(exit_code=1, stdout="failed check", stderr=""),
            CommandResult(exit_code=0, stdout="recovered", stderr=""),
        ]
    )
    agent = _agent(
        tmp_path,
        [
            [
                _call("read_file", {"path": "a.py"}, "read"),
                _call(
                    "apply_patch",
                    {
                        "path": "a.py",
                        "old_text": "before",
                        "new_text": "after",
                    },
                    "patch",
                ),
                _call(
                    "run_command",
                    {"argv": ["check"], "purpose": "verification"},
                    "verify",
                ),
            ],
            Message(role="assistant", content="premature"),
            [
                _call(
                    "run_command",
                    {"argv": ["check"], "purpose": "verification"},
                    "verify-again",
                )
            ],
            Message(role="assistant", content="done"),
        ],
        execution_backend=backend,
        completion_policy=EvidenceAwareCodingCompletionPolicy(),
    )

    assert agent.run("edit and verify") == "done"
    snapshot = agent.coding_evidence_snapshot
    assert snapshot.verification_exit_nonzero == 1
    assert snapshot.verification_exit_zero == 1
    assert snapshot.last_verification_outcome is VerificationOutcome.EXIT_ZERO
    assert snapshot.last_verification_exit_code == 0
    assert snapshot.executions_since_last_mutation == 2
    assert snapshot.verifications_since_last_mutation == 2
    assert agent.completion_rechecks_used == 1
    assert agent.trace.end_reason == "completed"
    requested = next(
        event
        for event in agent.events
        if event.type == "completion_recheck_requested"
    )
    assert requested.data == {
        "step": 1,
        "reason": "verification_failed_after_mutation",
        "recheck_number": 1,
        "max_rechecks": 1,
        "workspace_mutations": 1,
        "command_executions": 1,
        "process_starts": 0,
        "executions_since_last_mutation": 1,
        "verification_outcome": "exit_nonzero",
        "verification_exit_code": 1,
    }


def test_unresolved_failed_verification_respects_recheck_limit(tmp_path):
    (tmp_path / "a.py").write_text("before\n", encoding="utf-8")
    backend = FakeExecutionBackend(
        [CommandResult(exit_code=1, stdout="failed check", stderr="")]
    )
    agent = _agent(
        tmp_path,
        [
            [
                _call("read_file", {"path": "a.py"}, "read"),
                _call(
                    "apply_patch",
                    {
                        "path": "a.py",
                        "old_text": "before",
                        "new_text": "after",
                    },
                    "patch",
                ),
                _call(
                    "run_command",
                    {"argv": ["check"], "purpose": "verification"},
                    "verify",
                ),
            ],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ],
        execution_backend=backend,
        completion_policy=EvidenceAwareCodingCompletionPolicy(),
    )

    assert agent.run("edit and verify") == "second"
    assert agent.completion_rechecks_used == 1
    requested = [
        event
        for event in agent.events
        if event.type == "completion_recheck_requested"
    ]
    skipped = next(
        event
        for event in agent.events
        if event.type == "completion_recheck_skipped"
    )
    assert len(requested) == 1
    assert requested[0].data["reason"] == (
        "verification_failed_after_mutation"
    )
    assert skipped.data["reason"] == "verification_failed_after_mutation"
    assert skipped.data["skip_reason"] == "recheck_limit"
    assert skipped.data["verification_outcome"] == "exit_nonzero"
    assert skipped.data["verification_exit_code"] == 1
    assert agent.trace.end_reason == "completed"


def test_coding_evidence_is_not_persisted_in_run_record(tmp_path):
    backend = FakeExecutionBackend(
        [CommandResult(exit_code=0, stdout="ok", stderr="")]
    )
    agent = _agent(
        tmp_path,
        [
            [
                _call(
                    "write_file",
                    {"path": "new.py", "content": "x"},
                    "w",
                ),
                _call(
                    "run_command",
                    {"argv": ["check"], "purpose": "verification"},
                    "v",
                ),
            ]
        ],
        execution_backend=backend,
    )

    with pytest.raises(RuntimeError, match="exceeded max steps"):
        agent.run("write")

    assert agent.coding_evidence_snapshot.verification_attempts == 1
    assert agent.last_run_record is not None
    persisted = agent.last_run_record.to_dict()
    assert RUN_RECORD_SCHEMA_VERSION == 2
    assert persisted["schema_version"] == 2
    assert "coding_evidence" not in persisted
    assert "coding_evidence_snapshot" not in persisted
