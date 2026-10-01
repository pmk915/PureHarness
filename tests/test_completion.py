from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from pureharness.agent import Agent
from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.completion import (
    CompletionAssessment,
    CompletionDecision,
    CompletionReason,
    EvidenceAwareCodingCompletionPolicy,
    render_completion_recheck,
)
from pureharness.context import (
    ContextBudget,
    ContextBuilder,
    ContextLimits,
    RecentContextBuilder,
    TokenBudgetContextBuilder,
)
from pureharness.messages import AgentItem, Message, ToolCall
from pureharness.model import (
    ContextWindowExceededError,
    MalformedModelOutputError,
)
from pureharness.runtime import ExecutionBudget
from pureharness.session import Session
from pureharness.session_store import (
    DurableSession,
    JsonlDurableSessionStore,
)
from pureharness.skills import default_coding_skills, render_skill
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_selection import estimate_tool_schema_tokens
from pureharness.tools import RiskLevel, Tool, ToolRegistry
from pureharness.verification import VerificationOutcome
from pureharness.workspace_discipline import WorkspaceMutation


class ScriptedModel:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.contexts: list[list[AgentItem]] = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class MutationEvidencePrecondition:
    def reset(self) -> None:
        pass

    def prepare(self, tool, arguments):
        return None

    def revalidate(self, tool, arguments, prepared):
        return prepared

    def record_success(self, tool, arguments, prepared, result):
        if tool.name == "apply_patch":
            return WorkspaceMutation(
                path="a.py",
                tool_name="apply_patch",
                operation="patched",
            )
        return None


class CompletionEstimator:
    def __init__(
        self,
        *,
        message_tokens: int = 2,
        task_state_tokens: int = 1,
        recheck_tokens: int = 5,
        tool_tokens: int = 4,
    ) -> None:
        self.message_tokens = message_tokens
        self.task_state_tokens = task_state_tokens
        self.recheck_tokens = recheck_tokens
        self.tool_tokens = tool_tokens

    def estimate(self, items) -> int:
        if not items:
            return 0
        first = items[0]
        if isinstance(first, Message):
            if first.content.startswith("[PureHarness Completion Recheck]"):
                return self.recheck_tokens
            if first.role == "system":
                return self.task_state_tokens
            return self.message_tokens
        return self.tool_tokens


def _tool(name: str, category: str, result=None) -> Tool:
    return Tool(
        name=name,
        description=f"Test {name}.",
        parameters={"type": "object", "properties": {}},
        function=lambda: result,
        category=category,
        risk_level=RiskLevel.READ,
    )


def _agent(
    outputs,
    *,
    max_steps=None,
    execution_budget=None,
    context_builder=None,
    context_limits=None,
    session=None,
    session_id=None,
    skills=(),
) -> tuple[Agent, ScriptedModel]:
    registry = ToolRegistry()
    registry.register(_tool("run_command", "execution", {"exit_code": 0}))
    registry.register(_tool("start_process", "process", {"pid": 1}))
    registry.register(_tool("apply_patch", "workspace", "patched"))
    model = ScriptedModel(outputs)
    agent = Agent(
        model=model,
        tools=registry,
        tool_executor=ToolExecutor(
            registry,
            precondition=MutationEvidencePrecondition(),
        ),
        completion_policy=EvidenceAwareCodingCompletionPolicy(),
        max_steps=max_steps if max_steps is not None else len(outputs),
        execution_budget=execution_budget,
        context_builder=context_builder,
        context_limits=context_limits,
        session=session,
        session_id=session_id,
        skills=skills,
        run_id_factory=lambda: "completion-run",
    )
    return agent, model


def _call(name: str, call_id: str) -> ToolCall:
    return ToolCall(name=name, arguments={}, call_id=call_id)


def _event(agent: Agent, event_type: str):
    return next(event for event in agent.events if event.type == event_type)


def _recheck_messages(context) -> list[Message]:
    return [
        item
        for item in context
        if isinstance(item, Message)
        and item.content.startswith("[PureHarness Completion Recheck]")
    ]


@pytest.mark.parametrize(
    "snapshot, decision, reason",
    [
        (
            CodingEvidenceSnapshot(),
            CompletionDecision.ACCEPT,
            None,
        ),
        (
            CodingEvidenceSnapshot(command_executions=2),
            CompletionDecision.RECONSIDER,
            CompletionReason.EXECUTION_WITHOUT_STRUCTURED_MUTATION,
        ),
        (
            CodingEvidenceSnapshot(process_starts=1),
            CompletionDecision.RECONSIDER,
            CompletionReason.EXECUTION_WITHOUT_STRUCTURED_MUTATION,
        ),
        (
            CodingEvidenceSnapshot(workspace_mutations=1),
            CompletionDecision.RECONSIDER,
            CompletionReason.MUTATION_WITHOUT_POST_MUTATION_EXECUTION,
        ),
        (
            CodingEvidenceSnapshot(
                workspace_mutations=1,
                command_executions=1,
                executions_since_last_mutation=1,
            ),
            CompletionDecision.ACCEPT,
            None,
        ),
    ],
)
def test_coding_completion_policy_rules(snapshot, decision, reason):
    assessment = EvidenceAwareCodingCompletionPolicy().assess(snapshot)

    assert assessment == CompletionAssessment(decision, reason)


def test_command_error_counters_do_not_change_policy_decision():
    policy = EvidenceAwareCodingCompletionPolicy()

    assert policy.assess(
        CodingEvidenceSnapshot(
            workspace_mutations=1,
            command_executions=1,
            command_tool_errors=1,
            executions_since_last_mutation=1,
        )
    ).decision is CompletionDecision.ACCEPT


@pytest.mark.parametrize(
    "outcome, exit_code",
    [
        (VerificationOutcome.EXIT_NONZERO, 1),
        (VerificationOutcome.TOOL_ERROR, None),
    ],
)
def test_failed_verification_after_mutation_has_highest_priority(
    outcome,
    exit_code,
):
    assessment = EvidenceAwareCodingCompletionPolicy().assess(
        CodingEvidenceSnapshot(
            workspace_mutations=1,
            command_executions=1,
            command_tool_errors=(
                1 if outcome is VerificationOutcome.TOOL_ERROR else 0
            ),
            executions_since_last_mutation=1,
            verification_attempts=1,
            verification_exit_nonzero=(
                1 if outcome is VerificationOutcome.EXIT_NONZERO else 0
            ),
            verification_tool_errors=(
                1 if outcome is VerificationOutcome.TOOL_ERROR else 0
            ),
            verifications_since_last_mutation=1,
            last_mutation_step=1,
            last_execution_step=2,
            last_verification_outcome=outcome,
            last_verification_exit_code=exit_code,
            last_verification_step=2,
        )
    )

    assert assessment == CompletionAssessment(
        CompletionDecision.RECONSIDER,
        CompletionReason.VERIFICATION_FAILED_AFTER_MUTATION,
    )


def test_successful_verification_after_mutation_is_accepted():
    assessment = EvidenceAwareCodingCompletionPolicy().assess(
        CodingEvidenceSnapshot(
            workspace_mutations=1,
            command_executions=1,
            executions_since_last_mutation=1,
            verification_attempts=1,
            verification_exit_zero=1,
            verifications_since_last_mutation=1,
            last_mutation_step=1,
            last_execution_step=2,
            last_verification_outcome=VerificationOutcome.EXIT_ZERO,
            last_verification_exit_code=0,
            last_verification_step=2,
        )
    )

    assert assessment == CompletionAssessment(CompletionDecision.ACCEPT)


def test_failed_verification_before_mutation_is_ignored():
    assessment = EvidenceAwareCodingCompletionPolicy().assess(
        CodingEvidenceSnapshot(
            workspace_mutations=1,
            command_executions=1,
            verification_attempts=1,
            verification_exit_nonzero=1,
            last_mutation_step=2,
            last_execution_step=1,
            last_verification_outcome=VerificationOutcome.EXIT_NONZERO,
            last_verification_exit_code=1,
            last_verification_step=1,
        )
    )

    assert assessment == CompletionAssessment(CompletionDecision.ACCEPT)


def test_general_command_nonzero_after_mutation_is_accepted():
    assessment = EvidenceAwareCodingCompletionPolicy().assess(
        CodingEvidenceSnapshot(
            workspace_mutations=1,
            command_executions=1,
            executions_since_last_mutation=1,
            last_mutation_step=1,
            last_execution_step=2,
        )
    )

    assert assessment == CompletionAssessment(CompletionDecision.ACCEPT)


def test_mutation_after_failed_verification_is_a_reaction():
    assessment = EvidenceAwareCodingCompletionPolicy().assess(
        CodingEvidenceSnapshot(
            workspace_mutations=2,
            command_executions=1,
            verification_attempts=1,
            verification_exit_nonzero=1,
            last_mutation_step=2,
            last_execution_step=2,
            last_verification_outcome=VerificationOutcome.EXIT_NONZERO,
            last_verification_exit_code=1,
            last_verification_step=2,
        )
    )

    assert assessment == CompletionAssessment(CompletionDecision.ACCEPT)


def test_completion_values_are_immutable_and_rendering_is_neutral():
    policy = EvidenceAwareCodingCompletionPolicy()
    assessment = CompletionAssessment(CompletionDecision.ACCEPT)

    with pytest.raises(FrozenInstanceError):
        assessment.reason = (  # type: ignore[misc]
            CompletionReason.EXECUTION_WITHOUT_STRUCTURED_MUTATION
        )
    assert policy.max_rechecks == 1
    message = render_completion_recheck(
        CompletionReason.EXECUTION_WITHOUT_STRUCTURED_MUTATION
    )
    assert message.role == "system"
    assert "structured workspace mutation" in message.content
    assert "Re-evaluate" in message.content
    assert "must edit" not in message.content.lower()
    assert "pytest" not in message.content.lower()
    mutation_message = render_completion_recheck(
        CompletionReason.MUTATION_WITHOUT_POST_MUTATION_EXECUTION
    )
    assert "successful structured workspace mutation" in (
        mutation_message.content
    )
    assert "no execution was observed after" in mutation_message.content
    verification_message = render_completion_recheck(
        CompletionReason.VERIFICATION_FAILED_AFTER_MUTATION
    )
    assert verification_message == Message(
        role="system",
        content=(
            "[PureHarness Completion Recheck]\n\n"
            "A verification-marked command executed after your latest "
            "workspace mutation did not complete successfully.\n\n"
            "Review the current workspace state and verification evidence "
            "before deciding whether the task is complete."
        ),
    )
    assert "wrong" not in verification_message.content.lower()
    assert "fix the issue" not in verification_message.content.lower()


def test_generic_agent_and_zero_activity_coding_agent_complete_directly():
    generic_model = ScriptedModel([Message(role="assistant", content="done")])
    generic = Agent(model=generic_model)
    coding, _ = _agent([Message(role="assistant", content="done")])

    assert generic.run("question") == "done"
    assert coding.run("question") == "done"

    for agent in (generic, coding):
        assert agent.completion_rechecks_used == 0
        assert not any(
            event.type.startswith("completion_recheck")
            for event in agent.events
        )


def test_execution_without_mutation_rechecks_once_and_keeps_session_clean():
    rejected = Message(role="assistant", content="first final")
    accepted = Message(role="assistant", content="accepted final")
    agent, model = _agent(
        [[_call("run_command", "run-1")], rejected, accepted]
    )

    assert agent.run("task") == "accepted final"

    assert agent.completion_rechecks_used == 1
    assert len(_recheck_messages(model.contexts[2])) == 1
    assert [step.output for step in agent.trace.steps] == [
        [_call("run_command", "run-1")],
        rejected,
        accepted,
    ]
    assert rejected not in agent.session.items
    assert agent.session.items[-1] == accepted
    assert all(
        not (
            isinstance(item, Message)
            and item.content.startswith("[PureHarness Completion Recheck]")
        )
        for item in agent.session.items
    )
    assert sum(
        event.type == "completion_recheck_requested"
        for event in agent.events
    ) == 1
    assert sum(event.type == "agent_completed" for event in agent.events) == 1
    assert _event(
        agent, "completion_recheck_skipped"
    ).data["skip_reason"] == "recheck_limit"
    assert agent.last_run_record is not None
    assert agent.last_run_record.model_call_count == 3
    assert len(agent.last_run_record.trace.steps) == 3
    assert agent.last_run_record.schema_version == 2
    assert "completion_rechecks_used" not in agent.last_run_record.to_dict()
    terminal_progress = [
        event
        for event in agent.events
        if event.type == "progress_snapshot"
        and event.data["terminal"] is True
    ]
    assert len(terminal_progress) == 1


def test_requested_event_precedes_nonterminal_snapshots_and_next_step():
    agent, _ = _agent(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ]
    )

    agent.run("task")

    event_types = [event.type for event in agent.events]
    requested_index = event_types.index("completion_recheck_requested")
    assert event_types[requested_index - 1] == "model_completed"
    assert event_types[requested_index + 1 : requested_index + 3] == [
        "progress_snapshot",
        "coding_evidence_snapshot",
    ]
    next_build = event_types.index("context_build_started", requested_index)
    assert next_build == requested_index + 3
    progress = agent.events[requested_index + 1]
    assert progress.data["logical_steps_completed"] == 2
    assert progress.data["terminal"] is False


def test_recheck_model_context_order_is_skill_state_guidance_history():
    skills = default_coding_skills()
    agent, model = _agent(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ],
        skills=skills,
    )

    agent.run("task")

    context = model.contexts[2]
    assert context[0] == render_skill(skills[0])
    assert isinstance(context[1], Message)
    assert context[1].role == "system"
    assert context[2] == render_completion_recheck(
        CompletionReason.EXECUTION_WITHOUT_STRUCTURED_MUTATION
    )
    assert context[3] == Message(role="user", content="task")


def test_mutation_without_post_mutation_execution_requests_recheck():
    agent, _ = _agent(
        [
            [_call("apply_patch", "patch-1")],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ]
    )

    assert agent.run("task") == "second"

    requested = _event(agent, "completion_recheck_requested")
    assert requested.data["reason"] == (
        "mutation_without_post_mutation_execution"
    )
    assert requested.data["workspace_mutations"] == 1
    assert requested.data["executions_since_last_mutation"] == 0


def test_recheck_guidance_clears_after_valid_tool_decision():
    agent, model = _agent(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            [_call("apply_patch", "patch-1")],
            Message(role="assistant", content="second"),
        ]
    )

    assert agent.run("task") == "second"

    assert len(_recheck_messages(model.contexts[2])) == 1
    assert _recheck_messages(model.contexts[3]) == []


@pytest.mark.parametrize("same_step", [False, True])
def test_mutation_followed_by_execution_accepts_even_with_exit_one(same_step):
    if same_step:
        outputs = [
            [
                _call("apply_patch", "patch-1"),
                _call("run_command", "run-1"),
            ],
            Message(role="assistant", content="done"),
        ]
    else:
        outputs = [
            [_call("apply_patch", "patch-1")],
            [_call("run_command", "run-1")],
            Message(role="assistant", content="done"),
        ]
    agent, _ = _agent(outputs)
    agent.tools.get("run_command").function = lambda: {"exit_code": 1}

    assert agent.run("task") == "done"

    assert agent.coding_evidence_snapshot.executions_since_last_mutation == 1
    assert agent.completion_rechecks_used == 0
    assert not any(
        event.type.startswith("completion_recheck")
        for event in agent.events
    )


def test_final_step_skips_recheck_and_accepts_original_candidate():
    final = Message(role="assistant", content="original")
    agent, _ = _agent(
        [[_call("run_command", "run-1")], final],
        max_steps=2,
    )

    assert agent.run("task") == "original"

    skipped = _event(agent, "completion_recheck_skipped")
    assert skipped.data["skip_reason"] == "step_budget"
    assert agent.trace.end_reason == "completed"
    event_types = [event.type for event in agent.events]
    skipped_index = event_types.index("completion_recheck_skipped")
    assert event_types[skipped_index - 1] == "model_completed"
    assert event_types[skipped_index + 1 : skipped_index + 4] == [
        "progress_snapshot",
        "coding_evidence_snapshot",
        "agent_completed",
    ]


def test_exhausted_model_attempt_budget_skips_recheck():
    final = Message(role="assistant", content="original")
    agent, _ = _agent(
        [[_call("run_command", "run-1")], final],
        max_steps=3,
        execution_budget=ExecutionBudget(max_model_attempts=2),
    )

    assert agent.run("task") == "original"

    skipped = _event(agent, "completion_recheck_skipped")
    assert skipped.data["skip_reason"] == "model_attempt_budget"
    assert agent.trace.end_reason == "completed"
    assert all(
        event.type != "execution_budget_exhausted"
        for event in agent.events
    )


def test_recheck_limit_is_observable_and_accepts_second_final():
    agent, _ = _agent(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ],
        max_steps=4,
    )

    assert agent.run("task") == "second"

    requested = [
        event
        for event in agent.events
        if event.type == "completion_recheck_requested"
    ]
    skipped = _event(agent, "completion_recheck_skipped")
    assert len(requested) == 1
    assert skipped.data["skip_reason"] == "recheck_limit"
    assert skipped.data["rechecks_used"] == 1


def test_recheck_context_tokens_are_separate_and_can_bound_history():
    estimator = CompletionEstimator(recheck_tokens=5)
    builder = ContextBuilder(token_estimator=estimator)
    registry = ToolRegistry()
    registry.register(_tool("run_command", "execution"))
    schema_tokens = estimate_tool_schema_tokens(registry.list_tools())
    usable_tokens = schema_tokens + 10
    session = Session(items=[Message(role="user", content="old")])
    model = ScriptedModel(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ]
    )
    agent = Agent(
        model=model,
        tools=registry,
        completion_policy=EvidenceAwareCodingCompletionPolicy(),
        context_builder=builder,
        context_limits=ContextLimits(usable_tokens + 5, 5),
        session=session,
        max_steps=3,
    )

    agent.run("task")

    built = [event for event in agent.events if event.type == "context_built"]
    normal = built[0]
    recheck = built[2]
    assert normal.data["completion_recheck_present"] is False
    assert normal.data["estimated_completion_recheck_tokens"] == 0
    assert recheck.data["completion_recheck_present"] is True
    assert recheck.data["estimated_completion_recheck_tokens"] == 5
    assert recheck.data["available_history_tokens"] == 4
    assert recheck.data["bounded_history_applied"] is True
    assert len(_recheck_messages(model.contexts[2])) == 1


def test_context_capacity_preflight_skips_optional_recheck():
    estimator = CompletionEstimator(recheck_tokens=5)
    builder = ContextBuilder(token_estimator=estimator)
    registry = ToolRegistry()
    registry.register(_tool("run_command", "execution"))
    schema_tokens = estimate_tool_schema_tokens(registry.list_tools())
    usable_tokens = schema_tokens + 5
    model = ScriptedModel(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="original"),
        ]
    )
    agent = Agent(
        model=model,
        tools=registry,
        completion_policy=EvidenceAwareCodingCompletionPolicy(),
        context_builder=builder,
        context_limits=ContextLimits(usable_tokens + 5, 5),
        max_steps=3,
    )

    assert agent.run("task") == "original"

    assert _event(
        agent, "completion_recheck_skipped"
    ).data["skip_reason"] == "context_capacity"
    assert agent.trace.end_reason == "completed"
    assert all(event.type != "context_build_failed" for event in agent.events)
    assert len(model.contexts) == 2


@pytest.mark.parametrize(
    "builder",
    [
        RecentContextBuilder(max_items=1),
        TokenBudgetContextBuilder(ContextBudget(10_000)),
    ],
)
def test_history_strategies_cannot_drop_completion_recheck(builder):
    agent, model = _agent(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ],
        context_builder=builder,
    )

    agent.run("task")

    assert len(_recheck_messages(model.contexts[2])) == 1


def test_provider_overflow_recovery_preserves_one_recheck_message():
    agent, model = _agent(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            ContextWindowExceededError("too large"),
            Message(role="assistant", content="second"),
        ],
        max_steps=3,
        context_builder=ContextBuilder(
            token_estimator=CompletionEstimator(
                message_tokens=4,
                tool_tokens=4,
            )
        ),
    )

    assert agent.run("task") == "second"

    assert len(_recheck_messages(model.contexts[2])) == 1
    assert len(_recheck_messages(model.contexts[3])) == 1
    assert agent.progress_snapshot.context_recoveries == 1
    assert agent.last_run_record is not None
    assert agent.last_run_record.model_call_count == 3


def test_malformed_output_retry_remains_distinct_from_completion_recheck():
    agent, model = _agent(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            MalformedModelOutputError("bad"),
            Message(role="assistant", content="second"),
        ],
        max_steps=3,
    )

    assert agent.run("task") == "second"

    assert agent.completion_rechecks_used == 1
    assert agent.progress_snapshot.model_retries == 1
    assert agent.execution_usage.model_attempts == 4
    assert len(_recheck_messages(model.contexts[2])) == 1
    assert len(_recheck_messages(model.contexts[3])) == 1


def test_recheck_state_resets_between_runs():
    agent, model = _agent(
        [
            [_call("run_command", "run-1")],
            Message(role="assistant", content="first"),
            Message(role="assistant", content="accepted"),
            Message(role="assistant", content="next run"),
        ],
        max_steps=3,
    )

    assert agent.run("task") == "accepted"
    assert agent.completion_rechecks_used == 1
    assert agent.run("read-only follow-up") == "next run"

    assert agent.completion_rechecks_used == 0
    assert _recheck_messages(model.contexts[-1]) == []
    assert not any(
        event.type.startswith("completion_recheck")
        for event in agent.events
    )


def test_durable_session_excludes_rejected_final_and_recheck_guidance(
    tmp_path,
):
    rejected = Message(role="assistant", content="rejected")
    accepted = Message(role="assistant", content="accepted")
    agent, _ = _agent(
        [[_call("run_command", "run-1")], rejected, accepted],
        session=Session(),
        session_id="completion-session",
    )

    agent.run("task")
    assert agent.last_run_record is not None
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    state = DurableSession(
        session_id="completion-session",
        created_at=now,
        updated_at=now,
        workspace=tmp_path.resolve(),
        model="test-model",
        session=agent.session,
        run_records=[agent.last_run_record],
    )
    store = JsonlDurableSessionStore(tmp_path / "sessions")

    store.save(state)
    loaded = store.load("completion-session")

    assert rejected not in loaded.session.items
    assert loaded.session.items[-1] == accepted
    assert rejected in [step.output for step in loaded.run_records[0].trace.steps]
    assert all(
        not (
            isinstance(item, Message)
            and item.content.startswith("[PureHarness Completion Recheck]")
        )
        for item in loaded.session.items
    )
