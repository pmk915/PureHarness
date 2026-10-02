import json

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from pureharness.agent import Agent
from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.completion import (
    CompletionReason, EvidenceAwareCodingCompletionPolicy, render_completion_recheck,
)
from pureharness.context import (
    ContextBudget, ContextBuilder, ContextLimits, TokenBudgetContextBuilder,
)
from pureharness.evaluation.stagnation import (
    StagnationEvaluator, StagnationObservation, StagnationStep, StagnationTracker,
)
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.model import (
    ContextWindowExceededError, MalformedModelOutputError, ModelError,
    RecoverableModelError,
)
from pureharness.observability import JsonlEventRenderer, event_to_wire
from pureharness.run_record import RunRecord
from pureharness.runtime import ExecutionBudget, ExecutionBudgetExceeded
from pureharness.session_store import DurableSession, JsonlDurableSessionStore
from pureharness.stagnation_advisory import (
    StagnationAdvisoryPhase, StagnationAdvisoryPolicy, StagnationAdvisoryState,
    render_stagnation_advisory,
)
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_history import ToolInteraction
from pureharness.tool_selection import StaticToolSelector, estimate_tool_schema_tokens
from pureharness.tools import Tool, ToolRegistry
from pureharness.trajectory_compaction import IdentityTrajectoryCompactor
from pureharness.verification import CommandToolObservation, CommandPurpose
from pureharness.workspace_discipline import WorkspaceMutation


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.contexts = []
        self.tools = []

    def generate(self, messages, tools):
        self.contexts.append(deepcopy(messages))
        self.tools.append(list(tools))
        output = next(self.outputs)
        if isinstance(output, BaseException):
            raise output
        return output


def _call(step, name="observe", arguments=None):
    return [ToolCall(name, {"value": "a"} if arguments is None else arguments, str(step))]


def _calls(count=18):
    return [_call(i) for i in range(count)]


def _message(content="done"):
    return Message("assistant", content)


def _guidance(context):
    return [item for item in context if isinstance(item, Message) and (
        item.content.startswith("[PureHarness Stagnation Advisory]")
    )]


def _events(agent):
    return [event for event in agent.events if event.type == "runtime_advisory_emitted"]


def _agent(outputs, *, enabled=True, tools=None, **kwargs):
    registry = tools or ToolRegistry()
    if tools is None:
        registry.register(Tool(
            name="observe", description="Offline observation.",
            parameters={"type": "object", "properties": {}},
            function=lambda value="a": "fixed",
        ))
    model = ScriptedModel(outputs)
    agent = Agent(
        model, tools=registry, max_steps=100,
        stagnation_advisory=enabled, **kwargs,
    )
    return agent, model


def _steps(count=65, *, mutations=(), verifications=()):
    observation = StagnationObservation.from_interaction(ToolInteraction(
        _call(0)[0], ToolResult("observe", "fixed", "0"),
    ))
    return [StagnationStep(
        i, (observation,), int(i in mutations), int(i in verifications),
    ) for i in range(count)]


def _simulate(steps):
    tracker = StagnationTracker()
    policy = StagnationAdvisoryPolicy()
    state = StagnationAdvisoryState()
    evidence = CodingEvidenceSnapshot()
    decisions = []
    for step in steps:
        if state.pending_signal is not None:
            state = policy.delivered(state, evidence, step.step)
            decisions.append((state.advisory_count, step.step))
            state = policy.clear_pending(state)
        evidence = CodingEvidenceSnapshot(
            workspace_mutations=evidence.workspace_mutations + step.workspace_mutation_delta,
            verification_attempts=evidence.verification_attempts + step.verification_delta,
        )
        state = policy.after_tool_step(state, tracker.record_step(step), evidence)
    return decisions, state


def test_policy_false_signal_does_not_queue_or_rearm():
    policy = StagnationAdvisoryPolicy()
    state = StagnationAdvisoryState()
    snapshot = CodingEvidenceSnapshot()
    false = StagnationEvaluator().evaluate(_steps(1))
    assert policy.after_tool_step(state, false, snapshot) == state
    true = StagnationEvaluator().evaluate(_steps(17))
    queued = policy.after_tool_step(state, true, snapshot)
    assert queued.advisory_count == 0
    assert queued.pending_signal is true
    latched = policy.delivered(queued, snapshot, 17)
    # Physical retries must not count another intervention.
    assert policy.delivered(latched, snapshot, 17) == latched
    latched = policy.clear_pending(latched)
    assert latched.phase is StagnationAdvisoryPhase.LATCHED
    assert policy.after_tool_step(latched, false, snapshot) == latched
    assert policy.after_tool_step(latched, true, snapshot) == latched


@pytest.mark.parametrize("progress", ["mutations", "verifications"])
def test_progress_rearms_second_advisory_but_absolute_cap_blocks_third(progress):
    steps = _steps(**{progress: (20, 40)})
    before = deepcopy(steps)
    decisions, state = _simulate(steps)
    assert decisions == [(1, 17), (2, 37)]
    assert state.advisory_count == 2
    assert state.pending_signal is None
    assert _simulate(steps) == (decisions, state)
    assert steps == before


def test_rearm_requires_progress_after_delivery_not_earlier_progress():
    signal = StagnationEvaluator().evaluate(_steps(17))
    policy = StagnationAdvisoryPolicy()
    facts = CodingEvidenceSnapshot(workspace_mutations=3, verification_attempts=2)
    state = policy.after_tool_step(StagnationAdvisoryState(), signal, facts)
    state = policy.clear_pending(policy.delivered(state, facts, 17))
    assert policy.after_tool_step(state, signal, facts) == state


def test_policy_enforces_next_logical_step_delivery():
    policy = StagnationAdvisoryPolicy()
    state = policy.after_tool_step(
        StagnationAdvisoryState(), StagnationEvaluator().evaluate(_steps(17)),
        CodingEvidenceSnapshot(),
    )
    with pytest.raises(ValueError, match="next logical step"):
        policy.delivered(state, CodingEvidenceSnapshot(), 18)


def test_no_signal_and_disabled_runs_have_no_advisory(monkeypatch):
    agent, model = _agent([*_calls(3), _message()])
    assert agent.run("task") == "done"
    assert not _events(agent)
    assert not any(_guidance(c) for c in model.contexts)

    monkeypatch.setattr(
        "pureharness.agent.StagnationTracker",
        lambda: pytest.fail("disabled runs must not instantiate evaluator state"),
    )
    agent, model = _agent([*_calls(40), _message()], enabled=False)
    assert agent.run("task") == "done"
    assert agent.stagnation_advisories_emitted == 0
    assert not _events(agent)
    assert not any(_guidance(c) for c in model.contexts)
    assert all("stagnation_advisory_present" not in e.data for e in agent.events)


def test_first_signal_delivered_next_step_continuous_signals_emit_once():
    agent, model = _agent([*_calls(50), _message()])
    assert agent.run("task") == "done"
    assert [i for i,c in enumerate(model.contexts) if _guidance(c)] == [17]
    assert agent.stagnation_advisories_emitted == 1
    event, = _events(agent)
    assert event.data["detected_at_step"] == 16
    assert event.data["delivered_at_step"] == event.data["step"] == 17
    assert agent.events.index(event) > next(
        i for i,e in enumerate(agent.events) if e.type == "model_started" and e.data["step"] == 17
    )
    assert not any("Stagnation Advisory" in str(item) for item in agent.session.items)


@pytest.mark.parametrize("change", ["action", "result"])
def test_new_action_or_changed_result_alone_does_not_rearm(change):
    outputs = _calls(60)
    outputs[20] = _call(20, arguments={"value": "changed"})
    agent, model = _agent([*outputs, _message()])
    if change == "result":
        # Same action identity; change only its observable result once.
        outputs[20] = _call(20)
        seen = 0
        def observe(value="a"):
            nonlocal seen
            seen += 1
            return "different" if seen == 21 else "fixed"
        agent.tools.get("observe").function = observe
        model.outputs = iter([*outputs, _message()])
    assert agent.run("task") == "done"
    assert len(_events(agent)) == 1
    assert agent.stagnation_advisories_emitted == 1


class MutationPrecondition:
    def reset(self):
        pass

    def prepare(self, tool, arguments):
        return None

    def revalidate(self, tool, arguments, prepared):
        return prepared

    def record_success(self, tool, arguments, prepared, result):
        if tool.name == "apply_patch":
            return WorkspaceMutation("a.py", "apply_patch", "patched")
        return None


@pytest.mark.parametrize("progress", ["mutation", "verification"])
def test_runtime_progress_rearms_and_emits_two_not_three(progress):
    registry = ToolRegistry()
    registry.register(Tool("observe", "observe", {}, lambda value="a": "fixed"))
    registry.register(Tool("apply_patch", "mutate", {}, lambda: "ok"))
    def command() -> CommandToolObservation:
        return CommandToolObservation(CommandPurpose.VERIFICATION, 1, "check failed")
    registry.register(Tool(
        "run_command", "marked check",
        {"properties": {"purpose": {"enum": ["general", "verification"]}}},
        lambda purpose="verification": command(), category="execution",
    ))
    # Native evidence requires the actual function's structured return type.
    registry.get("run_command").function.__annotations__["return"] = CommandToolObservation
    outputs = _calls(65)
    for i in (20, 40):
        outputs[i] = _call(
            i, "apply_patch" if progress == "mutation" else "run_command",
            {} if progress == "mutation" else {"purpose": "verification"},
        )
    agent, model = _agent(
        [*outputs, _message()], tools=registry,
        tool_executor=ToolExecutor(registry, precondition=MutationPrecondition()),
    )
    assert agent.run("task") == "done"
    assert [e.data["advisory_index"] for e in _events(agent)] == [1, 2]
    assert [e.data["delivered_at_step"] for e in _events(agent)] == [17, 37]
    assert sum(bool(_guidance(c)) for c in model.contexts) == 2


def test_state_resets_between_runs_and_does_not_evaluate_previous_session():
    agent, model = _agent([*_calls(18), _message(), *_calls(18), _message()])
    assert agent.run("one") == "done"
    assert agent.stagnation_advisories_emitted == 1
    assert agent.run("two") == "done"
    assert agent.stagnation_advisories_emitted == 1
    assert len(_events(agent)) == 1
    assert _events(agent)[0].data["detected_at_step"] == 16
    assert sum(bool(_guidance(c)) for c in model.contexts) == 2


@pytest.mark.parametrize("error", [
    RecoverableModelError("temporary"), MalformedModelOutputError("malformed"),
    ContextWindowExceededError("provider rejected"),
])
def test_physical_retry_and_context_recovery_retain_advisory_and_count_once(error):
    builder = ContextBuilder(trajectory_compactor=IdentityTrajectoryCompactor())
    agent, model = _agent([*_calls(17), error, _call(17), _message()], context_builder=builder)
    assert agent.run("task") == "done"
    assert _guidance(model.contexts[17]) == _guidance(model.contexts[18]) == [
        render_stagnation_advisory(),
    ]
    assert not _guidance(model.contexts[19])
    assert len(_events(agent)) == agent.stagnation_advisories_emitted == 1
    assert agent.last_run_record.model_call_count == len(agent.last_run_record.model_invocations) == 19
    if isinstance(error, ContextWindowExceededError):
        recovery = next(e for e in agent.events if e.type == "context_recovering")
        assert len(model.contexts[18]) < len(model.contexts[17])
        assert recovery.data["previous_estimated_request_tokens"] > recovery.data["recovered_estimated_request_tokens"]


def test_mixed_retry_then_recovery_belong_to_one_advisory_invocation():
    agent, model = _agent(
        [*_calls(17), MalformedModelOutputError("malformed"),
         ContextWindowExceededError("overflow"), _call(17), _message()],
        context_builder=ContextBuilder(trajectory_compactor=IdentityTrajectoryCompactor()),
    )
    assert agent.run("task") == "done"
    assert all(_guidance(c) == [render_stagnation_advisory()] for c in model.contexts[17:20])
    assert not _guidance(model.contexts[20])
    assert len(_events(agent)) == agent.stagnation_advisories_emitted == 1


def test_delivered_advisory_is_counted_even_if_provider_terminally_fails():
    agent, model = _agent([*_calls(17), ModelError("terminal")])
    with pytest.raises(ModelError):
        agent.run("task")
    assert agent.stagnation_advisories_emitted == 1
    assert len(_events(agent)) == 1
    assert agent._stagnation_state.pending_signal is None
    assert _guidance(model.contexts[-1])


def test_step_or_physical_budget_prevents_delivery_without_false_emission():
    for physical_limit in (False, True):
        agent, model = _agent(
            _calls(17) if not physical_limit else [*_calls(17), _message()],
            execution_budget=ExecutionBudget(max_model_attempts=17) if physical_limit else None,
        )
        if not physical_limit:
            agent.max_steps = 17
        with pytest.raises(ExecutionBudgetExceeded if physical_limit else RuntimeError):
            agent.run("task")
        assert agent.stagnation_advisories_emitted == 0
        assert not _events(agent)
        assert not any(_guidance(c) for c in model.contexts)


class FixedEstimator:
    def __init__(self, advisory_tokens=5):
        self.advisory_tokens = advisory_tokens

    def estimate(self, items):
        if not items:
            return 0
        if isinstance(items[0], Message):
            if _guidance(items):
                return self.advisory_tokens
            return 2
        return 2


def test_advisory_is_accounted_as_pinned_guidance_and_can_bound_history():
    builder = TokenBudgetContextBuilder(
        ContextBudget(8000), token_estimator=FixedEstimator(),
        trajectory_compactor=IdentityTrajectoryCompactor(),
    )
    agent, model = _agent(
        [*_calls(18), _message()], context_builder=builder,
    )
    usable = estimate_tool_schema_tokens(agent.tools.list_tools()) + 2 + 5 + 30
    agent.context_limits = ContextLimits(usable + 1, 1)
    assert agent.run("task") == "done"
    delivered = next(e for e in agent.events if e.type == "context_built" and e.data["step"] == 17)
    assert delivered.data["stagnation_advisory_present"]
    assert delivered.data["estimated_stagnation_advisory_tokens"] == 5
    assert delivered.data["estimated_request_tokens"] == sum([
        delivered.data["estimated_history_tokens"],
        delivered.data["estimated_task_state_tokens"], 5,
        agent.last_run_record.model_invocations[17].estimated_tool_schema_tokens,
    ])
    assert delivered.data["estimated_request_tokens"] <= usable
    assert delivered.data["bounded_history_applied"] is True
    assert all(i.estimated_history_tokens <= 8000 for i in agent.last_run_record.model_invocations)
    assert _guidance(model.contexts[17])


@pytest.mark.parametrize("advisory_tokens", [100, 3])
def test_optional_advisory_that_cannot_fit_is_omitted_not_a_new_failure(advisory_tokens):
    builder = ContextBuilder(
        token_estimator=FixedEstimator(advisory_tokens=advisory_tokens),
        trajectory_compactor=IdentityTrajectoryCompactor(),
    )
    agent, model = _agent(
        [*_calls(18), _message()], context_builder=builder,
    )
    usable = estimate_tool_schema_tokens(agent.tools.list_tools()) + 2 + 4
    agent.context_limits = ContextLimits(usable + 1, 1)
    assert agent.run("task") == "done"
    assert agent.stagnation_advisories_emitted == 0
    assert not _events(agent)
    assert not any(_guidance(c) for c in model.contexts)


def test_disabled_default_and_explicit_false_have_identical_context_and_records():
    first, first_model = _agent([*_calls(18), _message()], enabled=False,
                                run_id_factory=lambda: "fixed")
    second_model = ScriptedModel([*_calls(18), _message()])
    second = Agent(second_model, tools=first.tools, max_steps=100,
                   run_id_factory=lambda: "fixed")
    assert first.run("task") == second.run("task")
    assert first_model.contexts == second_model.contexts
    assert first.last_run_record.to_dict() == second.last_run_record.to_dict()
    # Timings and timestamps are occurrence observations, not compatibility data.
    assert [e.type for e in first.events] == [e.type for e in second.events]
    assert [e.data for e in first.events if e.type == "context_built"] == [
        e.data for e in second.events if e.type == "context_built"
    ]


def test_completion_recheck_stays_independent_and_tool_selection_unchanged():
    registry = ToolRegistry()
    registry.register(Tool(
        "observe", "observe", {}, lambda value="a": "fixed",
    ))
    registry.register(Tool(
        "run_command", "execution", {}, lambda: "ok", category="execution",
    ))
    registry.register(Tool("hidden", "hidden", {}, lambda: pytest.fail("hidden")))
    agent, model = _agent(
        [*_calls(17), _call(17, "run_command", {}), _message("candidate"),
         _call(19), _message("accepted")],
        tools=registry, completion_policy=EvidenceAwareCodingCompletionPolicy(),
        tool_selector=StaticToolSelector(["observe", "run_command"]),
    )
    assert agent.run("task") == "accepted"
    assert agent.completion_rechecks_used == 1
    assert agent.stagnation_advisories_emitted == 1
    assert _guidance(model.contexts[17])
    assert not _guidance(model.contexts[19])
    assert any("Completion Recheck" in str(x) for x in model.contexts[19])
    assert all([t.name for t in ts] == ["observe", "run_command"] for ts in model.tools)
    assert not any("Advisory" in str(x) or "Completion Recheck" in str(x) for x in agent.session.items)


def test_both_ephemeral_guidance_items_have_deterministic_order_and_survive_recompile():
    agent, _ = _agent([], context_builder=ContextBuilder(
        token_estimator=FixedEstimator(),
        trajectory_compactor=IdentityTrajectoryCompactor(),
    ))
    completion = render_completion_recheck(
        CompletionReason.EXECUTION_WITHOUT_STRUCTURED_MUTATION,
    )
    agent._pending_completion_recheck = completion
    agent._stagnation_state = agent._stagnation_policy.after_tool_step(
        StagnationAdvisoryState(), StagnationEvaluator().evaluate(_steps(17)),
        CodingEvidenceSnapshot(),
    )
    history = [Message("user", "old"), Message("assistant", "previous"),
               Message("user", "task")]
    original = deepcopy(history)
    prepared = agent._prepare_context(17, history, agent.task_state_reducer.reduce(history))
    recovered = agent._recover_context(history, prepared, prepared.compiled.estimated_tokens // 2)
    for context in (prepared, recovered):
        items = context.model_items
        assert items.index(completion) < items.index(render_stagnation_advisory())
        assert context.estimated_completion_recheck_tokens == 2
        assert context.estimated_stagnation_advisory_tokens == 5
    assert history == original
    assert agent.session.items == []


def test_jsonl_event_is_additive_and_filters_raw_data_and_persistence_is_unchanged(tmp_path):
    output = []
    agent, model = _agent(
        [*_calls(18), _message()], listeners=[JsonlEventRenderer(output.append)],
        session_id="advisory-session",
    )
    assert agent.run("task") == "done"
    lines = [json.loads(line) for line in output]
    wire, = [line for line in lines if line["event"] == "runtime_advisory_emitted"]
    assert wire["schema_version"] == 1
    assert wire["step"] == 17
    assert wire["payload"] == {
        "kind": "stagnation", "advisory_index": 1, "detected_at_step": 16,
        "delivered_at_step": 17, "window_size": 16, "repeated_action_count": 16,
        "unchanged_result_repeat_count": 16, "new_action_count": 0,
        "workspace_mutation_delta": 0, "verification_delta": 0,
    }
    event = _events(agent)[0]
    event.data.update({"arguments": "private", "stdout": "private", "advisory": "private"})
    assert event_to_wire(event)["payload"] == wire["payload"]
    record = agent.last_run_record
    assert record.schema_version == 2
    assert RunRecord.from_json(record.to_json()).to_dict() == record.to_dict()
    assert "stagnation" not in record.to_json()
    now = datetime(2026, 10, 3, tzinfo=timezone.utc)
    durable = DurableSession("advisory-session", now, now, tmp_path.resolve(), "fake", agent.session, [record])
    store = JsonlDurableSessionStore(tmp_path / "sessions")
    store.save(durable)
    restored = store.load("advisory-session")
    assert restored.session.items == agent.session.items
    assert restored.run_records[0].to_dict() == record.to_dict()
    assert not any("Stagnation Advisory" in str(x) for x in restored.session.items)


def test_stable_advisory_is_generic_system_guidance():
    message = render_stagnation_advisory()
    assert message.role == "system"
    assert message == render_stagnation_advisory()
    for text in ("vm.js", "make-mips", "write_file", "run_command", "node", "task has failed"):
        assert text not in message.content


def test_incremental_tracker_matches_offline_windows_without_mutating_inputs():
    steps = _steps(mutations=(20,), verifications=(40,))
    before = deepcopy(steps)
    tracker = StagnationTracker()
    assert tuple(tracker.record_step(step) for step in steps) == (
        StagnationEvaluator().evaluate_windows(steps)
    )
    assert steps == before


def test_agent_rejects_non_boolean_opt_in():
    with pytest.raises(ValueError, match="must be bool"):
        Agent(ScriptedModel([]), stagnation_advisory=1)
