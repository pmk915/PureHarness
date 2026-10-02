import json

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.evaluation import (
    FailureDiagnoser,
    FailureType,
    StagnationEvaluator,
    StagnationObservation,
    StagnationStep,
    stagnation_steps_from_run_record,
    trajectory_from_run_record,
)
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.progress import ProgressTracker, tool_action_fingerprint
from pureharness.run_record import ModelInvocationRecord, RunRecordBuilder
from pureharness.runtime import ExecutionUsage
from pureharness.tool_history import ToolInteraction
from pureharness.trace import RunTrace, StepTrace


def _interaction(
    name="read_file", arguments=None, content="fixed", *,
    is_error=False, call_id="id",
):
    return ToolInteraction(
        ToolCall(
            name, {"path": "a.py"} if arguments is None else arguments, call_id,
        ),
        ToolResult(name, content, call_id, is_error),
    )


def _observation(*args, **kwargs):
    return StagnationObservation.from_interaction(_interaction(*args, **kwargs))


def _step(index, *observations, mutation=0, verification=0):
    return StagnationStep(index, observations, mutation, verification)


def _repeats(count=20, *, name="read_file", content="fixed", is_error=False):
    observation = _observation(name, content=content, is_error=is_error)
    return [_step(i, observation) for i in range(count)]


def _record(interactions):
    builder = RunRecordBuilder("offline", None)
    steps = []
    for index, interaction in enumerate(interactions):
        builder.record_model_invocation(ModelInvocationRecord(
            step=index, context_strategy="fixture", estimated_history_tokens=1,
            estimated_task_state_tokens=0, registered_tool_count=1,
            exposed_tool_count=1, estimated_tool_schema_tokens=0,
            selector_strategy="fixture", trajectory_compacted=False,
            compacted_source_units=0, compacted_tool_results=0,
        ))
        builder.record_tool_calls(1)
        builder.record_tool_execution()
        builder.record_tool_result(is_error=interaction.result.is_error)
        steps.append(StepTrace(index, [interaction.call], [interaction.result]))
    return builder.finalize(RunTrace(steps=steps, end_reason="max_steps_exceeded"))


def test_action_identity_reuses_progress_semantics():
    first = _interaction(arguments={"path": "a", "options": {"x": 1, "y": 2}})
    second = _interaction(
        arguments={"options": {"y": 2, "x": 1}, "path": "a"}, call_id="different",
    )
    observations = [StagnationObservation.from_interaction(i) for i in (first, second)]
    tracker = ProgressTracker()
    for interaction in (first, second):
        tracker.record_tool_action(interaction.call.name, interaction.call.arguments)
    snapshot = tracker.snapshot(ExecutionUsage(), logical_steps_completed=0)

    assert observations[0] == observations[1]
    assert observations[0].action_identity == tool_action_fingerprint(
        first.call.name, first.call.arguments,
    )
    assert snapshot.unique_tool_actions == 1
    assert snapshot.repeated_tool_actions == 1
    assert _observation(arguments={"items": [1, 2]}).action_identity != (
        _observation(arguments={"items": [2, 1]}).action_identity
    )
    assert _observation(arguments={"path": "./a"}).action_identity != (
        _observation(arguments={"path": "a"}).action_identity
    )


def test_identical_repeated_actions_and_results_produce_scalar_evidence():
    signal = StagnationEvaluator().evaluate(_repeats())

    assert signal.detected
    assert signal.reason == "unchanged_repetition_without_observed_progress"
    assert signal.evidence.to_dict() == {
        "window_size": 16, "observed_steps": 16, "start_step": 4, "end_step": 19,
        "active_steps": 16, "tool_action_count": 16, "unique_action_count": 1,
        "repeated_action_count": 16, "repeat_ratio": 1.0,
        "unchanged_result_repeat_count": 16, "changed_result_repeat_count": 0,
        "new_action_count": 0, "unknown_action_count": 0,
        "workspace_mutation_delta": 0, "verification_delta": 0,
        "tool_error_delta": 0,
    }
    assert json.loads(json.dumps(signal.to_dict(), allow_nan=False)) == signal.to_dict()


@pytest.mark.parametrize("name", ["read_file", "poll_process", "run_command"])
def test_changed_observations_are_progress_even_for_read_only_or_polling(name):
    steps = [_step(i, _observation(name, content=f"observed {i}")) for i in range(40)]

    signals = StagnationEvaluator().evaluate_windows(steps)

    assert not any(signal.detected for signal in signals)
    assert signals[-1].evidence.changed_result_repeat_count == 16
    assert signals[-1].evidence.new_action_count == 0


def test_even_one_changed_result_vetoes_a_repetition_warning():
    steps = _repeats()
    steps[-1] = _step(19, _observation(content="changed"))

    assert not StagnationEvaluator().evaluate(steps).detected


def test_return_to_an_old_result_is_still_a_change_from_the_last_observation():
    steps = _repeats()
    steps[-2] = _step(18, _observation(content="different"))
    signal = StagnationEvaluator().evaluate(steps)

    assert not signal.detected
    assert signal.evidence.changed_result_repeat_count == 2


def test_result_content_is_not_whitespace_normalized():
    assert _observation(content="a\nb").result_identity != (
        _observation(content="a b").result_identity
    )


@pytest.mark.parametrize("field", ["mutation", "verification"])
def test_repeated_reads_followed_by_explicit_progress_do_not_stagnate(field):
    steps = _repeats()
    steps[-1] = _step(19, _observation(), **{field: 1})

    signal = StagnationEvaluator().evaluate(steps)

    assert not signal.detected
    assert signal.reason == "mutation_or_verification_observed"


def test_continued_new_actions_prevent_stagnation():
    steps = [_step(i, *[
        _observation(arguments={"path": f"{i}-{j}"}) for j in range(5)
    ]) for i in range(40)]

    signals = StagnationEvaluator().evaluate_windows(steps)

    assert not any(signal.detected for signal in signals)
    assert signals[-1].evidence.new_action_count == 80


def test_short_benign_repetition_and_single_step_burst_are_not_stagnation():
    evaluator = StagnationEvaluator()
    assert not evaluator.evaluate(_repeats(15)).detected
    assert not evaluator.evaluate([_step(0, *([_observation()] * 100))]).detected
    empty = evaluator.evaluate([])
    assert not empty.detected
    assert empty.evidence.repeat_ratio == 0.0
    assert empty.evidence.start_step is None


def test_window_expires_progress_and_compares_to_last_result_not_first():
    steps = _repeats(6)
    steps[1] = _step(1, _observation(content="changed"), mutation=1)
    steps[2:] = [_step(i, _observation(content="changed")) for i in range(2, 6)]
    signals = StagnationEvaluator(window_size=4).evaluate_windows(steps)

    assert not signals[3].detected
    assert not signals[4].detected
    assert signals[5].detected
    assert signals[5].evidence.start_step == 2
    assert signals[5].evidence.unchanged_result_repeat_count == 4
    assert all(signal.evidence.observed_steps <= 4 for signal in signals)


def test_changed_error_state_participates_in_result_identity():
    assert _observation().result_identity != _observation(is_error=True).result_identity
    steps = _repeats()
    steps[-1] = _step(19, _observation(is_error=True))
    assert not StagnationEvaluator().evaluate(steps).detected


def test_tool_success_or_failure_alone_does_not_determine_progress():
    signal = StagnationEvaluator().evaluate(_repeats(is_error=True))
    assert signal.detected
    assert signal.evidence.tool_error_delta == 16
    # Different error observations remain new information, not stagnation.
    changing = [_step(i, _observation(content=str(i), is_error=True)) for i in range(20)]
    assert not StagnationEvaluator().evaluate(changing).detected


@pytest.mark.parametrize("field", ["workspace_mutation_delta", "verification_delta"])
def test_missing_progress_evidence_is_not_treated_as_zero(field):
    steps = _repeats()
    steps[-1] = replace(steps[-1], **{field: None})
    signal = StagnationEvaluator().evaluate(steps)
    assert not signal.detected
    assert signal.reason == "progress_evidence_unavailable"


def test_unknown_identity_is_not_conflated_with_another_action():
    observation = _observation(arguments={"unserializable": {1, 2}})
    assert observation.action_identity is None
    steps = [_step(i, observation) for i in range(20)]
    signal = StagnationEvaluator().evaluate(steps)
    assert not signal.detected
    assert signal.evidence.unknown_action_count == 16
    assert signal.evidence.repeated_action_count == 0


def test_inactive_steps_do_not_manufacture_an_active_stagnation_window():
    steps = _repeats()
    steps[-1] = _step(19)
    assert not StagnationEvaluator().evaluate(steps).detected


def test_deterministic_output_frozen_models_and_unchanged_source_history():
    interaction = _interaction(content="private-output" * 1000)
    before = deepcopy(interaction)
    observation = StagnationObservation.from_interaction(interaction)
    steps = [_step(i, observation) for i in range(20)]
    steps_before = deepcopy(steps)
    evaluator = StagnationEvaluator()
    assert evaluator.evaluate(steps) == evaluator.evaluate(iter(steps))
    assert interaction == before
    assert steps == steps_before
    assert "private-output" not in repr(observation)
    assert len(observation.result_identity) == 64
    with pytest.raises(FrozenInstanceError):
        evaluator.window_size = 2
    with pytest.raises(FrozenInstanceError):
        observation.is_error = True
    with pytest.raises(FrozenInstanceError):
        evaluator.evaluate(steps).detected = False


def test_make_mips_like_two_windows_without_benchmark_specific_logic():
    # Structural fixture only: exploratory identities, repeated stable reads,
    # another exploratory interval, and another repeated stable-read interval.
    known = [_observation(arguments={"path": str(i)}) for i in range(3)]
    steps = []
    for i in range(300):
        if 218 <= i <= 251 or 262 <= i <= 289 or i >= 291:
            observations = (known[i % 3], known[(i + 1) % 3])
        elif i < 3:
            observations = (known[i],)
        else:
            observations = (_observation(arguments={"path": f"new-{i}"}),)
        steps.append(_step(i, *observations))

    signals = StagnationEvaluator().evaluate_windows(steps)

    assert [s.evidence.end_step for s in signals if s.detected] == [
        *range(233, 252), *range(277, 290),
    ]
    assert signals[233].evidence.start_step == 218
    assert signals[277].evidence.start_step == 262


def test_non_stagnant_synthetic_trajectory_has_no_warning():
    steps = []
    for i in range(100):
        observations = [_observation()]
        if i % 8 == 0:
            observations.append(_observation(arguments={"path": f"new-{i}"}))
        steps.append(_step(i, *observations))
    assert not any(s.detected for s in StagnationEvaluator().evaluate_windows(steps))


def test_record_adapter_requires_explicit_coding_evidence_and_is_read_only():
    record = _record([_interaction(call_id=str(i)) for i in range(20)])
    before = record.to_dict()
    evaluator = StagnationEvaluator()
    without = stagnation_steps_from_run_record(record)
    assert evaluator.evaluate(without).reason == "progress_evidence_unavailable"
    snapshots = {i: CodingEvidenceSnapshot() for i in range(20)}
    steps = stagnation_steps_from_run_record(record, coding_evidence=snapshots)
    assert evaluator.evaluate(steps).detected
    snapshots[19] = CodingEvidenceSnapshot(workspace_mutations=1, verification_attempts=1)
    steps = stagnation_steps_from_run_record(record, coding_evidence=snapshots)
    assert steps[-1].workspace_mutation_delta == 1
    assert steps[-1].verification_delta == 1
    assert not evaluator.evaluate(steps).detected
    assert record.to_dict() == before
    assert FailureDiagnoser().diagnose(trajectory_from_run_record(record)).failure_type is (
        FailureType.UNKNOWN
    )


def test_record_adapter_uses_result_call_id_matching_not_zip_order():
    record = _record([_interaction()])
    other = _interaction(arguments={"path": "other"}, content="other", call_id="other")
    record.trace.steps[0].output.append(other.call)
    record.trace.steps[0].tool_result.insert(0, other.result)
    steps = stagnation_steps_from_run_record(record)
    assert steps[0].observations == (
        StagnationObservation.from_interaction(other), _observation(),
    )


def test_record_adapter_handles_message_steps_and_rejects_missing_pairs_or_counters():
    record = _record([_interaction()])
    record.trace.steps[0].output = Message("assistant", "done")
    record.trace.steps[0].tool_result = None
    assert stagnation_steps_from_run_record(record)[0].observations == ()
    with pytest.raises(ValueError, match="cover every trace step"):
        stagnation_steps_from_run_record(record, coding_evidence={})
    record.trace.steps[0].output = [_interaction().call]
    with pytest.raises(ValueError, match="completed tool pairs"):
        stagnation_steps_from_run_record(record)


def test_record_adapter_rejects_regressing_coding_counters():
    record = _record([_interaction(call_id=str(i)) for i in range(2)])
    with pytest.raises(ValueError, match="non-negative"):
        stagnation_steps_from_run_record(record, coding_evidence={
            0: CodingEvidenceSnapshot(workspace_mutations=1),
            1: CodingEvidenceSnapshot(),
        })


@pytest.mark.parametrize("size", [0, 1, -1, True, 2.0])
def test_invalid_window_size_is_rejected(size):
    with pytest.raises(ValueError):
        StagnationEvaluator(window_size=size)


@pytest.mark.parametrize("indexes", [(0, 0), (2, 1), (0, 2)])
def test_gaps_or_unordered_steps_are_not_treated_as_consecutive(indexes):
    with pytest.raises(ValueError, match="consecutive and ordered"):
        StagnationEvaluator().evaluate([_step(i, _observation()) for i in indexes])


def test_invalid_inputs_are_rejected():
    with pytest.raises(TypeError):
        StagnationEvaluator().evaluate([object()])
    with pytest.raises(ValueError):
        StagnationStep(True)
    with pytest.raises(ValueError):
        _step(0, _observation(), mutation=-1)
    with pytest.raises(ValueError):
        StagnationStep(0, (object(),))
