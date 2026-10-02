import json
import tracemalloc

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.evaluation import (
    ProgressGapEvidence,
    ProgressGapEvaluator,
    StagnationEvaluator,
    StagnationObservation,
    StagnationStep,
    stagnation_steps_from_run_record,
)
from pureharness.messages import ToolCall, ToolResult
from pureharness.run_record import ModelInvocationRecord, RunRecordBuilder
from pureharness.tool_history import ToolInteraction
from pureharness.trace import RunTrace, StepTrace


def observation(action="read", result="fixed", error=False):
    return StagnationObservation(action, result, error)


def step(index, *items, mutation=0, verification=0):
    return StagnationStep(index, items, mutation, verification)


def test_mutation_anchor_resets_gap_but_not_prefix_action_history():
    timeline = ProgressGapEvaluator().evaluate_steps([
        step(0, observation()), step(1, observation()),
        step(2, observation(), mutation=1), step(3, observation()),
    ])
    assert timeline[1].steps_since_anchor == 2
    anchor, later = timeline[2:]
    assert anchor.last_progress_anchor_step == 2
    assert anchor.steps_since_last_structured_mutation == 0
    assert anchor.span_start_step == 3
    assert anchor.active_steps == anchor.steps_since_anchor == 0
    assert anchor.tool_actions_since_anchor == 0
    assert later.steps_since_anchor == later.active_steps == 1
    assert later.steps_since_last_structured_mutation == 1
    assert later.repeated_action_count_since_anchor == 1
    assert later.unchanged_repeat_count_since_anchor == 1
    assert later.new_action_count_since_anchor == 0


def test_verification_anchor_resets_gap_and_its_distance_not_mutation_distance():
    timeline = ProgressGapEvaluator().evaluate_steps([
        step(0, observation(), mutation=1), step(1, observation()),
        # An explicit attempt is an anchor even when its result is an error.
        step(2, observation("verify", "failed", error=True), verification=1),
        step(3, observation()),
    ])
    assert timeline[2].last_verification_attempt_step == 2
    assert timeline[2].steps_since_last_verification_attempt == 0
    assert timeline[2].steps_since_last_structured_mutation == 2
    assert timeline[2].steps_since_anchor == 0
    assert timeline[3].steps_since_last_verification_attempt == 1
    assert timeline[3].steps_since_last_structured_mutation == 3


def test_novelty_and_changed_observations_do_not_reset_gap_age():
    evidence = ProgressGapEvaluator().evaluate([
        step(0, observation(), mutation=1),
        step(1, observation("new")), step(2, observation("new", "changed")),
        step(3, observation("another")),
    ])
    assert evidence.steps_since_anchor == 3
    assert evidence.last_progress_anchor_step == 0
    assert evidence.active_steps == 3
    assert evidence.new_action_count_since_anchor == evidence.unique_action_growth == 2
    assert evidence.repeated_action_count_since_anchor == 1
    assert evidence.changed_observation_count_since_anchor == 1


def test_repeat_and_result_comparison_include_within_step_order_and_error_flag():
    evidence = ProgressGapEvaluator().evaluate([
        step(0, observation("a", "first")),
        step(1, observation("a", "first"), observation("a", "second")),
        step(2, observation("a", "first"), observation("b", error=True)),
    ])
    assert evidence.tool_actions_since_anchor == 5
    assert evidence.new_action_count_since_anchor == 2
    assert evidence.repeated_action_count_since_anchor == 3
    assert evidence.unchanged_repeat_count_since_anchor == 1
    assert evidence.changed_observation_count_since_anchor == 2
    assert evidence.tool_error_count_since_anchor == 1
    assert evidence.repeat_ratio == 3 / 5
    assert evidence.unchanged_repeat_ratio == 1 / 5


def test_no_actions_do_not_fabricate_activity_or_reset_gap():
    evaluator = ProgressGapEvaluator()
    assert evaluator.evaluate([]) == ProgressGapEvidence()
    assert evaluator.evaluate_steps([]) == ()
    empty = evaluator.evaluate([step(0), step(1)])
    assert empty.steps_since_anchor == 2
    assert empty.active_steps == empty.tool_actions_since_anchor == 0
    assert empty.repeat_ratio == empty.unchanged_repeat_ratio == 0.0
    later = evaluator.evaluate([step(0, observation()), step(1), step(2, observation())])
    assert later.steps_since_anchor == 3
    assert later.active_steps == 2
    assert later.repeated_action_count_since_anchor == 1


def test_read_only_trajectory_is_representable_without_inventing_an_anchor():
    evidence = ProgressGapEvaluator().evaluate(step(i, observation()) for i in range(100))
    assert evidence.steps_since_anchor == evidence.active_steps == 100
    assert evidence.last_progress_anchor_step is None
    assert evidence.steps_since_last_structured_mutation is None
    assert evidence.steps_since_last_verification_attempt is None
    assert evidence.new_action_count_since_anchor == 1
    assert evidence.repeated_action_count_since_anchor == 99
    assert evidence.progress_evidence_complete


def test_unknown_counters_and_identities_are_not_inferred_as_zero():
    evaluator = ProgressGapEvaluator()
    unknown = evaluator.evaluate([
        step(0, observation(), mutation=1),
        step(1, observation(None), mutation=None), step(2, observation()),
    ])
    assert not unknown.progress_evidence_complete
    assert unknown.last_structured_mutation_step == 0
    assert unknown.steps_since_last_structured_mutation is None
    assert unknown.unknown_action_count_since_anchor == 1
    assert unknown.steps_since_anchor == 2
    # A later known anchor starts a fresh span; unknown earlier evidence stays
    # unknown for the distance to the earlier, different kind of anchor.
    later = evaluator.evaluate([
        step(0, observation(), mutation=1), step(1, mutation=None),
        step(2, verification=1), step(3, observation()),
    ])
    assert later.progress_evidence_complete
    assert later.steps_since_last_structured_mutation is None
    assert later.steps_since_last_verification_attempt == 1


def test_reobserving_a_counter_does_not_restore_a_distance_until_a_new_anchor():
    evidence = ProgressGapEvaluator().evaluate([
        step(0, verification=1), step(1, verification=None), step(2),
    ])
    assert evidence.steps_since_last_verification_attempt is None
    restored = ProgressGapEvaluator().evaluate([
        step(0, verification=1), step(1, verification=None),
        step(2, verification=1), step(3),
    ])
    assert restored.steps_since_last_verification_attempt == 1


def test_same_step_multiple_anchors_exclude_whole_batch_without_order_inference():
    evidence = ProgressGapEvaluator().evaluate([
        step(0, observation()),
        step(1, observation("x"), observation("y"), mutation=2, verification=3),
    ])
    assert evidence.workspace_mutation_delta == 2
    assert evidence.verification_delta == 3
    assert evidence.last_structured_mutation_step == evidence.last_verification_attempt_step == 1
    assert evidence.tool_actions_since_anchor == evidence.steps_since_anchor == 0


def test_partial_prefix_age_is_observed_steps_not_absolute_step_number():
    evidence = ProgressGapEvaluator().evaluate([step(100, observation()), step(101)])
    assert evidence.span_start_step == 100
    assert evidence.steps_since_anchor == 2
    assert evidence.last_progress_anchor_step is None


def test_synthetic_successful_exploration_and_failed_exploration_are_not_judged():
    # External labels are deliberately not inputs. A successful read-only task
    # can have exactly the same exploratory evidence as an unsuccessful one.
    exploration = [step(i, observation(str(i))) for i in range(40)]
    successful_control = ProgressGapEvaluator().evaluate(exploration)
    failed_control = ProgressGapEvaluator().evaluate(deepcopy(exploration))
    assert successful_control == failed_control
    assert successful_control.steps_since_anchor == 40
    assert successful_control.unique_action_growth == 40
    assert not hasattr(successful_control, "detected")
    assert not hasattr(successful_control, "healthy")
    assert not any(s.detected for s in StagnationEvaluator().evaluate_windows(exploration))


def test_failed_synthetic_repetition_with_periodic_novelty_retains_long_gap():
    steps = [step(i, observation(), *(
        [observation(f"new-{i}")] if i % 8 == 0 else []
    )) for i in range(100)]
    evidence = ProgressGapEvaluator().evaluate(steps)
    assert evidence.steps_since_anchor == evidence.active_steps == 100
    assert evidence.new_action_count_since_anchor == 14
    assert evidence.repeated_action_count_since_anchor == 99
    assert evidence.unchanged_repeat_count_since_anchor == 99
    assert not any(s.detected for s in StagnationEvaluator().evaluate_windows(steps))


def test_deterministic_frozen_json_serializable_evidence_and_unchanged_input():
    steps = [step(i, observation(str(i % 3))) for i in range(20)]
    before = deepcopy(steps)
    evaluator = ProgressGapEvaluator()
    first = evaluator.evaluate_steps(steps)
    assert first == evaluator.evaluate_steps(iter(steps))
    assert first[-1] == evaluator.evaluate(steps)
    assert steps == before
    assert json.loads(json.dumps(first[-1].to_dict(), allow_nan=False)) == first[-1].to_dict()
    with pytest.raises(FrozenInstanceError):
        first[-1].active_steps = 0
    with pytest.raises(FrozenInstanceError):
        evaluator.new_field = True


def test_streaming_is_lazy_and_repeated_steps_do_not_accumulate_history():
    consumed = []

    def source():
        for i in range(5):
            consumed.append(i)
            yield step(i, observation())

    iterator = ProgressGapEvaluator().iter_evidence(source())
    assert consumed == []
    assert next(iterator).step == 0
    assert consumed == [0]
    iterator.close()
    tracemalloc.start()
    try:
        evidence = ProgressGapEvaluator().evaluate(
            step(i, observation()) for i in range(20_000)
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert evidence.active_steps == 20_000
    assert peak < 100_000  # Only hashes/scalars, not 20,000 input/output steps.


def test_existing_record_adapter_preserves_raw_history_and_schemas():
    call = ToolCall("read_file", {"path": "a"}, "id")
    result = ToolResult("read_file", "private raw output", "id")
    builder = RunRecordBuilder("offline", None)
    builder.record_model_invocation(ModelInvocationRecord(
        step=0, context_strategy="fixture", estimated_history_tokens=1,
        estimated_task_state_tokens=0, registered_tool_count=1,
        exposed_tool_count=1, estimated_tool_schema_tokens=0,
        selector_strategy="fixture", trajectory_compacted=False,
        compacted_source_units=0, compacted_tool_results=0,
    ))
    builder.record_tool_calls(1)
    builder.record_tool_execution()
    builder.record_tool_result(is_error=False)
    record = builder.finalize(RunTrace(
        steps=[StepTrace(0, [call], [result])], end_reason="completed",
    ))
    before = deepcopy(record.to_dict())
    steps = stagnation_steps_from_run_record(record, coding_evidence={0: CodingEvidenceSnapshot()})
    evidence = ProgressGapEvaluator().evaluate(steps)
    assert evidence.progress_evidence_complete
    assert evidence.steps_since_anchor == 1
    assert "private raw output" not in repr(evidence)
    assert record.to_dict() == before
    without_counters = ProgressGapEvaluator().evaluate(stagnation_steps_from_run_record(record))
    assert not without_counters.progress_evidence_complete


@pytest.mark.parametrize("indexes", [(0, 0), (2, 1), (0, 2)])
def test_nonconsecutive_steps_are_rejected(indexes):
    with pytest.raises(ValueError, match="consecutive and ordered"):
        ProgressGapEvaluator().evaluate(step(i) for i in indexes)


def test_invalid_input_and_inconsistent_evidence_are_rejected():
    with pytest.raises(TypeError, match="StagnationStep"):
        ProgressGapEvaluator().evaluate([object()])
    for changes in [
        {"active_steps": True}, {"steps_since_anchor": -1},
        {"progress_evidence_complete": 1}, {"active_steps": 1},
        {"active_steps": None},
        {"new_action_count_since_anchor": 1}, {"tool_error_count_since_anchor": 1},
        {"unchanged_repeat_count_since_anchor": 1},
    ]:
        with pytest.raises(ValueError):
            replace(ProgressGapEvidence(), **changes)
