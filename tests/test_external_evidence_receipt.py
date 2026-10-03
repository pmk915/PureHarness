import json
import subprocess

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from scripts import external_evidence_receipt as extractor
from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.evaluation.external_evidence import ExternalEvidenceReceipt
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.progress import ProgressTracker
from pureharness.run_record import ModelInvocationRecord, RunRecordBuilder
from pureharness.runtime import ExecutionUsage
from pureharness.trace import RunTrace, StepTrace


REVISION = "a" * 40
SECRET = "fixture-API-key-and-private-output-never-copy"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def events_path(trial):
    return trial / "agent/pureharness-events.jsonl"


def write_events(trial, events):
    events_path(trial).write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def read_events(trial):
    return [json.loads(line) for line in events_path(trial).read_text().splitlines()]


def tree_bytes(path):
    return {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def make_trial(
    job, name="trial-a", *, reward=1.0, end_reason="completed",
    exception=None, agent="pureharness", count=20, mutation_at=None,
    verification_at=None, advisory=False,
):
    trial = job / name
    write_json(job / "result.json", {"id": "job-id", "private": SECRET})
    write_json(job / "lock.json", {"harbor": {"version": "0.23.0"}, "env": {"API_KEY": SECRET}})
    write_json(trial / "result.json", {
        "id": name + "-id", "trial_name": name, "task_name": "fixture/task",
        "task_id": {"ref": "sha256:task"}, "task_checksum": "c" * 64,
        "source": "fixture/dataset", "agent_info": {
            "name": agent, "version": REVISION,
            "model_info": {"name": "recorded-model", "provider": "recorded-provider"},
        },
        "config": {"agent": {"env": {"API_KEY": SECRET}, "model_name": "wrong-model"}},
        "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
        "exception_info": None if exception is None else {
            "exception_type": exception, "exception_message": SECRET,
            "exception_traceback": SECRET,
        },
    })
    if agent == "oracle":
        return trial
    builder = RunRecordBuilder("fixture-run", None)
    trace = []
    events = []
    progress = ProgressTracker()
    mutations = verifications = 0

    def event(kind, payload, index=None):
        return {"schema_version": 1, "run_id": "fixture-run", "event": kind,
                "payload": payload, **({"step": index} if index is not None else {})}

    for index in range(count + (end_reason == "completed")):
        builder.record_model_invocation(ModelInvocationRecord(
            step=index, context_strategy="fixture", estimated_history_tokens=100 + index,
            estimated_task_state_tokens=0, registered_tool_count=3, exposed_tool_count=3,
            estimated_tool_schema_tokens=0, selector_strategy="fixture",
            trajectory_compacted=False, compacted_source_units=0, compacted_tool_results=0,
        ))
        events.append(event("context_built", {
            "estimated_history_tokens": 100 + index, "history_token_budget": 8000,
            "trajectory_compacted": False, "private_raw_output": SECRET,
        }, index))
        if index == count:
            trace.append(StepTrace(index, Message("assistant", SECRET)))
        else:
            tool = "write_file" if index == mutation_at else "run_command" if index == verification_at else "read_file"
            args = {"path": "fixture", "private": SECRET}
            call = ToolCall(tool, args, str(index))
            result = ToolResult(tool, SECRET, str(index))
            trace.append(StepTrace(index, [call], [result]))
            builder.record_tool_calls(1)
            builder.record_tool_execution()
            builder.record_tool_result(is_error=False)
            progress.record_tool_action(tool, args)
            progress.record_tool_result(is_error=False)
        mutations += index == mutation_at
        verifications += index == verification_at
        snapshot = CodingEvidenceSnapshot(
            workspace_mutations=mutations, verification_attempts=verifications,
            verification_exit_zero=verifications,
        )
        events.append(event("coding_evidence_snapshot", {
            **snapshot.__dict__, "terminal": index == count,
        }, index))
        if advisory and index == 17:
            events.append(event("runtime_advisory_emitted", {
                "kind": "stagnation", "advisory_index": 1,
                "detected_at_step": 16, "delivered_at_step": 17,
                "command": SECRET, "stdout": SECRET,
            }, index))
    record = builder.finalize(RunTrace(steps=trace, end_reason=end_reason))
    write_json(trial / "agent/pureharness-run-record.json", record.to_dict())
    final_progress = progress.snapshot(
        ExecutionUsage(tool_calls=count, model_attempts=len(trace)),
        logical_steps_completed=len(trace),
    )
    events.append(event("progress_snapshot", {**final_progress.__dict__, "terminal": True}, len(trace) - 1))
    terminal = "agent_completed" if end_reason == "completed" else "agent_failed"
    events.append(event(terminal, {"end_reason": end_reason, "step_count": len(trace)}))
    write_events(trial, events)
    return trial


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline extraction must not spawn processes")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.mark.parametrize("reward,reason,exception", [
    (1.0, "completed", None), (0.0, "completed", None),
    (0.0, "max_steps_exceeded", "NonZeroAgentExitCodeError"),
    (0.0, "model_error", "AgentExecutionError"),
    (None, "model_error", "AgentExecutionError"),
    (1.0, "max_steps_exceeded", "NonZeroAgentExitCodeError"),
])
def test_external_reward_exception_and_protocol_completion_are_independent(tmp_path, reward, reason, exception):
    trial = make_trial(tmp_path / "job", reward=reward, end_reason=reason, exception=exception)
    receipt = extractor.create_receipt(trial)
    assert receipt.external.reward == reward
    assert receipt.external.exception_type == exception
    assert receipt.runtime.end_reason == reason
    assert receipt.runtime.protocol_completed is (reason == "completed")
    assert receipt.runtime.tool_call_count == receipt.runtime.tool_execution_count == 20
    assert receipt.external.pureharness_revision == REVISION
    assert receipt.external.harbor_version == "0.23.0"
    assert receipt.external.model == "recorded-model"
    assert "task_success" not in receipt.to_json()
    assert "overall_score" not in receipt.to_json()


def test_schema_is_frozen_versioned_and_json_serializable(tmp_path):
    receipt = extractor.create_receipt(make_trial(tmp_path / "job"))
    assert isinstance(receipt, ExternalEvidenceReceipt)
    assert receipt.schema_version == 1
    assert json.loads(receipt.to_json()) == receipt.to_dict()
    with pytest.raises(FrozenInstanceError):
        receipt.runtime.end_reason = "changed"
    with pytest.raises(ValueError, match="schema"):
        replace(receipt, schema_version=2)


def test_oracle_is_explicit_separate_and_task_matched(tmp_path):
    trial = make_trial(tmp_path / "job", reward=0.0)
    oracle = make_trial(tmp_path / "gate", name="oracle-a", agent="oracle", reward=1.0)
    assert extractor.create_receipt(trial).oracle is None
    receipt = extractor.create_receipt(trial, oracle_job=oracle.parent)
    assert receipt.external.reward == 0.0
    assert receipt.oracle.reward == 1.0
    assert receipt.oracle.exceptions == 0
    assert receipt.oracle.job == "gate"
    assert receipt.oracle.task_checksum == receipt.external.task_checksum
    assert receipt.runtime.tool_call_count == 20
    assert "tool_call_count" not in receipt.to_dict()["oracle"]


def test_unhealthy_oracle_preserves_health_facts_without_changing_agent(tmp_path):
    trial = make_trial(tmp_path / "job", reward=1.0)
    oracle = make_trial(tmp_path / "gate", agent="oracle", reward=0.0, exception="OracleError")
    receipt = extractor.create_receipt(trial, oracle_job=oracle)
    assert receipt.oracle.exceptions == 1
    assert receipt.oracle.reward == 0.0
    assert receipt.external.reward == 1.0


@pytest.mark.parametrize("field,value", [("source", "other"), ("task_name", "other"), ("task_checksum", "different"), ("task_id", {"ref": "different"})])
def test_mismatched_oracle_identity_is_rejected(tmp_path, field, value):
    trial = make_trial(tmp_path / "job")
    oracle = make_trial(tmp_path / "gate", agent="oracle")
    path = oracle / "result.json"
    data = json.loads(path.read_text()); data[field] = value; write_json(path, data)
    with pytest.raises(extractor.ReceiptError, match="does not match"):
        extractor.create_receipt(trial, oracle_job=oracle)


@pytest.mark.parametrize("relative", ["result.json", "agent/pureharness-run-record.json", "agent/pureharness-events.jsonl"])
def test_missing_required_artifacts_fail_clearly(tmp_path, relative):
    trial = make_trial(tmp_path / "job")
    (trial / relative).unlink()
    with pytest.raises(extractor.ReceiptError, match="Missing|Expected exactly one"):
        extractor.create_receipt(trial)


def test_ambiguous_trials_require_explicit_selection(tmp_path):
    job = tmp_path / "job"
    a = make_trial(job)
    make_trial(job, name="trial-b")
    with pytest.raises(extractor.ReceiptError, match="found 2"):
        extractor.create_receipt(job)
    assert extractor.create_receipt(job, trial=a.name).external.trial_name == a.name
    assert extractor.create_receipt(a).external.trial_name == a.name
    with pytest.raises(extractor.ReceiptError, match="found 0"):
        extractor.create_receipt(job, trial="missing")


def test_ambiguous_oracle_trials_require_selection(tmp_path):
    trial = make_trial(tmp_path / "job")
    a = make_trial(tmp_path / "gate", agent="oracle")
    make_trial(a.parent, name="oracle-b", agent="oracle")
    with pytest.raises(extractor.ReceiptError, match="found 2"):
        extractor.create_receipt(trial, oracle_job=a.parent)
    assert extractor.create_receipt(trial, oracle_job=a.parent, oracle_trial=a.name).oracle.trial_name == a.name


def test_deterministic_job_trial_and_cli_output_without_source_mutation(tmp_path, capsys):
    trial = make_trial(tmp_path / "job")
    before = tree_bytes(trial.parent)
    receipt = extractor.create_receipt(trial)
    assert receipt.to_json() == extractor.create_receipt(trial.parent).to_json()
    assert receipt.to_json() == extractor.create_receipt(trial).to_json()
    assert extractor.main([str(trial.parent)]) == 0
    printed = capsys.readouterr()
    assert printed.err == ""
    assert printed.out == receipt.to_json()
    output = tmp_path / "receipt.json"
    assert extractor.main([str(trial), "--output", str(output)]) == 0
    assert output.read_text() == printed.out
    assert tree_bytes(trial.parent) == before
    assert len(receipt.sources) == 5


def test_no_raw_history_secrets_config_or_unknown_event_leaks(tmp_path):
    trial = make_trial(tmp_path / "job", exception="RecordedError", advisory=True)
    events = read_events(trial)
    events.insert(0, {"schema_version": 1, "run_id": "fixture-run", "event": "future_additive_event", "payload": {"API_KEY": SECRET}})
    write_events(trial, events)
    serialized = extractor.create_receipt(trial).to_json()
    for forbidden in [SECRET, "API_KEY", "exception_message", "exception_traceback", "private_raw_output", "wrong-model", '"stdout"', '"argv"']:
        assert forbidden not in serialized


def test_stagnation_advisory_and_progress_gap_reuse(tmp_path):
    trial = make_trial(tmp_path / "job", count=40, end_reason="max_steps_exceeded", advisory=True)
    receipt = extractor.create_receipt(trial)
    assert receipt.progress.unique_tool_actions == 1
    assert receipt.progress.repeated_tool_actions == 39
    assert receipt.progress.max_identical_tool_action_count == 40
    assert receipt.stagnation_advisory.advisory_count == 1
    assert receipt.stagnation_advisory.first_detected_step == 16
    assert receipt.stagnation_advisory.first_delivered_step == 17
    strict = receipt.offline_evaluation.strict_stagnation
    assert strict.detected_step_count == 24
    assert [(i.first_detected_step, i.last_detected_step) for i in strict.intervals] == [(16, 39)]
    gap = receipt.offline_evaluation.progress_gap
    assert gap.longest_active_span.active_steps == 40
    assert gap.final.new_action_count_since_anchor == 1
    assert gap.final.unchanged_repeat_count_since_anchor == 39
    assert gap.anchor_steps == ()


def test_progress_gap_preserves_anchor_reset_without_task_health_inference(tmp_path):
    trial = make_trial(tmp_path / "job", count=40, end_reason="max_steps_exceeded", mutation_at=20, verification_at=24, reward=0.0)
    receipt = extractor.create_receipt(trial)
    gap = receipt.offline_evaluation.progress_gap
    assert gap.anchor_steps == (20, 24)
    assert gap.longest_active_span.active_steps == 20
    assert gap.final.active_steps == 15
    assert gap.final.steps_since_last_structured_mutation == 19
    assert gap.final.steps_since_last_verification_attempt == 15
    assert receipt.external.reward == 0.0


def test_incomplete_coding_evidence_is_unavailable_not_confirmed_zero(tmp_path):
    trial = make_trial(tmp_path / "job")
    events = [e for e in read_events(trial) if not (e["event"] == "coding_evidence_snapshot" and e["step"] == 3)]
    write_events(trial, events)
    receipt = extractor.create_receipt(trial)
    assert receipt.coding.workspace_mutations == 0
    assert receipt.offline_evaluation is None
    assert any("incomplete" in name for name in receipt.unavailable_evidence)


def test_missing_optional_facts_are_null_without_current_version_lookup(tmp_path):
    trial = make_trial(tmp_path / "job")
    (trial.parent / "lock.json").unlink()
    events = [e for e in read_events(trial) if e["event"] not in ("coding_evidence_snapshot", "progress_snapshot", "context_built")]
    write_events(trial, events)
    receipt = extractor.create_receipt(trial)
    assert receipt.external.harbor_version is None
    assert receipt.progress is receipt.coding is receipt.offline_evaluation is None
    assert receipt.context.maximum_history_tokens is None
    assert receipt.stagnation_advisory.advisory_count == 0


@pytest.mark.parametrize("corruption", ["mixed_run", "schema", "terminal_missing", "end_reason", "coding_regression", "progress_mismatch", "invalid_json"])
def test_inconsistent_or_corrupt_events_are_rejected_without_raw_errors(tmp_path, corruption):
    trial = make_trial(tmp_path / "job", mutation_at=1)
    events = read_events(trial)
    if corruption == "mixed_run": events[0]["run_id"] = "other"
    elif corruption == "schema": events[0]["schema_version"] = 2
    elif corruption == "terminal_missing": events = events[:-1]
    elif corruption == "end_reason": events[-1]["payload"]["end_reason"] = "model_error"
    elif corruption == "coding_regression":
        next(e for e in events if e["event"] == "coding_evidence_snapshot" and e["step"] == 3)["payload"]["workspace_mutations"] = 0
    elif corruption == "progress_mismatch": events[-2]["payload"]["tool_calls"] = 999
    write_events(trial, events)
    if corruption == "invalid_json": events_path(trial).write_text("not JSON " + SECRET)
    with pytest.raises(extractor.ReceiptError) as caught:
        extractor.create_receipt(trial)
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("destination", ["job", "trial", "oracle"])
def test_cli_refuses_output_within_source_trees(tmp_path, capsys, destination):
    trial = make_trial(tmp_path / "job")
    oracle = make_trial(tmp_path / "gate", agent="oracle")
    root = {"job": trial.parent, "trial": trial, "oracle": oracle.parent}[destination]
    output = root / "receipt.json"
    assert extractor.main([str(trial), "--oracle-job", str(oracle.parent), "--output", str(output)]) == 1
    assert "outside" in capsys.readouterr().err
    assert not output.exists()


def test_cli_refuses_to_overwrite_source_hardlink_outside_job(tmp_path, capsys):
    trial = make_trial(tmp_path / "job")
    original = trial / "result.json"
    output = tmp_path / "receipt.json"
    output.hardlink_to(original)
    before = original.read_bytes()
    assert extractor.main([str(trial), "--output", str(output)]) == 1
    assert capsys.readouterr().err
    assert original.read_bytes() == before


def test_artifact_symlink_escape_is_rejected(tmp_path):
    trial = make_trial(tmp_path / "job")
    record = trial / "agent/pureharness-run-record.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(record.read_bytes()); record.unlink(); record.symlink_to(outside)
    with pytest.raises(extractor.ReceiptError, match="escapes"):
        extractor.create_receipt(trial)
