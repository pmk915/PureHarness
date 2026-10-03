#!/usr/bin/env python3
"""Summarize recorded external evaluation evidence; never execute a trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys

from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.evaluation.external_evidence import (
    AdvisoryFacts, CodingFacts, ContextFacts, DetectedInterval,
    ExternalEvaluatorFacts, ExternalEvidenceReceipt, OfflineEvaluationEvidence,
    OracleHealthFacts, ProgressFacts, ProgressGapSummary, RuntimeFacts,
    SourceDigest, StrictStagnationSummary,
)
from pureharness.evaluation.progress_gap import ProgressGapEvaluator
from pureharness.evaluation.stagnation import (
    StagnationEvaluator, stagnation_steps_from_run_record,
)
from pureharness.run_record import RunRecord
from pureharness.verification import VerificationOutcome


class ReceiptError(ValueError):
    """Required artifacts are missing, ambiguous, or inconsistent."""


def _read(path: Path, root: Path, label: str, sources: list[SourceDigest]) -> bytes:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ReceiptError(f"{label} escapes its source directory")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReceiptError(f"Missing or unreadable {label}") from exc
    sources.append(SourceDigest(label, hashlib.sha256(data).hexdigest()))
    return data


def _object(data: object, label: str) -> dict:
    if not isinstance(data, dict):
        raise ReceiptError(f"{label} must be an object")
    return data


def _load(path: Path, root: Path, label: str, sources: list[SourceDigest]) -> dict:
    try:
        return _object(json.loads(_read(path, root, label, sources)), label)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"Invalid JSON in {label}") from exc


def _text(data: dict, key: str, *, required: bool = False) -> str | None:
    value = data.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{key} must be non-empty text")
    return value


def _count(data: dict, key: str, *, required: bool = False) -> int | None:
    value = data.get(key)
    if value is None and not required:
        return None
    if type(value) is not int or value < 0:
        raise ReceiptError(f"{key} must be a non-negative integer")
    return value


def _reward(result: dict) -> float | None:
    verifier = result.get("verifier_result")
    if verifier is None:
        return None
    rewards = _object(verifier, "verifier_result").get("rewards")
    if rewards is None:
        return None
    value = _object(rewards, "rewards").get("reward")
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ReceiptError("reward must be finite numeric data or null")
    return float(value)


def _exception(result: dict) -> str | None:
    info = result.get("exception_info")
    return None if info is None else _text(
        _object(info, "exception_info"), "exception_type", required=True,
    )


def _resolve_trial(
    directory: Path, agent: str, selection: str | None,
    sources: list[SourceDigest], prefix: str,
) -> tuple[Path, Path, dict, dict]:
    directory = directory.resolve()
    result = _load(directory / "result.json", directory, "source result.json", [])
    if "task_name" in result:
        trial = directory
        job = trial.parent if (trial.parent / "result.json").is_file() else trial
        job_result = (
            _load(job / "result.json", job, f"{prefix}/job-result.json", sources)
            if job != trial else {}
        )
        _read(trial / "result.json", trial, f"{prefix}/trial-result.json", sources)
        if selection is not None and selection != trial.name:
            raise ReceiptError("Explicit trial does not match the trial directory")
    else:
        job, job_result = directory, result
        _read(job / "result.json", job, f"{prefix}/job-result.json", sources)
        candidates = []
        for path in sorted(job.glob("*/result.json")):
            if selection is not None and path.parent.name != selection:
                continue
            # Discovery reads metadata only; only selected artifacts enter hashes.
            temporary: list[SourceDigest] = []
            value = _load(path, job, "trial discovery result", temporary)
            info = _object(value.get("agent_info"), "agent_info")
            if info.get("name") == agent:
                candidates.append((path.parent, value))
        if len(candidates) != 1:
            raise ReceiptError(
                f"Expected exactly one {agent} trial; found {len(candidates)}. "
                "Use an explicit trial directory or --trial/--oracle-trial."
            )
        trial, result = candidates[0]
        _read(trial / "result.json", job, f"{prefix}/trial-result.json", sources)
    if _object(result.get("agent_info"), "agent_info").get("name") != agent:
        raise ReceiptError(f"Selected trial is not a {agent} trial")
    if _text(result, "trial_name", required=True) != trial.name:
        raise ReceiptError("trial_name does not match its directory")
    return trial, job, result, job_result


def _artifact(trial: Path, filename: str) -> Path:
    paths = sorted({
        path for name in ("agent", "artifacts", "logs")
        for path in (trial / name).rglob(filename) if path.is_file()
    })
    if len(paths) != 1:
        raise ReceiptError(f"Expected exactly one {filename}; found {len(paths)}")
    if not paths[0].resolve().is_relative_to(trial.resolve()):
        raise ReceiptError(f"{filename} escapes trial directory")
    return paths[0]


def _identity(result: dict) -> tuple[str | None, str, str | None, str | None]:
    task_id = _object(result.get("task_id", {}), "task_id")
    return (
        _text(result, "source"), _text(result, "task_name", required=True),
        _text(task_id, "ref"), _text(result, "task_checksum"),
    )


def _events(path: Path, trial: Path, record: RunRecord, sources: list[SourceDigest]) -> list[dict]:
    raw = _read(path, trial, "agent/events.jsonl", sources)
    try:
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReceiptError("Invalid JSONL event artifact") from exc
    for event in events:
        event = _object(event, "event")
        if type(event.get("schema_version")) is not int or event["schema_version"] != 1:
            raise ReceiptError("Unsupported JSONL wire schema")
        if event.get("run_id") != record.run_id:
            raise ReceiptError("JSONL run_id does not match RunRecord")
        _text(event, "event", required=True)
        _object(event.get("payload"), "event payload")
        if "step" in event:
            _count(event, "step", required=True)
    terminals = [e for e in events if e["event"] in (
        "agent_completed", "agent_failed", "agent_interrupted",
    )]
    if len(terminals) != 1:
        raise ReceiptError("JSONL must contain exactly one terminal Agent event")
    payload = terminals[0]["payload"]
    if _text(payload, "end_reason", required=True) != record.end_reason or _count(payload, "step_count", required=True) != record.step_count:
        raise ReceiptError("Terminal JSONL facts do not match RunRecord")
    expected = "agent_completed" if record.end_reason == "completed" else (
        "agent_interrupted" if record.end_reason == "interrupted" else "agent_failed"
    )
    if terminals[0]["event"] != expected:
        raise ReceiptError("Terminal event type does not match end_reason")
    return events


def _last(events: list[dict], name: str) -> dict | None:
    matches = [e["payload"] for e in events if e["event"] == name]
    return matches[-1] if matches else None


def _offline(record: RunRecord, events: list[dict]) -> OfflineEvaluationEvidence | None:
    snapshots = {}
    names = {field.name for field in fields(CodingEvidenceSnapshot)}
    trace_indexes = {step.index for step in record.trace.steps}
    for event in events:
        if event["event"] != "coding_evidence_snapshot" or event.get("step") not in trace_indexes:
            continue
        payload = event["payload"]
        if payload.get("workspace_mutations") is None or payload.get("verification_attempts") is None:
            continue
        values = {key: value for key, value in payload.items() if key in names}
        try:
            if values.get("last_verification_outcome") is not None:
                values["last_verification_outcome"] = VerificationOutcome(values["last_verification_outcome"])
            snapshot = CodingEvidenceSnapshot(**values)
        except (TypeError, ValueError) as exc:
            raise ReceiptError("Invalid coding evidence snapshot") from exc
        index = event["step"]
        if index in snapshots and snapshots[index] != snapshot:
            raise ReceiptError("Conflicting post-step coding snapshots")
        snapshots[index] = snapshot
    if snapshots.keys() != trace_indexes:
        return None
    try:
        steps = stagnation_steps_from_run_record(record, coding_evidence=snapshots)
        evaluator = StagnationEvaluator()
        signals = evaluator.evaluate_windows(steps)
        gaps = ProgressGapEvaluator().evaluate_steps(steps)
    except (TypeError, ValueError) as exc:
        raise ReceiptError("Invalid completed trajectory or progress counters") from exc
    intervals = []
    for signal in signals:
        if not signal.detected:
            continue
        index = signal.evidence.end_step
        if intervals and index == intervals[-1].last_detected_step + 1:
            previous = intervals[-1]
            intervals[-1] = DetectedInterval(
                previous.first_detected_step, index, previous.first_window, signal.evidence,
            )
        else:
            intervals.append(DetectedInterval(index, index, signal.evidence, signal.evidence))
    empty = ProgressGapEvaluator().evaluate(())
    return OfflineEvaluationEvidence(
        StrictStagnationSummary(evaluator.window_size, sum(s.detected for s in signals), tuple(intervals)),
        ProgressGapSummary(
            tuple(e.step for e in gaps if e.workspace_mutation_delta or e.verification_delta),
            max(gaps, key=lambda e: (e.active_steps, e.steps_since_anchor), default=empty),
            gaps[-1] if gaps else empty,
        ),
    )


def create_receipt(
    directory: Path, *, trial: str | None = None,
    oracle_job: Path | None = None, oracle_trial: str | None = None,
) -> ExternalEvidenceReceipt:
    sources: list[SourceDigest] = []
    selected, job, result, job_result = _resolve_trial(directory, "pureharness", trial, sources, "agent")
    record_path = _artifact(selected, "pureharness-run-record.json")
    try:
        record = RunRecord.from_dict(_load(record_path, selected, "agent/run-record.json", sources))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ReceiptError):
            raise
        raise ReceiptError("Invalid PureHarness RunRecord") from exc
    events = _events(_artifact(selected, "pureharness-events.jsonl"), selected, record, sources)
    info = result["agent_info"]
    revision = _text(info, "version", required=True)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ReceiptError("PureHarness revision must be a recorded full Git SHA")
    model = _object(info["model_info"], "model_info") if info.get("model_info") is not None else {}
    version = None
    if (job / "lock.json").is_file():
        lock = _load(job / "lock.json", job, "agent/job-lock.json", sources)
        harbor = _object(lock["harbor"], "harbor") if lock.get("harbor") is not None else {}
        version = _text(harbor, "version")
    dataset, task, task_ref, checksum = _identity(result)
    external = ExternalEvaluatorFacts(
        dataset, task, task_ref, checksum, version,
        _text(model, "name"), _text(model, "provider"), revision.lower(),
        _reward(result), _exception(result),
        _text(job_result, "id"), job.name,
        _text(result, "id", required=True), result["trial_name"],
    )
    unavailable = []
    progress = _last(events, "progress_snapshot")
    if progress is not None:
        for key, expected in (("tool_calls", record.tool_call_count),
                              ("logical_steps_completed", record.step_count),
                              ("failed_tool_results", record.tool_result_error_count)):
            value = _count(progress, key)
            if value is not None and value != expected:
                raise ReceiptError("Final progress snapshot disagrees with RunRecord")
        progress = ProgressFacts(*[_count(progress, field.name) for field in fields(ProgressFacts)])
        known_counts = (progress.unique_tool_actions, progress.repeated_tool_actions)
        if (
            all(value is not None for value in known_counts)
            and sum(known_counts) > record.tool_call_count
        ) or (
            progress.max_identical_tool_action_count is not None
            and progress.max_identical_tool_action_count > record.tool_call_count
        ):
            raise ReceiptError("Progress action counts exceed recorded tool calls")
    else:
        unavailable.append("progress_snapshot")
    coding = _last(events, "coding_evidence_snapshot")
    if coding is not None:
        coding = CodingFacts(*[_count(coding, field.name) for field in fields(CodingFacts)])
    else:
        unavailable.append("coding_evidence_snapshot")
    contexts = [e["payload"] for e in events if e["event"] == "context_built"]
    history = [_count(c, "estimated_history_tokens", required=True) for c in contexts]
    budgets = [_count(c, "history_token_budget") for c in contexts]
    compacted = [c.get("trajectory_compacted") for c in contexts]
    if any(value is not None and type(value) is not bool for value in compacted):
        raise ReceiptError("trajectory_compacted must be bool")
    if not contexts:
        unavailable.append("context_built")
    context = ContextFacts(
        len(contexts), max(history, default=None), history[-1] if history else None,
        budgets[0] if budgets and len(set(budgets)) == 1 else None,
        sum(compacted) if contexts and all(v is not None for v in compacted) else None,
    )
    advisories = [e for e in events if e["event"] == "runtime_advisory_emitted" and e["payload"].get("kind") == "stagnation"]
    for index, event in enumerate(advisories, 1):
        payload = event["payload"]
        detected = _count(payload, "detected_at_step", required=True)
        delivered = _count(payload, "delivered_at_step", required=True)
        if _count(payload, "advisory_index", required=True) != index or delivered != detected + 1 or delivered != event.get("step"):
            raise ReceiptError("Inconsistent stagnation advisory delivery evidence")
    advisory = AdvisoryFacts(
        len(advisories), advisories[0]["payload"]["detected_at_step"] if advisories else None,
        advisories[0]["payload"]["delivered_at_step"] if advisories else None,
    )
    offline = _offline(record, events)
    if offline is None:
        unavailable.append("offline_evaluation: incomplete post-step coding evidence")
    oracle = None
    if oracle_trial is not None and oracle_job is None:
        raise ReceiptError("--oracle-trial requires --oracle-job")
    if oracle_job is not None:
        _, oracle_root, gate, gate_job = _resolve_trial(oracle_job, "oracle", oracle_trial, sources, "oracle")
        identity = _identity(gate)
        for left, right in zip((dataset, task, task_ref, checksum), identity, strict=True):
            if left is not None and right is not None and left != right:
                raise ReceiptError("Oracle task/dataset/ref/checksum does not match agent trial")
        exception = _exception(gate)
        oracle = OracleHealthFacts(
            _reward(gate), int(exception is not None), exception,
            _text(gate_job, "id"), oracle_root.name, _text(gate, "id", required=True),
            gate["trial_name"], *identity,
        )
    return ExternalEvidenceReceipt(
        external, RuntimeFacts(
            record.run_id, record.end_reason, record.end_reason == "completed",
            record.step_count, record.model_call_count, record.tool_call_count,
            record.tool_execution_count, record.tool_result_error_count,
            record.trajectory_compaction_count, record.tool_result_compaction_count,
        ), progress, coding, context, advisory, offline, oracle,
        tuple(unavailable), tuple(sorted(sources, key=lambda s: s.artifact)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harbor_path", type=Path)
    parser.add_argument("--trial")
    parser.add_argument("--oracle-job", type=Path)
    parser.add_argument("--oracle-trial")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        # Protect the parent job even when a trial directory was supplied.
        roots = []
        for source in (args.harbor_path, args.oracle_job):
            if source is None:
                continue
            root = source.resolve()
            if (root / "result.json").is_file() and (root.parent / "result.json").is_file():
                root = root.parent
            roots.append(root)
        if args.output is not None and any(args.output.resolve().is_relative_to(root) for root in roots):
            raise ReceiptError("--output must be outside agent and Oracle source directories")
        receipt = create_receipt(
            args.harbor_path, trial=args.trial,
            oracle_job=args.oracle_job, oracle_trial=args.oracle_trial,
        ).to_json()
        if args.output is None:
            sys.stdout.write(receipt)
        else:
            # Exclusive creation also prevents overwriting a hard link to a
            # source artifact outside the protected directory tree.
            with args.output.open("x", encoding="utf-8") as output:
                output.write(receipt)
    except (OSError, ReceiptError) as exc:
        print(f"external_evidence_receipt: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
