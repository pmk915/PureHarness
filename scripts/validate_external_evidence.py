#!/usr/bin/env python3
"""Validate an offline receipt pack, not benchmark correctness or authenticity."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

from pureharness.evaluation.external_evidence import ExternalEvidenceReceipt


DEFAULT_PACK = Path(__file__).resolve().parents[1] / "benchmarks/external"
INDEX_START = "<!-- external-evidence-index:start -->"
INDEX_END = "<!-- external-evidence-index:end -->"


@dataclass(frozen=True)
class PackValidationResult:
    receipt_count: int
    errors: tuple[str, ...]


def _receipt_errors(receipt: ExternalEvidenceReceipt) -> list[str]:
    """Relations among recorded facts only; no task-success implications."""
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    def bound(value, maximum, name):
        if value is not None and maximum is not None:
            check(value <= maximum, f"{name}: exceeds recorded bound")

    r = receipt.runtime
    check(bool(re.fullmatch(r"[0-9a-fA-F]{40}", receipt.external.pureharness_revision)),
          "external.pureharness_revision: expected full Git SHA")
    bound(r.tool_execution_count, r.tool_call_count, "runtime.tool_execution_count")
    # Policy errors can produce results without executions; calls are the bound.
    bound(r.tool_result_error_count, r.tool_call_count, "runtime.tool_result_error_count")
    bound(r.step_count, r.model_call_count, "runtime.step_count")
    bound(r.trajectory_compaction_count, r.model_call_count, "runtime.trajectory_compaction_count")
    if receipt.progress is not None:
        p = receipt.progress
        for key in ("unique_tool_actions", "repeated_tool_actions",
                    "max_identical_tool_action_count"):
            bound(getattr(p, key), r.tool_call_count, f"progress.{key}")
        if p.unique_tool_actions is not None and p.repeated_tool_actions is not None:
            bound(p.unique_tool_actions + p.repeated_tool_actions, r.tool_call_count,
                  "progress.unique_tool_actions + repeated_tool_actions")
    if receipt.coding is not None:
        c = receipt.coding
        outcomes = (c.verification_exit_zero, c.verification_exit_nonzero,
                    c.verification_tool_errors)
        bound(sum(value for value in outcomes if value is not None), c.verification_attempts,
              "coding.verification outcomes")
    c = receipt.context
    bound(c.final_history_tokens, c.maximum_history_tokens, "context.final_history_tokens")
    bound(c.maximum_history_tokens, c.configured_history_token_budget, "context.maximum_history_tokens")
    bound(c.final_history_tokens, c.configured_history_token_budget, "context.final_history_tokens budget")
    bound(c.compacted_context_count, c.context_build_count, "context.compacted_context_count")
    a = receipt.stagnation_advisory
    steps = (a.first_detected_step, a.first_delivered_step)
    if a.advisory_count == 0:
        check(steps == (None, None), "stagnation_advisory: zero count contradicts step evidence")
    elif None in steps:
        check(False, "stagnation_advisory: positive count requires detected/delivered steps")
    else:
        check(a.first_delivered_step >= a.first_detected_step,
              "stagnation_advisory: delivery precedes detection")
    if receipt.oracle is not None:
        o = receipt.oracle
        check(o.exceptions == int(o.exception_type is not None),
              "oracle.exceptions: must describe the selected trial's exception occurrence")
        for key in ("dataset", "task", "task_ref", "task_checksum"):
            left, right = getattr(receipt.external, key), getattr(o, key)
            if left is not None and right is not None:
                check(left == right, f"oracle.{key}: disagrees with agent task identity")
    labels = [source.artifact for source in receipt.sources]
    check(len(labels) == len(set(labels)), "sources: duplicate artifact labels")
    for source in receipt.sources:
        check(bool(re.fullmatch(r"[0-9a-fA-F]{64}", source.sha256)),
              "sources.sha256: expected SHA-256 digest")
    if receipt.offline_evaluation is not None:
        strict = receipt.offline_evaluation.strict_stagnation
        check(strict.window_size >= 2, "offline.strict_stagnation.window_size: must be at least 2")
        total, last = 0, -2
        for index, interval in enumerate(strict.intervals):
            name = f"offline.strict_stagnation.intervals[{index}]"
            first, end = interval.first_detected_step, interval.last_detected_step
            check(last + 1 < first <= end < r.step_count,
                  f"{name}: intervals must be ordered, disjoint, nonadjacent and within steps")
            total += end - first + 1
            last = end
            for step, window in ((first, interval.first_window), (end, interval.last_window)):
                check(window.end_step == step and window.window_size == strict.window_size,
                      f"{name}: window identity disagrees with interval")
                check(window.active_steps <= window.observed_steps <= window.window_size,
                      f"{name}: incompatible window step counts")
                check(window.start_step is not None and window.end_step is not None
                      and window.end_step - window.start_step + 1 == window.observed_steps,
                      f"{name}: window span disagrees with observed steps")
                check(window.new_action_count + window.repeated_action_count + window.unknown_action_count
                      == window.tool_action_count, f"{name}: action counts disagree")
                check(window.unchanged_result_repeat_count + window.changed_result_repeat_count
                      == window.repeated_action_count, f"{name}: repeated result counts disagree")
                ratio = window.repeated_action_count / window.tool_action_count if window.tool_action_count else 0.0
                check(0.0 <= window.repeat_ratio <= 1.0 and math.isclose(window.repeat_ratio, ratio),
                      f"{name}: repeat_ratio disagrees with counts")
                bound(window.unique_action_count, window.tool_action_count, f"{name}.unique_action_count")
                bound(window.tool_error_delta, window.tool_action_count, f"{name}.tool_error_delta")
        check(total == strict.detected_step_count,
              "offline.strict_stagnation.detected_step_count: disagrees with intervals")
        gap = receipt.offline_evaluation.progress_gap
        check(tuple(sorted(set(gap.anchor_steps))) == gap.anchor_steps,
              "offline.progress_gap.anchor_steps: must be ordered and unique")
        for step in gap.anchor_steps:
            check(step < r.step_count, "offline.progress_gap.anchor_steps: outside recorded steps")
        check(gap.longest_active_span.active_steps >= gap.final.active_steps,
              "offline.progress_gap: longest span is smaller than final span")
        for evidence in (gap.longest_active_span, gap.final):
            if evidence.step is not None:
                check(evidence.step < r.step_count, "offline.progress_gap: evidence outside recorded steps")
    return errors


def _cross_errors(receipts: list[tuple[str, ExternalEvidenceReceipt]]) -> list[str]:
    errors = []
    trials, locations, runs, oracles, tasks = {}, {}, {}, {}, {}

    def unique(table, key, path, label):
        if key in table:
            errors.append(f"{path}: duplicate {label}; also in {table[key]}")
        else:
            table[key] = path

    def compatible(table, key, facts, path, label):
        known = table.setdefault(key, {})
        for field, value in facts.items():
            if value is None:
                continue
            if field in known and known[field][0] != value:
                errors.append(f"{path}: conflicting {label}.{field}; also in {known[field][1]}")
            else:
                known.setdefault(field, (value, path))

    for path, receipt in receipts:
        e = receipt.external
        unique(trials, e.trial_id, path, "source trial_id")
        unique(locations, (e.job_name, e.trial_name), path, "source job/trial")
        unique(runs, receipt.runtime.run_id, path, "output run_id")
        suffix = re.search(r"_([0-9a-fA-F]{7,40})\.json$", path)
        if suffix and not e.pureharness_revision.lower().startswith(suffix[1].lower()):
            errors.append(f"{path}: filename SHA disagrees with external.pureharness_revision")
        if receipt.oracle is not None:
            o = receipt.oracle
            facts = {field.name: getattr(o, field.name) for field in fields(o)}
            compatible(oracles, ("trial_id", o.trial_id), facts, path, "shared Oracle")
            compatible(oracles, ("location", o.job, o.trial_name), facts, path, "shared Oracle")
            if o.job_id is not None:
                compatible(oracles, ("job_id", o.job_id, o.trial_name), facts, path, "shared Oracle")
        for identity in (e, receipt.oracle):
            if identity is None:
                continue
            facts = {key: getattr(identity, key) for key in ("dataset", "task")}
            for key in ("task_ref", "task_checksum"):
                value = getattr(identity, key)
                if value is not None:
                    compatible(tasks, (key, value), facts, path, "shared task identity")
    return errors


def _index_errors(root: Path, paths: set[str]) -> list[str]:
    readme = root / "README.md"
    if not readme.resolve().is_relative_to(root.resolve()):
        return ["README.md: path escapes evidence pack"]
    try:
        text = readme.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ["README.md: missing or unreadable"]
    if text.count(INDEX_START) != 1 or text.count(INDEX_END) != 1:
        return ["README.md: exactly one receipt-index start/end marker required"]
    start, end = text.index(INDEX_START), text.index(INDEX_END)
    if start >= end:
        return ["README.md: receipt-index markers are reversed"]
    section = text[start + len(INDEX_START):end]
    errors, indexed = [], []
    for number, row in enumerate(section.splitlines(), 1):
        if not row.strip():
            continue
        if not row.lstrip().startswith("|"):
            errors.append(f"README.md: index line {number}: expected a table row")
            continue
        links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", row)
        if not links:
            # Only the documented header and Markdown separator lack receipts.
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if cells[-1] != "Receipt" and not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                errors.append(f"README.md: index line {number}: missing receipt link")
            continue
        if len(links) != 1:
            errors.append(f"README.md: index line {number}: expected one receipt link")
        for link in links:
            path = PurePosixPath(link)
            if (path.is_absolute() or ".." in path.parts or path.as_posix() != link
                    or not link.startswith("receipts/") or path.suffix != ".json"):
                errors.append(f"README.md: index line {number}: invalid receipt path")
                continue
            indexed.append(link)
    counts = Counter(indexed)
    for path, count in sorted(counts.items()):
        if count > 1:
            errors.append(f"README.md: duplicate receipt index row: {path}")
        if path not in paths:
            errors.append(f"README.md: stale or nonexistent receipt link: {path}")
    for path in sorted(paths - counts.keys()):
        errors.append(f"README.md: receipt missing from index: {path}")
    return errors


def validate_pack(root: Path = DEFAULT_PACK) -> PackValidationResult:
    """Read only the pack: recursive *.json, never Git or source jobs.

    All on-disk JSON candidates are checked, including proposed new receipts;
    .gitkeep and all non-JSON files are ignored. This needs no Git executable.
    """
    root = Path(root)
    directory = root / "receipts"
    if not directory.is_dir() or not directory.resolve().is_relative_to(root.resolve()):
        return PackValidationResult(0, ("receipts/: missing directory or escaped path",))
    try:
        candidates = sorted(directory.rglob("*.json"), key=lambda p: p.relative_to(root).as_posix())
    except OSError:
        return PackValidationResult(0, ("receipts/: cannot enumerate JSON receipts",))
    errors, receipts, paths = [], [], set()
    if not candidates:
        errors.append("receipts/: no JSON receipts found")
    for path in candidates:
        name = path.relative_to(root).as_posix()
        paths.add(name)
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            errors.append(f"{name}: not a regular in-pack receipt file")
            continue
        try:
            receipt = ExternalEvidenceReceipt.from_dict(json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
            ))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            errors.append(f"{name}: unreadable or invalid JSON")
            continue
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        receipts.append((name, receipt))
        errors.extend(f"{name}: {message}" for message in _receipt_errors(receipt))
    errors.extend(_cross_errors(receipts))
    errors.extend(_index_errors(root, paths))
    return PackValidationResult(len(candidates), tuple(sorted(set(errors))))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object: duplicate field names are ambiguous")
        result[key] = value
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", nargs="?", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args(argv)
    result = validate_pack(args.pack)
    if result.errors:
        for error in result.errors:
            print(error, file=sys.stderr)
        print(f"External evidence pack invalid: {len(result.errors)} error(s)", file=sys.stderr)
        return 1
    print(f"External evidence pack valid: {result.receipt_count} receipt(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
