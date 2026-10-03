"""Versioned external evidence artifact, independent of runtime persistence.

These immutable, explicit fields contain no raw history, environment, prompts,
or provider errors. Extraction belongs to the offline script, not Agent. Reward
and exceptions are evaluator facts; protocol completion is a runtime fact.
Neither stagnation nor progress-gap evidence defines task success.
"""

import json
import math

from dataclasses import asdict, dataclass, fields, is_dataclass
from types import UnionType
from typing import get_args, get_origin, get_type_hints

from pureharness.evaluation.progress_gap import ProgressGapEvidence
from pureharness.evaluation.stagnation import StagnationEvidence


EXTERNAL_EVIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExternalEvaluatorFacts:
    dataset: str | None
    task: str
    task_ref: str | None
    task_checksum: str | None
    harbor_version: str | None
    model: str | None
    model_provider: str | None
    pureharness_revision: str
    reward: float | None
    exception_type: str | None
    job_id: str | None
    job_name: str
    trial_id: str
    trial_name: str


@dataclass(frozen=True)
class RuntimeFacts:
    run_id: str
    end_reason: str
    protocol_completed: bool
    step_count: int
    model_call_count: int
    tool_call_count: int
    tool_execution_count: int
    tool_result_error_count: int
    trajectory_compaction_count: int
    tool_result_compaction_count: int

    def __post_init__(self) -> None:
        if type(self.protocol_completed) is not bool or self.protocol_completed != (self.end_reason == "completed"):
            raise ValueError("protocol_completed must describe runtime completion only")


@dataclass(frozen=True)
class ProgressFacts:
    unique_tool_actions: int | None
    repeated_tool_actions: int | None
    max_identical_tool_action_count: int | None


@dataclass(frozen=True)
class CodingFacts:
    workspace_mutations: int | None
    verification_attempts: int | None
    verification_exit_zero: int | None
    verification_exit_nonzero: int | None
    verification_tool_errors: int | None


@dataclass(frozen=True)
class ContextFacts:
    context_build_count: int
    maximum_history_tokens: int | None
    final_history_tokens: int | None
    configured_history_token_budget: int | None
    compacted_context_count: int | None


@dataclass(frozen=True)
class AdvisoryFacts:
    advisory_count: int
    first_detected_step: int | None
    first_delivered_step: int | None


@dataclass(frozen=True)
class DetectedInterval:
    first_detected_step: int
    last_detected_step: int
    first_window: StagnationEvidence
    last_window: StagnationEvidence


@dataclass(frozen=True)
class StrictStagnationSummary:
    window_size: int
    detected_step_count: int
    intervals: tuple[DetectedInterval, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervals", tuple(self.intervals))


@dataclass(frozen=True)
class ProgressGapSummary:
    anchor_steps: tuple[int, ...]
    longest_active_span: ProgressGapEvidence
    final: ProgressGapEvidence

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_steps", tuple(self.anchor_steps))


@dataclass(frozen=True)
class OfflineEvaluationEvidence:
    strict_stagnation: StrictStagnationSummary
    progress_gap: ProgressGapSummary


@dataclass(frozen=True)
class OracleHealthFacts:
    """Separate selected Oracle trial; exceptions is its 0/1 occurrence count."""

    reward: float | None
    exceptions: int
    exception_type: str | None
    job_id: str | None
    job: str
    trial_id: str
    trial_name: str
    dataset: str | None
    task: str
    task_ref: str | None
    task_checksum: str | None


@dataclass(frozen=True)
class SourceDigest:
    artifact: str
    sha256: str


@dataclass(frozen=True)
class ExternalEvidenceReceipt:
    external: ExternalEvaluatorFacts
    runtime: RuntimeFacts
    progress: ProgressFacts | None
    coding: CodingFacts | None
    context: ContextFacts
    stagnation_advisory: AdvisoryFacts
    offline_evaluation: OfflineEvaluationEvidence | None
    oracle: OracleHealthFacts | None = None
    unavailable_evidence: tuple[str, ...] = ()
    sources: tuple[SourceDigest, ...] = ()
    schema_version: int = EXTERNAL_EVIDENCE_SCHEMA_VERSION
    receipt_type: str = "pureharness.external_evidence"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported external evidence receipt schema")
        if self.receipt_type != "pureharness.external_evidence":
            raise ValueError("Unsupported receipt type")
        for name, expected, optional in (
            ("external", ExternalEvaluatorFacts, False),
            ("runtime", RuntimeFacts, False),
            ("progress", ProgressFacts, True),
            ("coding", CodingFacts, True),
            ("context", ContextFacts, False),
            ("stagnation_advisory", AdvisoryFacts, False),
            ("offline_evaluation", OfflineEvaluationEvidence, True),
            ("oracle", OracleHealthFacts, True),
        ):
            value = getattr(self, name)
            if not (optional and value is None) and not isinstance(value, expected):
                raise ValueError(f"Invalid receipt section: {name}")
        object.__setattr__(self, "unavailable_evidence", tuple(self.unavailable_evidence))
        object.__setattr__(self, "sources", tuple(self.sources))
        if not all(isinstance(item, SourceDigest) for item in self.sources):
            raise ValueError("sources must contain SourceDigest values")

    def to_dict(self) -> dict[str, object]:
        # Round-trip makes immutable tuples JSON arrays, including nested
        # interval/source lists. No runtime models or raw history are included.
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, data: object) -> "ExternalEvidenceReceipt":
        """Read the explicit v1 shape without defaulting absent facts to zero.

        Field names and types come from these same dataclasses, including the
        existing offline evidence types. Null is valid only for nullable fields.
        Unknown fields are rejected rather than silently discarding evidence.
        This reader never reconstructs or accesses the original job artifacts.
        """
        return _decode_receipt_value(cls, data, "receipt")


def _decode_receipt_value(expected: object, value: object, path: str):
    """Narrow JSON decoder for the receipt's dataclasses and scalar types."""
    if get_origin(expected) is UnionType:
        choices = get_args(expected)
        if value is None and type(None) in choices:
            return None
        expected = next(choice for choice in choices if choice is not type(None))
    if get_origin(expected) is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected an array")
        item_type, _ = get_args(expected)
        return tuple(
            _decode_receipt_value(item_type, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if is_dataclass(expected):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected an object")
        names = {field.name for field in fields(expected)}
        missing = sorted(names - value.keys())
        if missing:
            raise ValueError(f"{path}.{missing[0]}: required field missing")
        if value.keys() - names:
            raise ValueError(f"{path}: unknown fields are not supported")
        hints = get_type_hints(expected)
        decoded = {
            field.name: _decode_receipt_value(
                hints[field.name], value[field.name], f"{path}.{field.name}",
            )
            for field in fields(expected)
        }
        try:
            return expected(**decoded)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if expected is str and isinstance(value, str) and value.strip():
        return value
    if expected is bool and type(value) is bool:
        return value
    if expected is int and type(value) is int and value >= 0:
        return value
    if expected is float and type(value) in (int, float):
        try:
            result = float(value)
        except OverflowError:
            result = math.inf
        if math.isfinite(result):
            return result
    labels = {
        str: "non-empty text", bool: "bool", int: "non-negative integer",
        float: "finite number",
    }
    raise ValueError(f"{path}: expected {labels[expected]}")
