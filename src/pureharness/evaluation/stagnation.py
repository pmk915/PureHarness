"""Observational stagnation evidence, never an execution or correctness policy.

Activity is not progress. Compare exact actions and byte-identical observations
separately from caller-supplied mutation/verification facts; do not interpret
shell commands or infer task intent. All state below is local to evaluation.
"""

import hashlib

from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.messages import Message
from pureharness.progress import tool_action_fingerprint
from pureharness.tool_history import ToolInteraction, match_tool_interactions

if TYPE_CHECKING:
    from pureharness.run_record import RunRecord


@dataclass(frozen=True)
class StagnationObservation:
    """Exact action/result fingerprints, with no raw arguments or outputs."""

    action_identity: str | None
    result_identity: str
    is_error: bool

    def __post_init__(self) -> None:
        for name in ("action_identity", "result_identity"):
            value = getattr(self, name)
            if name == "action_identity" and value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        if not isinstance(self.is_error, bool):
            raise ValueError("is_error must be bool")

    @classmethod
    def from_interaction(
        cls, interaction: ToolInteraction,
    ) -> "StagnationObservation":
        """Use ProgressTracker's action identity and exact result content.

        Error state participates in result identity; call IDs do not. No
        whitespace, path, shell, or semantic normalization is performed.
        """
        if not isinstance(interaction, ToolInteraction):
            raise TypeError("interaction must be ToolInteraction")
        result = interaction.result
        if not isinstance(result.content, str):
            raise ValueError("result content must be text")
        if not isinstance(result.is_error, bool):
            raise ValueError("result is_error must be bool")
        digest = hashlib.sha256()
        digest.update(b"error\0" if result.is_error else b"ok\0")
        digest.update(result.content.encode("utf-8"))
        return cls(
            action_identity=tool_action_fingerprint(
                interaction.call.name, interaction.call.arguments,
            ),
            result_identity=digest.hexdigest(),
            is_error=result.is_error,
        )


@dataclass(frozen=True)
class StagnationStep:
    """Completed-step observations and explicit per-step progress deltas.

    None means unavailable, NOT zero. Mutation means structured mutation
    evidence, not a filesystem audit; verification means marked attempts,
    not successful tests or proof of correctness.
    """

    step: int
    observations: tuple[StagnationObservation, ...] = ()
    workspace_mutation_delta: int | None = None
    verification_delta: int | None = None

    def __post_init__(self) -> None:
        _validate_count("step", self.step)
        observations = tuple(self.observations)
        if not all(
            isinstance(item, StagnationObservation) for item in observations
        ):
            raise ValueError("observations must contain StagnationObservation")
        object.__setattr__(self, "observations", observations)
        for name in ("workspace_mutation_delta", "verification_delta"):
            value = getattr(self, name)
            if value is not None:
                _validate_count(name, value)


@dataclass(frozen=True)
class StagnationEvidence:
    """Scalar facts for a trailing window of consecutive logical steps."""

    window_size: int
    observed_steps: int
    start_step: int | None
    end_step: int | None
    active_steps: int
    tool_action_count: int
    unique_action_count: int
    repeated_action_count: int
    repeat_ratio: float
    unchanged_result_repeat_count: int
    changed_result_repeat_count: int
    new_action_count: int
    unknown_action_count: int
    workspace_mutation_delta: int | None
    verification_delta: int | None
    tool_error_delta: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StagnationSignal:
    """A factual warning, not a diagnosis, recovery, or stop decision."""

    detected: bool
    reason: str
    evidence: StagnationEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "reason": self.reason,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class _StepEvidence:
    source: StagnationStep
    repeated: int
    unchanged: int
    changed: int
    new: int


@dataclass(frozen=True)
class StagnationEvaluator:
    """Conservative, deterministic rolling-window evidence evaluator.

    Default: 16 consecutive active steps, all actions previously observed in
    the supplied prefix, all repeated results unchanged, and known zero
    mutation/verification deltas. A changed result or new action vetoes the
    warning. This is deliberately NOT a repeat-count threshold.

    Window aggregates retain at most window_size steps. One last-result hash
    per distinct action is retained for prefix-relative novelty (O(unique
    actions)); old raw output/history is never retained by the evaluator.
    """

    window_size: int = 16

    def __post_init__(self) -> None:
        _validate_count("window_size", self.window_size)
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")

    def evaluate(self, steps: Iterable[StagnationStep]) -> StagnationSignal:
        """Evaluate the final window, with no state retained between calls."""
        signal = self._signal(())
        for signal in self._iter_signals(steps):
            pass
        return signal

    def evaluate_windows(
        self, steps: Iterable[StagnationStep],
    ) -> tuple[StagnationSignal, ...]:
        """Return one trailing-window result per supplied completed step."""
        return tuple(self._iter_signals(steps))

    def _iter_signals(
        self, steps: Iterable[StagnationStep],
    ) -> Iterator[StagnationSignal]:
        tracker = StagnationTracker(self)
        for step in steps:
            yield tracker.record_step(step)

    def _signal(self, window: tuple[_StepEvidence, ...]) -> StagnationSignal:
        observations = [
            item for step in window for item in step.source.observations
        ]
        count = len(observations)
        repeated = sum(step.repeated for step in window)
        evidence = StagnationEvidence(
            window_size=self.window_size,
            observed_steps=len(window),
            start_step=window[0].source.step if window else None,
            end_step=window[-1].source.step if window else None,
            active_steps=sum(bool(step.source.observations) for step in window),
            tool_action_count=count,
            unique_action_count=len({
                item.action_identity for item in observations
                if item.action_identity is not None
            }),
            repeated_action_count=repeated,
            repeat_ratio=repeated / count if count else 0.0,
            unchanged_result_repeat_count=sum(step.unchanged for step in window),
            changed_result_repeat_count=sum(step.changed for step in window),
            new_action_count=sum(step.new for step in window),
            unknown_action_count=sum(
                item.action_identity is None for item in observations
            ),
            workspace_mutation_delta=_sum_delta(
                window, "workspace_mutation_delta",
            ),
            verification_delta=_sum_delta(window, "verification_delta"),
            tool_error_delta=sum(item.is_error for item in observations),
        )
        if evidence.observed_steps < self.window_size:
            reason = "insufficient_window"
        elif evidence.active_steps < self.window_size:
            reason = "insufficient_tool_activity"
        elif evidence.unknown_action_count:
            reason = "action_identity_unavailable"
        elif (
            evidence.workspace_mutation_delta is None
            or evidence.verification_delta is None
        ):
            reason = "progress_evidence_unavailable"
        elif evidence.workspace_mutation_delta or evidence.verification_delta:
            reason = "mutation_or_verification_observed"
        elif evidence.new_action_count or evidence.changed_result_repeat_count:
            reason = "new_actions_or_changed_observations"
        else:
            reason = "unchanged_repetition_without_observed_progress"
        return StagnationSignal(
            detected=reason == "unchanged_repetition_without_observed_progress",
            reason=reason,
            evidence=evidence,
        )


class StagnationTracker:
    """Incremental factual observation, sharing the offline evaluator's rules.

    Retain only a rolling window and prefix action/result hashes, not raw
    outputs or a second history. This tracker has no execution authority.
    """

    def __init__(self, evaluator: StagnationEvaluator | None = None) -> None:
        self.evaluator = evaluator or StagnationEvaluator()
        self._previous: dict[str, str] = {}
        self._window: deque[_StepEvidence] = deque(
            maxlen=self.evaluator.window_size,
        )
        self.last_step: int | None = None

    def record_step(self, step: StagnationStep) -> StagnationSignal:
        if not isinstance(step, StagnationStep):
            raise TypeError("steps must contain StagnationStep")
        if self.last_step is not None and step.step != self.last_step + 1:
            raise ValueError("steps must be consecutive and ordered")
        self.last_step = step.step
        repeated = unchanged = changed = new = 0
        for item in step.observations:
            key = item.action_identity
            if key is None:
                continue
            if key not in self._previous:
                new += 1
            else:
                repeated += 1
                if self._previous[key] == item.result_identity:
                    unchanged += 1
                else:
                    changed += 1
            self._previous[key] = item.result_identity
        self._window.append(_StepEvidence(step, repeated, unchanged, changed, new))
        return self.evaluator._signal(tuple(self._window))


def stagnation_steps_from_run_record(
    record: "RunRecord",
    *,
    coding_evidence: Mapping[int, CodingEvidenceSnapshot] | None = None,
) -> tuple[StagnationStep, ...]:
    """Adapt completed trace pairs without replaying tools or changing schemas.

    RunRecord does not persist coding counters. Optionally supply post-step
    snapshots for EVERY trace step, taken from the same run's JSONL/runtime
    evidence. Counters must be monotonic and run-local (initially zero).
    Without snapshots progress deltas remain unknown, disabling detection.
    Observed pairs are not a claim that every tool was physically executed.
    """
    steps: list[StagnationStep] = []
    previous = CodingEvidenceSnapshot()
    for trace_step in record.trace.steps:
        mutation_delta = verification_delta = None
        if coding_evidence is not None:
            snapshot = coding_evidence.get(trace_step.index)
            if not isinstance(snapshot, CodingEvidenceSnapshot):
                raise ValueError("coding evidence must cover every trace step")
            mutation_delta = (
                snapshot.workspace_mutations - previous.workspace_mutations
            )
            verification_delta = (
                snapshot.verification_attempts - previous.verification_attempts
            )
            previous = snapshot
        if isinstance(trace_step.output, Message):
            interactions = ()
        else:
            interactions = match_tool_interactions([
                *trace_step.output, *(trace_step.tool_result or ()),
            ])
            if len(interactions) != len(trace_step.output):
                raise ValueError("trace step must contain completed tool pairs")
        steps.append(StagnationStep(
            step=trace_step.index,
            observations=tuple(
                StagnationObservation.from_interaction(item)
                for item in interactions
            ),
            workspace_mutation_delta=mutation_delta,
            verification_delta=verification_delta,
        ))
    return tuple(steps)


def _sum_delta(window: tuple[_StepEvidence, ...], name: str) -> int | None:
    values = [getattr(step.source, name) for step in window]
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def _validate_count(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
