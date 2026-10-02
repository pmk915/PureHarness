"""Offline activity since structured anchors, not task-progress judgments."""

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass

from pureharness.evaluation.stagnation import StagnationStep


@dataclass(frozen=True)
class ProgressGapEvidence:
    """Post-step facts for the open span after the last observed anchor.

    An anchor is a positive successful structured-mutation delta or explicit
    verification-attempt delta, regardless of verification outcome. Neither
    proves task progress. Before any anchor, the span starts at the supplied
    prefix's first step. Anchor steps are excluded in full: aggregate per-step
    evidence cannot locate actions before/after an anchor within a batch.

    Novelty and result changes are relative to the entire supplied prefix,
    including anchor steps. Unavailable progress evidence is never zero;
    progress_evidence_complete=False prevents interpreting a span as confirmed
    anchor-free. Distances to an individual anchor are None if never observed
    or if that counter became unavailable after its last observed anchor.
    """

    step: int | None = None
    span_start_step: int | None = None
    last_progress_anchor_step: int | None = None
    last_structured_mutation_step: int | None = None
    last_verification_attempt_step: int | None = None
    workspace_mutation_delta: int | None = None
    verification_delta: int | None = None
    progress_evidence_complete: bool = True
    steps_since_anchor: int = 0
    active_steps: int = 0
    steps_since_last_structured_mutation: int | None = None
    steps_since_last_verification_attempt: int | None = None
    tool_actions_since_anchor: int = 0
    new_action_count_since_anchor: int = 0
    repeated_action_count_since_anchor: int = 0
    unchanged_repeat_count_since_anchor: int = 0
    changed_observation_count_since_anchor: int = 0
    unknown_action_count_since_anchor: int = 0
    tool_error_count_since_anchor: int = 0

    def __post_init__(self) -> None:
        optional_counts = {
            "step", "span_start_step", "last_progress_anchor_step",
            "last_structured_mutation_step", "last_verification_attempt_step",
            "workspace_mutation_delta", "verification_delta",
            "steps_since_last_structured_mutation",
            "steps_since_last_verification_attempt",
        }
        for name, value in self.__dict__.items():
            if name == "progress_evidence_complete":
                if not isinstance(value, bool):
                    raise ValueError("progress_evidence_complete must be bool")
            elif not (value is None and name in optional_counts) and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.active_steps > self.steps_since_anchor:
            raise ValueError("active_steps cannot exceed steps_since_anchor")
        if (
            self.new_action_count_since_anchor
            + self.repeated_action_count_since_anchor
            + self.unknown_action_count_since_anchor
            != self.tool_actions_since_anchor
        ):
            raise ValueError("action counts must cover tool_actions_since_anchor")
        if (
            self.unchanged_repeat_count_since_anchor
            + self.changed_observation_count_since_anchor
            != self.repeated_action_count_since_anchor
        ):
            raise ValueError("result counts must cover repeated actions")
        if self.tool_error_count_since_anchor > self.tool_actions_since_anchor:
            raise ValueError("tool errors cannot exceed tool actions")

    @property
    def unique_action_growth(self) -> int:
        """Growth of prefix-global distinct identities since this anchor."""
        return self.new_action_count_since_anchor

    @property
    def repeat_ratio(self) -> float:
        return self._ratio(self.repeated_action_count_since_anchor)

    @property
    def unchanged_repeat_ratio(self) -> float:
        """Unchanged repeated observations divided by ALL span actions."""
        return self._ratio(self.unchanged_repeat_count_since_anchor)

    def _ratio(self, count: int) -> float:
        if not self.tool_actions_since_anchor:
            return 0.0
        return count / self.tool_actions_since_anchor

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "unique_action_growth": self.unique_action_growth,
            "repeat_ratio": self.repeat_ratio,
            "unchanged_repeat_ratio": self.unchanged_repeat_ratio,
        }


@dataclass(frozen=True)
class ProgressGapEvaluator:
    """Stateless offline evidence; no thresholds, classifications or effects.

    Reuse stagnation_steps_from_run_record(record, coding_evidence=...) to
    adapt raw completed pairs and same-run post-step snapshots. No RunRecord
    or JSONL schema changes are needed. iter_evidence is single-pass and keeps
    scalar counters plus one last-result hash per unique prefix action
    (O(unique actions), not constant memory). It does not retain input steps.
    evaluate_steps explicitly materializes the O(steps) output timeline.
    """

    def evaluate(self, steps: Iterable[StagnationStep]) -> ProgressGapEvidence:
        evidence = ProgressGapEvidence()
        for evidence in self.iter_evidence(steps):
            pass
        return evidence

    def evaluate_steps(
        self, steps: Iterable[StagnationStep],
    ) -> tuple[ProgressGapEvidence, ...]:
        return tuple(self.iter_evidence(steps))

    def iter_evidence(
        self, steps: Iterable[StagnationStep],
    ) -> Iterator[ProgressGapEvidence]:
        previous: dict[str, str] = {}
        last_step = last_anchor = last_mutation = last_verification = None
        mutation_known = verification_known = complete = True
        span_start = None
        age = active = actions = new = repeated = 0
        unchanged = changed = unknown = errors = 0
        for step in steps:
            if not isinstance(step, StagnationStep):
                raise TypeError("steps must contain StagnationStep")
            if last_step is not None and step.step != last_step + 1:
                raise ValueError("steps must be consecutive and ordered")
            last_step = step.step
            if span_start is None:
                span_start = step.step
            mutation = step.workspace_mutation_delta
            verification = step.verification_delta
            if mutation:
                last_mutation, mutation_known = step.step, True
            elif mutation is None:
                mutation_known = False
            if verification:
                last_verification, verification_known = step.step, True
            elif verification is None:
                verification_known = False
            anchor = bool(mutation or verification)
            if anchor:
                last_anchor, span_start, complete = step.step, step.step + 1, True
                age = active = actions = new = repeated = 0
                unchanged = changed = unknown = errors = 0
            else:
                complete = complete and mutation is not None and verification is not None
                age += 1
                active += bool(step.observations)
            for item in step.observations:
                key = item.action_identity
                if not anchor:
                    actions += 1
                    errors += item.is_error
                    if key is None:
                        unknown += 1
                    elif key not in previous:
                        new += 1
                    else:
                        repeated += 1
                        if previous[key] == item.result_identity:
                            unchanged += 1
                        else:
                            changed += 1
                if key is not None:
                    previous[key] = item.result_identity
            yield ProgressGapEvidence(
                step=step.step,
                span_start_step=span_start,
                last_progress_anchor_step=last_anchor,
                last_structured_mutation_step=last_mutation,
                last_verification_attempt_step=last_verification,
                workspace_mutation_delta=mutation,
                verification_delta=verification,
                progress_evidence_complete=complete,
                steps_since_anchor=age,
                active_steps=active,
                steps_since_last_structured_mutation=(
                    step.step - last_mutation
                    if last_mutation is not None and mutation_known else None
                ),
                steps_since_last_verification_attempt=(
                    step.step - last_verification
                    if last_verification is not None and verification_known else None
                ),
                tool_actions_since_anchor=actions,
                new_action_count_since_anchor=new,
                repeated_action_count_since_anchor=repeated,
                unchanged_repeat_count_since_anchor=unchanged,
                changed_observation_count_since_anchor=changed,
                unknown_action_count_since_anchor=unknown,
                tool_error_count_since_anchor=errors,
            )
