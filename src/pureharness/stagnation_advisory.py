"""Opt-in runtime policy over stagnation facts; no tools or task judgments."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar

from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.evaluation.stagnation import StagnationSignal
from pureharness.messages import Message


class StagnationAdvisoryPhase(str, Enum):
    ARMED = "armed"
    LATCHED = "latched"


@dataclass(frozen=True)
class StagnationAdvisoryState:
    """Bounded, per-run intervention state, never durable conversation state."""

    phase: StagnationAdvisoryPhase = StagnationAdvisoryPhase.ARMED
    advisory_count: int = 0
    mutation_count_at_delivery: int = 0
    verification_count_at_delivery: int = 0
    pending_signal: StagnationSignal | None = None
    delivered_at_step: int | None = None


@dataclass(frozen=True)
class StagnationAdvisoryPolicy:
    """Pure state transitions; emission and model dispatch belong to Agent."""

    max_advisories_per_run: ClassVar[int] = 2

    def after_tool_step(
        self,
        state: StagnationAdvisoryState,
        signal: StagnationSignal,
        evidence: CodingEvidenceSnapshot,
    ) -> StagnationAdvisoryState:
        if state.pending_signal is not None:
            return state
        if state.phase is StagnationAdvisoryPhase.LATCHED:
            progressed = (
                evidence.workspace_mutations > state.mutation_count_at_delivery
                or (
                    evidence.verification_attempts
                    > state.verification_count_at_delivery
                )
            )
            if not progressed:
                return state
            state = replace(state, phase=StagnationAdvisoryPhase.ARMED)
        if signal.detected and state.advisory_count < self.max_advisories_per_run:
            return replace(state, pending_signal=signal)
        return state

    def delivered(
        self,
        state: StagnationAdvisoryState,
        evidence: CodingEvidenceSnapshot,
        step: int,
    ) -> StagnationAdvisoryState:
        pending = state.pending_signal
        if pending is None or state.delivered_at_step is not None:
            return state
        if state.advisory_count >= self.max_advisories_per_run:
            raise ValueError("stagnation advisory cap exhausted")
        if (
            pending.evidence.end_step is None
            or step != pending.evidence.end_step + 1
        ):
            raise ValueError("advisory must be delivered on the next logical step")
        return replace(
            state,
            phase=StagnationAdvisoryPhase.LATCHED,
            advisory_count=state.advisory_count + 1,
            mutation_count_at_delivery=evidence.workspace_mutations,
            verification_count_at_delivery=evidence.verification_attempts,
            delivered_at_step=step,
        )

    def clear_pending(
        self, state: StagnationAdvisoryState,
    ) -> StagnationAdvisoryState:
        """Consume after logical completion, or discard undeliverable guidance.

        Clearing guidance does not re-arm a latched policy.
        """
        return replace(state, pending_signal=None, delivered_at_step=None)


def render_stagnation_advisory() -> Message:
    return Message(
        role="system",
        content=(
            "[PureHarness Stagnation Advisory]\n\n"
            "Recent execution contains repeated tool actions with unchanged "
            "observations and no recorded structured workspace mutation or "
            "verification-attempt progress in the observed window.\n\n"
            "Reassess the current plan and summarize what is already known. "
            "Consider a materially different next action toward the task goal. "
            "Continue intentional repetition only if a changed observation "
            "is expected. This observation does not determine task correctness."
        ),
    )
