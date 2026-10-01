from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.messages import Message
from pureharness.verification import VerificationOutcome


class CompletionDecision(str, Enum):
    ACCEPT = "accept"
    RECONSIDER = "reconsider"


class CompletionReason(str, Enum):
    VERIFICATION_FAILED_AFTER_MUTATION = (
        "verification_failed_after_mutation"
    )
    EXECUTION_WITHOUT_STRUCTURED_MUTATION = (
        "execution_without_structured_mutation"
    )
    MUTATION_WITHOUT_POST_MUTATION_EXECUTION = (
        "mutation_without_post_mutation_execution"
    )


class CompletionRecheckSkipReason(str, Enum):
    CONTEXT_CAPACITY = "context_capacity"
    STEP_BUDGET = "step_budget"
    MODEL_ATTEMPT_BUDGET = "model_attempt_budget"
    RECHECK_LIMIT = "recheck_limit"


@dataclass(frozen=True)
class CompletionAssessment:
    decision: CompletionDecision
    reason: CompletionReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, CompletionDecision):
            raise ValueError("decision must be CompletionDecision")
        if self.decision is CompletionDecision.ACCEPT:
            if self.reason is not None:
                raise ValueError("accepted completion cannot have a reason")
        elif not isinstance(self.reason, CompletionReason):
            raise ValueError("reconsidered completion requires a reason")


class CompletionPolicy(Protocol):
    max_rechecks: int

    def assess(
        self,
        evidence: CodingEvidenceSnapshot,
    ) -> CompletionAssessment:
        ...


@dataclass(frozen=True)
class EvidenceAwareCodingCompletionPolicy:
    """Request one reconsideration from narrow factual coding evidence."""

    max_rechecks: int = 1

    def __post_init__(self) -> None:
        if self.max_rechecks != 1:
            raise ValueError("coding completion policy allows one recheck")

    def assess(
        self,
        evidence: CodingEvidenceSnapshot,
    ) -> CompletionAssessment:
        if not isinstance(evidence, CodingEvidenceSnapshot):
            raise TypeError("evidence must be CodingEvidenceSnapshot")
        verification_failed = evidence.last_verification_outcome in {
            VerificationOutcome.EXIT_NONZERO,
            VerificationOutcome.TOOL_ERROR,
        }
        verification_failed_after_mutation = (
            verification_failed
            and evidence.last_mutation_step is not None
            and evidence.last_verification_step is not None
            and (
                evidence.last_verification_step
                > evidence.last_mutation_step
                or (
                    evidence.last_verification_step
                    == evidence.last_mutation_step
                    and evidence.verifications_since_last_mutation > 0
                )
            )
        )
        if verification_failed_after_mutation:
            return CompletionAssessment(
                CompletionDecision.RECONSIDER,
                CompletionReason.VERIFICATION_FAILED_AFTER_MUTATION,
            )
        mutation_after_failed_verification = (
            verification_failed
            and evidence.last_mutation_step is not None
            and evidence.last_verification_step is not None
            and evidence.verifications_since_last_mutation == 0
            and (
                evidence.last_mutation_step
                > evidence.last_verification_step
                or (
                    evidence.last_mutation_step
                    == evidence.last_verification_step
                )
            )
        )
        if (
            evidence.workspace_mutations > 0
            and evidence.executions_since_last_mutation == 0
            and not mutation_after_failed_verification
        ):
            return CompletionAssessment(
                CompletionDecision.RECONSIDER,
                CompletionReason.MUTATION_WITHOUT_POST_MUTATION_EXECUTION,
            )
        if (
            evidence.workspace_mutations == 0
            and evidence.command_executions + evidence.process_starts > 0
        ):
            return CompletionAssessment(
                CompletionDecision.RECONSIDER,
                CompletionReason.EXECUTION_WITHOUT_STRUCTURED_MUTATION,
            )
        return CompletionAssessment(CompletionDecision.ACCEPT)


_DEFAULT_CODING_COMPLETION_POLICY = EvidenceAwareCodingCompletionPolicy()


def default_coding_completion_policy() -> CompletionPolicy:
    return _DEFAULT_CODING_COMPLETION_POLICY


def render_completion_recheck(reason: CompletionReason) -> Message:
    if reason is CompletionReason.VERIFICATION_FAILED_AFTER_MUTATION:
        return Message(
            role="system",
            content=(
                "[PureHarness Completion Recheck]\n\n"
                "A verification-marked command executed after your latest "
                "workspace mutation did not complete successfully.\n\n"
                "Review the current workspace state and verification "
                "evidence before deciding whether the task is complete."
            ),
        )
    if reason is CompletionReason.MUTATION_WITHOUT_POST_MUTATION_EXECUTION:
        evidence = (
            "A successful structured workspace mutation was observed, but "
            "no execution was observed after the latest structured mutation."
        )
    elif reason is CompletionReason.EXECUTION_WITHOUT_STRUCTURED_MUTATION:
        evidence = (
            "Execution activity was observed, but no successful structured "
            "workspace mutation was observed."
        )
    else:
        raise ValueError("unsupported completion recheck reason")

    return Message(
        role="system",
        content=(
            "[PureHarness Completion Recheck]\n\n"
            "PureHarness observed that you attempted to finish, but the "
            "current run does not yet contain strong completion evidence "
            "for the latest coding work.\n\n"
            f"Reason: {reason.value}\n\n"
            f"{evidence}\n\n"
            "Re-evaluate whether the requested task is actually complete. "
            "If this task was intentionally read-only, no workspace change "
            "was appropriate, or verification cannot be performed, you may "
            "finish and state that clearly. Otherwise continue the "
            "implementation and obtain relevant evidence before reporting "
            "completion."
        ),
    )
