import math

from dataclasses import dataclass
from enum import Enum

from pureharness.evaluation.diagnosis import (
    DiagnosisResult,
    DiagnosisSignalValue,
)
from pureharness.evaluation.taxonomy import FailureType


class RecoveryAction(str, Enum):
    NONE = "none"
    RETRY_TOOL = "retry_tool"
    VERIFY_RESULT = "verify_result"
    INSPECT_STATE = "inspect_state"
    COLLECT_EVIDENCE = "collect_evidence"
    REVIEW_PLAN = "review_plan"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecoverySignal:
    """Advisory recovery guidance with no execution authority."""

    action: RecoveryAction
    reason: str
    confidence: float
    metadata: tuple[tuple[str, DiagnosisSignalValue], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.action, RecoveryAction):
            raise ValueError("action must be RecoveryAction")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty text")
        if (
            not isinstance(self.confidence, float)
            or not math.isfinite(self.confidence)
            or self.confidence < 0.0
            or self.confidence > 1.0
        ):
            raise ValueError(
                "confidence must be a float between 0.0 and 1.0"
            )
        metadata = tuple(self.metadata)
        names: set[str] = set()
        for item in metadata:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("metadata must contain name/value pairs")
            name, value = item
            if not isinstance(name, str) or not name:
                raise ValueError("metadata names must be non-empty text")
            if name in names:
                raise ValueError("metadata names must be unique")
            names.add(name)
            if not _is_metadata_value(value):
                raise ValueError(
                    "metadata values must be JSON scalar values"
                )
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


class RecoveryAdvisor:
    """Map a diagnosis to deterministic guidance without taking action."""

    _ACTIONS = {
        FailureType.NONE: RecoveryAction.NONE,
        FailureType.TOOL_FAILURE: RecoveryAction.RETRY_TOOL,
        FailureType.VERIFICATION_FAILURE: RecoveryAction.VERIFY_RESULT,
        FailureType.STATE_FAILURE: RecoveryAction.INSPECT_STATE,
        FailureType.EVIDENCE_FAILURE: RecoveryAction.COLLECT_EVIDENCE,
        FailureType.PLANNING_FAILURE: RecoveryAction.REVIEW_PLAN,
        FailureType.RECOVERY_FAILURE: RecoveryAction.UNKNOWN,
        FailureType.UNKNOWN: RecoveryAction.UNKNOWN,
    }
    _REASONS = {
        FailureType.NONE: "No recovery guidance is needed.",
        FailureType.TOOL_FAILURE: (
            "Retry the failed tool operation if it is safe to do so."
        ),
        FailureType.VERIFICATION_FAILURE: (
            "Review the verification result before proceeding."
        ),
        FailureType.STATE_FAILURE: (
            "Inspect the current state before proceeding."
        ),
        FailureType.EVIDENCE_FAILURE: (
            "Collect additional execution evidence before proceeding."
        ),
        FailureType.PLANNING_FAILURE: (
            "Review the current plan before proceeding."
        ),
        FailureType.RECOVERY_FAILURE: (
            "No specific recovery guidance is defined for a recovery "
            "failure."
        ),
        FailureType.UNKNOWN: (
            "No specific recovery guidance is available for this failure."
        ),
    }

    def advise(self, diagnosis: DiagnosisResult) -> RecoverySignal:
        if not isinstance(diagnosis, DiagnosisResult):
            raise TypeError("diagnosis must be DiagnosisResult")
        failure_type = diagnosis.failure_type
        return RecoverySignal(
            action=self._ACTIONS[failure_type],
            reason=self._REASONS[failure_type],
            confidence=diagnosis.confidence,
            metadata=(("failure_type", failure_type.value),),
        )


def _is_metadata_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)
