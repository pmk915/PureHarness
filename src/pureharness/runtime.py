from dataclasses import dataclass
from enum import Enum

from pureharness.model import RecoverableModelError
from pureharness.trace import RunEndReason


class RuntimeStage(str, Enum):
    """Deterministic stages where the Agent runtime can fail."""

    STATE_REDUCTION = "state_reduction"
    CONTEXT_PREPARATION = "context_preparation"
    TOOL_SELECTION = "tool_selection"
    MODEL_REQUEST = "model_request"
    TOOL_EXECUTION = "tool_execution"
    POLICY_EVALUATION = "policy_evaluation"


class FailureCategory(str, Enum):
    """Broad failure categories supported by current runtime boundaries."""

    CONTEXT = "context"
    MODEL = "model"
    TOOL = "tool"
    POLICY = "policy"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    FAIL = "fail"


@dataclass(frozen=True)
class RuntimeFailure:
    """Structured evidence about one runtime-level failure.

    This value describes a failure; it does not wrap or replace the original
    exception. Agent continues to re-raise that exception unchanged.
    """

    step: int
    stage: RuntimeStage
    category: FailureCategory
    error_type: str
    recoverable: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise ValueError("RuntimeFailure step must be non-negative")
        if not isinstance(self.stage, RuntimeStage):
            raise ValueError("RuntimeFailure stage must be RuntimeStage")
        if not isinstance(self.category, FailureCategory):
            raise ValueError(
                "RuntimeFailure category must be FailureCategory"
            )
        if not isinstance(self.error_type, str) or not self.error_type:
            raise ValueError("RuntimeFailure error_type must be non-empty")
        if not isinstance(self.recoverable, bool):
            raise ValueError("RuntimeFailure recoverable must be bool")


class RuntimeController:
    """Classify lifecycle failures without reasoning about the task."""

    _CATEGORIES = {
        RuntimeStage.STATE_REDUCTION: FailureCategory.CONTEXT,
        RuntimeStage.CONTEXT_PREPARATION: FailureCategory.CONTEXT,
        RuntimeStage.TOOL_SELECTION: FailureCategory.TOOL,
        RuntimeStage.MODEL_REQUEST: FailureCategory.MODEL,
        RuntimeStage.TOOL_EXECUTION: FailureCategory.TOOL,
        RuntimeStage.POLICY_EVALUATION: FailureCategory.POLICY,
    }
    _END_REASONS: dict[RuntimeStage, RunEndReason] = {
        RuntimeStage.STATE_REDUCTION: "context_error",
        RuntimeStage.CONTEXT_PREPARATION: "context_error",
        RuntimeStage.TOOL_SELECTION: "tool_selection_error",
        RuntimeStage.MODEL_REQUEST: "model_error",
    }

    def classify_failure(
        self,
        *,
        step: int,
        stage: RuntimeStage,
        error: Exception,
    ) -> RuntimeFailure:
        """Return current failure evidence without applying recovery policy."""
        return RuntimeFailure(
            step=step,
            stage=stage,
            category=self._CATEGORIES[stage],
            error_type=type(error).__name__,
            recoverable=(
                stage is RuntimeStage.MODEL_REQUEST
                and isinstance(error, RecoverableModelError)
            ),
        )

    def recovery_action(
        self,
        failure: RuntimeFailure,
        *,
        attempt: int,
        max_attempts: int,
    ) -> RecoveryAction:
        """Decide whether one failed attempt may be retried."""
        _validate_positive_integer("attempt", attempt)
        _validate_positive_integer("max_attempts", max_attempts)
        if attempt > max_attempts:
            raise ValueError("attempt cannot exceed max_attempts")

        if failure.recoverable and attempt < max_attempts:
            return RecoveryAction.RETRY
        return RecoveryAction.FAIL

    def end_reason(self, failure: RuntimeFailure) -> RunEndReason:
        """Map a currently fatal runtime failure to its compatible reason."""
        try:
            return self._END_REASONS[failure.stage]
        except KeyError as exc:
            raise ValueError(
                "Runtime stage has no fatal Run end reason: "
                f"{failure.stage.value}"
            ) from exc


def _validate_positive_integer(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
