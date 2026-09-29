from dataclasses import dataclass
from enum import Enum

from pureharness.model import (
    ContextWindowExceededError,
    RecoverableModelError,
)
from pureharness.trace import RunEndReason


class RuntimeStage(str, Enum):
    """Deterministic stages where the Agent runtime can fail."""

    STATE_REDUCTION = "state_reduction"
    CONTEXT_PREPARATION = "context_preparation"
    TOOL_SELECTION = "tool_selection"
    MODEL_REQUEST = "model_request"
    EXECUTION = "execution"
    TOOL_EXECUTION = "tool_execution"
    POLICY_EVALUATION = "policy_evaluation"


class FailureCategory(str, Enum):
    """Broad failure categories supported by current runtime boundaries."""

    CONTEXT = "context"
    MODEL = "model"
    BUDGET = "budget"
    TOOL = "tool"
    POLICY = "policy"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    REBUILD_CONTEXT = "rebuild_context"
    FAIL = "fail"


@dataclass(frozen=True)
class ExecutionBudget:
    """Optional cumulative expensive-action limits for one Agent run."""

    max_model_attempts: int | None = None
    max_tool_calls: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_model_attempts", self.max_model_attempts),
            ("max_tool_calls", self.max_tool_calls),
        ):
            if value is not None:
                _validate_positive_integer(name, value)


@dataclass(frozen=True)
class ExecutionUsage:
    """Immutable cumulative physical-action usage for the current run."""

    model_attempts: int = 0
    tool_calls: int = 0

    def __post_init__(self) -> None:
        _validate_non_negative_integer(
            "model_attempts",
            self.model_attempts,
        )
        _validate_non_negative_integer("tool_calls", self.tool_calls)


class ExecutionBudgetExceeded(RuntimeError):
    """The runtime refused work that exceeded a cumulative run limit."""

    def __init__(
        self,
        *,
        resource: str,
        used: int,
        limit: int,
        requested: int | None = None,
    ) -> None:
        if resource not in {"model_attempts", "tool_calls"}:
            raise ValueError("Unsupported execution-budget resource")
        _validate_non_negative_integer("used", used)
        _validate_positive_integer("limit", limit)
        if requested is not None:
            _validate_positive_integer("requested", requested)

        self.resource = resource
        self.used = used
        self.limit = limit
        self.requested = requested
        self.remaining = max(limit - used, 0)

        detail = ""
        if requested is not None:
            detail = (
                f"; requested {requested}, remaining {self.remaining}"
            )
        super().__init__(
            f"Execution budget exhausted for {resource}: "
            f"used {used} of {limit}{detail}"
        )


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
        RuntimeStage.EXECUTION: FailureCategory.BUDGET,
        RuntimeStage.TOOL_EXECUTION: FailureCategory.TOOL,
        RuntimeStage.POLICY_EVALUATION: FailureCategory.POLICY,
    }
    _END_REASONS: dict[RuntimeStage, RunEndReason] = {
        RuntimeStage.STATE_REDUCTION: "context_error",
        RuntimeStage.CONTEXT_PREPARATION: "context_error",
        RuntimeStage.TOOL_SELECTION: "tool_selection_error",
        RuntimeStage.MODEL_REQUEST: "model_error",
        RuntimeStage.EXECUTION: "execution_budget_exceeded",
    }

    def classify_failure(
        self,
        *,
        step: int,
        stage: RuntimeStage,
        error: Exception,
    ) -> RuntimeFailure:
        """Return current failure evidence without applying recovery policy."""
        context_window_exceeded = (
            stage is RuntimeStage.MODEL_REQUEST
            and isinstance(error, ContextWindowExceededError)
        )
        effective_stage = (
            RuntimeStage.CONTEXT_PREPARATION
            if context_window_exceeded
            else stage
        )
        return RuntimeFailure(
            step=step,
            stage=effective_stage,
            category=self._CATEGORIES[effective_stage],
            error_type=type(error).__name__,
            recoverable=(
                context_window_exceeded
                or (
                    stage is RuntimeStage.MODEL_REQUEST
                    and isinstance(error, RecoverableModelError)
                )
            ),
        )

    def recovery_action(
        self,
        failure: RuntimeFailure,
        *,
        attempt: int,
        max_attempts: int,
        context_recovery_attempt: int = 0,
        max_context_recoveries: int = 0,
    ) -> RecoveryAction:
        """Decide whether one failed attempt may be retried."""
        _validate_positive_integer("attempt", attempt)
        _validate_positive_integer("max_attempts", max_attempts)
        _validate_non_negative_integer(
            "context_recovery_attempt",
            context_recovery_attempt,
        )
        _validate_non_negative_integer(
            "max_context_recoveries",
            max_context_recoveries,
        )
        if attempt > max_attempts:
            raise ValueError("attempt cannot exceed max_attempts")

        if (
            failure.recoverable
            and failure.category is FailureCategory.CONTEXT
            and context_recovery_attempt < max_context_recoveries
        ):
            return RecoveryAction.REBUILD_CONTEXT
        if (
            failure.recoverable
            and failure.category is FailureCategory.MODEL
            and attempt < max_attempts
        ):
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


def _validate_non_negative_integer(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
