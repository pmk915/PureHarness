import hashlib
import json
from dataclasses import dataclass

from pureharness.runtime import ExecutionUsage


@dataclass(frozen=True)
class ProgressSnapshot:
    """Immutable factual activity observed during one Agent run."""

    logical_steps_completed: int = 0
    model_attempts: int = 0
    tool_calls: int = 0
    successful_tool_results: int = 0
    failed_tool_results: int = 0
    unique_tool_actions: int = 0
    repeated_tool_actions: int = 0
    max_identical_tool_action_count: int = 0
    model_retries: int = 0
    context_recoveries: int = 0
    context_pressure_count: int = 0
    context_window_exceeded_count: int = 0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"ProgressSnapshot {name} must be a non-negative integer"
                )


class ProgressTracker:
    """Collect deterministic activity facts without controlling execution."""

    def __init__(self) -> None:
        self._successful_tool_results = 0
        self._failed_tool_results = 0
        self._action_counts: dict[str, int] = {}
        self._model_retries = 0
        self._context_recoveries = 0
        self._context_pressure_count = 0
        self._context_window_exceeded_count = 0

    def record_tool_action(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> None:
        fingerprint = _tool_action_fingerprint(name, arguments)
        if fingerprint is None:
            return
        self._action_counts[fingerprint] = (
            self._action_counts.get(fingerprint, 0) + 1
        )

    def record_tool_result(self, *, is_error: bool) -> None:
        if is_error:
            self._failed_tool_results += 1
        else:
            self._successful_tool_results += 1

    def record_model_retry(self) -> None:
        self._model_retries += 1

    def record_context_recovery(self) -> None:
        self._context_recoveries += 1

    def record_context_pressure(self) -> None:
        self._context_pressure_count += 1

    def record_context_window_exceeded(self) -> None:
        self._context_window_exceeded_count += 1

    def snapshot(
        self,
        execution_usage: ExecutionUsage,
        *,
        logical_steps_completed: int,
    ) -> ProgressSnapshot:
        action_counts = tuple(self._action_counts.values())
        return ProgressSnapshot(
            logical_steps_completed=logical_steps_completed,
            model_attempts=execution_usage.model_attempts,
            tool_calls=execution_usage.tool_calls,
            successful_tool_results=self._successful_tool_results,
            failed_tool_results=self._failed_tool_results,
            unique_tool_actions=len(action_counts),
            repeated_tool_actions=sum(
                count - 1 for count in action_counts
            ),
            max_identical_tool_action_count=max(
                action_counts,
                default=0,
            ),
            model_retries=self._model_retries,
            context_recoveries=self._context_recoveries,
            context_pressure_count=self._context_pressure_count,
            context_window_exceeded_count=(
                self._context_window_exceeded_count
            ),
        )


def _tool_action_fingerprint(
    name: str,
    arguments: dict[str, object],
) -> str | None:
    try:
        canonical = json.dumps(
            {"arguments": arguments, "name": name},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        # Telemetry is auxiliary. Unknown identities are skipped rather than
        # conflated or allowed to interfere with tool dispatch.
        return None
