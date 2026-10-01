import math

from dataclasses import dataclass

from pureharness.evaluation.taxonomy import FailureType
from pureharness.evaluation.trajectory import Trajectory


DiagnosisSignalValue = str | int | float | bool | None


@dataclass(frozen=True)
class DiagnosisResult:
    failure_type: FailureType
    confidence: float
    reason: str
    signals: tuple[tuple[str, DiagnosisSignalValue], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.failure_type, FailureType):
            raise ValueError("failure_type must be FailureType")
        if (
            not isinstance(self.confidence, float)
            or not math.isfinite(self.confidence)
            or self.confidence < 0.0
            or self.confidence > 1.0
        ):
            raise ValueError(
                "confidence must be a float between 0.0 and 1.0"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty text")
        signals = tuple(self.signals)
        names: set[str] = set()
        for signal in signals:
            if not isinstance(signal, tuple) or len(signal) != 2:
                raise ValueError("signals must contain name/value pairs")
            name, value = signal
            if not isinstance(name, str) or not name:
                raise ValueError("signal names must be non-empty text")
            if name in names:
                raise ValueError("signal names must be unique")
            names.add(name)
            if not _is_signal_value(value):
                raise ValueError(
                    "signal values must be JSON scalar values"
                )
        object.__setattr__(self, "signals", signals)

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_type": self.failure_type.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "signals": dict(self.signals),
        }


class FailureDiagnoser:
    """Apply ordered, deterministic rules to one completed trajectory."""

    HIGH_TOOL_FAILURE_RATIO = 0.5

    def diagnose(self, trajectory: Trajectory) -> DiagnosisResult:
        if not isinstance(trajectory, Trajectory):
            raise TypeError("trajectory must be Trajectory")

        if trajectory.completed:
            return DiagnosisResult(
                failure_type=FailureType.NONE,
                confidence=1.0,
                reason="Trajectory completed successfully.",
                signals=(("completed", True),),
            )

        ratio = _tool_failure_ratio(trajectory)
        common_signals: tuple[
            tuple[str, DiagnosisSignalValue],
            ...,
        ] = (
            ("completed", False),
            ("tool_call_count", trajectory.tool_call_count),
            (
                "failed_tool_call_count",
                trajectory.failed_tool_call_count,
            ),
            ("tool_failure_ratio", ratio),
        )
        if (
            ratio is not None
            and ratio >= self.HIGH_TOOL_FAILURE_RATIO
        ):
            return DiagnosisResult(
                failure_type=FailureType.TOOL_FAILURE,
                confidence=ratio,
                reason=(
                    "Incomplete trajectory has a tool failure ratio "
                    "at or above 0.50."
                ),
                signals=common_signals
                + (
                    (
                        "high_tool_failure_ratio_threshold",
                        self.HIGH_TOOL_FAILURE_RATIO,
                    ),
                ),
            )

        return DiagnosisResult(
            failure_type=FailureType.UNKNOWN,
            confidence=0.0,
            reason=(
                "Trajectory is incomplete and no supported failure "
                "rule matched."
            ),
            signals=common_signals,
        )


def _tool_failure_ratio(trajectory: Trajectory) -> float | None:
    if trajectory.tool_call_count == 0:
        return None
    return (
        trajectory.failed_tool_call_count
        / trajectory.tool_call_count
    )


def _is_signal_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)
