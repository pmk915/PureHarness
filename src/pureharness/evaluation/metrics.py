import math

from dataclasses import dataclass
from typing import Protocol

from pureharness.evaluation.trajectory import Trajectory


def task_success_score(trajectory: Trajectory) -> float:
    """Return 1.0 only for a successfully completed trajectory."""

    _validate_trajectory(trajectory)
    return 1.0 if trajectory.completed else 0.0


def step_efficiency_score(trajectory: Trajectory) -> float:
    """Score steps as 1 / max(1, total_steps).

    With no external optimal-step oracle, zero or one step is the neutral
    maximum and each additional step deterministically lowers the score.
    """

    _validate_trajectory(trajectory)
    return 1.0 / max(1, trajectory.total_steps)


def tool_reliability_score(trajectory: Trajectory) -> float:
    """Return the fraction of tool calls without error results."""

    _validate_trajectory(trajectory)
    if trajectory.tool_call_count == 0:
        return 1.0
    return 1.0 - (
        trajectory.failed_tool_call_count / trajectory.tool_call_count
    )


class RecoveryScoreCalculator(Protocol):
    """Future recovery scoring interface; M22.1 provides no implementation."""

    def calculate(self, trajectory: Trajectory) -> float:
        ...


@dataclass(frozen=True)
class TrajectoryMetrics:
    task_success_score: float
    step_efficiency_score: float
    tool_reliability_score: float
    recovery_score: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_success_score",
            "step_efficiency_score",
            "tool_reliability_score",
        ):
            _validate_score(name, getattr(self, name))
        if self.recovery_score is not None:
            _validate_score("recovery_score", self.recovery_score)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "task_success_score": self.task_success_score,
            "step_efficiency_score": self.step_efficiency_score,
            "tool_reliability_score": self.tool_reliability_score,
            "recovery_score": self.recovery_score,
        }


def _validate_trajectory(trajectory: Trajectory) -> None:
    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be Trajectory")


def _validate_score(name: str, value: float) -> None:
    if (
        not isinstance(value, float)
        or not math.isfinite(value)
        or value < 0.0
        or value > 1.0
    ):
        raise ValueError(f"{name} must be a float between 0.0 and 1.0")
