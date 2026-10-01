import math

from dataclasses import dataclass

from pureharness.evaluation.metrics import (
    TrajectoryMetrics,
    step_efficiency_score,
    task_success_score,
    tool_reliability_score,
)
from pureharness.evaluation.trajectory import Trajectory


@dataclass(frozen=True)
class EvaluationResult:
    run_id: str
    metrics: TrajectoryMetrics
    overall_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be non-empty text")
        if not isinstance(self.metrics, TrajectoryMetrics):
            raise ValueError("metrics must be TrajectoryMetrics")
        if (
            not isinstance(self.overall_score, float)
            or not math.isfinite(self.overall_score)
            or self.overall_score < 0.0
            or self.overall_score > 1.0
        ):
            raise ValueError(
                "overall_score must be a float between 0.0 and 1.0"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "metrics": self.metrics.to_dict(),
            "overall_score": self.overall_score,
        }


class TrajectoryEvaluator:
    """Calculate deterministic, equal-weight M22.1 trajectory metrics."""

    def evaluate(self, trajectory: Trajectory) -> EvaluationResult:
        if not isinstance(trajectory, Trajectory):
            raise TypeError("trajectory must be Trajectory")
        metrics = TrajectoryMetrics(
            task_success_score=task_success_score(trajectory),
            step_efficiency_score=step_efficiency_score(trajectory),
            tool_reliability_score=tool_reliability_score(trajectory),
        )
        implemented_scores = (
            metrics.task_success_score,
            metrics.step_efficiency_score,
            metrics.tool_reliability_score,
        )
        return EvaluationResult(
            run_id=trajectory.run_id,
            metrics=metrics,
            overall_score=sum(implemented_scores) / len(implemented_scores),
        )
