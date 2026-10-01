import math

from dataclasses import dataclass

from pureharness.evaluation.diagnosis import (
    DiagnosisResult,
    FailureDiagnoser,
)
from pureharness.evaluation.evaluator import TrajectoryEvaluator
from pureharness.evaluation.metrics import TrajectoryMetrics
from pureharness.evaluation.trajectory import Trajectory


@dataclass(frozen=True)
class EvaluationReport:
    """Deterministic artifact combining existing metrics and diagnosis."""

    run_id: str
    metrics: TrajectoryMetrics
    diagnosis: DiagnosisResult
    overall_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be non-empty text")
        if not isinstance(self.metrics, TrajectoryMetrics):
            raise ValueError("metrics must be TrajectoryMetrics")
        if not isinstance(self.diagnosis, DiagnosisResult):
            raise ValueError("diagnosis must be DiagnosisResult")
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
            "diagnosis": self.diagnosis.to_dict(),
            "overall_score": self.overall_score,
        }

    def to_markdown(self) -> str:
        recovery = (
            "not available"
            if self.metrics.recovery_score is None
            else _format_number(self.metrics.recovery_score)
        )
        lines = [
            "# PureHarness Evaluation Report",
            "",
            "## Run",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Overall score: {_format_number(self.overall_score)}",
            "",
            "## Metrics",
            "",
            (
                "- Task success: "
                f"{_format_number(self.metrics.task_success_score)}"
            ),
            (
                "- Step efficiency: "
                f"{_format_number(self.metrics.step_efficiency_score)}"
            ),
            (
                "- Tool reliability: "
                f"{_format_number(self.metrics.tool_reliability_score)}"
            ),
            f"- Recovery: {recovery}",
            "",
            "## Diagnosis",
            "",
            f"- Failure type: `{self.diagnosis.failure_type.value}`",
            (
                "- Confidence: "
                f"{_format_number(self.diagnosis.confidence)}"
            ),
            f"- Reason: {self.diagnosis.reason}",
            "",
            "### Signals",
            "",
            *[
                f"- {name}: {_format_signal(value)}"
                for name, value in self.diagnosis.signals
            ],
        ]
        return "\n".join(lines)


class EvaluationReportBuilder:
    """Compose existing deterministic evaluation and diagnosis results."""

    def build(self, trajectory: Trajectory) -> EvaluationReport:
        if not isinstance(trajectory, Trajectory):
            raise TypeError("trajectory must be Trajectory")
        evaluation = TrajectoryEvaluator().evaluate(trajectory)
        diagnosis = FailureDiagnoser().diagnose(trajectory)
        return EvaluationReport(
            run_id=evaluation.run_id,
            metrics=evaluation.metrics,
            diagnosis=diagnosis,
            overall_score=evaluation.overall_score,
        )


def _format_number(value: float) -> str:
    return f"{value:.6f}"


def _format_signal(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return _format_number(value)
    return str(value)
