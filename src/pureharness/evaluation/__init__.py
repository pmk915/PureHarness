"""Deterministic, side-effect-free evaluation of completed trajectories.

Evaluation is separate from runtime execution so metrics cannot influence an
Agent run. Future diagnosis layers may consume these stable results without
coupling diagnostic policy back into the runtime kernel.
"""

from pureharness.evaluation.evaluator import (
    EvaluationResult,
    TrajectoryEvaluator,
)
from pureharness.evaluation.diagnosis import (
    DiagnosisResult,
    FailureDiagnoser,
)
from pureharness.evaluation.metrics import (
    RecoveryScoreCalculator,
    TrajectoryMetrics,
    step_efficiency_score,
    task_success_score,
    tool_reliability_score,
)
from pureharness.evaluation.trajectory import (
    Trajectory,
    TrajectoryEvent,
    trajectory_from_run_record,
)
from pureharness.evaluation.taxonomy import FailureType


__all__ = [
    "DiagnosisResult",
    "EvaluationResult",
    "FailureDiagnoser",
    "FailureType",
    "RecoveryScoreCalculator",
    "Trajectory",
    "TrajectoryEvaluator",
    "TrajectoryEvent",
    "TrajectoryMetrics",
    "step_efficiency_score",
    "task_success_score",
    "tool_reliability_score",
    "trajectory_from_run_record",
]
