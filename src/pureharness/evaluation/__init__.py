"""Deterministic, side-effect-free evaluation of completed trajectories.

M22 metrics and diagnosis do not control execution. Stagnation observations
may be consumed by an explicitly enabled runtime advisory policy; the factual
evaluator itself has no execution authority.
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
from pureharness.evaluation.report import (
    EvaluationReport,
    EvaluationReportBuilder,
)
from pureharness.evaluation.recovery import (
    RecoveryAction,
    RecoveryAdvisor,
    RecoverySignal,
)
from pureharness.evaluation.trajectory import (
    Trajectory,
    TrajectoryEvent,
    trajectory_from_run_record,
)
from pureharness.evaluation.taxonomy import FailureType
from pureharness.evaluation.progress_gap import (
    ProgressGapEvidence,
    ProgressGapEvaluator,
)
from pureharness.evaluation.stagnation import (
    StagnationEvidence,
    StagnationEvaluator,
    StagnationObservation,
    StagnationSignal,
    StagnationStep,
    StagnationTracker,
    stagnation_steps_from_run_record,
)


__all__ = [
    "DiagnosisResult",
    "EvaluationResult",
    "EvaluationReport",
    "EvaluationReportBuilder",
    "FailureDiagnoser",
    "FailureType",
    "ProgressGapEvidence",
    "ProgressGapEvaluator",
    "RecoveryScoreCalculator",
    "RecoveryAction",
    "RecoveryAdvisor",
    "RecoverySignal",
    "StagnationEvidence",
    "StagnationEvaluator",
    "StagnationObservation",
    "StagnationSignal",
    "StagnationStep",
    "StagnationTracker",
    "Trajectory",
    "TrajectoryEvaluator",
    "TrajectoryEvent",
    "TrajectoryMetrics",
    "step_efficiency_score",
    "task_success_score",
    "tool_reliability_score",
    "trajectory_from_run_record",
    "stagnation_steps_from_run_record",
]
