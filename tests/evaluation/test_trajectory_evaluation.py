from dataclasses import FrozenInstanceError

import pytest

from pureharness.agent import Agent
from pureharness.evaluation import (
    Trajectory,
    TrajectoryEvaluator,
    TrajectoryEvent,
    step_efficiency_score,
    task_success_score,
    tool_reliability_score,
    trajectory_from_run_record,
)
from pureharness.messages import Message


class CompletingModel:
    def generate(self, messages, tools):
        del messages, tools
        return Message(role="assistant", content="done")


def test_successful_trajectory_evaluation():
    trajectory = Trajectory(
        run_id="run-success",
        task_id="task-1",
        task_description="Update the fixture.",
        completed=True,
        total_steps=2,
        tool_call_count=2,
        failed_tool_call_count=0,
    )

    result = TrajectoryEvaluator().evaluate(trajectory)

    assert result.run_id == "run-success"
    assert result.metrics.task_success_score == 1.0
    assert result.metrics.step_efficiency_score == 0.5
    assert result.metrics.tool_reliability_score == 1.0
    assert result.metrics.recovery_score is None
    assert result.overall_score == pytest.approx(5 / 6)


def test_failed_tool_execution_lowers_reliability():
    trajectory = Trajectory(
        run_id="run-tool-error",
        completed=True,
        total_steps=1,
        tool_call_count=4,
        failed_tool_call_count=1,
    )

    result = TrajectoryEvaluator().evaluate(trajectory)

    assert result.metrics.tool_reliability_score == 0.75
    assert result.overall_score == pytest.approx((1.0 + 1.0 + 0.75) / 3)


def test_empty_minimal_trajectory_is_deterministic():
    trajectory = Trajectory(
        run_id="run-empty",
        completed=False,
        total_steps=0,
        tool_call_count=0,
        failed_tool_call_count=0,
    )

    result = TrajectoryEvaluator().evaluate(trajectory)

    assert result.metrics.task_success_score == 0.0
    assert result.metrics.step_efficiency_score == 1.0
    assert result.metrics.tool_reliability_score == 1.0
    assert result.overall_score == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "steps, expected",
    [(0, 1.0), (1, 1.0), (2, 0.5), (5, 0.2)],
)
def test_step_efficiency_formula(steps, expected):
    trajectory = Trajectory(
        run_id="run-steps",
        completed=False,
        total_steps=steps,
        tool_call_count=0,
        failed_tool_call_count=0,
    )

    assert step_efficiency_score(trajectory) == expected


@pytest.mark.parametrize(
    "completed, calls, failures, success_score, reliability_score",
    [
        (True, 0, 0, 1.0, 1.0),
        (False, 0, 0, 0.0, 1.0),
        (True, 2, 1, 1.0, 0.5),
        (False, 3, 3, 0.0, 0.0),
    ],
)
def test_task_success_and_tool_reliability_formulas(
    completed,
    calls,
    failures,
    success_score,
    reliability_score,
):
    trajectory = Trajectory(
        run_id="run-metrics",
        completed=completed,
        total_steps=1,
        tool_call_count=calls,
        failed_tool_call_count=failures,
    )

    assert task_success_score(trajectory) == success_score
    assert tool_reliability_score(trajectory) == reliability_score


def test_evaluator_output_format_and_optional_events():
    trajectory = Trajectory(
        run_id="run-format",
        task_id=None,
        task_description=None,
        completed=True,
        total_steps=1,
        tool_call_count=1,
        failed_tool_call_count=0,
        execution_events=(
            TrajectoryEvent(
                event_type="tool_result",
                step=0,
                tool_name="read_file",
                failed=False,
            ),
        ),
    )

    result = TrajectoryEvaluator().evaluate(trajectory)

    assert trajectory.to_dict() == {
        "run_id": "run-format",
        "task_id": None,
        "task_description": None,
        "completed": True,
        "total_steps": 1,
        "tool_call_count": 1,
        "failed_tool_call_count": 0,
        "execution_events": [
            {
                "event_type": "tool_result",
                "step": 0,
                "tool_name": "read_file",
                "failed": False,
            }
        ],
    }
    assert result.to_dict() == {
        "run_id": "run-format",
        "metrics": {
            "task_success_score": 1.0,
            "step_efficiency_score": 1.0,
            "tool_reliability_score": 1.0,
            "recovery_score": None,
        },
        "overall_score": 1.0,
    }


def test_run_record_adapter_is_read_only_and_explicit_about_task_data():
    agent = Agent(
        model=CompletingModel(),
        run_id_factory=lambda: "runtime-run",
    )

    assert agent.run("complete") == "done"
    assert agent.last_run_record is not None
    trajectory = trajectory_from_run_record(
        agent.last_run_record,
        task_id="task-from-caller",
        task_description="Caller-owned task information.",
    )

    assert trajectory == Trajectory(
        run_id="runtime-run",
        task_id="task-from-caller",
        task_description="Caller-owned task information.",
        completed=True,
        total_steps=1,
        tool_call_count=0,
        failed_tool_call_count=0,
    )
    assert agent.last_run_record.run_id == "runtime-run"


def test_trajectory_is_frozen_and_rejects_invalid_counts():
    trajectory = Trajectory(
        run_id="run-frozen",
        completed=True,
        total_steps=1,
        tool_call_count=1,
        failed_tool_call_count=0,
    )

    with pytest.raises(FrozenInstanceError):
        trajectory.total_steps = 2
    with pytest.raises(
        ValueError,
        match="cannot exceed tool_call_count",
    ):
        Trajectory(
            run_id="run-invalid",
            completed=False,
            total_steps=0,
            tool_call_count=0,
            failed_tool_call_count=1,
        )
