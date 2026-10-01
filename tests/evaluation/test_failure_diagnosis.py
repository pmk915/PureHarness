import pytest

from pureharness.evaluation import (
    FailureDiagnoser,
    FailureType,
    Trajectory,
)


def _trajectory(
    *,
    run_id="run-diagnosis",
    completed=False,
    total_steps=2,
    tool_call_count=0,
    failed_tool_call_count=0,
):
    return Trajectory(
        run_id=run_id,
        completed=completed,
        total_steps=total_steps,
        tool_call_count=tool_call_count,
        failed_tool_call_count=failed_tool_call_count,
    )


def test_successful_trajectory_has_no_failure():
    trajectory = _trajectory(
        completed=True,
        tool_call_count=2,
        failed_tool_call_count=2,
    )

    result = FailureDiagnoser().diagnose(trajectory)

    assert result.failure_type is FailureType.NONE
    assert result.confidence == 1.0
    assert result.reason == "Trajectory completed successfully."
    assert result.signals == (("completed", True),)


@pytest.mark.parametrize(
    "tool_calls, failures, expected_confidence",
    [(2, 1, 0.5), (3, 2, 2 / 3), (4, 4, 1.0)],
)
def test_high_tool_failure_ratio_is_diagnosed(
    tool_calls,
    failures,
    expected_confidence,
):
    trajectory = _trajectory(
        tool_call_count=tool_calls,
        failed_tool_call_count=failures,
    )

    result = FailureDiagnoser().diagnose(trajectory)

    assert result.failure_type is FailureType.TOOL_FAILURE
    assert result.confidence == pytest.approx(expected_confidence)
    assert result.reason == (
        "Incomplete trajectory has a tool failure ratio at or above 0.50."
    )
    signals = dict(result.signals)
    assert signals == {
        "completed": False,
        "tool_call_count": tool_calls,
        "failed_tool_call_count": failures,
        "tool_failure_ratio": pytest.approx(expected_confidence),
        "high_tool_failure_ratio_threshold": 0.5,
    }


def test_incomplete_trajectory_without_clear_signal_is_unknown():
    trajectory = _trajectory(
        tool_call_count=4,
        failed_tool_call_count=1,
    )

    result = FailureDiagnoser().diagnose(trajectory)

    assert result.failure_type is FailureType.UNKNOWN
    assert result.confidence == 0.0
    assert result.reason == (
        "Trajectory is incomplete and no supported failure rule matched."
    )
    assert dict(result.signals)["tool_failure_ratio"] == 0.25


def test_empty_minimal_trajectory_is_unknown_without_division():
    trajectory = _trajectory(
        run_id="run-empty",
        total_steps=0,
        tool_call_count=0,
        failed_tool_call_count=0,
    )

    result = FailureDiagnoser().diagnose(trajectory)

    assert result.failure_type is FailureType.UNKNOWN
    assert result.confidence == 0.0
    assert result.to_dict() == {
        "failure_type": "unknown",
        "confidence": 0.0,
        "reason": (
            "Trajectory is incomplete and no supported failure rule "
            "matched."
        ),
        "signals": {
            "completed": False,
            "tool_call_count": 0,
            "failed_tool_call_count": 0,
            "tool_failure_ratio": None,
        },
    }


def test_failure_taxonomy_has_stable_values():
    assert [failure_type.value for failure_type in FailureType] == [
        "none",
        "tool_failure",
        "verification_failure",
        "state_failure",
        "planning_failure",
        "evidence_failure",
        "recovery_failure",
        "unknown",
    ]


def test_diagnoser_requires_trajectory():
    with pytest.raises(TypeError, match="trajectory must be Trajectory"):
        FailureDiagnoser().diagnose(object())
