import pytest

from pureharness.evaluation import (
    DiagnosisResult,
    FailureType,
    RecoveryAction,
    RecoveryAdvisor,
)


def _diagnosis(
    failure_type: FailureType,
    *,
    confidence: float = 0.75,
) -> DiagnosisResult:
    return DiagnosisResult(
        failure_type=failure_type,
        confidence=confidence,
        reason=f"Observed {failure_type.value}.",
        signals=(("source", "test"),),
    )


def test_none_diagnosis_needs_no_recovery():
    signal = RecoveryAdvisor().advise(
        _diagnosis(FailureType.NONE, confidence=1.0)
    )

    assert signal.action is RecoveryAction.NONE
    assert signal.reason == "No recovery guidance is needed."
    assert signal.confidence == 1.0
    assert signal.metadata == (("failure_type", "none"),)


def test_tool_failure_advises_retry_tool():
    signal = RecoveryAdvisor().advise(
        _diagnosis(FailureType.TOOL_FAILURE, confidence=0.6)
    )

    assert signal.action is RecoveryAction.RETRY_TOOL
    assert signal.confidence == 0.6
    assert signal.to_dict() == {
        "action": "retry_tool",
        "reason": (
            "Retry the failed tool operation if it is safe to do so."
        ),
        "confidence": 0.6,
        "metadata": {"failure_type": "tool_failure"},
    }


def test_evidence_failure_advises_collect_evidence():
    signal = RecoveryAdvisor().advise(
        _diagnosis(FailureType.EVIDENCE_FAILURE)
    )

    assert signal.action is RecoveryAction.COLLECT_EVIDENCE
    assert signal.reason == (
        "Collect additional execution evidence before proceeding."
    )


def test_unknown_failure_has_unknown_recovery():
    signal = RecoveryAdvisor().advise(
        _diagnosis(FailureType.UNKNOWN, confidence=0.0)
    )

    assert signal.action is RecoveryAction.UNKNOWN
    assert signal.confidence == 0.0
    assert signal.metadata == (("failure_type", "unknown"),)


@pytest.mark.parametrize(
    "failure_type, expected_action",
    [
        (
            FailureType.VERIFICATION_FAILURE,
            RecoveryAction.VERIFY_RESULT,
        ),
        (FailureType.STATE_FAILURE, RecoveryAction.INSPECT_STATE),
        (FailureType.PLANNING_FAILURE, RecoveryAction.REVIEW_PLAN),
        (FailureType.RECOVERY_FAILURE, RecoveryAction.UNKNOWN),
    ],
)
def test_remaining_taxonomy_mappings(failure_type, expected_action):
    signal = RecoveryAdvisor().advise(_diagnosis(failure_type))

    assert signal.action is expected_action
    assert signal.confidence == 0.75


def test_recovery_action_has_stable_values():
    assert [action.value for action in RecoveryAction] == [
        "none",
        "retry_tool",
        "verify_result",
        "inspect_state",
        "collect_evidence",
        "review_plan",
        "unknown",
    ]


def test_recovery_advisor_requires_diagnosis():
    with pytest.raises(TypeError, match="diagnosis must be DiagnosisResult"):
        RecoveryAdvisor().advise(object())
