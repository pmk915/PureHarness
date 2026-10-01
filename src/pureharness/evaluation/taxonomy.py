from enum import Enum


class FailureType(str, Enum):
    """Stable trajectory-level failure taxonomy."""

    NONE = "none"
    TOOL_FAILURE = "tool_failure"
    VERIFICATION_FAILURE = "verification_failure"
    STATE_FAILURE = "state_failure"
    PLANNING_FAILURE = "planning_failure"
    EVIDENCE_FAILURE = "evidence_failure"
    RECOVERY_FAILURE = "recovery_failure"
    UNKNOWN = "unknown"
