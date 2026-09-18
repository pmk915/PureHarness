from enum import Enum
from typing import Protocol

from miniharness.tools import RiskLevel, Tool


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ToolPolicy(Protocol):
    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, object],
    ) -> PolicyDecision:
        ...


class DefaultToolPolicy:
    """Authorize tool capability classes using their declared risk level."""

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, object],
    ) -> PolicyDecision:
        decisions = {
            RiskLevel.READ: PolicyDecision.ALLOW,
            RiskLevel.WRITE: PolicyDecision.ALLOW,
            RiskLevel.EXECUTE: PolicyDecision.ALLOW,
            RiskLevel.DESTRUCTIVE: PolicyDecision.DENY,
        }

        return decisions[tool.risk_level]


class ToolPolicyError(Exception):
    """Raised when policy does not authorize a tool invocation."""

    def __init__(
        self,
        tool_name: str,
        decision: PolicyDecision,
    ) -> None:
        self.tool_name = tool_name
        self.decision = decision

        if decision is PolicyDecision.DENY:
            message = f"Tool '{tool_name}' was denied by policy."
        elif decision is PolicyDecision.REQUIRE_APPROVAL:
            message = (
                f"Tool '{tool_name}' requires approval and was not executed."
            )
        else:
            message = (
                f"Tool '{tool_name}' was not authorized by policy."
            )

        super().__init__(message)
