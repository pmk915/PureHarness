from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    arguments: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("tool_name must be non-empty text")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be a dictionary")
        object.__setattr__(self, "arguments", deepcopy(self.arguments))


class ApprovalHandler(Protocol):
    def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        ...


class AutoDenyApprovalHandler:
    def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        return ApprovalDecision.DENY


class AutoApproveApprovalHandler:
    def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        return ApprovalDecision.APPROVE


class ToolApprovalError(Exception):
    """Raised when an approval-gated tool call is not approved."""

    def __init__(self, tool_name: str, *, handler_configured: bool) -> None:
        self.tool_name = tool_name
        self.handler_configured = handler_configured
        if handler_configured:
            message = (
                f"Tool '{tool_name}' execution was denied during approval."
            )
        else:
            message = (
                f"Tool '{tool_name}' requires approval, but no approval "
                "handler was configured."
            )
        super().__init__(message)
