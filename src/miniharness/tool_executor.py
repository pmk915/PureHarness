from collections.abc import Callable

from miniharness.approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    ToolApprovalError,
)
from miniharness.tool_policy import (
    DefaultToolPolicy,
    PolicyDecision,
    ToolPolicy,
    ToolPolicyError,
)
from miniharness.tools import Tool, ToolRegistry


class ToolExecutor:
    """Authorize and invoke registered tools through one execution path."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.registry = registry
        self.policy = (
            policy
            if policy is not None
            else DefaultToolPolicy()
        )
        self.approval_handler = approval_handler

    def execute(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        on_policy_evaluated: (
            Callable[[Tool, PolicyDecision], None] | None
        ) = None,
        on_approval_requested: (
            Callable[[Tool, ApprovalRequest], None] | None
        ) = None,
        on_approval_resolved: (
            Callable[
                [Tool, ApprovalRequest, ApprovalDecision],
                None,
            ]
            | None
        ) = None,
        on_tool_started: Callable[[Tool], None] | None = None,
    ) -> object:
        tool = self.registry.get(name)
        decision = self.policy.evaluate(tool, arguments)

        if not isinstance(decision, PolicyDecision):
            raise TypeError("ToolPolicy must return PolicyDecision")

        if on_policy_evaluated is not None:
            on_policy_evaluated(tool, decision)

        if decision is PolicyDecision.DENY:
            raise ToolPolicyError(tool.name, decision)

        if decision is PolicyDecision.REQUIRE_APPROVAL:
            request = ApprovalRequest(
                tool_name=tool.name,
                arguments=arguments,
            )
            if on_approval_requested is not None:
                on_approval_requested(tool, request)

            handler_configured = self.approval_handler is not None
            approval_decision = (
                ApprovalDecision.DENY
                if self.approval_handler is None
                else self.approval_handler.request_approval(request)
            )
            if not isinstance(approval_decision, ApprovalDecision):
                raise TypeError(
                    "ApprovalHandler must return ApprovalDecision"
                )
            if on_approval_resolved is not None:
                on_approval_resolved(
                    tool,
                    request,
                    approval_decision,
                )
            if approval_decision is ApprovalDecision.DENY:
                raise ToolApprovalError(
                    tool.name,
                    handler_configured=handler_configured,
                )

        if on_tool_started is not None:
            on_tool_started(tool)

        return tool.execute(arguments)
