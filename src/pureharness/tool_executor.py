from collections.abc import Callable
from typing import Protocol

from pureharness.approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    ToolApprovalError,
)
from pureharness.tool_policy import (
    DefaultToolPolicy,
    PolicyDecision,
    ToolPolicy,
    ToolPolicyError,
)
from pureharness.tools import Tool, ToolRegistry


class ToolPreconditionError(Exception):
    """A tool's runtime state precondition was not satisfied."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        path: str | None = None,
        change: str | None = None,
    ) -> None:
        self.reason = reason
        self.path = path
        self.change = change
        super().__init__(message)


class ToolExecutionPrecondition(Protocol):
    """Optional run-scoped gate before policy and tool execution."""

    def reset(self) -> None:
        ...

    def prepare(
        self,
        tool: Tool,
        arguments: dict[str, object],
    ) -> object | None:
        ...

    def revalidate(
        self,
        tool: Tool,
        arguments: dict[str, object],
        prepared: object | None,
    ) -> object | None:
        ...

    def record_success(
        self,
        tool: Tool,
        arguments: dict[str, object],
        prepared: object | None,
        result: object,
    ) -> object | None:
        ...


class ToolExecutor:
    """Authorize and invoke registered tools through one execution path."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
        precondition: ToolExecutionPrecondition | None = None,
    ) -> None:
        self.registry = registry
        self.policy = (
            policy
            if policy is not None
            else DefaultToolPolicy()
        )
        self.approval_handler = approval_handler
        self.precondition = precondition

    def reset_run_state(self) -> None:
        self.registry.reset_run_state()
        if self.precondition is not None:
            self.precondition.reset()

    def cleanup_run_state(self) -> None:
        self.registry.cleanup_run_state()

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
        on_precondition_failed: (
            Callable[[Tool, ToolPreconditionError], None] | None
        ) = None,
        on_precondition_recorded: Callable[[object], None] | None = None,
    ) -> object:
        tool = self.registry.get(name)
        tool.validate_arguments(arguments)
        prepared: object | None = None
        if self.precondition is not None:
            try:
                prepared = self.precondition.prepare(tool, arguments)
            except ToolPreconditionError as exc:
                if on_precondition_failed is not None:
                    on_precondition_failed(tool, exc)
                raise

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

        if self.precondition is not None:
            try:
                prepared = self.precondition.revalidate(
                    tool,
                    arguments,
                    prepared,
                )
            except ToolPreconditionError as exc:
                if on_precondition_failed is not None:
                    on_precondition_failed(tool, exc)
                raise

        if on_tool_started is not None:
            on_tool_started(tool)

        result = tool.execute(arguments)
        if self.precondition is not None:
            evidence = self.precondition.record_success(
                tool,
                arguments,
                prepared,
                result,
            )
            if evidence is not None and on_precondition_recorded is not None:
                on_precondition_recorded(evidence)
        return result
