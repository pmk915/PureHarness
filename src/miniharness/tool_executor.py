from collections.abc import Callable

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
    ) -> None:
        self.registry = registry
        self.policy = (
            policy
            if policy is not None
            else DefaultToolPolicy()
        )

    def execute(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        on_policy_evaluated: (
            Callable[[Tool, PolicyDecision], None] | None
        ) = None,
        on_tool_started: Callable[[Tool], None] | None = None,
    ) -> object:
        tool = self.registry.get(name)
        decision = self.policy.evaluate(tool, arguments)

        if on_policy_evaluated is not None:
            on_policy_evaluated(tool, decision)

        if decision is not PolicyDecision.ALLOW:
            raise ToolPolicyError(tool.name, decision)

        if on_tool_started is not None:
            on_tool_started(tool)

        return tool.execute(arguments)
