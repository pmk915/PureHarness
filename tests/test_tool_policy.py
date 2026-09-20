from pathlib import Path

import pytest

from pureharness.approval import ToolApprovalError
from pureharness.coding_tools import create_run_command_tool
from pureharness.execution import CommandResult
from pureharness.tool_executor import ToolExecutor
from pureharness.tool_policy import (
    DefaultToolPolicy,
    PolicyDecision,
    ToolPolicyError,
)
from pureharness.tools import RiskLevel, Tool, ToolRegistry


class StaticPolicy:
    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[Tool, dict[str, object]]] = []

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, object],
    ) -> PolicyDecision:
        self.calls.append((tool, arguments))
        return self.decision


class FakeExecutionBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        argv: list[str],
        *,
        cwd: Path,
        timeout: float,
    ) -> CommandResult:
        self.calls.append(
            {
                "argv": argv,
                "cwd": cwd,
                "timeout": timeout,
            }
        )
        return CommandResult(0, "", "")


def _tool(
    function,
    *,
    risk_level: RiskLevel = RiskLevel.READ,
) -> Tool:
    return Tool(
        name="test_tool",
        description="A deterministic test tool.",
        parameters={"type": "object", "properties": {}},
        function=function,
        risk_level=risk_level,
    )


@pytest.mark.parametrize(
    ("risk_level", "expected"),
    [
        (RiskLevel.READ, PolicyDecision.ALLOW),
        (RiskLevel.WRITE, PolicyDecision.ALLOW),
        (RiskLevel.EXECUTE, PolicyDecision.ALLOW),
        (RiskLevel.DESTRUCTIVE, PolicyDecision.DENY),
    ],
)
def test_default_policy_maps_risk_levels(
    risk_level,
    expected,
):
    tool = _tool(lambda: None, risk_level=risk_level)

    assert (
        DefaultToolPolicy().evaluate(tool, {})
        is expected
    )


def test_executor_preserves_missing_tool_lookup_failure():
    executor = ToolExecutor(ToolRegistry())

    with pytest.raises(KeyError, match="Unknown tool: missing"):
        executor.execute("missing", {})


def test_executor_evaluates_policy_before_allowed_tool():
    calls = []

    class RecordingPolicy:
        def evaluate(self, tool, arguments):
            calls.append("policy")
            return PolicyDecision.ALLOW

    def function():
        calls.append("tool")
        return {"unchanged": True}

    registry = ToolRegistry()
    registry.register(_tool(function))
    executor = ToolExecutor(registry, RecordingPolicy())

    result = executor.execute("test_tool", {})

    assert result == {"unchanged": True}
    assert calls == ["policy", "tool"]


@pytest.mark.parametrize(
    ("decision", "error_type", "message"),
    [
        (
            PolicyDecision.DENY,
            ToolPolicyError,
            "Tool 'test_tool' was denied by policy.",
        ),
        (
            PolicyDecision.REQUIRE_APPROVAL,
            ToolApprovalError,
            "Tool 'test_tool' requires approval, but no approval ",
        ),
    ],
)
def test_executor_does_not_call_unauthorized_tool(
    decision,
    error_type,
    message,
):
    call_count = 0

    def function():
        nonlocal call_count
        call_count += 1

    registry = ToolRegistry()
    registry.register(_tool(function))
    policy = StaticPolicy(decision)
    executor = ToolExecutor(registry, policy)

    with pytest.raises(error_type, match=message):
        executor.execute("test_tool", {})

    assert call_count == 0
    assert len(policy.calls) == 1


def test_executor_propagates_tool_exception():
    def function():
        raise ValueError("tool failed")

    registry = ToolRegistry()
    registry.register(_tool(function))
    executor = ToolExecutor(registry)

    with pytest.raises(ValueError, match="tool failed"):
        executor.execute("test_tool", {})


@pytest.mark.parametrize(
    "decision",
    [
        PolicyDecision.DENY,
        PolicyDecision.REQUIRE_APPROVAL,
    ],
)
def test_unauthorized_command_does_not_call_backend(
    tmp_path,
    decision,
):
    backend = FakeExecutionBackend()
    tool = create_run_command_tool(
        tmp_path,
        execution_backend=backend,
    )
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry, StaticPolicy(decision))

    error_type = (
        ToolPolicyError
        if decision is PolicyDecision.DENY
        else ToolApprovalError
    )
    with pytest.raises(error_type):
        executor.execute(
            "run_command",
            {"argv": ["python", "-V"]},
        )

    assert backend.calls == []
