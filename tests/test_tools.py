import pytest

from miniharness.tools import (
    ADD_TOOL,
    RiskLevel,
    Tool,
    ToolRegistry,
)


def test_add_tool_executes():
    result = ADD_TOOL.execute(
        {
            "a": 12,
            "b": 17,
        }
    )

    assert result == 29


def test_risk_level_has_only_m2_values():
    assert list(RiskLevel) == [
        RiskLevel.READ,
        RiskLevel.WRITE,
        RiskLevel.EXECUTE,
        RiskLevel.DESTRUCTIVE,
    ]


def test_registry_executes_registered_tool():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    result = registry.execute(
        "add",
        {
            "a": 12,
            "b": 17,
        },
    )

    assert result == 29


def test_registry_rejects_unknown_tool():
    registry = ToolRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_tool_metadata_defaults_are_backward_compatible():
    tool = Tool(
        name="identity",
        description="Return the provided value.",
        parameters={
            "type": "object",
            "properties": {},
        },
        function=lambda: "ok",
    )

    assert tool.category == "general"
    assert tool.risk_level is RiskLevel.READ
    assert tool.side_effects is False


def test_registry_preserves_tool_metadata():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    listed = registry.list_tools()

    assert listed == [ADD_TOOL]
    assert listed[0].category == "utility"
    assert listed[0].risk_level is RiskLevel.READ
    assert listed[0].side_effects is False
