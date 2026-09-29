import pytest

from pureharness.tools import (
    ADD_TOOL,
    RiskLevel,
    Tool,
    ToolArgumentError,
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


def test_registry_gets_registered_tool():
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    assert registry.get("add") is ADD_TOOL


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


def test_registry_deduplicates_and_manages_shared_run_resources():
    calls = []

    class Resource:
        def reset_run_state(self):
            calls.append("reset")

        def cleanup_run_state(self):
            calls.append("cleanup")

    resource = Resource()
    registry = ToolRegistry()
    for name in ["one", "two"]:
        registry.register(
            Tool(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
                function=lambda: None,
                run_resource=resource,
            )
        )

    registry.reset_run_state()
    registry.cleanup_run_state()

    assert calls == ["reset", "cleanup"]


def test_tool_preserves_explicit_object_schema_contract():
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "additionalProperties": True,
    }

    tool = Tool(
        name="identity",
        description="Return a value.",
        parameters=parameters,
        function=lambda value: value,
    )

    assert tool.parameters["additionalProperties"] is True
    assert parameters["additionalProperties"] is True


def test_tool_rejects_unknown_arguments_with_deterministic_feedback():
    tool = Tool(
        name="inspect",
        description="Inspect a path.",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "path": {"type": "string"},
                "max_depth": {"type": "integer"},
            },
            "required": ["path"],
        },
        function=lambda path, max_depth=1: (path, max_depth),
    )

    with pytest.raises(
        ToolArgumentError,
        match=(
            r"Tool 'inspect' received unknown argument 'surprise'\. "
            r"Allowed arguments: max_depth, path\."
        ),
    ):
        tool.execute({"path": ".", "surprise": True})


def test_tool_rejects_missing_required_arguments_deterministically():
    with pytest.raises(
        ToolArgumentError,
        match=r"Tool 'add' is missing required arguments 'a', 'b'\.",
    ):
        ADD_TOOL.execute({})
