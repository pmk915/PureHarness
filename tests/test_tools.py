import pytest

from miniharness.tools import ADD_TOOL, ToolRegistry


def test_add_tool_executes():
    result = ADD_TOOL.execute(
        {
            "a": 12,
            "b": 17,
        }
    )

    assert result == 29


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
