import miniharness


def test_package_import():
    assert miniharness.__version__ == "0.1.0"


from miniharness.messages import ToolCall


def test_tool_call():
    tool_call = ToolCall(
        name="add",
        arguments={
            "a": 12,
            "b": 17,
        },
    )

    assert tool_call.name == "add"
    assert tool_call.arguments["a"] == 12
    assert tool_call.arguments["b"] == 17