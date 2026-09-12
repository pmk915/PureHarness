import sys

import pytest

from miniharness.coding_tools import (
    create_read_file_tool,
    create_run_command_tool,
    create_write_file_tool,
)


def test_read_file_tool_reads_workspace_file(tmp_path):
    target = tmp_path / "hello.txt"
    target.write_text(
        "hello MiniHarness",
        encoding="utf-8",
    )

    tool = create_read_file_tool(tmp_path)

    result = tool.execute(
        {
            "path": "hello.txt",
        }
    )

    assert result == "hello MiniHarness"


def test_write_file_tool_writes_workspace_file(tmp_path):
    tool = create_write_file_tool(tmp_path)

    result = tool.execute(
        {
            "path": "src/example.py",
            "content": "value = 42\n",
        }
    )

    target = tmp_path / "src" / "example.py"

    assert result == "Wrote src/example.py"
    assert target.read_text(
        encoding="utf-8",
    ) == "value = 42\n"


def test_read_file_rejects_path_outside_workspace(
    tmp_path,
):
    tool = create_read_file_tool(tmp_path)

    with pytest.raises(
        ValueError,
        match="Path escapes workspace",
    ):
        tool.execute(
            {
                "path": "../outside.txt",
            }
        )


def test_write_file_rejects_path_outside_workspace(
    tmp_path,
):
    tool = create_write_file_tool(tmp_path)

    with pytest.raises(
        ValueError,
        match="Path escapes workspace",
    ):
        tool.execute(
            {
                "path": "../outside.txt",
                "content": "nope",
            }
        )


def test_run_command_tool_executes_in_workspace(
    tmp_path,
):
    tool = create_run_command_tool(tmp_path)

    result = tool.execute(
        {
            "argv": [
                sys.executable,
                "-c",
                "print('hello from command')",
            ],
        }
    )

    assert "exit_code: 0" in result
    assert "hello from command" in result