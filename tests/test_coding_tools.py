import subprocess
import sys

import pytest

from miniharness.coding_tools import (
    create_apply_patch_tool,
    create_coding_tools,
    create_git_diff_tool,
    create_git_status_tool,
    create_list_files_tool,
    create_read_file_tool,
    create_run_command_tool,
    create_search_text_tool,
    create_write_file_tool,
)
from miniharness.tools import RiskLevel


def test_coding_tools_have_expected_metadata(tmp_path):
    tools = create_coding_tools(tmp_path)

    assert [tool.name for tool in tools] == [
        "list_files",
        "search_text",
        "read_file",
        "write_file",
        "apply_patch",
        "run_command",
        "git_status",
        "git_diff",
    ]

    metadata = {
        tool.name: (
            tool.category,
            tool.risk_level,
            tool.side_effects,
        )
        for tool in tools
    }

    assert metadata == {
        "list_files": ("filesystem", RiskLevel.READ, False),
        "search_text": ("filesystem", RiskLevel.READ, False),
        "read_file": ("filesystem", RiskLevel.READ, False),
        "write_file": ("filesystem", RiskLevel.WRITE, True),
        "apply_patch": ("filesystem", RiskLevel.WRITE, True),
        "run_command": ("execution", RiskLevel.EXECUTE, True),
        "git_status": ("git", RiskLevel.READ, False),
        "git_diff": ("git", RiskLevel.READ, False),
    }


def test_list_files_returns_nested_deterministic_listing(
    tmp_path,
):
    (tmp_path / "src" / "package").mkdir(parents=True)
    (tmp_path / "src" / "z.py").write_text(
        "z = 1\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "a.py").write_text(
        "a = 1\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "package" / "module.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(
        "ignored",
        encoding="utf-8",
    )

    tool = create_list_files_tool(tmp_path)
    arguments = {
        "path": ".",
        "max_depth": 2,
    }

    first = tool.execute(arguments)
    second = tool.execute(arguments)

    assert first == second
    assert first.splitlines() == [
        "src/",
        "src/a.py",
        "src/package/",
        "src/package/module.py",
        "src/z.py",
    ]


def test_list_files_respects_depth_limit(tmp_path):
    (tmp_path / "src" / "package").mkdir(parents=True)
    (tmp_path / "src" / "package" / "module.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    tool = create_list_files_tool(tmp_path)
    result = tool.execute(
        {
            "max_depth": 0,
        }
    )

    assert result.splitlines() == ["src/"]


def test_list_files_respects_entry_limit(tmp_path):
    for name in ["c.py", "a.py", "b.py"]:
        (tmp_path / name).write_text(
            name,
            encoding="utf-8",
        )

    tool = create_list_files_tool(tmp_path)
    result = tool.execute(
        {
            "max_entries": 2,
        }
    )

    assert result.splitlines() == [
        "a.py",
        "b.py",
        "... truncated after 2 entries",
    ]


def test_list_files_rejects_path_outside_workspace(tmp_path):
    tool = create_list_files_tool(tmp_path)

    with pytest.raises(
        ValueError,
        match="Path escapes workspace",
    ):
        tool.execute(
            {
                "path": "../outside",
            }
        )


def test_search_text_finds_sorted_path_and_line_matches(
    tmp_path,
):
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.py").write_text(
        "ToolRegistry at z\n",
        encoding="utf-8",
    )
    (tmp_path / "a.py").write_text(
        "first line\nToolRegistry at a\n",
        encoding="utf-8",
    )
    (tmp_path / "nested" / "b.py").write_text(
        "ToolRegistry nested\n",
        encoding="utf-8",
    )

    tool = create_search_text_tool(tmp_path)
    result = tool.execute(
        {
            "query": "ToolRegistry",
        }
    )

    assert result.splitlines() == [
        "a.py:2: ToolRegistry at a",
        "nested/b.py:1: ToolRegistry nested",
        "z.py:1: ToolRegistry at z",
    ]


def test_search_text_respects_match_limit(tmp_path):
    (tmp_path / "matches.txt").write_text(
        "needle one\nneedle two\nneedle three\n",
        encoding="utf-8",
    )

    tool = create_search_text_tool(tmp_path)
    result = tool.execute(
        {
            "query": "needle",
            "max_matches": 2,
        }
    )

    assert result.splitlines() == [
        "matches.txt:1: needle one",
        "matches.txt:2: needle two",
        "... truncated after 2 matches",
    ]


def test_search_text_skips_binary_and_undecodable_files(
    tmp_path,
):
    (tmp_path / "binary.dat").write_bytes(
        b"needle\x00binary"
    )
    (tmp_path / "invalid.txt").write_bytes(
        b"needle\xff"
    )
    (tmp_path / "valid.txt").write_text(
        "needle text\n",
        encoding="utf-8",
    )

    tool = create_search_text_tool(tmp_path)
    result = tool.execute(
        {
            "query": "needle",
        }
    )

    assert result == "valid.txt:1: needle text"


def test_search_text_rejects_path_outside_workspace(tmp_path):
    tool = create_search_text_tool(tmp_path)

    with pytest.raises(
        ValueError,
        match="Path escapes workspace",
    ):
        tool.execute(
            {
                "query": "needle",
                "path": "../outside",
            }
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


def test_apply_patch_replaces_one_exact_match(tmp_path):
    target = tmp_path / "calculator.py"
    target.write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    tool = create_apply_patch_tool(tmp_path)

    result = tool.execute(
        {
            "path": "calculator.py",
            "old_text": "return a - b",
            "new_text": "return a + b",
        }
    )

    assert result == "Patched calculator.py"
    assert target.read_text(encoding="utf-8") == (
        "def add(a, b):\n    return a + b\n"
    )


def test_apply_patch_zero_matches_does_not_modify_file(
    tmp_path,
):
    target = tmp_path / "example.py"
    original = "value = 1\n"
    target.write_text(original, encoding="utf-8")
    tool = create_apply_patch_tool(tmp_path)

    with pytest.raises(
        ValueError,
        match="old_text was not found",
    ):
        tool.execute(
            {
                "path": "example.py",
                "old_text": "value = 2",
                "new_text": "value = 3",
            }
        )

    assert target.read_text(encoding="utf-8") == original


def test_apply_patch_multiple_matches_does_not_modify_file(
    tmp_path,
):
    target = tmp_path / "example.py"
    original = "value = 1\nvalue = 1\n"
    target.write_text(original, encoding="utf-8")
    tool = create_apply_patch_tool(tmp_path)

    with pytest.raises(
        ValueError,
        match="old_text matched 2 times",
    ):
        tool.execute(
            {
                "path": "example.py",
                "old_text": "value = 1",
                "new_text": "value = 2",
            }
        )

    assert target.read_text(encoding="utf-8") == original


def test_apply_patch_rejects_path_outside_workspace(
    tmp_path,
):
    tool = create_apply_patch_tool(tmp_path)

    with pytest.raises(
        ValueError,
        match="Path escapes workspace",
    ):
        tool.execute(
            {
                "path": "../outside.py",
                "old_text": "before",
                "new_text": "after",
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


def test_git_status_reports_local_worktree(tmp_path):
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "example.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )
    tool = create_git_status_tool(tmp_path)

    result = tool.execute({})

    assert "?? example.py" in result


def test_git_diff_reports_local_modification(tmp_path):
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
    )
    target = tmp_path / "example.py"
    target.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "example.py"],
        cwd=tmp_path,
        check=True,
    )
    target.write_text("value = 2\n", encoding="utf-8")
    tool = create_git_diff_tool(tmp_path)

    result = tool.execute(
        {
            "path": "example.py",
        }
    )

    assert "diff --git a/example.py b/example.py" in result
    assert "-value = 1" in result
    assert "+value = 2" in result


def test_git_diff_rejects_path_outside_workspace(tmp_path):
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
    )
    tool = create_git_diff_tool(tmp_path)

    with pytest.raises(
        ValueError,
        match="Path escapes workspace",
    ):
        tool.execute(
            {
                "path": "../outside.py",
            }
        )


@pytest.mark.parametrize(
    "factory",
    [
        create_git_status_tool,
        create_git_diff_tool,
    ],
)
def test_git_tools_fail_clearly_outside_repository(
    tmp_path,
    factory,
):
    tool = factory(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="Git command failed",
    ):
        tool.execute({})
