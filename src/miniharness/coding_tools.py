import subprocess
from pathlib import Path

from miniharness.tools import Tool


def _resolve_workspace_path(
    workspace: Path,
    path: str,
) -> Path:
    workspace = workspace.resolve()
    target = (workspace / path).resolve()

    if target != workspace and workspace not in target.parents:
        raise ValueError(
            f"Path escapes workspace: {path}"
        )

    return target


def create_read_file_tool(workspace: Path) -> Tool:
    def read_file(path: str) -> str:
        target = _resolve_workspace_path(
            workspace,
            path,
        )

        return target.read_text(
            encoding="utf-8",
        )

    return Tool(
        name="read_file",
        description=(
            "Read a UTF-8 text file inside the workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path relative to the workspace."
                    ),
                },
            },
            "required": ["path"],
        },
        function=read_file,
    )


def create_write_file_tool(workspace: Path) -> Tool:
    def write_file(
        path: str,
        content: str,
    ) -> str:
        target = _resolve_workspace_path(
            workspace,
            path,
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            content,
            encoding="utf-8",
        )

        return f"Wrote {path}"

    return Tool(
        name="write_file",
        description=(
            "Write UTF-8 text to a file inside the workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path relative to the workspace."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Complete text content to write."
                    ),
                },
            },
            "required": [
                "path",
                "content",
            ],
        },
        function=write_file,
    )


def create_run_command_tool(
    workspace: Path,
    timeout_seconds: float = 10.0,
) -> Tool:
    def run_command(
        argv: list[str],
    ) -> str:
        if not argv:
            raise ValueError(
                "Command argv must not be empty."
            )

        result = subprocess.run(
            argv,
            cwd=workspace.resolve(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        return (
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return Tool(
        name="run_command",
        description=(
            "Run a command in the workspace and return "
            "its exit code, stdout, and stderr."
        ),
        parameters={
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": (
                        "Command and arguments as a list. "
                        'Example: ["pytest", "-q"].'
                    ),
                },
            },
            "required": ["argv"],
        },
        function=run_command,
    )


def create_coding_tools(
    workspace: Path,
) -> list[Tool]:
    return [
        create_read_file_tool(workspace),
        create_write_file_tool(workspace),
        create_run_command_tool(workspace),
    ]