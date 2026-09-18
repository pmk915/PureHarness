from collections.abc import Iterator
from pathlib import Path

from miniharness.execution import (
    ExecutionBackend,
    LocalExecutionBackend,
)
from miniharness.tools import RiskLevel, Tool


_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
_DEFAULT_LIST_DEPTH = 3
_DEFAULT_LIST_ENTRIES = 200
_MAX_LIST_DEPTH = 8
_MAX_LIST_ENTRIES = 500
_DEFAULT_SEARCH_MATCHES = 50
_MAX_SEARCH_MATCHES = 200
_MAX_SEARCH_FILE_BYTES = 1_000_000
_MATCH_PREVIEW_LENGTH = 160
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0


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


def _relative_path(
    workspace: Path,
    target: Path,
) -> str:
    relative = target.relative_to(workspace.resolve())

    if relative == Path("."):
        return "."

    return relative.as_posix()


def _validate_bounded_integer(
    name: str,
    value: int,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")

    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}."
        )


def _iter_workspace_files(directory: Path) -> Iterator[Path]:
    for child in sorted(
        directory.iterdir(),
        key=lambda item: item.name,
    ):
        if child.name in _IGNORED_DIRECTORY_NAMES:
            continue

        if child.is_symlink():
            continue

        if child.is_dir():
            yield from _iter_workspace_files(child)
        elif child.is_file():
            yield child


def _run_git(
    workspace: Path,
    arguments: list[str],
    execution_backend: ExecutionBackend,
) -> str:
    result = execution_backend.execute(
        ["git", *arguments],
        cwd=workspace.resolve(),
        timeout=_DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )

    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Git command failed with exit code "
            f"{result.exit_code}: {detail}"
        )

    return result.stdout.rstrip()


def create_list_files_tool(workspace: Path) -> Tool:
    def list_files(
        path: str = ".",
        max_depth: int = _DEFAULT_LIST_DEPTH,
        max_entries: int = _DEFAULT_LIST_ENTRIES,
    ) -> str:
        _validate_bounded_integer(
            "max_depth",
            max_depth,
            minimum=0,
            maximum=_MAX_LIST_DEPTH,
        )
        _validate_bounded_integer(
            "max_entries",
            max_entries,
            minimum=1,
            maximum=_MAX_LIST_ENTRIES,
        )

        workspace_root = workspace.resolve()
        target = _resolve_workspace_path(workspace_root, path)

        if not target.is_dir():
            raise NotADirectoryError(
                f"Not a directory inside workspace: {path}"
            )

        entries: list[str] = []
        truncated = False

        def visit(directory: Path, depth: int) -> None:
            nonlocal truncated

            for child in sorted(
                directory.iterdir(),
                key=lambda item: item.name,
            ):
                if child.name in _IGNORED_DIRECTORY_NAMES:
                    continue

                if child.is_symlink():
                    continue

                if len(entries) >= max_entries:
                    truncated = True
                    return

                relative = _relative_path(
                    workspace_root,
                    child,
                )

                if child.is_dir():
                    entries.append(f"{relative}/")

                    if depth < max_depth:
                        visit(child, depth + 1)

                        if truncated:
                            return
                elif child.is_file():
                    entries.append(relative)

        visit(target, 0)

        if not entries:
            return "(no files)"

        if truncated:
            entries.append(
                f"... truncated after {max_entries} entries"
            )

        return "\n".join(entries)

    return Tool(
        name="list_files",
        description=(
            "List files and directories under a workspace-relative directory "
            "for repository discovery. Results are sorted and bounded by "
            "depth and entry count; noisy internal directories and symlinks "
            "are skipped. Use search_text to locate known text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory to list.",
                    "default": ".",
                },
                "max_depth": {
                    "type": "integer",
                    "description": (
                        "Maximum recursion depth from the selected directory."
                    ),
                    "minimum": 0,
                    "maximum": _MAX_LIST_DEPTH,
                    "default": _DEFAULT_LIST_DEPTH,
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Maximum number of returned entries.",
                    "minimum": 1,
                    "maximum": _MAX_LIST_ENTRIES,
                    "default": _DEFAULT_LIST_ENTRIES,
                },
            },
            "required": [],
        },
        function=list_files,
        category="filesystem",
        risk_level=RiskLevel.READ,
        side_effects=False,
    )


def create_search_text_tool(workspace: Path) -> Tool:
    def search_text(
        query: str,
        path: str = ".",
        max_matches: int = _DEFAULT_SEARCH_MATCHES,
    ) -> str:
        if not query:
            raise ValueError("Search query must not be empty.")

        _validate_bounded_integer(
            "max_matches",
            max_matches,
            minimum=1,
            maximum=_MAX_SEARCH_MATCHES,
        )

        workspace_root = workspace.resolve()
        target = _resolve_workspace_path(workspace_root, path)

        if target.is_file():
            candidates: Iterator[Path] = iter([target])
        elif target.is_dir():
            candidates = _iter_workspace_files(target)
        else:
            raise FileNotFoundError(
                f"Search path does not exist: {path}"
            )

        matches: list[str] = []
        truncated = False

        for candidate in candidates:
            try:
                if candidate.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                    continue

                content_bytes = candidate.read_bytes()

                if b"\x00" in content_bytes:
                    continue

                content = content_bytes.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            relative = _relative_path(
                workspace_root,
                candidate,
            )

            for line_number, line in enumerate(
                content.splitlines(),
                start=1,
            ):
                if query not in line:
                    continue

                if len(matches) >= max_matches:
                    truncated = True
                    break

                matches.append(
                    f"{relative}:{line_number}: "
                    f"{_matching_line_preview(line, query)}"
                )

            if truncated:
                break

        if not matches:
            return f"No matches for {query!r}."

        if truncated:
            matches.append(
                f"... truncated after {max_matches} matches"
            )

        return "\n".join(matches)

    return Tool(
        name="search_text",
        description=(
            "Search UTF-8 text files recursively for a case-sensitive literal "
            "string and return sorted path:line previews. Results are bounded; "
            "binary, large, undecodable, noisy-directory, and symlink content "
            "is skipped. Use read_file to inspect a known match in full."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Case-sensitive literal text to find.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative file or directory to search."
                    ),
                    "default": ".",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum number of returned matches.",
                    "minimum": 1,
                    "maximum": _MAX_SEARCH_MATCHES,
                    "default": _DEFAULT_SEARCH_MATCHES,
                },
            },
            "required": ["query"],
        },
        function=search_text,
        category="filesystem",
        risk_level=RiskLevel.READ,
        side_effects=False,
    )


def _matching_line_preview(
    line: str,
    query: str,
) -> str:
    line = line.strip()

    if len(line) <= _MATCH_PREVIEW_LENGTH:
        return line

    match_index = line.find(query)
    start = max(0, match_index - 60)
    end = min(
        len(line),
        start + _MATCH_PREVIEW_LENGTH,
    )
    preview = line[start:end]

    if start > 0:
        preview = "…" + preview[1:]

    if end < len(line):
        preview = preview[:-1] + "…"

    return preview


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
            "Read a known UTF-8 text file inside the workspace. The path must "
            "stay inside the workspace. Use list_files for discovery or "
            "search_text to locate text across files."
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
        category="filesystem",
        risk_level=RiskLevel.READ,
        side_effects=False,
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
            "Create or fully rewrite a UTF-8 file inside the workspace. Use it "
            "for new files or complete replacements; prefer apply_patch for a "
            "small targeted change to an existing file."
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
        category="filesystem",
        risk_level=RiskLevel.WRITE,
        side_effects=True,
    )


def create_apply_patch_tool(workspace: Path) -> Tool:
    def apply_patch(
        path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        if not old_text:
            raise ValueError("old_text must not be empty.")

        target = _resolve_workspace_path(
            workspace,
            path,
        )
        content = target.read_text(encoding="utf-8")
        match_count = content.count(old_text)

        if match_count == 0:
            raise ValueError(
                f"old_text was not found in {path}; file was not changed."
            )

        if match_count > 1:
            raise ValueError(
                f"old_text matched {match_count} times in {path}; "
                "file was not changed."
            )

        target.write_text(
            content.replace(old_text, new_text, 1),
            encoding="utf-8",
        )

        return f"Patched {path}"

    return Tool(
        name="apply_patch",
        description=(
            "Replace one exact, unique text block in an existing UTF-8 "
            "workspace file. The edit fails without writing if old_text has "
            "zero or multiple matches. Use write_file for full rewrites."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text that must occur once.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
        function=apply_patch,
        category="filesystem",
        risk_level=RiskLevel.WRITE,
        side_effects=True,
    )


def create_run_command_tool(
    workspace: Path,
    timeout_seconds: float = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
    *,
    execution_backend: ExecutionBackend | None = None,
) -> Tool:
    backend = (
        execution_backend
        if execution_backend is not None
        else LocalExecutionBackend()
    )

    def run_command(
        argv: list[str],
    ) -> str:
        if not argv:
            raise ValueError(
                "Command argv must not be empty."
            )

        result = backend.execute(
            argv,
            cwd=workspace.resolve(),
            timeout=timeout_seconds,
        )

        return (
            f"exit_code: {result.exit_code}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return Tool(
        name="run_command",
        description=(
            "Run an argv command through the configured execution backend "
            "with the workspace as its current directory, returning exit "
            "code, stdout, and stderr. Use it for tests and validation. The "
            "default local backend has a timeout and is not a secure sandbox."
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
        category="execution",
        risk_level=RiskLevel.EXECUTE,
        side_effects=True,
    )


def create_git_status_tool(
    workspace: Path,
    *,
    execution_backend: ExecutionBackend | None = None,
) -> Tool:
    backend = (
        execution_backend
        if execution_backend is not None
        else LocalExecutionBackend()
    )

    def git_status() -> str:
        output = _run_git(
            workspace,
            ["status", "--short"],
            backend,
        )

        return output or "Working tree clean."

    return Tool(
        name="git_status",
        description=(
            "Inspect the local repository working tree with fixed read-only "
            "git status --short semantics. Use it to find changed or untracked "
            "files; use git_diff to review unstaged content changes."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=git_status,
        category="git",
        risk_level=RiskLevel.READ,
        side_effects=False,
    )


def create_git_diff_tool(
    workspace: Path,
    *,
    execution_backend: ExecutionBackend | None = None,
) -> Tool:
    backend = (
        execution_backend
        if execution_backend is not None
        else LocalExecutionBackend()
    )

    def git_diff(path: str | None = None) -> str:
        arguments = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
        ]

        if path is not None:
            target = _resolve_workspace_path(workspace, path)
            relative = _relative_path(workspace, target)
            arguments.extend(["--", relative])

        output = _run_git(
            workspace,
            arguments,
            backend,
        )

        return output or "No unstaged changes."

    return Tool(
        name="git_diff",
        description=(
            "Review local unstaged changes with fixed read-only git diff "
            "semantics, optionally limited to one workspace-relative path. "
            "Use git_status first to discover changed files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Optional workspace-relative path to constrain the diff."
                    ),
                },
            },
            "required": [],
        },
        function=git_diff,
        category="git",
        risk_level=RiskLevel.READ,
        side_effects=False,
    )


def create_coding_tools(
    workspace: Path,
    *,
    execution_backend: ExecutionBackend | None = None,
) -> list[Tool]:
    backend = (
        execution_backend
        if execution_backend is not None
        else LocalExecutionBackend()
    )

    return [
        create_list_files_tool(workspace),
        create_search_text_tool(workspace),
        create_read_file_tool(workspace),
        create_write_file_tool(workspace),
        create_apply_patch_tool(workspace),
        create_run_command_tool(
            workspace,
            execution_backend=backend,
        ),
        create_git_status_tool(
            workspace,
            execution_backend=backend,
        ),
        create_git_diff_tool(
            workspace,
            execution_backend=backend,
        ),
    ]
