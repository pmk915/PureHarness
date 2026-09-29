import re

from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from pureharness.execution import (
    ExecutionBackend,
    LocalExecutionBackend,
)
from pureharness.processes import (
    LocalProcessManager,
    ProcessManager,
    ProcessObservation,
    UnavailableProcessManager,
)
from pureharness.tools import RiskLevel, Tool
from pureharness.workspace_discipline import (
    resolve_workspace_path as _resolve_workspace_path,
    workspace_relative_path as _relative_path,
)


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
_DEFAULT_RANGE_LINES = 200
_MAX_RANGE_LINES = 500
_DEFAULT_FIND_MATCHES = 100
_MAX_FIND_MATCHES = 500
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0
_MIN_COMMAND_TIMEOUT_SECONDS = 1.0
_MAX_COMMAND_TIMEOUT_SECONDS = 120.0
_MAX_COMMAND_STREAM_CHARS = 20_000
_COMMAND_STREAM_HEAD_CHARS = 10_000
_COMMAND_STREAM_TAIL_CHARS = 10_000


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


def _validate_bounded_number(
    name: str,
    value: float,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be between {minimum:g} and {maximum:g}."
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
        regex: bool = False,
        case_sensitive: bool = True,
        file_glob: str | None = None,
    ) -> str:
        if not query:
            raise ValueError("Search query must not be empty.")
        if not isinstance(regex, bool):
            raise ValueError("regex must be bool.")
        if not isinstance(case_sensitive, bool):
            raise ValueError("case_sensitive must be bool.")
        if file_glob is not None and not file_glob:
            raise ValueError("file_glob must not be empty when provided.")

        _validate_bounded_integer(
            "max_matches",
            max_matches,
            minimum=1,
            maximum=_MAX_SEARCH_MATCHES,
        )

        workspace_root = workspace.resolve()
        target = _resolve_workspace_path(workspace_root, path)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags) if regex else None
        except re.error as exc:
            raise ValueError(f"Invalid search regex: {exc}") from exc

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
            relative = _relative_path(
                workspace_root,
                candidate,
            )
            if file_glob is not None and not _glob_matches(
                relative,
                file_glob,
            ):
                continue
            try:
                if candidate.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                    continue

                content_bytes = candidate.read_bytes()

                if b"\x00" in content_bytes:
                    continue

                content = content_bytes.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            for line_number, line in enumerate(
                content.splitlines(),
                start=1,
            ):
                match_start = _search_match_start(
                    line,
                    query,
                    pattern=pattern,
                    case_sensitive=case_sensitive,
                )
                if match_start is None:
                    continue

                if len(matches) >= max_matches:
                    truncated = True
                    break

                matches.append(
                    f"{relative}:{line_number}: "
                    f"{_matching_line_preview(line, match_start)}"
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
            "Search UTF-8 text files recursively using literal or regular-"
            "expression matching and return sorted path:line previews. Literal, "
            "case-sensitive matching remains the default. Results are bounded; "
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
                "regex": {
                    "type": "boolean",
                    "description": "Interpret query as a regular expression.",
                    "default": False,
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Use case-sensitive matching.",
                    "default": True,
                },
                "file_glob": {
                    "type": "string",
                    "description": (
                        "Optional glob matched against workspace-relative files."
                    ),
                },
            },
            "required": ["query"],
        },
        function=search_text,
        category="filesystem",
        risk_level=RiskLevel.READ,
        side_effects=False,
    )


def _search_match_start(
    line: str,
    query: str,
    *,
    pattern: re.Pattern[str] | None,
    case_sensitive: bool,
) -> int | None:
    if pattern is not None:
        match = pattern.search(line)
        return None if match is None else match.start()

    if case_sensitive:
        index = line.find(query)
    else:
        index = line.casefold().find(query.casefold())
    return None if index < 0 else index


def _matching_line_preview(line: str, match_start: int) -> str:
    leading_characters = len(line) - len(line.lstrip())
    line = line.strip()
    match_index = max(0, match_start - leading_characters)

    if len(line) <= _MATCH_PREVIEW_LENGTH:
        return line

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


def _glob_matches(path: str, pattern: str) -> bool:
    variants = {pattern}
    pending = [pattern]
    while pending:
        candidate = pending.pop()
        marker = candidate.find("**/")
        if marker < 0:
            continue
        without_recursive_directory = (
            candidate[:marker] + candidate[marker + 3 :]
        )
        if without_recursive_directory not in variants:
            variants.add(without_recursive_directory)
            pending.append(without_recursive_directory)
    relative = PurePosixPath(path)
    return any(relative.match(candidate) for candidate in variants)


def create_find_files_tool(workspace: Path) -> Tool:
    def find_files(
        pattern: str,
        path: str = ".",
        max_matches: int = _DEFAULT_FIND_MATCHES,
    ) -> str:
        if not pattern:
            raise ValueError("File pattern must not be empty.")
        _validate_bounded_integer(
            "max_matches",
            max_matches,
            minimum=1,
            maximum=_MAX_FIND_MATCHES,
        )
        workspace_root = workspace.resolve()
        target = _resolve_workspace_path(workspace_root, path)
        if not target.is_dir():
            raise NotADirectoryError(
                f"Find path is not a directory inside workspace: {path}"
            )

        matches: list[str] = []
        truncated = False
        for candidate in _iter_workspace_files(target):
            relative = _relative_path(workspace_root, candidate)
            if not _glob_matches(relative, pattern):
                continue
            if len(matches) >= max_matches:
                truncated = True
                break
            matches.append(relative)

        if not matches:
            return f"No files matching {pattern!r}."
        if truncated:
            matches.append(
                f"... truncated after {max_matches} matches"
            )
        return "\n".join(matches)

    return Tool(
        name="find_files",
        description=(
            "Find workspace files by glob pattern with deterministic, bounded "
            "results. Recursive patterns such as **/*.py are supported; noisy "
            "directories and symlinks are skipped."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for workspace-relative files.",
                },
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory to search.",
                    "default": ".",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum number of paths to return.",
                    "minimum": 1,
                    "maximum": _MAX_FIND_MATCHES,
                    "default": _DEFAULT_FIND_MATCHES,
                },
            },
            "required": ["pattern"],
        },
        function=find_files,
        category="filesystem",
        risk_level=RiskLevel.READ,
        side_effects=False,
    )


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


def create_read_file_range_tool(workspace: Path) -> Tool:
    def read_file_range(
        path: str,
        start_line: int = 1,
        max_lines: int = _DEFAULT_RANGE_LINES,
    ) -> str:
        _validate_bounded_integer(
            "start_line",
            start_line,
            minimum=1,
            maximum=2_147_483_647,
        )
        _validate_bounded_integer(
            "max_lines",
            max_lines,
            minimum=1,
            maximum=_MAX_RANGE_LINES,
        )
        workspace_root = workspace.resolve()
        target = _resolve_workspace_path(workspace_root, path)
        content = target.read_text(encoding="utf-8")
        lines = content.splitlines()
        total_lines = len(lines)
        start_index = start_line - 1
        selected = lines[start_index : start_index + max_lines]
        canonical_path = _relative_path(workspace_root, target)
        if not selected:
            return (
                f"{canonical_path} lines 0-0 of {total_lines}\n\n"
                "(no lines in requested range)"
            )

        end_line = start_line + len(selected) - 1
        output = [
            f"{canonical_path} lines {start_line}-{end_line} "
            f"of {total_lines}",
            "",
            *[
                f"{line_number} | {line}"
                for line_number, line in enumerate(
                    selected,
                    start=start_line,
                )
            ],
        ]
        if end_line < total_lines:
            output.append(
                f"... truncated; {total_lines - end_line} lines remain"
            )
        return "\n".join(output)

    return Tool(
        name="read_file_range",
        description=(
            "Inspect a bounded line range of one UTF-8 workspace file with "
            "line numbers. This partial view does not satisfy read-before-edit; "
            "use read_file before modifying an existing file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file to inspect.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "One-based first line to return.",
                    "minimum": 1,
                    "default": 1,
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return.",
                    "minimum": 1,
                    "maximum": _MAX_RANGE_LINES,
                    "default": _DEFAULT_RANGE_LINES,
                },
            },
            "required": ["path"],
        },
        function=read_file_range,
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
            "for new files or complete replacements. Existing files must be "
            "read successfully with read_file earlier in the current run; "
            "prefer apply_patch for a small targeted change."
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
            "workspace file after reading that file successfully with "
            "read_file in the current run. The edit fails without writing if "
            "old_text has zero or multiple matches. Use write_file for full "
            "rewrites."
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
    default_timeout_seconds = timeout_seconds

    def run_command(
        argv: list[str],
        cwd: str = ".",
        timeout_seconds: float = default_timeout_seconds,
    ) -> str:
        if not argv:
            raise ValueError(
                "Command argv must not be empty."
            )
        _validate_bounded_number(
            "timeout_seconds",
            timeout_seconds,
            minimum=_MIN_COMMAND_TIMEOUT_SECONDS,
            maximum=_MAX_COMMAND_TIMEOUT_SECONDS,
        )
        command_cwd = _resolve_workspace_path(workspace, cwd)
        if not command_cwd.is_dir():
            raise NotADirectoryError(
                f"Command cwd is not a directory inside workspace: {cwd}"
            )

        result = backend.execute(
            argv,
            cwd=command_cwd,
            timeout=timeout_seconds,
        )

        return (
            f"exit_code: {result.exit_code}\n"
            f"stdout:\n{_bound_command_stream(result.stdout, 'stdout')}\n"
            f"stderr:\n{_bound_command_stream(result.stderr, 'stderr')}"
        )

    return Tool(
        name="run_command",
        description=(
            "Run an argv command through the configured execution backend "
            "with a workspace-relative current directory and bounded timeout, "
            "returning exit code, stdout, and stderr. Use it for tests and "
            "validation. The default local backend is not a secure sandbox."
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
                "cwd": {
                    "type": "string",
                    "description": (
                        "Workspace-relative directory in which to run."
                    ),
                    "default": ".",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Per-call timeout in seconds.",
                    "minimum": _MIN_COMMAND_TIMEOUT_SECONDS,
                    "maximum": _MAX_COMMAND_TIMEOUT_SECONDS,
                    "default": default_timeout_seconds,
                },
            },
            "required": ["argv"],
        },
        function=run_command,
        category="execution",
        risk_level=RiskLevel.EXECUTE,
        side_effects=True,
    )


def _bound_command_stream(value: str, stream_name: str) -> str:
    if len(value) <= _MAX_COMMAND_STREAM_CHARS:
        return value

    omitted = len(value) - (
        _COMMAND_STREAM_HEAD_CHARS + _COMMAND_STREAM_TAIL_CHARS
    )
    notice = (
        f"\n... {stream_name} truncated; original character count: "
        f"{len(value)}; omitted: {omitted} ...\n"
    )
    return (
        value[:_COMMAND_STREAM_HEAD_CHARS]
        + notice
        + value[-_COMMAND_STREAM_TAIL_CHARS:]
    )


def create_process_tools(
    workspace: Path,
    *,
    process_manager: ProcessManager | None = None,
) -> list[Tool]:
    manager = (
        process_manager
        if process_manager is not None
        else LocalProcessManager(workspace)
    )

    def start_process(
        argv: list[str],
        cwd: str = ".",
    ) -> str:
        observation = manager.start(argv, cwd=Path(cwd))
        return _format_process_observation(
            observation,
            include_output=False,
        )

    def poll_process(job_id: str) -> str:
        return _format_process_observation(manager.poll(job_id))

    def stop_process(job_id: str) -> str:
        return _format_process_observation(manager.stop(job_id))

    return [
        Tool(
            name="start_process",
            description=(
                "Start a bounded background process from an argv list in a "
                "workspace-relative directory. Returns an opaque job ID for "
                "poll_process or stop_process. This is not a shell or PTY."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command and arguments as a list.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Workspace-relative directory in which to start."
                        ),
                        "default": ".",
                    },
                },
                "required": ["argv"],
            },
            function=start_process,
            category="process",
            risk_level=RiskLevel.EXECUTE,
            side_effects=True,
            run_resource=manager,
        ),
        Tool(
            name="poll_process",
            description=(
                "Poll a Run-scoped background job and return its status, exit "
                "code when available, and bounded output produced since the "
                "previous poll."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Opaque ID returned by start_process.",
                    },
                },
                "required": ["job_id"],
            },
            function=poll_process,
            category="process",
            risk_level=RiskLevel.READ,
            side_effects=False,
            run_resource=manager,
        ),
        Tool(
            name="stop_process",
            description=(
                "Stop a Run-scoped background job with graceful termination "
                "followed by forced cleanup if needed, then return final "
                "status and remaining bounded output."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Opaque ID returned by start_process.",
                    },
                },
                "required": ["job_id"],
            },
            function=stop_process,
            category="process",
            risk_level=RiskLevel.EXECUTE,
            side_effects=True,
            run_resource=manager,
        ),
    ]


def _format_process_observation(
    observation: ProcessObservation,
    *,
    include_output: bool = True,
) -> str:
    fields = [
        f"job_id: {observation.job_id}",
        f"status: {observation.status}",
    ]
    if observation.exit_code is not None:
        fields.append(f"exit_code: {observation.exit_code}")
    if include_output:
        fields.extend(
            [
                f"stdout:\n{observation.stdout}",
                f"stderr:\n{observation.stderr}",
            ]
        )
    return "\n".join(fields)


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
    process_manager: ProcessManager | None = None,
) -> list[Tool]:
    backend = (
        execution_backend
        if execution_backend is not None
        else LocalExecutionBackend()
    )

    manager = process_manager
    if manager is None:
        manager = (
            LocalProcessManager(workspace)
            if isinstance(backend, LocalExecutionBackend)
            else UnavailableProcessManager()
        )

    return [
        create_list_files_tool(workspace),
        create_find_files_tool(workspace),
        create_search_text_tool(workspace),
        create_read_file_range_tool(workspace),
        create_read_file_tool(workspace),
        create_write_file_tool(workspace),
        create_apply_patch_tool(workspace),
        create_run_command_tool(
            workspace,
            execution_backend=backend,
        ),
        *create_process_tools(
            workspace,
            process_manager=manager,
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
