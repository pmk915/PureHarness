import hashlib

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pureharness.tool_executor import ToolPreconditionError
from pureharness.tools import Tool


def resolve_workspace_path(workspace: Path, path: str) -> Path:
    """Resolve one path under a workspace using coding-tool semantics."""
    workspace_root = workspace.resolve()
    target = (workspace_root / path).resolve()

    if target != workspace_root and workspace_root not in target.parents:
        raise ValueError(f"Path escapes workspace: {path}")

    return target


def workspace_relative_path(workspace: Path, target: Path) -> str:
    relative = target.relative_to(workspace.resolve())
    if relative == Path("."):
        return "."
    return relative.as_posix()


class ReadBeforeEditError(ToolPreconditionError):
    """An existing structured-edit target was not read in this Run."""

    def __init__(self, path: str) -> None:
        super().__init__(
            "Existing file must be read successfully before modification: "
            f"{path}",
            reason="read_required",
            path=path,
        )


class StaleFileError(ToolPreconditionError):
    """An observed or intended structured-edit target changed state."""

    def __init__(self, path: str, change: str) -> None:
        super().__init__(
            "Workspace target changed since it was observed or prepared; "
            f"read/refresh before modification: {path}",
            reason="stale_observation",
            path=path,
            change=change,
        )


@dataclass(frozen=True)
class _PreparedWorkspaceAction:
    operation: Literal["observe", "create", "overwrite", "patch"]
    canonical_path: str
    expected_fingerprint: str | None = None
    current_text: str | None = None


@dataclass(frozen=True)
class WorkspaceMutation:
    """One successful structured workspace mutation in execution order."""

    path: str
    tool_name: str
    operation: Literal["created", "overwritten", "patched"]


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Immutable, non-persisted diagnostics for the current Agent Run."""

    observed_paths: tuple[str, ...]
    modified_paths: tuple[str, ...]
    mutations: tuple[WorkspaceMutation, ...]
    read_required_block_count: int
    stale_block_count: int


class WorkspaceDiscipline:
    """Enforce Run-scoped prior observation for structured file edits."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._observations: dict[str, str] = {}
        self._mutations: list[WorkspaceMutation] = []
        self._modified_paths: list[str] = []
        self._modified_path_set: set[str] = set()
        self._read_required_block_count = 0
        self._stale_block_count = 0

    @property
    def observed_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._observations))

    @property
    def snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(
            observed_paths=self.observed_paths,
            modified_paths=tuple(self._modified_paths),
            mutations=tuple(self._mutations),
            read_required_block_count=self._read_required_block_count,
            stale_block_count=self._stale_block_count,
        )

    def reset(self) -> None:
        self._observations.clear()
        self._mutations.clear()
        self._modified_paths.clear()
        self._modified_path_set.clear()
        self._read_required_block_count = 0
        self._stale_block_count = 0

    def canonicalize(self, path: str) -> str:
        return workspace_relative_path(
            self.workspace,
            resolve_workspace_path(self.workspace, path),
        )

    def mark_observed(self, path: str, content: str) -> None:
        """Record the exact model-visible text returned by a successful read."""
        self._observations[self.canonicalize(path)] = _fingerprint_text(content)

    def is_observed(self, path: str) -> bool:
        return self.canonicalize(path) in self._observations

    def require_observed(self, path: str) -> None:
        canonical_path = self.canonicalize(path)
        self._require_canonical_observed(canonical_path)

    def prepare(
        self,
        tool: Tool,
        arguments: dict[str, object],
    ) -> object | None:
        if tool.name not in {"read_file", "write_file", "apply_patch"}:
            return None

        path = arguments.get("path")
        if not isinstance(path, str) or not path:
            return None

        target = resolve_workspace_path(self.workspace, path)
        canonical_path = workspace_relative_path(self.workspace, target)

        if tool.name == "read_file":
            return _PreparedWorkspaceAction("observe", canonical_path)

        if tool.name == "apply_patch":
            fingerprint, content = self._require_fresh_observation(
                target,
                canonical_path,
            )
            return _PreparedWorkspaceAction(
                "patch",
                canonical_path,
                fingerprint,
                content,
            )

        if canonical_path in self._observations:
            fingerprint, content = self._require_fresh_observation(
                target,
                canonical_path,
            )
            return _PreparedWorkspaceAction(
                "overwrite",
                canonical_path,
                fingerprint,
                content,
            )

        if target.is_file():
            self._require_canonical_observed(canonical_path)

        if target.exists():
            return None

        return _PreparedWorkspaceAction("create", canonical_path)

    def revalidate(
        self,
        tool: Tool,
        arguments: dict[str, object],
        prepared: object | None,
    ) -> object | None:
        del tool, arguments
        if not isinstance(prepared, _PreparedWorkspaceAction):
            return prepared
        if prepared.operation == "observe":
            return prepared

        target = resolve_workspace_path(
            self.workspace,
            prepared.canonical_path,
        )
        if prepared.operation == "create":
            if target.exists():
                self._raise_stale(
                    prepared.canonical_path,
                    "target_appeared",
                )
            return prepared

        current_text = self._read_current_text(
            target,
            prepared.canonical_path,
        )
        current_fingerprint = _fingerprint_text(current_text)
        if current_fingerprint != prepared.expected_fingerprint:
            self._raise_stale(
                prepared.canonical_path,
                "content_changed",
            )
        return _PreparedWorkspaceAction(
            prepared.operation,
            prepared.canonical_path,
            current_fingerprint,
            current_text,
        )

    def record_success(
        self,
        tool: Tool,
        arguments: dict[str, object],
        prepared: object | None,
        result: object,
    ) -> object | None:
        if not isinstance(prepared, _PreparedWorkspaceAction):
            return None

        if prepared.operation == "observe":
            if isinstance(result, str):
                self._observations[prepared.canonical_path] = (
                    _fingerprint_text(result)
                )
            return None

        resulting_text = _resulting_text(arguments, prepared)
        if resulting_text is not None:
            self._observations[prepared.canonical_path] = (
                _fingerprint_text(resulting_text)
            )
        else:
            self._observations.pop(prepared.canonical_path, None)

        operation = {
            "create": "created",
            "overwrite": "overwritten",
            "patch": "patched",
        }.get(prepared.operation)
        if operation is None:
            return None
        mutation = WorkspaceMutation(
            path=prepared.canonical_path,
            tool_name=tool.name,
            operation=operation,
        )
        self._mutations.append(mutation)
        if prepared.canonical_path not in self._modified_path_set:
            self._modified_path_set.add(prepared.canonical_path)
            self._modified_paths.append(prepared.canonical_path)
        return mutation

    def _require_canonical_observed(self, canonical_path: str) -> None:
        if canonical_path not in self._observations:
            self._read_required_block_count += 1
            raise ReadBeforeEditError(canonical_path)

    def _require_fresh_observation(
        self,
        target: Path,
        canonical_path: str,
    ) -> tuple[str, str]:
        self._require_canonical_observed(canonical_path)
        current_text = self._read_current_text(target, canonical_path)
        current_fingerprint = _fingerprint_text(current_text)
        expected_fingerprint = self._observations[canonical_path]
        if current_fingerprint != expected_fingerprint:
            self._raise_stale(canonical_path, "content_changed")
        return current_fingerprint, current_text

    def _read_current_text(
        self,
        target: Path,
        canonical_path: str,
    ) -> str:
        if not target.is_file():
            self._raise_stale(canonical_path, "target_missing")
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self._raise_stale(canonical_path, "content_changed")

    def _raise_stale(self, canonical_path: str, change: str) -> None:
        self._observations.pop(canonical_path, None)
        self._stale_block_count += 1
        raise StaleFileError(canonical_path, change)


def _fingerprint_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _resulting_text(
    arguments: dict[str, object],
    prepared: _PreparedWorkspaceAction,
) -> str | None:
    if prepared.operation in {"create", "overwrite"}:
        content = arguments.get("content")
        return content if isinstance(content, str) else None

    if prepared.operation == "patch" and prepared.current_text is not None:
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        if isinstance(old_text, str) and isinstance(new_text, str):
            return prepared.current_text.replace(old_text, new_text, 1)

    return None
