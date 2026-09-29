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


@dataclass(frozen=True)
class _PreparedWorkspaceAction:
    operation: Literal["observe", "mutate"]
    canonical_path: str


class WorkspaceDiscipline:
    """Enforce Run-scoped prior observation for structured file edits."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._observed_paths: set[str] = set()

    @property
    def observed_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._observed_paths))

    def reset(self) -> None:
        self._observed_paths.clear()

    def canonicalize(self, path: str) -> str:
        return workspace_relative_path(
            self.workspace,
            resolve_workspace_path(self.workspace, path),
        )

    def mark_observed(self, path: str) -> None:
        self._observed_paths.add(self.canonicalize(path))

    def is_observed(self, path: str) -> bool:
        return self.canonicalize(path) in self._observed_paths

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

        if tool.name == "apply_patch" or (
            tool.name == "write_file" and target.is_file()
        ):
            self._require_canonical_observed(canonical_path)

        return _PreparedWorkspaceAction("mutate", canonical_path)

    def record_success(
        self,
        tool: Tool,
        arguments: dict[str, object],
        prepared: object | None,
    ) -> None:
        del tool, arguments
        if isinstance(prepared, _PreparedWorkspaceAction):
            self._observed_paths.add(prepared.canonical_path)

    def _require_canonical_observed(self, canonical_path: str) -> None:
        if canonical_path not in self._observed_paths:
            raise ReadBeforeEditError(canonical_path)
