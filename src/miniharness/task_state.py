import posixpath

from collections.abc import Sequence
from dataclasses import dataclass

from miniharness.messages import AgentItem, Message, ToolCall, ToolResult
from miniharness.tool_history import (
    ToolHistoryError,
    ToolInteraction,
    match_tool_interactions,
)


DEFAULT_MAX_CURRENT_REQUEST_CHARS = 4_000
DEFAULT_MAX_COMPLETED_ACTIONS = 20
DEFAULT_MAX_FAILED_ACTIONS = 10
DEFAULT_MAX_RECENT_ERRORS = 5
DEFAULT_MAX_ERROR_CHARS = 400

_FILE_READ_TOOLS = frozenset({"read_file"})
_FILE_MODIFICATION_TOOLS = frozenset(
    {
        "write_file",
        "apply_patch",
    }
)


class TaskStateError(RuntimeError):
    """Raised when TaskState cannot be derived from raw history."""


@dataclass(frozen=True)
class TaskAction:
    tool_name: str
    call_id: str | None


@dataclass(frozen=True)
class RecentTaskError:
    tool_name: str
    call_id: str | None
    message: str


@dataclass(frozen=True)
class TaskState:
    current_request: str | None
    completed_actions: tuple[TaskAction, ...]
    failed_actions: tuple[TaskAction, ...]
    files_read: tuple[str, ...]
    files_modified: tuple[str, ...]
    recent_errors: tuple[RecentTaskError, ...]


class TaskStateReducer:
    """Derive bounded current working state from complete raw history."""

    def __init__(
        self,
        *,
        max_current_request_chars: int = (
            DEFAULT_MAX_CURRENT_REQUEST_CHARS
        ),
        max_completed_actions: int = DEFAULT_MAX_COMPLETED_ACTIONS,
        max_failed_actions: int = DEFAULT_MAX_FAILED_ACTIONS,
        max_recent_errors: int = DEFAULT_MAX_RECENT_ERRORS,
        max_error_chars: int = DEFAULT_MAX_ERROR_CHARS,
    ) -> None:
        _validate_text_limit(
            "max_current_request_chars",
            max_current_request_chars,
        )
        _validate_positive_integer(
            "max_completed_actions",
            max_completed_actions,
        )
        _validate_positive_integer(
            "max_failed_actions",
            max_failed_actions,
        )
        _validate_positive_integer(
            "max_recent_errors",
            max_recent_errors,
        )
        _validate_text_limit(
            "max_error_chars",
            max_error_chars,
        )

        self.max_current_request_chars = max_current_request_chars
        self.max_completed_actions = max_completed_actions
        self.max_failed_actions = max_failed_actions
        self.max_recent_errors = max_recent_errors
        self.max_error_chars = max_error_chars

    def reduce(
        self,
        items: Sequence[AgentItem],
    ) -> TaskState:
        current_request = next(
            (
                item.content
                for item in reversed(items)
                if isinstance(item, Message)
                and item.role == "user"
            ),
            None,
        )
        if current_request is not None:
            current_request = _bounded_text(
                current_request,
                self.max_current_request_chars,
            )

        try:
            interactions = _match_history_interactions(items)
        except ToolHistoryError as exc:
            raise TaskStateError(str(exc)) from exc

        completed_actions: list[TaskAction] = []
        failed_actions: list[TaskAction] = []
        recent_errors: list[RecentTaskError] = []
        files_read: list[str] = []
        files_modified: list[str] = []
        seen_files_read: set[str] = set()
        seen_files_modified: set[str] = set()

        for interaction in interactions:
            action = TaskAction(
                tool_name=interaction.call.name,
                call_id=interaction.call.call_id,
            )

            if interaction.result.is_error:
                failed_actions.append(action)
                recent_errors.append(
                    RecentTaskError(
                        tool_name=interaction.call.name,
                        call_id=interaction.call.call_id,
                        message=_error_summary(
                            interaction.result.content,
                            self.max_error_chars,
                        ),
                    )
                )
                continue

            completed_actions.append(action)
            path = _structured_path(interaction.call)

            if (
                path is not None
                and interaction.call.name in _FILE_READ_TOOLS
            ):
                _append_unique(path, files_read, seen_files_read)

            if (
                path is not None
                and interaction.call.name
                in _FILE_MODIFICATION_TOOLS
            ):
                _append_unique(
                    path,
                    files_modified,
                    seen_files_modified,
                )

        return TaskState(
            current_request=current_request,
            completed_actions=tuple(
                completed_actions[-self.max_completed_actions :]
            ),
            failed_actions=tuple(
                failed_actions[-self.max_failed_actions :]
            ),
            files_read=tuple(files_read),
            files_modified=tuple(files_modified),
            recent_errors=tuple(
                recent_errors[-self.max_recent_errors :]
            ),
        )


def render_task_state(state: TaskState) -> Message:
    """Render a deterministic, model-facing system context item."""
    sections = [
        "[MiniHarness Derived Task State]",
        "Current request:\n"
        + (
            state.current_request
            if state.current_request is not None
            else "(none)"
        ),
        "Files read:\n" + _render_values(state.files_read),
        "Files modified:\n" + _render_values(state.files_modified),
        "Recent completed actions:\n"
        + _render_actions(state.completed_actions),
        "Recent failed actions:\n"
        + _render_actions(state.failed_actions),
        "Recent tool errors:\n"
        + _render_errors(state.recent_errors),
    ]

    return Message(
        role="system",
        content="\n\n".join(sections),
    )


def _match_history_interactions(
    items: Sequence[AgentItem],
) -> tuple[ToolInteraction, ...]:
    interactions: list[ToolInteraction] = []
    tool_items: list[ToolCall | ToolResult] = []

    for item in items:
        if isinstance(item, Message):
            if tool_items:
                interactions.extend(match_tool_interactions(tool_items))
                tool_items = []
            continue

        tool_items.append(item)

    if tool_items:
        interactions.extend(match_tool_interactions(tool_items))

    return tuple(interactions)


def _structured_path(call: ToolCall) -> str | None:
    path = call.arguments.get("path")

    if not isinstance(path, str) or not path:
        return None

    return posixpath.normpath(path)


def _append_unique(
    value: str,
    values: list[str],
    seen: set[str],
) -> None:
    if value in seen:
        return

    seen.add(value)
    values.append(value)


def _error_summary(
    content: str,
    max_chars: int,
) -> str:
    normalized = " ".join(content.split())

    return _bounded_text(
        normalized or "(no error text)",
        max_chars,
    )


def _bounded_text(
    value: str,
    max_chars: int,
) -> str:
    if len(value) <= max_chars:
        return value

    marker = (
        "\n... text compacted "
        f"({len(value)} original characters; middle omitted) ...\n"
    )
    available_chars = max_chars - len(marker)
    head_chars = (available_chars * 2) // 3
    tail_chars = available_chars - head_chars

    return value[:head_chars] + marker + value[-tail_chars:]


def _render_values(values: Sequence[str]) -> str:
    if not values:
        return "- (none)"

    return "\n".join(f"- {value}" for value in values)


def _render_actions(actions: Sequence[TaskAction]) -> str:
    if not actions:
        return "- (none)"

    return "\n".join(
        f"- {_action_identity(action.tool_name, action.call_id)}"
        for action in actions
    )


def _render_errors(errors: Sequence[RecentTaskError]) -> str:
    if not errors:
        return "- (none)"

    return "\n".join(
        f"- {_action_identity(error.tool_name, error.call_id)}: "
        f"{error.message}"
        for error in errors
    )


def _action_identity(
    tool_name: str,
    call_id: str | None,
) -> str:
    if call_id is None:
        return tool_name

    return f"{tool_name} (call_id={call_id})"


def _validate_positive_integer(
    name: str,
    value: int,
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _validate_text_limit(
    name: str,
    value: int,
) -> None:
    _validate_positive_integer(name, value)

    if value < 80:
        raise ValueError(f"{name} must be at least 80")
