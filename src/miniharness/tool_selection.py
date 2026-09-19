import json

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from miniharness.task_state import TaskState
from miniharness.token_estimation import approximate_text_tokens
from miniharness.tools import Tool


class ToolSelectionError(RuntimeError):
    """Raised when tools cannot be selected for a model inference."""


class ToolNotExposedError(RuntimeError):
    """Raised when a model calls a tool absent from its exposed schemas."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"Tool '{tool_name}' was not exposed for this inference."
        )


@dataclass(frozen=True)
class ToolSelectionContext:
    step: int
    task_state: TaskState

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise ValueError("step must be a non-negative integer")


class ToolSelector(Protocol):
    def select(
        self,
        tools: Sequence[Tool],
        context: ToolSelectionContext,
    ) -> Sequence[Tool]:
        ...


class AllToolsSelector:
    """Expose every registered tool in registry order."""

    strategy = "AllTools"

    def select(
        self,
        tools: Sequence[Tool],
        context: ToolSelectionContext,
    ) -> Sequence[Tool]:
        del context
        return tuple(tools)


class StaticToolSelector:
    """Expose an explicit set of tool names in registry order."""

    strategy = "StaticNames"

    def __init__(self, tool_names: Iterable[str]) -> None:
        if isinstance(tool_names, str):
            raise ValueError(
                "tool_names must be an iterable of names, not one string"
            )

        names = tuple(tool_names)

        if any(not isinstance(name, str) for name in names):
            raise ValueError("tool_names must contain only strings")

        if len(set(names)) != len(names):
            raise ValueError("tool_names must not contain duplicates")

        self.tool_names = frozenset(names)

    def select(
        self,
        tools: Sequence[Tool],
        context: ToolSelectionContext,
    ) -> Sequence[Tool]:
        del context
        registered_names = {tool.name for tool in tools}
        unknown_names = sorted(self.tool_names - registered_names)

        if unknown_names:
            names = ", ".join(repr(name) for name in unknown_names)
            raise ToolSelectionError(
                f"StaticToolSelector references unknown tool(s): {names}."
            )

        return tuple(
            tool
            for tool in tools
            if tool.name in self.tool_names
        )


@dataclass(frozen=True)
class ToolSelection:
    tools: tuple[Tool, ...]
    registered_tool_count: int
    exposed_tool_count: int
    estimated_tool_schema_tokens: int
    estimated_all_tool_schema_tokens: int
    estimated_tool_schema_tokens_saved: int
    selector_strategy: str


def prepare_tool_selection(
    registered_tools: Sequence[Tool],
    selector: ToolSelector,
    context: ToolSelectionContext,
) -> ToolSelection:
    """Select, validate, normalize, and measure model-facing tools."""
    registered = tuple(registered_tools)

    try:
        selected_output = selector.select(registered, context)
        selected = tuple(selected_output)
    except ToolSelectionError:
        raise
    except Exception as exc:
        raise ToolSelectionError(
            "Tool selector failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    registered_by_name = {
        tool.name: tool
        for tool in registered
    }
    selected_names: set[str] = set()

    for tool in selected:
        if not isinstance(tool, Tool):
            raise ToolSelectionError(
                "ToolSelector must return registered Tool instances."
            )

        if tool.name in selected_names:
            raise ToolSelectionError(
                f"ToolSelector returned duplicate tool {tool.name!r}."
            )

        registered_tool = registered_by_name.get(tool.name)
        if registered_tool is not tool:
            raise ToolSelectionError(
                "ToolSelector returned an unregistered Tool instance: "
                f"{tool.name!r}."
            )

        selected_names.add(tool.name)

    normalized = tuple(
        tool
        for tool in registered
        if tool.name in selected_names
    )
    all_schema_tokens = estimate_tool_schema_tokens(registered)
    selected_schema_tokens = estimate_tool_schema_tokens(normalized)
    strategy = getattr(selector, "strategy", type(selector).__name__)

    if not isinstance(strategy, str) or not strategy:
        raise ToolSelectionError(
            "ToolSelector strategy must be non-empty text."
        )

    return ToolSelection(
        tools=normalized,
        registered_tool_count=len(registered),
        exposed_tool_count=len(normalized),
        estimated_tool_schema_tokens=selected_schema_tokens,
        estimated_all_tool_schema_tokens=all_schema_tokens,
        estimated_tool_schema_tokens_saved=(
            all_schema_tokens - selected_schema_tokens
        ),
        selector_strategy=strategy,
    )


def estimate_tool_schema_tokens(tools: Sequence[Tool]) -> int:
    """Estimate complete model-facing tool schemas deterministically."""
    if not tools:
        return 0

    schemas = [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in tools
    ]

    try:
        serialized = json.dumps(
            schemas,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ToolSelectionError(
            "Tool schemas must be deterministically JSON serializable."
        ) from exc

    return approximate_text_tokens(serialized)
