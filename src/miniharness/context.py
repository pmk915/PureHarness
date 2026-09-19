import json

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from miniharness.messages import (
    AgentItem,
    Message,
    ToolCall,
    ToolResult,
)
from miniharness.tool_result_projection import (
    DeterministicToolResultProjector,
    ToolResultProjector,
)


class ContextCompileError(RuntimeError):
    """Raised when history cannot be compiled safely."""


class ContextBudgetExceeded(ContextCompileError):
    """Raised when the newest atomic unit cannot fit the history budget."""


@dataclass(frozen=True)
class ContextUnit:
    items: tuple[AgentItem, ...]


class TokenEstimator(Protocol):
    def estimate(
        self,
        items: Sequence[AgentItem],
    ) -> int:
        ...


class ApproximateTokenEstimator:
    """Estimate history tokens from deterministic serialized character size."""

    def estimate(
        self,
        items: Sequence[AgentItem],
    ) -> int:
        if not items:
            return 0

        serialized = json.dumps(
            [_serialize_item(item) for item in items],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

        return max(1, (len(serialized) + 3) // 4)


@dataclass(frozen=True)
class ContextBudget:
    max_estimated_tokens: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_estimated_tokens, int)
            or isinstance(self.max_estimated_tokens, bool)
            or self.max_estimated_tokens <= 0
        ):
            raise ValueError(
                "max_estimated_tokens must be greater than 0"
            )


@dataclass
class CompiledContext:
    items: list[AgentItem]
    estimated_tokens: int
    total_units: int
    included_units: int
    dropped_units: int
    strategy: str
    history_token_budget: int | None = None
    projected_tool_results: int = 0
    compacted_tool_results: int = 0
    raw_tool_result_chars: int = 0
    projected_tool_result_chars: int = 0


@dataclass(frozen=True)
class _UnitProjectionStats:
    projected_tool_results: int = 0
    compacted_tool_results: int = 0
    raw_tool_result_chars: int = 0
    projected_tool_result_chars: int = 0


class ContextBuilder:
    strategy = "FullHistory"

    def __init__(
        self,
        token_estimator: TokenEstimator | None = None,
        tool_result_projector: ToolResultProjector | None = None,
    ) -> None:
        self.token_estimator = (
            token_estimator
            if token_estimator is not None
            else ApproximateTokenEstimator()
        )
        self.tool_result_projector = (
            tool_result_projector
            if tool_result_projector is not None
            else DeterministicToolResultProjector()
        )

    def _prepare(
        self,
        history: Sequence[AgentItem],
    ) -> tuple[
        list[ContextUnit],
        list[int],
        list[_UnitProjectionStats],
    ]:
        raw_units = _group_context_units(history)
        units, projection_stats = _project_context_units(
            raw_units,
            self.tool_result_projector,
        )
        costs = _estimate_units(units, self.token_estimator)

        return units, costs, projection_stats

    def compile(
        self,
        history: Sequence[AgentItem],
    ) -> CompiledContext:
        units, costs, projection_stats = self._prepare(history)

        return _compiled_context(
            units,
            costs,
            projection_stats,
            start=0,
            strategy=self.strategy,
        )

    def build(
        self,
        history: list[AgentItem],
    ) -> list[AgentItem]:
        """Compatibility view over the canonical compile path."""
        return self.compile(history).items


class RecentContextBuilder(ContextBuilder):
    strategy = "Recent"

    def __init__(
        self,
        max_items: int = 20,
        token_estimator: TokenEstimator | None = None,
        tool_result_projector: ToolResultProjector | None = None,
    ) -> None:
        if max_items <= 0:
            raise ValueError(
                "max_items must be greater than 0"
            )

        super().__init__(
            token_estimator,
            tool_result_projector,
        )
        self.max_items = max_items

    def compile(
        self,
        history: Sequence[AgentItem],
    ) -> CompiledContext:
        units, costs, projection_stats = self._prepare(history)
        total_items = sum(len(unit.items) for unit in units)

        if total_items <= self.max_items:
            start = 0
        else:
            start = len(units)
            included_items = 0

            while start > 0 and included_items < self.max_items:
                start -= 1
                included_items += len(units[start].items)

            while start > 0 and not _is_user_message(units[start]):
                start -= 1

        return _compiled_context(
            units,
            costs,
            projection_stats,
            start=start,
            strategy=self.strategy,
        )


class TokenBudgetContextBuilder(ContextBuilder):
    strategy = "TokenBudget"

    def __init__(
        self,
        budget: ContextBudget,
        token_estimator: TokenEstimator | None = None,
        tool_result_projector: ToolResultProjector | None = None,
    ) -> None:
        super().__init__(
            token_estimator,
            tool_result_projector,
        )
        self.budget = budget

    def compile(
        self,
        history: Sequence[AgentItem],
    ) -> CompiledContext:
        units, costs, projection_stats = self._prepare(history)
        start = len(units)
        estimated_tokens = 0

        while start > 0:
            cost = costs[start - 1]

            if (
                estimated_tokens + cost
                > self.budget.max_estimated_tokens
            ):
                if start == len(units):
                    raise ContextBudgetExceeded(
                        "Newest indivisible context unit requires "
                        f"{cost} estimated history tokens, exceeding "
                        "the budget of "
                        f"{self.budget.max_estimated_tokens}."
                    )

                break

            start -= 1
            estimated_tokens += cost

        return _compiled_context(
            units,
            costs,
            projection_stats,
            start=start,
            strategy=self.strategy,
            history_token_budget=(
                self.budget.max_estimated_tokens
            ),
        )


def _group_context_units(
    history: Sequence[AgentItem],
) -> list[ContextUnit]:
    units: list[ContextUnit] = []
    tool_items: list[ToolCall | ToolResult] = []

    for item in history:
        if isinstance(item, Message):
            if tool_items:
                units.append(_tool_context_unit(tool_items))
                tool_items = []

            units.append(ContextUnit(items=(item,)))
        else:
            tool_items.append(item)

    if tool_items:
        units.append(_tool_context_unit(tool_items))

    return units


def _tool_context_unit(
    items: Sequence[ToolCall | ToolResult],
) -> ContextUnit:
    calls: list[ToolCall] = []
    matched_call_indexes: set[int] = set()
    call_ids: set[str] = set()

    for item in items:
        if isinstance(item, ToolCall):
            if item.call_id is not None:
                if item.call_id in call_ids:
                    raise ContextCompileError(
                        "Tool history contains duplicate ToolCall "
                        f"call_id {item.call_id!r}."
                    )

                call_ids.add(item.call_id)

            calls.append(item)
            continue

        match = next(
            (
                index
                for index, call in enumerate(calls)
                if index not in matched_call_indexes
                and _tool_result_matches(call, item)
            ),
            None,
        )

        if match is None:
            raise ContextCompileError(
                "ToolResult has no matching ToolCall in its "
                f"execution group: {item.name!r}."
            )

        matched_call_indexes.add(match)

    return ContextUnit(items=tuple(items))


def _tool_result_matches(
    call: ToolCall,
    result: ToolResult,
) -> bool:
    if call.name != result.name:
        return False

    if call.call_id is None or result.call_id is None:
        return True

    return call.call_id == result.call_id


def _estimate_units(
    units: Sequence[ContextUnit],
    token_estimator: TokenEstimator,
) -> list[int]:
    costs = []

    for unit in units:
        cost = token_estimator.estimate(unit.items)

        if (
            not isinstance(cost, int)
            or isinstance(cost, bool)
            or cost < 0
        ):
            raise ContextCompileError(
                "TokenEstimator must return a non-negative integer."
            )

        costs.append(cost)

    return costs


def _project_context_units(
    units: Sequence[ContextUnit],
    projector: ToolResultProjector,
) -> tuple[list[ContextUnit], list[_UnitProjectionStats]]:
    projected_units = []
    unit_stats = []

    for unit in units:
        projected_items: list[AgentItem] = []
        projected_tool_results = 0
        compacted_tool_results = 0
        raw_tool_result_chars = 0
        projected_tool_result_chars = 0

        for item in unit.items:
            if not isinstance(item, ToolResult):
                projected_items.append(item)
                continue

            projection_input = ToolResult(
                name=item.name,
                content=item.content,
                call_id=item.call_id,
                is_error=item.is_error,
            )
            projected = projector.project(projection_input)
            _validate_projected_tool_result(item, projected)
            projected_items.append(projected)
            projected_tool_results += 1
            raw_tool_result_chars += len(item.content)
            projected_tool_result_chars += len(
                projected.content
            )

            if projected.content != item.content:
                compacted_tool_results += 1

        projected_units.append(
            ContextUnit(items=tuple(projected_items))
        )
        unit_stats.append(
            _UnitProjectionStats(
                projected_tool_results=projected_tool_results,
                compacted_tool_results=compacted_tool_results,
                raw_tool_result_chars=raw_tool_result_chars,
                projected_tool_result_chars=(
                    projected_tool_result_chars
                ),
            )
        )

    return projected_units, unit_stats


def _validate_projected_tool_result(
    raw: ToolResult,
    projected: ToolResult,
) -> None:
    if not isinstance(projected, ToolResult):
        raise ContextCompileError(
            "ToolResultProjector must return a ToolResult."
        )

    if (
        projected.name != raw.name
        or projected.call_id != raw.call_id
        or projected.is_error is not raw.is_error
    ):
        raise ContextCompileError(
            "ToolResultProjector must preserve name, call_id, "
            "and is_error."
        )

    if not isinstance(projected.content, str):
        raise ContextCompileError(
            "Projected ToolResult content must be text."
        )


def _compiled_context(
    units: Sequence[ContextUnit],
    costs: Sequence[int],
    projection_stats: Sequence[_UnitProjectionStats],
    *,
    start: int,
    strategy: str,
    history_token_budget: int | None = None,
) -> CompiledContext:
    selected_units = units[start:]
    selected_stats = projection_stats[start:]

    return CompiledContext(
        items=[
            item
            for unit in selected_units
            for item in unit.items
        ],
        estimated_tokens=sum(costs[start:]),
        total_units=len(units),
        included_units=len(selected_units),
        dropped_units=start,
        strategy=strategy,
        projected_tool_results=sum(
            stats.projected_tool_results
            for stats in selected_stats
        ),
        compacted_tool_results=sum(
            stats.compacted_tool_results
            for stats in selected_stats
        ),
        raw_tool_result_chars=sum(
            stats.raw_tool_result_chars
            for stats in selected_stats
        ),
        projected_tool_result_chars=sum(
            stats.projected_tool_result_chars
            for stats in selected_stats
        ),
        history_token_budget=history_token_budget,
    )


def _is_user_message(unit: ContextUnit) -> bool:
    return (
        len(unit.items) == 1
        and isinstance(unit.items[0], Message)
        and unit.items[0].role == "user"
    )


def _serialize_item(item: AgentItem) -> dict[str, object]:
    if isinstance(item, Message):
        return {
            "type": "message",
            "role": item.role,
            "content": item.content,
        }

    if isinstance(item, ToolCall):
        return {
            "type": "tool_call",
            "name": item.name,
            "arguments": item.arguments,
            "call_id": item.call_id,
        }

    return {
        "type": "tool_result",
        "name": item.name,
        "content": item.content,
        "call_id": item.call_id,
        "is_error": item.is_error,
    }
