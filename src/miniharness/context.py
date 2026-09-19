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
from miniharness.tool_history import (
    ToolHistoryError,
    match_tool_interactions,
)
from miniharness.token_estimation import approximate_text_tokens
from miniharness.trajectory_compaction import (
    CompactedTrajectory,
    DeterministicToolTrajectoryCompactor,
    ModelContextUnit,
    TrajectoryCompactionError,
    TrajectoryCompactor,
    default_compactor_for_history_budget,
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

        return approximate_text_tokens(serialized)


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
    trajectory_compacted: bool = False
    compacted_source_units: int = 0
    compacted_tool_actions: int = 0
    original_trajectory_estimated_tokens: int = 0
    compacted_trajectory_estimated_tokens: int = 0
    recent_raw_units: int = 0
    recent_raw_estimated_tokens: int = 0
    trajectory_compaction_strategy: str = "Identity"


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
        trajectory_compactor: TrajectoryCompactor | None = None,
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
        self.trajectory_compactor = (
            trajectory_compactor
            if trajectory_compactor is not None
            else DeterministicToolTrajectoryCompactor()
        )

    def _prepare(
        self,
        history: Sequence[AgentItem],
    ) -> CompactedTrajectory:
        raw_units = _group_context_units(history)
        units, projection_stats = _project_context_units(
            raw_units,
            self.tool_result_projector,
        )
        costs = _estimate_units(units, self.token_estimator)
        model_units = [
            ModelContextUnit(
                items=unit.items,
                estimated_tokens=cost,
                projected_tool_results=(
                    stats.projected_tool_results
                ),
                compacted_tool_results=(
                    stats.compacted_tool_results
                ),
                raw_tool_result_chars=(
                    stats.raw_tool_result_chars
                ),
                projected_tool_result_chars=(
                    stats.projected_tool_result_chars
                ),
            )
            for unit, cost, stats in zip(
                units,
                costs,
                projection_stats,
                strict=True,
            )
        ]

        try:
            trajectory = self.trajectory_compactor.compact(
                model_units,
                self.token_estimator,
            )
        except (ToolHistoryError, TrajectoryCompactionError) as exc:
            raise ContextCompileError(str(exc)) from exc

        _validate_compacted_trajectory(trajectory)
        return trajectory

    def compile(
        self,
        history: Sequence[AgentItem],
    ) -> CompiledContext:
        trajectory = self._prepare(history)

        return _compiled_context(
            trajectory,
            start=0,
            strategy=self.strategy,
        )

    def build(
        self,
        history: list[AgentItem],
    ) -> list[AgentItem]:
        """Compatibility view over the canonical compile path."""
        return self.compile(history).items

    def estimate_tokens(
        self,
        items: Sequence[AgentItem],
    ) -> int:
        """Estimate and validate model-facing items with this compiler."""
        return _estimate_items(items, self.token_estimator)


class RecentContextBuilder(ContextBuilder):
    strategy = "Recent"

    def __init__(
        self,
        max_items: int = 20,
        token_estimator: TokenEstimator | None = None,
        tool_result_projector: ToolResultProjector | None = None,
        trajectory_compactor: TrajectoryCompactor | None = None,
    ) -> None:
        if max_items <= 0:
            raise ValueError(
                "max_items must be greater than 0"
            )

        super().__init__(
            token_estimator,
            tool_result_projector,
            trajectory_compactor,
        )
        self.max_items = max_items

    def compile(
        self,
        history: Sequence[AgentItem],
    ) -> CompiledContext:
        trajectory = self._prepare(history)
        units = trajectory.units
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
            trajectory,
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
        trajectory_compactor: TrajectoryCompactor | None = None,
    ) -> None:
        if trajectory_compactor is None:
            trajectory_compactor = default_compactor_for_history_budget(
                budget.max_estimated_tokens
            )

        super().__init__(
            token_estimator,
            tool_result_projector,
            trajectory_compactor,
        )
        self.budget = budget

    def compile(
        self,
        history: Sequence[AgentItem],
    ) -> CompiledContext:
        trajectory = self._prepare(history)
        units = trajectory.units
        start = len(units)
        estimated_tokens = 0

        while start > 0:
            cost = units[start - 1].estimated_tokens

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
            trajectory,
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
    try:
        match_tool_interactions(items)
    except ToolHistoryError as exc:
        raise ContextCompileError(str(exc)) from exc

    return ContextUnit(items=tuple(items))


def _estimate_units(
    units: Sequence[ContextUnit],
    token_estimator: TokenEstimator,
) -> list[int]:
    costs = []

    for unit in units:
        costs.append(_estimate_items(unit.items, token_estimator))

    return costs


def _estimate_items(
    items: Sequence[AgentItem],
    token_estimator: TokenEstimator,
) -> int:
    cost = token_estimator.estimate(items)

    if (
        not isinstance(cost, int)
        or isinstance(cost, bool)
        or cost < 0
    ):
        raise ContextCompileError(
            "TokenEstimator must return a non-negative integer."
        )

    return cost


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
    trajectory: CompactedTrajectory,
    *,
    start: int,
    strategy: str,
    history_token_budget: int | None = None,
) -> CompiledContext:
    units = trajectory.units
    selected_units = units[start:]
    total_units = sum(unit.source_unit_count for unit in units)
    included_units = sum(
        unit.source_unit_count for unit in selected_units
    )

    return CompiledContext(
        items=[
            item
            for unit in selected_units
            for item in unit.items
        ],
        estimated_tokens=sum(
            unit.estimated_tokens for unit in selected_units
        ),
        total_units=total_units,
        included_units=included_units,
        dropped_units=total_units - included_units,
        strategy=strategy,
        projected_tool_results=sum(
            unit.projected_tool_results
            for unit in selected_units
        ),
        compacted_tool_results=sum(
            unit.compacted_tool_results
            for unit in selected_units
        ),
        raw_tool_result_chars=sum(
            unit.raw_tool_result_chars
            for unit in selected_units
        ),
        projected_tool_result_chars=sum(
            unit.projected_tool_result_chars
            for unit in selected_units
        ),
        history_token_budget=history_token_budget,
        trajectory_compacted=trajectory.trajectory_compacted,
        compacted_source_units=trajectory.compacted_source_units,
        compacted_tool_actions=trajectory.compacted_tool_actions,
        original_trajectory_estimated_tokens=(
            trajectory.original_estimated_tokens
        ),
        compacted_trajectory_estimated_tokens=(
            trajectory.compacted_estimated_tokens
        ),
        recent_raw_units=trajectory.recent_raw_units,
        recent_raw_estimated_tokens=(
            trajectory.recent_raw_estimated_tokens
        ),
        trajectory_compaction_strategy=trajectory.strategy,
    )


def _validate_compacted_trajectory(
    trajectory: CompactedTrajectory,
) -> None:
    if not isinstance(trajectory, CompactedTrajectory):
        raise ContextCompileError(
            "TrajectoryCompactor must return CompactedTrajectory."
        )

    if not isinstance(trajectory.units, tuple):
        raise ContextCompileError(
            "CompactedTrajectory units must be a tuple."
        )

    for unit in trajectory.units:
        if not isinstance(unit, ModelContextUnit):
            raise ContextCompileError(
                "CompactedTrajectory must contain ModelContextUnit "
                "values."
            )

        _validate_non_negative_integer(
            "ModelContextUnit estimated_tokens",
            unit.estimated_tokens,
        )

        if (
            not isinstance(unit.source_unit_count, int)
            or isinstance(unit.source_unit_count, bool)
            or unit.source_unit_count <= 0
        ):
            raise ContextCompileError(
                "ModelContextUnit source_unit_count must be a "
                "positive integer."
            )

        for name in (
            "projected_tool_results",
            "compacted_tool_results",
            "raw_tool_result_chars",
            "projected_tool_result_chars",
        ):
            _validate_non_negative_integer(
                f"ModelContextUnit {name}",
                getattr(unit, name),
            )

    for name in (
        "compacted_source_units",
        "compacted_tool_actions",
        "original_estimated_tokens",
        "compacted_estimated_tokens",
        "recent_raw_units",
        "recent_raw_estimated_tokens",
    ):
        _validate_non_negative_integer(
            f"CompactedTrajectory {name}",
            getattr(trajectory, name),
        )

    if not isinstance(trajectory.trajectory_compacted, bool):
        raise ContextCompileError(
            "CompactedTrajectory trajectory_compacted must be bool."
        )

    if not isinstance(trajectory.strategy, str) or not trajectory.strategy:
        raise ContextCompileError(
            "CompactedTrajectory strategy must be non-empty text."
        )

    actual_tokens = sum(
        unit.estimated_tokens for unit in trajectory.units
    )
    if actual_tokens != trajectory.compacted_estimated_tokens:
        raise ContextCompileError(
            "CompactedTrajectory token total does not match its units."
        )

    if trajectory.trajectory_compacted != (
        trajectory.compacted_source_units > 0
    ):
        raise ContextCompileError(
            "CompactedTrajectory compaction metrics are inconsistent."
        )


def _validate_non_negative_integer(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ContextCompileError(
            f"{name} must be a non-negative integer."
        )


def _is_user_message(unit: ContextUnit | ModelContextUnit) -> bool:
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
