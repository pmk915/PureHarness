import re
import shlex

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pureharness.messages import AgentItem, Message, ToolCall, ToolResult
from pureharness.tool_history import match_tool_interactions


DEFAULT_COMPACTION_TRIGGER_TOKENS = 12_000
DEFAULT_RECENT_RAW_TOKENS = 4_000
DEFAULT_MAX_ARGUMENT_CHARS = 120
DEFAULT_MAX_FAILURE_DETAIL_CHARS = 120

_COMPACTED_HISTORY_TITLE = "[PureHarness Compacted Tool History]"
_PATH_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "apply_patch",
        "list_files",
        "git_diff",
    }
)
_SENSITIVE_TOKEN = re.compile(
    r"(?i)(password|secret|token|api[_-]?key|authorization)"
)


class TrajectoryCompactionError(RuntimeError):
    """Raised when a trajectory compactor cannot produce valid context."""


class TrajectoryTokenEstimator(Protocol):
    def estimate(
        self,
        items: Sequence[AgentItem],
    ) -> int:
        ...


@dataclass(frozen=True)
class ModelContextUnit:
    items: tuple[AgentItem, ...]
    estimated_tokens: int
    source_unit_count: int = 1
    projected_tool_results: int = 0
    compacted_tool_results: int = 0
    raw_tool_result_chars: int = 0
    projected_tool_result_chars: int = 0


@dataclass(frozen=True)
class CompactedTrajectory:
    units: tuple[ModelContextUnit, ...]
    trajectory_compacted: bool
    compacted_source_units: int
    compacted_tool_actions: int
    original_estimated_tokens: int
    compacted_estimated_tokens: int
    recent_raw_units: int
    recent_raw_estimated_tokens: int
    strategy: str


class TrajectoryCompactor(Protocol):
    def compact(
        self,
        units: Sequence[ModelContextUnit],
        token_estimator: TrajectoryTokenEstimator,
    ) -> CompactedTrajectory:
        ...


class IdentityTrajectoryCompactor:
    """Preserve projected trajectory units without M9 compaction."""

    def compact(
        self,
        units: Sequence[ModelContextUnit],
        token_estimator: TrajectoryTokenEstimator,
    ) -> CompactedTrajectory:
        del token_estimator
        units_tuple = tuple(units)
        estimated_tokens = sum(
            unit.estimated_tokens for unit in units_tuple
        )

        return CompactedTrajectory(
            units=units_tuple,
            trajectory_compacted=False,
            compacted_source_units=0,
            compacted_tool_actions=0,
            original_estimated_tokens=estimated_tokens,
            compacted_estimated_tokens=estimated_tokens,
            recent_raw_units=sum(
                unit.source_unit_count for unit in units_tuple
            ),
            recent_raw_estimated_tokens=estimated_tokens,
            strategy="Identity",
        )


class DeterministicToolTrajectoryCompactor:
    """Compact eligible old tool units into structural system messages."""

    def __init__(
        self,
        *,
        compaction_trigger_tokens: int = (
            DEFAULT_COMPACTION_TRIGGER_TOKENS
        ),
        recent_raw_tokens: int = DEFAULT_RECENT_RAW_TOKENS,
        max_argument_chars: int = DEFAULT_MAX_ARGUMENT_CHARS,
        max_failure_detail_chars: int = (
            DEFAULT_MAX_FAILURE_DETAIL_CHARS
        ),
    ) -> None:
        _validate_positive_integer(
            "compaction_trigger_tokens",
            compaction_trigger_tokens,
        )
        _validate_positive_integer(
            "recent_raw_tokens",
            recent_raw_tokens,
        )
        _validate_positive_integer(
            "max_argument_chars",
            max_argument_chars,
        )
        _validate_positive_integer(
            "max_failure_detail_chars",
            max_failure_detail_chars,
        )

        if recent_raw_tokens >= compaction_trigger_tokens:
            raise ValueError(
                "recent_raw_tokens must be less than "
                "compaction_trigger_tokens"
            )

        self.compaction_trigger_tokens = compaction_trigger_tokens
        self.recent_raw_tokens = recent_raw_tokens
        self.max_argument_chars = max_argument_chars
        self.max_failure_detail_chars = max_failure_detail_chars

    def compact(
        self,
        units: Sequence[ModelContextUnit],
        token_estimator: TrajectoryTokenEstimator,
    ) -> CompactedTrajectory:
        units_tuple = tuple(units)
        original_tokens = sum(
            unit.estimated_tokens for unit in units_tuple
        )

        if original_tokens <= self.compaction_trigger_tokens:
            return _unchanged_trajectory(
                units_tuple,
                original_tokens,
                strategy="DeterministicOldTools",
            )

        recent_start = _recent_raw_start(
            units_tuple,
            self.recent_raw_tokens,
        )
        output_units: list[ModelContextUnit] = []
        compacted_source_units = 0
        compacted_tool_actions = 0

        for index, unit in enumerate(units_tuple):
            if index >= recent_start or not _is_tool_unit(unit):
                output_units.append(unit)
                continue

            compacted_unit, action_count = self._compact_tool_unit(
                unit,
                token_estimator,
            )

            if compacted_unit.estimated_tokens >= unit.estimated_tokens:
                output_units.append(unit)
                continue

            output_units.append(compacted_unit)
            compacted_source_units += unit.source_unit_count
            compacted_tool_actions += action_count

        compacted_tokens = sum(
            unit.estimated_tokens for unit in output_units
        )
        recent_units = units_tuple[recent_start:]

        return CompactedTrajectory(
            units=tuple(output_units),
            trajectory_compacted=compacted_source_units > 0,
            compacted_source_units=compacted_source_units,
            compacted_tool_actions=compacted_tool_actions,
            original_estimated_tokens=original_tokens,
            compacted_estimated_tokens=compacted_tokens,
            recent_raw_units=sum(
                unit.source_unit_count for unit in recent_units
            ),
            recent_raw_estimated_tokens=sum(
                unit.estimated_tokens for unit in recent_units
            ),
            strategy="DeterministicOldTools",
        )

    def _compact_tool_unit(
        self,
        unit: ModelContextUnit,
        token_estimator: TrajectoryTokenEstimator,
    ) -> tuple[ModelContextUnit, int]:
        interactions = match_tool_interactions(unit.items)
        call_count = sum(
            isinstance(item, ToolCall) for item in unit.items
        )
        result_count = sum(
            isinstance(item, ToolResult) for item in unit.items
        )

        if (
            not interactions
            or len(interactions) != call_count
            or len(interactions) != result_count
        ):
            return unit, 0

        lines = [
            _render_interaction(
                interaction.call,
                interaction.result,
                max_argument_chars=self.max_argument_chars,
                max_failure_detail_chars=(
                    self.max_failure_detail_chars
                ),
            )
            for interaction in interactions
        ]
        message = Message(
            role="system",
            content=(
                f"{_COMPACTED_HISTORY_TITLE}\n"
                "Historical tool step:\n"
                + "\n".join(lines)
            ),
        )
        estimated_tokens = _estimate_tokens(
            token_estimator,
            (message,),
        )

        return (
            ModelContextUnit(
                items=(message,),
                estimated_tokens=estimated_tokens,
                source_unit_count=unit.source_unit_count,
            ),
            len(interactions),
        )


def default_compactor_for_history_budget(
    history_budget: int,
) -> DeterministicToolTrajectoryCompactor:
    _validate_positive_integer("history_budget", history_budget)
    recent_raw_tokens = max(1, history_budget // 3)
    compaction_trigger_tokens = max(
        recent_raw_tokens + 1,
        (history_budget * 4) // 5,
    )

    return DeterministicToolTrajectoryCompactor(
        compaction_trigger_tokens=compaction_trigger_tokens,
        recent_raw_tokens=recent_raw_tokens,
    )


def _unchanged_trajectory(
    units: tuple[ModelContextUnit, ...],
    estimated_tokens: int,
    *,
    strategy: str,
) -> CompactedTrajectory:
    return CompactedTrajectory(
        units=units,
        trajectory_compacted=False,
        compacted_source_units=0,
        compacted_tool_actions=0,
        original_estimated_tokens=estimated_tokens,
        compacted_estimated_tokens=estimated_tokens,
        recent_raw_units=sum(
            unit.source_unit_count for unit in units
        ),
        recent_raw_estimated_tokens=estimated_tokens,
        strategy=strategy,
    )


def _recent_raw_start(
    units: Sequence[ModelContextUnit],
    recent_raw_tokens: int,
) -> int:
    start = len(units)
    estimated_tokens = 0
    included_unit = False

    while start > 0:
        next_cost = units[start - 1].estimated_tokens

        if (
            included_unit
            and estimated_tokens + next_cost > recent_raw_tokens
        ):
            break

        start -= 1
        estimated_tokens += next_cost
        included_unit = True

        if estimated_tokens >= recent_raw_tokens:
            break

    return start


def _is_tool_unit(unit: ModelContextUnit) -> bool:
    return bool(unit.items) and all(
        isinstance(item, (ToolCall, ToolResult))
        for item in unit.items
    )


def _render_interaction(
    call: ToolCall,
    result: ToolResult,
    *,
    max_argument_chars: int,
    max_failure_detail_chars: int,
) -> str:
    target = _safe_target(call, max_argument_chars)
    status = "failed" if result.is_error else "success"
    line = f"- {call.name}"

    if target:
        line += f" {target}"

    line += f" -> {status}"

    if result.is_error:
        detail = _bounded_tail(
            " ".join(result.content.split()),
            max_failure_detail_chars,
        )
        if detail:
            line += f"\n  detail: {detail}"

    return line


def _safe_target(
    call: ToolCall,
    max_chars: int,
) -> str:
    if call.name in _PATH_TOOLS:
        path = call.arguments.get("path")

        if isinstance(path, str):
            return _bounded(_single_line(path), max_chars)

        return ""

    if call.name == "search_text":
        parts = []
        query = call.arguments.get("query")
        path = call.arguments.get("path")

        if isinstance(query, str):
            parts.append(f"query={_single_line(query)}")
        if isinstance(path, str):
            parts.append(f"path={_single_line(path)}")

        return _bounded(" ".join(parts), max_chars)

    if call.name == "run_command":
        argv = call.arguments.get("argv")

        if not isinstance(argv, list):
            return ""

        safe_argv = _safe_argv(argv[:6])
        rendered = shlex.join(safe_argv)

        if len(argv) > 6:
            rendered += f" … ({len(argv)} argv items)"

        return _bounded(rendered, max_chars)

    return ""


def _safe_argv(values: Sequence[object]) -> list[str]:
    safe_values = []
    redact_next = False

    for value in values:
        text = _single_line(str(value))

        if redact_next:
            safe_values.append("[REDACTED]")
            redact_next = False
            continue

        if _SENSITIVE_TOKEN.search(text):
            safe_values.append("[REDACTED]")
            redact_next = "=" not in text
            continue

        safe_values.append(text)

    return safe_values


def _bounded(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    return value[: max_chars - 1] + "…"


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _bounded_tail(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    if max_chars == 1:
        return "…"

    return "…" + value[-(max_chars - 1) :]


def _estimate_tokens(
    token_estimator: TrajectoryTokenEstimator,
    items: Sequence[AgentItem],
) -> int:
    estimated_tokens = token_estimator.estimate(items)

    if (
        not isinstance(estimated_tokens, int)
        or isinstance(estimated_tokens, bool)
        or estimated_tokens < 0
    ):
        raise TrajectoryCompactionError(
            "Token estimator must return a non-negative integer."
        )

    return estimated_tokens


def _validate_positive_integer(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
