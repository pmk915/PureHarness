from collections.abc import Sequence
from dataclasses import dataclass

from miniharness.messages import ToolCall, ToolResult


class ToolHistoryError(ValueError):
    """Raised when a raw tool execution group is structurally invalid."""


@dataclass(frozen=True)
class ToolInteraction:
    call: ToolCall
    result: ToolResult


def match_tool_interactions(
    items: Sequence[ToolCall | ToolResult],
) -> tuple[ToolInteraction, ...]:
    """Match results to earlier calls using the canonical history rules."""
    calls: list[ToolCall] = []
    matched_call_indexes: set[int] = set()
    call_ids: set[str] = set()
    interactions: list[ToolInteraction] = []

    for item in items:
        if isinstance(item, ToolCall):
            if item.call_id is not None:
                if item.call_id in call_ids:
                    raise ToolHistoryError(
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
            raise ToolHistoryError(
                "ToolResult has no matching ToolCall in its "
                f"execution group: {item.name!r}."
            )

        matched_call_indexes.add(match)
        interactions.append(
            ToolInteraction(
                call=calls[match],
                result=item,
            )
        )

    return tuple(interactions)


def _tool_result_matches(
    call: ToolCall,
    result: ToolResult,
) -> bool:
    if call.name != result.name:
        return False

    if call.call_id is None or result.call_id is None:
        return True

    return call.call_id == result.call_id
