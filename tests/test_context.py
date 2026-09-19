from collections.abc import Sequence
from copy import deepcopy

import pytest

from miniharness.context import (
    ApproximateTokenEstimator,
    ContextBudget,
    ContextBudgetExceeded,
    ContextBuilder,
    ContextCompileError,
    RecentContextBuilder,
    TokenBudgetContextBuilder,
)
from miniharness.messages import (
    AgentItem,
    Message,
    ToolCall,
    ToolResult,
)
from miniharness.session import Session
from miniharness.tool_result_projection import (
    DeterministicToolResultProjector,
    IdentityToolResultProjector,
)


class ContentCostEstimator:
    def __init__(self, costs: dict[str, int]) -> None:
        self.costs = costs
        self.calls: list[tuple[AgentItem, ...]] = []

    def estimate(
        self,
        items: Sequence[AgentItem],
    ) -> int:
        unit = tuple(items)
        self.calls.append(unit)
        first = unit[0]

        if isinstance(first, Message):
            key = first.content
        else:
            key = first.name

        return self.costs[key]


class ModelContentLengthEstimator:
    def estimate(self, items):
        return sum(
            len(item.content)
            if isinstance(item, (Message, ToolResult))
            else 1
            for item in items
        )


class FixedToolResultProjector:
    def project(self, result):
        return ToolResult(
            name=result.name,
            content="fixed projection",
            call_id=result.call_id,
            is_error=result.is_error,
        )


class MutatingToolResultProjector:
    def project(self, result):
        result.content = "mutated projection"
        return result


def _multi_tool_history() -> list[AgentItem]:
    return [
        ToolCall(
            name="read_file",
            arguments={"path": "a.py"},
            call_id="call-a",
        ),
        ToolResult(
            name="read_file",
            content="a",
            call_id="call-a",
        ),
        ToolCall(
            name="read_file",
            arguments={"path": "b.py"},
            call_id="call-b",
        ),
        ToolResult(
            name="read_file",
            content="b",
            call_id="call-b",
        ),
    ]


def test_approximate_estimator_is_deterministic_and_sensible():
    estimator = ApproximateTokenEstimator()
    empty = estimator.estimate([])
    small = estimator.estimate(
        [Message(role="user", content="hello")]
    )
    repeated = estimator.estimate(
        [Message(role="user", content="hello")]
    )
    large = estimator.estimate(
        [Message(role="user", content="hello" * 100)]
    )

    assert empty == 0
    assert small > 0
    assert repeated == small
    assert large > small


def test_approximate_estimator_accounts_for_all_item_types():
    estimator = ApproximateTokenEstimator()
    base = estimator.estimate(
        [Message(role="user", content="read")]
    )
    complete = estimator.estimate(
        [
            Message(role="user", content="read"),
            ToolCall(
                name="read_file",
                arguments={"path": "module.py"},
                call_id="1",
            ),
            ToolResult(
                name="read_file",
                content="file contents",
                call_id="1",
            ),
        ]
    )

    assert complete > base


@pytest.mark.parametrize("value", [0, -1, True])
def test_context_budget_rejects_invalid_values(value):
    with pytest.raises(
        ValueError,
        match="max_estimated_tokens must be greater than 0",
    ):
        ContextBudget(value)


def test_full_history_preserves_items_order_and_statistics():
    history = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    ]
    builder = ContextBuilder()

    compiled = builder.compile(history)

    assert compiled.items == history
    assert compiled.items is not history
    assert compiled.estimated_tokens > 0
    assert compiled.total_units == 2
    assert compiled.included_units == 2
    assert compiled.dropped_units == 0
    assert compiled.strategy == "FullHistory"
    assert compiled.history_token_budget is None
    assert compiled.projected_tool_results == 0
    assert compiled.compacted_tool_results == 0
    assert compiled.raw_tool_result_chars == 0
    assert compiled.projected_tool_result_chars == 0
    assert builder.build(history) == history


def test_small_tool_result_is_unchanged_with_projection_statistics():
    history = [
        ToolCall(name="add", arguments={}, call_id="1"),
        ToolResult(name="add", content="42", call_id="1"),
    ]

    compiled = ContextBuilder().compile(history)
    projected = compiled.items[1]

    assert isinstance(projected, ToolResult)
    assert projected.content == "42"
    assert compiled.projected_tool_results == 1
    assert compiled.compacted_tool_results == 0
    assert compiled.raw_tool_result_chars == 2
    assert compiled.projected_tool_result_chars == 2


def test_large_tool_result_projection_does_not_mutate_session():
    raw_content = "head" + "x" * 500 + "tail"
    session = Session(
        items=[
            ToolCall(
                name="read_file",
                arguments={"path": "large.txt"},
                call_id="1",
            ),
            ToolResult(
                name="read_file",
                content=raw_content,
                call_id="1",
                is_error=True,
            ),
        ]
    )
    before = deepcopy(session.snapshot())
    builder = ContextBuilder(
        tool_result_projector=(
            DeterministicToolResultProjector(
                max_chars=100,
                head_chars=20,
                tail_chars=20,
            )
        )
    )

    compiled = builder.compile(session.snapshot())
    projected = compiled.items[1]

    assert isinstance(projected, ToolResult)
    assert projected.content != raw_content
    assert projected.name == "read_file"
    assert projected.call_id == "1"
    assert projected.is_error is True
    assert compiled.projected_tool_results == 1
    assert compiled.compacted_tool_results == 1
    assert compiled.raw_tool_result_chars == len(raw_content)
    assert compiled.projected_tool_result_chars == len(
        projected.content
    )
    assert session.snapshot() == before
    assert session.items[1].content == raw_content


def test_projector_cannot_mutate_raw_session_result_in_place():
    session = Session(
        items=[
            ToolCall(name="read_file", arguments={}, call_id="1"),
            ToolResult(
                name="read_file",
                content="raw output",
                call_id="1",
            ),
        ]
    )

    compiled = ContextBuilder(
        tool_result_projector=MutatingToolResultProjector()
    ).compile(session.snapshot())

    assert compiled.items[1].content == "mutated projection"
    assert session.items[1].content == "raw output"


@pytest.mark.parametrize(
    "builder",
    [
        ContextBuilder(
            tool_result_projector=FixedToolResultProjector()
        ),
        RecentContextBuilder(
            max_items=10,
            tool_result_projector=FixedToolResultProjector(),
        ),
        TokenBudgetContextBuilder(
            ContextBudget(max_estimated_tokens=100),
            ModelContentLengthEstimator(),
            FixedToolResultProjector(),
        ),
    ],
)
def test_all_context_strategies_use_replaceable_projector(builder):
    history = [
        Message(role="user", content="inspect"),
        ToolCall(
            name="read_file",
            arguments={},
            call_id="1",
        ),
        ToolResult(
            name="read_file",
            content="raw output",
            call_id="1",
        ),
        Message(role="assistant", content="done"),
    ]

    compiled = builder.compile(history)
    result = next(
        item
        for item in compiled.items
        if isinstance(item, ToolResult)
    )

    assert result.content == "fixed projection"
    assert compiled.compacted_tool_results == 1


def test_projection_turns_m6_budget_failure_into_success():
    raw_content = "x" * 1_000
    session = Session(
        items=[
            ToolCall(
                name="read_file",
                arguments={},
                call_id="1",
            ),
            ToolResult(
                name="read_file",
                content=raw_content,
                call_id="1",
            ),
        ]
    )
    before = deepcopy(session.snapshot())
    budget = ContextBudget(max_estimated_tokens=200)

    with pytest.raises(ContextBudgetExceeded):
        TokenBudgetContextBuilder(
            budget,
            ModelContentLengthEstimator(),
            IdentityToolResultProjector(),
        ).compile(session.snapshot())

    compiled = TokenBudgetContextBuilder(
        budget,
        ModelContentLengthEstimator(),
        DeterministicToolResultProjector(
            max_chars=100,
            head_chars=20,
            tail_chars=20,
        ),
    ).compile(session.snapshot())

    assert compiled.estimated_tokens <= 200
    assert compiled.compacted_tool_results == 1
    assert compiled.items[1].content != raw_content
    assert session.snapshot() == before
    assert session.items[1].content == raw_content


def test_budget_still_fails_when_projected_unit_is_too_large():
    history = [
        ToolCall(name="read_file", arguments={}, call_id="1"),
        ToolResult(
            name="read_file",
            content="x" * 1_000,
            call_id="1",
        ),
    ]
    builder = TokenBudgetContextBuilder(
        ContextBudget(max_estimated_tokens=10),
        ModelContentLengthEstimator(),
        DeterministicToolResultProjector(
            max_chars=100,
            head_chars=20,
            tail_chars=20,
        ),
    )

    with pytest.raises(ContextBudgetExceeded):
        builder.compile(history)


def test_recent_context_keeps_recent_user_turn():
    history = [
        Message(role="user", content="first"),
        Message(role="assistant", content="first answer"),
        Message(role="user", content="second"),
        Message(role="assistant", content="second answer"),
    ]
    builder = RecentContextBuilder(max_items=2)

    compiled = builder.compile(history)

    assert [item.content for item in compiled.items] == [
        "second",
        "second answer",
    ]
    assert compiled.total_units == 4
    assert compiled.included_units == 2
    assert compiled.dropped_units == 2
    assert compiled.strategy == "Recent"


def test_recent_context_preserves_single_tool_turn():
    history = [
        Message(role="user", content="old question"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="calculate"),
        ToolCall(
            name="add",
            arguments={"a": 12, "b": 17},
            call_id="1",
        ),
        ToolResult(name="add", content="29", call_id="1"),
        Message(role="assistant", content="The result is 29"),
    ]
    builder = RecentContextBuilder(max_items=2)

    context = builder.compile(history).items

    assert len(context) == 4
    assert isinstance(context[0], Message)
    assert context[0].content == "calculate"
    assert isinstance(context[1], ToolCall)
    assert isinstance(context[2], ToolResult)
    assert isinstance(context[3], Message)


def test_recent_context_never_splits_multi_tool_step():
    tool_history = _multi_tool_history()
    history = [
        Message(role="user", content="old"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="inspect"),
        *tool_history,
        Message(role="assistant", content="done"),
    ]
    builder = RecentContextBuilder(max_items=3)

    first = builder.compile(history)
    second = builder.compile(history)

    assert first.items == second.items
    assert first.items == [history[2], *tool_history, history[-1]]
    assert first.included_units == 3
    assert first.dropped_units == 2


def test_recent_context_rejects_invalid_max_items():
    with pytest.raises(
        ValueError,
        match="max_items must be greater than 0",
    ):
        RecentContextBuilder(max_items=0)


def test_token_budget_selects_newest_units_in_original_order():
    history = [
        Message(role="user", content="A"),
        Message(role="assistant", content="B"),
        Message(role="user", content="C"),
        Message(role="assistant", content="D"),
        Message(role="user", content="E"),
    ]
    estimator = ContentCostEstimator(
        {"A": 7, "B": 12, "C": 9, "D": 8, "E": 6}
    )
    builder = TokenBudgetContextBuilder(
        ContextBudget(max_estimated_tokens=35),
        estimator,
    )

    compiled = builder.compile(history)

    assert compiled.items == history[1:]
    assert compiled.estimated_tokens == 35
    assert compiled.total_units == 5
    assert compiled.included_units == 4
    assert compiled.dropped_units == 1
    assert compiled.strategy == "TokenBudget"
    assert compiled.history_token_budget == 35
    assert len(estimator.calls) == 5


def test_token_budget_never_splits_multi_tool_step():
    tool_history = _multi_tool_history()
    history = [
        Message(role="user", content="old"),
        *tool_history,
        Message(role="assistant", content="done"),
    ]
    estimator = ContentCostEstimator(
        {"old": 5, "read_file": 6, "done": 2}
    )
    builder = TokenBudgetContextBuilder(
        ContextBudget(max_estimated_tokens=8),
        estimator,
    )

    compiled = builder.compile(history)

    assert compiled.items == [*tool_history, history[-1]]
    assert compiled.estimated_tokens == 8
    assert compiled.included_units == 2
    assert compiled.dropped_units == 1


def test_token_budget_raises_for_oversized_newest_unit():
    session = Session(
        items=[Message(role="user", content="oversized")]
    )
    before = session.snapshot()
    builder = TokenBudgetContextBuilder(
        ContextBudget(max_estimated_tokens=3),
        ContentCostEstimator({"oversized": 4}),
    )

    with pytest.raises(
        ContextBudgetExceeded,
        match="Newest indivisible context unit",
    ):
        builder.compile(session.snapshot())

    assert session.snapshot() == before


@pytest.mark.parametrize(
    "builder",
    [
        ContextBuilder(),
        RecentContextBuilder(max_items=1),
        TokenBudgetContextBuilder(
            ContextBudget(max_estimated_tokens=10),
            ContentCostEstimator({"first": 6, "second": 4}),
        ),
    ],
)
def test_context_compilation_never_mutates_session(builder):
    session = Session(
        items=[
            Message(role="user", content="first"),
            Message(role="assistant", content="second"),
        ]
    )
    before = session.snapshot()

    builder.compile(session.snapshot())

    assert session.snapshot() == before


@pytest.mark.parametrize(
    "history",
    [
        [ToolResult(name="read_file", content="x", call_id="1")],
        [
            ToolCall(
                name="read_file",
                arguments={},
                call_id="1",
            ),
            ToolResult(
                name="read_file",
                content="x",
                call_id="2",
            ),
        ],
    ],
)
def test_context_rejects_unmatched_tool_results(history):
    with pytest.raises(
        ContextCompileError,
        match="ToolResult has no matching ToolCall",
    ):
        ContextBuilder().compile(history)


def test_context_supports_legacy_missing_call_ids():
    history = [
        ToolCall(name="read_file", arguments={}, call_id=None),
        ToolResult(name="read_file", content="x", call_id=None),
    ]

    compiled = ContextBuilder().compile(history)

    assert compiled.items == history
    assert compiled.total_units == 1


def test_context_rejects_duplicate_tool_call_ids():
    history = [
        ToolCall(name="read_file", arguments={}, call_id="1"),
        ToolCall(name="read_file", arguments={}, call_id="1"),
    ]

    with pytest.raises(
        ContextCompileError,
        match="duplicate ToolCall call_id",
    ):
        ContextBuilder().compile(history)
