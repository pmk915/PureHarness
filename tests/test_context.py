from collections.abc import Sequence

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
    assert builder.build(history) == history


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
