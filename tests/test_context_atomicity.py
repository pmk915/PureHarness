"""Context boundaries follow closed tool batches, not entire tool-only spans."""

from copy import deepcopy

import pytest

from pureharness.context import (
    ApproximateTokenEstimator,
    ContextBudget,
    ContextBudgetExceeded,
    ContextBuilder,
    ContextCompileError,
    TokenBudgetContextBuilder,
)
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.tool_history import match_tool_interactions
from pureharness.tool_result_projection import (
    DeterministicToolResultProjector,
)
from pureharness.trajectory_compaction import IdentityTrajectoryCompactor


def _pair(call_id, size=10):
    return [
        ToolCall(name="inspect", arguments={}, call_id=call_id),
        ToolResult(name="inspect", content="x" * size, call_id=call_id),
    ]


def _long_history():
    return [
        Message(role="user", content="Inspect the workspace."),
        *[item for i in range(10) for item in _pair(str(i), 2_500)],
        *_pair("latest", 11_992),
    ]


def _assert_complete_tools(items):
    interactions = []
    tools = []
    for item in [*items, Message(role="system", content="boundary")]:
        if isinstance(item, Message):
            matched = match_tool_interactions(tools)
            assert len(tools) == 2 * len(matched)
            interactions.extend(matched)
            tools = []
        else:
            tools.append(item)
    return interactions


def test_long_tool_only_history_compiles_with_default_8000_budget():
    history = _long_history()
    before = deepcopy(history)
    builder = TokenBudgetContextBuilder(ContextBudget(8_000))

    compiled = builder.compile(history)

    assert compiled == builder.compile(history)
    assert history == before
    assert compiled.estimated_tokens <= 8_000
    assert compiled.items[0] == history[0]
    assert compiled.items[-2:] == history[-2:]
    assert compiled.total_units == 12
    assert compiled.total_units == (
        compiled.included_units + compiled.dropped_units
    )
    assert compiled.trajectory_compacted
    assert compiled.compacted_source_units == 10
    assert compiled.compacted_tool_actions == 10
    assert compiled.recent_raw_units == 1
    assert compiled.compacted_trajectory_estimated_tokens < (
        compiled.original_trajectory_estimated_tokens
    )
    assert compiled.estimated_tokens == (
        compiled.compacted_trajectory_estimated_tokens
    )
    assert compiled.projected_tool_results == 1
    assert compiled.compacted_tool_results == 0
    assert compiled.raw_tool_result_chars == 11_992
    assert compiled.projected_tool_result_chars == 11_992
    assert len(_assert_complete_tools(compiled.items)) == 1


def test_full_history_separates_independent_pairs_without_reordering():
    history = _long_history()
    compiled = ContextBuilder(
        trajectory_compactor=IdentityTrajectoryCompactor()
    ).compile(history)

    assert compiled.items == history
    assert compiled.total_units == compiled.included_units == 12
    assert compiled.dropped_units == 0
    assert len(_assert_complete_tools(compiled.items)) == 11
    estimator = ApproximateTokenEstimator()
    assert compiled.estimated_tokens == (
        estimator.estimate(history[:1])
        + sum(
            estimator.estimate(history[i:i + 2])
            for i in range(1, len(history), 2)
        )
    )


def test_budget_can_drop_old_pairs_without_trajectory_compaction():
    old = _pair("old", 2_500)
    recent = _pair("recent", 2_500)
    budget = ApproximateTokenEstimator().estimate(recent)
    compiled = TokenBudgetContextBuilder(
        ContextBudget(budget),
        trajectory_compactor=IdentityTrajectoryCompactor(),
    ).compile([*old, *recent])

    assert compiled.items == recent
    assert compiled.estimated_tokens == budget
    assert compiled.total_units == 2
    assert compiled.included_units == compiled.dropped_units == 1
    assert len(_assert_complete_tools(compiled.items)) == 1


@pytest.mark.parametrize("reverse_results", [False, True])
@pytest.mark.parametrize("call_ids", [("a", "b"), (None, None)])
def test_explicit_multi_call_batch_is_indivisible(call_ids, reverse_results):
    a, b = (_pair(call_id, 200) for call_id in call_ids)
    results = [a[1], b[1]]
    batch = [a[0], b[0], *(reversed(results) if reverse_results else results)]
    estimator = ApproximateTokenEstimator()
    budget = estimator.estimate(batch)
    compiled = TokenBudgetContextBuilder(
        ContextBudget(budget),
        trajectory_compactor=IdentityTrajectoryCompactor(),
    ).compile([*_pair("old", 200), *batch])

    assert compiled.items == batch
    assert compiled.total_units == 2
    assert compiled.included_units == 1
    assert len(_assert_complete_tools(compiled.items)) == 2
    with pytest.raises(ContextBudgetExceeded):
        TokenBudgetContextBuilder(ContextBudget(budget - 1)).compile(batch)


def test_overlapping_calls_are_kept_until_all_pending_calls_close():
    a, b, c = (_pair(call_id) for call_id in ("a", "b", "c"))
    batch = [a[0], b[0], a[1], c[0], b[1], c[1]]
    history = [*batch, *_pair("later")]
    compiled = ContextBuilder(
        trajectory_compactor=IdentityTrajectoryCompactor()
    ).compile(history)

    assert compiled.items == history
    assert compiled.total_units == 2
    assert len(_assert_complete_tools(compiled.items)) == 4


def test_compaction_counts_multi_call_batch_as_one_source_unit():
    a, b = _pair("a", 11_000), _pair("b", 11_000)
    latest = _pair("latest", 11_992)
    history = [a[0], b[0], b[1], a[1], *latest]

    compiled = TokenBudgetContextBuilder(ContextBudget(8_000)).compile(history)

    assert compiled.items[-2:] == latest
    assert compiled.estimated_tokens <= 8_000
    assert compiled.total_units == compiled.included_units == 2
    assert compiled.dropped_units == 0
    assert compiled.compacted_source_units == 1
    assert compiled.compacted_tool_actions == 2
    assert compiled.projected_tool_results == 1
    assert compiled.recent_raw_units == 1
    assert len(_assert_complete_tools(compiled.items)) == 1


def test_runtime_interleaved_multi_call_layout_preserves_each_pair():
    history = [*_pair("a"), *_pair("b"), *_pair("c")]
    budget = ApproximateTokenEstimator().estimate(history[-2:])
    compiled = TokenBudgetContextBuilder(
        ContextBudget(budget),
        trajectory_compactor=IdentityTrajectoryCompactor(),
    ).compile(history)

    # Flat AgentItems contain no assistant-turn envelope: a closed pair is valid.
    assert compiled.items == history[-2:]
    assert compiled.total_units == 3
    assert len(_assert_complete_tools(compiled.items)) == 1


def test_pending_call_keeps_its_overlapping_batch_uncompacted():
    a, b = _pair("a"), _pair("b")
    history = [a[0], b[0], b[1]]
    compiled = ContextBuilder().compile(history)

    assert compiled.items == history
    assert compiled.total_units == 1
    assert not compiled.trajectory_compacted


def test_duplicate_ids_across_closed_pairs_still_fail():
    with pytest.raises(ContextCompileError, match="duplicate ToolCall call_id"):
        ContextBuilder().compile([*_pair("same"), *_pair("same")])


def test_result_cannot_match_across_message_boundary():
    a = _pair("a")
    with pytest.raises(ContextCompileError, match="no matching ToolCall"):
        ContextBuilder().compile(
            [a[0], Message(role="user", content="next"), a[1]]
        )


def test_projected_newest_single_pair_may_still_exceed_budget():
    history = _pair("latest", 40_000)
    before = deepcopy(history)
    projector = DeterministicToolResultProjector()
    projected = [history[0], projector.project(history[1])]
    budget = ApproximateTokenEstimator().estimate(projected) - 1

    assert len(projected[1].content) < len(history[1].content)
    with pytest.raises(
        ContextBudgetExceeded, match="Newest indivisible context unit"
    ):
        TokenBudgetContextBuilder(ContextBudget(budget)).compile(history)
    assert history == before
