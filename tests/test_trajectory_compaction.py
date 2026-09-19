from collections.abc import Sequence
from copy import deepcopy

import pytest

from miniharness.agent import Agent
from miniharness.context import (
    ContextBudget,
    ContextBuilder,
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
)
from miniharness.trajectory_compaction import (
    CompactedTrajectory,
    DeterministicToolTrajectoryCompactor,
    IdentityTrajectoryCompactor,
    ModelContextUnit,
    default_compactor_for_history_budget,
)


_COMPACTED_TITLE = "[MiniHarness Compacted Tool History]"


class CompactCostEstimator:
    def __init__(self, compact_cost: int = 3) -> None:
        self.compact_cost = compact_cost
        self.calls: list[tuple[AgentItem, ...]] = []

    def estimate(self, items: Sequence[AgentItem]) -> int:
        unit = tuple(items)
        self.calls.append(unit)
        assert len(unit) == 1
        assert isinstance(unit[0], Message)
        assert unit[0].content.startswith(_COMPACTED_TITLE)
        return self.compact_cost


class ModelContentLengthEstimator:
    def estimate(self, items: Sequence[AgentItem]) -> int:
        return sum(
            len(item.content)
            if isinstance(item, (Message, ToolResult))
            else 1
            for item in items
        )


class RecordingModel:
    def __init__(self) -> None:
        self.contexts: list[list[AgentItem]] = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        return Message(role="assistant", content="done")


def _tool_unit(
    name: str,
    *,
    call_id: str,
    estimated_tokens: int,
    arguments: dict[str, object] | None = None,
    content: str = "ok",
    is_error: bool = False,
) -> ModelContextUnit:
    return ModelContextUnit(
        items=(
            ToolCall(
                name=name,
                arguments=arguments or {},
                call_id=call_id,
            ),
            ToolResult(
                name=name,
                content=content,
                call_id=call_id,
                is_error=is_error,
            ),
        ),
        estimated_tokens=estimated_tokens,
    )


def _message_unit(
    role: str,
    content: str,
    estimated_tokens: int = 2,
) -> ModelContextUnit:
    return ModelContextUnit(
        items=(Message(role=role, content=content),),
        estimated_tokens=estimated_tokens,
    )


def _compact_text(unit: ModelContextUnit) -> str:
    assert len(unit.items) == 1
    item = unit.items[0]
    assert isinstance(item, Message)
    assert item.role == "system"
    return item.content


def test_identity_compactor_preserves_units_and_reports_baseline():
    units = (
        _message_unit("user", "request"),
        _tool_unit(
            "read_file",
            call_id="read-1",
            estimated_tokens=20,
            arguments={"path": "a.py"},
        ),
    )

    compacted = IdentityTrajectoryCompactor().compact(
        units,
        CompactCostEstimator(),
    )

    assert compacted == CompactedTrajectory(
        units=units,
        trajectory_compacted=False,
        compacted_source_units=0,
        compacted_tool_actions=0,
        original_estimated_tokens=22,
        compacted_estimated_tokens=22,
        recent_raw_units=2,
        recent_raw_estimated_tokens=22,
        strategy="Identity",
    )


def test_trigger_boundary_is_exact_and_above_trigger_compacts():
    units = (
        _tool_unit(
            "read_file",
            call_id="read-1",
            estimated_tokens=10,
            arguments={"path": "a.py"},
        ),
        _message_unit("user", "latest", 1),
    )
    estimator = CompactCostEstimator(compact_cost=2)

    at_trigger = DeterministicToolTrajectoryCompactor(
        compaction_trigger_tokens=11,
        recent_raw_tokens=1,
    ).compact(units, estimator)
    above_trigger = DeterministicToolTrajectoryCompactor(
        compaction_trigger_tokens=10,
        recent_raw_tokens=1,
    ).compact(units, estimator)

    assert at_trigger.units == units
    assert at_trigger.trajectory_compacted is False
    assert above_trigger.trajectory_compacted is True
    assert above_trigger.compacted_source_units == 1
    assert above_trigger.compacted_estimated_tokens == 3
    assert _compact_text(above_trigger.units[0]).endswith(
        "- read_file a.py -> success"
    )
    assert above_trigger.units[1] == units[1]


def test_zero_cost_newest_unit_does_not_expand_recent_reserve():
    old_tool = _tool_unit(
        "read_file",
        call_id="read-1",
        estimated_tokens=20,
        arguments={"path": "a.py"},
    )
    zero_cost_latest = _message_unit("user", "latest", 0)

    compacted = DeterministicToolTrajectoryCompactor(
        compaction_trigger_tokens=10,
        recent_raw_tokens=5,
    ).compact(
        (old_tool, zero_cost_latest),
        CompactCostEstimator(),
    )

    assert compacted.trajectory_compacted is True
    assert compacted.recent_raw_units == 1
    assert compacted.recent_raw_estimated_tokens == 0
    assert compacted.units[1] == zero_cost_latest


def test_recent_window_preserves_atomic_units_messages_and_order():
    multi_tool = ModelContextUnit(
        items=(
            ToolCall(
                name="read_file",
                arguments={"path": "b.py"},
                call_id="b",
            ),
            ToolCall(
                name="read_file",
                arguments={"path": "c.py"},
                call_id="c",
            ),
            ToolResult(
                name="read_file",
                content="b",
                call_id="b",
            ),
            ToolResult(
                name="read_file",
                content="c",
                call_id="c",
            ),
        ),
        estimated_tokens=40,
    )
    units = (
        _message_unit("user", "old user"),
        _tool_unit(
            "read_file",
            call_id="a",
            estimated_tokens=40,
            arguments={"path": "a.py"},
        ),
        _message_unit("assistant", "old assistant"),
        multi_tool,
        _message_unit("user", "recent user"),
        _tool_unit(
            "read_file",
            call_id="d",
            estimated_tokens=10,
            arguments={"path": "d.py"},
        ),
        _message_unit("assistant", "recent assistant"),
        _tool_unit(
            "read_file",
            call_id="e",
            estimated_tokens=10,
            arguments={"path": "e.py"},
        ),
    )
    compacted = DeterministicToolTrajectoryCompactor(
        compaction_trigger_tokens=100,
        recent_raw_tokens=24,
    ).compact(units, CompactCostEstimator())

    assert compacted.trajectory_compacted is True
    assert compacted.compacted_source_units == 2
    assert compacted.compacted_tool_actions == 3
    assert compacted.recent_raw_units == 4
    assert compacted.recent_raw_estimated_tokens == 24
    assert compacted.units[0] == units[0]
    assert "a.py" in _compact_text(compacted.units[1])
    assert compacted.units[2] == units[2]
    assert "b.py" in _compact_text(compacted.units[3])
    assert "c.py" in _compact_text(compacted.units[3])
    assert compacted.units[4:] == units[4:]


def test_compact_record_bounds_arguments_redacts_and_keeps_failure():
    long_text = "x" * 500
    unit = ModelContextUnit(
        items=(
            ToolCall(
                name="apply_patch",
                arguments={
                    "path": "src/auth.py",
                    "old_text": long_text,
                    "new_text": long_text,
                },
                call_id="patch",
            ),
            ToolCall(
                name="search_text",
                arguments={
                    "query": "needle-\n" + long_text,
                    "path": "src",
                },
                call_id="search",
            ),
            ToolCall(
                name="run_command",
                arguments={
                    "argv": [
                        "python",
                        "--api-key",
                        "super-secret",
                        "script.py",
                        long_text,
                        "tail",
                        "omitted",
                    ]
                },
                call_id="command",
            ),
            ToolResult(
                name="apply_patch",
                content="patched",
                call_id="patch",
            ),
            ToolResult(
                name="search_text",
                content="matches",
                call_id="search",
            ),
            ToolResult(
                name="run_command",
                content="prefix-" + long_text + "-useful-tail",
                call_id="command",
                is_error=True,
            ),
        ),
        estimated_tokens=500,
    )
    latest = _message_unit("user", "latest", 1)
    compacted = DeterministicToolTrajectoryCompactor(
        compaction_trigger_tokens=100,
        recent_raw_tokens=1,
        max_argument_chars=40,
        max_failure_detail_chars=20,
    ).compact((unit, latest), CompactCostEstimator())
    text = _compact_text(compacted.units[0])

    assert "apply_patch src/auth.py -> success" in text
    assert "old_text" not in text
    assert "new_text" not in text
    assert long_text not in text
    assert "search_text query=needle-" in text
    assert "query=needle-\n" not in text
    assert "[REDACTED]" in text
    assert "super-secret" not in text
    assert "run_command" in text
    assert "-> failed" in text
    assert "detail: …xxxxxxx-useful-tail" in text


def test_unpaired_tool_call_is_not_compacted():
    pending = ModelContextUnit(
        items=(
            ToolCall(
                name="read_file",
                arguments={"path": "pending.py"},
                call_id="pending",
            ),
        ),
        estimated_tokens=20,
    )
    latest = _message_unit("user", "latest", 1)

    compacted = DeterministicToolTrajectoryCompactor(
        compaction_trigger_tokens=10,
        recent_raw_tokens=1,
    ).compact((pending, latest), CompactCostEstimator())

    assert compacted.units == (pending, latest)
    assert compacted.trajectory_compacted is False


def test_context_compaction_is_deterministic_immutable_and_after_m7():
    raw_old = "old-start" + "o" * 1_000 + "old-finish"
    raw_recent = "recent-start" + "r" * 1_000 + "recent-finish"
    session = Session(
        items=[
            Message(role="user", content="old request"),
            ToolCall(
                name="read_file",
                arguments={"path": "old.py"},
                call_id="old",
            ),
            ToolResult(
                name="read_file",
                content=raw_old,
                call_id="old",
            ),
            Message(role="assistant", content="bridge"),
            ToolCall(
                name="read_file",
                arguments={"path": "recent.py"},
                call_id="recent",
            ),
            ToolResult(
                name="read_file",
                content=raw_recent,
                call_id="recent",
            ),
            Message(role="user", content="latest"),
        ]
    )
    before = deepcopy(session.snapshot())
    builder = ContextBuilder(
        token_estimator=ModelContentLengthEstimator(),
        tool_result_projector=DeterministicToolResultProjector(
            max_chars=200,
            head_chars=50,
            tail_chars=50,
        ),
        trajectory_compactor=(
            DeterministicToolTrajectoryCompactor(
                compaction_trigger_tokens=300,
                recent_raw_tokens=210,
            )
        ),
    )

    first = builder.compile(session.snapshot())
    second = builder.compile(session.snapshot())

    assert first == second
    assert session.snapshot() == before
    assert session.items[2].content == raw_old
    assert session.items[5].content == raw_recent
    assert first.trajectory_compacted is True
    assert first.compacted_source_units == 1
    assert first.compacted_tool_actions == 1
    assert first.compacted_trajectory_estimated_tokens < (
        first.original_trajectory_estimated_tokens
    )
    assert first.items[0] == before[0]
    assert isinstance(first.items[1], Message)
    assert first.items[1].content.startswith(_COMPACTED_TITLE)
    assert first.items[2] == before[3]
    assert first.items[-1] == before[-1]
    recent_result = next(
        item
        for item in first.items
        if isinstance(item, ToolResult)
    )
    assert recent_result.call_id == "recent"
    assert recent_result.content != raw_recent
    assert "tool output compacted" in recent_result.content
    assert first.projected_tool_results == 1
    assert first.compacted_tool_results == 1


def test_token_budget_is_enforced_after_trajectory_compaction():
    raw_output = "x" * 1_000
    history = [
        ToolCall(
            name="read_file",
            arguments={"path": "large.py"},
            call_id="large",
        ),
        ToolResult(
            name="read_file",
            content=raw_output,
            call_id="large",
        ),
        Message(role="user", content="latest request"),
    ]
    builder = TokenBudgetContextBuilder(
        ContextBudget(max_estimated_tokens=150),
        ModelContentLengthEstimator(),
    )

    compiled = builder.compile(history)

    assert compiled.trajectory_compacted is True
    assert compiled.estimated_tokens <= 150
    assert compiled.included_units == 2
    assert compiled.dropped_units == 0
    assert isinstance(compiled.items[0], Message)
    assert compiled.items[0].content.startswith(_COMPACTED_TITLE)
    assert compiled.items[-1] == history[-1]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("compaction_trigger_tokens", 0),
        ("recent_raw_tokens", False),
        ("max_argument_chars", -1),
        ("max_failure_detail_chars", 0),
    ],
)
def test_compactor_rejects_invalid_positive_limits(name, value):
    arguments = {
        "compaction_trigger_tokens": 10,
        "recent_raw_tokens": 2,
        "max_argument_chars": 20,
        "max_failure_detail_chars": 20,
    }
    arguments[name] = value

    with pytest.raises(ValueError, match=name):
        DeterministicToolTrajectoryCompactor(**arguments)


def test_compactor_rejects_recent_reserve_at_or_above_trigger():
    with pytest.raises(
        ValueError,
        match="recent_raw_tokens must be less",
    ):
        DeterministicToolTrajectoryCompactor(
            compaction_trigger_tokens=10,
            recent_raw_tokens=10,
        )


def test_budget_relative_defaults_are_centralized_and_validated():
    compactor = default_compactor_for_history_budget(300)

    assert compactor.compaction_trigger_tokens == 240
    assert compactor.recent_raw_tokens == 100

    with pytest.raises(ValueError, match="history_budget"):
        default_compactor_for_history_budget(0)


def test_agent_uses_compacted_view_but_keeps_it_out_of_session():
    raw_output = "start" + "x" * 1_000 + "finish"
    session = Session(
        items=[
            ToolCall(
                name="read_file",
                arguments={"path": "old.py"},
                call_id="old",
            ),
            ToolResult(
                name="read_file",
                content=raw_output,
                call_id="old",
            ),
            Message(role="assistant", content="continue"),
        ]
    )
    model = RecordingModel()
    agent = Agent(
        model=model,
        session=session,
        context_builder=ContextBuilder(
            token_estimator=ModelContentLengthEstimator(),
            trajectory_compactor=(
                DeterministicToolTrajectoryCompactor(
                    compaction_trigger_tokens=150,
                    recent_raw_tokens=50,
                )
            ),
        ),
    )

    assert agent.run("latest") == "done"

    model_messages = model.contexts[0]
    compacted_messages = [
        item
        for item in model_messages
        if isinstance(item, Message)
        and item.content.startswith(_COMPACTED_TITLE)
    ]
    assert len(compacted_messages) == 1
    assert all(
        not (
            isinstance(item, Message)
            and item.content.startswith(_COMPACTED_TITLE)
        )
        for item in session.items
    )
    assert session.items[1].content == raw_output
    event = next(
        event
        for event in agent.events
        if event.type == "context_built"
    )
    assert event.data["trajectory_compacted"] is True
    assert event.data["compacted_source_units"] == 1
    assert event.data["compacted_tool_actions"] == 1
    assert event.data["original_trajectory_estimated_tokens"] > (
        event.data["compacted_trajectory_estimated_tokens"]
    )


def test_small_agent_trajectory_keeps_normal_context_and_task_state():
    model = RecordingModel()
    agent = Agent(model=model)

    assert agent.run("small request") == "done"

    assert model.contexts[0][0].role == "system"
    assert model.contexts[0][0].content.startswith(
        "[MiniHarness Derived Task State]"
    )
    assert model.contexts[0][1] == Message(
        role="user",
        content="small request",
    )
    event = next(
        event
        for event in agent.events
        if event.type == "context_built"
    )
    assert event.data["trajectory_compacted"] is False
