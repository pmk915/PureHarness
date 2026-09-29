from importlib.resources import files

import pytest

from pureharness.agent import Agent
from pureharness.context import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextBuilder,
    ContextLimits,
    RecentContextBuilder,
    TokenBudgetContextBuilder,
)
from pureharness.messages import AgentItem, Message, ToolCall
from pureharness.model import ContextWindowExceededError
from pureharness.session import Session
from pureharness.skills import (
    Skill,
    SkillDocumentError,
    default_coding_skills,
    load_builtin_skill,
    parse_skill_document,
    render_skill,
)
from pureharness.tools import Tool, ToolRegistry


VALID_DOCUMENT = """---
name: coding-task
version: 1
description: Focused coding procedure.
---

# Procedure

Inspect, change, and verify.
"""


class RecordingModel:
    def __init__(self, outputs=None) -> None:
        self.outputs = list(
            outputs
            if outputs is not None
            else [Message(role="assistant", content="done")]
        )
        self.contexts: list[list[AgentItem]] = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class SkillAwareEstimator:
    def __init__(
        self,
        message_costs: dict[str, int],
        *,
        skill_tokens: int = 2,
        task_state_tokens: int = 1,
    ) -> None:
        self.message_costs = message_costs
        self.skill_tokens = skill_tokens
        self.task_state_tokens = task_state_tokens

    def estimate(self, items) -> int:
        if not items:
            return 0
        first = items[0]
        if isinstance(first, Message) and first.role == "system":
            if first.content.startswith("[PureHarness Active Skill]"):
                return self.skill_tokens
            return self.task_state_tokens
        if isinstance(first, Message):
            return self.message_costs[first.content]
        raise AssertionError("unexpected non-message context unit")


def _skill(name: str = "coding-task", version: int = 1) -> Skill:
    return Skill(
        name=name,
        version=version,
        description=f"Procedure for {name}.",
        instructions=f"Follow the {name} procedure.",
    )


def _skill_messages(items) -> list[Message]:
    return [
        item
        for item in items
        if isinstance(item, Message)
        and item.role == "system"
        and item.content.startswith("[PureHarness Active Skill]")
    ]


def test_skill_is_frozen_and_validated():
    skill = _skill()

    assert skill.identifier == "coding-task@1"
    with pytest.raises(AttributeError):
        skill.name = "changed"  # type: ignore[misc]

    for name in ("", "Coding Task", "coding_task", "-coding"):
        with pytest.raises(ValueError, match="stable identifier"):
            _skill(name=name)
    for version in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            _skill(version=version)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="description"):
        Skill("coding", 1, " ", "instructions")
    with pytest.raises(ValueError, match="instructions"):
        Skill("coding", 1, "description", " ")


def test_parse_skill_document_preserves_metadata_and_body():
    skill = parse_skill_document(VALID_DOCUMENT)

    assert skill == Skill(
        name="coding-task",
        version=1,
        description="Focused coding procedure.",
        instructions="# Procedure\n\nInspect, change, and verify.",
    )


@pytest.mark.parametrize(
    "document, message",
    [
        ("name: coding-task\n---\nbody", "opening"),
        ("---\nname: coding-task\nbody", "closing"),
        (
            VALID_DOCUMENT.replace(
                "description: Focused coding procedure.\n",
                "description: Focused coding procedure.\nunknown: value\n",
            ),
            "Unknown",
        ),
        (
            VALID_DOCUMENT.replace(
                "version: 1\n",
                "version: 1\nversion: 2\n",
            ),
            "Duplicate",
        ),
        (VALID_DOCUMENT.replace("description: Focused coding procedure.\n", ""), "Missing"),
        (VALID_DOCUMENT.replace("version: 1", "version: false"), "positive integer"),
        (VALID_DOCUMENT.replace("version: 1", "version: 0"), "positive integer"),
        (VALID_DOCUMENT.split("# Procedure")[0], "body"),
    ],
)
def test_parse_skill_document_rejects_invalid_documents(document, message):
    with pytest.raises(SkillDocumentError, match=message):
        parse_skill_document(document)


def test_builtin_skill_loads_through_package_resources():
    resource = files("pureharness").joinpath(
        "builtin_skills", "coding-task", "SKILL.md"
    )

    assert resource.is_file()
    skill = load_builtin_skill("coding-task")
    assert skill.name == "coding-task"
    assert skill.version == 1
    assert skill.description == (
        "Complete coding changes through focused implementation and "
        "verification."
    )
    assert "make a focused change" in skill.instructions.lower()
    assert default_coding_skills() == (skill,)


def test_skill_rendering_is_deterministic_and_path_free():
    skill = _skill()

    assert render_skill(skill) == render_skill(skill)
    assert render_skill(skill) == Message(
        role="system",
        content=(
            "[PureHarness Active Skill]\n"
            "Name: coding-task\n"
            "Version: 1\n"
            "Description: Procedure for coding-task.\n\n"
            "Follow the coding-task procedure."
        ),
    )
    assert "/home/" not in render_skill(skill).content


def test_generic_agent_has_no_skill_and_reports_explicit_zero_telemetry():
    model = RecordingModel()
    agent = Agent(model=model)

    assert agent.run("request") == "done"

    assert agent.active_skill_ids == ()
    assert _skill_messages(model.contexts[0]) == []
    event = next(event for event in agent.events if event.type == "context_built")
    assert event.data["active_skill_count"] == 0
    assert event.data["active_skill_ids"] == []
    assert event.data["estimated_skill_tokens"] == 0


def test_skill_precedes_task_state_and_trajectory_without_entering_session():
    skill = _skill()
    model = RecordingModel()
    session = Session()
    agent = Agent(model=model, session=session, skills=(skill,))

    assert agent.run("request") == "done"

    context = model.contexts[0]
    assert context[0] == render_skill(skill)
    assert isinstance(context[1], Message)
    assert context[1].role == "system"
    assert context[2] == Message(role="user", content="request")
    assert all(item.role != "system" for item in session.items if isinstance(item, Message))
    assert agent.active_skill_ids == ("coding-task@1",)


def test_skill_remains_when_task_state_is_disabled():
    skill = _skill()
    model = RecordingModel()
    agent = Agent(
        model=model,
        include_task_state=False,
        skills=(skill,),
    )

    agent.run("request")

    assert model.contexts[0] == [
        render_skill(skill),
        Message(role="user", content="request"),
    ]


def test_skill_appears_once_per_tool_loop_inference_and_never_in_session():
    skill = _skill()
    model = RecordingModel(
        [
            [ToolCall(name="lookup", arguments={}, call_id="lookup-1")],
            Message(role="assistant", content="done"),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="Lookup.",
            parameters={"type": "object", "properties": {}},
            function=lambda: "result",
        )
    )
    session = Session()
    agent = Agent(
        model=model,
        tools=registry,
        session=session,
        skills=(skill,),
        max_steps=2,
    )

    agent.run("request")

    assert len(model.contexts) == 2
    assert all(len(_skill_messages(context)) == 1 for context in model.contexts)
    assert _skill_messages(session.items) == []


def test_repeated_runs_reinject_without_accumulating_skill_in_session():
    skill = _skill()
    model = RecordingModel(
        [
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
        ]
    )
    session = Session()
    agent = Agent(model=model, session=session, skills=(skill,))

    assert agent.run("one") == "first"
    assert agent.run("two") == "second"

    assert all(len(_skill_messages(context)) == 1 for context in model.contexts)
    assert _skill_messages(session.items) == []
    assert [item.content for item in session.items if isinstance(item, Message)] == [
        "one",
        "first",
        "two",
        "second",
    ]


def test_active_skill_order_is_stable_and_duplicate_names_are_rejected():
    first = _skill("first")
    second = _skill("second", version=2)
    agent = Agent(model=RecordingModel(), skills=(second, first))

    assert agent.active_skill_ids == ("second@2", "first@1")
    with pytest.raises(ValueError, match="duplicates"):
        Agent(model=RecordingModel(), skills=(first, _skill("first", 2)))


@pytest.mark.parametrize(
    "builder",
    [
        ContextBuilder(),
        RecentContextBuilder(max_items=1),
        TokenBudgetContextBuilder(ContextBudget(100)),
    ],
)
def test_context_strategies_keep_skill_outside_compiled_history(builder):
    skill = _skill()
    model = RecordingModel()
    agent = Agent(
        model=model,
        session=Session(items=[Message(role="user", content="old")]),
        context_builder=builder,
        include_task_state=False,
        skills=(skill,),
    )

    agent.run("new")

    assert model.contexts[0][0] == render_skill(skill)
    assert len(_skill_messages(model.contexts[0])) == 1


def test_skill_cost_reduces_history_budget_and_triggers_bounding():
    skill = _skill()

    def run(skills):
        model = RecordingModel()
        agent = Agent(
            model=model,
            session=Session(items=[Message(role="user", content="old")]),
            context_builder=ContextBuilder(
                token_estimator=SkillAwareEstimator(
                    {"old": 3, "new": 3}, skill_tokens=2
                )
            ),
            include_task_state=False,
            context_limits=ContextLimits(10, 4),
            skills=skills,
        )
        agent.run("new")
        event = next(
            event for event in agent.events if event.type == "context_built"
        )
        return model, event

    plain_model, plain_event = run(())
    skill_model, skill_event = run((skill,))

    assert plain_event.data["bounded_history_applied"] is False
    assert plain_model.contexts[0] == [
        Message(role="user", content="old"),
        Message(role="user", content="new"),
    ]
    assert skill_event.data["estimated_skill_tokens"] == 2
    assert skill_event.data["available_history_tokens"] == 4
    assert skill_event.data["bounded_history_applied"] is True
    assert skill_model.contexts[0] == [
        render_skill(skill),
        Message(role="user", content="new"),
    ]


def test_pinned_skill_fixed_cost_can_fail_before_model_invocation():
    model = RecordingModel()
    agent = Agent(
        model=model,
        context_builder=ContextBuilder(
            token_estimator=SkillAwareEstimator(
                {"new": 1}, skill_tokens=3
            )
        ),
        include_task_state=False,
        context_limits=ContextLimits(5, 2),
        skills=(_skill(),),
    )

    with pytest.raises(ContextBudgetExceeded, match="active Skills"):
        agent.run("new")

    assert model.contexts == []
    assert any(event.type == "context_build_failed" for event in agent.events)


def test_provider_overflow_recovery_preserves_one_pinned_skill():
    skill = _skill()
    model = RecordingModel(
        [
            ContextWindowExceededError("too large"),
            Message(role="assistant", content="done"),
        ]
    )
    agent = Agent(
        model=model,
        session=Session(items=[Message(role="user", content="old")]),
        context_builder=ContextBuilder(
            token_estimator=SkillAwareEstimator(
                {"old": 4, "new": 4}, skill_tokens=2
            )
        ),
        include_task_state=False,
        skills=(skill,),
    )

    assert agent.run("new") == "done"

    assert len(model.contexts) == 2
    assert all(context[0] == render_skill(skill) for context in model.contexts)
    assert all(len(_skill_messages(context)) == 1 for context in model.contexts)
    assert model.contexts[1][1:] == [Message(role="user", content="new")]
    recovery = next(
        event for event in agent.events if event.type == "context_recovering"
    )
    assert recovery.data["previous_estimated_request_tokens"] == 10
    assert recovery.data["recovered_estimated_request_tokens"] == 6
    record = agent.last_run_record
    assert record is not None
    assert record.model_call_count == len(record.model_invocations) == 1
    assert "estimated_skill_tokens" not in record.to_dict()


def test_skill_guidance_does_not_delay_message_completion():
    model = RecordingModel()
    agent = Agent(model=model, skills=(_skill(),))

    assert agent.run("request") == "done"
    assert len(model.contexts) == 1
    assert agent.trace.end_reason == "completed"
