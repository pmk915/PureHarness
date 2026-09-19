import pytest

from miniharness.messages import ToolResult
from miniharness.tool_result_projection import (
    DeterministicToolResultProjector,
    IdentityToolResultProjector,
)


def test_identity_projector_returns_equivalent_fresh_result():
    raw = ToolResult(
        name="read_file",
        content="small result",
        call_id="call-1",
        is_error=False,
    )

    projected = IdentityToolResultProjector().project(raw)

    assert projected == raw
    assert projected is not raw


def test_compacting_projector_leaves_small_result_unchanged():
    raw = ToolResult(
        name="read_file",
        content="small result",
        call_id="call-1",
    )
    projector = DeterministicToolResultProjector(
        max_chars=100,
        head_chars=20,
        tail_chars=20,
    )

    projected = projector.project(raw)

    assert projected == raw
    assert projected.content == "small result"


def test_compacting_projector_keeps_head_tail_and_metadata():
    content = "H" * 100 + "M" * 300 + "T" * 100
    raw = ToolResult(
        name="run_command",
        content=content,
        call_id="call-2",
        is_error=True,
    )
    projector = DeterministicToolResultProjector(
        max_chars=100,
        head_chars=20,
        tail_chars=30,
    )

    projected = projector.project(raw)

    assert projected.content.startswith("H" * 20)
    assert projected.content.endswith("T" * 30)
    assert "450 characters omitted" in projected.content
    assert "original: 500" in projected.content
    assert "retained: 50" in projected.content
    assert projected.name == raw.name
    assert projected.call_id == raw.call_id
    assert projected.is_error is True


def test_compacting_projector_is_deterministic():
    raw = ToolResult(
        name="git_diff",
        content="a" * 500,
        call_id=None,
    )
    projector = DeterministicToolResultProjector(
        max_chars=100,
        head_chars=20,
        tail_chars=20,
    )

    assert projector.project(raw) == projector.project(raw)


def test_compacting_projector_does_not_expand_near_threshold_result():
    raw = ToolResult(
        name="read_file",
        content="x" * 101,
    )
    projector = DeterministicToolResultProjector(
        max_chars=100,
        head_chars=20,
        tail_chars=20,
    )

    assert projector.project(raw) == raw


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 0, "head_chars": 1, "tail_chars": 1},
        {"max_chars": True, "head_chars": 1, "tail_chars": 1},
        {"max_chars": 10, "head_chars": 0, "tail_chars": 1},
        {"max_chars": 10, "head_chars": 1, "tail_chars": -1},
        {"max_chars": 10, "head_chars": 5, "tail_chars": 5},
    ],
)
def test_compacting_projector_rejects_invalid_configuration(
    kwargs,
):
    with pytest.raises(ValueError):
        DeterministicToolResultProjector(**kwargs)
