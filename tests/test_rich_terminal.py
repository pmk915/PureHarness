from io import StringIO

import pytest


pytest.importorskip("rich")

from rich.console import Console

from miniharness.agent import Agent
from miniharness.events import AgentEvent
from miniharness.model import AddModel
from miniharness.rich_terminal import RichTerminalRenderer
from miniharness.tools import ADD_TOOL, ToolRegistry


def _render_run(locale: str) -> str:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=False,
        color_system=None,
        width=100,
    )
    renderer = RichTerminalRenderer(
        locale=locale,
        console=console,
    )
    registry = ToolRegistry()
    registry.register(ADD_TOOL)

    agent = Agent(
        model=AddModel(),
        tools=registry,
        max_steps=2,
        listeners=[renderer],
    )

    assert agent.run("calculate") == "The result is 29"

    return output.getvalue()


def test_rich_terminal_renders_english_labels():
    output = _render_run("en")

    assert "Building context" in output
    assert "estimated history tokens:" in output
    assert "units: 1/1" in output
    assert "tool outputs compacted: 0" in output
    assert "task state: 0 modified files · 0 recent errors" in output
    assert "tools exposed: 1/1 · ~" in output
    assert "schema tokens" in output
    assert "Tool policy evaluated: add" in output
    assert "decision: allow" in output
    assert "Calling tool: add" in output
    assert "a: 12" in output
    assert "Tool completed" in output
    assert "Task completed" in output


def test_rich_terminal_renders_chinese_labels():
    output = _render_run("zh-CN")

    assert "正在构建上下文" in output
    assert "估算历史 tokens：" in output
    assert "单元：1/1" in output
    assert "工具输出压缩：0" in output
    assert "任务状态：0 个修改文件 · 0 个近期错误" in output
    assert "工具暴露：1/1 · 约 " in output
    assert "schema tokens" in output
    assert "工具策略已评估：add" in output
    assert "决策：allow" in output
    assert "正在调用工具：add" in output
    assert "a: 12" in output
    assert "工具执行完成" in output
    assert "任务完成" in output


def test_rich_terminal_rejects_unsupported_locale():
    with pytest.raises(
        ValueError,
        match="Unsupported locale: fr",
    ):
        RichTerminalRenderer(locale="fr")


def test_rich_terminal_renders_context_build_failure():
    output = StringIO()
    renderer = RichTerminalRenderer(
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
        )
    )

    renderer(
        AgentEvent(
            type="context_build_failed",
            data={
                "step": 0,
                "reason": "context_error",
                "error_type": "ContextBudgetExceeded",
            },
        )
    )

    assert "Context build failed" in output.getvalue()


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        (
            "en",
            "trajectory compacted: 3 old units → 90 estimated tokens",
        ),
        (
            "zh-CN",
            "轨迹已压缩：3 个旧单元 → 90 估算 tokens",
        ),
    ],
)
def test_rich_terminal_renders_one_compaction_line(locale, expected):
    output = StringIO()
    renderer = RichTerminalRenderer(
        locale=locale,
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
        ),
    )
    renderer(
        AgentEvent(
            type="context_built",
            data={
                "history_item_count": 10,
                "context_item_count": 5,
                "context_strategy": "FullHistory",
                "estimated_history_tokens": 100,
                "included_units": 5,
                "total_units": 5,
                "compacted_tool_results": 1,
                "files_modified_count": 0,
                "recent_errors_count": 0,
                "trajectory_compacted": True,
                "compacted_source_units": 3,
                "compacted_trajectory_estimated_tokens": 90,
            },
        )
    )

    rendered = output.getvalue()
    assert expected in rendered
    assert rendered.count(expected) == 1


def test_broken_renderer_does_not_stop_other_observers():
    class BrokenRenderer:
        def __call__(self, event):
            raise RuntimeError("render failed")

    received_events = []
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    agent = Agent(
        model=AddModel(),
        tools=registry,
        max_steps=2,
        listeners=[
            BrokenRenderer(),
            received_events.append,
        ],
    )

    assert agent.run("calculate") == "The result is 29"
    assert received_events == agent.events
    assert len(agent.listener_errors) == len(agent.events)
    assert all(
        isinstance(error, RuntimeError)
        for error in agent.listener_errors
    )
