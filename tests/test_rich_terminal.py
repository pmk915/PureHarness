import json

from io import StringIO

import pytest


pytest.importorskip("rich")

from rich.console import Console

from pureharness.agent import Agent
from pureharness.events import AgentEvent, safe_arguments_preview
from pureharness.observability import JsonlEventRenderer, event_to_wire
from pureharness.model import AddModel
from pureharness.rich_terminal import RichTerminalRenderer
from pureharness.tools import ADD_TOOL, ToolRegistry


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
        ("en", "Retrying model request: 2/2"),
        ("zh-CN", "正在重试模型请求：2/2"),
    ],
)
def test_rich_terminal_renders_model_retry(locale, expected):
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
            type="model_retrying",
            data={"attempt": 2, "max_attempts": 2},
        )
    )

    assert expected in output.getvalue()


def test_rich_terminal_renders_interruption_separately_from_failure():
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
            type="agent_interrupted",
            data={"reason": "interrupted", "step_count": 0},
        )
    )

    rendered = output.getvalue()
    assert "Agent interrupted" in rendered
    assert "Agent failed" not in rendered


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        ("approval_requested", "Approval requested: effect"),
        ("approval_granted", "Approval granted: effect"),
        ("approval_denied", "Approval denied: effect"),
    ],
)
def test_rich_terminal_renders_approval_events(event_type, expected):
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
            type=event_type,
            data={
                "run_id": "run-1",
                "step": 0,
                "name": "effect",
                "call_id": "call-1",
            },
        )
    )

    assert expected in output.getvalue()


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


@pytest.mark.parametrize("locale, label", [
    ("en", "Verification evidence: exit_nonzero"),
    ("zh-CN", "验证执行证据：exit_nonzero"),
])
def test_verification_rendering_uses_structured_evidence_only(locale, label):
    output = StringIO()
    renderer = RichTerminalRenderer(
        locale=locale,
        console=Console(file=output, force_terminal=False, width=120),
    )
    renderer(AgentEvent(type="coding_evidence_snapshot", data={
        "last_verification_outcome": "exit_nonzero",
        "last_verification_exit_code": 1,
    }))
    assert label in output.getvalue()
    assert "exit_code=1" in output.getvalue()
    assert "task failed" not in output.getvalue().lower()


def _compact(locale="en"):
    output = StringIO()
    renderer = RichTerminalRenderer(
        locale=locale, interactive=True, detail_level="compact",
        console=Console(file=output, force_terminal=False, color_system=None, width=240),
    )
    return renderer, output


def _tool(renderer, name, arguments, *, step=0, call_id=None, is_error=False):
    identity = {"step": step, "name": name, "call_id": call_id}
    renderer(AgentEvent(type="tool_started", data={
        **identity, "arguments_preview": safe_arguments_preview(arguments),
    }))
    renderer(AgentEvent(type="tool_completed", data={**identity, "is_error": is_error}))


def _verification(renderer, count, outcome, code, *, step=2, run_id=None):
    renderer(AgentEvent(type="coding_evidence_snapshot", run_id=run_id, data={
        "verification_attempts": count,
        "last_verification_step": step,
        "last_verification_outcome": outcome,
        "last_verification_exit_code": code,
    }))


def test_direct_renderer_still_defaults_to_verbose_and_rejects_invalid_detail():
    assert RichTerminalRenderer().detail_level == "verbose"
    with pytest.raises(ValueError, match="Unsupported detail level"):
        RichTerminalRenderer(detail_level="quiet")


def test_compact_hides_routine_diagnostics_but_verbose_keeps_rechecks():
    renderer, output = _compact()
    events = [
        AgentEvent(type="context_build_started", data={}),
        AgentEvent(type="context_built", data={"context_strategy": "FullHistory", "estimated_history_tokens": 99}),
        AgentEvent(type="model_started", data={}),
        AgentEvent(type="model_completed", data={"tool_call_count": 2}),
        AgentEvent(type="tool_policy_evaluated", data={"name": "read_file", "decision": "allow"}),
        AgentEvent(type="completion_recheck_requested", data={"reason": "verification_failed_after_mutation"}),
        AgentEvent(type="completion_recheck_skipped", data={"reason": "verification_failed_after_mutation", "skip_reason": "limit"}),
        AgentEvent(type="coding_evidence_snapshot", data={"verification_attempts": 0}),
    ]
    for event in events:
        renderer(event)
    assert output.getvalue() == ""
    verbose = RichTerminalRenderer(console=Console(file=output, color_system=None))
    for event in events[5:7]:
        verbose(event)
    assert "Completion recheck requested" in output.getvalue()
    assert "Completion recheck skipped" in output.getvalue()


@pytest.mark.parametrize("locale, expected", [
    ("en", ["Inspected workspace", "Read calc.py", "Wrote result.py", "Updated calc.py"]),
    ("zh-CN", ["已查看 工作区", "已读取 calc.py", "已写入 result.py", "已更新 calc.py"]),
])
def test_compact_file_actions_are_semantic_and_do_not_dump_content(locale, expected):
    renderer, output = _compact(locale)
    _tool(renderer, "list_files", {})
    _tool(renderer, "read_file", {"path": "calc.py"}, call_id="read")
    _tool(renderer, "read_file_range", {"path": "calc.py", "start_line": 1, "end_line": 2})
    _tool(renderer, "write_file", {"path": "result.py", "content": "private-file-content"})
    _tool(renderer, "apply_patch", {"path": "calc.py", "old_text": "old-content", "new_text": "new-content"})
    _tool(renderer, "custom_tool", {"body": "private-custom-content"})
    text = output.getvalue()
    for label in expected:
        assert f"✓ {label}" in text
    assert "✓ custom_tool" in text
    assert "content" not in text
    assert "Calling tool" not in text
    assert "Tool completed" not in text


def test_compact_call_identity_does_not_attach_wrong_path_to_completion():
    renderer, output = _compact()
    renderer(AgentEvent(type="tool_started", data={"step": 1, "name": "read_file", "call_id": "first", "arguments_preview": {"path": "first.py"}}))
    renderer(AgentEvent(type="tool_completed", data={"step": 1, "name": "read_file", "call_id": "second", "is_error": False}))
    assert output.getvalue().strip() == "✓ read_file"
    renderer(AgentEvent(type="tool_completed", data={"step": 1, "name": "read_file", "call_id": "first", "is_error": False}))
    assert "✓ Read first.py" in output.getvalue()


def test_compact_commands_use_only_bounded_redacted_preview_without_claiming_exit_success():
    renderer, output = _compact()
    _tool(renderer, "run_command", {"argv": ["python3", "-m", "unittest", "-v"]})
    _tool(renderer, "run_command", {"argv": ["echo", "two words"]})
    _tool(renderer, "run_command", {"argv": ["echo", "two words", "api_key=topsecret"]})
    _tool(renderer, "run_command", {"argv": ["echo", "x" * 200]})
    text = output.getvalue()
    assert "● Run python3 -m unittest -v" in text
    assert "'two words'" in text
    assert "[REDACTED]" in text
    assert "topsecret" not in text
    assert "x" * 121 not in text
    assert "…" in text
    assert "Verification" not in text
    assert "✓" not in text
    assert "success" not in text.lower()


def test_compact_previews_preserve_spaces_and_escape_newlines_without_shell_parsing():
    renderer, output = _compact()
    _tool(renderer, "read_file", {"path": "two  spaces.py"})
    _tool(renderer, "run_command", {"argv": ["echo", "two  spaces", "line\nbreak"]})
    assert "✓ Read two  spaces.py" in output.getvalue()
    assert "'two  spaces'" in output.getvalue()
    assert r"line\nbreak" in output.getvalue()
    assert len(output.getvalue().splitlines()) == 2


@pytest.mark.parametrize("event_type, data, expected", [
    ("tool_policy_evaluated", {"name": "write_file", "decision": "deny"}, "Policy denied"),
    ("tool_policy_evaluated", {"name": "run_command", "decision": "require_approval"}, "Approval required"),
    ("approval_requested", {"name": "run_command"}, "Approval requested"),
    ("approval_granted", {"name": "run_command"}, "Approval granted"),
    ("approval_denied", {"name": "run_command"}, "Approval denied"),
    ("workspace_precondition_failed", {"name": "write_file", "path": "calc.py", "reason": "not_read"}, "Workspace precondition failed"),
    ("context_build_failed", {}, "Context build failed"),
    ("context_window_exceeded", {}, "Provider context window exceeded"),
    ("context_recovering", {}, "Rebuilding smaller context"),
    ("model_retrying", {"attempt": 2, "max_attempts": 2}, "Retrying model request · 2/2"),
    ("model_failed", {}, "Model request failed"),
    ("execution_budget_exhausted", {}, "Execution budget exhausted"),
    ("agent_interrupted", {}, "Agent interrupted"),
    ("agent_failed", {"reason": "model_error"}, "Run failed · model_error"),
])
def test_compact_keeps_exceptional_events_visible(event_type, data, expected):
    renderer, output = _compact()
    renderer(AgentEvent(type=event_type, data=data))
    assert expected in output.getvalue()


def test_compact_tool_errors_and_early_rejections_stay_visible_without_count_duplicates():
    renderer, output = _compact()
    _tool(renderer, "read_file", {"path": "missing.py"}, is_error=True)
    renderer(AgentEvent(type="tool_policy_evaluated", data={"name": "write_file", "decision": "deny"}))
    renderer(AgentEvent(type="progress_snapshot", data={"failed_tool_results": 2}))
    assert "Tool failed · Read missing.py" in output.getvalue()
    assert "Tool error observations" not in output.getvalue()
    renderer(AgentEvent(type="progress_snapshot", data={"failed_tool_results": 3}))
    assert "Tool error observations · 1" in output.getvalue()


@pytest.mark.parametrize("locale, failed, passed, error", [
    ("en", "Verification failed · exit 1", "Verification passed", "Verification could not complete"),
    ("zh-CN", "验证失败 · 退出码 1", "验证通过", "验证未能完成"),
])
def test_compact_verification_deduplicates_repeated_snapshots_but_not_new_attempts(locale, failed, passed, error):
    renderer, output = _compact(locale)
    for _ in range(3):
        _verification(renderer, 1, "exit_nonzero", 1)
    _tool(renderer, "apply_patch", {"path": "calc.py"}, step=3)
    _verification(renderer, 1, "exit_nonzero", 1)
    for _ in range(3):
        _verification(renderer, 2, "exit_zero", 0, step=4)
    text = output.getvalue()
    assert text.count(failed) == 1
    assert text.count(passed) == 1
    # Identical outcomes in the same step still count as genuine new attempts.
    _verification(renderer, 3, "exit_zero", 0, step=4)
    assert output.getvalue().count(passed) == 2
    _verification(renderer, 4, "tool_error", None, step=5)
    assert error in output.getvalue()
    assert "exit_nonzero" not in output.getvalue()


def test_compact_verification_identity_fallback_and_new_run_reset():
    renderer, output = _compact()
    data = {"last_verification_step": 0, "last_verification_outcome": "exit_nonzero", "last_verification_exit_code": 1}
    for _ in range(2):
        renderer(AgentEvent(type="coding_evidence_snapshot", data=data))
    assert output.getvalue().count("Verification failed") == 1
    renderer(AgentEvent(type="agent_started", data={}))
    _verification(renderer, 1, "exit_zero", 0, run_id="run-one")
    _verification(renderer, 1, "exit_zero", 0, run_id="run-two")
    assert output.getvalue().count("Verification passed") == 2


def test_compact_summary_is_run_scoped_and_counts_reset_between_runs():
    renderer, output = _compact()
    for _ in range(2):
        renderer(AgentEvent(type="agent_started", data={}))
        renderer(AgentEvent(type="model_completed", data={"tool_call_count": 3}))
        renderer(AgentEvent(type="model_completed", data={"tool_call_count": 0}))
        renderer(AgentEvent(type="progress_snapshot", data={"tool_calls": 2}))
        renderer(AgentEvent(type="agent_completed", data={"step_count": 2}))
    assert output.getvalue().count("Run completed · 2 steps · 3 tool calls") == 2
    assert "Task" not in output.getvalue()
    assert "success" not in output.getvalue().lower()


def test_compact_is_only_presentation_and_full_jsonl_evidence_is_unchanged():
    renderer, human = _compact()
    machine = StringIO()
    received = []
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    agent = Agent(model=AddModel(), tools=registry, max_steps=2,
                  listeners=[renderer, JsonlEventRenderer(lambda line: machine.write(line + "\n")), received.append])
    assert agent.run("calculate") == "The result is 29"
    assert received == agent.events
    assert [json.loads(line) for line in machine.getvalue().splitlines()] == [event_to_wire(event) for event in received]
    assert any(event.type == "context_built" for event in received)
    assert "Building context" not in human.getvalue()
    record = agent.last_run_record
    assert record.model_call_count == len(record.model_invocations) == 2
    assert record.tool_call_count == 1
    assert "Run completed · 2 steps · 1 tool calls" in human.getvalue()
    assert len(agent.session.items) == 4
    assert agent.listener_errors == []


def test_broken_compact_console_does_not_stop_agent_or_other_listeners():
    class BrokenConsole:
        def print(self, *args, **kwargs):
            raise RuntimeError("console failed")

    renderer = RichTerminalRenderer(console=BrokenConsole(), detail_level="compact")
    registry = ToolRegistry()
    registry.register(ADD_TOOL)
    received = []
    agent = Agent(model=AddModel(), tools=registry, max_steps=2, listeners=[renderer, received.append])
    assert agent.run("calculate") == "The result is 29"
    assert received == agent.events
    assert agent.listener_errors
    assert all(isinstance(error, RuntimeError) for error in agent.listener_errors)
