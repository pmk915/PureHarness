import builtins
import json
import sys

from datetime import datetime, timedelta, timezone
from io import StringIO

import pytest

import pureharness.cli as cli
from pureharness.agent import Agent
from pureharness.evaluation import EvaluationReportBuilder, trajectory_from_run_record
from pureharness.interactive import InteractiveCommands
from pureharness.messages import Message, ToolCall
from pureharness.model import ModelError
from pureharness.session import Session
from pureharness.session_store import DurableSession, JsonlDurableSessionStore
from pureharness.tools import ToolRegistry


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.contexts = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        result = next(self.outputs)
        if isinstance(result, BaseException):
            raise result
        return result


def _input(values):
    values = iter(values)
    return lambda prompt: next(values)


def _no_model(name):
    pytest.fail("read-only commands must not construct a model")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("PUREHARNESS_HOME", str(home))
    return workspace, JsonlDurableSessionStore(home / "sessions")


def _seed(store, workspace, session_id, updated, *, with_run=False):
    session = Session()
    records = []
    if with_run:
        agent = Agent(
            model=ScriptedModel([Message(role="assistant", content="historical")]),
            tools=ToolRegistry(),
            session=session,
            session_id=session_id,
            run_id_factory=lambda: f"{session_id}-run",
        )
        agent.run("historical request")
        records.append(agent.last_run_record)
    state = DurableSession(
        session_id=session_id,
        created_at=updated - timedelta(hours=1),
        updated_at=updated,
        workspace=workspace.resolve(),
        model="saved-model",
        session=session,
        run_records=records,
    )
    store.save(state)
    return state


def test_all_slash_commands_are_read_only_and_need_no_credentials(environment, monkeypatch):
    workspace, store = environment
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: pytest.fail("no credentials needed"))
    output = []
    assert cli.main(
        ["--workspace", str(workspace)],
        model_factory=_no_model,
        input_fn=_input(["/help", "/status", "/session", "/runs", "/eval", "/invalid", "/exit"]),
        output_fn=output.append,
    ) == 0
    state = store.load(store.list_sessions()[0].session_id)
    assert state.run_records == []
    assert state.session.items == []
    assert "History items: 0" in output
    assert "Runs in session: 0" in output
    assert "No finalized Run to evaluate yet." in output
    assert "No finalized Runs in this session yet." in output
    assert "Unknown command: /invalid" in output
    for command in ("/help", "/status", "/session", "/runs", "/eval", "/exit"):
        assert any(line.lstrip().startswith(command + " ") for line in output)
    assert f"Created (UTC): {state.created_at.strftime('%Y-%m-%dT%H:%M:%SZ')}" in output
    assert f"Updated (UTC): {state.updated_at.strftime('%Y-%m-%dT%H:%M:%SZ')}" in output


def test_compact_evaluation_and_eval_use_existing_m22_rules(environment):
    workspace, store = environment
    model = ScriptedModel([
        Message(role="assistant", content="first response"),
        Message(role="assistant", content="second response"),
    ])
    output = []
    assert cli.main(
        ["--workspace", str(workspace), "--model", "scripted"],
        model_factory=lambda name: model,
        input_fn=_input(["first request", "/eval", "/session", "/runs", "second request", "/exit"]),
        output_fn=output.append,
    ) == 0
    state = store.load(store.list_sessions()[0].session_id)
    report = EvaluationReportBuilder().build(trajectory_from_run_record(state.run_records[0]))
    assert output.count("Run evaluation") == 2
    assert output.count("Execution evaluation") == 1
    assert f"  Completion: {report.metrics.task_success_score:.3f}" in output
    assert "  Step efficiency: 1.000" in output
    assert "  Tool reliability: 1.000" in output
    assert "  Type: none" in output
    assert "  Action: none" in output
    assert "History items: 2" in output
    record = state.run_records[0]
    assert f"{record.run_id}\tcompleted\t1\t0" in output
    assert len(model.contexts) == 2
    assert not any(
        isinstance(item, Message) and (
            "Run evaluation" in item.content or "Recovery (advisory only)" in item.content
        )
        for item in model.contexts[1]
    )
    assert "Completion describes protocol execution, not verified task correctness." in output


def test_incomplete_run_evaluation_is_advisory_and_does_not_retry(environment):
    workspace, store = environment
    model = ScriptedModel([
        [ToolCall(name="read_file", arguments={"path": "missing.txt"}, call_id="missing")],
        ModelError("scripted failure"),
    ])
    output = []
    assert cli.main(
        ["--workspace", str(workspace)],
        model_factory=lambda name: model,
        input_fn=_input(["read missing file", "/eval", "/status", "/exit"]),
        output_fn=output.append,
    ) == 0
    record = store.load(store.list_sessions()[0].session_id).run_records[0]
    assert record.end_reason == "model_error"
    assert record.tool_result_error_count == 1
    assert len(model.contexts) == 2
    assert "  Completion: 0.000" in output
    assert "  Tool reliability: 0.000" in output
    assert "  Diagnosis: tool_failure" in output
    assert "  Recovery (advisory only): retry_tool" in output
    assert "  Type: tool_failure" in output
    assert "  Action: retry_tool" in output
    assert "Last end reason: model_error" in output


def test_eval_of_interrupted_run_returns_to_prompt(environment):
    workspace, _ = environment
    output = []
    assert cli.main(
        ["--workspace", str(workspace)],
        model_factory=lambda name: ScriptedModel([KeyboardInterrupt()]),
        input_fn=_input(["interrupt", "/eval", "/exit"]),
        output_fn=output.append,
    ) == 0
    assert "Run interrupted." in output
    assert "  End reason: interrupted" in output
    assert "  Completion: 0.000" in output


@pytest.mark.parametrize("schema_version", [1, 2])
def test_eval_after_resume_reads_persisted_record_without_model(environment, monkeypatch, schema_version):
    workspace, store = environment
    state = _seed(store, workspace, "persisted", datetime.now(timezone.utc), with_run=True)
    from pureharness.run_record import RunRecord

    data = state.run_records[0].to_dict()
    data["schema_version"] = schema_version
    state.run_records = [RunRecord.from_dict(data)]
    store.save(state)
    before = store.load(state.session_id)
    monkeypatch.setattr(cli, "_create_agent", lambda *a, **k: pytest.fail("resume is passive"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: pytest.fail("no credentials needed"))
    output = []
    assert cli.main(
        ["resume", state.session_id],
        model_factory=_no_model,
        input_fn=_input(["/eval", "/runs", "/session", "/exit"]),
        output_fn=output.append,
    ) == 0
    after = store.load(state.session_id)
    assert "  ID: persisted-run" in output
    assert "  End reason: completed" in output
    assert "  Action: none" in output
    assert after.run_records == before.run_records
    assert after.session.items == before.session.items
    assert after.updated_at == before.updated_at


def test_continue_does_not_reexecute_historical_mutation(environment):
    workspace, store = environment
    model = ScriptedModel([
        [ToolCall(name="write_file", arguments={"path": "marker.txt", "content": "original"})],
        Message(role="assistant", content="done"),
        Message(role="assistant", content="confirmed"),
    ])
    assert cli.main(
        ["--workspace", str(workspace)],
        model_factory=lambda name: model,
        input_fn=_input(["create marker", "/exit"]),
        output_fn=lambda value: None,
    ) == 0
    marker = workspace / "marker.txt"
    marker.write_text("changed after run", encoding="utf-8")
    assert cli.main(
        ["--workspace", str(workspace), "--continue"],
        model_factory=_no_model,
        input_fn=_input(["/eval", "/runs", "/exit"]),
        output_fn=lambda value: None,
    ) == 0
    assert marker.read_text(encoding="utf-8") == "changed after run"
    assert len(model.contexts) == 3
    assert len(store.load(store.list_sessions()[0].session_id).run_records) == 1


def test_runs_limits_display_to_latest_ten_without_changing_history(environment):
    workspace, store = environment
    state = _seed(store, workspace, "many", datetime.now(timezone.utc))
    agent = Agent(
        model=ScriptedModel([Message(role="assistant", content="done")] * 12),
        tools=ToolRegistry(), session_id=state.session_id,
    )
    for _ in range(12):
        agent.run("hello")
        state.run_records.append(agent.last_run_record)
    output = []
    InteractiveCommands(state, output.append).runs()
    assert len(output) == 11
    assert state.run_records[0].run_id not in "\n".join(output)
    assert state.run_records[2].run_id in "\n".join(output)
    assert state.run_records[-1].run_id in "\n".join(output)
    assert len(state.run_records) == 12


def test_continue_selects_newest_matching_workspace_and_is_passive(environment, monkeypatch):
    workspace, store = environment
    other = workspace.parent / "other"
    other.mkdir()
    now = datetime.now(timezone.utc)
    # Deliberately choose IDs whose lexical ordering differs from timestamp order.
    _seed(store, workspace, "z-old", now - timedelta(hours=2))
    expected = _seed(store, workspace, "a-new", now, with_run=True)
    _seed(store, other, "other-newest", now + timedelta(hours=1))
    before = store.load(expected.session_id)
    monkeypatch.setattr(cli, "_create_agent", lambda *a, **k: pytest.fail("must not replay tools"))
    output = []
    assert cli.main(
        ["--workspace", str(workspace / "."), "--continue", "--model", "ignored"],
        model_factory=_no_model,
        input_fn=_input(["/status", "/eval", "/exit"]),
        output_fn=output.append,
    ) == 0
    assert "Resumed session a-new" in output
    assert "Model: saved-model" in output
    assert "  ID: a-new-run" in output
    after = store.load(expected.session_id)
    assert after.run_records == before.run_records
    assert after.session.items == before.session.items
    assert len(store.list_sessions()) == 3


def test_continue_new_turn_reuses_existing_resume_path(environment):
    workspace, store = environment
    _seed(store, workspace, "continue-turn", datetime.now(timezone.utc), with_run=True)
    model = ScriptedModel([Message(role="assistant", content="new response")])
    assert cli.main(
        ["--workspace", str(workspace), "--continue"],
        model_factory=lambda name: model,
        input_fn=_input(["new request", "/exit"]),
        output_fn=lambda value: None,
    ) == 0
    assert len(store.load("continue-turn").run_records) == 2
    assert any(
        isinstance(item, Message) and item.content == "historical"
        for item in model.contexts[0]
    )


@pytest.mark.parametrize("other_session", [False, True])
def test_continue_without_matching_session_fails_without_creating_one(environment, other_session):
    workspace, store = environment
    if other_session:
        other = workspace.parent / "other"
        other.mkdir()
        _seed(store, other, "other", datetime.now(timezone.utc))
    before = store.list_sessions()
    output = []
    assert cli.main(
        ["--workspace", str(workspace), "--continue"],
        model_factory=_no_model,
        input_fn=lambda prompt: pytest.fail("no session should open"),
        output_fn=output.append,
    ) == 2
    assert output == [f"Error: No saved session for workspace: {workspace.resolve()}"]
    assert store.list_sessions() == before


def test_continue_resolves_workspace_alias(environment):
    workspace, store = environment
    _seed(store, workspace, "canonical", datetime.now(timezone.utc))
    alias = workspace.parent / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    output = []
    assert cli.main(
        ["--workspace", str(alias), "--continue"],
        model_factory=_no_model,
        input_fn=_input(["/exit"]),
        output_fn=output.append,
    ) == 0
    assert "Resumed session canonical" in output


def test_locale_validation(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--locale", "fr"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--locale", "zh-CN", "resume", "id"], ["resume", "id", "--locale", "zh-CN", "--plain"]])
def test_resume_accepts_presentation_options_on_either_side(argv):
    arguments = cli.build_parser().parse_args(argv)
    assert arguments.locale == "zh-CN"


@pytest.mark.parametrize("verbose, detail_level", [(False, "compact"), (True, "verbose")])
def test_rich_selection_for_interactive_terminal(monkeypatch, verbose, detail_level):
    rich = pytest.importorskip("pureharness.rich_terminal")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    renderer = cli._select_interactive_renderer(print, plain=False, locale="zh-CN", verbose=verbose)
    assert isinstance(renderer, rich.RichTerminalRenderer)
    assert renderer.locale == "zh-CN"
    assert renderer.interactive
    assert renderer.detail_level == detail_level


@pytest.mark.parametrize("plain, terminal, injected", [
    (True, True, False), (False, False, False), (False, True, True),
])
@pytest.mark.parametrize("verbose", [False, True])
def test_plain_fallback_for_flags_redirects_and_injected_output(monkeypatch, plain, terminal, injected, verbose):
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: terminal)
    output = (lambda value: None) if injected else print
    assert isinstance(
        cli._select_interactive_renderer(output, plain=plain, locale="zh-CN", verbose=verbose),
        cli.PlainTerminalRenderer,
    )


def test_plain_fallback_when_rich_unavailable(monkeypatch):
    original_import = builtins.__import__

    def without_rich(name, *args, **kwargs):
        if name == "pureharness.rich_terminal":
            raise ModuleNotFoundError("rich unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rich)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert isinstance(
        cli._select_interactive_renderer(print, plain=False, locale="en"),
        cli.PlainTerminalRenderer,
    )


def test_real_cli_rich_header_events_response_and_chinese_evaluation(environment, monkeypatch):
    rich = pytest.importorskip("pureharness.rich_terminal")
    from rich.console import Console

    workspace, _ = environment
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=160)
    monkeypatch.setattr(rich, "Console", lambda: console)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    agents = []
    original_create = cli._create_agent

    def capture(*args, **kwargs):
        agent = original_create(*args, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(cli, "_create_agent", capture)
    model = ScriptedModel([
        [ToolCall(name="list_files", arguments={}, call_id="list")],
        Message(role="assistant", content="[literal] response"),
    ])
    assert cli.main(
        ["--workspace", str(workspace), "--locale", "zh-CN", "--verbose"],
        model_factory=lambda name: model,
        input_fn=_input(["inspect workspace", "/eval", "/exit"]),
    ) == 0
    text = output.getvalue()
    assert "PureHarness" in text
    assert f"工作区: {workspace}" in text
    assert "模型: deepseek-v4-flash" in text
    assert "会话 ID:" in text
    assert "正在构建上下文" in text
    assert "正在请求模型" in text
    assert "正在调用工具：list_files" in text
    assert "Run 执行完成" in text
    assert "助手" in text
    assert "[literal] response" in text
    assert "Run 执行评估" in text
    assert "协议完成度: 1.000" in text
    assert "恢复建议（仅供参考）" in text
    assert agents[0].listener_errors == []


def test_plain_cli_stays_english_with_chinese_locale(environment, monkeypatch, capsys):
    workspace, _ = environment
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert cli.main(
        ["--workspace", str(workspace), "--plain", "--locale", "zh-CN"],
        model_factory=_no_model,
        input_fn=_input(["/help", "/exit"]),
    ) == 0
    text = capsys.readouterr().out
    assert "Workspace:" in text
    assert "Interactive commands" in text
    assert "\x1b" not in text


def test_jsonl_ignores_interactive_presentation_even_on_tty(environment, monkeypatch, capsys):
    workspace, _ = environment
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli, "_select_interactive_renderer", lambda *a, **k: pytest.fail("human UI in JSONL"))
    assert cli.main(
        ["--verbose", "--locale", "zh-CN", "run", "hello", "--workspace", str(workspace), "--output", "jsonl"],
        model_factory=lambda name: ScriptedModel([Message(role="assistant", content="response")]),
        input_fn=lambda prompt: pytest.fail("no input in JSONL"),
    ) == 0
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert all(event["schema_version"] == 1 for event in events)
    assert events[-1]["event"] == "agent_completed"
    assert "Run evaluation" not in captured.out
    assert "response" not in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("argv", [
    ["--plain", "--verbose"],
    ["--verbose", "resume", "id", "--plain"],
    ["--plain", "resume", "id", "--verbose"],
    ["resume", "id", "--verbose", "--plain"],
])
def test_conflicting_presentation_flags_fail_before_session_or_model(argv, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_create_durable_store", lambda: pytest.fail("must reject before persistence"))
    with pytest.raises(SystemExit) as error:
        cli.main(argv, model_factory=_no_model)
    assert error.value.code == 2
    assert "--plain and --verbose cannot be used together" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["resume_before", "resume_after", "continue"])
def test_verbose_resume_and_continue_are_passive(environment, monkeypatch, mode):
    workspace, store = environment
    state = _seed(store, workspace, "saved", datetime.now(timezone.utc), with_run=True)
    selected = []
    original = cli._select_interactive_renderer

    def capture(*args, **kwargs):
        selected.append(kwargs["verbose"])
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "_select_interactive_renderer", capture)
    monkeypatch.setattr(cli, "_create_agent", lambda *a, **k: pytest.fail("passive resume"))
    argv = {
        "resume_before": ["--verbose", "resume", "saved"],
        "resume_after": ["resume", "saved", "--verbose"],
        "continue": ["--workspace", str(workspace), "--continue", "--verbose"],
    }[mode]
    output = []
    assert cli.main(argv, model_factory=_no_model, input_fn=_input(["/eval", "/exit"]), output_fn=output.append) == 0
    assert selected == [True]
    assert "Execution evaluation" in output
    assert store.load("saved") == state


@pytest.mark.parametrize("locale, inspected, completion, evaluation", [
    ("en", "Inspected workspace", "Run completed · 2 steps · 1 tool calls", "Protocol completion: 1.000"),
    ("zh-CN", "已查看 工作区", "Run 执行完成 · 2 步骤 · 1 工具调用", "协议完成度: 1.000"),
])
def test_real_cli_default_compact_and_detailed_eval(environment, monkeypatch, locale, inspected, completion, evaluation):
    rich = pytest.importorskip("pureharness.rich_terminal")
    from rich.console import Console

    workspace, _ = environment
    output = StringIO()
    monkeypatch.setattr(rich, "Console", lambda: Console(file=output, force_terminal=False, color_system=None, width=160))
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    model = ScriptedModel([
        [ToolCall(name="list_files", arguments={})],
        Message(role="assistant", content="[literal] response"),
    ])
    assert cli.main(
        ["--workspace", str(workspace), "--locale", locale],
        model_factory=lambda name: model,
        input_fn=_input(["inspect", "/eval", "/exit"]),
    ) == 0
    text = output.getvalue()
    automatic, detailed = text.split("Execution evaluation" if locale == "en" else "执行评估")
    assert inspected in automatic
    assert completion in automatic
    assert evaluation in automatic
    assert "[literal] response" in automatic
    for label in ("Building context", "正在构建上下文", "Requesting model", "正在请求模型", "Tool policy", "工具策略", "Step efficiency", "步骤效率", "Tool reliability", "工具可靠性", "Recovery (advisory only)", "恢复建议（仅供参考）"):
        assert label not in automatic
    assert "Step efficiency" in detailed if locale == "en" else "步骤效率" in detailed
    assert "Tool reliability" in detailed if locale == "en" else "工具可靠性" in detailed
    assert "End reason" in detailed if locale == "en" else "结束原因" in detailed
    assert "Recovery" in detailed if locale == "en" else "恢复建议" in detailed
    assert "not verified task correctness" in detailed if locale == "en" else "不代表已验证任务正确性" in detailed


def test_real_compact_cli_failed_verification_edit_then_pass_is_shown_once(environment, monkeypatch):
    rich = pytest.importorskip("pureharness.rich_terminal")
    from rich.console import Console

    workspace, store = environment
    (workspace / "calc.py").write_text("value = 1\n", encoding="utf-8")
    output = StringIO()
    monkeypatch.setattr(rich, "Console", lambda: Console(file=output, force_terminal=False, color_system=None, width=160))
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    agents = []
    original = cli._create_agent

    def capture(*args, **kwargs):
        agent = original(*args, **kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(cli, "_create_agent", capture)
    model = ScriptedModel([
        [ToolCall(name="read_file", arguments={"path": "calc.py"})],
        [ToolCall(name="run_command", arguments={"argv": [sys.executable, "-c", "raise SystemExit(1)"], "purpose": "verification"})],
        [ToolCall(name="apply_patch", arguments={"path": "calc.py", "old_text": "value = 1", "new_text": "value = 2"})],
        [ToolCall(name="run_command", arguments={"argv": [sys.executable, "-c", "raise SystemExit(0)"], "purpose": "verification"})],
        Message(role="assistant", content="done"),
    ])
    assert cli.main(
        ["--workspace", str(workspace), "--locale", "zh-CN"],
        model_factory=lambda name: model, input_fn=_input(["edit and verify", "/exit"]),
    ) == 0
    text = output.getvalue()
    assert text.count("验证失败 · 退出码 1") == 1
    assert text.count("✓ 验证通过") == 1
    assert text.index("已读取 calc.py") < text.index("验证失败") < text.index("已更新 calc.py") < text.index("验证通过")
    assert "Run 执行完成 · 5 步骤 · 4 工具调用" in text
    assert "正在构建上下文" not in text
    assert agents[0].listener_errors == []
    evidence = [event for event in agents[0].events if event.type == "coding_evidence_snapshot"]
    assert len(evidence) > 2
    assert evidence[-1].data["verification_attempts"] == 2
    record = store.load(store.list_sessions()[0].session_id).run_records[0]
    assert record.end_reason == "completed"
    assert record.tool_result_error_count == 0  # nonzero exit is not a tool execution error
    assert record.model_call_count == len(record.model_invocations) == 5
    assert (workspace / "calc.py").read_text(encoding="utf-8") == "value = 2\n"
