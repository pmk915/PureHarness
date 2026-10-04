"""The example drives the real CLI and local tools, without a provider."""

import json
import os
from io import StringIO

import pytest

import pureharness.cli as cli
from examples.cli_demo import DEMO_TASK, FIXTURE, ScriptedDemoModel, main
from pureharness.messages import Message, ToolResult
from pureharness.model import ModelError
from pureharness.run_record import RunRecord


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_default_model_factory", lambda name: pytest.fail("no provider"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: pytest.fail("no credential loading"))


def test_jsonl_demo_executes_real_tools_and_preserves_fixture(tmp_path, capsys):
    before = {p.name: p.read_bytes() for p in FIXTURE.glob("*.py")}
    root = tmp_path / "demo"
    assert main(["--jsonl", "--output-dir", str(root)]) == 0
    captured = capsys.readouterr()
    assert "Offline scripted" in captured.err
    assert (root / "events.jsonl").read_text() == captured.out
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert events[0]["event"] == "agent_started"
    assert events[-1]["event"] == "agent_completed"
    assert all(event["schema_version"] == 1 for event in events)
    assert [e["payload"]["tool_name"] for e in events if e["event"] == "tool_started"] == [
        "read_file", "read_file", "apply_patch", "run_command",
    ]
    coding = [e["payload"] for e in events if e["event"] == "coding_evidence_snapshot"][-1]
    assert coding["workspace_mutations"] == 1
    assert coding["verification_attempts"] == coding["verification_exit_zero"] == 1
    assert coding["verification_exit_nonzero"] == coding["verification_tool_errors"] == 0
    record = RunRecord.from_json((root / "run.json").read_text())
    assert record.end_reason == "completed"
    assert record.step_count == record.model_call_count == len(record.model_invocations) == 5
    assert record.tool_call_count == record.tool_execution_count == 4
    assert record.tool_result_error_count == 0
    assert "return left + right" in (root / "workspace/calculator.py").read_text()
    assert any("Addition checks passed (3 cases)." in result.content
               for step in record.trace.steps for result in (step.tool_result or []))
    assert before == {p.name: p.read_bytes() for p in FIXTURE.glob("*.py")}
    assert not (root / "state").exists()  # One-shot mode has no durable interactive session.


def test_plain_interactive_demo_records_and_isolates_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PUREHARNESS_HOME", str(tmp_path / "original-home"))
    monkeypatch.setenv("DEMO_UNUSED_SECRET", "not-for-demo-output")
    root = tmp_path / "plain"
    inputs = iter([DEMO_TASK, "/eval", "/exit"])
    assert main(["--plain", "--output-dir", str(root)], input_fn=lambda prompt: next(inputs)) == 0
    captured = capsys.readouterr()
    assert "Model: offline-scripted-demo" in captured.out
    assert "[tool] apply_patch (ok)" in captured.out
    assert "Run record:" in captured.out
    assert "Completion: 1.000" in captured.out
    assert "not-for-demo-output" not in captured.out + captured.err
    assert len(list((root / "records").glob("*.json"))) == 1
    assert len(list((root / "state/sessions").glob("*.jsonl"))) == 1
    assert not (tmp_path / "original-home").exists()
    assert os.environ["PUREHARNESS_HOME"] == str(tmp_path / "original-home")


@pytest.mark.parametrize("verbose", [False, True])
def test_demo_uses_existing_rich_modes(tmp_path, monkeypatch, verbose):
    rich = pytest.importorskip("pureharness.rich_terminal")
    from rich.console import Console

    output = StringIO()
    monkeypatch.setattr(rich, "Console", lambda: Console(file=output, width=110, color_system=None))
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    inputs = iter([DEMO_TASK, "/exit"])
    args = ["--output-dir", str(tmp_path / "rich")]
    if verbose:
        args.append("--verbose")
    assert main(args, input_fn=lambda prompt: next(inputs)) == 0
    text = output.getvalue()
    assert "offline-scripted-demo" in text
    assert "calculator.py" in text
    assert "Run completed" in text
    assert ("Building context" in text) is verbose
    assert ("Tool policy" in text) is verbose


def test_demo_refuses_existing_output_and_does_not_overwrite(tmp_path, capsys):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    assert main(["--jsonl", "--output-dir", str(tmp_path)]) == 2
    assert marker.read_text() == "keep"
    assert not (tmp_path / "workspace").exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("flags", [
    ["--plain", "--verbose"], ["--jsonl", "--plain"], ["--jsonl", "--verbose"],
])
def test_demo_rejects_conflicting_flags_before_artifacts(tmp_path, flags):
    root = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        main([*flags, "--output-dir", str(root)])
    assert error.value.code == 2
    assert not root.exists()


def test_script_requires_actual_tool_results_and_exact_task():
    model = ScriptedDemoModel()
    with pytest.raises(ModelError, match="only accepts"):
        model.generate([Message("user", "unrelated task")], [])
    from pureharness.coding_tools import create_coding_tools
    call = model.generate([Message("user", DEMO_TASK)], create_coding_tools(FIXTURE))[0]
    with pytest.raises(ModelError, match="successful tool result"):
        model.generate([ToolResult(call.name, "real error", call.call_id, is_error=True)], [])


def test_demo_help_has_no_side_effects(tmp_path, capsys):
    root = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        main(["--help", "--output-dir", str(root)])
    assert error.value.code == 0
    assert "Only the responses are scripted" in " ".join(capsys.readouterr().out.split())
    assert not root.exists()
