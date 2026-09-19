import tomllib

import pytest

import miniharness.cli as cli_module
from miniharness.agent import Agent
from miniharness.cli import main
from miniharness.experiment import load_experiment_results
from miniharness.messages import Message, ToolCall
from miniharness.model import ModelError
from miniharness.tool_executor import ToolExecutor
from miniharness.tool_policy import PolicyDecision
from miniharness.tools import Tool, ToolRegistry


@pytest.fixture(autouse=True)
def isolated_miniharness_home(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "MINIHARNESS_HOME",
        str(tmp_path / "miniharness-home"),
    )


class MultiTurnModel:
    def __init__(self):
        self.contexts = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        user_count = sum(
            isinstance(item, Message) and item.role == "user"
            for item in messages
        )
        return Message(
            role="assistant",
            content=f"response-{user_count}",
        )


class FailingModel:
    def generate(self, messages, tools):
        raise ModelError("offline failure")


class InterruptingModel:
    def generate(self, messages, tools):
        raise KeyboardInterrupt


class FixSimpleTaskModel:
    def __init__(self):
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1
        if self.call_count == 1:
            return [
                ToolCall(
                    name="apply_patch",
                    arguments={
                        "path": "calculator.py",
                        "old_text": "return left - right",
                        "new_text": "return left + right",
                    },
                    call_id="fix-1",
                )
            ]
        return Message(role="assistant", content="fixed")


def _input(values):
    iterator = iter(values)
    return lambda prompt: next(iterator)


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "miniharness" in output
    assert "run" in output
    assert "inspect" in output
    assert "benchmark" in output
    assert "sessions" in output
    assert "resume" in output


def test_interactive_session_is_multi_turn_and_supports_commands(tmp_path):
    model = MultiTurnModel()
    output = []

    exit_code = main(
        ["--workspace", str(tmp_path), "--model", "fake-model"],
        model_factory=lambda name: model,
        input_fn=_input(
            [
                "/help",
                "first request",
                "/status",
                "second request",
                "/exit",
            ]
        ),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert len(model.contexts) == 2
    assert sum(
        isinstance(item, Message) and item.role == "user"
        for item in model.contexts[1]
    ) == 2
    rendered = "\n".join(output)
    assert "Commands: /help, /status, /exit" in rendered
    assert "response-1" in rendered
    assert "response-2" in rendered
    assert "Runs in session: 1" in rendered
    assert "Last end reason: completed" in rendered
    assert "Goodbye." in rendered


def test_interactive_status_and_exit_do_not_require_api_key(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        cli_module,
        "load_dotenv",
        lambda: pytest.fail("slash commands must not load API credentials"),
    )
    output = []

    exit_code = main(
        ["--workspace", str(tmp_path)],
        model_factory=lambda name: pytest.fail(
            "slash commands must not create a model"
        ),
        input_fn=_input(["/status", "/exit"]),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert "Runs in session: 0" in output
    assert "Last run ID: (none)" in output
    assert output[-1] == "Goodbye."


def test_interactive_help_and_exit_do_not_create_model(tmp_path):
    output = []

    exit_code = main(
        ["--workspace", str(tmp_path)],
        model_factory=lambda name: pytest.fail(
            "slash commands must not create a model"
        ),
        input_fn=_input(["/help", "/exit"]),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert any("Commands: /help, /status, /exit" in line for line in output)
    assert output[-1] == "Goodbye."


def test_interactive_lazily_creates_and_reuses_one_agent(tmp_path):
    model = MultiTurnModel()
    factory_calls = []
    calls_seen_before_input = []
    values = iter(
        ["/status", "first request", "second request", "/exit"]
    )

    def model_factory(name):
        factory_calls.append(name)
        return model

    def tracked_input(prompt):
        calls_seen_before_input.append(len(factory_calls))
        return next(values)

    exit_code = main(
        ["--workspace", str(tmp_path), "--model", "fake-model"],
        model_factory=model_factory,
        input_fn=tracked_input,
        output_fn=lambda value: None,
    )

    assert exit_code == 0
    assert calls_seen_before_input == [0, 0, 1, 1]
    assert factory_calls == ["fake-model"]
    assert len(model.contexts) == 2
    assert sum(
        isinstance(item, Message) and item.role == "user"
        for item in model.contexts[1]
    ) == 2


def test_interactive_ctrl_d_exits_cleanly(tmp_path):
    def eof(prompt):
        raise EOFError

    output = []
    exit_code = main(
        ["--workspace", str(tmp_path)],
        model_factory=lambda name: MultiTurnModel(),
        input_fn=eof,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert output[-1] == "Goodbye."


def test_interactive_ctrl_c_at_prompt_does_not_traceback(tmp_path):
    calls = 0

    def interrupted_then_exit(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return "/exit"

    output = []
    exit_code = main(
        ["--workspace", str(tmp_path)],
        model_factory=lambda name: MultiTurnModel(),
        input_fn=interrupted_then_exit,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert "Interrupted. Use /exit to leave." in output
    assert output[-1] == "Goodbye."


def test_interactive_ctrl_c_during_run_returns_to_prompt(tmp_path):
    output = []

    exit_code = main(
        ["--workspace", str(tmp_path)],
        model_factory=lambda name: InterruptingModel(),
        input_fn=_input(["start work", "/exit"]),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert "Run interrupted." in output
    assert output[-1] == "Goodbye."


def test_interactive_ctrl_c_during_approval_denies_without_execution(
    tmp_path,
    monkeypatch,
):
    execution_count = 0

    class RequireApprovalPolicy:
        def evaluate(self, tool, arguments):
            return PolicyDecision.REQUIRE_APPROVAL

    class ApprovalModel:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return [
                    ToolCall(
                        name="effect",
                        arguments={},
                        call_id="approval-ctrl-c",
                    )
                ]
            return Message(role="assistant", content="denied safely")

    def create_agent(
        workspace,
        model,
        *,
        session_id,
        output_fn,
        session=None,
        approval_handler=None,
    ):
        nonlocal execution_count
        assert approval_handler is not None

        def effect():
            nonlocal execution_count
            execution_count += 1

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="effect",
                description="Approval-gated effect.",
                parameters={"type": "object", "properties": {}},
                function=effect,
                side_effects=True,
            )
        )
        return Agent(
            model=model,
            tools=registry,
            tool_executor=ToolExecutor(
                registry,
                RequireApprovalPolicy(),
                approval_handler=approval_handler,
            ),
            listeners=[cli_module.PlainTerminalRenderer(output_fn)],
            session_id=session_id,
            session=session,
            max_steps=2,
        )

    monkeypatch.setattr(cli_module, "_create_agent", create_agent)
    input_calls = 0

    def interrupt_approval(prompt):
        nonlocal input_calls
        input_calls += 1
        if input_calls == 1:
            return "perform effect"
        if input_calls == 2:
            raise KeyboardInterrupt
        return "/exit"

    output = []
    exit_code = main(
        ["--workspace", str(tmp_path), "--model", "fake-model"],
        model_factory=lambda name: ApprovalModel(),
        input_fn=interrupt_approval,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert execution_count == 0
    assert "Approval cancelled; action denied." in output
    assert "[approval] denied: effect" in output
    assert "denied safely" in output
    assert not any("Traceback" in line for line in output)


def test_one_shot_writes_record_and_inspect_reads_it(tmp_path):
    record_path = tmp_path / "evidence" / "run.json"
    output = []

    run_exit = main(
        [
            "run",
            "inspect this",
            "--workspace",
            str(tmp_path),
            "--model",
            "fake-model",
            "--record",
            str(record_path),
        ],
        model_factory=lambda name: MultiTurnModel(),
        output_fn=output.append,
    )

    assert run_exit == 0
    assert record_path.is_file()
    inspect_output = []
    inspect_exit = main(
        ["inspect", str(record_path)],
        model_factory=lambda name: pytest.fail(
            "inspect must not construct a model"
        ),
        output_fn=inspect_output.append,
    )

    assert inspect_exit == 0
    rendered = "\n".join(inspect_output)
    assert "End reason: completed" in rendered
    assert "Model calls: 1" in rendered
    assert "Estimated history tokens:" in rendered


def test_one_shot_agent_has_no_interactive_approval_handler(tmp_path):
    agent = cli_module._create_agent(
        tmp_path,
        MultiTurnModel(),
        session_id="one-shot-session",
        output_fn=lambda value: None,
    )

    assert agent.tool_executor.approval_handler is None


def test_one_shot_failure_has_nonzero_exit_without_traceback(tmp_path):
    output = []

    exit_code = main(
        ["run", "fail", "--workspace", str(tmp_path)],
        model_factory=lambda name: FailingModel(),
        output_fn=output.append,
    )

    assert exit_code == 1
    assert any("Run failed: ModelError" in line for line in output)


def test_one_shot_missing_api_key_has_friendly_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "load_dotenv", lambda: False)
    output = []

    exit_code = main(
        ["run", "hello", "--workspace", str(tmp_path)],
        output_fn=output.append,
    )

    assert exit_code == 2
    assert output == ["Error: DEEPSEEK_API_KEY is not set."]
    assert "KeyError" not in output[0]
    assert "Traceback" not in output[0]


def test_benchmark_command_reuses_harness_without_real_api(tmp_path):
    output_path = tmp_path / "results.jsonl"
    output = []
    created = []

    def model_factory(name):
        model = FixSimpleTaskModel()
        created.append(model)
        return model

    exit_code = main(
        [
            "benchmark",
            "--tasks-root",
            "benchmarks/tasks",
            "--task",
            "simple_fix",
            "--config",
            "raw_baseline",
            "--repetitions",
            "1",
            "--model",
            "fake-model",
            "--output",
            str(output_path),
        ],
        model_factory=model_factory,
        output_fn=output.append,
    )

    assert exit_code == 0
    results = load_experiment_results(output_path)
    assert len(results) == 1
    assert results[0].model_id == "fake-model"
    assert results[0].benchmark_result.task_success is True
    assert len(created) == 1
    assert any("raw_baseline: 1/1 successful" in line for line in output)


def test_invalid_record_path_is_clean_cli_error(tmp_path):
    output = []

    exit_code = main(
        ["inspect", str(tmp_path / "missing.json")],
        output_fn=output.append,
    )

    assert exit_code == 2
    assert output == [
        f"Error: Could not read RunRecord: {tmp_path / 'missing.json'}"
    ]


def test_console_script_is_registered():
    with open("pyproject.toml", "rb") as handle:
        project = tomllib.load(handle)["project"]

    assert project["version"] == "0.1.0"
    assert project["readme"] == "README.md"
    assert project["optional-dependencies"]["dev"] == ["pytest>=8"]
    assert project["scripts"]["miniharness"] == (
        "miniharness.cli:main"
    )
