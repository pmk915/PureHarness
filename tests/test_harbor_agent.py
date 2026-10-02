import asyncio
import json
import os
import shlex
import subprocess

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest


pytest.importorskip("harbor")

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext

from pureharness.harbor_agent import PureHarnessHarborAgent


COMMIT = "a" * 40
MODEL = "deepseek/deepseek-chat"


class FakeResult:
    def __init__(self, return_code=0, stdout="", stderr="") -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class FakeEnvironment:
    def __init__(
        self,
        workdir: str | None = "/workspace",
        results: list[FakeResult] | None = None,
    ) -> None:
        self.task_env_config = SimpleNamespace(workdir=workdir)
        self.default_user = None
        self.calls: list[dict[str, object]] = []
        self.results = list(results or [])

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        if self.results:
            return self.results.pop(0)
        return FakeResult()


class LocalShellEnvironment:
    def __init__(self, workdir: Path) -> None:
        self.task_env_config = SimpleNamespace(workdir=str(workdir))
        self.default_user = None
        self.calls: list[dict[str, object]] = []
        self.results: list[FakeResult] = []

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        command_env = os.environ.copy()
        command_env.update(kwargs.get("env") or {})
        completed = subprocess.run(
            ["bash", "-c", str(kwargs["command"])],
            cwd=kwargs.get("cwd"),
            env=command_env,
            capture_output=True,
            text=True,
            check=False,
        )
        result = FakeResult(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        self.results.append(result)
        return result


@pytest.fixture(autouse=True)
def clear_harbor_env(monkeypatch):
    monkeypatch.delenv("PUREHARNESS_HARBOR_REF", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def make_agent(tmp_path, **overrides):
    values = {
        "logs_dir": tmp_path / "logs",
        "model_name": MODEL,
        "extra_env": {
            "PUREHARNESS_HARBOR_REF": COMMIT,
            "DEEPSEEK_API_KEY": "secret-key",
        },
    }
    values.update(overrides)
    return PureHarnessHarborAgent(**values)


def test_adapter_name(tmp_path):
    agent = make_agent(tmp_path)

    assert agent.name() == "pureharness"
    assert agent.version() == COMMIT


def test_missing_pureharness_ref_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="PUREHARNESS_HARBOR_REF must be set"):
        make_agent(
            tmp_path,
            extra_env={"DEEPSEEK_API_KEY": "secret-key"},
        )


def test_ref_must_be_a_full_commit_sha(tmp_path):
    with pytest.raises(ValueError, match="full 40-character Git commit SHA"):
        make_agent(
            tmp_path,
            extra_env={
                "PUREHARNESS_HARBOR_REF": "main",
                "DEEPSEEK_API_KEY": "secret-key",
            },
        )


@pytest.mark.parametrize("model_name", [None, "", "deepseek", "deepseek/"])
def test_missing_or_malformed_model_is_rejected(tmp_path, model_name):
    with pytest.raises(ValueError, match="Model name|model name"):
        make_agent(tmp_path, model_name=model_name)


def test_unsupported_provider_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="only 'deepseek' is supported"):
        make_agent(tmp_path, model_name="openai/gpt-5")


def test_missing_deepseek_credential_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY must be set"):
        make_agent(
            tmp_path,
            extra_env={"PUREHARNESS_HARBOR_REF": COMMIT},
        )


def test_install_succeeds_immediately_and_pins_exact_revision(
    tmp_path,
    monkeypatch,
):
    agent = make_agent(tmp_path)
    environment = FakeEnvironment()
    dependency_calls = []

    async def dependencies(dependency_environment, dependencies):
        dependency_calls.append((dependency_environment, dependencies))

    monkeypatch.setattr(agent, "ensure_system_dependencies", dependencies)
    asyncio.run(agent.install(environment))

    assert dependency_calls == []
    assert len(environment.calls) == 3
    capability_call = environment.calls[0]
    capability_command = str(capability_call["command"])
    assert capability_call["user"] == "root"
    assert "command -v git" in capability_command
    assert "command -v python3" in capability_command
    assert "ssl.create_default_context().get_ca_certs()" in capability_command
    assert "TemporaryDirectory" in capability_command
    assert '["python3", "-m", "venv"' in capability_command

    venv_command = str(environment.calls[1]["command"])
    assert "python3 -m venv /installed-agent/pureharness/venv" in venv_command

    install_command = str(environment.calls[2]["command"])
    assert "pip install" in install_command
    assert (
        "git+https://github.com/pmk915/pureharness.git@" + COMMIT
        in install_command
    )
    assert "@main" not in install_command
    assert "secret-key" not in str(environment.calls)


def test_install_retries_transient_failure_then_succeeds(
    tmp_path,
    monkeypatch,
):
    agent = make_agent(tmp_path)
    environment = FakeEnvironment(
        results=[
            FakeResult(),
            FakeResult(),
            FakeResult(return_code=1, stderr="transient TLS failure"),
            FakeResult(),
        ]
    )
    retry_delays = []

    async def dependencies(_environment, _dependencies):
        pytest.fail("dependency fallback should not run")

    async def sleep(delay):
        retry_delays.append(delay)

    monkeypatch.setattr(agent, "ensure_system_dependencies", dependencies)
    monkeypatch.setattr("pureharness.harbor_agent.asyncio.sleep", sleep)
    asyncio.run(agent.install(environment))

    commands = [str(call["command"]) for call in environment.calls]
    venv_commands = [
        command
        for command in commands
        if "python3 -m venv /installed-agent/pureharness/venv" in command
    ]
    install_commands = [
        command for command in commands if "pip install" in command
    ]

    assert len(venv_commands) == 1
    assert len(install_commands) == 2
    assert retry_delays == [1]
    assert all(
        "git+https://github.com/pmk915/pureharness.git@" + COMMIT
        in command
        for command in install_commands
    )
    assert "secret-key" not in str(environment.calls)


def test_install_propagates_after_all_attempts_fail(
    tmp_path,
    monkeypatch,
):
    agent = make_agent(tmp_path)
    environment = FakeEnvironment(
        results=[
            FakeResult(),
            FakeResult(),
            FakeResult(return_code=1, stderr="transient TLS failure"),
            FakeResult(return_code=1, stderr="transient TLS failure"),
            FakeResult(return_code=1, stderr="transient TLS failure"),
        ]
    )
    retry_delays = []

    async def dependencies(_environment, _dependencies):
        pytest.fail("dependency fallback should not run")

    async def sleep(delay):
        retry_delays.append(delay)

    monkeypatch.setattr(agent, "ensure_system_dependencies", dependencies)
    monkeypatch.setattr("pureharness.harbor_agent.asyncio.sleep", sleep)

    with pytest.raises(NonZeroAgentExitCodeError):
        asyncio.run(agent.install(environment))

    commands = [str(call["command"]) for call in environment.calls]
    venv_commands = [
        command
        for command in commands
        if "python3 -m venv /installed-agent/pureharness/venv" in command
    ]
    install_commands = [
        command for command in commands if "pip install" in command
    ]

    assert len(venv_commands) == 1
    assert len(install_commands) == 3
    assert retry_delays == [1, 2]
    assert all(
        "git+https://github.com/pmk915/pureharness.git@" + COMMIT
        in command
        for command in install_commands
    )
    assert "secret-key" not in str(environment.calls)


def test_install_falls_back_when_capabilities_are_missing(
    tmp_path,
    monkeypatch,
):
    agent = make_agent(tmp_path)
    environment = FakeEnvironment(results=[FakeResult(return_code=1)])
    dependency_calls = []

    async def dependencies(dependency_environment, dependencies):
        dependency_calls.append((dependency_environment, dependencies))

    monkeypatch.setattr(agent, "ensure_system_dependencies", dependencies)
    asyncio.run(agent.install(environment))

    assert dependency_calls == [
        (
            environment,
            ("git", "python3", "python_venv", "ca_certificates"),
        )
    ]


def test_run_invokes_public_cli_with_safe_arguments_and_record(tmp_path):
    instruction = "fix 'quoted' input; touch /tmp/not-run"
    agent = make_agent(tmp_path)
    agent.environment_logs_dir = PurePosixPath("/logs/agent")
    environment = FakeEnvironment()

    asyncio.run(agent.run(instruction, environment, AgentContext()))

    assert len(environment.calls) == 1
    call = environment.calls[0]
    command = str(call["command"])
    assert "pureharness/venv/bin/pureharness run" in command
    assert "--workspace /workspace" in command
    assert "--model deepseek-chat" in command
    assert "--max-steps 300" in command
    assert "--history-token-budget 8000" in command
    assert "--record /logs/agent/pureharness-run-record.json" in command
    assert "--output jsonl" in command
    assert "| tee /logs/agent/pureharness-events.jsonl" in command
    assert command.startswith("set -o pipefail; ")
    assert shlex.quote(instruction) in command
    assert call["cwd"] == "/workspace"
    assert call["env"] == {"DEEPSEEK_API_KEY": "secret-key"}
    assert agent.extra_env == {}


@pytest.mark.parametrize("instruction", ["summarize a directory", "repair an arithmetic function"])
def test_evaluation_command_is_task_agnostic(tmp_path, instruction):
    agent = make_agent(tmp_path)
    agent.environment_logs_dir = PurePosixPath("/logs/agent")
    environment = FakeEnvironment()
    asyncio.run(agent.run(instruction, environment, AgentContext()))

    command = str(environment.calls[0]["command"])
    tokens = shlex.split(command)
    executable = "/installed-agent/pureharness/venv/bin/pureharness"
    argv = tokens[tokens.index(executable):tokens.index("|")]
    assert argv == [
        executable, "run", instruction,
        "--workspace", "/workspace", "--model", "deepseek-chat",
        "--max-steps", "300", "--history-token-budget", "8000",
        "--record", "/logs/agent/pureharness-run-record.json", "--output", "jsonl",
    ]
    for hint in ("regex-log", "terminal-bench", "findall", "capturing group"):
        assert hint not in command


@pytest.mark.parametrize("exit_code", [0, 7], ids=["success", "failure"])
def test_run_persists_clean_jsonl_and_preserves_exit_status(
    tmp_path,
    monkeypatch,
    exit_code,
):
    event_values = [
        {
            "schema_version": 1,
            "event": "model_retrying",
            "run_id": "run-1",
            "payload": {"attempt": 2},
        },
        {
            "schema_version": 1,
            "event": "agent_failed",
            "run_id": "run-1",
            "payload": {"end_reason": "model_error"},
        },
    ]
    event_lines = [
        json.dumps(value, separators=(",", ":"))
        for value in event_values
    ]
    executable = tmp_path / "venv/bin/pureharness"
    executable.parent.mkdir(parents=True)
    script_lines = [
        "#!/bin/sh",
        *[
            f"printf '%s\\n' {shlex.quote(line)}"
            for line in event_lines
        ],
        f"exit {exit_code}",
    ]
    executable.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(
        "pureharness.harbor_agent._VENV_DIR",
        PurePosixPath(str(tmp_path / "venv")),
    )

    agent = make_agent(tmp_path)
    log_directory = tmp_path / "logs/agent"
    agent.environment_logs_dir = PurePosixPath(str(log_directory))
    environment = LocalShellEnvironment(tmp_path)

    if exit_code == 0:
        asyncio.run(agent.run("test task", environment, AgentContext()))
    else:
        with pytest.raises(NonZeroAgentExitCodeError):
            asyncio.run(agent.run("test task", environment, AgentContext()))

    event_log = log_directory / "pureharness-events.jsonl"
    persisted_lines = event_log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in persisted_lines] == event_values
    assert persisted_lines == event_lines
    assert environment.results[-1].stdout.splitlines() == event_lines
    assert environment.results[-1].return_code == exit_code


def test_run_resolves_missing_workdir_from_container_pwd(tmp_path):
    instruction = "create answer.txt"
    agent = make_agent(tmp_path)
    agent.environment_logs_dir = PurePosixPath("/logs/agent")
    environment = FakeEnvironment(
        workdir=None,
        results=[
            FakeResult(stdout="/app\n"),
            FakeResult(),
        ],
    )

    asyncio.run(agent.run(instruction, environment, AgentContext()))

    assert len(environment.calls) == 2

    pwd_call = environment.calls[0]
    assert str(pwd_call["command"]).endswith("pwd")

    run_call = environment.calls[1]
    command = str(run_call["command"])

    assert "--workspace /app" in command
    assert run_call["cwd"] == "/app"


def test_run_resolves_relative_workdir_against_container_pwd(tmp_path):
    agent = make_agent(tmp_path)
    agent.environment_logs_dir = PurePosixPath("/logs/agent")
    environment = FakeEnvironment(
        workdir="repo",
        results=[
            FakeResult(stdout="/app\n"),
            FakeResult(),
        ],
    )

    asyncio.run(agent.run("test task", environment, AgentContext()))

    assert len(environment.calls) == 2

    run_call = environment.calls[1]
    command = str(run_call["command"])

    assert "--workspace /app/repo" in command
    assert run_call["cwd"] == "/app/repo"
