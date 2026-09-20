import asyncio
import shlex

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest


pytest.importorskip("harbor")

from harbor.models.agent.context import AgentContext

from pureharness.harbor_agent import PureHarnessHarborAgent


COMMIT = "a" * 40
MODEL = "deepseek/deepseek-chat"


class FakeResult:
    return_code = 0
    stdout = ""
    stderr = ""


class FakeEnvironment:
    def __init__(self, workdir: str | None = "/workspace") -> None:
        self.task_env_config = SimpleNamespace(workdir=workdir)
        self.default_user = None
        self.calls: list[dict[str, object]] = []

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResult()


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


def test_install_command_pins_exact_pureharness_revision(
    tmp_path,
    monkeypatch,
):
    agent = make_agent(tmp_path)
    environment = FakeEnvironment()

    async def dependencies(_environment, _dependencies):
        return None

    monkeypatch.setattr(agent, "ensure_system_dependencies", dependencies)
    asyncio.run(agent.install(environment))

    assert len(environment.calls) == 1
    command = str(environment.calls[0]["command"])
    assert "python3 -m venv /installed-agent/pureharness/venv" in command
    assert (
        "git+https://github.com/pmk915/pureharness.git@" + COMMIT
        in command
    )
    assert "@main" not in command


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
    assert "--record /logs/agent/pureharness-run-record.json" in command
    assert "--output jsonl" in command
    assert shlex.quote(instruction) in command
    assert call["cwd"] == "/workspace"
    assert call["env"] == {"DEEPSEEK_API_KEY": "secret-key"}
    assert agent.extra_env == {}
