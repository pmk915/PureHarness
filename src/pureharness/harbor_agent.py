"""Optional Harbor Installed Agent adapter for PureHarness.

Harbor owns environment lifecycle and evaluation. This adapter only installs a
pinned PureHarness revision and invokes its public one-shot CLI.
"""

import asyncio
import os
import re
import shlex

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, override

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
)
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


_AGENT_NAME = "pureharness"
_API_KEY_ENV = "DEEPSEEK_API_KEY"
_REF_ENV = "PUREHARNESS_HARBOR_REF"
_REPOSITORY_URL = "https://github.com/pmk915/pureharness.git"
_INSTALL_DIR = PurePosixPath("/installed-agent/pureharness")
_VENV_DIR = _INSTALL_DIR / "venv"
_RUN_RECORD_NAME = "pureharness-run-record.json"
_EVALUATION_MAX_STEPS = 300
_INSTALL_ATTEMPTS = 3
_INSTALL_RETRY_DELAY_SECONDS = 1
_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}\Z")
_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_CAPABILITY_PROBE = """\
import ssl
import subprocess
import tempfile

if not ssl.create_default_context().get_ca_certs():
    raise SystemExit(1)

with tempfile.TemporaryDirectory(prefix="pureharness-venv-check-") as temp_dir:
    result = subprocess.run(
        ["python3", "-m", "venv", f"{temp_dir}/venv"],
        check=False,
    )

raise SystemExit(result.returncode)
"""


def _require_value(
    sources: tuple[Mapping[str, str], ...],
    name: str,
) -> str:
    for source in sources:
        value = source.get(name)
        if value:
            return value
    raise ValueError(f"{name} must be set")


def _validate_ref(value: str) -> str:
    if not _COMMIT_PATTERN.fullmatch(value):
        raise ValueError(
            f"{_REF_ENV} must be a full 40-character Git commit SHA"
        )
    return value.lower()


def _deepseek_model_name(value: str | None) -> str:
    if not value:
        raise ValueError("Model name is required; use deepseek/<model>")
    if value.count("/") != 1:
        raise ValueError("Model name must use the form deepseek/<model>")

    provider, model = value.split("/", maxsplit=1)
    if provider != "deepseek":
        raise ValueError(
            f"Unsupported model provider {provider!r}; only 'deepseek' is supported"
        )
    if not _MODEL_PATTERN.fullmatch(model):
        raise ValueError("DeepSeek model name is malformed")
    return model


class PureHarnessHarborAgent(BaseInstalledAgent):
    """Run a commit-pinned PureHarness CLI inside a Harbor environment."""

    options_model = InstalledAgentOptions

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            extra_env=extra_env,
            **kwargs,
        )
        sources = (self._extra_env, os.environ)
        self._pureharness_ref = _validate_ref(
            _require_value(sources, _REF_ENV)
        )
        self._deepseek_api_key = _require_value(sources, _API_KEY_ENV)
        self._deepseek_model = _deepseek_model_name(model_name)

        # Harbor otherwise scopes every --agent-env value over setup and run.
        # The ref is consumed host-side and only the API credential is forwarded,
        # explicitly and only for the PureHarness CLI process.
        self._extra_env.clear()

    @staticmethod
    @override
    def name() -> str:
        return _AGENT_NAME

    @override
    def version(self) -> str:
        return self._pureharness_ref

    @classmethod
    @override
    def preflight(
        cls,
        kwargs: dict[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        super().preflight(kwargs, env)
        sources = (env or {}, os.environ)
        _validate_ref(_require_value(sources, _REF_ENV))
        _require_value(sources, _API_KEY_ENV)

    async def _has_required_install_capabilities(
        self,
        environment: BaseEnvironment,
    ) -> bool:
        result = await environment.exec(
            command=(
                "command -v git >/dev/null 2>&1 && "
                "command -v python3 >/dev/null 2>&1 && "
                f"python3 -c {shlex.quote(_CAPABILITY_PROBE)}"
            ),
            user="root",
        )
        return result.return_code == 0

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if not await self._has_required_install_capabilities(environment):
            await self.ensure_system_dependencies(
                environment,
                ("git", "python3", "python_venv", "ca_certificates"),
            )
        package = (
            f"pureharness @ git+{_REPOSITORY_URL}@{self._pureharness_ref}"
        )
        await self.exec_as_root(
            environment,
            command=f"python3 -m venv {shlex.quote(str(_VENV_DIR))}",
        )
        install_command = (
            f"{shlex.quote(str(_VENV_DIR / 'bin/python'))} "
            "-m pip install --disable-pip-version-check "
            f"{shlex.quote(package)}"
        )
        for attempt in range(1, _INSTALL_ATTEMPTS + 1):
            try:
                await self.exec_as_root(
                    environment,
                    command=install_command,
                )
                break
            except NonZeroAgentExitCodeError:
                if attempt == _INSTALL_ATTEMPTS:
                    raise
                await asyncio.sleep(
                    _INSTALL_RETRY_DELAY_SECONDS * attempt
                )

    async def _resolve_workspace(self, environment: BaseEnvironment) -> str:
        configured = environment.task_env_config.workdir

        if configured:
            path = PurePosixPath(configured)
            if path.is_absolute():
                return str(path)

        result = await self.exec_as_agent(
            environment,
            command="pwd",
        )
        container_cwd = result.stdout.strip()

        if not container_cwd:
            raise RuntimeError("Could not determine container working directory")

        base = PurePosixPath(container_cwd)
        if not base.is_absolute():
            raise RuntimeError(
                f"Container working directory must be absolute: {container_cwd!r}"
            )

        if configured:
            return str(base / configured)

        return str(base)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        workspace = await self._resolve_workspace(environment)
        record_path = self.environment_logs_dir / _RUN_RECORD_NAME
        executable = _VENV_DIR / "bin/pureharness"

        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(str(record_path.parent))} && "
                f"exec {shlex.quote(str(executable))} run "
                f"{shlex.quote(instruction)} "
                f"--workspace {shlex.quote(workspace)} "
                f"--model {shlex.quote(self._deepseek_model)} "
                f"--max-steps {_EVALUATION_MAX_STEPS} "
                f"--record {shlex.quote(str(record_path))} "
                "--output jsonl"
            ),
            env={_API_KEY_ENV: self._deepseek_api_key},
            cwd=workspace,
        )
