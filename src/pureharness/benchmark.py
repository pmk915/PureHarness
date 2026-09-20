import json
import shutil

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

from pureharness.agent import Agent
from pureharness.coding_tools import create_coding_tools
from pureharness.context import (
    ContextBudget,
    ContextBuilder,
    TokenBudgetContextBuilder,
)
from pureharness.execution import (
    ExecutionBackend,
    ExecutionError,
    ExecutionTimeoutError,
    LocalExecutionBackend,
)
from pureharness.model import Model
from pureharness.run_record import RunRecord
from pureharness.tool_result_projection import (
    DeterministicToolResultProjector,
    IdentityToolResultProjector,
)
from pureharness.tool_selection import (
    AllToolsSelector,
    StaticToolSelector,
    ToolSelector,
)
from pureharness.tools import ToolRegistry
from pureharness.trace import RUN_END_REASONS, RunEndReason
from pureharness.trajectory_compaction import (
    DeterministicToolTrajectoryCompactor,
    IdentityTrajectoryCompactor,
    default_compactor_for_history_budget,
)


BENCHMARK_RESULT_SCHEMA_VERSION = 1
BENCHMARK_TASK_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_HISTORY_TOKEN_BUDGET = 8_000
DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 30.0
DEFAULT_VERIFICATION_PREVIEW_CHARS = 2_000
TRUSTED_VERIFIER_PLACEHOLDER = "{trusted_verifier}"

_CONTEXT_STRATEGIES = frozenset({"full_history", "token_budget"})
_COMPONENT_MODES = frozenset({"identity", "deterministic"})
_TOOL_SELECTION_MODES = frozenset({"all", "static"})


class BenchmarkError(RuntimeError):
    """Raised when benchmark infrastructure cannot run a valid case."""


class BenchmarkSerializationError(BenchmarkError):
    """Raised when benchmark data cannot be serialized or loaded."""


class BenchmarkVerifierError(BenchmarkError):
    """Raised when trusted verification infrastructure fails."""


class BenchmarkVerifierTimeout(BenchmarkVerifierError):
    """Raised when trusted verification exceeds its timeout."""


class BenchmarkIntegrityError(BenchmarkError):
    """Raised when trusted benchmark inputs change during a case."""


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    prompt: str
    fixture_path: Path
    verification_argv: tuple[str, ...]
    trusted_verifier_path: Path | None = None
    selective_tool_names: tuple[str, ...] = ()
    declared_required_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_text("task_id", self.task_id)
        _validate_non_empty_text("prompt", self.prompt)
        object.__setattr__(self, "fixture_path", Path(self.fixture_path))
        if self.trusted_verifier_path is not None:
            object.__setattr__(
                self,
                "trusted_verifier_path",
                Path(self.trusted_verifier_path),
            )
        _validate_string_tuple(
            "verification_argv",
            self.verification_argv,
            allow_empty=False,
        )
        placeholder_count = self.verification_argv.count(
            TRUSTED_VERIFIER_PLACEHOLDER
        )
        if self.trusted_verifier_path is None and placeholder_count:
            raise ValueError(
                "verification_argv cannot reference a missing "
                "trusted_verifier_path"
            )
        if (
            self.trusted_verifier_path is not None
            and placeholder_count != 1
        ):
            raise ValueError(
                "verification_argv must contain exactly one "
                "trusted verifier placeholder"
            )
        _validate_string_tuple(
            "selective_tool_names",
            self.selective_tool_names,
        )
        _validate_string_tuple(
            "declared_required_tools",
            self.declared_required_tools,
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> "BenchmarkTask":
        task_directory = Path(directory)
        metadata_path = task_directory / "task.json"

        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkSerializationError(
                f"Could not load benchmark task metadata: {metadata_path}"
            ) from exc

        if not isinstance(data, dict):
            raise BenchmarkSerializationError(
                "Benchmark task metadata must be a JSON object"
            )

        try:
            version = data["schema_version"]
            if (
                type(version) is not int
                or version != BENCHMARK_TASK_SCHEMA_VERSION
            ):
                raise BenchmarkSerializationError(
                    "Unsupported BenchmarkTask schema version: "
                    f"{version!r}"
                )

            return cls(
                task_id=_require_string(data, "id"),
                prompt=_require_string(data, "prompt"),
                fixture_path=task_directory / "workspace",
                verification_argv=_require_string_tuple(
                    data,
                    "verification_argv",
                ),
                trusted_verifier_path=_optional_relative_path(
                    data,
                    "trusted_verifier",
                    base=task_directory,
                ),
                selective_tool_names=_optional_string_tuple(
                    data,
                    "selective_tool_names",
                ),
                declared_required_tools=_optional_string_tuple(
                    data,
                    "declared_required_tools",
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, BenchmarkSerializationError):
                raise
            raise BenchmarkSerializationError(
                f"Invalid BenchmarkTask metadata: {exc}"
            ) from exc


@dataclass(frozen=True)
class BenchmarkConfig:
    config_id: str
    context_strategy: str
    history_token_budget: int | None
    tool_result_projection: str
    trajectory_compaction: str
    include_task_state: bool
    tool_selection: str
    max_steps: int = 10

    def __post_init__(self) -> None:
        _validate_non_empty_text("config_id", self.config_id)
        if self.context_strategy not in _CONTEXT_STRATEGIES:
            raise ValueError(
                f"Unsupported context strategy: {self.context_strategy!r}"
            )
        if self.tool_result_projection not in _COMPONENT_MODES:
            raise ValueError(
                "tool_result_projection must be identity or deterministic"
            )
        if self.trajectory_compaction not in _COMPONENT_MODES:
            raise ValueError(
                "trajectory_compaction must be identity or deterministic"
            )
        if not isinstance(self.include_task_state, bool):
            raise ValueError("include_task_state must be bool")
        if self.tool_selection not in _TOOL_SELECTION_MODES:
            raise ValueError("tool_selection must be all or static")
        _validate_positive_int("max_steps", self.max_steps)

        if self.context_strategy == "token_budget":
            _validate_positive_int(
                "history_token_budget",
                self.history_token_budget,
            )
        elif self.history_token_budget is not None:
            raise ValueError(
                "FullHistory config must not define a history token budget"
            )

    def create_context_builder(self) -> ContextBuilder:
        if self.tool_result_projection == "identity":
            projector = IdentityToolResultProjector()
        else:
            projector = DeterministicToolResultProjector()

        if self.trajectory_compaction == "identity":
            compactor = IdentityTrajectoryCompactor()
        elif self.history_token_budget is None:
            compactor = DeterministicToolTrajectoryCompactor()
        else:
            compactor = default_compactor_for_history_budget(
                self.history_token_budget
            )

        if self.context_strategy == "full_history":
            return ContextBuilder(
                tool_result_projector=projector,
                trajectory_compactor=compactor,
            )

        assert self.history_token_budget is not None
        return TokenBudgetContextBuilder(
            ContextBudget(
                max_estimated_tokens=self.history_token_budget
            ),
            tool_result_projector=projector,
            trajectory_compactor=compactor,
        )

    def create_tool_selector(
        self,
        task: BenchmarkTask,
    ) -> ToolSelector:
        if self.tool_selection == "all":
            return AllToolsSelector()

        return StaticToolSelector(task.selective_tool_names)


def default_benchmark_configs(
    *,
    history_token_budget: int = (
        DEFAULT_BENCHMARK_HISTORY_TOKEN_BUDGET
    ),
    max_steps: int = 10,
) -> tuple[BenchmarkConfig, ...]:
    return (
        BenchmarkConfig(
            config_id="raw_baseline",
            context_strategy="full_history",
            history_token_budget=None,
            tool_result_projection="identity",
            trajectory_compaction="identity",
            include_task_state=False,
            tool_selection="all",
            max_steps=max_steps,
        ),
        BenchmarkConfig(
            config_id="budget_only",
            context_strategy="token_budget",
            history_token_budget=history_token_budget,
            tool_result_projection="identity",
            trajectory_compaction="identity",
            include_task_state=False,
            tool_selection="all",
            max_steps=max_steps,
        ),
        BenchmarkConfig(
            config_id="context_engineered",
            context_strategy="token_budget",
            history_token_budget=history_token_budget,
            tool_result_projection="deterministic",
            trajectory_compaction="deterministic",
            include_task_state=True,
            tool_selection="all",
            max_steps=max_steps,
        ),
        BenchmarkConfig(
            config_id="full_pureharness",
            context_strategy="token_budget",
            history_token_budget=history_token_budget,
            tool_result_projection="deterministic",
            trajectory_compaction="deterministic",
            include_task_state=True,
            tool_selection="static",
            max_steps=max_steps,
        ),
    )


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: ClassVar[int] = BENCHMARK_RESULT_SCHEMA_VERSION

    task_id: str
    config_id: str
    task_success: bool
    verification_exit_code: int
    verification_stdout_preview: str
    verification_stderr_preview: str
    agent_end_reason: RunEndReason
    agent_error_type: str | None
    max_steps: int
    context_strategy: str
    selector_strategy: str
    run_record: RunRecord

    def __post_init__(self) -> None:
        _validate_non_empty_text("task_id", self.task_id)
        _validate_non_empty_text("config_id", self.config_id)
        if not isinstance(self.task_success, bool):
            raise ValueError("task_success must be bool")
        if (
            not isinstance(self.verification_exit_code, int)
            or isinstance(self.verification_exit_code, bool)
        ):
            raise ValueError("verification_exit_code must be an integer")
        if self.task_success != (self.verification_exit_code == 0):
            raise ValueError(
                "task_success must reflect verification exit code"
            )
        for name in (
            "verification_stdout_preview",
            "verification_stderr_preview",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be text")
        _validate_non_empty_text(
            "context_strategy",
            self.context_strategy,
        )
        _validate_non_empty_text(
            "selector_strategy",
            self.selector_strategy,
        )
        if self.agent_error_type is not None:
            _validate_non_empty_text(
                "agent_error_type",
                self.agent_error_type,
            )
        if self.agent_end_reason not in RUN_END_REASONS:
            raise ValueError(
                f"Unsupported agent end reason: {self.agent_end_reason!r}"
            )
        if self.agent_end_reason != self.run_record.end_reason:
            raise ValueError(
                "agent_end_reason must match RunRecord end_reason"
            )
        _validate_positive_int("max_steps", self.max_steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "config_id": self.config_id,
            "task_success": self.task_success,
            "verification_exit_code": self.verification_exit_code,
            "verification_stdout_preview": (
                self.verification_stdout_preview
            ),
            "verification_stderr_preview": (
                self.verification_stderr_preview
            ),
            "agent_end_reason": self.agent_end_reason,
            "agent_error_type": self.agent_error_type,
            "max_steps": self.max_steps,
            "context_strategy": self.context_strategy,
            "selector_strategy": self.selector_strategy,
            "run_record": self.run_record.to_dict(),
        }

    def to_json(self) -> str:
        try:
            return json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkSerializationError(
                f"BenchmarkResult is not JSON serializable: {exc}"
            ) from exc

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "BenchmarkResult":
        try:
            version = data["schema_version"]
            if (
                type(version) is not int
                or version != BENCHMARK_RESULT_SCHEMA_VERSION
            ):
                raise BenchmarkSerializationError(
                    "Unsupported BenchmarkResult schema version: "
                    f"{version!r}"
                )
            run_record_data = data["run_record"]
            if not isinstance(run_record_data, dict):
                raise BenchmarkSerializationError(
                    "run_record must be an object"
                )
            task_success = data["task_success"]
            if not isinstance(task_success, bool):
                raise BenchmarkSerializationError(
                    "task_success must be bool"
                )
            error_type = data["agent_error_type"]
            if error_type is not None and not isinstance(error_type, str):
                raise BenchmarkSerializationError(
                    "agent_error_type must be text or null"
                )
            end_reason = _require_string(data, "agent_end_reason")
            if end_reason not in RUN_END_REASONS:
                raise BenchmarkSerializationError(
                    f"Unsupported agent end reason: {end_reason!r}"
                )

            return cls(
                task_id=_require_string(data, "task_id"),
                config_id=_require_string(data, "config_id"),
                task_success=task_success,
                verification_exit_code=_require_int(
                    data,
                    "verification_exit_code",
                ),
                verification_stdout_preview=_require_string(
                    data,
                    "verification_stdout_preview",
                ),
                verification_stderr_preview=_require_string(
                    data,
                    "verification_stderr_preview",
                ),
                agent_end_reason=end_reason,
                agent_error_type=error_type,
                max_steps=_require_positive_int(data, "max_steps"),
                context_strategy=_require_string(
                    data,
                    "context_strategy",
                ),
                selector_strategy=_require_string(
                    data,
                    "selector_strategy",
                ),
                run_record=RunRecord.from_dict(run_record_data),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, BenchmarkSerializationError):
                raise
            raise BenchmarkSerializationError(
                f"Invalid BenchmarkResult data: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, value: str) -> "BenchmarkResult":
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise BenchmarkSerializationError(
                f"Malformed BenchmarkResult JSON: {exc.msg}"
            ) from exc

        if not isinstance(data, dict):
            raise BenchmarkSerializationError(
                "BenchmarkResult JSON must contain an object"
            )
        return cls.from_dict(data)


ModelFactory = Callable[[BenchmarkTask, BenchmarkConfig], Model]


class BenchmarkRunner:
    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        verification_backend: ExecutionBackend | None = None,
        verification_timeout_seconds: float = (
            DEFAULT_VERIFICATION_TIMEOUT_SECONDS
        ),
        verification_preview_chars: int = (
            DEFAULT_VERIFICATION_PREVIEW_CHARS
        ),
        workspace_parent: str | Path | None = None,
        run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not callable(model_factory):
            raise ValueError("model_factory must be callable")
        _validate_positive_number(
            "verification_timeout_seconds",
            verification_timeout_seconds,
        )
        _validate_positive_int(
            "verification_preview_chars",
            verification_preview_chars,
        )

        self.model_factory = model_factory
        self.verification_backend = (
            verification_backend
            if verification_backend is not None
            else LocalExecutionBackend()
        )
        self.verification_timeout_seconds = (
            verification_timeout_seconds
        )
        self.verification_preview_chars = verification_preview_chars
        self.workspace_parent = (
            None
            if workspace_parent is None
            else Path(workspace_parent)
        )
        self.run_id_factory = run_id_factory

    def run_case(
        self,
        task: BenchmarkTask,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        fixture = task.fixture_path.resolve()
        if not fixture.is_dir():
            raise BenchmarkError(
                f"Benchmark fixture directory does not exist: {fixture}"
            )
        trusted_verifier_snapshot: bytes | None = None
        trusted_verifier_path: Path | None = None
        if task.trusted_verifier_path is not None:
            trusted_verifier_path = (
                task.trusted_verifier_path.resolve()
            )
            if (
                trusted_verifier_path == fixture
                or fixture in trusted_verifier_path.parents
            ):
                raise BenchmarkError(
                    "Trusted verifier must be outside the Agent workspace "
                    "fixture"
                )
            try:
                trusted_verifier_snapshot = (
                    trusted_verifier_path.read_bytes()
                )
            except OSError as exc:
                raise BenchmarkError(
                    "Could not read trusted verifier: "
                    f"{trusted_verifier_path}"
                ) from exc
        if (
            self.workspace_parent is not None
            and not self.workspace_parent.is_dir()
        ):
            raise BenchmarkError(
                "Benchmark workspace parent does not exist: "
                f"{self.workspace_parent}"
            )

        try:
            temporary = TemporaryDirectory(
                prefix="pureharness-benchmark-",
                dir=self.workspace_parent,
            )
        except OSError as exc:
            raise BenchmarkError(
                "Could not create benchmark temporary directory"
            ) from exc

        with temporary as temporary_path:
            workspace = Path(temporary_path) / "workspace"
            try:
                shutil.copytree(fixture, workspace)
            except OSError as exc:
                raise BenchmarkError(
                    f"Could not copy benchmark fixture: {fixture}"
                ) from exc

            registry = ToolRegistry()
            agent_backend = LocalExecutionBackend()
            for tool in create_coding_tools(
                workspace,
                execution_backend=agent_backend,
            ):
                registry.register(tool)

            try:
                model = self.model_factory(task, config)
                context_builder = config.create_context_builder()
                tool_selector = config.create_tool_selector(task)
            except Exception as exc:
                raise BenchmarkError(
                    f"Could not construct benchmark case: {exc}"
                ) from exc

            agent = Agent(
                model=model,
                tools=registry,
                max_steps=config.max_steps,
                context_builder=context_builder,
                tool_selector=tool_selector,
                include_task_state=config.include_task_state,
                run_id_factory=self.run_id_factory,
            )
            agent_error_type: str | None = None

            try:
                agent.run(task.prompt)
            except Exception as exc:
                if agent.last_run_record is None:
                    raise BenchmarkError(
                        "Agent failed without producing a RunRecord"
                    ) from exc
                agent_error_type = type(exc).__name__

            record = agent.last_run_record
            if record is None:
                raise BenchmarkError(
                    "Agent completed without producing a RunRecord"
                )

            verification_argv = list(task.verification_argv)
            if (
                trusted_verifier_path is not None
                and trusted_verifier_snapshot is not None
            ):
                try:
                    current_verifier = trusted_verifier_path.read_bytes()
                except OSError as exc:
                    raise BenchmarkIntegrityError(
                        "Trusted verifier became unreadable during "
                        "Agent execution"
                    ) from exc
                if current_verifier != trusted_verifier_snapshot:
                    raise BenchmarkIntegrityError(
                        "Trusted verifier changed during Agent execution"
                    )

                verifier_copy = (
                    Path(temporary_path)
                    / "trusted-verifier"
                    / trusted_verifier_path.name
                )
                try:
                    verifier_copy.parent.mkdir()
                    verifier_copy.write_bytes(trusted_verifier_snapshot)
                except OSError as exc:
                    raise BenchmarkError(
                        "Could not materialize trusted verifier"
                    ) from exc
                verification_argv = [
                    (
                        str(verifier_copy)
                        if item == TRUSTED_VERIFIER_PLACEHOLDER
                        else item
                    )
                    for item in verification_argv
                ]

            try:
                verification = self.verification_backend.execute(
                    verification_argv,
                    cwd=workspace,
                    timeout=self.verification_timeout_seconds,
                )
            except ExecutionTimeoutError as exc:
                raise BenchmarkVerifierTimeout(
                    "Benchmark verification timed out after "
                    f"{self.verification_timeout_seconds} seconds"
                ) from exc
            except ExecutionError as exc:
                raise BenchmarkVerifierError(
                    "Benchmark verification could not be executed"
                ) from exc

            return BenchmarkResult(
                task_id=task.task_id,
                config_id=config.config_id,
                task_success=verification.exit_code == 0,
                verification_exit_code=verification.exit_code,
                verification_stdout_preview=_bounded_preview(
                    verification.stdout,
                    self.verification_preview_chars,
                ),
                verification_stderr_preview=_bounded_preview(
                    verification.stderr,
                    self.verification_preview_chars,
                ),
                agent_end_reason=record.end_reason,
                agent_error_type=agent_error_type,
                max_steps=config.max_steps,
                context_strategy=context_builder.strategy,
                selector_strategy=getattr(
                    tool_selector,
                    "strategy",
                    type(tool_selector).__name__,
                ),
                run_record=record,
            )

    def run_suite(
        self,
        tasks: Sequence[BenchmarkTask],
        configs: Sequence[BenchmarkConfig],
    ) -> "BenchmarkSuiteResult":
        _validate_unique_non_empty_ids(
            "task",
            [task.task_id for task in tasks],
        )
        _validate_unique_non_empty_ids(
            "config",
            [config.config_id for config in configs],
        )

        results = tuple(
            self.run_case(task, config)
            for task in tasks
            for config in configs
        )
        return BenchmarkSuiteResult(
            results=results,
            summaries=summarize_results(results),
        )


@dataclass(frozen=True)
class BenchmarkConfigSummary:
    config_id: str
    case_count: int
    success_count: int
    success_rate: float
    end_reason_counts: tuple[tuple[str, int], ...]
    total_model_calls: int
    mean_model_calls: float
    total_steps: int
    mean_steps: float
    total_tool_calls: int
    total_tool_executions: int
    total_tool_result_errors: int
    sum_estimated_history_tokens: int
    sum_estimated_task_state_tokens: int
    sum_estimated_tool_schema_tokens: int
    trajectory_compaction_count: int
    compacted_source_unit_count: int
    tool_result_compaction_count: int


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    results: tuple[BenchmarkResult, ...]
    summaries: tuple[BenchmarkConfigSummary, ...]


def summarize_results(
    results: Sequence[BenchmarkResult],
) -> tuple[BenchmarkConfigSummary, ...]:
    if not results:
        raise BenchmarkError("Cannot summarize an empty benchmark result set")

    config_ids = tuple(dict.fromkeys(
        result.config_id for result in results
    ))
    summaries = []

    for config_id in config_ids:
        config_results = tuple(
            result
            for result in results
            if result.config_id == config_id
        )
        records = tuple(
            result.run_record for result in config_results
        )
        case_count = len(config_results)
        success_count = sum(
            result.task_success for result in config_results
        )
        end_reasons = sorted({
            result.agent_end_reason for result in config_results
        })

        total_model_calls = sum(
            record.model_call_count for record in records
        )
        total_steps = sum(record.step_count for record in records)
        summaries.append(
            BenchmarkConfigSummary(
                config_id=config_id,
                case_count=case_count,
                success_count=success_count,
                success_rate=success_count / case_count,
                end_reason_counts=tuple(
                    (
                        reason,
                        sum(
                            result.agent_end_reason == reason
                            for result in config_results
                        ),
                    )
                    for reason in end_reasons
                ),
                total_model_calls=total_model_calls,
                mean_model_calls=total_model_calls / case_count,
                total_steps=total_steps,
                mean_steps=total_steps / case_count,
                total_tool_calls=sum(
                    record.tool_call_count for record in records
                ),
                total_tool_executions=sum(
                    record.tool_execution_count for record in records
                ),
                total_tool_result_errors=sum(
                    record.tool_result_error_count for record in records
                ),
                sum_estimated_history_tokens=sum(
                    record.sum_estimated_history_tokens
                    for record in records
                ),
                sum_estimated_task_state_tokens=sum(
                    record.sum_estimated_task_state_tokens
                    for record in records
                ),
                sum_estimated_tool_schema_tokens=sum(
                    record.sum_estimated_tool_schema_tokens
                    for record in records
                ),
                trajectory_compaction_count=sum(
                    record.trajectory_compaction_count
                    for record in records
                ),
                compacted_source_unit_count=sum(
                    record.compacted_source_unit_count
                    for record in records
                ),
                tool_result_compaction_count=sum(
                    record.tool_result_compaction_count
                    for record in records
                ),
            )
        )

    return tuple(summaries)


def write_benchmark_results(
    path: str | Path,
    results: Iterable[BenchmarkResult],
    *,
    append: bool = False,
) -> None:
    output_path = Path(path)
    lines = tuple(result.to_json() + "\n" for result in results)
    mode = "a" if append else "w"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(mode, encoding="utf-8") as handle:
            handle.writelines(lines)
            handle.flush()
    except OSError as exc:
        raise BenchmarkError(
            f"Could not write benchmark results: {output_path}"
        ) from exc


def load_benchmark_results(
    path: str | Path,
) -> tuple[BenchmarkResult, ...]:
    input_path = Path(path)
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkError(
            f"Could not read benchmark results: {input_path}"
        ) from exc

    results = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise BenchmarkSerializationError(
                f"Empty benchmark result line: {line_number}"
            )
        try:
            results.append(BenchmarkResult.from_json(line))
        except BenchmarkSerializationError as exc:
            raise BenchmarkSerializationError(
                "Invalid benchmark result at line "
                f"{line_number}: {exc}"
            ) from exc

    return tuple(results)


def load_benchmark_tasks(
    root: str | Path,
) -> tuple[BenchmarkTask, ...]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise BenchmarkError(
            f"Benchmark task directory does not exist: {root_path}"
        )

    tasks = tuple(
        BenchmarkTask.from_directory(path)
        for path in sorted(root_path.iterdir())
        if path.is_dir() and (path / "task.json").is_file()
    )
    _validate_unique_non_empty_ids(
        "task",
        [task.task_id for task in tasks],
    )
    return tasks


def _bounded_preview(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    marker = "\n... verification output omitted ...\n"
    if max_chars <= len(marker):
        return value[:max_chars]

    available = max_chars - len(marker)
    head_chars = max(0, available // 2)
    tail_chars = max(0, available - head_chars)
    return value[:head_chars] + marker + value[-tail_chars:]


def _validate_unique_non_empty_ids(
    kind: str,
    values: Sequence[str],
) -> None:
    if not values:
        raise BenchmarkError(
            f"Benchmark suite requires at least one {kind}"
        )
    duplicates = sorted({
        value for value in values if values.count(value) > 1
    })
    if duplicates:
        raise BenchmarkError(
            f"Duplicate benchmark {kind} ID(s): "
            + ", ".join(duplicates)
        )


def _validate_non_empty_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")


def _validate_string_tuple(
    name: str,
    value: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain non-empty text")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")


def _validate_positive_int(name: str, value: object) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _validate_positive_number(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be greater than zero")


def _require_string(
    data: dict[str, object],
    key: str,
) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise BenchmarkSerializationError(f"{key} must be text")
    return value


def _require_int(
    data: dict[str, object],
    key: str,
) -> int:
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise BenchmarkSerializationError(f"{key} must be an integer")
    return value


def _require_positive_int(
    data: dict[str, object],
    key: str,
) -> int:
    value = _require_int(data, key)
    if value <= 0:
        raise BenchmarkSerializationError(
            f"{key} must be a positive integer"
        )
    return value


def _require_string_tuple(
    data: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or not value:
        raise BenchmarkSerializationError(
            f"{key} must be a non-empty list"
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise BenchmarkSerializationError(
            f"{key} must contain non-empty text"
        )
    return tuple(value)


def _optional_string_tuple(
    data: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise BenchmarkSerializationError(f"{key} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise BenchmarkSerializationError(
            f"{key} must contain non-empty text"
        )
    return tuple(value)


def _optional_relative_path(
    data: dict[str, object],
    key: str,
    *,
    base: Path,
) -> Path | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BenchmarkSerializationError(
            f"{key} must be non-empty text"
        )

    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkSerializationError(
            f"{key} must be a safe relative path"
        )
    return base / relative
