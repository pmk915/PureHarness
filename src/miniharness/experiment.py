import json

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import ClassVar

from miniharness.benchmark import (
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkTask,
)


EXPERIMENT_RESULT_SCHEMA_VERSION = 1


class ExperimentSerializationError(BenchmarkError):
    """Raised when experiment evidence cannot be serialized or loaded."""


@dataclass(frozen=True)
class ExperimentResult:
    schema_version: ClassVar[int] = EXPERIMENT_RESULT_SCHEMA_VERSION

    model_id: str
    repetition: int
    duration_seconds: float
    benchmark_result: BenchmarkResult

    def __post_init__(self) -> None:
        _validate_non_empty_text("model_id", self.model_id)
        _validate_positive_int("repetition", self.repetition)
        if (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or not isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError(
                "duration_seconds must be a finite non-negative number"
            )

    @property
    def task_id(self) -> str:
        return self.benchmark_result.task_id

    @property
    def config_id(self) -> str:
        return self.benchmark_result.config_id

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "repetition": self.repetition,
            "duration_seconds": self.duration_seconds,
            "benchmark_result": self.benchmark_result.to_dict(),
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
            raise ExperimentSerializationError(
                f"ExperimentResult is not JSON serializable: {exc}"
            ) from exc

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "ExperimentResult":
        try:
            version = data["schema_version"]
            if (
                type(version) is not int
                or version != EXPERIMENT_RESULT_SCHEMA_VERSION
            ):
                raise ExperimentSerializationError(
                    "Unsupported ExperimentResult schema version: "
                    f"{version!r}"
                )
            benchmark_data = data["benchmark_result"]
            if not isinstance(benchmark_data, dict):
                raise ExperimentSerializationError(
                    "benchmark_result must be an object"
                )
            duration = data["duration_seconds"]
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
            ):
                raise ExperimentSerializationError(
                    "duration_seconds must be a number"
                )

            return cls(
                model_id=_require_string(data, "model_id"),
                repetition=_require_positive_int(
                    data,
                    "repetition",
                ),
                duration_seconds=float(duration),
                benchmark_result=BenchmarkResult.from_dict(
                    benchmark_data
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ExperimentSerializationError):
                raise
            raise ExperimentSerializationError(
                f"Invalid ExperimentResult data: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, value: str) -> "ExperimentResult":
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ExperimentSerializationError(
                f"Malformed ExperimentResult JSON: {exc.msg}"
            ) from exc
        if not isinstance(data, dict):
            raise ExperimentSerializationError(
                "ExperimentResult JSON must contain an object"
            )
        return cls.from_dict(data)


class ExperimentRunner:
    def __init__(
        self,
        benchmark_runner: BenchmarkRunner,
        *,
        model_id: str,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        _validate_non_empty_text("model_id", model_id)
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.benchmark_runner = benchmark_runner
        self.model_id = model_id
        self.clock = clock

    def run(
        self,
        tasks: Sequence[BenchmarkTask],
        configs: Sequence[BenchmarkConfig],
        *,
        repetitions: int = 1,
    ) -> tuple[ExperimentResult, ...]:
        _validate_positive_int("repetitions", repetitions)
        _validate_unique_non_empty_ids(
            "task",
            [task.task_id for task in tasks],
        )
        _validate_unique_non_empty_ids(
            "config",
            [config.config_id for config in configs],
        )

        results = []
        for repetition in range(1, repetitions + 1):
            for task in tasks:
                for config in configs:
                    started_at = self.clock()
                    result = self.benchmark_runner.run_case(task, config)
                    duration = self.clock() - started_at
                    if duration < 0:
                        raise BenchmarkError(
                            "Experiment clock moved backwards"
                        )
                    results.append(
                        ExperimentResult(
                            model_id=self.model_id,
                            repetition=repetition,
                            duration_seconds=duration,
                            benchmark_result=result,
                        )
                    )

        return tuple(results)


@dataclass(frozen=True)
class ExperimentConfigSummary:
    config_id: str
    run_count: int
    success_count: int
    success_rate: float
    average_duration_seconds: float
    average_steps: float
    average_model_calls: float
    average_tool_calls: float
    average_tool_failures: float
    sum_estimated_history_tokens: int
    sum_estimated_task_state_tokens: int
    sum_estimated_tool_schema_tokens: int
    trajectory_compaction_count: int
    tool_result_compaction_count: int


def summarize_experiment(
    results: Sequence[ExperimentResult],
) -> tuple[ExperimentConfigSummary, ...]:
    if not results:
        raise BenchmarkError("Cannot summarize an empty experiment")

    config_ids = tuple(dict.fromkeys(
        result.config_id for result in results
    ))
    summaries = []

    for config_id in config_ids:
        selected = tuple(
            result
            for result in results
            if result.config_id == config_id
        )
        count = len(selected)
        successes = sum(
            result.benchmark_result.task_success
            for result in selected
        )
        records = tuple(
            result.benchmark_result.run_record
            for result in selected
        )
        summaries.append(
            ExperimentConfigSummary(
                config_id=config_id,
                run_count=count,
                success_count=successes,
                success_rate=successes / count,
                average_duration_seconds=(
                    sum(
                        result.duration_seconds
                        for result in selected
                    )
                    / count
                ),
                average_steps=(
                    sum(record.step_count for record in records)
                    / count
                ),
                average_model_calls=(
                    sum(
                        record.model_call_count
                        for record in records
                    )
                    / count
                ),
                average_tool_calls=(
                    sum(
                        record.tool_call_count
                        for record in records
                    )
                    / count
                ),
                average_tool_failures=(
                    sum(
                        record.tool_result_error_count
                        for record in records
                    )
                    / count
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
                tool_result_compaction_count=sum(
                    record.tool_result_compaction_count
                    for record in records
                ),
            )
        )

    return tuple(summaries)


def write_experiment_results(
    path: str | Path,
    results: Iterable[ExperimentResult],
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
            f"Could not write experiment results: {output_path}"
        ) from exc


def load_experiment_results(
    path: str | Path,
) -> tuple[ExperimentResult, ...]:
    input_path = Path(path)
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkError(
            f"Could not read experiment results: {input_path}"
        ) from exc

    results = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ExperimentSerializationError(
                f"Empty experiment result line: {line_number}"
            )
        try:
            results.append(ExperimentResult.from_json(line))
        except ExperimentSerializationError as exc:
            raise ExperimentSerializationError(
                "Invalid experiment result at line "
                f"{line_number}: {exc}"
            ) from exc
    return tuple(results)


def _validate_non_empty_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")


def _validate_positive_int(name: str, value: object) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def _validate_unique_non_empty_ids(
    kind: str,
    values: Sequence[str],
) -> None:
    if not values:
        raise BenchmarkError(
            f"Experiment requires at least one {kind}"
        )
    duplicates = sorted({
        value for value in values if values.count(value) > 1
    })
    if duplicates:
        raise BenchmarkError(
            f"Duplicate experiment {kind} ID(s): "
            + ", ".join(duplicates)
        )


def _require_string(
    data: dict[str, object],
    key: str,
) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise ExperimentSerializationError(f"{key} must be text")
    return value


def _require_positive_int(
    data: dict[str, object],
    key: str,
) -> int:
    value = data[key]
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ExperimentSerializationError(
            f"{key} must be a positive integer"
        )
    return value
