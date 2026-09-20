import sys

from dataclasses import replace
from pathlib import Path

import pytest

from pureharness.benchmark import (
    BenchmarkError,
    BenchmarkRunner,
    BenchmarkTask,
    TRUSTED_VERIFIER_PLACEHOLDER,
    default_benchmark_configs,
)
from pureharness.experiment import (
    EXPERIMENT_RESULT_SCHEMA_VERSION,
    ExperimentResult,
    ExperimentRunner,
    ExperimentSerializationError,
    load_experiment_results,
    summarize_experiment,
    write_experiment_results,
)
from pureharness.messages import Message


class CompletingModel:
    def generate(self, messages, tools):
        return Message(role="assistant", content="done")


def _task(tmp_path: Path) -> BenchmarkTask:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "value.txt").write_text(
        "original",
        encoding="utf-8",
    )
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "from pathlib import Path\n"
        "value = Path('value.txt').read_text(encoding='utf-8')\n"
        "raise SystemExit(0 if value == 'fixed' else 1)\n",
        encoding="utf-8",
    )
    return BenchmarkTask(
        task_id="experiment-task",
        prompt="Fix value.txt.",
        fixture_path=fixture,
        verification_argv=(
            sys.executable,
            TRUSTED_VERIFIER_PLACEHOLDER,
        ),
        trusted_verifier_path=verifier,
        selective_tool_names=("read_file", "write_file"),
    )


def _configs():
    raw = default_benchmark_configs(max_steps=1)[0]
    return (
        replace(raw, config_id="config-a"),
        replace(raw, config_id="config-b"),
    )


def test_repeated_experiment_order_duration_and_fresh_models(tmp_path):
    task = _task(tmp_path)
    created_models = []

    def model_factory(task, config):
        model = CompletingModel()
        created_models.append(model)
        return model

    run_ids = iter(["run-1", "run-2", "run-3", "run-4"])
    clock_values = iter([0.0, 1.0, 2.0, 4.0, 5.0, 8.0, 9.0, 13.0])
    runner = ExperimentRunner(
        BenchmarkRunner(
            model_factory,
            run_id_factory=run_ids.__next__,
        ),
        model_id="test-model",
        clock=clock_values.__next__,
    )

    results = runner.run(
        (task,),
        _configs(),
        repetitions=2,
    )

    assert [
        (
            result.repetition,
            result.task_id,
            result.config_id,
            result.duration_seconds,
        )
        for result in results
    ] == [
        (1, "experiment-task", "config-a", 1.0),
        (1, "experiment-task", "config-b", 2.0),
        (2, "experiment-task", "config-a", 3.0),
        (2, "experiment-task", "config-b", 4.0),
    ]
    assert len(created_models) == 4
    assert all(
        result.model_id == "test-model"
        and result.benchmark_result.agent_end_reason == "completed"
        and result.benchmark_result.task_success is False
        for result in results
    )


def test_experiment_result_round_trip_and_schema_rejection(tmp_path):
    task = _task(tmp_path)
    result = ExperimentRunner(
        BenchmarkRunner(
            lambda task, config: CompletingModel(),
            run_id_factory=lambda: "experiment-record",
        ),
        model_id="test-model",
        clock=iter([10.0, 10.5]).__next__,
    ).run((task,), (_configs()[0],))[0]

    restored = ExperimentResult.from_json(result.to_json())

    assert restored == result
    assert restored.to_dict()["schema_version"] == (
        EXPERIMENT_RESULT_SCHEMA_VERSION
    )

    data = result.to_dict()
    data["schema_version"] = 2
    with pytest.raises(
        ExperimentSerializationError,
        match="Unsupported ExperimentResult schema version",
    ):
        ExperimentResult.from_dict(data)


def test_experiment_jsonl_and_summary(tmp_path):
    task = _task(tmp_path)
    clock_values = iter([0.0, 1.0, 2.0, 4.0, 5.0, 8.0, 9.0, 13.0])
    run_ids = iter(["a", "b", "c", "d"])
    results = ExperimentRunner(
        BenchmarkRunner(
            lambda task, config: CompletingModel(),
            run_id_factory=run_ids.__next__,
        ),
        model_id="test-model",
        clock=clock_values.__next__,
    ).run((task,), _configs(), repetitions=2)
    output = tmp_path / "experiment.jsonl"

    write_experiment_results(output, results[:2])
    write_experiment_results(output, results[2:], append=True)
    restored = load_experiment_results(output)
    summaries = summarize_experiment(restored)

    assert restored == results
    assert [summary.config_id for summary in summaries] == [
        "config-a",
        "config-b",
    ]
    assert [summary.run_count for summary in summaries] == [2, 2]
    assert [summary.success_count for summary in summaries] == [0, 0]
    assert [summary.success_rate for summary in summaries] == [0.0, 0.0]
    assert [
        summary.average_duration_seconds for summary in summaries
    ] == [2.0, 3.0]
    assert all(summary.average_steps == 1.0 for summary in summaries)
    assert all(
        summary.average_model_calls == 1.0
        for summary in summaries
    )


@pytest.mark.parametrize("repetitions", [0, -1, True])
def test_experiment_rejects_invalid_repetitions(tmp_path, repetitions):
    task = _task(tmp_path)
    runner = ExperimentRunner(
        BenchmarkRunner(
            lambda task, config: CompletingModel()
        ),
        model_id="test-model",
    )

    with pytest.raises(ValueError, match="positive integer"):
        runner.run(
            (task,),
            (_configs()[0],),
            repetitions=repetitions,
        )


def test_experiment_rejects_empty_or_duplicate_inputs(tmp_path):
    task = _task(tmp_path)
    config = _configs()[0]
    runner = ExperimentRunner(
        BenchmarkRunner(
            lambda task, config: CompletingModel()
        ),
        model_id="test-model",
    )

    with pytest.raises(BenchmarkError, match="at least one task"):
        runner.run((), (config,))
    with pytest.raises(BenchmarkError, match="Duplicate.*task"):
        runner.run((task, task), (config,))
    with pytest.raises(BenchmarkError, match="Duplicate.*config"):
        runner.run((task,), (config, config))


def test_experiment_layer_has_no_provider_dependency():
    source = Path("src/pureharness/experiment.py").read_text(
        encoding="utf-8"
    ).lower()

    assert "deepseek" not in source
    assert "openai" not in source
