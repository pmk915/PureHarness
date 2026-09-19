import argparse

from pathlib import Path

from dotenv import load_dotenv

from miniharness.benchmark import (
    BenchmarkRunner,
    default_benchmark_configs,
    load_benchmark_tasks,
)
from miniharness.deepseek_model import DeepSeekModel
from miniharness.experiment import (
    ExperimentRunner,
    summarize_experiment,
    write_experiment_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled MiniHarness benchmark with DeepSeek. "
            "This makes paid, nondeterministic API calls."
        )
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("benchmarks/tasks"),
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="task_ids",
        help="Task ID to include; repeat as needed. Defaults to all.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="config_ids",
        help="Config ID to include; repeat as needed. Defaults to all.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    arguments = parser.parse_args()

    load_dotenv()
    tasks = load_benchmark_tasks(arguments.tasks_root)
    configs = default_benchmark_configs()
    if arguments.task_ids:
        requested = set(arguments.task_ids)
        tasks = tuple(
            task for task in tasks if task.task_id in requested
        )
        _require_all_requested(
            "task",
            requested,
            {task.task_id for task in tasks},
        )
    if arguments.config_ids:
        requested = set(arguments.config_ids)
        configs = tuple(
            config
            for config in configs
            if config.config_id in requested
        )
        _require_all_requested(
            "config",
            requested,
            {config.config_id for config in configs},
        )

    model_name = arguments.model
    runner = ExperimentRunner(
        BenchmarkRunner(
            lambda task, config: DeepSeekModel(model=model_name)
        ),
        model_id=f"deepseek:{model_name}",
    )
    results = runner.run(
        tasks,
        configs,
        repetitions=arguments.repetitions,
    )
    write_experiment_results(arguments.output, results)

    for summary in summarize_experiment(results):
        print(
            f"{summary.config_id}: "
            f"{summary.success_count}/{summary.run_count} successful; "
            f"success_rate={summary.success_rate:.3f}; "
            f"avg_steps={summary.average_steps:.2f}; "
            f"avg_tool_calls={summary.average_tool_calls:.2f}; "
            f"avg_duration={summary.average_duration_seconds:.2f}s"
        )
    print(f"Wrote {len(results)} results to {arguments.output}")
    return 0


def _require_all_requested(
    kind: str,
    requested: set[str],
    selected: set[str],
) -> None:
    missing = sorted(requested - selected)
    if missing:
        raise SystemExit(
            f"Unknown {kind} ID(s): {', '.join(missing)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
