#!/usr/bin/env python3
"""Create a reproducibility receipt from an existing Harbor job directory."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from collections.abc import Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}\Z")
_RUN_RECORD_NAME = "pureharness-run-record.json"


class ReceiptError(RuntimeError):
    """Raised when a trustworthy receipt cannot be produced."""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReceiptError(f"Missing {label}: {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReceiptError(f"Invalid JSON in {label}: {path}") from exc
    except OSError as exc:
        raise ReceiptError(f"Could not read {label}: {path}") from exc

    if not isinstance(value, dict):
        raise ReceiptError(f"{label} must contain a JSON object: {path}")
    return value


def _required_string(
    data: dict[str, Any],
    key: str,
    label: str,
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{label}.{key} must be a non-empty string")
    return value


def _required_count(
    data: dict[str, Any],
    key: str,
    label: str,
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReceiptError(f"{label}.{key} must be a non-negative integer")
    return value


def _job_mean_reward(stats: dict[str, Any]) -> float | int:
    evals = stats.get("evals")
    if not isinstance(evals, dict):
        raise ReceiptError("Harbor job result.stats.evals must be an object")
    if len(evals) != 1:
        raise ReceiptError(
            "Harbor job result.stats.evals must contain exactly one eval entry"
        )

    eval_name, eval_result = next(iter(evals.items()))
    if not isinstance(eval_result, dict):
        raise ReceiptError(
            f"Harbor eval {eval_name!r} must contain an object"
        )

    metrics = eval_result.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ReceiptError(
            f"Harbor eval {eval_name!r}.metrics must be a non-empty list"
        )
    first_metric = metrics[0]
    if not isinstance(first_metric, dict):
        raise ReceiptError(
            f"Harbor eval {eval_name!r}.metrics[0] must be an object"
        )

    mean = first_metric.get("mean")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)):
        raise ReceiptError(
            f"Harbor eval {eval_name!r}.metrics[0].mean must be numeric"
        )
    return mean


def _run_command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    label: str,
) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ReceiptError(f"Could not determine {label}: {argv[0]} not found") from exc
    except subprocess.CalledProcessError as exc:
        raise ReceiptError(f"Could not determine {label}") from exc
    return result.stdout.strip()


def _repository_state(repository_root: Path) -> tuple[str, bool]:
    commit = _run_command(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        label="repository commit",
    )
    if not _SHA_PATTERN.fullmatch(commit):
        raise ReceiptError("Repository HEAD is not a full 40-character Git SHA")

    status = _run_command(
        ("git", "status", "--porcelain=v1"),
        cwd=repository_root,
        label="repository worktree status",
    )
    return commit.lower(), not status


def _harbor_version() -> str:
    version = _run_command(
        ("harbor", "--version"),
        label="Harbor version",
    )
    if not version:
        raise ReceiptError("Harbor version command returned no version")
    return version


def _find_run_record(trial_dir: Path, job_dir: Path) -> str | None:
    candidates: set[Path] = set()
    for directory_name in ("agent", "artifacts", "logs"):
        search_root = trial_dir / directory_name
        if search_root.is_dir():
            candidates.update(
                path
                for path in search_root.rglob(_RUN_RECORD_NAME)
                if path.is_file()
            )

    ordered = sorted(candidates)
    if len(ordered) > 1:
        relative_paths = ", ".join(
            path.relative_to(job_dir).as_posix() for path in ordered
        )
        raise ReceiptError(
            f"Multiple PureHarness RunRecords found for {trial_dir.name}: "
            f"{relative_paths}"
        )
    if not ordered:
        return None

    candidate = ordered[0]
    try:
        candidate.resolve().relative_to(job_dir.resolve())
    except ValueError as exc:
        raise ReceiptError(
            f"RunRecord escapes Harbor job directory: {candidate}"
        ) from exc
    return candidate.relative_to(job_dir).as_posix()


def _trial_receipt(result_path: Path, job_dir: Path) -> dict[str, Any]:
    data = _load_json_object(result_path, "Harbor trial result")
    trial_label = f"trial {result_path.parent.name}"

    task_id = data.get("task_id")
    if not isinstance(task_id, dict):
        raise ReceiptError(f"{trial_label}.task_id must be an object")
    task_path = task_id.get("path")
    if task_path is not None and not isinstance(task_path, str):
        raise ReceiptError(f"{trial_label}.task_id.path must be a string or null")

    agent_info = data.get("agent_info")
    if not isinstance(agent_info, dict):
        raise ReceiptError(f"{trial_label}.agent_info must be an object")

    model_info = agent_info.get("model_info")
    if model_info is None:
        model_provider = None
        model_name = None
    elif isinstance(model_info, dict):
        model_provider = model_info.get("provider")
        model_name = model_info.get("name")
        if model_provider is not None and not isinstance(model_provider, str):
            raise ReceiptError(
                f"{trial_label}.agent_info.model_info.provider "
                "must be a string or null"
            )
        if model_name is not None and not isinstance(model_name, str):
            raise ReceiptError(
                f"{trial_label}.agent_info.model_info.name "
                "must be a string or null"
            )
    else:
        raise ReceiptError(f"{trial_label}.agent_info.model_info must be an object")

    config = data.get("config")
    if not isinstance(config, dict):
        raise ReceiptError(f"{trial_label}.config must be an object")
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise ReceiptError(f"{trial_label}.config.environment must be an object")
    environment_type = environment.get("type")
    if environment_type is not None and not isinstance(environment_type, str):
        raise ReceiptError(
            f"{trial_label}.config.environment.type must be a string or null"
        )

    verifier_result = data.get("verifier_result")
    reward: float | int | None = None
    if verifier_result is not None:
        if not isinstance(verifier_result, dict):
            raise ReceiptError(f"{trial_label}.verifier_result must be an object")
        rewards = verifier_result.get("rewards")
        if rewards is not None:
            if not isinstance(rewards, dict):
                raise ReceiptError(
                    f"{trial_label}.verifier_result.rewards must be an object"
                )
            reward = rewards.get("reward")
            if (
                reward is not None
                and (
                    isinstance(reward, bool)
                    or not isinstance(reward, (int, float))
                )
            ):
                raise ReceiptError(
                    f"{trial_label}.verifier_result.rewards.reward "
                    "must be numeric or null"
                )

    exception_info = data.get("exception_info")
    if exception_info is None:
        exception = None
    elif isinstance(exception_info, dict):
        exception = {
            "type": _required_string(
                exception_info,
                "exception_type",
                f"{trial_label}.exception_info",
            ),
            "occurred_at": exception_info.get("occurred_at"),
        }
        if (
            exception["occurred_at"] is not None
            and not isinstance(exception["occurred_at"], str)
        ):
            raise ReceiptError(
                f"{trial_label}.exception_info.occurred_at "
                "must be a string or null"
            )
    else:
        raise ReceiptError(f"{trial_label}.exception_info must be an object or null")

    return {
        "trial_name": _required_string(data, "trial_name", trial_label),
        "task_name": _required_string(data, "task_name", trial_label),
        "task_path": task_path,
        "task_checksum": _required_string(data, "task_checksum", trial_label),
        "agent_name": _required_string(agent_info, "name", f"{trial_label}.agent_info"),
        "agent_version": _required_string(
            agent_info,
            "version",
            f"{trial_label}.agent_info",
        ),
        "model_provider": model_provider,
        "model_name": model_name,
        "environment_type": environment_type,
        "reward": reward,
        "exception": exception,
        "run_record": _find_run_record(result_path.parent, job_dir),
    }


def create_receipt(
    job_dir: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    if not job_dir.is_dir():
        raise ReceiptError(f"Harbor job directory does not exist: {job_dir}")

    root = repository_root or Path(__file__).resolve().parents[1]
    repository_commit, worktree_clean = _repository_state(root)
    job = _load_json_object(job_dir / "result.json", "Harbor job result")

    stats = job.get("stats")
    if not isinstance(stats, dict):
        raise ReceiptError("Harbor job result.stats must be an object")

    trial_paths = sorted(job_dir.glob("*/result.json"))
    if not trial_paths:
        raise ReceiptError(f"No Harbor trial results found under: {job_dir}")
    trials = [_trial_receipt(path, job_dir) for path in trial_paths]

    for trial in trials:
        if (
            trial["agent_name"] == "pureharness"
            and trial["agent_version"] != repository_commit
        ):
            raise ReceiptError(
                f"PureHarness trial {trial['trial_name']!r} used "
                f"{trial['agent_version']}, but repository HEAD is "
                f"{repository_commit}"
            )

    mean_reward = _job_mean_reward(stats)

    return {
        "schema_version": SCHEMA_VERSION,
        "pureharness": {
            "repository_commit": repository_commit,
            "worktree_clean": worktree_clean,
        },
        "harbor": {
            "version": _harbor_version(),
            "job_id": _required_string(job, "id", "Harbor job result"),
        },
        "evaluation": {
            "started_at": _required_string(
                job,
                "started_at",
                "Harbor job result",
            ),
            "finished_at": job.get("finished_at"),
            "total_trials": _required_count(
                job,
                "n_total_trials",
                "Harbor job result",
            ),
            "completed_trials": _required_count(
                stats,
                "n_completed_trials",
                "Harbor job result.stats",
            ),
            "errored_trials": _required_count(
                stats,
                "n_errored_trials",
                "Harbor job result.stats",
            ),
            "mean_reward": mean_reward,
        },
        "trials": trials,
    }


def _output_is_inside_job(output: Path, job_dir: Path) -> bool:
    try:
        output.resolve().relative_to(job_dir.resolve())
    except ValueError:
        return False
    return True


def _serialize(receipt: dict[str, Any]) -> str:
    return json.dumps(receipt, indent=2, sort_keys=True) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reproducibility receipt from a Harbor job.",
    )
    parser.add_argument("harbor_job_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if (
            args.output is not None
            and _output_is_inside_job(args.output, args.harbor_job_dir)
        ):
            raise ReceiptError(
                "--output must be outside the Harbor job directory"
            )
        serialized = _serialize(create_receipt(args.harbor_job_dir))
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            args.output.write_text(serialized, encoding="utf-8")
    except (OSError, ReceiptError) as exc:
        print(f"benchmark_receipt: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
