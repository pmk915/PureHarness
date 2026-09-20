import json

from pathlib import Path

import pytest

from scripts import benchmark_receipt


HEAD = "86c67d0a35c784dee30cfaae70bee70d5d5431bd"
JOB_ID = "11111111-2222-3333-4444-555555555555"
TRIAL_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SECRET = "fixture-secret-that-must-not-appear"


@pytest.fixture(autouse=True)
def fixed_versions(monkeypatch):
    monkeypatch.setattr(
        benchmark_receipt,
        "_repository_state",
        lambda _root: (HEAD, True),
    )
    monkeypatch.setattr(
        benchmark_receipt,
        "_harbor_version",
        lambda: "0.23.0",
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _job_result(
    *,
    errored_trials: int = 0,
    mean_reward: float | int = 0.5,
    evals: dict | None = None,
) -> dict:
    if evals is None:
        evals = {
            "misleading-display-key": {
                "metrics": [{"mean": mean_reward}],
            }
        }
    return {
        "id": JOB_ID,
        "started_at": "2026-09-21T00:45:10Z",
        "updated_at": "2026-09-21T00:47:03Z",
        "finished_at": "2026-09-21T00:47:03Z",
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": errored_trials,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "evals": evals,
        },
    }


def _trial_result(
    *,
    agent_version: str = HEAD,
    reward: float | None = 0.75,
    exception_info: dict | None = None,
) -> dict:
    verifier_result = (
        None
        if reward is None
        else {
            "rewards": {
                "reward": reward,
                "unrelated_metric": 999,
            }
        }
    )
    return {
        "id": TRIAL_ID,
        "task_name": "structured-task-name",
        "trial_name": "opaque-trial-name",
        "trial_uri": "trial://opaque",
        "task_id": {"path": "structured/task/path"},
        "source": None,
        "task_checksum": "c" * 64,
        "config": {
            "task": {"path": "misleading/task/path"},
            "agent": {
                "model_name": "wrong-provider/wrong-model",
                "kwargs": {
                    "DEEPSEEK_API_KEY": SECRET,
                },
            },
            "environment": {
                "type": "docker",
                "kwargs": {
                    "SECRET_TOKEN": SECRET,
                },
            },
        },
        "agent_info": {
            "name": "pureharness",
            "version": agent_version,
            "model_info": {
                "provider": "deepseek",
                "name": "structured-model",
            },
        },
        "agent_result": None,
        "verifier_result": verifier_result,
        "exception_info": exception_info,
        "started_at": "2026-09-21T00:45:11Z",
        "finished_at": "2026-09-21T00:47:03Z",
    }


def _create_job(
    tmp_path: Path,
    *,
    trial: dict | None = None,
    errored_trials: int = 0,
    job_mean_reward: float | int = 0.5,
    job_evals: dict | None = None,
    include_run_record: bool = True,
) -> Path:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "opaque-trial-name"
    _write_json(
        job_dir / "result.json",
        _job_result(
            errored_trials=errored_trials,
            mean_reward=job_mean_reward,
            evals=job_evals,
        ),
    )
    _write_json(
        trial_dir / "result.json",
        trial or _trial_result(),
    )
    if include_run_record:
        _write_json(
            trial_dir / "agent" / "pureharness-run-record.json",
            {
                "schema_version": 1,
                "sensitive_fixture_value": SECRET,
            },
        )
    return job_dir


def test_successful_one_trial_receipt_uses_structured_fields(tmp_path):
    job_dir = _create_job(tmp_path)

    receipt = benchmark_receipt.create_receipt(
        job_dir,
        repository_root=tmp_path,
    )

    assert receipt["schema_version"] == 1
    assert receipt["pureharness"] == {
        "repository_commit": HEAD,
        "worktree_clean": True,
    }
    assert receipt["harbor"] == {
        "version": "0.23.0",
        "job_id": JOB_ID,
    }
    assert receipt["evaluation"] == {
        "started_at": "2026-09-21T00:45:10Z",
        "finished_at": "2026-09-21T00:47:03Z",
        "total_trials": 1,
        "completed_trials": 1,
        "errored_trials": 0,
        "mean_reward": 0.5,
    }
    assert receipt["trials"] == [
        {
            "trial_name": "opaque-trial-name",
            "task_name": "structured-task-name",
            "task_path": "structured/task/path",
            "task_checksum": "c" * 64,
            "agent_name": "pureharness",
            "agent_version": HEAD,
            "model_provider": "deepseek",
            "model_name": "structured-model",
            "environment_type": "docker",
            "reward": 0.75,
            "exception": None,
            "run_record": (
                "opaque-trial-name/agent/pureharness-run-record.json"
            ),
        }
    ]

    serialized = json.dumps(receipt)
    assert len(receipt["pureharness"]["repository_commit"]) == 40
    assert SECRET not in serialized
    assert "DEEPSEEK_API_KEY" not in serialized
    assert "wrong-provider" not in serialized
    assert "misleading-display-key" not in serialized


def test_stdout_and_output_file_are_deterministic(tmp_path, capsys):
    job_dir = _create_job(tmp_path)
    before = {
        path.relative_to(job_dir): path.read_bytes()
        for path in job_dir.rglob("*")
        if path.is_file()
    }

    assert benchmark_receipt.main([str(job_dir)]) == 0
    captured = capsys.readouterr()
    stdout_receipt = json.loads(captured.out)

    output = tmp_path / "receipt.json"
    assert (
        benchmark_receipt.main(
            [str(job_dir), "--output", str(output)]
        )
        == 0
    )

    assert captured.err == ""
    assert captured.out.endswith("\n")
    assert output.read_text(encoding="utf-8") == captured.out
    assert stdout_receipt["harbor"]["job_id"] == JOB_ID
    assert before == {
        path.relative_to(job_dir): path.read_bytes()
        for path in job_dir.rglob("*")
        if path.is_file()
    }


def test_repository_head_agent_version_mismatch_is_rejected(tmp_path):
    job_dir = _create_job(
        tmp_path,
        trial=_trial_result(agent_version="b" * 40),
    )

    with pytest.raises(
        benchmark_receipt.ReceiptError,
        match="used .* but repository HEAD is",
    ):
        benchmark_receipt.create_receipt(
            job_dir,
            repository_root=tmp_path,
        )


def test_errored_trial_is_represented_without_sensitive_details(tmp_path):
    job_dir = _create_job(
        tmp_path,
        trial=_trial_result(
            reward=None,
            exception_info={
                "exception_type": "NetworkConnectionError",
                "exception_message": f"request failed with {SECRET}",
                "exception_traceback": f"traceback containing {SECRET}",
                "occurred_at": "2026-09-21T00:46:00Z",
            },
        ),
        errored_trials=1,
        job_mean_reward=0.0,
        include_run_record=False,
    )

    receipt = benchmark_receipt.create_receipt(
        job_dir,
        repository_root=tmp_path,
    )
    trial = receipt["trials"][0]

    assert receipt["evaluation"]["errored_trials"] == 1
    assert receipt["evaluation"]["mean_reward"] == 0.0
    assert trial["reward"] is None
    assert trial["exception"] == {
        "type": "NetworkConnectionError",
        "occurred_at": "2026-09-21T00:46:00Z",
    }
    assert trial["run_record"] is None
    assert SECRET not in json.dumps(receipt)


@pytest.mark.parametrize(
    ("evals", "message"),
    [
        ({}, "exactly one eval entry"),
        (
            {
                "first": {"metrics": [{"mean": 0.0}]},
                "second": {"metrics": [{"mean": 1.0}]},
            },
            "exactly one eval entry",
        ),
        (
            {"only": {"metrics": []}},
            "metrics must be a non-empty list",
        ),
        (
            {"only": {"metrics": [{"mean": "0.0"}]}},
            "mean must be numeric",
        ),
    ],
)
def test_job_level_mean_requires_one_structured_numeric_metric(
    tmp_path,
    evals,
    message,
):
    job_dir = _create_job(tmp_path, job_evals=evals)

    with pytest.raises(benchmark_receipt.ReceiptError, match=message):
        benchmark_receipt.create_receipt(
            job_dir,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "Missing Harbor job result"),
        ("{not-json", "Invalid JSON in Harbor job result"),
        ("[]", "must contain a JSON object"),
    ],
)
def test_missing_or_invalid_job_result_fails_clearly(
    tmp_path,
    contents,
    message,
):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    if contents is not None:
        (job_dir / "result.json").write_text(contents, encoding="utf-8")

    with pytest.raises(benchmark_receipt.ReceiptError, match=message):
        benchmark_receipt.create_receipt(
            job_dir,
            repository_root=tmp_path,
        )


def test_invalid_trial_result_fails_clearly(tmp_path):
    job_dir = _create_job(tmp_path)
    (job_dir / "opaque-trial-name" / "result.json").write_text(
        "{not-json",
        encoding="utf-8",
    )

    with pytest.raises(
        benchmark_receipt.ReceiptError,
        match="Invalid JSON in Harbor trial result",
    ):
        benchmark_receipt.create_receipt(
            job_dir,
            repository_root=tmp_path,
        )


def test_output_inside_job_directory_is_rejected(tmp_path, capsys):
    job_dir = _create_job(tmp_path)
    output = job_dir / "receipt.json"

    assert (
        benchmark_receipt.main(
            [str(job_dir), "--output", str(output)]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert "must be outside the Harbor job directory" in captured.err
    assert not output.exists()
