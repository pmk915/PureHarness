import json

import pytest

from pureharness.evaluation import (
    EvaluationReport,
    EvaluationReportBuilder,
    FailureType,
    Trajectory,
)


def _trajectory(
    *,
    run_id="report-run",
    completed=True,
    total_steps=2,
    tool_call_count=2,
    failed_tool_call_count=0,
):
    return Trajectory(
        run_id=run_id,
        completed=completed,
        total_steps=total_steps,
        tool_call_count=tool_call_count,
        failed_tool_call_count=failed_tool_call_count,
    )


def test_successful_report_creation():
    report = EvaluationReportBuilder().build(_trajectory())

    assert isinstance(report, EvaluationReport)
    assert report.run_id == "report-run"
    assert report.metrics.task_success_score == 1.0
    assert report.metrics.step_efficiency_score == 0.5
    assert report.metrics.tool_reliability_score == 1.0
    assert report.metrics.recovery_score is None
    assert report.diagnosis.failure_type is FailureType.NONE
    assert report.overall_score == pytest.approx(5 / 6)


def test_failure_report_creation():
    report = EvaluationReportBuilder().build(
        _trajectory(
            completed=False,
            tool_call_count=4,
            failed_tool_call_count=3,
        )
    )

    assert report.metrics.task_success_score == 0.0
    assert report.metrics.tool_reliability_score == 0.25
    assert report.diagnosis.failure_type is FailureType.TOOL_FAILURE
    assert report.diagnosis.confidence == 0.75
    assert report.overall_score == pytest.approx(0.25)


def test_report_dictionary_is_json_serializable():
    report = EvaluationReportBuilder().build(
        _trajectory(
            run_id="json-run",
            completed=False,
            total_steps=0,
            tool_call_count=0,
        )
    )

    serialized = json.dumps(
        report.to_dict(),
        sort_keys=True,
        allow_nan=False,
    )

    assert json.loads(serialized) == report.to_dict()
    assert report.to_dict()["run_id"] == "json-run"
    assert report.to_dict()["diagnosis"]["failure_type"] == "unknown"


def test_markdown_generation_is_simple_and_deterministic():
    report = EvaluationReportBuilder().build(
        _trajectory(
            run_id="markdown-run",
            completed=True,
            total_steps=1,
            tool_call_count=0,
        )
    )

    assert report.to_markdown() == (
        "# PureHarness Evaluation Report\n"
        "\n"
        "## Run\n"
        "\n"
        "- Run ID: `markdown-run`\n"
        "- Overall score: 1.000000\n"
        "\n"
        "## Metrics\n"
        "\n"
        "- Task success: 1.000000\n"
        "- Step efficiency: 1.000000\n"
        "- Tool reliability: 1.000000\n"
        "- Recovery: not available\n"
        "\n"
        "## Diagnosis\n"
        "\n"
        "- Failure type: `none`\n"
        "- Confidence: 1.000000\n"
        "- Reason: Trajectory completed successfully.\n"
        "\n"
        "### Signals\n"
        "\n"
        "- completed: true"
    )
