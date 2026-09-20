import json
import sys

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from pureharness.agent import Agent
from pureharness.benchmark import (
    BENCHMARK_RESULT_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkIntegrityError,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSerializationError,
    BenchmarkTask,
    BenchmarkVerifierError,
    BenchmarkVerifierTimeout,
    TRUSTED_VERIFIER_PLACEHOLDER,
    default_benchmark_configs,
    load_benchmark_results,
    load_benchmark_tasks,
    summarize_results,
    write_benchmark_results,
)
from pureharness.context import (
    ContextBuilder,
    TokenBudgetContextBuilder,
)
from pureharness.execution import (
    CommandResult,
    ExecutionTimeoutError,
)
from pureharness.messages import Message, ToolCall, ToolResult
from pureharness.tool_result_projection import (
    DeterministicToolResultProjector,
    IdentityToolResultProjector,
)
from pureharness.tool_selection import (
    AllToolsSelector,
    StaticToolSelector,
)
from pureharness.trajectory_compaction import (
    DeterministicToolTrajectoryCompactor,
    IdentityTrajectoryCompactor,
)


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.contexts = []
        self.tool_names = []

    def generate(self, messages, tools):
        self.contexts.append(list(messages))
        self.tool_names.append(tuple(tool.name for tool in tools))
        return self.outputs.pop(0)


def _fixture_task(tmp_path: Path) -> BenchmarkTask:
    fixture = tmp_path / "fixture"
    fixture.mkdir(parents=True)
    (fixture / "value.txt").write_text(
        "broken",
        encoding="utf-8",
    )
    verifier = tmp_path / "trusted" / "verify.py"
    verifier.parent.mkdir()
    verifier.write_text(
        "from pathlib import Path\n"
        "value = Path('value.txt').read_text(encoding='utf-8')\n"
        "raise SystemExit(0 if value == 'fixed' else 1)\n",
        encoding="utf-8",
    )
    return BenchmarkTask(
        task_id="fixture-task",
        prompt="Fix value.txt.",
        fixture_path=fixture,
        verification_argv=(
            sys.executable,
            TRUSTED_VERIFIER_PLACEHOLDER,
        ),
        trusted_verifier_path=verifier,
        selective_tool_names=("read_file", "write_file"),
        declared_required_tools=("write_file",),
    )


def _config(
    config_id: str = "test-config",
    *,
    max_steps: int = 2,
) -> BenchmarkConfig:
    return replace(
        default_benchmark_configs(max_steps=max_steps)[0],
        config_id=config_id,
    )


def _passing_model():
    return ScriptedModel(
        [
            [
                ToolCall(
                    name="write_file",
                    arguments={
                        "path": "value.txt",
                        "content": "fixed",
                    },
                    call_id="write-1",
                )
            ],
            Message(role="assistant", content="done"),
        ]
    )


def test_benchmark_task_validates_static_metadata(tmp_path):
    task = _fixture_task(tmp_path)

    assert task.verification_argv[0] == sys.executable
    assert task.fixture_path.is_dir()
    assert task.trusted_verifier_path is not None
    assert task.trusted_verifier_path.is_file()

    with pytest.raises(ValueError, match="task_id"):
        BenchmarkTask(
            task_id="",
            prompt="prompt",
            fixture_path=tmp_path,
            verification_argv=("python",),
        )

    with pytest.raises(ValueError, match="duplicates"):
        BenchmarkTask(
            task_id="duplicate-tools",
            prompt="prompt",
            fixture_path=tmp_path,
            verification_argv=("python",),
            selective_tool_names=("read_file", "read_file"),
        )


def test_task_loader_validates_schema_and_loads_curated_tasks(tmp_path):
    tasks = load_benchmark_tasks(Path("benchmarks/tasks"))

    assert [task.task_id for task in tasks] == [
        "exposure_sensitive",
        "large_output",
        "long_horizon",
        "multi_file",
        "simple_fix",
    ]
    assert all(
        task.trusted_verifier_path is not None
        and task.trusted_verifier_path.is_file()
        and task.fixture_path
        not in task.trusted_verifier_path.parents
        for task in tasks
    )

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "id": "invalid",
                "prompt": "invalid",
                "verification_argv": ["python", "verify.py"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        BenchmarkSerializationError,
        match="Unsupported BenchmarkTask schema version",
    ):
        BenchmarkTask.from_directory(invalid)


def test_default_configs_are_explicit_and_stable():
    configs = default_benchmark_configs(
        history_token_budget=321,
        max_steps=7,
    )

    assert [config.config_id for config in configs] == [
        "raw_baseline",
        "budget_only",
        "context_engineered",
        "full_pureharness",
    ]
    assert all(config.max_steps == 7 for config in configs)
    assert [config.history_token_budget for config in configs] == [
        None,
        321,
        321,
        321,
    ]


def test_config_validation_rejects_invalid_combinations():
    with pytest.raises(ValueError, match="FullHistory"):
        BenchmarkConfig(
            config_id="invalid",
            context_strategy="full_history",
            history_token_budget=100,
            tool_result_projection="identity",
            trajectory_compaction="identity",
            include_task_state=False,
            tool_selection="all",
        )

    with pytest.raises(ValueError, match="positive integer"):
        replace(
            default_benchmark_configs()[1],
            history_token_budget=0,
        )


def test_representative_configs_use_expected_runtime_components(tmp_path):
    task = _fixture_task(tmp_path)
    raw, budget, engineered, full = default_benchmark_configs()

    raw_builder = raw.create_context_builder()
    assert type(raw_builder) is ContextBuilder
    assert isinstance(
        raw_builder.tool_result_projector,
        IdentityToolResultProjector,
    )
    assert isinstance(
        raw_builder.trajectory_compactor,
        IdentityTrajectoryCompactor,
    )
    assert raw.include_task_state is False
    assert isinstance(raw.create_tool_selector(task), AllToolsSelector)

    budget_builder = budget.create_context_builder()
    assert isinstance(budget_builder, TokenBudgetContextBuilder)
    assert isinstance(
        budget_builder.tool_result_projector,
        IdentityToolResultProjector,
    )
    assert isinstance(
        budget_builder.trajectory_compactor,
        IdentityTrajectoryCompactor,
    )
    assert budget.include_task_state is False

    engineered_builder = engineered.create_context_builder()
    assert isinstance(engineered_builder, TokenBudgetContextBuilder)
    assert isinstance(
        engineered_builder.tool_result_projector,
        DeterministicToolResultProjector,
    )
    assert isinstance(
        engineered_builder.trajectory_compactor,
        DeterministicToolTrajectoryCompactor,
    )
    assert engineered.include_task_state is True
    assert isinstance(
        engineered.create_tool_selector(task),
        AllToolsSelector,
    )

    full_builder = full.create_context_builder()
    assert isinstance(full_builder, TokenBudgetContextBuilder)
    selector = full.create_tool_selector(task)
    assert isinstance(selector, StaticToolSelector)
    assert selector.tool_names == frozenset(task.selective_tool_names)


def test_completed_agent_can_fail_external_oracle(tmp_path):
    task = _fixture_task(tmp_path)
    model = ScriptedModel(
        [Message(role="assistant", content="finished")]
    )
    runner = BenchmarkRunner(
        lambda task, config: model,
        run_id_factory=lambda: "completed-but-wrong",
    )

    result = runner.run_case(task, _config())

    assert result.agent_end_reason == "completed"
    assert result.run_record.end_reason == "completed"
    assert result.task_success is False
    assert result.verification_exit_code == 1


def test_workspace_verifier_edit_cannot_forge_success(tmp_path):
    task = _fixture_task(tmp_path)
    assert task.trusted_verifier_path is not None
    trusted_before = task.trusted_verifier_path.read_bytes()
    model = ScriptedModel(
        [
            [
                ToolCall(
                    name="write_file",
                    arguments={
                        "path": "verify.py",
                        "content": "raise SystemExit(0)\n",
                    },
                    call_id="forge-1",
                )
            ],
            Message(role="assistant", content="done"),
        ]
    )

    result = BenchmarkRunner(
        lambda task, config: model
    ).run_case(task, _config())

    assert result.agent_end_reason == "completed"
    assert result.task_success is False
    assert task.trusted_verifier_path.read_bytes() == trusted_before


def test_trusted_verifier_mutation_is_integrity_failure(tmp_path):
    task = _fixture_task(tmp_path)
    assert task.trusted_verifier_path is not None
    trusted_path = task.trusted_verifier_path
    trusted_before = trusted_path.read_bytes()

    class MutatingModel:
        def generate(self, messages, tools):
            trusted_path.write_text(
                "raise SystemExit(0)\n",
                encoding="utf-8",
            )
            return Message(role="assistant", content="done")

    try:
        with pytest.raises(
            BenchmarkIntegrityError,
            match="Trusted verifier changed",
        ):
            BenchmarkRunner(
                lambda task, config: MutatingModel()
            ).run_case(task, _config())
    finally:
        trusted_path.write_bytes(trusted_before)


def test_scripted_model_can_pass_external_oracle(tmp_path):
    task = _fixture_task(tmp_path)
    runner = BenchmarkRunner(
        lambda task, config: _passing_model(),
        run_id_factory=lambda: "passing-run",
    )

    result = runner.run_case(task, _config())

    assert result.task_success is True
    assert result.verification_exit_code == 0
    assert result.agent_end_reason == "completed"
    assert result.run_record.tool_call_count == 1
    assert result.run_record.tool_execution_count == 1
    assert (
        task.fixture_path / "value.txt"
    ).read_text(encoding="utf-8") == "broken"


def test_failed_agent_run_still_runs_oracle_and_returns_record(tmp_path):
    task = _fixture_task(tmp_path)
    runner = BenchmarkRunner(
        lambda task, config: _passing_model(),
        run_id_factory=lambda: "max-step-run",
    )

    result = runner.run_case(
        task,
        _config(max_steps=1),
    )

    assert result.agent_end_reason == "max_steps_exceeded"
    assert result.agent_error_type == "RuntimeError"
    assert result.task_success is True
    assert result.run_record.end_reason == "max_steps_exceeded"


def test_task_state_disabled_baseline_sends_only_trajectory(tmp_path):
    task = _fixture_task(tmp_path)
    model = ScriptedModel(
        [Message(role="assistant", content="done")]
    )
    runner = BenchmarkRunner(
        lambda task, config: model,
        run_id_factory=lambda: "raw-baseline-run",
    )

    result = runner.run_case(
        task,
        default_benchmark_configs()[0],
    )

    assert [
        item.content
        for item in model.contexts[0]
        if isinstance(item, Message)
    ] == [task.prompt]
    assert result.run_record.sum_estimated_task_state_tokens == 0


def test_large_output_fixture_records_tool_result_projection():
    task = BenchmarkTask.from_directory(
        Path("benchmarks/tasks/large_output")
    )
    model = ScriptedModel(
        [
            [
                ToolCall(
                    name="run_command",
                    arguments={"argv": ["python3", "diagnose.py"]},
                    call_id="verify-1",
                )
            ],
            Message(role="assistant", content="done"),
        ]
    )
    runner = BenchmarkRunner(lambda task, config: model)

    result = runner.run_case(
        task,
        default_benchmark_configs()[2],
    )

    assert result.task_success is False
    assert result.run_record.tool_result_compaction_count == 1
    assert (
        result.run_record.model_invocations[
            1
        ].compacted_tool_results
        == 1
    )


def test_static_exposure_reduces_recorded_schema_cost(tmp_path):
    task = _fixture_task(tmp_path)
    raw, _, _, full = default_benchmark_configs()
    models = []

    def model_factory(task, config):
        model = ScriptedModel(
            [Message(role="assistant", content="done")]
        )
        models.append(model)
        return model

    runner = BenchmarkRunner(model_factory)
    all_result = runner.run_case(task, raw)
    static_result = runner.run_case(task, full)
    all_invocation = all_result.run_record.model_invocations[0]
    static_invocation = static_result.run_record.model_invocations[0]

    assert all_invocation.selector_strategy == "AllTools"
    assert static_invocation.selector_strategy == "StaticNames"
    assert static_invocation.exposed_tool_count == 2
    assert (
        static_invocation.exposed_tool_count
        < all_invocation.exposed_tool_count
    )
    assert (
        static_invocation.estimated_tool_schema_tokens
        < all_invocation.estimated_tool_schema_tokens
    )
    assert (
        static_result.run_record.sum_estimated_task_state_tokens
        > 0
    )
    assert models[1].tool_names[0] == task.selective_tool_names


def test_default_agent_still_injects_task_state():
    model = ScriptedModel(
        [Message(role="assistant", content="done")]
    )

    Agent(model=model).run("default behavior")

    assert isinstance(model.contexts[0][0], Message)
    assert (
        "[PureHarness Derived Task State]"
        in model.contexts[0][0].content
    )


def test_suite_uses_fresh_workspace_for_every_config(tmp_path):
    task = _fixture_task(tmp_path)
    observations = []

    def model_factory(task, config):
        if config.config_id == "mutating":
            return _passing_model()

        class ReadOriginalModel:
            call_count = 0

            def generate(self, messages, tools):
                self.call_count += 1
                if self.call_count == 1:
                    return [
                        ToolCall(
                            name="read_file",
                            arguments={"path": "value.txt"},
                            call_id="read-1",
                        )
                    ]
                result = next(
                    item
                    for item in reversed(messages)
                    if isinstance(item, ToolResult)
                )
                observations.append(result.content)
                return Message(role="assistant", content="done")

        return ReadOriginalModel()

    configs = (
        _config("mutating"),
        _config("observing"),
    )
    suite = BenchmarkRunner(
        model_factory,
        run_id_factory=iter(["run-a", "run-b"]).__next__,
    ).run_suite((task,), configs)

    assert [
        result.config_id for result in suite.results
    ] == ["mutating", "observing"]
    assert observations == ["broken"]
    assert (
        task.fixture_path / "value.txt"
    ).read_text(encoding="utf-8") == "broken"


def test_suite_iteration_is_task_then_config_order(tmp_path):
    first = _fixture_task(tmp_path / "first")
    second = _fixture_task(tmp_path / "second")
    second = replace(second, task_id="second-task")
    configs = (_config("a"), _config("b"))

    suite = BenchmarkRunner(
        lambda task, config: ScriptedModel(
            [Message(role="assistant", content="done")]
        )
    ).run_suite((first, second), configs)

    assert [
        (result.task_id, result.config_id)
        for result in suite.results
    ] == [
        ("fixture-task", "a"),
        ("fixture-task", "b"),
        ("second-task", "a"),
        ("second-task", "b"),
    ]


@pytest.mark.parametrize("kind", ["task", "config"])
def test_suite_rejects_duplicate_and_empty_ids(tmp_path, kind):
    task = _fixture_task(tmp_path)
    runner = BenchmarkRunner(
        lambda task, config: ScriptedModel(
            [Message(role="assistant", content="done")]
        )
    )

    if kind == "task":
        with pytest.raises(BenchmarkError, match="Duplicate.*task"):
            runner.run_suite((task, task), (_config(),))
        with pytest.raises(BenchmarkError, match="at least one task"):
            runner.run_suite((), (_config(),))
    else:
        config = _config()
        with pytest.raises(BenchmarkError, match="Duplicate.*config"):
            runner.run_suite((task,), (config, config))
        with pytest.raises(BenchmarkError, match="at least one config"):
            runner.run_suite((task,), ())


def test_result_serialization_round_trip_and_schema_rejection(tmp_path):
    task = _fixture_task(tmp_path)
    result = BenchmarkRunner(
        lambda task, config: _passing_model(),
        run_id_factory=lambda: "serialization-run",
    ).run_case(task, _config())

    restored = BenchmarkResult.from_json(result.to_json())

    assert restored == result
    assert restored.to_dict()["schema_version"] == (
        BENCHMARK_RESULT_SCHEMA_VERSION
    )

    data = result.to_dict()
    data["schema_version"] = 999
    with pytest.raises(
        BenchmarkSerializationError,
        match="Unsupported BenchmarkResult schema version",
    ):
        BenchmarkResult.from_dict(data)


def test_jsonl_writer_overwrites_appends_and_loads(tmp_path):
    task = _fixture_task(tmp_path)
    runner = BenchmarkRunner(
        lambda task, config: _passing_model(),
        run_id_factory=iter(["jsonl-a", "jsonl-b"]).__next__,
    )
    first = runner.run_case(task, _config("a"))
    second = runner.run_case(task, _config("b"))
    path = tmp_path / "results" / "run.jsonl"

    write_benchmark_results(path, (first,))
    write_benchmark_results(path, (second,), append=True)

    assert load_benchmark_results(path) == (first, second)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_summary_uses_run_record_metrics_without_composite_score(tmp_path):
    task = _fixture_task(tmp_path)
    ids = iter(["summary-a", "summary-b"])
    runner = BenchmarkRunner(
        lambda task, config: _passing_model(),
        run_id_factory=ids.__next__,
    )
    results = (
        runner.run_case(task, _config("same")),
        runner.run_case(task, _config("same")),
    )

    summary = summarize_results(results)[0]

    assert summary.config_id == "same"
    assert summary.case_count == 2
    assert summary.success_count == 2
    assert summary.success_rate == 1.0
    assert summary.total_model_calls == 4
    assert summary.mean_model_calls == 2.0
    assert summary.total_steps == 4
    assert summary.mean_steps == 2.0
    assert summary.total_tool_calls == 2
    assert summary.total_tool_executions == 2
    assert summary.sum_estimated_history_tokens == sum(
        result.run_record.sum_estimated_history_tokens
        for result in results
    )
    assert not hasattr(summary, "score")


def test_verifier_nonzero_is_task_failure_but_start_error_is_infrastructure(
    tmp_path,
):
    task = _fixture_task(tmp_path)
    runner = BenchmarkRunner(
        lambda task, config: ScriptedModel(
            [Message(role="assistant", content="done")]
        )
    )

    result = runner.run_case(task, _config())
    assert result.task_success is False

    missing_verifier = replace(
        task,
        verification_argv=(
            "definitely-missing-pureharness-command",
            TRUSTED_VERIFIER_PLACEHOLDER,
        ),
    )
    with pytest.raises(
        BenchmarkVerifierError,
        match="verification could not be executed",
    ):
        runner.run_case(missing_verifier, _config())


def test_verifier_uses_independent_backend_and_configured_timeout(tmp_path):
    task = _fixture_task(tmp_path)

    class RecordingBackend:
        def __init__(self):
            self.calls = []

        def execute(self, argv, *, cwd, timeout):
            self.calls.append((tuple(argv), cwd, timeout))
            return CommandResult(
                exit_code=0,
                stdout="verified",
                stderr="",
            )

    backend = RecordingBackend()
    runner = BenchmarkRunner(
        lambda task, config: ScriptedModel(
            [Message(role="assistant", content="done")]
        ),
        verification_backend=backend,
        verification_timeout_seconds=4.5,
    )

    result = runner.run_case(task, _config())

    assert result.task_success is True
    assert len(backend.calls) == 1
    assert backend.calls[0][0][0] == sys.executable
    assert backend.calls[0][0][1] != (
        str(task.trusted_verifier_path)
    )
    assert Path(backend.calls[0][0][1]).name == "verify.py"
    assert backend.calls[0][2] == 4.5
    assert backend.calls[0][1] != task.fixture_path


def test_verifier_timeout_has_distinct_infrastructure_error(tmp_path):
    task = _fixture_task(tmp_path)

    class TimeoutBackend:
        def execute(self, argv, *, cwd, timeout):
            raise ExecutionTimeoutError("timed out")

    runner = BenchmarkRunner(
        lambda task, config: ScriptedModel(
            [Message(role="assistant", content="done")]
        ),
        verification_backend=TimeoutBackend(),
        verification_timeout_seconds=0.25,
    )

    with pytest.raises(
        BenchmarkVerifierTimeout,
        match="timed out after 0.25 seconds",
    ):
        runner.run_case(task, _config())


def test_trusted_verifier_must_be_outside_fixture(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    verifier = fixture / "verify.py"
    verifier.write_text("raise SystemExit(0)\n", encoding="utf-8")
    task = BenchmarkTask(
        task_id="unsafe-verifier",
        prompt="do nothing",
        fixture_path=fixture,
        verification_argv=(
            sys.executable,
            TRUSTED_VERIFIER_PLACEHOLDER,
        ),
        trusted_verifier_path=verifier,
    )

    with pytest.raises(
        BenchmarkError,
        match="outside the Agent workspace",
    ):
        BenchmarkRunner(
            lambda task, config: ScriptedModel(
                [Message(role="assistant", content="done")]
            )
        ).run_case(task, _config())


def test_verification_output_is_bounded(tmp_path):
    task = BenchmarkTask.from_directory(
        Path("benchmarks/tasks/large_output")
    )
    runner = BenchmarkRunner(
        lambda task, config: ScriptedModel(
            [Message(role="assistant", content="done")]
        ),
        verification_preview_chars=200,
    )

    result = runner.run_case(task, _config())

    assert result.task_success is False
    assert len(result.verification_stdout_preview) <= 200
    assert "verification output omitted" in (
        result.verification_stdout_preview
    )


def test_benchmark_core_has_no_provider_or_network_dependency():
    source = Path("src/pureharness/benchmark.py").read_text(
        encoding="utf-8"
    )

    assert "deepseek" not in source.lower()
    assert "openai" not in source.lower()
    assert "socket" not in source.lower()


def test_finalized_benchmark_values_are_frozen(tmp_path):
    task = _fixture_task(tmp_path)
    result = BenchmarkRunner(
        lambda task, config: _passing_model()
    ).run_case(task, _config())

    with pytest.raises(FrozenInstanceError):
        result.task_success = False
