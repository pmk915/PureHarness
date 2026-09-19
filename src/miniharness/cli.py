import argparse
import os
import sys

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from miniharness.approval import ApprovalHandler
from miniharness.agent import Agent
from miniharness.benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkTask,
    default_benchmark_configs,
    load_benchmark_tasks,
)
from miniharness.coding_tools import create_coding_tools
from miniharness.deepseek_model import DeepSeekModel
from miniharness.events import AgentEvent
from miniharness.experiment import (
    ExperimentRunner,
    summarize_experiment,
    write_experiment_results,
)
from miniharness.model import Model, ModelError
from miniharness.observability import (
    JsonlEventRenderer,
    dumps_wire,
    run_record_to_wire,
    session_summaries_to_wire,
)
from miniharness.run_record import (
    RunRecord,
    RunRecordSerializationError,
)
from miniharness.session import Session
from miniharness.session_store import (
    DurableSession,
    DurableSessionStore,
    JsonlDurableSessionStore,
    SessionStoreError,
)
from miniharness.terminal_approval import TerminalApprovalHandler
from miniharness.tool_executor import ToolExecutor
from miniharness.tools import ToolRegistry


DEFAULT_MODEL = "deepseek-v4-flash"
OutputFunction = Callable[[str], None]
InputFunction = Callable[[str], str]
ModelFactory = Callable[[str], Model]


class CLIError(RuntimeError):
    """Raised when CLI setup or local evidence handling fails."""


class PlainTerminalRenderer:
    """Render a compact event stream without controlling Agent execution."""

    def __init__(self, output: OutputFunction) -> None:
        self.output = output

    def __call__(self, event: AgentEvent) -> None:
        data = event.data
        if event.type == "model_started":
            self.output(
                "[model] request "
                f"(tools={data.get('exposed_tool_count', 0)})"
            )
        elif event.type == "tool_started":
            self.output(f"[tool] {data['name']}")
        elif event.type == "tool_completed":
            status = "error" if data["is_error"] else "ok"
            self.output(f"[tool] {data['name']} ({status})")
        elif event.type == "context_build_failed":
            self.output("[context] build failed")
        elif event.type == "model_failed":
            self.output("[model] request failed")
        elif event.type == "agent_failed":
            self.output(f"[agent] failed: {data['reason']}")
        elif event.type == "agent_interrupted":
            self.output("[agent] interrupted")
        elif event.type == "approval_requested":
            self.output(f"[approval] requested: {data['name']}")
        elif event.type == "approval_granted":
            self.output(f"[approval] granted: {data['name']}")
        elif event.type == "approval_denied":
            self.output(f"[approval] denied: {data['name']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniharness",
        description=(
            "A small, transparent CLI harness for long-horizon, "
            "tool-using agents."
        ),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace for interactive mode (default: current directory).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"DeepSeek model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="Write one RunRecord JSON file per interactive turn.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run one task without entering interactive mode.",
    )
    run_parser.add_argument("prompt")
    run_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
    )
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument(
        "--record",
        type=Path,
        help="Write the finalized RunRecord as JSON.",
    )
    run_parser.add_argument(
        "--output",
        dest="output_mode",
        choices=("human", "jsonl"),
        default="human",
        help="Render human text or a versioned JSONL event stream.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a serialized RunRecord JSON file.",
    )
    inspect_parser.add_argument("record", type=Path)
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write the versioned RunRecord JSON document.",
    )

    sessions_parser = subparsers.add_parser(
        "sessions",
        help="List saved interactive sessions.",
    )
    sessions_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write a versioned JSON session summary.",
    )

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a saved interactive session.",
    )
    resume_parser.add_argument("session_id")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the controlled API-backed benchmark.",
    )
    benchmark_parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("benchmarks/tasks"),
    )
    benchmark_parser.add_argument(
        "--task",
        action="append",
        dest="task_ids",
    )
    benchmark_parser.add_argument(
        "--config",
        action="append",
        dest="config_ids",
    )
    benchmark_parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
    )
    benchmark_parser.add_argument("--model", default=DEFAULT_MODEL)
    benchmark_parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/experiment.jsonl"),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: ModelFactory | None = None,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    error_fn: OutputFunction | None = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    factory = model_factory or _default_model_factory
    error_output = (
        error_fn
        if error_fn is not None
        else (_stderr_output if output_fn is print else output_fn)
    )
    jsonl_mode = (
        arguments.command == "run"
        and arguments.output_mode == "jsonl"
    )

    try:
        if arguments.command == "run":
            return _run_once(arguments, factory, output_fn)
        if arguments.command == "inspect":
            return _inspect_record(
                arguments.record,
                output_fn,
                as_json=arguments.json_output,
            )
        if arguments.command == "benchmark":
            return _run_benchmark(arguments, factory, output_fn)
        durable_store = _create_durable_store()
        if arguments.command == "sessions":
            return _list_sessions(
                durable_store,
                output_fn,
                as_json=arguments.json_output,
            )
        if arguments.command == "resume":
            state = _load_durable_session(
                durable_store,
                arguments.session_id,
            )
            return _run_interactive(
                workspace=state.workspace,
                model_name=state.model,
                record_dir=arguments.record_dir,
                model_factory=factory,
                input_fn=input_fn,
                output_fn=output_fn,
                durable_store=durable_store,
                durable_state=state,
            )
        return _run_interactive(
            workspace=arguments.workspace,
            model_name=arguments.model,
            record_dir=arguments.record_dir,
            model_factory=factory,
            input_fn=input_fn,
            output_fn=output_fn,
            durable_store=durable_store,
        )
    except KeyboardInterrupt:
        if not jsonl_mode:
            output_fn("Interrupted.")
        return 130
    except CLIError as exc:
        error_output(f"Error: {exc}")
        return 2
    except Exception as exc:
        error_output(f"Error: {type(exc).__name__}: {exc}")
        return 1


def _run_interactive(
    *,
    workspace: Path,
    model_name: str,
    record_dir: Path | None,
    model_factory: ModelFactory,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    durable_store: DurableSessionStore,
    durable_state: DurableSession | None = None,
) -> int:
    resumed = durable_state is not None
    if durable_state is None:
        resolved_workspace = _resolve_workspace(workspace)
        now = _utc_now()
        durable_state = DurableSession(
            session_id=str(uuid4()),
            created_at=now,
            updated_at=now,
            workspace=resolved_workspace,
            model=model_name,
            session=Session(),
            run_records=[],
        )
        _save_durable_session(durable_store, durable_state)
    else:
        resolved_workspace = _resolve_session_workspace(
            durable_state.workspace
        )
        model_name = durable_state.model

    session_id = durable_state.session_id
    agent: Agent | None = None
    output_fn("MiniHarness")
    if resumed:
        output_fn(f"Resumed session {session_id}")
    output_fn(f"Workspace: {resolved_workspace}")
    output_fn(f"Model: {model_name}")
    output_fn("Type /help for commands.")

    while True:
        try:
            value = input_fn("> ")
        except EOFError:
            _save_durable_session(durable_store, durable_state)
            output_fn("Goodbye.")
            return 0
        except KeyboardInterrupt:
            output_fn("Interrupted. Use /exit to leave.")
            continue

        prompt = value.strip()
        if not prompt:
            continue
        if prompt == "/exit":
            _save_durable_session(durable_store, durable_state)
            output_fn("Goodbye.")
            return 0
        if prompt == "/help":
            output_fn(
                "Commands: /help, /status, /exit. "
                "Any other text starts an Agent run."
            )
            continue
        if prompt == "/status":
            _render_status(
                agent,
                session_id=session_id,
                workspace=resolved_workspace,
                model_name=model_name,
                run_records=durable_state.run_records,
                output_fn=output_fn,
            )
            continue
        if prompt.startswith("/"):
            output_fn(f"Unknown command: {prompt}")
            continue

        if agent is None:
            try:
                agent = _create_agent(
                    resolved_workspace,
                    model_factory(model_name),
                    session_id=session_id,
                    output_fn=output_fn,
                    session=durable_state.session,
                    approval_handler=TerminalApprovalHandler(
                        input_fn=input_fn,
                        output_fn=output_fn,
                    ),
                )
            except CLIError as exc:
                output_fn(f"Error: {exc}")
                continue
            except Exception as exc:
                output_fn(
                    f"Run failed: {type(exc).__name__}: {exc}"
                )
                continue

        try:
            response = agent.run(prompt)
        except KeyboardInterrupt:
            output_fn("Run interrupted.")
            continue
        except Exception as exc:
            output_fn(f"Run failed: {type(exc).__name__}: {exc}")
        else:
            output_fn(response)
        finally:
            record = agent.last_run_record
            if record is not None:
                durable_state.session = agent.session
                durable_state.run_records.append(record)
                durable_state.updated_at = _utc_now()
                _save_durable_session(
                    durable_store,
                    durable_state,
                )
            if record_dir is not None and record is not None:
                destination = (
                    record_dir
                    / f"{record.run_id}.json"
                )
                _write_run_record(
                    destination,
                    record,
                )
                output_fn(f"Run record: {destination}")


def _create_durable_store() -> DurableSessionStore:
    configured_home = os.environ.get("MINIHARNESS_HOME")
    if configured_home is None:
        root = Path.home() / ".miniharness"
    elif not configured_home:
        raise CLIError("MINIHARNESS_HOME must not be empty")
    else:
        root = Path(configured_home).expanduser()

    try:
        root = root.resolve()
    except OSError as exc:
        raise CLIError(
            f"Could not resolve MiniHarness home: {root}"
        ) from exc
    return JsonlDurableSessionStore(root / "sessions")


def _load_durable_session(
    store: DurableSessionStore,
    session_id: str,
) -> DurableSession:
    try:
        return store.load(session_id)
    except SessionStoreError as exc:
        raise CLIError(str(exc)) from exc


def _save_durable_session(
    store: DurableSessionStore,
    state: DurableSession,
) -> None:
    try:
        store.save(state)
    except SessionStoreError as exc:
        raise CLIError(str(exc)) from exc


def _list_sessions(
    store: DurableSessionStore,
    output_fn: OutputFunction,
    *,
    as_json: bool = False,
) -> int:
    try:
        summaries = store.list_sessions()
    except SessionStoreError as exc:
        raise CLIError(str(exc)) from exc

    if as_json:
        output_fn(dumps_wire(session_summaries_to_wire(summaries)))
        return 0

    if not summaries:
        output_fn("No saved sessions.")
        return 0

    output_fn("SESSION ID\tUPDATED (UTC)\tRUNS\tMODEL\tWORKSPACE")
    for summary in summaries:
        updated = summary.updated_at.astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        output_fn(
            f"{summary.session_id}\t{updated}\t"
            f"{summary.run_count}\t{summary.model}\t"
            f"{summary.workspace}"
        )
    return 0


def _resolve_session_workspace(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CLIError(
            f"session workspace no longer exists: {path}"
        ) from exc
    if not resolved.is_dir():
        raise CLIError(
            f"session workspace no longer exists: {path}"
        )
    return resolved


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_once(
    arguments: argparse.Namespace,
    model_factory: ModelFactory,
    output_fn: OutputFunction,
) -> int:
    workspace = _resolve_workspace(arguments.workspace)
    jsonl_mode = arguments.output_mode == "jsonl"
    renderer = (
        JsonlEventRenderer(output_fn)
        if jsonl_mode
        else PlainTerminalRenderer(output_fn)
    )
    agent = _create_agent(
        workspace,
        model_factory(arguments.model),
        session_id=str(uuid4()),
        output_fn=output_fn,
        event_listener=renderer,
    )
    exit_code = 0
    try:
        response = agent.run(arguments.prompt)
    except Exception as exc:
        if not jsonl_mode:
            output_fn(f"Run failed: {type(exc).__name__}: {exc}")
        exit_code = 1
    else:
        if not jsonl_mode:
            output_fn(response)

    if jsonl_mode and agent.listener_errors:
        first_error = agent.listener_errors[0]
        raise CLIError(
            "JSONL event output failed: "
            f"{type(first_error).__name__}: {first_error}"
        )

    if arguments.record is not None:
        if agent.last_run_record is None:
            raise CLIError("Agent produced no RunRecord to save")
        _write_run_record(arguments.record, agent.last_run_record)
        if not jsonl_mode:
            output_fn(f"Run record: {arguments.record}")
    return exit_code


def _inspect_record(
    path: Path,
    output_fn: OutputFunction,
    *,
    as_json: bool = False,
) -> int:
    try:
        value = path.read_text(encoding="utf-8")
        record = RunRecord.from_json(value)
    except OSError as exc:
        raise CLIError(f"Could not read RunRecord: {path}") from exc
    except RunRecordSerializationError as exc:
        raise CLIError(f"Invalid RunRecord: {exc}") from exc

    if as_json:
        output_fn(dumps_wire(run_record_to_wire(record)))
        return 0

    output_fn(f"Run ID: {record.run_id}")
    output_fn(f"Session ID: {record.session_id or '(none)'}")
    output_fn(f"End reason: {record.end_reason}")
    output_fn(f"Steps: {record.step_count}")
    output_fn(f"Model calls: {record.model_call_count}")
    output_fn(f"Tool calls: {record.tool_call_count}")
    output_fn(f"Tool executions: {record.tool_execution_count}")
    output_fn(f"Tool result errors: {record.tool_result_error_count}")
    output_fn(f"Approvals: {len(record.trace.approvals)}")
    for approval in record.trace.approvals:
        output_fn(
            "Approval: "
            f"step={approval.step} "
            f"tool={approval.tool_name} "
            f"call_id={approval.call_id or '(none)'} "
            f"decision={approval.decision.value}"
        )
    output_fn(
        "Trajectory compactions: "
        f"{record.trajectory_compaction_count}"
    )
    output_fn(
        "ToolResult compactions: "
        f"{record.tool_result_compaction_count}"
    )
    output_fn(
        "Estimated history tokens: "
        f"{record.sum_estimated_history_tokens}"
    )
    output_fn(
        "Estimated TaskState tokens: "
        f"{record.sum_estimated_task_state_tokens}"
    )
    output_fn(
        "Estimated tool-schema tokens: "
        f"{record.sum_estimated_tool_schema_tokens}"
    )
    return 0


def _run_benchmark(
    arguments: argparse.Namespace,
    model_factory: ModelFactory,
    output_fn: OutputFunction,
) -> int:
    tasks = _select_tasks(
        load_benchmark_tasks(arguments.tasks_root),
        arguments.task_ids,
    )
    configs = _select_configs(
        default_benchmark_configs(),
        arguments.config_ids,
    )
    model_name = arguments.model
    runner = ExperimentRunner(
        BenchmarkRunner(
            lambda task, config: model_factory(model_name)
        ),
        model_id=model_name,
    )
    results = runner.run(
        tasks,
        configs,
        repetitions=arguments.repetitions,
    )
    write_experiment_results(arguments.output, results)
    for summary in summarize_experiment(results):
        output_fn(
            f"{summary.config_id}: "
            f"{summary.success_count}/{summary.run_count} successful; "
            f"success_rate={summary.success_rate:.3f}; "
            f"avg_steps={summary.average_steps:.2f}; "
            f"avg_tool_calls={summary.average_tool_calls:.2f}; "
            f"avg_duration={summary.average_duration_seconds:.2f}s"
        )
    output_fn(
        f"Wrote {len(results)} experiment result(s) "
        f"to {arguments.output}"
    )
    return 0


def _create_agent(
    workspace: Path,
    model: Model,
    *,
    session_id: str,
    output_fn: OutputFunction,
    session: Session | None = None,
    approval_handler: ApprovalHandler | None = None,
    event_listener: Callable[[AgentEvent], None] | None = None,
) -> Agent:
    registry = ToolRegistry()
    for tool in create_coding_tools(workspace):
        registry.register(tool)
    tool_executor = ToolExecutor(
        registry,
        approval_handler=approval_handler,
    )
    return Agent(
        model=model,
        tools=registry,
        tool_executor=tool_executor,
        listeners=[
            event_listener
            if event_listener is not None
            else PlainTerminalRenderer(output_fn)
        ],
        session_id=session_id,
        session=session,
    )


def _stderr_output(value: str) -> None:
    print(value, file=sys.stderr)


def _render_status(
    agent: Agent | None,
    *,
    session_id: str,
    workspace: Path,
    model_name: str,
    run_records: Sequence[RunRecord],
    output_fn: OutputFunction,
) -> None:
    record = (
        agent.last_run_record
        if agent is not None and agent.last_run_record is not None
        else (run_records[-1] if run_records else None)
    )
    output_fn(f"Session ID: {session_id}")
    output_fn(f"Workspace: {workspace}")
    output_fn(f"Model: {model_name}")
    output_fn(f"Runs in session: {len(run_records)}")
    output_fn(
        "Last run ID: "
        f"{record.run_id if record is not None else '(none)'}"
    )
    output_fn(
        "Last end reason: "
        f"{record.end_reason if record is not None else '(none)'}"
    )
    if record is not None:
        output_fn(f"Last tool calls: {record.tool_call_count}")
        output_fn(
            "Last estimated context tokens: "
            f"{record.sum_estimated_history_tokens}"
        )


def _write_run_record(path: Path, record: RunRecord) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.to_json() + "\n", encoding="utf-8")
    except OSError as exc:
        raise CLIError(f"Could not write RunRecord: {path}") from exc


def _resolve_workspace(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CLIError(f"Workspace does not exist: {path}") from exc
    if not resolved.is_dir():
        raise CLIError(f"Workspace is not a directory: {path}")
    return resolved


def _select_tasks(
    tasks: tuple[BenchmarkTask, ...],
    requested_ids: list[str] | None,
) -> tuple[BenchmarkTask, ...]:
    if not requested_ids:
        return tasks
    requested = set(requested_ids)
    selected = tuple(
        task for task in tasks if task.task_id in requested
    )
    _require_requested(
        "task",
        requested,
        {task.task_id for task in selected},
    )
    return selected


def _select_configs(
    configs: tuple[BenchmarkConfig, ...],
    requested_ids: list[str] | None,
) -> tuple[BenchmarkConfig, ...]:
    if not requested_ids:
        return configs
    requested = set(requested_ids)
    selected = tuple(
        config
        for config in configs
        if config.config_id in requested
    )
    _require_requested(
        "config",
        requested,
        {config.config_id for config in selected},
    )
    return selected


def _require_requested(
    kind: str,
    requested: set[str],
    selected: set[str],
) -> None:
    missing = sorted(requested - selected)
    if missing:
        raise CLIError(
            f"Unknown {kind} ID(s): {', '.join(missing)}"
        )


def _default_model_factory(model_name: str) -> Model:
    load_dotenv()
    try:
        return DeepSeekModel(model=model_name)
    except ModelError as exc:
        raise CLIError(str(exc)) from exc
