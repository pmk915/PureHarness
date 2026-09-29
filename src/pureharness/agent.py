from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from pureharness.approval import ApprovalDecision, ApprovalRequest
from pureharness.coding_evidence import (
    CodingEvidenceSnapshot,
    CodingEvidenceTracker,
)
from pureharness.context import (
    CompiledContext,
    ContextBudgetExceeded,
    ContextBuilder,
    ContextCompileError,
    ContextLimits,
)
from pureharness.events import AgentEvent, safe_arguments_preview
from pureharness.messages import AgentItem, Message, ToolCall, ToolResult
from pureharness.model import (
    ContextWindowExceededError,
    Model,
    ModelOutput,
)
from pureharness.progress import ProgressSnapshot, ProgressTracker
from pureharness.run_record import (
    ModelInvocationRecord,
    RunRecord,
    RunRecordBuilder,
)
from pureharness.runtime import (
    ExecutionBudget,
    ExecutionBudgetExceeded,
    ExecutionUsage,
    FailureCategory,
    RecoveryAction,
    RuntimeController,
    RuntimeFailure,
    RuntimeStage,
)
from pureharness.session import Session
from pureharness.task_state import (
    TaskState,
    TaskStateError,
    TaskStateReducer,
    render_task_state,
)
from pureharness.tool_executor import ToolExecutor, ToolPreconditionError
from pureharness.tool_policy import PolicyDecision
from pureharness.tool_selection import (
    AllToolsSelector,
    ToolNotExposedError,
    ToolSelection,
    ToolSelectionContext,
    ToolSelectionError,
    ToolSelector,
    prepare_tool_selection,
)
from pureharness.tools import Tool, ToolRegistry
from pureharness.trace import ApprovalTrace, RunTrace, StepTrace
from pureharness.workspace_discipline import (
    WorkspaceMutation,
    WorkspaceSnapshot,
)


@dataclass(frozen=True)
class _PreparedContext:
    task_state: TaskState
    compiled: CompiledContext
    model_items: tuple[AgentItem, ...]
    estimated_task_state_tokens: int
    task_state_item: Message | None


class Agent:
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry | None = None,
        max_steps: int = 10,
        listeners: list[Callable[[AgentEvent], None]] | None = None,
        context_builder: ContextBuilder | None = None,
        session: Session | None = None,
        tool_executor: ToolExecutor | None = None,
        task_state_reducer: TaskStateReducer | None = None,
        tool_selector: ToolSelector | None = None,
        session_id: str | None = None,
        run_id_factory: Callable[[], str] | None = None,
        include_task_state: bool = True,
        max_model_retries: int = 1,
        max_context_recoveries: int = 1,
        context_limits: ContextLimits | None = None,
        execution_budget: ExecutionBudget | None = None,
    ):
        if not isinstance(include_task_state, bool):
            raise ValueError("include_task_state must be bool")
        if (
            not isinstance(max_model_retries, int)
            or isinstance(max_model_retries, bool)
            or max_model_retries < 0
        ):
            raise ValueError("max_model_retries must be non-negative")
        if (
            not isinstance(max_context_recoveries, int)
            or isinstance(max_context_recoveries, bool)
            or max_context_recoveries not in {0, 1}
        ):
            raise ValueError(
                "max_context_recoveries must be 0 or 1"
            )
        if context_limits is not None and not isinstance(
            context_limits,
            ContextLimits,
        ):
            raise ValueError("context_limits must be ContextLimits or None")
        if execution_budget is not None and not isinstance(
            execution_budget,
            ExecutionBudget,
        ):
            raise ValueError(
                "execution_budget must be ExecutionBudget or None"
            )

        self.model = model
        if tool_executor is None:
            self.tools = tools if tools is not None else ToolRegistry()
            self.tool_executor = ToolExecutor(self.tools)
        else:
            if tools is not None and tools is not tool_executor.registry:
                raise ValueError(
                    "Agent tools and ToolExecutor registry must match."
                )

            self.tools = tool_executor.registry
            self.tool_executor = tool_executor
        self.session = session if session is not None else Session()
        self.max_steps = max_steps
        self.trace = RunTrace()
        self.events: list[AgentEvent] = []
        self.listeners = listeners or []
        self.listener_errors: list[Exception] = []
        self.context_builder = context_builder or ContextBuilder()
        self.task_state_reducer = (
            task_state_reducer
            if task_state_reducer is not None
            else TaskStateReducer()
        )
        self.tool_selector = (
            tool_selector
            if tool_selector is not None
            else AllToolsSelector()
        )
        self.session_id = session_id
        self.include_task_state = include_task_state
        self.max_model_retries = max_model_retries
        self.max_context_recoveries = max_context_recoveries
        self.context_limits = context_limits
        self.execution_budget = execution_budget or ExecutionBudget()
        self._execution_usage = ExecutionUsage()
        self._progress_tracker = ProgressTracker()
        self._coding_evidence_tracker = CodingEvidenceTracker()
        self._run_id_factory = (
            run_id_factory
            if run_id_factory is not None
            else lambda: str(uuid4())
        )
        self.runtime_controller = RuntimeController()
        self.last_runtime_failure: RuntimeFailure | None = None
        self.last_run_record: RunRecord | None = None
        self._active_record_builder: RunRecordBuilder | None = None
        self._active_session_size: int | None = None

    @property
    def messages(self):
        return self.session.items

    @property
    def execution_usage(self) -> ExecutionUsage:
        return self._execution_usage

    @property
    def progress_snapshot(self) -> ProgressSnapshot:
        return self._progress_tracker.snapshot(
            self._execution_usage,
            logical_steps_completed=len(self.trace.steps),
        )

    @property
    def coding_evidence_snapshot(self) -> CodingEvidenceSnapshot:
        return self._coding_evidence_tracker.snapshot

    @property
    def workspace_snapshot(self) -> WorkspaceSnapshot | None:
        snapshot = getattr(self.tool_executor.precondition, "snapshot", None)
        return snapshot if isinstance(snapshot, WorkspaceSnapshot) else None

    def _emit(self, event: AgentEvent) -> None:
        if event.run_id is None and self._active_record_builder is not None:
            event.run_id = self._active_record_builder.run_id
        if event.session_id is None:
            event.session_id = self.session_id
        self.events.append(event)

        for listener in self.listeners:
            try:
                listener(event)
            except Exception as exc:
                self.listener_errors.append(exc)

    def _finalize_run_record(
        self,
        builder: RunRecordBuilder,
    ) -> None:
        self.last_run_record = builder.finalize(self.trace)

    def run(self, user_input: str) -> str:
        try:
            return self._run(user_input)
        except KeyboardInterrupt:
            if self.trace.end_reason is None:
                if self._active_session_size is not None:
                    del self.session.items[self._active_session_size :]
                self.trace.end_reason = "interrupted"
                if self._active_record_builder is not None:
                    self._finalize_run_record(
                        self._active_record_builder
                    )
                self._emit_coding_evidence_snapshot(
                    step=(
                        self.trace.steps[-1].index
                        if self.trace.steps
                        else None
                    ),
                    terminal=True,
                )
                self._emit(
                    AgentEvent(
                        type="agent_interrupted",
                        data={
                            "reason": "interrupted",
                            "step_count": len(self.trace.steps),
                        },
                    )
                )
            raise
        finally:
            try:
                self.tool_executor.cleanup_run_state()
            finally:
                self._active_record_builder = None
                self._active_session_size = None

    def _run(self, user_input: str) -> str:
        record_builder = self._start_run(user_input)

        for step in range(self.max_steps):
            history = self._start_step(step)

            try:
                task_state = self.task_state_reducer.reduce(history)
            except TaskStateError as exc:
                self._record_runtime_failure(
                    step=step,
                    stage=RuntimeStage.STATE_REDUCTION,
                    error=exc,
                    builder=record_builder,
                )
                raise

            try:
                prepared = self._prepare_context(
                    step,
                    history,
                    task_state,
                )
            except ContextCompileError as exc:
                self._record_runtime_failure(
                    step=step,
                    stage=RuntimeStage.CONTEXT_PREPARATION,
                    error=exc,
                    builder=record_builder,
                )
                raise

            try:
                tool_selection = self._select_tools(step, task_state)
            except ToolSelectionError as exc:
                self._record_runtime_failure(
                    step=step,
                    stage=RuntimeStage.TOOL_SELECTION,
                    error=exc,
                    builder=record_builder,
                )
                raise

            if self.context_limits is not None:
                try:
                    prepared = self._apply_context_limits(
                        step,
                        history,
                        prepared,
                        tool_selection,
                    )
                except ContextCompileError as exc:
                    self._record_runtime_failure(
                        step=step,
                        stage=RuntimeStage.CONTEXT_PREPARATION,
                        error=exc,
                        builder=record_builder,
                    )
                    raise

            self._start_model_request(
                step,
                prepared,
                tool_selection,
                record_builder,
            )

            output = self._request_model(
                step,
                history,
                prepared,
                tool_selection,
                record_builder,
            )
            self._complete_model_request(step, output, record_builder)

            if isinstance(output, Message):
                return self._complete_run(
                    step,
                    output,
                    record_builder,
                )

            if isinstance(output, list):
                self._execute_tool_calls(
                    step,
                    output,
                    frozenset(
                        tool.name for tool in tool_selection.tools
                    ),
                    record_builder,
                )

        self.trace.end_reason = "max_steps_exceeded"
        self._finalize_run_record(record_builder)
        self._emit_progress_snapshot(
            step=(self.trace.steps[-1].index if self.trace.steps else None),
            terminal=True,
        )
        self._emit_coding_evidence_snapshot(
            step=(self.trace.steps[-1].index if self.trace.steps else None),
            terminal=True,
        )
        self._emit(
            AgentEvent(
                type="agent_failed",
                data={
                    "reason": "max_steps_exceeded",
                    "step_count": len(self.trace.steps),
                },
            )
        )
        raise RuntimeError("Agent exceeded max steps")

    def _start_run(self, user_input: str) -> RunRecordBuilder:
        self.trace = RunTrace()
        self.events = []
        self.listener_errors = []
        self.last_runtime_failure = None
        self.last_run_record = None
        self._execution_usage = ExecutionUsage()
        self._progress_tracker = ProgressTracker()
        self._coding_evidence_tracker = CodingEvidenceTracker()
        self.tool_executor.reset_run_state()
        self._active_session_size = len(self.session.items)
        builder = RunRecordBuilder(
            run_id=self._run_id_factory(),
            session_id=self.session_id,
        )
        self._active_record_builder = builder

        self._emit(
            AgentEvent(
                type="agent_started",
                data={"history_item_count": len(self.session.items)},
            )
        )
        self.session.append(Message(role="user", content=user_input))
        return builder

    def _start_step(self, step: int) -> list[AgentItem]:
        history = self.session.snapshot()
        self._emit(
            AgentEvent(
                type="context_build_started",
                data={
                    "step": step,
                    "history_item_count": len(history),
                },
            )
        )
        return history

    def _prepare_context(
        self,
        step: int,
        history: list[AgentItem],
        task_state: TaskState,
    ) -> _PreparedContext:
        if self.include_task_state:
            task_state_item = render_task_state(task_state)
            estimated_task_state_tokens = (
                self.context_builder.estimate_tokens([task_state_item])
            )
        else:
            task_state_item = None
            estimated_task_state_tokens = 0

        compiled = self.context_builder.compile(history)
        model_items = list(compiled.items)
        if task_state_item is not None:
            model_items.insert(0, task_state_item)

        prepared = _PreparedContext(
            task_state=task_state,
            compiled=compiled,
            model_items=tuple(model_items),
            estimated_task_state_tokens=estimated_task_state_tokens,
            task_state_item=task_state_item,
        )
        if self.context_limits is None:
            self._emit_context_built(step, history, prepared)
        return prepared

    def _apply_context_limits(
        self,
        step: int,
        history: list[AgentItem],
        prepared: _PreparedContext,
        selection: ToolSelection,
    ) -> _PreparedContext:
        limits = self.context_limits
        assert limits is not None
        usable_input_tokens = limits.usable_input_tokens
        non_history_tokens = (
            prepared.estimated_task_state_tokens
            + selection.estimated_tool_schema_tokens
        )
        available_history_tokens = (
            usable_input_tokens - non_history_tokens
        )

        if available_history_tokens <= 0:
            raise ContextBudgetExceeded(
                "TaskState and exposed tool schemas require "
                f"{non_history_tokens} estimated input tokens, leaving "
                "no positive history budget within the usable input "
                f"capacity of {usable_input_tokens}."
            )

        candidate_request_tokens = (
            prepared.compiled.estimated_tokens + non_history_tokens
        )
        pressure_detected = candidate_request_tokens > usable_input_tokens
        final_prepared = prepared

        if pressure_detected:
            self._progress_tracker.record_context_pressure()
            compiled = self.context_builder.compile_bounded(
                history,
                available_history_tokens,
            )
            model_items = list(compiled.items)
            if prepared.task_state_item is not None:
                model_items.insert(0, prepared.task_state_item)
            final_prepared = _PreparedContext(
                task_state=prepared.task_state,
                compiled=compiled,
                model_items=tuple(model_items),
                estimated_task_state_tokens=(
                    prepared.estimated_task_state_tokens
                ),
                task_state_item=prepared.task_state_item,
            )

        estimated_request_tokens = (
            final_prepared.compiled.estimated_tokens
            + non_history_tokens
        )
        self._emit_context_built(
            step,
            history,
            final_prepared,
            pressure_data={
                "context_window_tokens": limits.context_window_tokens,
                "reserved_output_tokens": limits.reserved_output_tokens,
                "usable_input_tokens": usable_input_tokens,
                "estimated_request_tokens": estimated_request_tokens,
                "context_pressure_detected": pressure_detected,
                "available_history_tokens": available_history_tokens,
                "bounded_history_applied": pressure_detected,
            },
        )
        return final_prepared

    def _emit_context_built(
        self,
        step: int,
        history: list[AgentItem],
        prepared: _PreparedContext,
        *,
        pressure_data: dict[str, object] | None = None,
    ) -> None:
        data = self._context_event_data(
            step,
            history,
            prepared.task_state,
            prepared.compiled,
            len(prepared.model_items),
            prepared.estimated_task_state_tokens,
        )
        if pressure_data is not None:
            data.update(pressure_data)
        self._emit(AgentEvent(type="context_built", data=data))

    def _context_event_data(
        self,
        step: int,
        history: list[AgentItem],
        task_state: TaskState,
        compiled: CompiledContext,
        context_item_count: int,
        estimated_task_state_tokens: int,
    ) -> dict[str, object]:
        data = {
            "step": step,
            "history_item_count": len(history),
            "context_item_count": context_item_count,
            "trajectory_item_count": len(compiled.items),
            "context_strategy": compiled.strategy,
            "estimated_history_tokens": compiled.estimated_tokens,
            "estimated_task_state_tokens": estimated_task_state_tokens,
            "current_request_present": (
                task_state.current_request is not None
            ),
            "completed_actions_count": len(task_state.completed_actions),
            "failed_actions_count": len(task_state.failed_actions),
            "files_read_count": len(task_state.files_read),
            "files_modified_count": len(task_state.files_modified),
            "recent_errors_count": len(task_state.recent_errors),
            "total_units": compiled.total_units,
            "included_units": compiled.included_units,
            "dropped_units": compiled.dropped_units,
            "projected_tool_results": compiled.projected_tool_results,
            "compacted_tool_results": compiled.compacted_tool_results,
            "raw_tool_result_chars": compiled.raw_tool_result_chars,
            "projected_tool_result_chars": (
                compiled.projected_tool_result_chars
            ),
            "trajectory_compacted": compiled.trajectory_compacted,
            "compacted_source_units": compiled.compacted_source_units,
            "compacted_tool_actions": compiled.compacted_tool_actions,
            "original_trajectory_estimated_tokens": (
                compiled.original_trajectory_estimated_tokens
            ),
            "compacted_trajectory_estimated_tokens": (
                compiled.compacted_trajectory_estimated_tokens
            ),
            "recent_raw_units": compiled.recent_raw_units,
            "recent_raw_estimated_tokens": (
                compiled.recent_raw_estimated_tokens
            ),
            "trajectory_compaction_strategy": (
                compiled.trajectory_compaction_strategy
            ),
        }
        if compiled.history_token_budget is not None:
            data["history_token_budget"] = compiled.history_token_budget
        return data

    def _select_tools(
        self,
        step: int,
        task_state: TaskState,
    ) -> ToolSelection:
        return prepare_tool_selection(
            self.tools.list_tools(),
            self.tool_selector,
            ToolSelectionContext(step=step, task_state=task_state),
        )

    def _start_model_request(
        self,
        step: int,
        prepared: _PreparedContext,
        selection: ToolSelection,
        builder: RunRecordBuilder,
    ) -> None:
        builder.record_model_invocation(
            self._model_invocation_record(step, prepared, selection)
        )
        self._emit(
            AgentEvent(
                type="model_started",
                data={
                    "step": step,
                    "registered_tool_count": (
                        selection.registered_tool_count
                    ),
                    "exposed_tool_count": selection.exposed_tool_count,
                    "estimated_tool_schema_tokens": (
                        selection.estimated_tool_schema_tokens
                    ),
                    "estimated_all_tool_schema_tokens": (
                        selection.estimated_all_tool_schema_tokens
                    ),
                    "estimated_tool_schema_tokens_saved": (
                        selection.estimated_tool_schema_tokens_saved
                    ),
                    "selector_strategy": selection.selector_strategy,
                },
            )
        )

    def _model_invocation_record(
        self,
        step: int,
        prepared: _PreparedContext,
        selection: ToolSelection,
    ) -> ModelInvocationRecord:
        compiled = prepared.compiled
        return ModelInvocationRecord(
            step=step,
            context_strategy=compiled.strategy,
            estimated_history_tokens=compiled.estimated_tokens,
            estimated_task_state_tokens=(
                prepared.estimated_task_state_tokens
            ),
            registered_tool_count=selection.registered_tool_count,
            exposed_tool_count=selection.exposed_tool_count,
            estimated_tool_schema_tokens=(
                selection.estimated_tool_schema_tokens
            ),
            selector_strategy=selection.selector_strategy,
            trajectory_compacted=compiled.trajectory_compacted,
            compacted_source_units=compiled.compacted_source_units,
            compacted_tool_results=compiled.compacted_tool_results,
        )

    def _request_model(
        self,
        step: int,
        history: list[AgentItem],
        prepared: _PreparedContext,
        selection: ToolSelection,
        builder: RunRecordBuilder,
    ) -> ModelOutput:
        max_attempts = self.max_model_retries + 1
        attempt = 1
        context_recovery_attempt = 0
        current_prepared = prepared
        retry_pending = False

        while True:
            try:
                self._consume_model_attempt()
            except ExecutionBudgetExceeded as exc:
                self._record_runtime_failure(
                    step=step,
                    stage=RuntimeStage.EXECUTION,
                    error=exc,
                    builder=builder,
                )
                raise
            if retry_pending:
                self._progress_tracker.record_model_retry()
                retry_pending = False
            try:
                return self.model.generate(
                    list(current_prepared.model_items),
                    list(selection.tools),
                )
            except Exception as exc:
                failure = self.runtime_controller.classify_failure(
                    step=step,
                    stage=RuntimeStage.MODEL_REQUEST,
                    error=exc,
                )
                self.last_runtime_failure = failure
                action = self.runtime_controller.recovery_action(
                    failure,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    context_recovery_attempt=(
                        context_recovery_attempt
                    ),
                    max_context_recoveries=(
                        self.max_context_recoveries
                    ),
                )

                if isinstance(exc, ContextWindowExceededError):
                    self._progress_tracker.record_context_window_exceeded()
                    recovery_available = (
                        action is RecoveryAction.REBUILD_CONTEXT
                    )
                    event_data: dict[str, object] = {
                        "step": step,
                        "error_type": failure.error_type,
                        "context_recovery_available": (
                            recovery_available
                        ),
                        "max_context_recoveries": (
                            self.max_context_recoveries
                        ),
                    }
                    if recovery_available:
                        event_data["recovery_attempt"] = (
                            context_recovery_attempt + 1
                        )
                    self._emit(
                        AgentEvent(
                            type="context_window_exceeded",
                            data=event_data,
                        )
                    )

                if action is RecoveryAction.RETRY:
                    attempt += 1
                    retry_pending = True
                    self._emit(
                        AgentEvent(
                            type="model_retrying",
                            data={
                                "step": step,
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                                "error_type": failure.error_type,
                                "failure_category": (
                                    failure.category.value
                                ),
                            },
                        )
                    )
                    continue

                if action is RecoveryAction.REBUILD_CONTEXT:
                    next_recovery_attempt = context_recovery_attempt + 1
                    previous_history_tokens = (
                        current_prepared.compiled.estimated_tokens
                    )
                    recovery_history_budget = (
                        previous_history_tokens // 2
                    )
                    try:
                        recovered = self._recover_context(
                            history,
                            current_prepared,
                            recovery_history_budget,
                        )
                    except ContextCompileError as recovery_exc:
                        self._emit_context_recovering(
                            step=step,
                            recovery_attempt=next_recovery_attempt,
                            error_type=failure.error_type,
                            previous=current_prepared,
                            selection=selection,
                            recovery_history_budget=(
                                recovery_history_budget
                            ),
                        )
                        recovery_failure = (
                            self.runtime_controller.classify_failure(
                                step=step,
                                stage=(
                                    RuntimeStage.CONTEXT_PREPARATION
                                ),
                                error=recovery_exc,
                            )
                        )
                        self.last_runtime_failure = recovery_failure
                        self._finalize_runtime_failure(
                            recovery_failure,
                            builder,
                        )
                        raise recovery_exc from exc

                    context_recovery_attempt = next_recovery_attempt
                    self._progress_tracker.record_context_recovery()
                    builder.replace_model_invocation(
                        self._model_invocation_record(
                            step,
                            recovered,
                            selection,
                        )
                    )
                    self._emit_context_recovering(
                        step=step,
                        recovery_attempt=context_recovery_attempt,
                        error_type=failure.error_type,
                        previous=current_prepared,
                        selection=selection,
                        recovery_history_budget=(
                            recovery_history_budget
                        ),
                        recovered=recovered,
                    )
                    current_prepared = recovered
                    continue

                self._finalize_runtime_failure(
                    failure,
                    builder,
                    emit_context_build_failed=(
                        not isinstance(
                            exc,
                            ContextWindowExceededError,
                        )
                    ),
                )
                raise

    def _recover_context(
        self,
        history: list[AgentItem],
        previous: _PreparedContext,
        recovery_history_budget: int,
    ) -> _PreparedContext:
        if recovery_history_budget <= 0:
            raise ContextBudgetExceeded(
                "Provider context recovery cannot derive a positive "
                "history budget from the previous estimated history "
                f"size of {previous.compiled.estimated_tokens}."
            )

        compiled = self.context_builder.compile_bounded(
            history,
            recovery_history_budget,
        )
        if compiled.estimated_tokens >= previous.compiled.estimated_tokens:
            raise ContextBudgetExceeded(
                "Provider context recovery did not produce strictly "
                "smaller estimated history."
            )

        model_items = list(compiled.items)
        if previous.task_state_item is not None:
            model_items.insert(0, previous.task_state_item)
        return _PreparedContext(
            task_state=previous.task_state,
            compiled=compiled,
            model_items=tuple(model_items),
            estimated_task_state_tokens=(
                previous.estimated_task_state_tokens
            ),
            task_state_item=previous.task_state_item,
        )

    def _emit_context_recovering(
        self,
        *,
        step: int,
        recovery_attempt: int,
        error_type: str,
        previous: _PreparedContext,
        selection: ToolSelection,
        recovery_history_budget: int,
        recovered: _PreparedContext | None = None,
    ) -> None:
        non_history_tokens = (
            previous.estimated_task_state_tokens
            + selection.estimated_tool_schema_tokens
        )
        data: dict[str, object] = {
            "step": step,
            "recovery_attempt": recovery_attempt,
            "max_recoveries": self.max_context_recoveries,
            "error_type": error_type,
            "previous_history_tokens": (
                previous.compiled.estimated_tokens
            ),
            "recovery_history_budget": recovery_history_budget,
            "previous_estimated_request_tokens": (
                previous.compiled.estimated_tokens
                + non_history_tokens
            ),
        }
        if recovered is not None:
            data.update(
                {
                    "recovered_history_tokens": (
                        recovered.compiled.estimated_tokens
                    ),
                    "recovered_estimated_request_tokens": (
                        recovered.compiled.estimated_tokens
                        + non_history_tokens
                    ),
                }
            )
        self._emit(AgentEvent(type="context_recovering", data=data))

    def _complete_model_request(
        self,
        step: int,
        output: ModelOutput,
        builder: RunRecordBuilder,
    ) -> None:
        output_kind = (
            "message" if isinstance(output, Message) else "tool_calls"
        )
        tool_call_count = len(output) if isinstance(output, list) else 0
        builder.record_tool_calls(tool_call_count)
        self._emit(
            AgentEvent(
                type="model_completed",
                data={
                    "step": step,
                    "output_kind": output_kind,
                    "tool_call_count": tool_call_count,
                },
            )
        )

    def _complete_run(
        self,
        step: int,
        output: Message,
        builder: RunRecordBuilder,
    ) -> str:
        self.session.append(output)
        self.trace.steps.append(
            StepTrace(index=step, output=output, tool_result=None)
        )
        self.trace.end_reason = "completed"
        self._finalize_run_record(builder)
        self._emit_progress_snapshot(step=step, terminal=True)
        self._emit_coding_evidence_snapshot(step=step, terminal=True)
        self._emit(
            AgentEvent(
                type="agent_completed",
                data={
                    "reason": "completed",
                    "step_count": len(self.trace.steps),
                },
            )
        )
        return output.content

    def _record_runtime_failure(
        self,
        *,
        step: int,
        stage: RuntimeStage,
        error: Exception,
        builder: RunRecordBuilder,
    ) -> None:
        failure = self.runtime_controller.classify_failure(
            step=step,
            stage=stage,
            error=error,
        )
        self.last_runtime_failure = failure
        if isinstance(error, ExecutionBudgetExceeded):
            event_data: dict[str, object] = {
                "step": step,
                "resource": error.resource,
                "used": error.used,
                "limit": error.limit,
            }
            if error.requested is not None:
                event_data.update(
                    {
                        "requested": error.requested,
                        "remaining": error.remaining,
                    }
                )
            self._emit(
                AgentEvent(
                    type="execution_budget_exhausted",
                    data=event_data,
                )
            )
        self._finalize_runtime_failure(failure, builder)

    def _consume_model_attempt(self) -> None:
        used = self._execution_usage.model_attempts
        limit = self.execution_budget.max_model_attempts
        if limit is not None and used >= limit:
            raise ExecutionBudgetExceeded(
                resource="model_attempts",
                used=used,
                limit=limit,
            )
        self._execution_usage = ExecutionUsage(
            model_attempts=used + 1,
            tool_calls=self._execution_usage.tool_calls,
        )

    def _consume_tool_calls(self, requested: int) -> None:
        if requested == 0:
            return
        used = self._execution_usage.tool_calls
        limit = self.execution_budget.max_tool_calls
        if limit is not None and requested > limit - used:
            raise ExecutionBudgetExceeded(
                resource="tool_calls",
                used=used,
                limit=limit,
                requested=requested,
            )
        self._execution_usage = ExecutionUsage(
            model_attempts=self._execution_usage.model_attempts,
            tool_calls=used + requested,
        )

    def _finalize_runtime_failure(
        self,
        failure: RuntimeFailure,
        builder: RunRecordBuilder,
        *,
        emit_context_build_failed: bool = True,
    ) -> None:
        reason = self.runtime_controller.end_reason(failure)
        self.trace.end_reason = reason

        if (
            failure.category is FailureCategory.CONTEXT
            and emit_context_build_failed
        ):
            self._emit(
                AgentEvent(
                    type="context_build_failed",
                    data={
                        "step": failure.step,
                        "reason": reason,
                        "error_type": failure.error_type,
                    },
                )
            )
        elif failure.category is FailureCategory.MODEL:
            self._emit(
                AgentEvent(
                    type="model_failed",
                    data={
                        "step": failure.step,
                        "reason": reason,
                        "error_type": failure.error_type,
                    },
                )
            )

        self._finalize_run_record(builder)
        self._emit_progress_snapshot(
            step=failure.step,
            terminal=True,
        )
        self._emit_coding_evidence_snapshot(
            step=failure.step,
            terminal=True,
        )
        event_data = {
            "reason": reason,
            "step_count": len(self.trace.steps),
        }
        if failure.category is not FailureCategory.CONTEXT:
            event_data["error_type"] = failure.error_type
        self._emit(AgentEvent(type="agent_failed", data=event_data))

    def _emit_progress_snapshot(
        self,
        *,
        step: int | None,
        terminal: bool,
    ) -> None:
        snapshot = self.progress_snapshot
        data: dict[str, object] = {
            "logical_steps_completed": (
                snapshot.logical_steps_completed
            ),
            "model_attempts": snapshot.model_attempts,
            "tool_calls": snapshot.tool_calls,
            "successful_tool_results": (
                snapshot.successful_tool_results
            ),
            "failed_tool_results": snapshot.failed_tool_results,
            "unique_tool_actions": snapshot.unique_tool_actions,
            "repeated_tool_actions": snapshot.repeated_tool_actions,
            "max_identical_tool_action_count": (
                snapshot.max_identical_tool_action_count
            ),
            "model_retries": snapshot.model_retries,
            "context_recoveries": snapshot.context_recoveries,
            "context_pressure_count": snapshot.context_pressure_count,
            "context_window_exceeded_count": (
                snapshot.context_window_exceeded_count
            ),
            "terminal": terminal,
        }
        if step is not None:
            data["step"] = step
        self._emit(AgentEvent(type="progress_snapshot", data=data))

    def _emit_coding_evidence_snapshot(
        self,
        *,
        step: int | None,
        terminal: bool,
    ) -> None:
        snapshot = self.coding_evidence_snapshot
        data: dict[str, object] = {
            "workspace_mutations": snapshot.workspace_mutations,
            "command_executions": snapshot.command_executions,
            "command_tool_errors": snapshot.command_tool_errors,
            "process_starts": snapshot.process_starts,
            "process_polls": snapshot.process_polls,
            "process_stops": snapshot.process_stops,
            "process_tool_errors": snapshot.process_tool_errors,
            "executions_since_last_mutation": (
                snapshot.executions_since_last_mutation
            ),
            "last_mutation_step": snapshot.last_mutation_step,
            "last_execution_step": snapshot.last_execution_step,
            "terminal": terminal,
        }
        if step is not None:
            data["step"] = step
        self._emit(
            AgentEvent(type="coding_evidence_snapshot", data=data)
        )

    def _execute_tool_calls(
        self,
        step: int,
        tool_calls: list[ToolCall],
        exposed_tool_names: frozenset[str],
        builder: RunRecordBuilder,
    ) -> None:
        accepted_call_count = sum(
            tool_call.name in exposed_tool_names
            for tool_call in tool_calls
        )
        try:
            self._consume_tool_calls(accepted_call_count)
        except ExecutionBudgetExceeded as exc:
            self._record_runtime_failure(
                step=step,
                stage=RuntimeStage.EXECUTION,
                error=exc,
                builder=builder,
            )
            raise

        tool_results: list[ToolResult] = []

        for tool_call in tool_calls:
            self.session.append(tool_call)
            tool_started_at: float | None = None
            started_tool: Tool | None = None
            workspace_mutation: WorkspaceMutation | None = None

            def on_policy_evaluated(
                tool: Tool,
                decision: PolicyDecision,
            ) -> None:
                self._emit(
                    AgentEvent(
                        type="tool_policy_evaluated",
                        data={
                            "step": step,
                            "name": tool.name,
                            "call_id": tool_call.call_id,
                            "risk_level": tool.risk_level.value,
                            "decision": decision.value,
                        },
                    )
                )

            def on_tool_started(tool: Tool) -> None:
                nonlocal tool_started_at, started_tool
                tool_started_at = perf_counter()
                started_tool = tool
                self._coding_evidence_tracker.record_tool_started(
                    tool,
                    step=step,
                )
                self._emit(
                    AgentEvent(
                        type="tool_started",
                        data={
                            "step": step,
                            "name": tool.name,
                            "call_id": tool_call.call_id,
                            "arguments_preview": safe_arguments_preview(
                                tool_call.arguments
                            ),
                        },
                    )
                )

            def on_precondition_failed(
                tool: Tool,
                error: ToolPreconditionError,
            ) -> None:
                data: dict[str, object] = {
                    "step": step,
                    "name": tool.name,
                    "call_id": tool_call.call_id,
                    "reason": error.reason,
                }
                if error.path is not None:
                    data["path"] = error.path
                if error.change is not None:
                    data["change"] = error.change
                self._emit(
                    AgentEvent(
                        type="workspace_precondition_failed",
                        data=data,
                    )
                )

            def on_precondition_recorded(evidence: object) -> None:
                nonlocal workspace_mutation
                if isinstance(evidence, WorkspaceMutation):
                    workspace_mutation = evidence

            def on_approval_requested(
                tool: Tool,
                request: ApprovalRequest,
            ) -> None:
                self._emit(
                    AgentEvent(
                        type="approval_requested",
                        data={
                            "run_id": builder.run_id,
                            "step": step,
                            "name": tool.name,
                            "call_id": tool_call.call_id,
                            "arguments_preview": safe_arguments_preview(
                                request.arguments
                            ),
                        },
                    )
                )

            def on_approval_resolved(
                tool: Tool,
                _request: ApprovalRequest,
                decision: ApprovalDecision,
            ) -> None:
                self.trace.approvals.append(
                    ApprovalTrace(
                        step=step,
                        tool_name=tool.name,
                        call_id=tool_call.call_id,
                        decision=decision,
                    )
                )
                event_type = (
                    "approval_granted"
                    if decision is ApprovalDecision.APPROVE
                    else "approval_denied"
                )
                self._emit(
                    AgentEvent(
                        type=event_type,
                        data={
                            "run_id": builder.run_id,
                            "step": step,
                            "name": tool.name,
                            "call_id": tool_call.call_id,
                            "approval_decision": decision.value,
                        },
                    )
                )

            try:
                if tool_call.name not in exposed_tool_names:
                    raise ToolNotExposedError(
                        tool_call.name,
                        exposed_tool_names,
                    )

                self._progress_tracker.record_tool_action(
                    tool_call.name,
                    tool_call.arguments,
                )
                result = self.tool_executor.execute(
                    tool_call.name,
                    tool_call.arguments,
                    on_policy_evaluated=on_policy_evaluated,
                    on_approval_requested=on_approval_requested,
                    on_approval_resolved=on_approval_resolved,
                    on_tool_started=on_tool_started,
                    on_precondition_failed=(
                        on_precondition_failed
                    ),
                    on_precondition_recorded=(
                        on_precondition_recorded
                    ),
                )
                content = str(result)
                is_error = False
            except Exception as exc:
                content = f"Tool error: {type(exc).__name__}: {exc}"
                is_error = True

            tool_result = ToolResult(
                name=tool_call.name,
                content=content,
                call_id=tool_call.call_id,
                is_error=is_error,
            )
            self.session.append(tool_result)
            tool_results.append(tool_result)
            self._progress_tracker.record_tool_result(
                is_error=is_error
            )
            if started_tool is not None:
                self._coding_evidence_tracker.record_tool_result(
                    started_tool,
                    is_error=is_error,
                )

            if tool_started_at is not None:
                builder.record_tool_execution()
            builder.record_tool_result(is_error=is_error)

            if tool_started_at is not None:
                duration_seconds = perf_counter() - tool_started_at
                self._emit(
                    AgentEvent(
                        type="tool_completed",
                        data={
                            "step": step,
                            "name": tool_call.name,
                            "call_id": tool_call.call_id,
                            "is_error": is_error,
                            "duration_seconds": round(
                                duration_seconds,
                                6,
                            ),
                            "result_character_count": len(content),
                        },
                    )
                )
                if workspace_mutation is not None:
                    self._coding_evidence_tracker.record_workspace_mutation(
                        workspace_mutation,
                        step=step,
                    )
                    self._emit(
                        AgentEvent(
                            type="workspace_mutated",
                            data={
                                "step": step,
                                "name": workspace_mutation.tool_name,
                                "call_id": tool_call.call_id,
                                "path": workspace_mutation.path,
                                "operation": workspace_mutation.operation,
                            },
                        )
                    )

        self.trace.steps.append(
            StepTrace(
                index=step,
                output=tool_calls,
                tool_result=tool_results,
            )
        )
        self._emit_progress_snapshot(step=step, terminal=False)
        self._emit_coding_evidence_snapshot(step=step, terminal=False)
