from miniharness.messages import AgentItem, Message, ToolCall, ToolResult
from miniharness.model import Model, ModelError
from miniharness.run_record import (
    ModelInvocationRecord,
    RunRecord,
    RunRecordBuilder,
)
from miniharness.tool_executor import ToolExecutor
from miniharness.tool_policy import PolicyDecision
from miniharness.tool_selection import (
    AllToolsSelector,
    ToolNotExposedError,
    ToolSelectionContext,
    ToolSelectionError,
    ToolSelector,
    prepare_tool_selection,
)
from miniharness.tools import Tool, ToolRegistry
from miniharness.trace import RunTrace, StepTrace
from miniharness.events import AgentEvent, safe_arguments_preview
from miniharness.context import (
    ContextBuilder,
    ContextCompileError,
)

from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

from miniharness.session import Session
from miniharness.task_state import (
    TaskStateError,
    TaskStateReducer,
    render_task_state,
)


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
    ):
        if not isinstance(include_task_state, bool):
            raise ValueError("include_task_state must be bool")

        self.model = model
        if tool_executor is None:
            self.tools = (
                tools
                if tools is not None
                else ToolRegistry()
            )
            self.tool_executor = ToolExecutor(self.tools)
        else:
            if (
                tools is not None
                and tools is not tool_executor.registry
            ):
                raise ValueError(
                    "Agent tools and ToolExecutor registry must match."
                )

            self.tools = tool_executor.registry
            self.tool_executor = tool_executor
        self.session = (
            session
            if session is not None
            else Session()
        )
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
        self._run_id_factory = (
            run_id_factory
            if run_id_factory is not None
            else lambda: str(uuid4())
        )
        self.last_run_record: RunRecord | None = None
        self._active_record_builder: RunRecordBuilder | None = None
        self._active_session_size: int | None = None


    @property
    def messages(self):
        return self.session.items


    def _emit(self, event: AgentEvent) -> None:
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
                    del self.session.items[self._active_session_size:]
                self.trace.end_reason = "interrupted"
                if self._active_record_builder is not None:
                    self._finalize_run_record(
                        self._active_record_builder
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
            self._active_record_builder = None
            self._active_session_size = None


    def _run(self, user_input: str) -> str:
        self.trace = RunTrace()
        self.events = []
        self.listener_errors = []
        self.last_run_record = None
        self._active_session_size = len(self.session.items)
        record_builder = RunRecordBuilder(
            run_id=self._run_id_factory(),
            session_id=self.session_id,
        )
        self._active_record_builder = record_builder

        self._emit(
            AgentEvent(
                type="agent_started",
                data={
                    "history_item_count": len(
                        self.session.items
                    ),
                },
            )
        )

        user_message = Message(
            role="user",
            content=user_input,
        )

        self.session.append(user_message)

        for step in range(self.max_steps):

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

            try:
                task_state = self.task_state_reducer.reduce(history)
                if self.include_task_state:
                    task_state_item = render_task_state(task_state)
                    estimated_task_state_tokens = (
                        self.context_builder.estimate_tokens(
                            [task_state_item]
                        )
                    )
                else:
                    task_state_item = None
                    estimated_task_state_tokens = 0
                compiled_context = self.context_builder.compile(
                    history
                )
            except (ContextCompileError, TaskStateError) as exc:
                self.trace.end_reason = "context_error"

                self._emit(
                    AgentEvent(
                        type="context_build_failed",
                        data={
                            "step": step,
                            "reason": "context_error",
                            "error_type": type(exc).__name__,
                        },
                    )
                )

                self._finalize_run_record(record_builder)

                self._emit(
                    AgentEvent(
                        type="agent_failed",
                        data={
                            "reason": "context_error",
                            "step_count": len(
                                self.trace.steps
                            ),
                        },
                    )
                )

                raise

            model_context = list(compiled_context.items)
            if task_state_item is not None:
                model_context.insert(0, task_state_item)
            context_event_data = {
                "step": step,
                "history_item_count": len(history),
                "context_item_count": len(model_context),
                "trajectory_item_count": len(
                    compiled_context.items
                ),
                "context_strategy": compiled_context.strategy,
                "estimated_history_tokens": (
                    compiled_context.estimated_tokens
                ),
                "estimated_task_state_tokens": (
                    estimated_task_state_tokens
                ),
                "current_request_present": (
                    task_state.current_request is not None
                ),
                "completed_actions_count": len(
                    task_state.completed_actions
                ),
                "failed_actions_count": len(
                    task_state.failed_actions
                ),
                "files_read_count": len(task_state.files_read),
                "files_modified_count": len(
                    task_state.files_modified
                ),
                "recent_errors_count": len(
                    task_state.recent_errors
                ),
                "total_units": compiled_context.total_units,
                "included_units": (
                    compiled_context.included_units
                ),
                "dropped_units": compiled_context.dropped_units,
                "projected_tool_results": (
                    compiled_context.projected_tool_results
                ),
                "compacted_tool_results": (
                    compiled_context.compacted_tool_results
                ),
                "raw_tool_result_chars": (
                    compiled_context.raw_tool_result_chars
                ),
                "projected_tool_result_chars": (
                    compiled_context.projected_tool_result_chars
                ),
                "trajectory_compacted": (
                    compiled_context.trajectory_compacted
                ),
                "compacted_source_units": (
                    compiled_context.compacted_source_units
                ),
                "compacted_tool_actions": (
                    compiled_context.compacted_tool_actions
                ),
                "original_trajectory_estimated_tokens": (
                    compiled_context.original_trajectory_estimated_tokens
                ),
                "compacted_trajectory_estimated_tokens": (
                    compiled_context.compacted_trajectory_estimated_tokens
                ),
                "recent_raw_units": (
                    compiled_context.recent_raw_units
                ),
                "recent_raw_estimated_tokens": (
                    compiled_context.recent_raw_estimated_tokens
                ),
                "trajectory_compaction_strategy": (
                    compiled_context.trajectory_compaction_strategy
                ),
            }

            if compiled_context.history_token_budget is not None:
                context_event_data["history_token_budget"] = (
                    compiled_context.history_token_budget
                )

            self._emit(
                AgentEvent(
                    type="context_built",
                    data=context_event_data,
                )
            )

            try:
                tool_selection = prepare_tool_selection(
                    self.tools.list_tools(),
                    self.tool_selector,
                    ToolSelectionContext(
                        step=step,
                        task_state=task_state,
                    ),
                )
            except ToolSelectionError as exc:
                self.trace.end_reason = "tool_selection_error"

                self._finalize_run_record(record_builder)

                self._emit(
                    AgentEvent(
                        type="agent_failed",
                        data={
                            "reason": "tool_selection_error",
                            "error_type": type(exc).__name__,
                            "step_count": len(self.trace.steps),
                        },
                    )
                )

                raise

            exposed_tool_names = frozenset(
                tool.name for tool in tool_selection.tools
            )
            record_builder.record_model_invocation(
                ModelInvocationRecord(
                    step=step,
                    context_strategy=compiled_context.strategy,
                    estimated_history_tokens=(
                        compiled_context.estimated_tokens
                    ),
                    estimated_task_state_tokens=(
                        estimated_task_state_tokens
                    ),
                    registered_tool_count=(
                        tool_selection.registered_tool_count
                    ),
                    exposed_tool_count=(
                        tool_selection.exposed_tool_count
                    ),
                    estimated_tool_schema_tokens=(
                        tool_selection.estimated_tool_schema_tokens
                    ),
                    selector_strategy=(
                        tool_selection.selector_strategy
                    ),
                    trajectory_compacted=(
                        compiled_context.trajectory_compacted
                    ),
                    compacted_source_units=(
                        compiled_context.compacted_source_units
                    ),
                    compacted_tool_results=(
                        compiled_context.compacted_tool_results
                    ),
                )
            )

            self._emit(
                AgentEvent(
                    type="model_started",
                    data={
                        "step": step,
                        "registered_tool_count": (
                            tool_selection.registered_tool_count
                        ),
                        "exposed_tool_count": (
                            tool_selection.exposed_tool_count
                        ),
                        "estimated_tool_schema_tokens": (
                            tool_selection.estimated_tool_schema_tokens
                        ),
                        "estimated_all_tool_schema_tokens": (
                            tool_selection.estimated_all_tool_schema_tokens
                        ),
                        "estimated_tool_schema_tokens_saved": (
                            tool_selection.estimated_tool_schema_tokens_saved
                        ),
                        "selector_strategy": (
                            tool_selection.selector_strategy
                        ),
                    },
                )
            )

            try:
                output = self.model.generate(
                    model_context,
                    list(tool_selection.tools),
                )
            except ModelError as exc:
                self.trace.end_reason = "model_error"

                self._emit(
                    AgentEvent(
                        type="model_failed",
                        data={
                            "step": step,
                            "reason": "model_error",
                            "error_type": type(exc).__name__,
                        },
                    )
                )

                self._finalize_run_record(record_builder)

                self._emit(
                    AgentEvent(
                        type="agent_failed",
                        data={
                            "reason": "model_error",
                            "step_count": len(
                                self.trace.steps
                            ),
                        },
                    )
                )

                raise

            output_kind = (
                "message"
                if isinstance(output, Message)
                else "tool_calls"
            )
            tool_call_count = (
                len(output)
                if isinstance(output, list)
                else 0
            )
            record_builder.record_tool_calls(tool_call_count)

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

            # 情况1：模型直接回答
            if isinstance(output, Message):

                self.session.append(output)

                self.trace.steps.append(
                    StepTrace(
                        index=step,
                        output=output,
                        tool_result=None,
                    )
                )

                self.trace.end_reason = "completed"
                self._finalize_run_record(record_builder)
                self._emit(
                    AgentEvent(
                        type="agent_completed",
                        data={
                            "reason": "completed",
                            "step_count": len(
                                self.trace.steps
                            ),
                        },
                    )
                )
                return output.content


            # 情况2：模型调用工具
            if isinstance(output, list):

                tool_results = []

                for tool_call in output:

                    self.session.append(tool_call)

                    tool_started_at: float | None = None

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
                        nonlocal tool_started_at
                        tool_started_at = perf_counter()

                        self._emit(
                            AgentEvent(
                                type="tool_started",
                                data={
                                    "step": step,
                                    "name": tool.name,
                                    "call_id": tool_call.call_id,
                                    "arguments_preview": (
                                        safe_arguments_preview(
                                            tool_call.arguments
                                        )
                                    ),
                                },
                            )
                        )

                    try:
                        if (
                            tool_call.name
                            not in exposed_tool_names
                        ):
                            raise ToolNotExposedError(
                                tool_call.name
                            )

                        result = self.tool_executor.execute(
                            tool_call.name,
                            tool_call.arguments,
                            on_policy_evaluated=(
                                on_policy_evaluated
                            ),
                            on_tool_started=on_tool_started,
                        )

                        content = str(result)
                        is_error = False

                    except Exception as exc:

                        content = (
                            f"Tool error: "
                            f"{type(exc).__name__}: {exc}"
                        )

                        is_error = True

                    tool_result = ToolResult(
                        name=tool_call.name,
                        content=content,
                        call_id=tool_call.call_id,
                        is_error=is_error,
                    )

                    self.session.append(tool_result)

                    tool_results.append(tool_result)

                    if tool_started_at is not None:
                        record_builder.record_tool_execution()
                    record_builder.record_tool_result(
                        is_error=is_error
                    )

                    if tool_started_at is not None:
                        duration_seconds = (
                            perf_counter() - tool_started_at
                        )

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
                                    "result_character_count": len(
                                        content
                                    ),
                                },
                            )
                        )

                self.trace.steps.append(
                    StepTrace(
                        index=step,
                        output=output,
                        tool_result=tool_results,
                    )
                )


                continue
        self.trace.end_reason = "max_steps_exceeded"
        self._finalize_run_record(record_builder)

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
