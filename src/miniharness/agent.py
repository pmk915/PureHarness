from miniharness.messages import AgentItem, Message, ToolCall, ToolResult
from miniharness.model import Model, ModelError
from miniharness.tool_executor import ToolExecutor
from miniharness.tool_policy import PolicyDecision
from miniharness.tools import Tool, ToolRegistry
from miniharness.trace import RunTrace, StepTrace
from miniharness.events import AgentEvent, safe_arguments_preview
from miniharness.context import (
    ContextBuilder,
    ContextCompileError,
)

from collections.abc import Callable
from time import perf_counter

from miniharness.session import Session


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
    ):
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


    def run(self, user_input: str) -> str:
        self.trace = RunTrace()
        self.events = []
        self.listener_errors = []

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
                compiled_context = self.context_builder.compile(
                    history
                )
            except ContextCompileError as exc:
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

            context_event_data = {
                "step": step,
                "history_item_count": len(history),
                "context_item_count": len(
                    compiled_context.items
                ),
                "context_strategy": compiled_context.strategy,
                "estimated_history_tokens": (
                    compiled_context.estimated_tokens
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

            self._emit(
                AgentEvent(
                    type="model_started",
                    data={
                        "step": step,
                    },
                )
            )

            try:
                output = self.model.generate(
                    compiled_context.items,
                    self.tools.list_tools(),
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
