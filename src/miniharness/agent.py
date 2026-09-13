from miniharness.messages import AgentItem, Message, ToolCall, ToolResult
from miniharness.model import Model, ModelError
from miniharness.tools import ToolRegistry
from miniharness.trace import RunTrace, StepTrace
from miniharness.events import AgentEvent


class Agent:
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry | None = None,
        max_steps: int = 10,
    ):
        self.model = model
        self.tools = tools or ToolRegistry()
        self.messages: list[AgentItem] = []
        self.max_steps = max_steps
        self.trace = RunTrace()
        self.events: list[AgentEvent] = []

    def run(self, user_input: str) -> str:
        self.trace = RunTrace()

        self.events = [
            AgentEvent(
                type="agent_started",
                data={},
            )
        ]

        user_message = Message(
            role="user",
            content=user_input,
        )

        self.messages.append(user_message)

        for step in range(self.max_steps):

            try:
                output = self.model.generate(
                    self.messages,
                    self.tools.list_tools(),
                )

            except ModelError:
                self.trace.end_reason = "model_error"

                self.events.append(
                    AgentEvent(
                        type="agent_failed",
                        data={
                            "reason": "model_error",
                        },
                    )
                )

                raise


            # 情况1：模型直接回答
            if isinstance(output, Message):

                self.messages.append(output)

                self.trace.steps.append(
                    StepTrace(
                        index=step,
                        output=output,
                        tool_result=None,
                    )
                )

                self.trace.end_reason = "completed"
                self.events.append(
                    AgentEvent(
                        type="agent_completed",
                        data={
                            "reason": "completed",
                        },
                    )
                )
                return output.content


            # 情况2：模型调用工具
            if isinstance(output, list):

                tool_results = []


                for tool_call in output:
                    self.messages.append(tool_call)

                    try:
                        result = self.tools.execute(
                            tool_call.name,
                            tool_call.arguments,
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

                    self.messages.append(tool_result)

                    tool_results.append(tool_result)


                self.trace.steps.append(
                    StepTrace(
                        index=step,
                        output=output,
                        tool_result=tool_results,
                    )
                )


                continue
        self.trace.end_reason = "max_steps_exceeded"

        self.events.append(
            AgentEvent(
                type="agent_failed",
                data={
                    "reason": "max_steps_exceeded",
                },
            )
        )

        raise RuntimeError("Agent exceeded max steps")