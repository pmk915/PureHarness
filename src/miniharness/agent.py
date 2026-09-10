from miniharness.messages import AgentItem, Message, ToolCall, ToolResult
from miniharness.model import Model
from miniharness.tools import ToolRegistry


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

    def run(self, user_input: str) -> str:
        user_message = Message(
            role="user",
            content=user_input,
        )

        self.messages.append(user_message)

        for step in range(self.max_steps):
            output = self.model.generate(
                self.messages,
                self.tools.list_tools(),
            )

            if isinstance(output, Message):
                self.messages.append(output)
                return output.content

            if isinstance(output, ToolCall):
                self.messages.append(output)

                try:
                    result = self.tools.execute(
                        output.name,
                        output.arguments,
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
                    name=output.name,
                    content=content,
                    call_id=output.call_id,
                    is_error=is_error,
                )

                self.messages.append(tool_result)

        raise RuntimeError("Agent exceeded max steps")