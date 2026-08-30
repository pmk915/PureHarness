from miniharness.messages import AgentItem, Message, ToolCall, ToolResult
from miniharness.model import Model
from miniharness.tools import ToolRegistry


class Agent:
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry | None = None,
    ):
        self.model = model
        self.tools = tools or ToolRegistry()
        self.messages: list[AgentItem] = []

    def run(self, user_input: str) -> str:
        user_message = Message(
            role="user",
            content=user_input,
        )

        self.messages.append(user_message)

        while True:
            output = self.model.generate(
                self.messages,
                self.tools.list_tools(),
            )

            if isinstance(output, Message):
                self.messages.append(output)
                return output.content

            if isinstance(output, ToolCall):
                self.messages.append(output)

                result = self.tools.execute(
                    output.name,
                    output.arguments,
                )

                tool_result = ToolResult(
                    name=output.name,
                    content=str(result),
                    call_id=output.call_id,
                )

                self.messages.append(tool_result)