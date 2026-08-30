from miniharness.messages import Message, ToolCall
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
        self.messages: list[Message] = []

    def run(self, user_input: str) -> str:
        user_message = Message(
            role="user",
            content=user_input,
        )

        self.messages.append(user_message)


        while True:
            output = self.model.generate(self.messages)

            if isinstance(output, Message):
                self.messages.append(output)
                return output.content

            if isinstance(output, ToolCall):
                result = self.tools.execute(
                    output.name,
                    output.arguments,
                )

                tool_message = Message(
                    role="tool",
                    content=str(result),
                )

                self.messages.append(tool_message)
