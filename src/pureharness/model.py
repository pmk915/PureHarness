from typing import Protocol

from pureharness.messages import AgentItem, Message, ToolCall, ToolResult
from pureharness.tools import Tool


ModelOutput = Message | list[ToolCall]

class ModelError(RuntimeError):
    pass

class Model(Protocol):
    def generate(
        self,
        messages: list[AgentItem],
        tools: list[Tool],
    ) -> ModelOutput:
        ...


class EchoModel:
    def generate(
        self,
        messages: list[AgentItem],
        tools: list[Tool],
    ) -> ModelOutput:
        last_message = messages[-1]

        if not isinstance(last_message, Message):
            raise ValueError("EchoModel expects a Message.")

        return Message(
            role="assistant",
            content=f"Echo: {last_message.content}",
        )


class AddModel:
    def generate(
        self,
        messages: list[AgentItem],
        tools: list[Tool],
    ) -> ModelOutput:
        last_message = messages[-1]

        if isinstance(last_message, ToolResult):
            return Message(
                role="assistant",
                content=f"The result is {last_message.content}",
            )

        return [
            ToolCall(
                name="add",
                arguments={
                    "a":12,
                    "b":17,
                },
                call_id="1",
            )
        ]