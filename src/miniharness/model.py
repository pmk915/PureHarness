from typing import Protocol

from miniharness.messages import AgentItem, Message, ToolCall, ToolResult
from miniharness.tools import Tool


ModelOutput = Message | ToolCall


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

        return ToolCall(
            name="add",
            arguments={
                "a": 12,
                "b": 17,
            },
        )