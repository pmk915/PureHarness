from typing import Protocol

from miniharness.messages import Message, ToolCall


ModelOutput = Message | ToolCall


class Model(Protocol):
    def generate(self, messages: list[Message]) -> ModelOutput:
        ...


class EchoModel:
    def generate(self, messages: list[Message]) -> ModelOutput:
        last_message = messages[-1]

        return Message(
            role="assistant",
            content=f"Echo: {last_message.content}",
        )


class AddModel:
    def generate(self, messages: list[Message]) -> ModelOutput:
        last_message = messages[-1]

        if last_message.role == "tool":
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