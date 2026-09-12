import json
import os

import openai
from openai import OpenAI

from miniharness.messages import AgentItem, Message, ToolCall, ToolResult
from miniharness.model import ModelError, ModelOutput
from miniharness.tools import Tool


class DeepSeekModel:
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
    ):
        self.model = model

        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

    def generate(
        self,
        messages: list[AgentItem],
        tools: list[Tool],
    ) -> ModelOutput:
        input_items = [
            self._convert_input_item(item)
            for item in messages
        ]

        tool_schemas = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]

        try:
            response = self.client.responses.create(
                model=self.model,
                input=input_items,
                tools=tool_schemas,
                tool_choice="auto",
                reasoning={
                    "effort": "none",
                },
            )
        except openai.APIError as exc:
            raise ModelError(
                f"DeepSeek request failed: {exc}"
            ) from exc

        for item in response.output:
            if item.type == "function_call":
                return ToolCall(
                    name=item.name,
                    arguments=json.loads(item.arguments),
                    call_id=item.call_id,
                )

        return Message(
            role="assistant",
            content=response.output_text,
        )

    def _convert_input_item(
        self,
        item: AgentItem,
    ) -> dict[str, object]:
        if isinstance(item, Message):
            return {
                "role": item.role,
                "content": item.content,
            }

        if isinstance(item, ToolCall):
            if item.call_id is None:
                raise ValueError(
                    "ToolCall must have a call_id when using DeepSeek."
                )

            return {
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": json.dumps(item.arguments),
            }

        if isinstance(item, ToolResult):
            if item.call_id is None:
                raise ValueError(
                    "ToolResult must have a call_id when using DeepSeek."
                )

            return {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": item.content,
            }

        raise TypeError(
            f"Unsupported agent item: {type(item)}"
        )