import json
import os
import re
from collections.abc import Mapping

import openai
from openai import OpenAI

from pureharness.messages import AgentItem, Message, ToolCall, ToolResult
from pureharness.model import (
    ContextWindowExceededError,
    MalformedModelOutputError,
    ModelError,
    ModelOutput,
)
from pureharness.tools import Tool


class DeepSeekModel:
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
    ):
        self.model = model
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ModelError("DEEPSEEK_API_KEY is not set.")

        self.client = OpenAI(
            api_key=api_key,
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
        except openai.BadRequestError as exc:
            if _is_context_window_exceeded(exc):
                raise ContextWindowExceededError(
                    "DeepSeek rejected the request because its context "
                    "window was exceeded."
                ) from exc
            raise ModelError(
                f"DeepSeek request failed: {exc}"
            ) from exc
        except openai.APIError as exc:
            raise ModelError(
                f"DeepSeek request failed: {exc}"
            ) from exc


        tool_calls = []

        for item in response.output:

            if item.type == "function_call":

                try:
                    arguments = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise MalformedModelOutputError(
                        "DeepSeek returned malformed tool arguments."
                    ) from exc
                if not isinstance(arguments, dict):
                    raise MalformedModelOutputError(
                        "DeepSeek tool arguments must be a JSON object."
                    )

                tool_calls.append(
                    ToolCall(
                        name=item.name,
                        arguments=arguments,
                        call_id=item.call_id,
                    )
                )

        if tool_calls:
            return tool_calls

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


_CONTEXT_WINDOW_CODES = frozenset(
    {"context_length_exceeded", "context_window_exceeded"}
)
_DEEPSEEK_INPUT_LIMIT_MESSAGE = re.compile(
    r"Input token exceed the limit"
    r"(?: \(request id: [^)]+\))?\.?",
    re.IGNORECASE,
)
_MAX_CONTEXT_LENGTH_MESSAGE = re.compile(
    r"This model's maximum context length is \d+ tokens\. "
    r"However, you requested (?:about )?\d+ tokens"
    r"(?: \([^)]*\))?\. Please reduce the length of "
    r"(?:either )?(?:the )?messages or completion\.?",
    re.IGNORECASE,
)


def _is_context_window_exceeded(
    error: openai.BadRequestError,
) -> bool:
    body = error.body
    if not isinstance(body, Mapping):
        return False

    code = body.get("code")
    error_type = body.get("type")
    if code in _CONTEXT_WINDOW_CODES or error_type in _CONTEXT_WINDOW_CODES:
        return True

    message = body.get("message")
    deepseek_input_limit = (
        code == "quota_limit_reached"
        and error_type == "api_error"
        and body.get("param") in {None, ""}
        and isinstance(message, str)
        and _DEEPSEEK_INPUT_LIMIT_MESSAGE.fullmatch(message.strip())
        is not None
    )
    maximum_context_length = (
        code in {None, "invalid_request_error"}
        and error_type == "invalid_request_error"
        and body.get("param") in {None, ""}
        and isinstance(message, str)
        and _MAX_CONTEXT_LENGTH_MESSAGE.fullmatch(message.strip())
        is not None
    )
    return deepseek_input_limit or maximum_context_length
