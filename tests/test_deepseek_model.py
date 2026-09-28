import json

from types import SimpleNamespace

import httpx
import openai
import pytest

from pureharness.deepseek_model import DeepSeekModel
from pureharness.messages import ToolCall
from pureharness.model import (
    ContextWindowExceededError,
    MalformedModelOutputError,
    ModelError,
)


class StaticResponses:
    def __init__(self, response) -> None:
        self.response = response

    def create(self, **kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _model_with_tool_arguments(arguments):
    return _model_with_response(
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="read_file",
                    arguments=arguments,
                    call_id="call-1",
                )
            ],
            output_text="",
        )
    )


def _model_with_response(response):
    model = object.__new__(DeepSeekModel)
    model.model = "test-model"
    model.client = SimpleNamespace(
        responses=StaticResponses(response)
    )
    return model


def _bad_request(body: dict[str, object]) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.deepseek.com/responses")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError(
        "DeepSeek bad request",
        response=response,
        body=body,
    )


def test_deepseek_normalizes_malformed_tool_arguments():
    model = _model_with_tool_arguments("{")

    with pytest.raises(
        MalformedModelOutputError,
        match="malformed tool arguments",
    ) as exc_info:
        model.generate([], [])

    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_deepseek_rejects_non_object_tool_arguments():
    model = _model_with_tool_arguments("[]")

    with pytest.raises(
        MalformedModelOutputError,
        match="must be a JSON object",
    ):
        model.generate([], [])


def test_deepseek_preserves_valid_tool_arguments():
    model = _model_with_tool_arguments('{"path": "README.md"}')

    assert model.generate([], []) == [
        ToolCall(
            name="read_file",
            arguments={"path": "README.md"},
            call_id="call-1",
        )
    ]


@pytest.mark.parametrize(
    "body",
    [
        {
            "message": "request is too large",
            "type": "invalid_request_error",
            "param": None,
            "code": "context_length_exceeded",
        },
        {
            "message": (
                "Input token exceed the limit "
                "(request id: test-request)"
            ),
            "type": "api_error",
            "param": "",
            "code": "quota_limit_reached",
        },
        {
            "message": (
                "This model's maximum context length is 1048576 tokens. "
                "However, you requested 1053235 tokens "
                "(797235 in the messages, 256000 in the completion). "
                "Please reduce the length of the messages or completion."
            ),
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_request_error",
        },
    ],
)
def test_deepseek_normalizes_structured_context_window_errors(body):
    error = _bad_request(body)
    model = _model_with_response(error)

    with pytest.raises(ContextWindowExceededError) as exc_info:
        model.generate([], [])

    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize(
    "body",
    [
        {
            "message": "Invalid tool schema.",
            "type": "invalid_request_error",
            "param": "tools",
            "code": "invalid_request_error",
        },
        {
            "message": "Account quota exceeded.",
            "type": "api_error",
            "param": "",
            "code": "quota_limit_reached",
        },
    ],
)
def test_deepseek_does_not_normalize_unrelated_bad_requests(body):
    error = _bad_request(body)
    model = _model_with_response(error)

    with pytest.raises(ModelError) as exc_info:
        model.generate([], [])

    assert not isinstance(exc_info.value, ContextWindowExceededError)
    assert exc_info.value.__cause__ is error
