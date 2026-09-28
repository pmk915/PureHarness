import json

from types import SimpleNamespace

import pytest

from pureharness.deepseek_model import DeepSeekModel
from pureharness.messages import ToolCall
from pureharness.model import MalformedModelOutputError


class StaticResponses:
    def __init__(self, response) -> None:
        self.response = response

    def create(self, **kwargs):
        return self.response


def _model_with_tool_arguments(arguments):
    model = object.__new__(DeepSeekModel)
    model.model = "test-model"
    model.client = SimpleNamespace(
        responses=StaticResponses(
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
    )
    return model


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
