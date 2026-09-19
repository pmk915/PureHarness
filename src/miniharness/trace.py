from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from miniharness.approval import ApprovalDecision
from miniharness.messages import Message, ToolCall, ToolResult


RunEndReason = Literal[
    "completed",
    "max_steps_exceeded",
    "model_error",
    "context_error",
    "tool_selection_error",
    "interrupted",
]

RUN_END_REASONS = frozenset(
    {
        "completed",
        "max_steps_exceeded",
        "model_error",
        "context_error",
        "tool_selection_error",
        "interrupted",
    }
)


class RunTraceSerializationError(ValueError):
    """Raised when serialized RunTrace data is invalid."""


@dataclass(frozen=True)
class ApprovalTrace:
    step: int
    tool_name: str
    call_id: str | None
    decision: ApprovalDecision

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise ValueError("ApprovalTrace step must be non-negative")
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("ApprovalTrace tool_name must be non-empty")
        if self.call_id is not None and not isinstance(self.call_id, str):
            raise ValueError("ApprovalTrace call_id must be text or null")
        if not isinstance(self.decision, ApprovalDecision):
            raise ValueError(
                "ApprovalTrace decision must be ApprovalDecision"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "decision": self.decision.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ApprovalTrace":
        try:
            step = data["step"]
            if (
                not isinstance(step, int)
                or isinstance(step, bool)
                or step < 0
            ):
                raise ValueError("step must be non-negative")
            tool_name = _require_string(data, "tool_name")
            call_id = data["call_id"]
            if call_id is not None and not isinstance(call_id, str):
                raise ValueError("call_id must be text or null")
            decision = ApprovalDecision(
                _require_string(data, "decision")
            )
            return cls(
                step=step,
                tool_name=tool_name,
                call_id=call_id,
                decision=decision,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunTraceSerializationError(
                f"Invalid ApprovalTrace data: {exc}"
            ) from exc


@dataclass
class StepTrace:
    index: int
    output: Message | list[ToolCall]
    tool_result: list[ToolResult] | None = None

    def to_dict(self) -> dict[str, object]:
        if isinstance(self.output, Message):
            output = {
                "kind": "message",
                "item": _serialize_message(self.output),
            }
        else:
            output = {
                "kind": "tool_calls",
                "items": [
                    _serialize_tool_call(item)
                    for item in self.output
                ],
            }

        return {
            "index": self.index,
            "output": output,
            "tool_results": (
                None
                if self.tool_result is None
                else [
                    _serialize_tool_result(item)
                    for item in self.tool_result
                ]
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StepTrace":
        try:
            index = _require_non_negative_int(data, "index")
            output_data = _require_dict(data, "output")
            kind = _require_string(output_data, "kind")

            if kind == "message":
                output: Message | list[ToolCall] = _deserialize_message(
                    _require_dict(output_data, "item")
                )
            elif kind == "tool_calls":
                output = [
                    _deserialize_tool_call(item)
                    for item in _require_dict_list(output_data, "items")
                ]
            else:
                raise RunTraceSerializationError(
                    f"Unsupported trace output kind: {kind!r}."
                )

            results_data = data.get("tool_results")
            if results_data is None:
                results = None
            elif isinstance(results_data, list):
                results = [
                    _deserialize_tool_result(item)
                    for item in results_data
                    if isinstance(item, dict)
                ]
                if len(results) != len(results_data):
                    raise RunTraceSerializationError(
                        "Trace tool_results must contain objects."
                    )
            else:
                raise RunTraceSerializationError(
                    "Trace tool_results must be a list or null."
                )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RunTraceSerializationError):
                raise
            raise RunTraceSerializationError(
                f"Invalid StepTrace data: {exc}"
            ) from exc

        if isinstance(output, Message) and results is not None:
            raise RunTraceSerializationError(
                "Message trace output cannot have tool results."
            )
        if isinstance(output, list) and (
            results is None or len(output) != len(results)
        ):
            raise RunTraceSerializationError(
                "ToolCall trace output must have one result per call."
            )

        return cls(
            index=index,
            output=output,
            tool_result=results,
        )


@dataclass
class RunTrace:
    steps: list[StepTrace] = field(default_factory=list)
    approvals: list[ApprovalTrace] = field(default_factory=list)
    end_reason: RunEndReason | None = None

    def snapshot(self) -> "RunTrace":
        return deepcopy(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "end_reason": self.end_reason,
            "steps": [step.to_dict() for step in self.steps],
            "approvals": [
                approval.to_dict()
                for approval in self.approvals
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RunTrace":
        try:
            end_reason = data.get("end_reason")
            if (
                end_reason is not None
                and end_reason not in RUN_END_REASONS
            ):
                raise RunTraceSerializationError(
                    f"Unsupported run end reason: {end_reason!r}."
                )

            steps_data = data["steps"]
            if not isinstance(steps_data, list):
                raise RunTraceSerializationError(
                    "RunTrace steps must be a list."
                )

            steps = [
                StepTrace.from_dict(item)
                for item in steps_data
                if isinstance(item, dict)
            ]
            if len(steps) != len(steps_data):
                raise RunTraceSerializationError(
                    "RunTrace steps must contain objects."
                )

            approvals_data = data.get("approvals", [])
            if not isinstance(approvals_data, list) or not all(
                isinstance(item, dict) for item in approvals_data
            ):
                raise RunTraceSerializationError(
                    "RunTrace approvals must be a list of objects."
                )
            approvals = [
                ApprovalTrace.from_dict(item)
                for item in approvals_data
            ]
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RunTraceSerializationError):
                raise
            raise RunTraceSerializationError(
                f"Invalid RunTrace data: {exc}"
            ) from exc

        if any(
            step.index != expected
            for expected, step in enumerate(steps)
        ):
            raise RunTraceSerializationError(
                "RunTrace step indexes must be contiguous from zero."
            )

        return cls(
            steps=steps,
            approvals=approvals,
            end_reason=end_reason,
        )


def _serialize_message(message: Message) -> dict[str, object]:
    return {
        "role": message.role,
        "content": message.content,
    }


def _serialize_tool_call(call: ToolCall) -> dict[str, object]:
    return {
        "name": call.name,
        "arguments": deepcopy(call.arguments),
        "call_id": call.call_id,
    }


def _serialize_tool_result(result: ToolResult) -> dict[str, object]:
    return {
        "name": result.name,
        "content": result.content,
        "call_id": result.call_id,
        "is_error": result.is_error,
    }


def _deserialize_message(data: dict[str, object]) -> Message:
    return Message(
        role=_require_string(data, "role"),
        content=_require_string(data, "content"),
    )


def _deserialize_tool_call(data: dict[str, object]) -> ToolCall:
    arguments = data["arguments"]
    if not isinstance(arguments, dict):
        raise RunTraceSerializationError(
            "ToolCall arguments must be an object."
        )

    return ToolCall(
        name=_require_string(data, "name"),
        arguments=deepcopy(arguments),
        call_id=_optional_string(data, "call_id"),
    )


def _deserialize_tool_result(data: dict[str, object]) -> ToolResult:
    is_error = data["is_error"]
    if not isinstance(is_error, bool):
        raise RunTraceSerializationError(
            "ToolResult is_error must be bool."
        )

    return ToolResult(
        name=_require_string(data, "name"),
        content=_require_string(data, "content"),
        call_id=_optional_string(data, "call_id"),
        is_error=is_error,
    )


def _require_dict(
    data: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = data[key]
    if not isinstance(value, dict):
        raise RunTraceSerializationError(f"{key} must be an object.")
    return value


def _require_dict_list(
    data: dict[str, object],
    key: str,
) -> list[dict[str, object]]:
    value = data[key]
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise RunTraceSerializationError(
            f"{key} must be a list of objects."
        )
    return value


def _require_string(data: dict[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise RunTraceSerializationError(f"{key} must be text.")
    return value


def _optional_string(
    data: dict[str, object],
    key: str,
) -> str | None:
    value = data[key]
    if value is not None and not isinstance(value, str):
        raise RunTraceSerializationError(
            f"{key} must be text or null."
        )
    return value


def _require_non_negative_int(
    data: dict[str, object],
    key: str,
) -> int:
    value = data[key]
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise RunTraceSerializationError(
            f"{key} must be a non-negative integer."
        )
    return value
