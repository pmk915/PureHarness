import json
import re

from dataclasses import dataclass
from typing import Literal, TypedDict


AgentEventType = Literal[
    "agent_started",
    "context_build_started",
    "context_build_failed",
    "context_built",
    "model_started",
    "model_completed",
    "model_failed",
    "tool_policy_evaluated",
    "tool_started",
    "tool_completed",
    "agent_completed",
    "agent_failed",
]


class AgentEventData(TypedDict, total=False):
    step: int
    step_count: int
    reason: str
    error_type: str
    history_item_count: int
    context_item_count: int
    context_strategy: str
    estimated_history_tokens: int
    total_units: int
    included_units: int
    dropped_units: int
    history_token_budget: int
    output_kind: Literal["message", "tool_calls"]
    tool_call_count: int
    name: str
    call_id: str | None
    risk_level: str
    decision: Literal["allow", "deny", "require_approval"]
    arguments_preview: dict[str, str]
    is_error: bool
    duration_seconds: float
    result_character_count: int


@dataclass
class AgentEvent:
    type: AgentEventType
    data: AgentEventData


_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(password|secret|token|api[_-]?key|authorization)\b"
    r"\s*[:=]\s*[^,;\n]+"
)


def safe_arguments_preview(
    arguments: dict[str, object],
    *,
    max_items: int = 8,
    max_value_length: int = 120,
) -> dict[str, str]:
    """Return deterministic, bounded, redacted tool argument previews."""
    preview: dict[str, str] = {}

    items = sorted(
        arguments.items(),
        key=lambda item: str(item[0]),
    )

    for key, value in items[:max_items]:
        key_text = str(key)

        if _is_sensitive_key(key_text):
            value_text = _REDACTED
        else:
            value_text = _format_preview_value(value)

        preview[_truncate(key_text, 40)] = _truncate(
            value_text,
            max_value_length,
        )

    omitted_count = len(items) - max_items

    if omitted_count > 0:
        preview["..."] = f"{omitted_count} more argument(s)"

    return preview


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")

    return any(
        part in normalized
        for part in _SENSITIVE_KEY_PARTS
    )


def _redact_nested(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): (
                _REDACTED
                if _is_sensitive_key(str(key))
                else _redact_nested(item)
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _redact_nested(item)
            for item in value
        ]

    return value


def _format_preview_value(value: object) -> str:
    redacted = _redact_nested(value)

    if isinstance(redacted, str):
        text = redacted
    else:
        try:
            text = json.dumps(
                redacted,
                ensure_ascii=False,
                sort_keys=True,
            )
        except TypeError:
            text = str(redacted)

    return _SENSITIVE_ASSIGNMENT.sub(
        lambda match: (
            f"{match.group(1)}={_REDACTED}"
        ),
        text,
    )


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value

    return value[: max_length - 1] + "…"
