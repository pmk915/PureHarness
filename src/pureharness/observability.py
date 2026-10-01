import json
import math

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum

from pureharness.events import AgentEvent
from pureharness.run_record import RunRecord
from pureharness.session_store import DurableSessionSummary


EVENT_WIRE_SCHEMA_VERSION = 1
SESSION_LIST_SCHEMA_VERSION = 1


class EventSerializationError(ValueError):
    """Raised when a runtime event cannot enter the stable wire format."""


_EVENT_PAYLOAD_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "agent_started": (("history_item_count", "history_item_count"),),
    "context_build_started": (
        ("history_item_count", "history_item_count"),
    ),
    "context_build_failed": (
        ("reason", "reason"),
        ("error_type", "error_type"),
    ),
    "context_built": (
        ("history_item_count", "history_item_count"),
        ("context_item_count", "context_item_count"),
        ("trajectory_item_count", "trajectory_item_count"),
        ("context_strategy", "context_strategy"),
        ("estimated_history_tokens", "estimated_history_tokens"),
        ("active_skill_count", "active_skill_count"),
        ("active_skill_ids", "active_skill_ids"),
        ("estimated_skill_tokens", "estimated_skill_tokens"),
        ("estimated_task_state_tokens", "estimated_task_state_tokens"),
        ("completion_recheck_present", "completion_recheck_present"),
        (
            "estimated_completion_recheck_tokens",
            "estimated_completion_recheck_tokens",
        ),
        ("context_window_tokens", "context_window_tokens"),
        ("reserved_output_tokens", "reserved_output_tokens"),
        ("usable_input_tokens", "usable_input_tokens"),
        ("estimated_request_tokens", "estimated_request_tokens"),
        (
            "context_pressure_detected",
            "context_pressure_detected",
        ),
        ("available_history_tokens", "available_history_tokens"),
        ("bounded_history_applied", "bounded_history_applied"),
        ("current_request_present", "current_request_present"),
        ("completed_actions_count", "completed_actions_count"),
        ("failed_actions_count", "failed_actions_count"),
        ("files_read_count", "files_read_count"),
        ("files_modified_count", "files_modified_count"),
        ("recent_errors_count", "recent_errors_count"),
        ("total_units", "total_units"),
        ("included_units", "included_units"),
        ("dropped_units", "dropped_units"),
        ("history_token_budget", "history_token_budget"),
        ("projected_tool_results", "projected_tool_results"),
        ("compacted_tool_results", "compacted_tool_results"),
        ("raw_tool_result_chars", "raw_tool_result_chars"),
        ("projected_tool_result_chars", "projected_tool_result_chars"),
        ("trajectory_compacted", "trajectory_compacted"),
        ("compacted_source_units", "compacted_source_units"),
        ("compacted_tool_actions", "compacted_tool_actions"),
        (
            "original_trajectory_estimated_tokens",
            "original_trajectory_estimated_tokens",
        ),
        (
            "compacted_trajectory_estimated_tokens",
            "compacted_trajectory_estimated_tokens",
        ),
        ("recent_raw_units", "recent_raw_units"),
        ("recent_raw_estimated_tokens", "recent_raw_estimated_tokens"),
        (
            "trajectory_compaction_strategy",
            "trajectory_compaction_strategy",
        ),
    ),
    "context_recovering": (
        ("recovery_attempt", "recovery_attempt"),
        ("max_recoveries", "max_recoveries"),
        ("error_type", "error_type"),
        ("previous_history_tokens", "previous_history_tokens"),
        ("recovery_history_budget", "recovery_history_budget"),
        ("recovered_history_tokens", "recovered_history_tokens"),
        (
            "previous_estimated_request_tokens",
            "previous_estimated_request_tokens",
        ),
        (
            "recovered_estimated_request_tokens",
            "recovered_estimated_request_tokens",
        ),
    ),
    "context_window_exceeded": (
        ("error_type", "error_type"),
        (
            "context_recovery_available",
            "context_recovery_available",
        ),
        ("recovery_attempt", "recovery_attempt"),
        ("max_context_recoveries", "max_context_recoveries"),
    ),
    "completion_recheck_requested": (
        ("reason", "reason"),
        ("recheck_number", "recheck_number"),
        ("max_rechecks", "max_rechecks"),
        ("workspace_mutations", "workspace_mutations"),
        ("command_executions", "command_executions"),
        ("process_starts", "process_starts"),
        (
            "executions_since_last_mutation",
            "executions_since_last_mutation",
        ),
        ("verification_outcome", "verification_outcome"),
        ("verification_exit_code", "verification_exit_code"),
    ),
    "completion_recheck_skipped": (
        ("reason", "reason"),
        ("skip_reason", "skip_reason"),
        ("rechecks_used", "rechecks_used"),
        ("max_rechecks", "max_rechecks"),
        ("workspace_mutations", "workspace_mutations"),
        ("command_executions", "command_executions"),
        ("process_starts", "process_starts"),
        (
            "executions_since_last_mutation",
            "executions_since_last_mutation",
        ),
        ("verification_outcome", "verification_outcome"),
        ("verification_exit_code", "verification_exit_code"),
    ),
    "execution_budget_exhausted": (
        ("resource", "resource"),
        ("used", "used"),
        ("limit", "limit"),
        ("requested", "requested"),
        ("remaining", "remaining"),
    ),
    "progress_snapshot": (
        ("logical_steps_completed", "logical_steps_completed"),
        ("model_attempts", "model_attempts"),
        ("tool_calls", "tool_calls"),
        ("successful_tool_results", "successful_tool_results"),
        ("failed_tool_results", "failed_tool_results"),
        ("unique_tool_actions", "unique_tool_actions"),
        ("repeated_tool_actions", "repeated_tool_actions"),
        (
            "max_identical_tool_action_count",
            "max_identical_tool_action_count",
        ),
        ("model_retries", "model_retries"),
        ("context_recoveries", "context_recoveries"),
        ("context_pressure_count", "context_pressure_count"),
        (
            "context_window_exceeded_count",
            "context_window_exceeded_count",
        ),
        ("terminal", "terminal"),
    ),
    "coding_evidence_snapshot": (
        ("workspace_mutations", "workspace_mutations"),
        ("command_executions", "command_executions"),
        ("command_tool_errors", "command_tool_errors"),
        ("process_starts", "process_starts"),
        ("process_polls", "process_polls"),
        ("process_stops", "process_stops"),
        ("process_tool_errors", "process_tool_errors"),
        (
            "executions_since_last_mutation",
            "executions_since_last_mutation",
        ),
        ("verification_attempts", "verification_attempts"),
        ("verification_exit_zero", "verification_exit_zero"),
        ("verification_exit_nonzero", "verification_exit_nonzero"),
        ("verification_tool_errors", "verification_tool_errors"),
        (
            "verifications_since_last_mutation",
            "verifications_since_last_mutation",
        ),
        ("last_mutation_step", "last_mutation_step"),
        ("last_execution_step", "last_execution_step"),
        ("last_verification_outcome", "last_verification_outcome"),
        ("last_verification_exit_code", "last_verification_exit_code"),
        ("last_verification_step", "last_verification_step"),
        ("terminal", "terminal"),
    ),
    "model_started": (
        ("registered_tool_count", "registered_tool_count"),
        ("exposed_tool_count", "exposed_tool_count"),
        ("estimated_tool_schema_tokens", "estimated_tool_schema_tokens"),
        (
            "estimated_all_tool_schema_tokens",
            "estimated_all_tool_schema_tokens",
        ),
        (
            "estimated_tool_schema_tokens_saved",
            "estimated_tool_schema_tokens_saved",
        ),
        ("selector_strategy", "selector_strategy"),
    ),
    "model_retrying": (
        ("attempt", "attempt"),
        ("max_attempts", "max_attempts"),
        ("error_type", "error_type"),
        ("failure_category", "failure_category"),
    ),
    "model_completed": (
        ("output_kind", "output_kind"),
        ("tool_call_count", "tool_call_count"),
    ),
    "model_failed": (
        ("reason", "reason"),
        ("error_type", "error_type"),
    ),
    "tool_policy_evaluated": (
        ("name", "tool_name"),
        ("call_id", "call_id"),
        ("risk_level", "risk_level"),
        ("decision", "decision"),
    ),
    "workspace_precondition_failed": (
        ("name", "tool_name"),
        ("call_id", "call_id"),
        ("path", "path"),
        ("reason", "reason"),
        ("change", "change"),
    ),
    "workspace_mutated": (
        ("name", "tool_name"),
        ("call_id", "call_id"),
        ("path", "path"),
        ("operation", "operation"),
    ),
    "approval_requested": (
        ("name", "tool_name"),
        ("call_id", "call_id"),
        ("arguments_preview", "arguments_preview"),
    ),
    "approval_granted": (
        ("name", "tool_name"),
        ("call_id", "call_id"),
        ("approval_decision", "decision"),
    ),
    "approval_denied": (
        ("name", "tool_name"),
        ("call_id", "call_id"),
        ("approval_decision", "decision"),
    ),
    "tool_started": (
        ("name", "tool_name"),
        ("call_id", "call_id"),
        ("arguments_preview", "arguments_preview"),
    ),
    "tool_completed": (
        ("name", "tool_name"),
        ("call_id", "call_id"),
        ("is_error", "is_error"),
        ("duration_seconds", "duration_seconds"),
        ("result_character_count", "result_character_count"),
    ),
    "agent_completed": (
        ("reason", "end_reason"),
        ("step_count", "step_count"),
    ),
    "agent_interrupted": (
        ("reason", "end_reason"),
        ("step_count", "step_count"),
    ),
    "agent_failed": (
        ("reason", "end_reason"),
        ("error_type", "error_type"),
        ("step_count", "step_count"),
    ),
}
SUPPORTED_EVENT_TYPES = frozenset(_EVENT_PAYLOAD_FIELDS)


def event_to_wire(event: AgentEvent) -> dict[str, object]:
    fields = _EVENT_PAYLOAD_FIELDS.get(event.type)
    if fields is None:
        raise EventSerializationError(
            f"Unsupported runtime event: {event.type!r}"
        )
    if not isinstance(event.run_id, str) or not event.run_id:
        raise EventSerializationError(
            f"Runtime event {event.type!r} is missing run_id"
        )

    wire: dict[str, object] = {
        "schema_version": EVENT_WIRE_SCHEMA_VERSION,
        "event": event.type,
        "timestamp": _format_timestamp(event.timestamp),
        "run_id": event.run_id,
    }
    if event.session_id is not None:
        if not isinstance(event.session_id, str) or not event.session_id:
            raise EventSerializationError("session_id must be non-empty text")
        wire["session_id"] = event.session_id
    if "step" in event.data:
        wire["step"] = _to_json_value(event.data["step"])

    payload = {
        wire_name: _to_json_value(event.data[source_name])
        for source_name, wire_name in fields
        if source_name in event.data
    }
    wire["payload"] = payload
    return wire


def run_record_to_wire(record: RunRecord) -> dict[str, object]:
    return record.to_dict()


def session_summaries_to_wire(
    summaries: Sequence[DurableSessionSummary],
) -> dict[str, object]:
    return {
        "schema_version": SESSION_LIST_SCHEMA_VERSION,
        "sessions": [
            {
                "session_id": summary.session_id,
                "created_at": _format_timestamp(summary.created_at),
                "updated_at": _format_timestamp(summary.updated_at),
                "workspace": str(summary.workspace),
                "model": summary.model,
                "run_count": summary.run_count,
                "last_run_id": summary.last_run_id,
                "last_end_reason": summary.last_end_reason,
            }
            for summary in summaries
        ],
    }


def dumps_wire(value: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EventSerializationError(
            f"Machine output is not JSON serializable: {exc}"
        ) from exc


class JsonlEventRenderer:
    """Write one stable wire event per output line in observation order."""

    def __init__(self, output: Callable[[str], None]) -> None:
        self.output = output

    def __call__(self, event: AgentEvent) -> None:
        self.output(dumps_wire(event_to_wire(event)))


def _format_timestamp(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise EventSerializationError("timestamp must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _to_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EventSerializationError("non-finite float in event payload")
        return value
    if isinstance(value, Enum):
        return _to_json_value(value.value)
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise EventSerializationError(
                "event payload object keys must be text"
            )
        return {
            key: _to_json_value(item)
            for key, item in value.items()
        }
    raise EventSerializationError(
        f"unsupported event payload value: {type(value).__name__}"
    )
