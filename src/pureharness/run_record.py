import json

from dataclasses import dataclass
from typing import ClassVar

from pureharness.trace import RUN_END_REASONS, RunEndReason, RunTrace


RUN_RECORD_SCHEMA_VERSION = 1


class RunRecordSerializationError(ValueError):
    """Raised when serialized RunRecord data is unsupported or invalid."""


@dataclass(frozen=True)
class ModelInvocationRecord:
    step: int
    context_strategy: str
    estimated_history_tokens: int
    estimated_task_state_tokens: int
    registered_tool_count: int
    exposed_tool_count: int
    estimated_tool_schema_tokens: int
    selector_strategy: str
    trajectory_compacted: bool
    compacted_source_units: int
    compacted_tool_results: int

    def __post_init__(self) -> None:
        for name in (
            "step",
            "estimated_history_tokens",
            "estimated_task_state_tokens",
            "registered_tool_count",
            "exposed_tool_count",
            "estimated_tool_schema_tokens",
            "compacted_source_units",
            "compacted_tool_results",
        ):
            _validate_non_negative_int(name, getattr(self, name))

        if self.exposed_tool_count > self.registered_tool_count:
            raise ValueError(
                "exposed_tool_count cannot exceed registered_tool_count"
            )

        _validate_non_empty_text(
            "context_strategy",
            self.context_strategy,
        )
        _validate_non_empty_text(
            "selector_strategy",
            self.selector_strategy,
        )

        if not isinstance(self.trajectory_compacted, bool):
            raise ValueError("trajectory_compacted must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "context_strategy": self.context_strategy,
            "estimated_history_tokens": self.estimated_history_tokens,
            "estimated_task_state_tokens": (
                self.estimated_task_state_tokens
            ),
            "registered_tool_count": self.registered_tool_count,
            "exposed_tool_count": self.exposed_tool_count,
            "estimated_tool_schema_tokens": (
                self.estimated_tool_schema_tokens
            ),
            "selector_strategy": self.selector_strategy,
            "trajectory_compacted": self.trajectory_compacted,
            "compacted_source_units": self.compacted_source_units,
            "compacted_tool_results": self.compacted_tool_results,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "ModelInvocationRecord":
        try:
            trajectory_compacted = data["trajectory_compacted"]
            if not isinstance(trajectory_compacted, bool):
                raise RunRecordSerializationError(
                    "trajectory_compacted must be bool"
                )

            return cls(
                step=_require_non_negative_int(data, "step"),
                context_strategy=_require_string(
                    data,
                    "context_strategy",
                ),
                estimated_history_tokens=_require_non_negative_int(
                    data,
                    "estimated_history_tokens",
                ),
                estimated_task_state_tokens=_require_non_negative_int(
                    data,
                    "estimated_task_state_tokens",
                ),
                registered_tool_count=_require_non_negative_int(
                    data,
                    "registered_tool_count",
                ),
                exposed_tool_count=_require_non_negative_int(
                    data,
                    "exposed_tool_count",
                ),
                estimated_tool_schema_tokens=(
                    _require_non_negative_int(
                        data,
                        "estimated_tool_schema_tokens",
                    )
                ),
                selector_strategy=_require_string(
                    data,
                    "selector_strategy",
                ),
                trajectory_compacted=trajectory_compacted,
                compacted_source_units=_require_non_negative_int(
                    data,
                    "compacted_source_units",
                ),
                compacted_tool_results=_require_non_negative_int(
                    data,
                    "compacted_tool_results",
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RunRecordSerializationError):
                raise
            raise RunRecordSerializationError(
                f"Invalid model invocation record: {exc}"
            ) from exc


@dataclass(frozen=True)
class RunRecord:
    schema_version: ClassVar[int] = RUN_RECORD_SCHEMA_VERSION

    run_id: str
    session_id: str | None
    end_reason: RunEndReason
    trace: RunTrace
    model_invocations: tuple[ModelInvocationRecord, ...]
    model_call_count: int
    tool_call_count: int
    tool_execution_count: int
    tool_result_error_count: int
    step_count: int
    sum_estimated_history_tokens: int
    sum_estimated_task_state_tokens: int
    sum_estimated_tool_schema_tokens: int
    trajectory_compaction_count: int
    compacted_source_unit_count: int
    tool_result_compaction_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", self.trace.snapshot())
        _validate_non_empty_text("run_id", self.run_id)
        if self.session_id is not None:
            _validate_non_empty_text("session_id", self.session_id)

        if self.end_reason not in RUN_END_REASONS:
            raise ValueError(
                f"Unsupported run end reason: {self.end_reason!r}"
            )
        if self.trace.end_reason != self.end_reason:
            raise ValueError(
                "RunRecord end_reason must match RunTrace end_reason"
            )

        invocations = tuple(self.model_invocations)
        object.__setattr__(self, "model_invocations", invocations)
        if any(
            invocation.step != expected
            for expected, invocation in enumerate(invocations)
        ):
            raise ValueError(
                "Model invocation steps must be contiguous from zero"
            )

        expected_values = {
            "model_call_count": len(invocations),
            "step_count": len(self.trace.steps),
            "sum_estimated_history_tokens": sum(
                item.estimated_history_tokens
                for item in invocations
            ),
            "sum_estimated_task_state_tokens": sum(
                item.estimated_task_state_tokens
                for item in invocations
            ),
            "sum_estimated_tool_schema_tokens": sum(
                item.estimated_tool_schema_tokens
                for item in invocations
            ),
            "trajectory_compaction_count": sum(
                item.trajectory_compacted
                for item in invocations
            ),
            "compacted_source_unit_count": sum(
                item.compacted_source_units
                for item in invocations
            ),
            "tool_result_compaction_count": sum(
                item.compacted_tool_results
                for item in invocations
            ),
        }

        for name in (
            "model_call_count",
            "tool_call_count",
            "tool_execution_count",
            "tool_result_error_count",
            "step_count",
            "sum_estimated_history_tokens",
            "sum_estimated_task_state_tokens",
            "sum_estimated_tool_schema_tokens",
            "trajectory_compaction_count",
            "compacted_source_unit_count",
            "tool_result_compaction_count",
        ):
            _validate_non_negative_int(name, getattr(self, name))

        for name, expected in expected_values.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} does not match recorded runtime facts"
                )

        if self.tool_execution_count > self.tool_call_count:
            raise ValueError(
                "tool_execution_count cannot exceed tool_call_count"
            )
        if self.tool_result_error_count > self.tool_call_count:
            raise ValueError(
                "tool_result_error_count cannot exceed tool_call_count"
            )

        trace_steps = {step.index for step in self.trace.steps}
        invocation_steps = {item.step for item in invocations}
        if not trace_steps.issubset(invocation_steps):
            raise ValueError(
                "Every trace step must have a model invocation record"
            )
        if not {
            approval.step for approval in self.trace.approvals
        }.issubset(invocation_steps):
            raise ValueError(
                "Every approval must have a model invocation record"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "end_reason": self.end_reason,
            "trace": self.trace.to_dict(),
            "model_invocations": [
                item.to_dict()
                for item in self.model_invocations
            ],
            "model_call_count": self.model_call_count,
            "tool_call_count": self.tool_call_count,
            "tool_execution_count": self.tool_execution_count,
            "tool_result_error_count": self.tool_result_error_count,
            "step_count": self.step_count,
            "sum_estimated_history_tokens": (
                self.sum_estimated_history_tokens
            ),
            "sum_estimated_task_state_tokens": (
                self.sum_estimated_task_state_tokens
            ),
            "sum_estimated_tool_schema_tokens": (
                self.sum_estimated_tool_schema_tokens
            ),
            "trajectory_compaction_count": (
                self.trajectory_compaction_count
            ),
            "compacted_source_unit_count": (
                self.compacted_source_unit_count
            ),
            "tool_result_compaction_count": (
                self.tool_result_compaction_count
            ),
        }

    def to_json(self) -> str:
        try:
            return json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise RunRecordSerializationError(
                f"RunRecord is not JSON serializable: {exc}"
            ) from exc

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RunRecord":
        try:
            schema_version = data["schema_version"]
            if type(schema_version) is not int:
                raise RunRecordSerializationError(
                    "RunRecord schema version must be an integer"
                )
            if schema_version != RUN_RECORD_SCHEMA_VERSION:
                raise RunRecordSerializationError(
                    "Unsupported RunRecord schema version: "
                    f"{schema_version}"
                )

            session_id = data["session_id"]
            if session_id is not None and not isinstance(session_id, str):
                raise RunRecordSerializationError(
                    "session_id must be text or null"
                )

            end_reason = _require_string(data, "end_reason")
            if end_reason not in RUN_END_REASONS:
                raise RunRecordSerializationError(
                    f"Unsupported run end reason: {end_reason!r}"
                )

            invocations_data = data["model_invocations"]
            if not isinstance(invocations_data, list) or not all(
                isinstance(item, dict) for item in invocations_data
            ):
                raise RunRecordSerializationError(
                    "model_invocations must be a list of objects"
                )

            trace_data = data["trace"]
            if not isinstance(trace_data, dict):
                raise RunRecordSerializationError(
                    "trace must be an object"
                )

            return cls(
                run_id=_require_string(data, "run_id"),
                session_id=session_id,
                end_reason=end_reason,
                trace=RunTrace.from_dict(trace_data),
                model_invocations=tuple(
                    ModelInvocationRecord.from_dict(item)
                    for item in invocations_data
                ),
                model_call_count=_require_non_negative_int(
                    data,
                    "model_call_count",
                ),
                tool_call_count=_require_non_negative_int(
                    data,
                    "tool_call_count",
                ),
                tool_execution_count=_require_non_negative_int(
                    data,
                    "tool_execution_count",
                ),
                tool_result_error_count=_require_non_negative_int(
                    data,
                    "tool_result_error_count",
                ),
                step_count=_require_non_negative_int(
                    data,
                    "step_count",
                ),
                sum_estimated_history_tokens=(
                    _require_non_negative_int(
                        data,
                        "sum_estimated_history_tokens",
                    )
                ),
                sum_estimated_task_state_tokens=(
                    _require_non_negative_int(
                        data,
                        "sum_estimated_task_state_tokens",
                    )
                ),
                sum_estimated_tool_schema_tokens=(
                    _require_non_negative_int(
                        data,
                        "sum_estimated_tool_schema_tokens",
                    )
                ),
                trajectory_compaction_count=(
                    _require_non_negative_int(
                        data,
                        "trajectory_compaction_count",
                    )
                ),
                compacted_source_unit_count=(
                    _require_non_negative_int(
                        data,
                        "compacted_source_unit_count",
                    )
                ),
                tool_result_compaction_count=(
                    _require_non_negative_int(
                        data,
                        "tool_result_compaction_count",
                    )
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RunRecordSerializationError):
                raise
            raise RunRecordSerializationError(
                f"Invalid RunRecord data: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, value: str) -> "RunRecord":
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RunRecordSerializationError(
                f"Malformed RunRecord JSON: {exc.msg}"
            ) from exc

        if not isinstance(data, dict):
            raise RunRecordSerializationError(
                "RunRecord JSON must contain an object"
            )

        return cls.from_dict(data)


class RunRecordBuilder:
    """Collect factual metrics for exactly one Agent.run() invocation."""

    def __init__(self, run_id: str, session_id: str | None) -> None:
        _validate_non_empty_text("run_id", run_id)
        if session_id is not None:
            _validate_non_empty_text("session_id", session_id)

        self.run_id = run_id
        self.session_id = session_id
        self.model_invocations: list[ModelInvocationRecord] = []
        self.tool_call_count = 0
        self.tool_execution_count = 0
        self.tool_result_error_count = 0

    def record_model_invocation(
        self,
        invocation: ModelInvocationRecord,
    ) -> None:
        if invocation.step != len(self.model_invocations):
            raise ValueError(
                "Model invocation steps must be recorded in order"
            )
        self.model_invocations.append(invocation)

    def record_tool_calls(self, count: int) -> None:
        _validate_non_negative_int("count", count)
        self.tool_call_count += count

    def record_tool_execution(self) -> None:
        self.tool_execution_count += 1

    def record_tool_result(self, *, is_error: bool) -> None:
        if not isinstance(is_error, bool):
            raise ValueError("is_error must be bool")
        if is_error:
            self.tool_result_error_count += 1

    def finalize(self, trace: RunTrace) -> RunRecord:
        if trace.end_reason is None:
            raise ValueError(
                "Cannot finalize RunRecord without an end reason"
            )

        invocations = tuple(self.model_invocations)
        return RunRecord(
            run_id=self.run_id,
            session_id=self.session_id,
            end_reason=trace.end_reason,
            trace=trace.snapshot(),
            model_invocations=invocations,
            model_call_count=len(invocations),
            tool_call_count=self.tool_call_count,
            tool_execution_count=self.tool_execution_count,
            tool_result_error_count=self.tool_result_error_count,
            step_count=len(trace.steps),
            sum_estimated_history_tokens=sum(
                item.estimated_history_tokens
                for item in invocations
            ),
            sum_estimated_task_state_tokens=sum(
                item.estimated_task_state_tokens
                for item in invocations
            ),
            sum_estimated_tool_schema_tokens=sum(
                item.estimated_tool_schema_tokens
                for item in invocations
            ),
            trajectory_compaction_count=sum(
                item.trajectory_compacted
                for item in invocations
            ),
            compacted_source_unit_count=sum(
                item.compacted_source_units
                for item in invocations
            ),
            tool_result_compaction_count=sum(
                item.compacted_tool_results
                for item in invocations
            ),
        )


def _validate_non_negative_int(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_non_empty_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")


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
        raise RunRecordSerializationError(
            f"{key} must be a non-negative integer"
        )
    return value


def _require_string(data: dict[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise RunRecordSerializationError(f"{key} must be text")
    return value
