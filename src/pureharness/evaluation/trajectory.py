from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pureharness.run_record import RunRecord


@dataclass(frozen=True)
class TrajectoryEvent:
    """Small evaluation-facing execution event, independent of AgentEvent."""

    event_type: str
    step: int | None = None
    tool_name: str | None = None
    failed: bool | None = None

    def __post_init__(self) -> None:
        _validate_optional_non_empty_text("tool_name", self.tool_name)
        _validate_non_empty_text("event_type", self.event_type)
        _validate_optional_non_negative_int("step", self.step)
        if self.failed is not None and not isinstance(self.failed, bool):
            raise ValueError("failed must be bool or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "step": self.step,
            "tool_name": self.tool_name,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class Trajectory:
    """Stable factual input for post-run evaluation."""

    run_id: str
    completed: bool
    total_steps: int
    tool_call_count: int
    failed_tool_call_count: int
    task_id: str | None = None
    task_description: str | None = None
    execution_events: tuple[TrajectoryEvent, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_text("run_id", self.run_id)
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be bool")
        for name in (
            "total_steps",
            "tool_call_count",
            "failed_tool_call_count",
        ):
            _validate_non_negative_int(name, getattr(self, name))
        if self.failed_tool_call_count > self.tool_call_count:
            raise ValueError(
                "failed_tool_call_count cannot exceed tool_call_count"
            )
        _validate_optional_non_empty_text("task_id", self.task_id)
        _validate_optional_non_empty_text(
            "task_description",
            self.task_description,
        )
        events = tuple(self.execution_events)
        if not all(isinstance(event, TrajectoryEvent) for event in events):
            raise ValueError(
                "execution_events must contain TrajectoryEvent values"
            )
        object.__setattr__(self, "execution_events", events)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "completed": self.completed,
            "total_steps": self.total_steps,
            "tool_call_count": self.tool_call_count,
            "failed_tool_call_count": self.failed_tool_call_count,
            "execution_events": [
                event.to_dict() for event in self.execution_events
            ],
        }


def trajectory_from_run_record(
    record: "RunRecord",
    *,
    task_id: str | None = None,
    task_description: str | None = None,
    execution_events: tuple[TrajectoryEvent, ...] = (),
) -> Trajectory:
    """Adapt persisted runtime facts without changing the RunRecord model."""

    return Trajectory(
        run_id=record.run_id,
        task_id=task_id,
        task_description=task_description,
        completed=record.end_reason == "completed",
        total_steps=record.step_count,
        tool_call_count=record.tool_call_count,
        failed_tool_call_count=record.tool_result_error_count,
        execution_events=execution_events,
    )


def _validate_non_empty_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")


def _validate_optional_non_empty_text(
    name: str,
    value: str | None,
) -> None:
    if value is not None:
        _validate_non_empty_text(name, value)


def _validate_non_negative_int(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_optional_non_negative_int(
    name: str,
    value: int | None,
) -> None:
    if value is not None:
        _validate_non_negative_int(name, value)
