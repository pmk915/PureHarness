from dataclasses import dataclass

from miniharness.messages import Message
from miniharness.run_record import RunRecord


ReplayValue = str | int | bool | None


@dataclass(frozen=True)
class ReplayEntry:
    kind: str
    step: int | None
    summary: str
    metadata: tuple[tuple[str, ReplayValue], ...] = ()


def replay_run(record: RunRecord) -> tuple[ReplayEntry, ...]:
    """Build a deterministic, side-effect-free view of a recorded run."""
    entries = [
        ReplayEntry(
            kind="run_started",
            step=None,
            summary="Run started.",
            metadata=(
                ("run_id", record.run_id),
                ("session_id", record.session_id),
            ),
        )
    ]
    trace_steps = {
        step.index: step
        for step in record.trace.steps
    }

    for invocation in record.model_invocations:
        entries.append(
            ReplayEntry(
                kind="model_invocation",
                step=invocation.step,
                summary="Model invocation recorded.",
                metadata=(
                    ("context_strategy", invocation.context_strategy),
                    (
                        "estimated_history_tokens",
                        invocation.estimated_history_tokens,
                    ),
                    (
                        "estimated_task_state_tokens",
                        invocation.estimated_task_state_tokens,
                    ),
                    (
                        "estimated_tool_schema_tokens",
                        invocation.estimated_tool_schema_tokens,
                    ),
                    (
                        "registered_tool_count",
                        invocation.registered_tool_count,
                    ),
                    (
                        "exposed_tool_count",
                        invocation.exposed_tool_count,
                    ),
                    ("selector_strategy", invocation.selector_strategy),
                    (
                        "trajectory_compacted",
                        invocation.trajectory_compacted,
                    ),
                ),
            )
        )

        trace_step = trace_steps.get(invocation.step)
        if trace_step is None:
            continue

        if isinstance(trace_step.output, Message):
            entries.append(
                ReplayEntry(
                    kind="model_message",
                    step=invocation.step,
                    summary="Assistant message recorded.",
                    metadata=(
                        ("role", trace_step.output.role),
                        (
                            "content_character_count",
                            len(trace_step.output.content),
                        ),
                    ),
                )
            )
            continue

        results = trace_step.tool_result or []
        for call, result in zip(
            trace_step.output,
            results,
            strict=True,
        ):
            entries.append(
                ReplayEntry(
                    kind="tool_call",
                    step=invocation.step,
                    summary=f"Tool call recorded: {call.name}.",
                    metadata=(
                        ("name", call.name),
                        ("call_id", call.call_id),
                    ),
                )
            )
            entries.append(
                ReplayEntry(
                    kind="tool_result",
                    step=invocation.step,
                    summary=f"Tool result recorded: {result.name}.",
                    metadata=(
                        ("name", result.name),
                        ("call_id", result.call_id),
                        ("is_error", result.is_error),
                        (
                            "content_character_count",
                            len(result.content),
                        ),
                    ),
                )
            )

    entries.append(
        ReplayEntry(
            kind="run_ended",
            step=None,
            summary=f"Run ended: {record.end_reason}.",
            metadata=(
                ("end_reason", record.end_reason),
                ("step_count", record.step_count),
                ("model_call_count", record.model_call_count),
                ("tool_call_count", record.tool_call_count),
                ("tool_execution_count", record.tool_execution_count),
                (
                    "tool_result_error_count",
                    record.tool_result_error_count,
                ),
            ),
        )
    )

    return tuple(entries)
