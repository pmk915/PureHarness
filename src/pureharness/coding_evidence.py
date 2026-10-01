from dataclasses import dataclass

from pureharness.tools import Tool
from pureharness.verification import (
    CommandPurpose,
    CommandToolObservation,
    VerificationOutcome,
)
from pureharness.workspace_discipline import WorkspaceMutation


@dataclass(frozen=True)
class CodingEvidenceSnapshot:
    """Immutable coding activity facts observed during one Agent run."""

    workspace_mutations: int = 0
    command_executions: int = 0
    command_tool_errors: int = 0
    process_starts: int = 0
    process_polls: int = 0
    process_stops: int = 0
    process_tool_errors: int = 0
    executions_since_last_mutation: int = 0
    verification_attempts: int = 0
    verification_exit_zero: int = 0
    verification_exit_nonzero: int = 0
    verification_tool_errors: int = 0
    verifications_since_last_mutation: int = 0
    last_mutation_step: int | None = None
    last_execution_step: int | None = None
    last_verification_outcome: VerificationOutcome | None = None
    last_verification_exit_code: int | None = None
    last_verification_step: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "workspace_mutations",
            "command_executions",
            "command_tool_errors",
            "process_starts",
            "process_polls",
            "process_stops",
            "process_tool_errors",
            "executions_since_last_mutation",
            "verification_attempts",
            "verification_exit_zero",
            "verification_exit_nonzero",
            "verification_tool_errors",
            "verifications_since_last_mutation",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"CodingEvidenceSnapshot {name} must be a "
                    "non-negative integer"
                )
        for name in (
            "last_mutation_step",
            "last_execution_step",
            "last_verification_step",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"CodingEvidenceSnapshot {name} must be None or a "
                    "non-negative integer"
                )
        if (
            self.last_verification_outcome is not None
            and not isinstance(
                self.last_verification_outcome,
                VerificationOutcome,
            )
        ):
            raise ValueError(
                "CodingEvidenceSnapshot last_verification_outcome must "
                "be None or VerificationOutcome"
            )
        if self.last_verification_exit_code is not None and (
            not isinstance(self.last_verification_exit_code, int)
            or isinstance(self.last_verification_exit_code, bool)
        ):
            raise ValueError(
                "CodingEvidenceSnapshot last_verification_exit_code "
                "must be None or an integer"
            )


class CodingEvidenceTracker:
    """Collect coding-specific facts without making completion decisions."""

    def __init__(self) -> None:
        self._workspace_mutations = 0
        self._command_executions = 0
        self._command_tool_errors = 0
        self._process_starts = 0
        self._process_polls = 0
        self._process_stops = 0
        self._process_tool_errors = 0
        self._executions_since_last_mutation = 0
        self._verification_attempts = 0
        self._verification_exit_zero = 0
        self._verification_exit_nonzero = 0
        self._verification_tool_errors = 0
        self._verifications_since_last_mutation = 0
        self._last_mutation_step: int | None = None
        self._last_execution_step: int | None = None
        self._last_verification_outcome: VerificationOutcome | None = None
        self._last_verification_exit_code: int | None = None
        self._last_verification_step: int | None = None

    @property
    def snapshot(self) -> CodingEvidenceSnapshot:
        return CodingEvidenceSnapshot(
            workspace_mutations=self._workspace_mutations,
            command_executions=self._command_executions,
            command_tool_errors=self._command_tool_errors,
            process_starts=self._process_starts,
            process_polls=self._process_polls,
            process_stops=self._process_stops,
            process_tool_errors=self._process_tool_errors,
            executions_since_last_mutation=(
                self._executions_since_last_mutation
            ),
            verification_attempts=self._verification_attempts,
            verification_exit_zero=self._verification_exit_zero,
            verification_exit_nonzero=self._verification_exit_nonzero,
            verification_tool_errors=self._verification_tool_errors,
            verifications_since_last_mutation=(
                self._verifications_since_last_mutation
            ),
            last_mutation_step=self._last_mutation_step,
            last_execution_step=self._last_execution_step,
            last_verification_outcome=(
                self._last_verification_outcome
            ),
            last_verification_exit_code=(
                self._last_verification_exit_code
            ),
            last_verification_step=self._last_verification_step,
        )

    def record_tool_started(
        self,
        tool: Tool,
        arguments: dict[str, object] | None = None,
        *,
        step: int,
    ) -> None:
        _validate_step(step)
        arguments = {} if arguments is None else arguments
        operation = _coding_operation(tool)
        if operation == "command":
            self._command_executions += 1
            self._record_execution(step)
            if (
                _native_command_purpose(tool, arguments)
                is CommandPurpose.VERIFICATION
            ):
                self._verification_attempts += 1
                self._last_verification_step = step
                self._last_verification_outcome = None
                self._last_verification_exit_code = None
                if self._workspace_mutations > 0:
                    self._verifications_since_last_mutation += 1
        elif operation == "process_start":
            self._process_starts += 1
            self._record_execution(step)
        elif operation == "process_poll":
            self._process_polls += 1
        elif operation == "process_stop":
            self._process_stops += 1

    def record_tool_result(
        self,
        tool: Tool,
        arguments: dict[str, object] | None = None,
        *,
        result: object | None = None,
        is_error: bool,
        step: int | None = None,
    ) -> None:
        if not isinstance(is_error, bool):
            raise ValueError("is_error must be bool")
        if step is not None:
            _validate_step(step)
        arguments = {} if arguments is None else arguments
        operation = _coding_operation(tool)
        if is_error and operation == "command":
            self._command_tool_errors += 1
        elif is_error and operation in {
            "process_start",
            "process_poll",
            "process_stop",
        }:
            self._process_tool_errors += 1

        if (
            operation != "command"
            or _native_command_purpose(tool, arguments)
            is not CommandPurpose.VERIFICATION
        ):
            return
        if is_error:
            self._verification_tool_errors += 1
            self._last_verification_outcome = (
                VerificationOutcome.TOOL_ERROR
            )
            self._last_verification_exit_code = None
            return
        if not isinstance(result, CommandToolObservation):
            return
        if result.purpose is not CommandPurpose.VERIFICATION:
            return
        self._last_verification_exit_code = result.exit_code
        if result.exit_code == 0:
            self._verification_exit_zero += 1
            self._last_verification_outcome = VerificationOutcome.EXIT_ZERO
        else:
            self._verification_exit_nonzero += 1
            self._last_verification_outcome = (
                VerificationOutcome.EXIT_NONZERO
            )

    def record_workspace_mutation(
        self,
        mutation: WorkspaceMutation,
        *,
        step: int,
    ) -> None:
        if not isinstance(mutation, WorkspaceMutation):
            raise TypeError("mutation must be WorkspaceMutation")
        _validate_step(step)
        self._workspace_mutations += 1
        self._executions_since_last_mutation = 0
        self._verifications_since_last_mutation = 0
        self._last_mutation_step = step

    def _record_execution(self, step: int) -> None:
        self._last_execution_step = step
        if self._workspace_mutations > 0:
            self._executions_since_last_mutation += 1


def _coding_operation(tool: Tool) -> str | None:
    if tool.category == "execution" and tool.name == "run_command":
        return "command"
    if tool.category != "process":
        return None
    return {
        "start_process": "process_start",
        "poll_process": "process_poll",
        "stop_process": "process_stop",
    }.get(tool.name)


def _native_command_purpose(
    tool: Tool,
    arguments: dict[str, object],
) -> CommandPurpose | None:
    if _coding_operation(tool) != "command":
        return None
    if (
        tool.function.__annotations__.get("return")
        is not CommandToolObservation
    ):
        return None
    properties = tool.parameters.get("properties")
    if not isinstance(properties, dict):
        return None
    purpose_schema = properties.get("purpose")
    expected_values = [
        CommandPurpose.GENERAL.value,
        CommandPurpose.VERIFICATION.value,
    ]
    if (
        not isinstance(purpose_schema, dict)
        or purpose_schema.get("enum") != expected_values
    ):
        return None
    try:
        return CommandPurpose(
            arguments.get("purpose", CommandPurpose.GENERAL.value)
        )
    except (TypeError, ValueError):
        return None


def _validate_step(step: int) -> None:
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer")
