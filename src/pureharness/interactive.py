"""Read-only interactive commands and post-run human evaluation.

The CLI supplies durable facts and presentation callables. These commands never
construct an Agent or model, and recovery guidance has no execution authority.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum

from pureharness.evaluation import (
    EvaluationReportBuilder,
    RecoveryAction,
    RecoveryAdvisor,
    trajectory_from_run_record,
)
from pureharness.run_record import RunRecord
from pureharness.session_store import DurableSession


INTERACTIVE_COMMANDS = {
    "/help": "Show interactive commands",
    "/status": "Show current harness/run status",
    "/session": "Show durable session information",
    "/runs": "Show recent Runs in this session",
    "/eval": "Evaluate the latest Run",
    "/exit": "Exit PureHarness",
}


class CommandResult(Enum):
    NOT_COMMAND = "not_command"
    HANDLED = "handled"
    EXIT = "exit"


class InteractiveCommands:
    def __init__(
        self,
        state: DurableSession,
        output: Callable[[str], None],
        translate: Callable[[str], str] = lambda value: value,
    ) -> None:
        self.state = state
        self.output = output
        self.translate = translate
        self._handlers = {
            "/help": self.help,
            "/status": self.status,
            "/session": self.session,
            "/runs": self.runs,
            "/eval": self.evaluate_latest,
        }

    def handle(self, prompt: str) -> CommandResult:
        if not prompt.startswith("/"):
            return CommandResult.NOT_COMMAND
        if prompt == "/exit":
            return CommandResult.EXIT
        handler = self._handlers.get(prompt)
        if handler is None:
            self.output(f"{self.translate('Unknown command')}: {prompt}")
        else:
            handler()
        return CommandResult.HANDLED

    def help(self) -> None:
        self.output(self.translate("Interactive commands"))
        for command, description in INTERACTIVE_COMMANDS.items():
            self.output(f"  {command:<10} {self.translate(description)}")
        self.output(self.translate("Any other text starts an Agent run."))

    def status(self) -> None:
        record = self.latest_record
        self.field("Session ID", self.state.session_id)
        self.field("Workspace", self.state.workspace)
        self.field("Model", self.state.model)
        self.field("Runs in session", len(self.state.run_records))
        self.field("Last run ID", record.run_id if record else "(none)")
        self.field("Last end reason", record.end_reason if record else "(none)")
        if record is not None:
            self.field("Last tool calls", record.tool_call_count)
            self.field(
                "Last estimated context tokens",
                record.sum_estimated_history_tokens,
            )

    def session(self) -> None:
        self.field("Session ID", self.state.session_id)
        self.field("Created (UTC)", _utc_timestamp(self.state.created_at))
        self.field("Updated (UTC)", _utc_timestamp(self.state.updated_at))
        self.field("Workspace", self.state.workspace)
        self.field("Model", self.state.model)
        self.field("Runs in session", len(self.state.run_records))
        self.field("History items", len(self.state.session.items))

    def runs(self) -> None:
        if not self.state.run_records:
            self.output(self.translate("No finalized Runs in this session yet."))
            return
        self.output(self.translate("RUN ID\tEND REASON\tSTEPS\tTOOLS"))
        for record in self.state.run_records[-10:]:
            self.output(
                f"{record.run_id}\t{record.end_reason}\t"
                f"{record.step_count}\t{record.tool_call_count}"
            )

    @property
    def latest_record(self) -> RunRecord | None:
        return self.state.run_records[-1] if self.state.run_records else None

    def evaluate_latest(self) -> None:
        if self.latest_record is None:
            self.output(self.translate("No finalized Run to evaluate yet."))
            return
        self.render_evaluation(self.latest_record, detailed=True)

    def render_evaluation(
        self,
        record: RunRecord,
        *,
        detailed: bool = False,
    ) -> None:
        report = EvaluationReportBuilder().build(trajectory_from_run_record(record))
        recovery = RecoveryAdvisor().advise(report.diagnosis)
        self.output("")
        self.output(self.translate(
            "Execution evaluation" if detailed else "Run evaluation"
        ))
        if detailed:
            self.output(self.translate("Run"))
            self.field("ID", record.run_id, indent=True)
            self.field("End reason", record.end_reason, indent=True)
            self.field("Steps", record.step_count, indent=True)
            self.field("Tool calls", record.tool_call_count, indent=True)
            self.field("Tool result errors", record.tool_result_error_count, indent=True)
            self.output(self.translate("Metrics"))
        self.field("Completion", f"{report.metrics.task_success_score:.3f}", indent=True)
        self.field("Step efficiency", f"{report.metrics.step_efficiency_score:.3f}", indent=True)
        self.field("Tool reliability", f"{report.metrics.tool_reliability_score:.3f}", indent=True)
        if detailed:
            self.output(self.translate("Diagnosis"))
            self.field("Type", report.diagnosis.failure_type.value, indent=True)
            self.field("Confidence", f"{report.diagnosis.confidence:.3f}", indent=True)
            self.field("Reason", report.diagnosis.reason, indent=True)
            self.output(self.translate("Recovery (advisory only)"))
            self.field("Action", recovery.action.value, indent=True)
            self.field("Confidence", f"{recovery.confidence:.3f}", indent=True)
            self.field("Reason", recovery.reason, indent=True)
        else:
            self.field("Diagnosis", report.diagnosis.failure_type.value, indent=True)
            if recovery.action is not RecoveryAction.NONE:
                self.field("Recovery (advisory only)", recovery.action.value, indent=True)
        self.output(self.translate(
            "Completion describes protocol execution, not verified task correctness."
        ))

    def field(self, label: str, value: object, *, indent: bool = False) -> None:
        prefix = "  " if indent else ""
        self.output(f"{prefix}{self.translate(label)}: {value}")


def _utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
