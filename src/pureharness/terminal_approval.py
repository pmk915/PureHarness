from collections.abc import Callable

from pureharness.approval import (
    ApprovalDecision,
    ApprovalRequest,
)
from pureharness.events import safe_arguments_preview


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class TerminalApprovalHandler:
    """Obtain a one-time, explicit approval decision from a terminal user."""

    def __init__(
        self,
        *,
        input_fn: InputFunction = input,
        output_fn: OutputFunction = print,
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn

    def request_approval(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        self.output_fn("Approval required")
        self.output_fn(f"Tool: {request.tool_name}")
        self.output_fn("Arguments:")
        preview = safe_arguments_preview(request.arguments)
        if preview:
            for name, value in preview.items():
                self.output_fn(f"  {name}: {value}")
        else:
            self.output_fn("  (none)")

        try:
            answer = self.input_fn("Approve this action? [y/N]: ")
        except (EOFError, KeyboardInterrupt):
            self.output_fn("Approval cancelled; action denied.")
            return ApprovalDecision.DENY

        if answer.strip().lower() in {"y", "yes"}:
            return ApprovalDecision.APPROVE
        return ApprovalDecision.DENY
