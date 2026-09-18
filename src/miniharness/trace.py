from dataclasses import dataclass, field
from typing import Literal

from miniharness.messages import Message, ToolCall, ToolResult


RunEndReason = Literal[
    "completed",
    "max_steps_exceeded",
    "model_error",
    "context_error",
]

@dataclass
class StepTrace:
    index: int
    output: Message | list[ToolCall]
    tool_result: list[ToolResult] | None = None


@dataclass
class RunTrace:
    steps: list[StepTrace] = field(default_factory=list)
    end_reason: RunEndReason | None = None
