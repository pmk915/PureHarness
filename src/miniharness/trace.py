from dataclasses import dataclass, field
from typing import Literal

from miniharness.messages import Message, ToolCall, ToolResult


RunEndReason = Literal[
    "completed",
    "max_steps_exceeded",
]

@dataclass
class StepTrace:
    index: int
    output: Message | ToolCall
    tool_result: ToolResult | None = None


@dataclass
class RunTrace:
    steps: list[StepTrace] = field(default_factory=list)