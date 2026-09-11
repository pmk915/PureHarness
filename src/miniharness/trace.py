from dataclasses import dataclass, field

from miniharness.messages import Message, ToolCall, ToolResult


@dataclass
class StepTrace:
    index: int
    output: Message | ToolCall
    tool_result: ToolResult | None = None


@dataclass
class RunTrace:
    steps: list[StepTrace] = field(default_factory=list)