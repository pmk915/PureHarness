from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, object]
    call_id: str | None = None


@dataclass
class ToolResult:
    name: str
    content: str
    call_id: str | None = None
    is_error: bool = False


AgentItem = Message | ToolCall | ToolResult