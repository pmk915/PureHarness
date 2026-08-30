from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, object]
