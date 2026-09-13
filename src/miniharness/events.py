from dataclasses import dataclass
from typing import Literal


AgentEventType = Literal[
    "agent_started",
    "model_completed",
    "tool_started",
    "tool_completed",
    "agent_completed",
    "agent_failed",
]


@dataclass
class AgentEvent:
    type: AgentEventType
    data: dict[str, object]