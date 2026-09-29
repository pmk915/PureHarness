from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class ToolRunResource(Protocol):
    """Run-scoped resource shared by one or more Tool definitions."""

    def reset_run_state(self) -> None:
        ...

    def cleanup_run_state(self) -> None:
        ...


class RiskLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DESTRUCTIVE = "destructive"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, object]
    function: Callable[..., object]
    category: str = "general"
    risk_level: RiskLevel = RiskLevel.READ
    side_effects: bool = False
    run_resource: ToolRunResource | None = None

    def execute(self, arguments: dict[str, object]) -> object:
        return self.function(**arguments)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._run_resources: list[ToolRunResource] = []

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        resource = tool.run_resource
        if resource is not None and not any(
            existing is resource for existing in self._run_resources
        ):
            self._run_resources.append(resource)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")

        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def reset_run_state(self) -> None:
        for resource in self._run_resources:
            resource.reset_run_state()

    def cleanup_run_state(self) -> None:
        for resource in reversed(self._run_resources):
            resource.cleanup_run_state()


def add(a: int, b: int) -> int:
    return a + b


ADD_TOOL = Tool(
    name="add",
    description="Add two integers and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "description": "The first integer.",
            },
            "b": {
                "type": "integer",
                "description": "The second integer.",
            },
        },
        "required": ["a", "b"],
    },
    function=add,
    category="utility",
    risk_level=RiskLevel.READ,
    side_effects=False,
)
