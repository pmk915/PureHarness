from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


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

    def execute(self, arguments: dict[str, object]) -> object:
        return self.function(**arguments)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")

        return self._tools[name]

    def execute(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> object:
        tool = self.get(name)
        return tool.execute(arguments)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())


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
