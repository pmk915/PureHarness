import json

from dataclasses import dataclass, field
from pathlib import Path

from miniharness.messages import (
    AgentItem,
    Message,
    ToolCall,
    ToolResult,
)


@dataclass
class Session:
    items: list[AgentItem] = field(
        default_factory=list
    )

    def append(
        self,
        item: AgentItem,
    ) -> None:
        self.items.append(item)

    def snapshot(
        self,
    ) -> list[AgentItem]:
        return list(self.items)

    def save_jsonl(
        self,
        path: Path,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for item in self.items:
                record = _serialize_item(item)

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )

                file.write("\n")

    @classmethod
    def load_jsonl(
        cls,
        path: Path,
    ) -> "Session":
        session = cls()

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                item = _deserialize_item(
                    record
                )

                session.append(item)

        return session


def _serialize_item(
    item: AgentItem,
) -> dict[str, object]:

    if isinstance(item, Message):
        return {
            "type": "message",
            "role": item.role,
            "content": item.content,
        }

    if isinstance(item, ToolCall):
        return {
            "type": "tool_call",
            "name": item.name,
            "arguments": item.arguments,
            "call_id": item.call_id,
        }

    if isinstance(item, ToolResult):
        return {
            "type": "tool_result",
            "name": item.name,
            "content": item.content,
            "call_id": item.call_id,
            "is_error": item.is_error,
        }

    raise TypeError(
        f"Unsupported session item: {type(item)}"
    )


def _deserialize_item(
    record: dict[str, object],
) -> AgentItem:

    item_type = record.get("type")

    if item_type == "message":
        return Message(
            role=str(record["role"]),
            content=str(record["content"]),
        )

    if item_type == "tool_call":
        arguments = record["arguments"]

        if not isinstance(arguments, dict):
            raise ValueError(
                "ToolCall arguments must be an object"
            )

        call_id = record.get("call_id")

        return ToolCall(
            name=str(record["name"]),
            arguments=arguments,
            call_id=(
                str(call_id)
                if call_id is not None
                else None
            ),
        )

    if item_type == "tool_result":
        call_id = record.get("call_id")

        return ToolResult(
            name=str(record["name"]),
            content=str(record["content"]),
            call_id=(
                str(call_id)
                if call_id is not None
                else None
            ),
            is_error=bool(
                record.get(
                    "is_error",
                    False,
                )
            ),
        )

    raise ValueError(
        f"Unknown session item type: {item_type}"
    )