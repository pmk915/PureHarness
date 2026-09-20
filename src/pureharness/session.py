from dataclasses import dataclass, field

from pureharness.messages import AgentItem


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
