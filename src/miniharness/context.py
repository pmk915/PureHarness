from miniharness.messages import AgentItem, Message


class ContextBuilder:
    def build(
        self,
        history: list[AgentItem],
    ) -> list[AgentItem]:
        return list(history)


class RecentContextBuilder(ContextBuilder):
    def __init__(
        self,
        max_items: int = 20,
    ):
        if max_items <= 0:
            raise ValueError(
                "max_items must be greater than 0"
            )

        self.max_items = max_items

    def build(
        self,
        history: list[AgentItem],
    ) -> list[AgentItem]:
        if len(history) <= self.max_items:
            return list(history)

        start = len(history) - self.max_items

        while start > 0:
            item = history[start]

            if (
                isinstance(item, Message)
                and item.role == "user"
            ):
                break

            start -= 1

        return list(history[start:])