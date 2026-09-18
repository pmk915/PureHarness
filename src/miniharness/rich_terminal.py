try:
    from rich.console import Console
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "RichTerminalRenderer requires the optional CLI dependencies. "
        "Install them with: pip install 'miniharness[cli]'"
    ) from exc

from miniharness.events import AgentEvent


_TEXT = {
    "en": {
        "title": "MiniHarness",
        "context_build_started": "● Building context...",
        "context_built": "✓ Context built",
        "history": "history",
        "context": "context",
        "strategy": "strategy",
        "item_unit": "items",
        "separator": ": ",
        "model_started": "● Requesting model...",
        "model_completed": "✓ Model response received",
        "tool_calls": "tool calls",
        "tool_policy_evaluated": "◆ Tool policy evaluated",
        "risk": "risk",
        "decision": "decision",
        "tool_started": "◆ Calling tool",
        "tool_completed": "✓ Tool completed",
        "tool_failed": "✗ Tool failed",
        "model_failed": "✗ Model request failed",
        "agent_completed": "✓ Task completed",
        "agent_failed": "✗ Agent failed",
        "steps": "steps",
    },
    "zh-CN": {
        "title": "MiniHarness",
        "context_build_started": "● 正在构建上下文...",
        "context_built": "✓ 上下文已构建",
        "history": "历史记录",
        "context": "上下文",
        "strategy": "策略",
        "item_unit": "项",
        "separator": "：",
        "model_started": "● 正在请求模型...",
        "model_completed": "✓ 已收到模型响应",
        "tool_calls": "工具调用",
        "tool_policy_evaluated": "◆ 工具策略已评估",
        "risk": "风险",
        "decision": "决策",
        "tool_started": "◆ 正在调用工具",
        "tool_completed": "✓ 工具执行完成",
        "tool_failed": "✗ 工具执行失败",
        "model_failed": "✗ 模型请求失败",
        "agent_completed": "✓ 任务完成",
        "agent_failed": "✗ Agent 执行失败",
        "steps": "步骤",
    },
}


class RichTerminalRenderer:
    """Render Agent events without controlling Agent execution."""

    def __init__(
        self,
        locale: str = "en",
        console: Console | None = None,
    ):
        if locale not in _TEXT:
            supported = ", ".join(_TEXT)
            raise ValueError(
                f"Unsupported locale: {locale}. "
                f"Supported locales: {supported}"
            )

        self.locale = locale
        self.console = console or Console()

    def __call__(self, event: AgentEvent) -> None:
        text = _TEXT[self.locale]
        data = event.data

        if event.type == "agent_started":
            self._print(text["title"])

        elif event.type == "context_build_started":
            self._print(text["context_build_started"])

        elif event.type == "context_built":
            self._print(text["context_built"])
            self._print(
                f"  {text['history']}"
                f"{text['separator']}"
                f"{data['history_item_count']} "
                f"{text['item_unit']}"
            )
            self._print(
                f"  {text['context']}"
                f"{text['separator']}"
                f"{data['context_item_count']} "
                f"{text['item_unit']}"
            )
            self._print(
                f"  {text['strategy']}"
                f"{text['separator']}"
                f"{data['context_strategy']}"
            )

        elif event.type == "model_started":
            self._print(text["model_started"])

        elif event.type == "model_completed":
            self._print(text["model_completed"])

            if data["output_kind"] == "tool_calls":
                self._print(
                    f"  {text['tool_calls']}"
                    f"{text['separator']}"
                    f"{data['tool_call_count']}"
                )

        elif event.type == "tool_policy_evaluated":
            self._print(
                f"{text['tool_policy_evaluated']}"
                f"{text['separator']}"
                f"{data['name']}"
            )
            self._print(
                f"  {text['risk']}"
                f"{text['separator']}"
                f"{data['risk_level']}"
            )
            self._print(
                f"  {text['decision']}"
                f"{text['separator']}"
                f"{data['decision']}"
            )

        elif event.type == "tool_started":
            self._print(
                f"{text['tool_started']}"
                f"{text['separator']}"
                f"{data['name']}"
            )

            for key, value in data["arguments_preview"].items():
                self._print(f"  {key}: {value}")

        elif event.type == "tool_completed":
            key = (
                "tool_failed"
                if data["is_error"]
                else "tool_completed"
            )
            self._print(text[key])

        elif event.type == "model_failed":
            self._print(text["model_failed"])

        elif event.type == "agent_completed":
            self._print(text["agent_completed"])
            self._print(
                f"  {text['steps']}"
                f"{text['separator']}"
                f"{data['step_count']}"
            )

        elif event.type == "agent_failed":
            self._print(
                f"{text['agent_failed']}"
                f"{text['separator']}"
                f"{data['reason']}"
            )

    def _print(self, value: str) -> None:
        self.console.print(
            value,
            markup=False,
            highlight=False,
        )
