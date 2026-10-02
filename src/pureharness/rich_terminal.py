try:
    from rich.console import Console
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "RichTerminalRenderer requires the optional CLI dependencies. "
        "Install them with: pip install 'pureharness[cli]'"
    ) from exc

from pureharness.events import AgentEvent


_TEXT = {
    "en": {
        "title": "PureHarness",
        "context_build_started": "● Building context...",
        "context_built": "✓ Context built",
        "history": "history",
        "context": "context",
        "strategy": "strategy",
        "estimated_tokens": "estimated history tokens",
        "units": "units",
        "history_budget": "history token budget",
        "compacted_results": "tool outputs compacted",
        "trajectory_compacted": "trajectory compacted",
        "old_units": "old units",
        "estimated_token_unit": "estimated tokens",
        "task_state": "task state",
        "modified_files": "modified files",
        "recent_errors": "recent errors",
        "context_build_failed": "✗ Context build failed",
        "item_unit": "items",
        "separator": ": ",
        "model_started": "● Requesting model...",
        "tools_exposed": "tools exposed",
        "schema_approx": "~",
        "schema_tokens": "schema tokens",
        "model_retrying": "↻ Retrying model request",
        "model_completed": "✓ Model response received",
        "tool_calls": "tool calls",
        "tool_policy_evaluated": "◆ Tool policy evaluated",
        "approval_requested": "? Approval requested",
        "approval_granted": "✓ Approval granted",
        "approval_denied": "✗ Approval denied",
        "risk": "risk",
        "decision": "decision",
        "tool_started": "◆ Calling tool",
        "tool_completed": "✓ Tool completed",
        "tool_failed": "✗ Tool failed",
        "model_failed": "✗ Model request failed",
        "agent_completed": "✓ Task completed",
        "agent_interrupted": "! Agent interrupted",
        "agent_failed": "✗ Agent failed",
        "steps": "steps",
        "run_completed": "✓ Run completed",
        "verification": "Verification evidence",
        "context_window_exceeded": "! Provider context window exceeded",
        "context_recovering": "↻ Rebuilding smaller context",
        "completion_recheck_requested": "↻ Completion recheck requested",
        "completion_recheck_skipped": "! Completion recheck skipped",
        "execution_budget_exhausted": "! Execution budget exhausted",
    },
    "zh-CN": {
        "title": "PureHarness",
        "context_build_started": "● 正在构建上下文...",
        "context_built": "✓ 上下文已构建",
        "history": "历史记录",
        "context": "上下文",
        "strategy": "策略",
        "estimated_tokens": "估算历史 tokens",
        "units": "单元",
        "history_budget": "历史 token 预算",
        "compacted_results": "工具输出压缩",
        "trajectory_compacted": "轨迹已压缩",
        "old_units": "个旧单元",
        "estimated_token_unit": "估算 tokens",
        "task_state": "任务状态",
        "modified_files": "个修改文件",
        "recent_errors": "个近期错误",
        "context_build_failed": "✗ 上下文构建失败",
        "item_unit": "项",
        "separator": "：",
        "model_started": "● 正在请求模型...",
        "tools_exposed": "工具暴露",
        "schema_approx": "约 ",
        "schema_tokens": "schema tokens",
        "model_retrying": "↻ 正在重试模型请求",
        "model_completed": "✓ 已收到模型响应",
        "tool_calls": "工具调用",
        "tool_policy_evaluated": "◆ 工具策略已评估",
        "approval_requested": "? 请求操作批准",
        "approval_granted": "✓ 操作已批准",
        "approval_denied": "✗ 操作已拒绝",
        "risk": "风险",
        "decision": "决策",
        "tool_started": "◆ 正在调用工具",
        "tool_completed": "✓ 工具执行完成",
        "tool_failed": "✗ 工具执行失败",
        "model_failed": "✗ 模型请求失败",
        "agent_completed": "✓ 任务完成",
        "agent_interrupted": "! Agent 执行已中断",
        "agent_failed": "✗ Agent 执行失败",
        "steps": "步骤",
        "run_completed": "✓ Run 执行完成",
        "verification": "验证执行证据",
        "context_window_exceeded": "! 提供方上下文窗口超限",
        "context_recovering": "↻ 正在重建更小的上下文",
        "completion_recheck_requested": "↻ 请求完成复查",
        "completion_recheck_skipped": "! 跳过完成复查",
        "execution_budget_exhausted": "! 执行预算耗尽",
        "Workspace": "工作区",
        "Model": "模型",
        "Session ID": "会话 ID",
        "Resumed": "已恢复",
        "Type /help for commands.": "输入 /help 查看命令。",
        "Assistant": "助手",
        "Interactive commands": "交互命令",
        "Show interactive commands": "显示交互命令",
        "Show current harness/run status": "显示当前 harness/Run 状态",
        "Show durable session information": "显示持久会话信息",
        "Show recent Runs in this session": "显示本会话最近的 Runs",
        "Evaluate the latest Run": "评估最近的 Run",
        "Exit PureHarness": "退出 PureHarness",
        "Any other text starts an Agent run.": "其他文本将启动一个 Agent Run。",
        "Unknown command": "未知命令",
        "Runs in session": "会话中的 Runs",
        "Last run ID": "最近 Run ID",
        "Last end reason": "最近结束原因",
        "Last tool calls": "最近工具调用数",
        "Last estimated context tokens": "最近估算上下文 tokens",
        "Created (UTC)": "创建时间 (UTC)",
        "Updated (UTC)": "更新时间 (UTC)",
        "History items": "历史记录项",
        "No finalized Runs in this session yet.": "本会话尚无已结束的 Run。",
        "No finalized Run to evaluate yet.": "尚无已结束的 Run 可供评估。",
        "RUN ID\tEND REASON\tSTEPS\tTOOLS": "RUN ID\t结束原因\t步骤\t工具",
        "Run evaluation": "Run 执行评估",
        "Execution evaluation": "执行评估",
        "Run": "Run",
        "ID": "ID",
        "End reason": "结束原因",
        "Steps": "步骤",
        "Tool calls": "工具调用",
        "Tool result errors": "工具错误结果",
        "Metrics": "指标",
        "Completion": "协议完成度",
        "Step efficiency": "步骤效率",
        "Tool reliability": "工具可靠性",
        "Diagnosis": "诊断",
        "Type": "类型",
        "Confidence": "置信度",
        "Reason": "原因",
        "Recovery (advisory only)": "恢复建议（仅供参考）",
        "Action": "操作建议",
        "Completion describes protocol execution, not verified task correctness.": (
            "完成度描述协议执行状态，不代表已验证任务正确性。"
        ),
    },
}


class RichTerminalRenderer:
    """Render Agent events without controlling Agent execution."""

    def __init__(
        self,
        locale: str = "en",
        console: Console | None = None,
        *,
        interactive: bool = False,
    ):
        if locale not in _TEXT:
            supported = ", ".join(_TEXT)
            raise ValueError(
                f"Unsupported locale: {locale}. "
                f"Supported locales: {supported}"
            )

        self.locale = locale
        self.console = console or Console()
        self.interactive = interactive

    def translate(self, value: str) -> str:
        return _TEXT[self.locale].get(value, value)

    def write(self, value: str) -> None:
        self._print(value)

    def render_header(
        self,
        *,
        workspace: object,
        model: str,
        session_id: str,
        resumed: bool,
    ) -> None:
        self.console.print("PureHarness", style="bold", markup=False)
        for label, value in (
            ("Workspace", workspace),
            ("Model", model),
            ("Session ID", session_id),
        ):
            self._print(f"  {self.translate(label)}: {value}")
        if resumed:
            self._print(f"  {self.translate('Resumed')}: {session_id}")
        self._print(self.translate("Type /help for commands."))

    def render_response(self, response: str) -> None:
        self._print("")
        self._print(self.translate("Assistant"))
        self.console.rule()
        self._print(response)
        self.console.rule()

    def __call__(self, event: AgentEvent) -> None:
        text = _TEXT[self.locale]
        data = event.data

        if event.type == "agent_started":
            if not self.interactive:
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
            self._print(
                f"  {text['estimated_tokens']}"
                f"{text['separator']}"
                f"{data['estimated_history_tokens']}"
            )
            self._print(
                f"  {text['units']}"
                f"{text['separator']}"
                f"{data['included_units']}/"
                f"{data['total_units']}"
            )
            self._print(
                f"  {text['compacted_results']}"
                f"{text['separator']}"
                f"{data['compacted_tool_results']}"
            )
            self._print(
                f"  {text['task_state']}"
                f"{text['separator']}"
                f"{data['files_modified_count']} "
                f"{text['modified_files']} · "
                f"{data['recent_errors_count']} "
                f"{text['recent_errors']}"
            )

            if "history_token_budget" in data:
                self._print(
                    f"  {text['history_budget']}"
                    f"{text['separator']}"
                    f"{data['history_token_budget']}"
                )

            if data.get("trajectory_compacted", False):
                self._print(
                    f"  {text['trajectory_compacted']}"
                    f"{text['separator']}"
                    f"{data['compacted_source_units']} "
                    f"{text['old_units']} → "
                    f"{data['compacted_trajectory_estimated_tokens']} "
                    f"{text['estimated_token_unit']}"
                )

        elif event.type == "context_build_failed":
            self._print(text["context_build_failed"])

        elif event.type == "model_started":
            self._print(text["model_started"])

            if "registered_tool_count" in data:
                self._print(
                    f"  {text['tools_exposed']}"
                    f"{text['separator']}"
                    f"{data['exposed_tool_count']}/"
                    f"{data['registered_tool_count']} · "
                    f"{text['schema_approx']}"
                    f"{data['estimated_tool_schema_tokens']} "
                    f"{text['schema_tokens']}"
                )

        elif event.type == "model_retrying":
            self._print(
                f"{text['model_retrying']}"
                f"{text['separator']}"
                f"{data['attempt']}/{data['max_attempts']}"
            )

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

        elif event.type in {
            "approval_requested",
            "approval_granted",
            "approval_denied",
        }:
            self._print(
                f"{text[event.type]}"
                f"{text['separator']}"
                f"{data['name']}"
            )

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
            self._print(text[
                "run_completed" if self.interactive else "agent_completed"
            ])
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

        elif event.type == "agent_interrupted":
            self._print(text["agent_interrupted"])

        elif event.type == "coding_evidence_snapshot":
            if data.get("last_verification_outcome") is not None:
                self._print(
                    f"  {text['verification']}{text['separator']}"
                    f"{data['last_verification_outcome']} "
                    f"(exit_code={data.get('last_verification_exit_code')})"
                )

        elif event.type in {
            "context_window_exceeded",
            "context_recovering",
            "completion_recheck_requested",
            "completion_recheck_skipped",
            "execution_budget_exhausted",
        }:
            self._print(text[event.type])

    def _print(self, value: str) -> None:
        self.console.print(
            value,
            markup=False,
            highlight=False,
        )
