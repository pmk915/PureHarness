import json
import shlex

from dataclasses import dataclass
from typing import Literal

try:
    from rich.console import Console
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "RichTerminalRenderer requires the optional CLI dependencies. "
        "Install them with: pip install 'pureharness[cli]'"
    ) from exc

from pureharness.events import AgentEvent, AgentEventData


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
        "compact_inspect": "Inspected {target}",
        "compact_workspace": "workspace",
        "compact_read": "Read {target}",
        "compact_write": "Wrote {target}",
        "compact_update": "Updated {target}",
        "compact_run": "Run {target}",
        "compact_verification_passed": "✓ Verification passed",
        "compact_verification_failed": "✗ Verification failed",
        "compact_verification_error": "✗ Verification could not complete",
        "compact_exit": "exit",
        "compact_policy_denied": "✗ Policy denied",
        "compact_approval_required": "! Approval required",
        "compact_precondition_failed": "✗ Workspace precondition failed",
        "compact_tool_errors": "✗ Tool error observations",
        "compact_run_failed": "✗ Run failed",
        "compact_step_count": "{count} steps",
        "compact_tool_count": "{count} tool calls",
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
        "compact_inspect": "已查看 {target}",
        "compact_workspace": "工作区",
        "compact_read": "已读取 {target}",
        "compact_write": "已写入 {target}",
        "compact_update": "已更新 {target}",
        "compact_run": "执行 {target}",
        "compact_verification_passed": "✓ 验证通过",
        "compact_verification_failed": "✗ 验证失败",
        "compact_verification_error": "✗ 验证未能完成",
        "compact_exit": "退出码",
        "compact_policy_denied": "✗ 策略拒绝",
        "compact_approval_required": "! 需要批准",
        "compact_precondition_failed": "✗ 工作区前置条件未满足",
        "compact_tool_errors": "✗ 工具错误记录",
        "compact_run_failed": "✗ Run 执行失败",
        "compact_step_count": "{count} 步骤",
        "compact_tool_count": "{count} 工具调用",
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
        "Protocol completion": "协议完成度",
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


# Explicit compact hierarchy: routine diagnostics are hidden; tool actions and
# verification are progress; approval and terminal events are transitions;
# failures/recovery stay visible. The verbose event path below is unchanged.
_COMPACT_DIAGNOSTICS = frozenset({
    "context_build_started", "context_built", "model_started",
    "completion_recheck_requested", "completion_recheck_skipped",
    "workspace_mutated",
})
_COMPACT_ALERTS = frozenset({
    "context_build_failed", "context_window_exceeded", "context_recovering",
    "model_failed", "execution_budget_exhausted", "agent_interrupted",
})
_FILE_ACTIONS = {
    "list_files": "compact_inspect",
    "read_file": "compact_read",
    "read_file_range": "compact_read",
    "write_file": "compact_write",
    "apply_patch": "compact_update",
}


@dataclass(frozen=True)
class _ToolDisplay:
    identity: tuple[int | None, str, str | None]
    action: str
    command: bool


class RichTerminalRenderer:
    """Render Agent events without controlling Agent execution."""

    def __init__(
        self,
        locale: str = "en",
        console: Console | None = None,
        *,
        interactive: bool = False,
        detail_level: Literal["compact", "verbose"] = "verbose",
    ):
        if locale not in _TEXT:
            supported = ", ".join(_TEXT)
            raise ValueError(
                f"Unsupported locale: {locale}. "
                f"Supported locales: {supported}"
            )
        if detail_level not in {"compact", "verbose"}:
            raise ValueError(f"Unsupported detail level: {detail_level}")

        self.locale = locale
        self.console = console or Console()
        self.interactive = interactive
        self.detail_level = detail_level
        self._reset_compact_state()

    def _reset_compact_state(self) -> None:
        self._run_id: str | None = None
        self._active_tool: _ToolDisplay | None = None
        self._tool_call_count: int | None = None
        self._verification_attempts = 0
        self._last_verification: tuple[object, ...] | None = None
        self._visible_tool_errors = 0

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
        if self.detail_level == "compact":
            self._render_compact(event)
            return
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

    def _render_compact(self, event: AgentEvent) -> None:
        if event.type == "agent_started" or (
            event.run_id is not None and event.run_id != self._run_id
        ):
            self._reset_compact_state()
            self._run_id = event.run_id
        text = _TEXT[self.locale]
        data = event.data
        if event.type in _COMPACT_DIAGNOSTICS:
            return
        if event.type == "agent_started":
            if not self.interactive:
                self._print(text["title"])
        elif event.type == "model_completed":
            # Match RunRecord's returned-call count, not accepted dispatches.
            if "tool_call_count" in data:
                self._tool_call_count = (
                    (self._tool_call_count or 0) + data["tool_call_count"]
                )
        elif event.type == "tool_policy_evaluated":
            if data["decision"] == "deny":
                self._visible_tool_errors += 1
                self._print(
                    f"{text['compact_policy_denied']} · {_single_line(data['name'])}"
                )
            elif data["decision"] == "require_approval":
                self._print(
                    f"{text['compact_approval_required']} · {_single_line(data['name'])}"
                )
        elif event.type in {"approval_requested", "approval_granted", "approval_denied"}:
            if event.type == "approval_denied":
                self._visible_tool_errors += 1
            self._print(f"{text[event.type]} · {_single_line(data['name'])}")
        elif event.type == "workspace_precondition_failed":
            self._visible_tool_errors += 1
            self._print(
                f"{text['compact_precondition_failed']} · {_single_line(data['name'])}"
                f" · {_single_line(data['path'])} · {_single_line(data['reason'])}"
            )
        elif event.type == "tool_started":
            self._active_tool = self._tool_display(event)
            if self._active_tool.command:
                self._print(f"● {self._active_tool.action}")
        elif event.type == "tool_completed":
            display = self._active_tool
            if display is None or display.identity != _tool_identity(event):
                display = _ToolDisplay(
                    _tool_identity(event), _single_line(data["name"]), False,
                )
            else:
                self._active_tool = None
            if data["is_error"]:
                self._visible_tool_errors += 1
                self._print(f"{text['tool_failed']} · {display.action}")
            elif not display.command:
                self._print(f"✓ {display.action}")
            # A successful command ToolResult alone says nothing about its exit
            # code. Verification outcomes come only from structured evidence.
        elif event.type == "coding_evidence_snapshot":
            self._render_new_verification(data)
        elif event.type == "progress_snapshot":
            # Schema/exposure rejection may produce an error result without a
            # tool_completed, policy or precondition event. Surface the factual
            # count without guessing which tool/command failed from output text.
            errors = data.get("failed_tool_results", 0)
            if errors > self._visible_tool_errors:
                self._print(
                    f"{text['compact_tool_errors']} · {errors - self._visible_tool_errors}"
                )
                self._visible_tool_errors = errors
        elif event.type == "model_retrying":
            self._print(f"{text['model_retrying']} · {data['attempt']}/{data['max_attempts']}")
        elif event.type in _COMPACT_ALERTS:
            self._print(text[event.type])
            self._active_tool = None
        elif event.type == "agent_failed":
            self._print(f"{text['compact_run_failed']} · {_single_line(data['reason'])}")
            self._active_tool = None
        elif event.type == "agent_completed":
            parts = [text["run_completed"]]
            if "step_count" in data:
                parts.append(text["compact_step_count"].format(count=data["step_count"]))
            if self._tool_call_count is not None:
                parts.append(text["compact_tool_count"].format(count=self._tool_call_count))
            self._print(" · ".join(parts))
            self._active_tool = None

    def _tool_display(self, event: AgentEvent) -> _ToolDisplay:
        data = event.data
        name = data["name"]
        preview = data.get("arguments_preview", {})
        text = _TEXT[self.locale]
        if name in _FILE_ACTIONS:
            target = preview.get("path", "")
            if name == "list_files" and target in {"", "."}:
                target = text["compact_workspace"]
            action = (
                text[_FILE_ACTIONS[name]].format(target=_single_line(target))
                if target else _single_line(name)
            )
        elif name == "run_command":
            action = text["compact_run"].format(
                target=_command_preview(preview.get("argv", "run_command"))
            )
        else:
            action = _single_line(name)
        return _ToolDisplay(_tool_identity(event), action, name in {"run_command", "start_process"})

    def _render_new_verification(self, data: AgentEventData) -> None:
        outcome = data.get("last_verification_outcome")
        if outcome not in {"exit_zero", "exit_nonzero", "tool_error"}:
            return
        identity = (
            data.get("last_verification_step"), outcome,
            data.get("last_verification_exit_code"),
        )
        attempts = data.get("verification_attempts")
        if type(attempts) is int:
            if attempts <= self._verification_attempts:
                return
            self._verification_attempts = attempts
        elif identity == self._last_verification:
            return
        self._last_verification = identity
        text = _TEXT[self.locale]
        key = {
            "exit_zero": "compact_verification_passed",
            "exit_nonzero": "compact_verification_failed",
            "tool_error": "compact_verification_error",
        }[outcome]
        message = text[key]
        exit_code = data.get("last_verification_exit_code")
        if outcome == "exit_nonzero" and type(exit_code) is int:
            message += f" · {text['compact_exit']} {exit_code}"
        self._print(message)

    def _print(self, value: str) -> None:
        self.console.print(
            value,
            markup=False,
            highlight=False,
        )


def _tool_identity(event: AgentEvent) -> tuple[int | None, str, str | None]:
    return event.data.get("step"), event.data["name"], event.data.get("call_id")


def _single_line(value: str) -> str:
    # Keep meaningful spaces in paths/argv intact; escape line-breaking controls.
    value = value.replace("\r", r"\r").replace("\n", r"\n").replace("\t", r"\t")
    return value if len(value) <= 120 else value[:119] + "…"


def _command_preview(value: str) -> str:
    # Decode only complete JSON from the bounded, redacted preview. Truncated
    # or redacted JSON stays literal; never consult raw argv or parse a shell.
    try:
        argv = json.loads(value)
    except (ValueError, TypeError):
        return _single_line(value)
    if isinstance(argv, list) and all(isinstance(arg, str) for arg in argv):
        return _single_line(shlex.join(argv))
    return _single_line(value)
