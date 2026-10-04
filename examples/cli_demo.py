"""Offline CLI demo using the existing model_factory seam, not a provider.

Only the responses are scripted. Tools, policy/preconditions, rendering,
verification, Session persistence, and RunRecord/JSONL generation are real.
"""

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pureharness.cli import InputFunction, main as cli_main
from pureharness.messages import AgentItem, Message, ToolCall, ToolResult
from pureharness.model import ModelError, ModelOutput
from pureharness.tools import Tool


DEMO_TASK = "Fix add in calculator.py and verify it with check_add.py."
MODEL_NAME = "offline-scripted-demo"
FIXTURE = Path(__file__).parent / "cli_demo_workspace"


class ScriptedDemoModel:
    """One fixed task, with real preceding tool results required at each step."""

    def __init__(self) -> None:
        self.index = 0
        self.calls = (
            ToolCall("read_file", {"path": "calculator.py"}, "demo-read"),
            ToolCall("read_file", {"path": "check_add.py"}, "demo-read-check"),
            ToolCall(
                "apply_patch",
                {
                    "path": "calculator.py",
                    "old_text": "return left - right",
                    "new_text": "return left + right",
                },
                "demo-patch",
            ),
            ToolCall(
                "run_command",
                {
                    "argv": [sys.executable, "check_add.py"],
                    "purpose": "verification",
                    "timeout_seconds": 10,
                },
                "demo-verify",
            ),
        )

    def generate(
        self, messages: list[AgentItem], tools: list[Tool],
    ) -> ModelOutput:
        if self.index == 0:
            request = next(
                (item for item in reversed(messages)
                 if isinstance(item, Message) and item.role == "user"),
                None,
            )
            if request is None or request.content != DEMO_TASK:
                raise ModelError(f"This scripted demo only accepts: {DEMO_TASK}")
        elif self.index <= len(self.calls):
            previous = self.calls[self.index - 1]
            result = next(
                (item for item in reversed(messages)
                 if isinstance(item, ToolResult) and item.call_id == previous.call_id),
                None,
            )
            if result is None or result.is_error:
                raise ModelError("The scripted demo did not receive a successful tool result.")
        else:
            raise ModelError("The fixed demo is exhausted; start a fresh demo for another run.")

        if self.index == len(self.calls):
            self.index += 1
            return Message(
                "assistant",
                "Applied the scripted addition correction and ran the local check. "
                "Review the verification outcome in the recorded evidence. "
                "This demonstrates CLI execution, not benchmark performance.",
            )
        call = self.calls[self.index]
        if call.name not in {tool.name for tool in tools}:
            raise ModelError(f"Required demo tool is not exposed: {call.name}")
        self.index += 1
        return [call]


def prepare_demo(output_dir: Path | None) -> Path:
    """Copy the fixture into a fresh directory; never overwrite a prior run."""
    if output_dir is None:
        root = Path(tempfile.mkdtemp(prefix="pureharness-cli-demo-"))
    else:
        root = output_dir.expanduser().absolute()
        root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(FIXTURE, root / "workspace")
    return root


def main(
    argv: Sequence[str] | None = None, *, input_fn: InputFunction = input,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        help="New artifacts directory; defaults to a new temporary directory.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Use the existing verbose interactive CLI presentation.",
    )
    parser.add_argument(
        "--plain", action="store_true",
        help="Use the existing plain interactive CLI presentation.",
    )
    parser.add_argument("--locale", choices=("en", "zh-CN"), default="en")
    parser.add_argument(
        "--jsonl", action="store_true",
        help="Run the fixed task once using run --output jsonl.",
    )
    args = parser.parse_args(argv)
    if args.plain and args.verbose:
        parser.error("--plain and --verbose cannot be used together")
    if args.jsonl and (args.plain or args.verbose):
        parser.error("--plain and --verbose are interactive demo options, not JSONL options")
    try:
        root = prepare_demo(args.output_dir)
    except OSError as exc:
        print(f"Demo setup failed: {exc}", file=sys.stderr)
        return 2

    print("Offline scripted CLI demo: no LLM or API calls.", file=sys.stderr)
    print(f"Artifacts: {root}", file=sys.stderr)
    print(f"Task: {DEMO_TASK}", file=sys.stderr)
    def factory(name: str) -> ScriptedDemoModel:
        return ScriptedDemoModel()
    if args.jsonl:
        arguments = [
            "run", DEMO_TASK, "--workspace", str(root / "workspace"),
            "--model", MODEL_NAME, "--record", str(root / "run.json"),
            "--output", "jsonl",
        ]
        with (root / "events.jsonl").open("x", encoding="utf-8") as events:
            def emit(line: str) -> None:
                events.write(line + "\n")
                events.flush()
                print(line)

            return cli_main(
                arguments, model_factory=factory, output_fn=emit,
                error_fn=lambda message: print(message, file=sys.stderr),
            )

    arguments = [
        "--workspace", str(root / "workspace"), "--model", MODEL_NAME,
        "--record-dir", str(root / "records"), "--locale", args.locale,
    ]
    if args.verbose:
        arguments.append("--verbose")
    if args.plain:
        arguments.append("--plain")
    previous_home = os.environ.get("PUREHARNESS_HOME")
    os.environ["PUREHARNESS_HOME"] = str(root / "state")
    try:
        return cli_main(arguments, model_factory=factory, input_fn=input_fn)
    finally:
        if previous_home is None:
            os.environ.pop("PUREHARNESS_HOME", None)
        else:
            os.environ["PUREHARNESS_HOME"] = previous_home


if __name__ == "__main__":
    raise SystemExit(main())
