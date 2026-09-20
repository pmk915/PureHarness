import argparse

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from tempfile import TemporaryDirectory

from pureharness.agent import Agent
from pureharness.coding_tools import create_coding_tools
from pureharness.deepseek_model import DeepSeekModel
from pureharness.messages import Message, ToolCall
from pureharness.tools import ToolRegistry


def main(
    show_terminal: bool = False,
    locale: str = "en",
) -> None:
    with TemporaryDirectory(
        prefix="pureharness-coding-demo-"
    ) as temporary_directory:
        workspace = Path(temporary_directory)

        calculator_file = workspace / "calculator.py"
        test_file = workspace / "test_calculator.py"

        calculator_file.write_text(
            (
                "def add(a: int, b: int) -> int:\n"
                "    return a - b\n"
            ),
            encoding="utf-8",
        )

        test_file.write_text(
            (
                "from calculator import add\n"
                "\n"
                "\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
            ),
            encoding="utf-8",
        )

        registry = ToolRegistry()

        for tool in create_coding_tools(workspace):
            registry.register(tool)

        listeners = []

        if show_terminal:
            from pureharness.rich_terminal import (
                RichTerminalRenderer,
            )

            listeners.append(
                RichTerminalRenderer(locale=locale)
            )

        agent = Agent(
            model=DeepSeekModel(),
            tools=registry,
            max_steps=10,
            listeners=listeners,
        )

        task = (
            "You are a coding agent working inside the provided workspace. "
            "There is a bug in the calculator project. "
            "Inspect the relevant files, fix the bug, and run the tests. "
            "Do not claim completion until the tests pass. "
            "Use only paths relative to the workspace. "
            "Call at most one tool per model response."
        )

        result = agent.run(task)

        print("\n=== FINAL ANSWER ===")
        print(result)

        print("\n=== FINAL calculator.py ===")
        print(
            calculator_file.read_text(
                encoding="utf-8",
            )
        )

        print("\n=== TRACE ===")


        for step in agent.trace.steps:

            print(f"Step {step.index}")

            if step.tool_result:

                for result in step.tool_result:

                    print(
                        f"ToolResult "
                        f"(error={result.is_error}):"
                    )

                    print(result.content)

            else:

                print(
                    f"Message: {step.output.content}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Render structured Agent events with Rich.",
    )
    parser.add_argument(
        "--locale",
        choices=["en", "zh-CN"],
        default="en",
        help="Terminal presentation language.",
    )
    arguments = parser.parse_args()

    main(
        show_terminal=arguments.terminal,
        locale=arguments.locale,
    )
