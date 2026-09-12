from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from tempfile import TemporaryDirectory

from miniharness.agent import Agent
from miniharness.coding_tools import create_coding_tools
from miniharness.deepseek_model import DeepSeekModel
from miniharness.messages import Message, ToolCall
from miniharness.tools import ToolRegistry


def main() -> None:
    with TemporaryDirectory(
        prefix="miniharness-coding-demo-"
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

        agent = Agent(
            model=DeepSeekModel(),
            tools=registry,
            max_steps=10,
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
            print(f"\nStep {step.index}")

            if isinstance(step.output, ToolCall):
                print(
                    f"ToolCall: "
                    f"{step.output.name}"
                    f"({step.output.arguments})"
                )

            elif isinstance(step.output, Message):
                print(
                    f"Message: "
                    f"{step.output.content}"
                )

            if step.tool_result is not None:
                print(
                    f"ToolResult "
                    f"(error={step.tool_result.is_error}):"
                )
                print(step.tool_result.content)

        print(
            "\nEnd reason:",
            agent.trace.end_reason,
        )


if __name__ == "__main__":
    main()