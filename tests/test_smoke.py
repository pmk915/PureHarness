import os
import subprocess
import sys
from pathlib import Path

import miniharness


def test_package_import():
    assert miniharness.__version__ == "0.1.0"


from miniharness.messages import ToolCall


def test_tool_call():
    tool_call = ToolCall(
        name="add",
        arguments={
            "a": 12,
            "b": 17,
        },
    )

    assert tool_call.name == "add"
    assert tool_call.arguments["a"] == 12
    assert tool_call.arguments["b"] == 17


def test_core_import_does_not_require_optional_dependencies():
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(
        project_root / "src"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import miniharness; "
                "from miniharness.agent import Agent"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
