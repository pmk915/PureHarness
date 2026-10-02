import builtins

import pytest

import pureharness.interactive_input as input_module
from pureharness.interactive_input import create_interactive_input


def test_custom_input_is_authoritative_even_with_tty(monkeypatch):
    custom = lambda prompt: "/exit"
    monkeypatch.setattr(input_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(input_module.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(input_module, "PromptToolkitInput", lambda: pytest.fail("custom input replaced"))
    assert create_interactive_input(custom, print) is custom


@pytest.mark.parametrize("stdin_tty, stdout_tty, custom_output", [
    (False, True, False), (True, False, False), (True, True, True),
])
def test_redirected_or_injected_output_preserves_builtin_input(
    monkeypatch, stdin_tty, stdout_tty, custom_output,
):
    monkeypatch.setattr(input_module.sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(input_module.sys.stdout, "isatty", lambda: stdout_tty)
    monkeypatch.setattr(input_module, "PromptToolkitInput", lambda: pytest.fail("non-terminal input enhanced"))
    output = (lambda value: None) if custom_output else print
    assert create_interactive_input(input, output) is input


def test_factory_selects_toolkit_for_builtin_terminal_io(monkeypatch):
    adapter = lambda prompt: "hello"
    monkeypatch.setattr(input_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(input_module.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(input_module, "PromptToolkitInput", lambda: adapter)
    assert create_interactive_input(input, print) is adapter


def test_factory_falls_back_if_toolkit_is_missing(monkeypatch):
    monkeypatch.setattr(input_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(input_module.sys.stdout, "isatty", lambda: True)

    def missing():
        raise ModuleNotFoundError("prompt_toolkit")

    monkeypatch.setattr(input_module, "PromptToolkitInput", missing)
    assert create_interactive_input(input, print) is input


@pytest.fixture
def toolkit_input(monkeypatch):
    toolkit = pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    original_session = toolkit.PromptSession
    with create_pipe_input() as pipe:
        monkeypatch.setattr(toolkit, "PromptSession", lambda **kwargs: original_session(
            input=pipe, output=DummyOutput(), **kwargs,
        ))
        yield input_module.PromptToolkitInput(), pipe


def test_toolkit_enter_submits_and_up_recalls_in_memory_history(toolkit_input):
    adapter, pipe = toolkit_input
    pipe.send_text("first request\r")
    assert adapter("You › ") == "first request"
    pipe.send_text("\x1b[A\r")
    assert adapter("You › ") == "first request"
    assert adapter.session.history.get_strings() == ["first request"]


def test_toolkit_slash_completion_and_alt_enter_newline(toolkit_input):
    adapter, pipe = toolkit_input
    def submit_completed(buffer):
        if buffer.text == "/eval":
            pipe.send_text("\r")

    # Tab completion is scheduled asynchronously by prompt-toolkit; submit
    # only after the actual completed text is available, without timing sleeps.
    adapter.session.default_buffer.on_text_changed += submit_completed
    pipe.send_text("/eva\t")
    assert adapter("You › ") == "/eval"
    adapter.session.default_buffer.on_text_changed -= submit_completed
    pipe.send_text("line one\x1b\rline two\r")
    assert adapter("You › ") == "line one\nline two"


def test_toolkit_propagates_ctrl_c_and_eof(toolkit_input):
    adapter, pipe = toolkit_input
    pipe.send_text("\x03")
    with pytest.raises(KeyboardInterrupt):
        adapter("You › ")
    pipe.send_text("\x04")
    with pytest.raises(EOFError):
        adapter("You › ")


def test_approval_input_does_not_enter_task_history(toolkit_input, monkeypatch):
    adapter, _ = toolkit_input
    prompts = []

    def approval_input(prompt):
        prompts.append(prompt)
        return "yes"

    monkeypatch.setattr(builtins, "input", approval_input)
    assert adapter("Approve this action? [y/N]: ") == "yes"
    assert prompts == ["Approve this action? [y/N]: "]
    assert adapter.session.history.get_strings() == []
