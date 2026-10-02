"""Optional terminal input; injected callables and pipes stay authoritative."""

import sys

from collections.abc import Callable

from pureharness.interactive import INTERACTIVE_COMMANDS


def create_interactive_input(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> Callable[[str], str]:
    if (
        input_fn is not input
        or output_fn is not print
        or not sys.stdin.isatty()
        or not sys.stdout.isatty()
    ):
        return input_fn
    try:
        return PromptToolkitInput()
    except ImportError:
        return input_fn


class PromptToolkitInput:
    """Keep task history in memory; approvals use a separate plain prompt."""

    def __init__(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings

        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(event):
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        def newline(event):
            event.current_buffer.insert_text("\n")

        self.session = PromptSession(
            history=InMemoryHistory(),
            completer=WordCompleter(list(INTERACTIVE_COMMANDS), sentence=True),
            complete_while_typing=False,
            multiline=True,
            key_bindings=bindings,
        )

    def __call__(self, prompt: str) -> str:
        if prompt != "You › ":
            # Approval responses must not enter task history or completion.
            return input(prompt)
        return self.session.prompt(prompt)
