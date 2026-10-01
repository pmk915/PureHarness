from dataclasses import dataclass
from enum import Enum


class CommandPurpose(str, Enum):
    GENERAL = "general"
    VERIFICATION = "verification"


class VerificationOutcome(str, Enum):
    EXIT_ZERO = "exit_zero"
    EXIT_NONZERO = "exit_nonzero"
    TOOL_ERROR = "tool_error"


@dataclass(frozen=True)
class CommandToolObservation:
    purpose: CommandPurpose
    exit_code: int
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, CommandPurpose):
            raise ValueError("purpose must be CommandPurpose")
        if not isinstance(self.exit_code, int) or isinstance(
            self.exit_code,
            bool,
        ):
            raise ValueError("exit_code must be an integer")
        if not isinstance(self.content, str):
            raise ValueError("content must be text")

    def __str__(self) -> str:
        return self.content
