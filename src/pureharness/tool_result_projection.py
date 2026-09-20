from typing import Protocol

from pureharness.messages import ToolResult


DEFAULT_MAX_TOOL_RESULT_CHARS = 12_000
DEFAULT_TOOL_RESULT_HEAD_CHARS = 6_000
DEFAULT_TOOL_RESULT_TAIL_CHARS = 4_000


class ToolResultProjector(Protocol):
    def project(self, result: ToolResult) -> ToolResult:
        ...


class IdentityToolResultProjector:
    """Return an equivalent model-facing copy without compacting content."""

    def project(self, result: ToolResult) -> ToolResult:
        return _copy_tool_result(result, result.content)


class DeterministicToolResultProjector:
    """Compact large text results by retaining deterministic head and tail."""

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
        head_chars: int = DEFAULT_TOOL_RESULT_HEAD_CHARS,
        tail_chars: int = DEFAULT_TOOL_RESULT_TAIL_CHARS,
    ) -> None:
        _validate_positive_integer("max_chars", max_chars)
        _validate_positive_integer("head_chars", head_chars)
        _validate_positive_integer("tail_chars", tail_chars)

        if head_chars + tail_chars >= max_chars:
            raise ValueError(
                "head_chars + tail_chars must be less than max_chars"
            )

        self.max_chars = max_chars
        self.head_chars = head_chars
        self.tail_chars = tail_chars

    def project(self, result: ToolResult) -> ToolResult:
        original_chars = len(result.content)

        if original_chars <= self.max_chars:
            return _copy_tool_result(result, result.content)

        retained_chars = self.head_chars + self.tail_chars
        omitted_chars = original_chars - retained_chars
        marker = (
            "\n\n... tool output compacted: "
            f"{omitted_chars} characters omitted "
            f"(original: {original_chars}; retained: {retained_chars}) "
            "...\n\n"
        )
        projected_content = (
            result.content[: self.head_chars]
            + marker
            + result.content[-self.tail_chars :]
        )

        if len(projected_content) >= original_chars:
            return _copy_tool_result(result, result.content)

        return _copy_tool_result(result, projected_content)


def _copy_tool_result(
    result: ToolResult,
    content: str,
) -> ToolResult:
    return ToolResult(
        name=result.name,
        content=content,
        call_id=result.call_id,
        is_error=result.is_error,
    )


def _validate_positive_integer(
    name: str,
    value: int,
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
