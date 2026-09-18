import json
import os
import tempfile

from copy import deepcopy
from pathlib import Path
from typing import Protocol, TextIO

from miniharness.messages import (
    AgentItem,
    Message,
    ToolCall,
    ToolResult,
)
from miniharness.session import Session


SCHEMA_VERSION = 1


class SessionStoreError(Exception):
    """Raised when explicitly requested Session persistence fails."""


class SessionStore(Protocol):
    def save(
        self,
        session_id: str,
        session: Session,
    ) -> None:
        ...

    def load(
        self,
        session_id: str,
    ) -> Session:
        ...


class MemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def save(
        self,
        session_id: str,
        session: Session,
    ) -> None:
        try:
            stored_session = _copy_session(session)
        except Exception as exc:
            raise SessionStoreError(
                f"Could not save session {session_id!r}: {exc}"
            ) from exc

        self._sessions[session_id] = stored_session

    def load(
        self,
        session_id: str,
    ) -> Session:
        try:
            session = self._sessions[session_id]
        except KeyError as exc:
            raise SessionStoreError(
                f"Session not found: {session_id}"
            ) from exc

        try:
            return _copy_session(session)
        except Exception as exc:
            raise SessionStoreError(
                f"Could not load session {session_id!r}: {exc}"
            ) from exc


class JsonlSessionStore:
    def __init__(
        self,
        directory: str | Path,
    ) -> None:
        self.directory = Path(directory)

    def save(
        self,
        session_id: str,
        session: Session,
    ) -> None:
        path = self._path_for(session_id)
        temporary_path: Path | None = None

        try:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)

                _write_record(
                    file,
                    {
                        "type": "session_meta",
                        "schema_version": SCHEMA_VERSION,
                    },
                )

                for item in session.items:
                    _write_record(
                        file,
                        _serialize_item(item),
                    )

                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, path)
        except (OSError, TypeError, ValueError) as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

            raise SessionStoreError(
                f"Could not save session {session_id!r}: {exc}"
            ) from exc

    def load(
        self,
        session_id: str,
    ) -> Session:
        path = self._path_for(session_id)

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                records = [
                    _parse_record(line, line_number)
                    for line_number, line in enumerate(
                        file,
                        start=1,
                    )
                    if line.strip()
                ]

            _validate_metadata(records)

            return Session(
                items=[
                    _deserialize_item(record)
                    for record in records[1:]
                ]
            )
        except SessionStoreError:
            raise
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SessionStoreError(
                f"Could not load session {session_id!r}: {exc}"
            ) from exc

    def _path_for(
        self,
        session_id: str,
    ) -> Path:
        if (
            not session_id
            or session_id in {".", ".."}
            or "/" in session_id
            or "\\" in session_id
        ):
            raise SessionStoreError(
                f"Invalid session ID: {session_id!r}"
            )

        return self.directory / f"{session_id}.jsonl"


def _copy_session(
    session: Session,
) -> Session:
    return Session(
        items=deepcopy(session.items)
    )


def _write_record(
    file: TextIO,
    record: dict[str, object],
) -> None:
    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
    )
    file.write("\n")


def _parse_record(
    line: str,
    line_number: int,
) -> dict[str, object]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SessionStoreError(
            f"Malformed JSON on line {line_number}: {exc.msg}"
        ) from exc

    if not isinstance(record, dict):
        raise SessionStoreError(
            f"Session record on line {line_number} must be an object"
        )

    return record


def _validate_metadata(
    records: list[dict[str, object]],
) -> None:
    if not records:
        raise SessionStoreError(
            "Session data is missing schema metadata"
        )

    metadata = records[0]

    if metadata.get("type") != "session_meta":
        raise SessionStoreError(
            "First session record must be session metadata"
        )

    try:
        _validate_keys(
            metadata,
            {"type", "schema_version"},
        )
    except ValueError as exc:
        raise SessionStoreError(str(exc)) from exc

    schema_version = metadata.get("schema_version")

    if type(schema_version) is not int:
        raise SessionStoreError(
            "Session schema version must be an integer"
        )

    if schema_version != SCHEMA_VERSION:
        raise SessionStoreError(
            "Unsupported session schema version: "
            f"{schema_version}"
        )


def _serialize_item(
    item: AgentItem,
) -> dict[str, object]:
    if isinstance(item, Message):
        return {
            "type": "message",
            "role": item.role,
            "content": item.content,
        }

    if isinstance(item, ToolCall):
        return {
            "type": "tool_call",
            "name": item.name,
            "arguments": item.arguments,
            "call_id": item.call_id,
        }

    if isinstance(item, ToolResult):
        return {
            "type": "tool_result",
            "name": item.name,
            "content": item.content,
            "call_id": item.call_id,
            "is_error": item.is_error,
        }

    raise TypeError(
        f"Unsupported session item: {type(item)}"
    )


def _deserialize_item(
    record: dict[str, object],
) -> AgentItem:
    item_type = record.get("type")

    if item_type == "message":
        _validate_keys(
            record,
            {"type", "role", "content"},
        )
        return Message(
            role=_require_string(record, "role"),
            content=_require_string(record, "content"),
        )

    if item_type == "tool_call":
        _validate_keys(
            record,
            {"type", "name", "arguments", "call_id"},
        )
        arguments = record["arguments"]

        if not isinstance(arguments, dict):
            raise ValueError(
                "ToolCall arguments must be an object"
            )

        return ToolCall(
            name=_require_string(record, "name"),
            arguments=arguments,
            call_id=_require_optional_string(
                record,
                "call_id",
            ),
        )

    if item_type == "tool_result":
        _validate_keys(
            record,
            {
                "type",
                "name",
                "content",
                "call_id",
                "is_error",
            },
        )
        is_error = record["is_error"]

        if not isinstance(is_error, bool):
            raise ValueError(
                "ToolResult is_error must be a boolean"
            )

        return ToolResult(
            name=_require_string(record, "name"),
            content=_require_string(record, "content"),
            call_id=_require_optional_string(
                record,
                "call_id",
            ),
            is_error=is_error,
        )

    raise ValueError(
        f"Unknown session item type: {item_type}"
    )


def _validate_keys(
    record: dict[str, object],
    expected: set[str],
) -> None:
    actual = set(record)

    if actual != expected:
        raise ValueError(
            "Invalid fields for session record: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )


def _require_string(
    record: dict[str, object],
    key: str,
) -> str:
    value = record[key]

    if not isinstance(value, str):
        raise ValueError(
            f"Session field {key!r} must be a string"
        )

    return value


def _require_optional_string(
    record: dict[str, object],
    key: str,
) -> str | None:
    value = record[key]

    if value is not None and not isinstance(value, str):
        raise ValueError(
            f"Session field {key!r} must be a string or null"
        )

    return value
