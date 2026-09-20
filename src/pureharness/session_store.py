import json
import os
import tempfile

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO

from pureharness.messages import (
    AgentItem,
    Message,
    ToolCall,
    ToolResult,
)
from pureharness.session import Session
from pureharness.run_record import RunRecord


SCHEMA_VERSION = 1
DURABLE_SESSION_SCHEMA_VERSION = 1


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


@dataclass
class DurableSession:
    session_id: str
    created_at: datetime
    updated_at: datetime
    workspace: Path
    model: str
    session: Session
    run_records: list[RunRecord]

    def __post_init__(self) -> None:
        _validate_session_id_value(self.session_id)
        _validate_timestamp("created_at", self.created_at)
        _validate_timestamp("updated_at", self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")

        self.workspace = Path(self.workspace)
        if not self.workspace.is_absolute():
            raise ValueError("workspace must be an absolute path")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be non-empty text")
        if not isinstance(self.session, Session):
            raise ValueError("session must be a Session")

        self.run_records = list(self.run_records)
        run_ids = set()
        for record in self.run_records:
            if not isinstance(record, RunRecord):
                raise ValueError(
                    "run_records must contain RunRecord values"
                )
            if record.session_id != self.session_id:
                raise ValueError(
                    "RunRecord session_id must match durable session"
                )
            if record.run_id in run_ids:
                raise ValueError(
                    f"Duplicate run ID: {record.run_id}"
                )
            run_ids.add(record.run_id)


@dataclass(frozen=True)
class DurableSessionSummary:
    session_id: str
    created_at: datetime
    updated_at: datetime
    workspace: Path
    model: str
    run_count: int
    last_run_id: str | None
    last_end_reason: str | None


class DurableSessionStore(Protocol):
    def save(self, state: DurableSession) -> None:
        ...

    def load(self, session_id: str) -> DurableSession:
        ...

    def list_sessions(self) -> tuple[DurableSessionSummary, ...]:
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
        try:
            _atomic_replace_records(
                path,
                [
                    {
                        "type": "session_meta",
                        "schema_version": SCHEMA_VERSION,
                    },
                    *(
                        _serialize_item(item)
                        for item in session.items
                    ),
                ],
            )
        except (OSError, TypeError, ValueError) as exc:
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
        try:
            _validate_session_id_value(session_id)
        except ValueError as exc:
            raise SessionStoreError(str(exc)) from exc

        return self.directory / f"{session_id}.jsonl"


class JsonlDurableSessionStore:
    """Persist one complete durable logical session per JSONL file."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def save(self, state: DurableSession) -> None:
        try:
            snapshot = DurableSession(
                session_id=state.session_id,
                created_at=state.created_at,
                updated_at=state.updated_at,
                workspace=state.workspace,
                model=state.model,
                session=Session(items=state.session.snapshot()),
                run_records=list(state.run_records),
            )
            path = self._path_for(snapshot.session_id)
            records = [
                {
                    "type": "durable_session_meta",
                    "schema_version": DURABLE_SESSION_SCHEMA_VERSION,
                    "session_id": snapshot.session_id,
                    "created_at": snapshot.created_at.isoformat(),
                    "updated_at": snapshot.updated_at.isoformat(),
                    "workspace": str(snapshot.workspace),
                    "model": snapshot.model,
                },
                *(
                    _serialize_item(item)
                    for item in snapshot.session.items
                ),
                *(
                    {
                        "type": "run_record",
                        "value": record.to_dict(),
                    }
                    for record in snapshot.run_records
                ),
            ]
            _atomic_replace_records(path, records)
        except (OSError, TypeError, ValueError) as exc:
            raise SessionStoreError(
                "Could not save durable session "
                f"{state.session_id!r}: {exc}"
            ) from exc

    def load(self, session_id: str) -> DurableSession:
        path = self._path_for(session_id)
        if not path.is_file():
            raise SessionStoreError(
                f"Session not found: {session_id}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                records = [
                    _parse_record(line, line_number)
                    for line_number, line in enumerate(file, start=1)
                    if line.strip()
                ]
            return _deserialize_durable_session(
                records,
                expected_session_id=session_id,
            )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            SessionStoreError,
        ) as exc:
            raise SessionStoreError(
                "Could not load durable session "
                f"{session_id!r} from {path}: {exc}"
            ) from exc

    def list_sessions(self) -> tuple[DurableSessionSummary, ...]:
        if not self.directory.exists():
            return ()
        if not self.directory.is_dir():
            raise SessionStoreError(
                "Durable session path is not a directory: "
                f"{self.directory}"
            )

        states = []
        for path in sorted(self.directory.glob("*.jsonl")):
            try:
                states.append(self.load(path.stem))
            except SessionStoreError as exc:
                raise SessionStoreError(
                    f"Could not list session file {path}: {exc}"
                ) from exc

        states.sort(
            key=lambda state: state.updated_at,
            reverse=True,
        )
        return tuple(_summarize_durable_session(state) for state in states)

    def _path_for(self, session_id: str) -> Path:
        try:
            _validate_session_id_value(session_id)
        except ValueError as exc:
            raise SessionStoreError(str(exc)) from exc
        return self.directory / f"{session_id}.jsonl"


def _copy_session(
    session: Session,
) -> Session:
    return Session(
        items=deepcopy(session.items)
    )


def _atomic_replace_records(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            for record in records:
                _write_record(file, record)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _deserialize_durable_session(
    records: list[dict[str, object]],
    *,
    expected_session_id: str,
) -> DurableSession:
    if not records:
        raise ValueError("Durable session is missing schema metadata")

    metadata = records[0]
    _validate_keys(
        metadata,
        {
            "type",
            "schema_version",
            "session_id",
            "created_at",
            "updated_at",
            "workspace",
            "model",
        },
    )
    if metadata["type"] != "durable_session_meta":
        raise ValueError(
            "First durable session record must be metadata"
        )
    version = metadata["schema_version"]
    if type(version) is not int:
        raise ValueError(
            "Durable session schema version must be an integer"
        )
    if version != DURABLE_SESSION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported durable session schema version: "
            f"{version}"
        )

    session_id = _require_string(metadata, "session_id")
    if session_id != expected_session_id:
        raise ValueError(
            "Durable session ID does not match its file name"
        )

    items = []
    run_records = []
    reading_run_records = False
    for record in records[1:]:
        if record.get("type") == "run_record":
            reading_run_records = True
            _validate_keys(record, {"type", "value"})
            value = record["value"]
            if not isinstance(value, dict):
                raise ValueError("run_record value must be an object")
            run_records.append(RunRecord.from_dict(value))
            continue

        if reading_run_records:
            raise ValueError(
                "Session items cannot follow durable RunRecords"
            )
        items.append(_deserialize_item(record))

    return DurableSession(
        session_id=session_id,
        created_at=_parse_timestamp(
            "created_at",
            _require_string(metadata, "created_at"),
        ),
        updated_at=_parse_timestamp(
            "updated_at",
            _require_string(metadata, "updated_at"),
        ),
        workspace=Path(_require_string(metadata, "workspace")),
        model=_require_string(metadata, "model"),
        session=Session(items=items),
        run_records=run_records,
    )


def _summarize_durable_session(
    state: DurableSession,
) -> DurableSessionSummary:
    last_record = (
        state.run_records[-1]
        if state.run_records
        else None
    )
    return DurableSessionSummary(
        session_id=state.session_id,
        created_at=state.created_at,
        updated_at=state.updated_at,
        workspace=state.workspace,
        model=state.model,
        run_count=len(state.run_records),
        last_run_id=(
            None if last_record is None else last_record.run_id
        ),
        last_end_reason=(
            None if last_record is None else last_record.end_reason
        ),
    )


def _parse_timestamp(name: str, value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 timestamp") from exc
    _validate_timestamp(name, timestamp)
    return timestamp


def _validate_timestamp(name: str, value: datetime) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware")


def _validate_session_id_value(session_id: str) -> None:
    if (
        not isinstance(session_id, str)
        or not session_id
        or session_id in {".", ".."}
        or "/" in session_id
        or "\\" in session_id
    ):
        raise ValueError(f"Invalid session ID: {session_id!r}")


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
