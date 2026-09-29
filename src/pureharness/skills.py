import re

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files

from pureharness.messages import Message


_SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_METADATA_FIELDS = frozenset(
    {"name", "version", "description"}
)
_BUILTIN_SKILL_RESOURCES = {
    "coding-task": ("builtin_skills", "coding-task", "SKILL.md"),
}


class SkillDocumentError(ValueError):
    """A built-in Skill document does not match the strict small format."""


@dataclass(frozen=True)
class Skill:
    name: str
    version: int
    description: str
    instructions: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not _SKILL_NAME_PATTERN.fullmatch(self.name)
        ):
            raise ValueError(
                "Skill name must be a non-empty stable identifier"
            )
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version <= 0
        ):
            raise ValueError("Skill version must be a positive integer")
        for field_name in ("description", "instructions"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Skill {field_name} must be non-empty text"
                )

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.version}"


def parse_skill_document(document: str) -> Skill:
    """Parse PureHarness's strict built-in Skill document format."""
    if not isinstance(document, str):
        raise SkillDocumentError("Skill document must be text")
    lines = document.splitlines()
    if not lines or lines[0] != "---":
        raise SkillDocumentError(
            "Skill document is missing the opening metadata boundary"
        )
    try:
        closing_index = lines.index("---", 1)
    except ValueError as exc:
        raise SkillDocumentError(
            "Skill document is missing the closing metadata boundary"
        ) from exc

    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if ":" not in line:
            raise SkillDocumentError(
                f"Invalid Skill metadata line: {line!r}"
            )
        name, value = (part.strip() for part in line.split(":", 1))
        if name not in _METADATA_FIELDS:
            raise SkillDocumentError(
                f"Unknown Skill metadata field: {name!r}"
            )
        if name in metadata:
            raise SkillDocumentError(
                f"Duplicate Skill metadata field: {name!r}"
            )
        if not value:
            raise SkillDocumentError(
                f"Skill metadata field {name!r} must not be empty"
            )
        metadata[name] = value

    missing = sorted(_METADATA_FIELDS - metadata.keys())
    if missing:
        raise SkillDocumentError(
            "Missing required Skill metadata field(s): "
            + ", ".join(missing)
        )
    version_text = metadata["version"]
    if not version_text.isdecimal():
        raise SkillDocumentError(
            "Skill version metadata must be a positive integer"
        )
    version = int(version_text)
    instructions = "\n".join(lines[closing_index + 1 :]).strip()
    if not instructions:
        raise SkillDocumentError("Skill instruction body must not be empty")
    try:
        return Skill(
            name=metadata["name"],
            version=version,
            description=metadata["description"],
            instructions=instructions,
        )
    except ValueError as exc:
        raise SkillDocumentError(str(exc)) from exc


def load_builtin_skill(name: str) -> Skill:
    try:
        resource_parts = _BUILTIN_SKILL_RESOURCES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown built-in Skill: {name!r}") from exc
    resource = files("pureharness").joinpath(*resource_parts)
    try:
        document = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise SkillDocumentError(
            f"Built-in Skill resource is unavailable: {name!r}"
        ) from exc
    skill = parse_skill_document(document)
    if skill.name != name:
        raise SkillDocumentError(
            f"Built-in Skill resource {name!r} declares {skill.name!r}"
        )
    return skill


def default_coding_skills() -> tuple[Skill, ...]:
    return (load_builtin_skill("coding-task"),)


def normalize_skills(skills: Sequence[Skill]) -> tuple[Skill, ...]:
    if isinstance(skills, (str, bytes)):
        raise ValueError("skills must be a sequence of Skill values")
    normalized = tuple(skills)
    if any(not isinstance(skill, Skill) for skill in normalized):
        raise ValueError("skills must contain only Skill values")
    names = [skill.name for skill in normalized]
    if len(names) != len(set(names)):
        raise ValueError("active Skill names must not contain duplicates")
    return normalized


def render_skill(skill: Skill) -> Message:
    return Message(
        role="system",
        content=(
            "[PureHarness Active Skill]\n"
            f"Name: {skill.name}\n"
            f"Version: {skill.version}\n"
            f"Description: {skill.description}\n\n"
            f"{skill.instructions}"
        ),
    )
