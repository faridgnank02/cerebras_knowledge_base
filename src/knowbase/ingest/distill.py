from dataclasses import dataclass


@dataclass
class Artifact:
    question: str
    summary: str
    resolution: str
    systems: list[str]
    code_refs: list[str]


ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "summary": {"type": "string"},
        "resolution": {"type": "string"},
        "systems": {"type": "array", "items": {"type": "string"}},
        "code_refs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "summary", "resolution", "systems", "code_refs"],
    "additionalProperties": False,
}


def render(title: str, a: Artifact) -> str:
    parts = [f"# {title}", f"Question: {a.question}", f"Summary: {a.summary}"]
    if a.resolution:
        parts.append(f"Resolution: {a.resolution}")
    if a.systems:
        parts.append("Systems: " + ", ".join(a.systems))
    if a.code_refs:
        parts.append("Code refs: " + ", ".join(a.code_refs))
    return "\n\n".join(parts)
