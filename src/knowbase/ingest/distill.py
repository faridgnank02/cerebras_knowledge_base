import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


SYSTEM_PROMPT = (
    "You distill GitHub issue threads into a compact retrieval artifact. "
    "Extract only what the thread itself establishes; do not invent. "
    "If the thread reaches no accepted answer, leave resolution empty."
)


class Distiller:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_input_chars: int = 60000,
        client=None,
        sleep=time.sleep,
    ):
        if client is None:
            # lazy so tests with injected fakes never import the SDK
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.max_input_chars = max_input_chars
        self._sleep = sleep

    def distill(self, title: str, thread: str) -> Artifact | None:
        content = thread[: self.max_input_chars]
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Title: {title}\n\n{content}"},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "issue_artifact",
                            "strict": True,
                            "schema": ARTIFACT_SCHEMA,
                        },
                    },
                )
                return Artifact(**json.loads(resp.choices[0].message.content))
            except Exception:
                if attempt == 2:
                    logger.warning("distillation failed for %r", title, exc_info=True)
                    return None
                self._sleep(2**attempt)
        return None

    def distill_document(self, title: str, thread: str) -> str | None:
        artifact = self.distill(title, thread)
        return render(title, artifact) if artifact else None
