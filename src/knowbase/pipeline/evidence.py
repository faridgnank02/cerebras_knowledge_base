from dataclasses import dataclass, replace
from datetime import datetime

from knowbase.retrieval.expand import Expanded


@dataclass
class Evidence:
    n: int
    text: str
    source: str
    source_id: str
    url: str
    score: float
    updated_at: datetime | None


def source_url(source: str, metadata: dict) -> str:
    if source == "github_code":
        path = metadata.get("path", "")
        start, end = metadata.get("start_line"), metadata.get("end_line")
        if start is not None and end is not None:
            return f"{path}#L{start}-L{end}"
        return path
    return metadata.get("url", "")


def build_evidence(expanded: list[Expanded], max_chars: int = 2000) -> list[Evidence]:
    out = []
    for n, e in enumerate(expanded, start=1):
        r = e.result
        text = r.document
        if e.context:
            text = f"{text}\n\n[context]\n{e.context}"
        out.append(
            Evidence(
                n=n, text=text[:max_chars], source=r.source, source_id=r.source_id,
                url=source_url(r.source, r.metadata or {}), score=r.score,
                updated_at=r.updated_at,
            )
        )
    return out


def merge_evidence(*lists: list[Evidence]) -> list[Evidence]:
    out: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for evs in lists:
        for e in evs:
            key = (e.source, e.source_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(replace(e, n=len(out) + 1))
    return out
