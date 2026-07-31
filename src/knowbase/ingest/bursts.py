from dataclasses import dataclass
from typing import Callable


@dataclass
class Burst:
    author: str
    text: str
    reactions: int


def split_bursts(comments: list[dict]) -> list[Burst]:
    bursts: list[Burst] = []
    for c in comments:
        body = (c.get("body") or "").strip()
        if not body:
            continue
        if bursts and bursts[-1].author == c["author"]:
            bursts[-1].text += "\n\n" + body
            bursts[-1].reactions += c.get("reactions", 0)
        else:
            bursts.append(Burst(c["author"], body, c.get("reactions", 0)))
    return bursts


def make_burst_scorer(
    idf: dict[str, float],
    lexemize: Callable[[str], list[str]],
    *,
    idf_threshold: float = 4.0,
    min_chars: int = 200,
) -> Callable[[Burst], int]:
    def score(burst: Burst) -> int:
        s = 0
        if idf and any(idf.get(lex, 0.0) >= idf_threshold for lex in lexemize(burst.text)):
            s += 1
        if len(burst.text) >= min_chars:
            s += 1
        if burst.reactions > 0:
            s += 1
        return s

    return score
