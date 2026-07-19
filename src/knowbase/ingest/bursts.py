from dataclasses import dataclass


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
