import re
from dataclasses import dataclass

LEVELS = [
    re.compile(r"^class\s+(\w+)"),
    re.compile(r"^(?:async\s+)?def\s+(\w+)"),
    re.compile(r"^\s{4}(?:async\s+)?def\s+(\w+)"),
]


@dataclass
class Chunk:
    text: str
    start_line: int
    end_line: int
    symbol: str | None


def chunk_python(text: str, max_chars: int = 2000) -> list[Chunk]:
    return _split(text.rstrip("\n").splitlines(), offset=0, level=0, max_chars=max_chars)


def _split(lines: list[str], offset: int, level: int, max_chars: int) -> list[Chunk]:
    text = "\n".join(lines)
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [Chunk(text, offset + 1, offset + len(lines), None)]
    if level >= len(LEVELS):
        return _hard_split(lines, offset, max_chars)
    pattern = LEVELS[level]
    starts = [i for i, line in enumerate(lines) if pattern.match(line)]
    if not starts:
        return _split(lines, offset, level + 1, max_chars)
    bounds = ([0] if starts[0] != 0 else []) + starts + [len(lines)]
    chunks: list[Chunk] = []
    for a, b in zip(bounds, bounds[1:]):
        seg = lines[a:b]
        seg_text = "\n".join(seg)
        if not seg_text.strip():
            continue
        match = pattern.match(seg[0])
        symbol = match.group(1) if match else None
        if len(seg_text) <= max_chars:
            chunks.append(Chunk(seg_text, offset + a + 1, offset + a + len(seg), symbol))
        else:
            sub = _split(seg, offset + a, level + 1, max_chars)
            if symbol:
                sub = [
                    Chunk(
                        c.text,
                        c.start_line,
                        c.end_line,
                        f"{symbol}.{c.symbol}" if c.symbol else symbol,
                    )
                    for c in sub
                ]
            chunks.extend(sub)
    return chunks


def _hard_split(lines: list[str], offset: int, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []
    start = 0
    size = 0
    for i, line in enumerate(lines):
        if buf and size + len(line) > max_chars:
            chunks.append(Chunk("\n".join(buf), offset + start + 1, offset + i, None))
            buf, start, size = [], i, 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        chunks.append(
            Chunk("\n".join(buf), offset + start + 1, offset + len(lines), None)
        )
    return chunks
