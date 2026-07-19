import logging
import shutil
import subprocess
from pathlib import Path

import psycopg

from knowbase.retrieval.vector import SearchResult

logger = logging.getLogger(__name__)


def parse_matches(output: str) -> dict[str, list[int]]:
    """Parse `path:line:text` grep output into {relpath: [line numbers]}."""
    matches: dict[str, list[int]] = {}
    for line in output.splitlines():
        path, sep, rest = line.partition(":")
        lineno_s, sep2, _ = rest.partition(":")
        if not sep or not sep2:
            continue
        try:
            lineno = int(lineno_s)
        except ValueError:
            continue
        matches.setdefault(path, []).append(lineno)
    return matches


def rg_matches(
    clone_path: Path, pattern: str, max_files: int = 20
) -> dict[str, list[int]]:
    clone_path = Path(clone_path)
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", "--color", "never", pattern, "."]
    else:
        cmd = ["git", "grep", "-n", "--untracked", "-e", pattern]
    try:
        proc = subprocess.run(
            cmd, cwd=clone_path, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("grep failed for %r in %s", pattern, clone_path, exc_info=True)
        return {}
    if proc.returncode != 0:  # 1 = no matches; >1 = error — both empty
        return {}
    matches = parse_matches(proc.stdout)
    normalized = {path.removeprefix("./"): lines for path, lines in matches.items()}
    return dict(list(normalized.items())[:max_files])


def grep_code(
    conn: psycopg.Connection, clone_path: Path, pattern: str, limit: int = 10
) -> list[SearchResult]:
    matches = rg_matches(clone_path, pattern)
    if not matches:
        return []
    rows = conn.execute(
        """
        SELECT id, source, source_id, document, metadata, updated_at
        FROM embeddings
        WHERE source = 'github_code' AND metadata->>'path' = ANY(%s)
        ORDER BY (metadata->>'start_line')::int
        """,
        (list(matches),),
    ).fetchall()
    by_path: dict[str, list] = {}
    for row in rows:
        by_path.setdefault(row[4]["path"], []).append(row)
    scored: list[SearchResult] = []
    for path, lines in matches.items():
        chunks = by_path.get(path)
        if not chunks:
            continue
        path_results: list[SearchResult] = []
        covered: set[int] = set()
        for id_, source, source_id, document, metadata, updated_at in chunks:
            start, end = metadata.get("start_line"), metadata.get("end_line")
            if start is None or end is None:
                continue
            in_range = [ln for ln in lines if start <= ln <= end]
            covered.update(in_range)
            if in_range:
                path_results.append(SearchResult(id_, source, source_id, document,
                                                 metadata, float(len(in_range)),
                                                 updated_at))
        # matched lines outside every chunk still count for the file,
        # at half weight, credited to its best (or first) chunk
        uncovered = sum(1 for ln in lines if ln not in covered)
        if uncovered:
            if path_results:
                best = max(path_results, key=lambda r: r.score)
                best.score += 0.5 * uncovered
            else:
                id_, source, source_id, document, metadata, updated_at = chunks[0]
                path_results.append(SearchResult(id_, source, source_id, document,
                                                 metadata, 0.5 * uncovered,
                                                 updated_at))
        scored.extend(path_results)
    scored.sort(key=lambda r: (-r.score, r.source_id))
    return scored[:limit]
