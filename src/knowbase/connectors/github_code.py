import subprocess
from pathlib import Path
from typing import Iterator

from knowbase.connectors.base import Row
from knowbase.ingest.chunker import chunk_python


class GitHubCodeConnector:
    name = "github_code"

    def __init__(
        self,
        repo_path: Path,
        paths: list[str],
        exclude_dirs: list[str],
        max_chars: int = 2000,
    ):
        self.repo_path = Path(repo_path)
        self.paths = paths
        self.exclude_dirs = set(exclude_dirs)
        self.max_chars = max_chars

    def _head(self) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def fetch(self, since: str | None) -> Iterator[Row]:
        head = self._head()
        if since is not None and head is not None and since == head:
            return
        for base in self.paths:
            root = self.repo_path / base
            for path in sorted(root.rglob("*.py")):
                rel = path.relative_to(self.repo_path)
                if self.exclude_dirs.intersection(rel.parts):
                    continue
                counts: dict[str, int] = {}
                for chunk in chunk_python(path.read_text(errors="replace"), self.max_chars):
                    symbol = chunk.symbol or "module"
                    counts[symbol] = counts.get(symbol, 0) + 1
                    n = counts[symbol]
                    sid = f"{rel}#{symbol}" if n == 1 else f"{rel}#{symbol}@{n}"
                    prefix = f"{rel} {chunk.symbol}" if chunk.symbol else str(rel)
                    yield Row(
                        source="github_code",
                        source_id=sid,
                        document=f"{prefix}\n{chunk.text}",
                        raw_content=chunk.text,
                        metadata={
                            "path": str(rel),
                            "symbol": chunk.symbol,
                            "commit": head,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                        },
                    )

    def watermark(self) -> str | None:
        return self._head()
