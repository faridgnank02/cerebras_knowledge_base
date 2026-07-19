from pathlib import Path

from knowbase.connectors.github_code import GitHubCodeConnector

CODE = '''class Loader:
    def load(self):
        return 1
'''


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "loader.py").write_text(CODE)
    (tmp_path / "pkg" / "tests").mkdir()
    (tmp_path / "pkg" / "tests" / "test_loader.py").write_text("def test(): pass\n")
    (tmp_path / "pkg" / "notes.md").write_text("not python\n")
    return tmp_path


def test_fetch_chunks_python_files_only(tmp_path):
    conn = GitHubCodeConnector(make_repo(tmp_path), paths=["pkg"], exclude_dirs=["tests"])
    rows = list(conn.fetch(None))
    assert len(rows) == 1
    row = rows[0]
    assert row.source == "github_code"
    assert row.source_id == "pkg/loader.py#L1-L3"
    assert row.document.startswith("pkg/loader.py")
    assert "class Loader" in row.raw_content
    assert row.metadata["path"] == "pkg/loader.py"


def test_fetch_skips_when_watermark_matches_head(tmp_path, monkeypatch):
    conn = GitHubCodeConnector(make_repo(tmp_path), paths=["pkg"], exclude_dirs=[])
    monkeypatch.setattr(conn, "_head", lambda: "abc123")
    assert list(conn.fetch("abc123")) == []
    assert list(conn.fetch("other")) != []
    assert conn.watermark() == "abc123"
