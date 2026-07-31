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
    assert row.source_id == "pkg/loader.py#module"
    assert row.document.startswith("pkg/loader.py")
    assert "class Loader" in row.raw_content
    assert row.metadata["path"] == "pkg/loader.py"


def test_fetch_skips_when_watermark_matches_head(tmp_path, monkeypatch):
    conn = GitHubCodeConnector(make_repo(tmp_path), paths=["pkg"], exclude_dirs=[])
    monkeypatch.setattr(conn, "_head", lambda: "abc123")
    assert list(conn.fetch("abc123")) == []
    assert list(conn.fetch("other")) != []
    assert conn.watermark() == "abc123"


def test_symbol_ids_with_suffixes(tmp_path):
    (tmp_path / "pkg").mkdir()
    body = "\n".join(f"    x{i} = {i}" for i in range(60))
    (tmp_path / "pkg" / "mod.py").write_text(
        "import os\n\n"
        f"def alpha():\n{body}\n\n"
        f"def alpha():\n{body}\n\n"
        "def beta():\n    return 1\n"
    )
    conn = GitHubCodeConnector(tmp_path, paths=["pkg"], exclude_dirs=[], max_chars=400)
    ids = [r.source_id for r in conn.fetch(None)]
    assert "pkg/mod.py#alpha" in ids
    assert "pkg/mod.py#alpha@2" in ids          # second chunk named alpha
    assert "pkg/mod.py#beta" in ids
    assert any(i.startswith("pkg/mod.py#module") for i in ids)  # the import header
    assert len(ids) == len(set(ids))            # ids unique within a file
    assert all("#L" not in i for i in ids)


def test_symbol_id_rows_carry_line_metadata(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("def f():\n    return 1\n")
    conn = GitHubCodeConnector(tmp_path, paths=["pkg"], exclude_dirs=[])
    row = next(iter(conn.fetch(None)))
    assert row.metadata["start_line"] == 1
    assert row.metadata["end_line"] == 2
