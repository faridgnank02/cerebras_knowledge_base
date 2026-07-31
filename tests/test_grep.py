import subprocess

import pytest

from knowbase.connectors.base import Row
from knowbase.db import upsert_rows
from knowbase.retrieval.grep import grep_code, parse_matches, rg_matches


def test_parse_matches_groups_lines_by_relpath():
    out = "fastapi/app.py:10:def foo\nfastapi/app.py:42:foo again\nfastapi/other.py:1:foo\n"
    assert parse_matches(out) == {
        "fastapi/app.py": [10, 42],
        "fastapi/other.py": [1],
    }


def test_parse_matches_skips_malformed_lines():
    assert parse_matches("garbage\nfile.py:notanint:x\nok.py:3:x\n") == {"ok.py": [3]}


@pytest.fixture()
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "def alpha():\n    return needle\n\ndef beta():\n    return 2\n"
    )
    (tmp_path / "pkg" / "b.py").write_text("needle = 1\n")
    return tmp_path


def test_rg_matches_finds_lines_in_untracked_files(repo):
    matches = rg_matches(repo, "needle")
    assert matches == {"pkg/a.py": [2], "pkg/b.py": [1]}


def test_rg_matches_no_hits_returns_empty(repo):
    assert rg_matches(repo, "no_such_token_xyz") == {}


def test_rg_matches_bad_repo_returns_empty(tmp_path):
    assert rg_matches(tmp_path / "missing", "x") == {}


def seed_chunks(conn):
    upsert_rows(conn, [
        Row(source="github_code", source_id="pkg/a.py#alpha", document="d",
            raw_content="def alpha(): ...",
            metadata={"path": "pkg/a.py", "start_line": 1, "end_line": 2}),
        Row(source="github_code", source_id="pkg/a.py#beta", document="d",
            raw_content="def beta(): ...",
            metadata={"path": "pkg/a.py", "start_line": 4, "end_line": 5}),
        Row(source="github_code", source_id="pkg/b.py#module", document="d",
            raw_content="needle = 1",
            metadata={"path": "pkg/b.py", "start_line": 1, "end_line": 1}),
    ])


def test_grep_code_maps_matches_to_containing_chunks(clean_db, repo):
    seed_chunks(clean_db)
    results = grep_code(clean_db, repo, "needle")
    ids = [r.source_id for r in results]
    assert "pkg/a.py#alpha" in ids  # line 2 falls in [1, 2]
    assert "pkg/b.py#module" in ids
    assert "pkg/a.py#beta" not in ids


def test_grep_code_scores_by_match_count(clean_db, repo):
    (repo / "pkg" / "b.py").write_text("needle = 1\nneedle2 = needle\n")
    seed_chunks(clean_db)
    results = grep_code(clean_db, repo, "needle")
    assert results[0].source_id == "pkg/b.py#module"  # 1 in-range + fallback ordering
    assert results[0].score >= results[-1].score


def test_grep_code_unindexed_path_falls_back_to_any_row_for_path(clean_db, repo):
    # match on a line outside every chunk range still surfaces the file's rows
    (repo / "pkg" / "a.py").write_text("x = 0\n" * 98 + "tail_needle = 1\n")
    seed_chunks(clean_db)
    results = grep_code(clean_db, repo, "tail_needle")
    assert [r.source_id for r in results] == ["pkg/a.py#alpha"]


def test_grep_code_no_matches(clean_db, repo):
    seed_chunks(clean_db)
    assert grep_code(clean_db, repo, "zzz_nothing") == []
