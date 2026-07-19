from knowbase.cli import _first_line


def test_first_line_truncates():
    assert _first_line("a" * 200) == "a" * 100


def test_first_line_takes_first_line_only():
    assert _first_line("title\nbody") == "title"


def test_first_line_empty_string():
    assert _first_line("") == ""
