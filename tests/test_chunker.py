from knowbase.ingest.chunker import chunk_python

SMALL = '''import os

def helper():
    return 1
'''

TWO_CLASSES = (
    "class Alpha:\n"
    + "".join(f"    def a{i}(self):\n        return {i}  # {'x' * 40}\n" for i in range(10))
    + "\n\nclass Beta:\n"
    + "".join(f"    def b{i}(self):\n        return {i}  # {'x' * 40}\n" for i in range(10))
)


def test_small_file_is_one_chunk():
    chunks = chunk_python(SMALL, max_chars=2000)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].text == SMALL.rstrip("\n")


def test_splits_on_class_boundaries_first():
    chunks = chunk_python(TWO_CLASSES, max_chars=700)
    symbols = [c.symbol for c in chunks]
    assert "Alpha" in symbols
    assert "Beta" in symbols


def test_oversized_class_falls_back_to_methods():
    chunks = chunk_python(TWO_CLASSES, max_chars=200)
    assert all(len(c.text) <= 200 or c.symbol for c in chunks)
    assert any(c.symbol and c.symbol.startswith("a") for c in chunks)


def test_line_numbers_cover_the_file():
    chunks = chunk_python(TWO_CLASSES, max_chars=700)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == len(TWO_CLASSES.splitlines())


def test_hard_split_when_no_boundaries():
    blob = "\n".join(f"x{i} = {i}" for i in range(500))
    chunks = chunk_python(blob, max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
