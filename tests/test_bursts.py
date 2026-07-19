from knowbase.ingest.bursts import Burst, split_bursts


def c(author, body, reactions=0):
    return {"author": author, "body": body, "reactions": reactions}


def test_consecutive_comments_by_same_author_merge():
    bursts = split_bursts([c("a", "one"), c("a", "two"), c("b", "three")])
    assert [b.author for b in bursts] == ["a", "b"]
    assert bursts[0].text == "one\n\ntwo"
    assert bursts[1].text == "three"


def test_alternating_authors_do_not_merge():
    bursts = split_bursts([c("a", "1"), c("b", "2"), c("a", "3")])
    assert [b.author for b in bursts] == ["a", "b", "a"]


def test_reactions_sum_within_a_burst():
    bursts = split_bursts([c("a", "one", 2), c("a", "two", 1)])
    assert bursts[0].reactions == 3


def test_empty_bodies_are_skipped():
    bursts = split_bursts([c("a", "  "), c("b", "real")])
    assert [b.author for b in bursts] == ["b"]


def test_no_comments_no_bursts():
    assert split_bursts([]) == []
