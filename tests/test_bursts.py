from knowbase.ingest.bursts import Burst, make_burst_scorer, split_bursts


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


IDF = {"rare": 5.0, "common": 0.5}


def lexemize(text):
    return text.lower().split()


def test_rare_token_scores_one():
    score = make_burst_scorer(IDF, lexemize)
    assert score(Burst("a", "rare", 0)) == 1


def test_common_tokens_score_zero():
    score = make_burst_scorer(IDF, lexemize)
    assert score(Burst("a", "common common", 0)) == 0


def test_length_scores_one():
    score = make_burst_scorer(IDF, lexemize, min_chars=10)
    assert score(Burst("a", "common " * 3, 0)) == 1


def test_reactions_score_one():
    score = make_burst_scorer(IDF, lexemize)
    assert score(Burst("a", "common", 2)) == 1


def test_signals_are_additive():
    score = make_burst_scorer(IDF, lexemize, min_chars=4)
    assert score(Burst("a", "rare and long", 1)) == 3


def test_empty_idf_disables_rare_token_signal_only():
    score = make_burst_scorer({}, lexemize, min_chars=4)
    assert score(Burst("a", "anything here", 0)) == 1  # length only
