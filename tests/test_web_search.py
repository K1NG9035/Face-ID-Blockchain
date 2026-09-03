from app.web_search import Candidate, unique_candidates


def test_candidates_are_valid_and_deduplicated():
    candidates = unique_candidates([Candidate("https://example.com/a"), Candidate("https://example.com/a"), Candidate("ftp://bad")])
    assert candidates == [Candidate("https://example.com/a")]
