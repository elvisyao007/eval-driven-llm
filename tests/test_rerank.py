"""Reranker plumbing tests — deterministic lexical reranker behaviour and the
compare path's per-query accounting. The cross-encoder path needs weights and
is not exercised here (it is config-swapped on the target machine)."""

from elv.rerank.reranker import LexicalOverlapReranker, build_reranker


def test_lexical_reranker_promotes_overlap():
    rr = LexicalOverlapReranker()
    q = "オンプレミスでLLMを動かす"
    cands = [
        ("d_far", "野球の試合は九回裏に逆転した"),
        ("d_near", "オンプレミスでLLMを動かす構成について"),
    ]
    assert rr.rerank(q, cands)[0] == "d_near"


def test_lexical_reranker_stable_on_ties():
    rr = LexicalOverlapReranker()
    cands = [("a", "無関係"), ("b", "無関係")]  # equal overlap -> keep input order
    assert rr.rerank("クエリ", cands) == ["a", "b"]


def test_lexical_reranker_empty():
    assert LexicalOverlapReranker().rerank("q", []) == []


def test_build_reranker_none():
    assert build_reranker("none") is None
    assert build_reranker("") is None


def test_per_query_p1_accounting():
    # baseline gets q1 wrong, q2 right; tuned flips q1 right, q2 wrong
    from elv.eval.runner import per_query_p1
    from elv.eval.golden import GoldenSet, GoldenQuery

    g = GoldenSet(
        name="t", version="v0",
        queries=[
            GoldenQuery(id="q1", query="a", relevant_doc_ids={"A"}),
            GoldenQuery(id="q2", query="b", relevant_doc_ids={"B"}),
        ],
        doc_id_hashes={},
    )
    base = per_query_p1(g, lambda q: ["X", "A"] if q == "a" else ["B"])
    tuned = per_query_p1(g, lambda q: ["A"] if q == "a" else ["Y", "B"])
    assert base == {"q1": 0, "q2": 1}
    assert tuned == {"q1": 1, "q2": 0}
