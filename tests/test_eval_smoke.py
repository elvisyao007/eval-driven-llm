"""Smoke test: the deterministic eval core runs and is correct.

This is the proof that 'the harness produces repeatable scores' is real and not
a promise — these assertions are exact because retrieval metrics are
deterministic (DECISIONS.md ADR-0001)."""

from elv.eval import metrics_retrieval as m


def test_perfect_ranking():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert m.precision_at_1(retrieved, relevant) == 1.0
    assert m.recall_at_k(retrieved, relevant, 2) == 1.0
    assert m.reciprocal_rank(retrieved, relevant) == 1.0


def test_first_relevant_at_rank_2():
    retrieved = ["x", "a", "b"]
    relevant = {"a"}
    assert m.precision_at_1(retrieved, relevant) == 0.0
    assert m.reciprocal_rank(retrieved, relevant) == 0.5


def test_recall_partial():
    retrieved = ["a", "x", "y"]
    relevant = {"a", "b"}
    assert m.recall_at_k(retrieved, relevant, 3) == 0.5


def test_ndcg_monotonic():
    relevant = {"a"}
    top = m.ndcg_at_k(["a", "x"], relevant, 2)
    low = m.ndcg_at_k(["x", "a"], relevant, 2)
    assert top > low


def test_aggregate_runs():
    runs = [(["a", "b"], {"a"}), (["x", "a"], {"a"})]
    out = m.aggregate(runs, ks=(2,))
    assert out["p@1"] == 0.5
    assert out["mrr"] == 0.75  # (1.0 + 0.5) / 2


def test_duplicate_ids_rejected():
    import pytest
    with pytest.raises(ValueError):
        m.recall_at_k(["a", "a"], {"a"}, 2)
