"""Deterministic retrieval metrics.

No LLM judge, no randomness: given a ranked list of retrieved document ids and
the set of relevant ids from a frozen golden set, these scores are fully
reproducible. This is the part of the eval that anchors everything else
(see DECISIONS.md ADR-0001).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _check(retrieved: Sequence[str], relevant: set[str]) -> None:
    if not isinstance(relevant, set):
        raise TypeError("relevant must be a set of doc ids")
    if len(set(retrieved)) != len(retrieved):
        # Duplicates in a ranked list silently corrupt rank-based metrics.
        raise ValueError("retrieved list contains duplicate ids")


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found within the top-k."""
    _check(retrieved, relevant)
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / len(relevant)


def precision_at_1(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1.0 if the top-ranked doc is relevant, else 0.0."""
    _check(retrieved, relevant)
    if not retrieved:
        return 0.0
    return 1.0 if retrieved[0] in relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant doc (0 if none in list)."""
    _check(retrieved, relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k. Ideal DCG assumes all relevant docs ranked first."""
    _check(retrieved, relevant)
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(
    runs: list[tuple[Sequence[str], set[str]]],
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Mean metrics across a golden set.

    `runs` is a list of (retrieved_ids, relevant_ids) per query.
    Returns macro-averaged metrics — fully deterministic given the inputs.
    """
    if not runs:
        return {}
    n = len(runs)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in runs) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, rel, k) for r, rel in runs) / n
    out["mrr"] = sum(reciprocal_rank(r, rel) for r, rel in runs) / n
    out["p@1"] = sum(precision_at_1(r, rel) for r, rel in runs) / n
    return out
