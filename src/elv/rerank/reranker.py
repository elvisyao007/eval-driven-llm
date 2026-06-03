"""Rerankers — second stage of two-stage retrieval (ADR-0002).

Two are real:
  - LexicalOverlapReranker: deterministic bigram-Jaccard reorder. No downloads,
    CPU. A genuine (weak) scorer used to validate the rerank plumbing and as a
    lexical reranking baseline.
  - CrossEncoderReranker: production cross-encoder (BGE-Reranker-v2-m3). Scores
    each (query, passage) pair jointly — this is the precision workhorse that a
    bi-encoder dense pass cannot match, at the cost of latency. Needs weights on
    the target machine.

Interface: rerank(query, candidates) -> reordered doc_ids (most relevant first).
`candidates` is [(doc_id, passage_text)] from the first-pass retriever.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[str]: ...


def _bigrams(text: str) -> set[str]:
    text = (text or "").strip()
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


class LexicalOverlapReranker:
    """Reorder candidates by bigram-Jaccard overlap with the query.

    Deterministic; ties keep the first-pass order (stable). A real, if crude,
    reranking signal — the cross-encoder must beat it to justify its latency.
    """

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[str]:
        qb = _bigrams(query)

        def score(text: str) -> float:
            tb = _bigrams(text)
            if not qb or not tb:
                return 0.0
            return len(qb & tb) / len(qb | tb)

        order = sorted(
            range(len(candidates)),
            key=lambda i: (-score(candidates[i][1]), i),  # stable tie-break
        )
        return [candidates[i][0] for i in order]


class CrossEncoderReranker:
    """Production cross-encoder reranker (BGE-Reranker-v2-m3 by default)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True) -> None:
        from FlagEmbedding import FlagReranker  # lazy import; needs weights

        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[str]:
        if not candidates:
            return []
        scores = self.model.compute_score([[query, text] for _, text in candidates])
        if not isinstance(scores, list):
            scores = [scores]
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-scores[i], i),  # stable tie-break
        )
        return [candidates[i][0] for i in order]


def build_reranker(name: str, **kwargs) -> Reranker | None:
    if name in ("none", None, ""):
        return None
    if name in ("lexical", "baseline"):
        return LexicalOverlapReranker()
    if name in ("cross-encoder", "bge", "bge-reranker-v2-m3"):
        return CrossEncoderReranker(**kwargs)
    raise ValueError(f"unknown reranker: {name}")
