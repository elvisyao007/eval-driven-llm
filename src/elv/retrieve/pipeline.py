"""Two-stage retrieval: dense first pass -> optional rerank (ADR-0002).

Reranking is config-gated (BGE-Reranker-v2-m3 by default). Empirically a wide
first pass + rerank beats a single dense pass on P@1, which is why the reranker
is the precision workhorse, not an optional nicety.
"""

from __future__ import annotations

from typing import Sequence


class Retriever:
    def __init__(self, embedder, store, reranker=None, widen: int = 50) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker  # None -> dense-only (the baseline)
        self.widen = widen
        self._text_by_id: dict[str, str] = {}  # cross-encoder needs the text

    def index_corpus(self, doc_ids: Sequence[str], texts: Sequence[str], metadata=None):
        doc_ids, texts = list(doc_ids), list(texts)
        vecs = self.embedder.encode_docs(texts)
        meta = list(metadata) if metadata is not None else [{} for _ in doc_ids]
        self.store.upsert(doc_ids, vecs, meta)
        self._text_by_id.update(dict(zip(doc_ids, texts)))

    def retrieve(self, query: str, k: int = 5, acl_filter: dict | None = None) -> list[str]:
        qv = self.embedder.encode_queries([query])[0]
        # over-fetch when reranking: a wide first pass + rerank beats a single
        # dense pass on top-1 precision (ADR-0002).
        fetch = self.widen if self.reranker else k
        hits = self.store.search(qv, fetch, acl_filter=acl_filter)
        ranked = [doc_id for doc_id, _ in hits]
        if self.reranker:
            pairs = [(doc_id, self._text_by_id.get(doc_id, "")) for doc_id in ranked]
            ranked = self.reranker.rerank(query, pairs)
        return ranked[:k]
