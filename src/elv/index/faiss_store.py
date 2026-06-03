"""Faiss-backed vector store (benchmark track).

Cosine similarity via inner product on L2-normalized vectors. Note the
limitation that justifies ADR-0002: Faiss has no native metadata filtering, so
per-user permission (`acl_filter`) is enforced by post-filtering here — fine for
a benchmark, not acceptable for multi-tenant production, which is why Qdrant is
the customer-facing store.
"""

from __future__ import annotations

import numpy as np


class FaissStore:
    def __init__(self, dim: int) -> None:
        import faiss  # lazy import

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.metadata: list[dict] = []

    def upsert(self, ids, vectors: np.ndarray, metadata) -> None:
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index.add(vectors)
        self.ids.extend(ids)
        self.metadata.extend(metadata)

    def search(self, vector: np.ndarray, k: int, acl_filter: dict | None = None):
        # over-fetch when post-filtering by ACL (Faiss limitation, see ADR-0002)
        fetch = k * 5 if acl_filter else k
        q = np.ascontiguousarray(vector.reshape(1, -1).astype(np.float32))
        scores, idxs = self.index.search(q, min(fetch, len(self.ids)))
        out: list[tuple[str, float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            if acl_filter and not _acl_ok(self.metadata[i], acl_filter):
                continue
            out.append((self.ids[i], float(score)))
            if len(out) >= k:
                break
        return out


def _acl_ok(meta: dict, acl_filter: dict) -> bool:
    for key, allowed in acl_filter.items():
        if meta.get(key) not in allowed:
            return False
    return True
