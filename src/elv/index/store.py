"""Vector store abstraction.

Interface-first so Faiss (benchmark track) and Qdrant (realism / customer track)
are config swaps. Qdrant is chosen for customer-facing work specifically because
its metadata filtering enforces per-user permissions at query time (ADR-0002).
"""

from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors, metadata: list[dict]) -> None: ...
    def search(
        self, vector, k: int, acl_filter: dict | None = None
    ) -> list[tuple[str, float]]:
        """Return [(doc_id, score)]. `acl_filter` enforces per-user permissions
        — a store that cannot do this is unsuitable for multi-user deployment."""
        ...
