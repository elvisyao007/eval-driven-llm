"""Embedders.

Two are real:
  - HashingBoWEmbedder: a deterministic lexical baseline (character bigram
    hashing). No GPU, no downloads. It is a legitimate (if weak) embedding and
    a legitimate eval baseline — semantic models are measured *against* it.
  - SentenceTransformerEmbedder: the production semantic embedder (ruri-v3 by
    default). Requires the model weights on the target machine.

Selecting one is a config choice (configs/models.yaml). The eval harness does
not care which it is — that's the point of measuring against a frozen golden
set (DECISIONS.md ADR-0001/0003).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_docs(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingBoWEmbedder:
    """Deterministic character-bigram hashing embedder (lexical baseline).

    Tokenizer-free (works for Japanese), fully reproducible, CPU-only. Documents
    and queries sharing surface terms get higher cosine similarity, so relevant
    passages rank above unrelated ones — enough to validate the pipeline and to
    serve as the lexical baseline the semantic models must beat.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _bigrams(self, text: str) -> list[str]:
        text = (text or "").strip()
        if len(text) < 2:
            return [text] if text else []
        return [text[i : i + 2] for i in range(len(text) - 1)]

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._bigrams(text):
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        return v

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.vstack([self._vec(t) for t in texts]))

    # lexical baseline treats queries and docs symmetrically
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class SentenceTransformerEmbedder:
    """Production semantic embedder (ruri-v3 default).

    Ruri-style models expect instruction prefixes on queries vs documents; the
    exact prefix strings live in config and MUST be verified against the model
    card, because a wrong prefix silently degrades retrieval.
    """

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-310m",
        query_prefix: str = "検索クエリ: ",
        doc_prefix: str = "検索文書: ",
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        out = self.model.encode(
            [self.query_prefix + t for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return out.astype(np.float32)

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        out = self.model.encode(
            [self.doc_prefix + t for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return out.astype(np.float32)


def build_embedder(name: str, **kwargs) -> Embedder:
    if name in ("hashing", "lexical", "dummy"):
        return HashingBoWEmbedder(dim=int(kwargs.get("dim", 1024)))
    if name in ("ruri", "sentence-transformer", "st"):
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name}")
