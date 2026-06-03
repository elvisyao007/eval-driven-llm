"""Stable document ids and content hashes.

The same function must produce the doc_id used in golden labels and the doc_id
used when indexing the corpus, or retrieval metrics score against the wrong
ids. Centralized here so they can never drift.
"""
from __future__ import annotations

import hashlib


def content_hash(title: str, text: str) -> str:
    h = hashlib.sha256()
    h.update((title or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def stable_doc_id(title: str, text: str) -> str:
    """Deterministic id from content. Doubles as a dedup key in ingestion."""
    return content_hash(title, text)[:16]
