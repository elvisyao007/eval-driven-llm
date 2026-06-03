"""Deduplication: exact + near-duplicate.

Exact dedup is by content hash. Near-dup uses 64-bit SimHash + Hamming distance,
which is dependency-free and far cheaper than pairwise Jaccard. The pairwise
Hamming scan here is fine for repo-scale corpora; at production scale swap in
LSH banding over the SimHash bits (documented, not implemented — a deliberate
scope choice).

SimHash near-dup is conservative: it is reliable on longer texts but noisy on
short ones, so the default threshold (3) catches only very close duplicates.
Raise the threshold or switch to shingled-Jaccard/MinHash for aggressive
near-dup matching — a tunable, not a silent default.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence


def _shingles(text: str, n: int = 5) -> set[str]:
    text = re.sub(r"\s+", "", text or "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def simhash(text: str, bits: int = 64) -> int:
    v = [0] * bits
    for sh in _shingles(text):
        h = int(hashlib.md5(sh.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_duplicates(
    texts: Sequence[str], near_threshold: int = 3
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return (keep_indices, dropped_pairs).

    Exact duplicates and near-duplicates (Hamming distance <= near_threshold on
    SimHash) are dropped, keeping the first occurrence. `dropped_pairs` is
    [(dropped_idx, kept_idx)] for the audit log."""
    seen_exact: dict[str, int] = {}
    kept: list[int] = []
    kept_sims: list[tuple[int, int]] = []  # (simhash, index)
    dropped: list[tuple[int, int]] = []

    for i, t in enumerate(texts):
        digest = hashlib.sha256((t or "").encode("utf-8")).hexdigest()
        if digest in seen_exact:
            dropped.append((i, seen_exact[digest]))
            continue
        sh = simhash(t)
        near = next((idx for s, idx in kept_sims if _hamming(s, sh) <= near_threshold), None)
        if near is not None:
            dropped.append((i, near))
            continue
        seen_exact[digest] = i
        kept_sims.append((sh, i))
        kept.append(i)
    return kept, dropped
