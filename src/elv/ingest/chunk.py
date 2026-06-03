"""Structure-aware chunking.

Splits on blank-line / heading boundaries first, then packs blocks into chunks
up to `max_chars` with overlap, so a chunk rarely cuts mid-paragraph. A single
oversized block is hard-split with overlap as a fallback. Each chunk keeps its
source metadata (incl. owner/ACL) so retrieval can filter per user (ADR-0002).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_BLOCK = re.compile(r"\n\s*\n|\n(?=#{1,6}\s)")  # blank line or before a heading


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    out, i = [], 0
    step = max(1, max_chars - overlap)
    while i < len(text):
        out.append(text[i : i + max_chars])
        i += step
    return out


def chunk_text(text: str, max_chars: int = 512, overlap: int = 64,
               metadata: dict | None = None) -> list[Chunk]:
    meta = dict(metadata or {})
    blocks = [b.strip() for b in _BLOCK.split(text) if b and b.strip()]
    chunks: list[str] = []
    buf = ""
    for b in blocks:
        if len(b) > max_chars:
            if buf:
                chunks.append(buf); buf = ""
            chunks.extend(_hard_split(b, max_chars, overlap))
            continue
        if len(buf) + len(b) + 1 <= max_chars:
            buf = f"{buf}\n{b}" if buf else b
        else:
            chunks.append(buf); buf = b
    if buf:
        chunks.append(buf)
    return [
        Chunk(text=c, metadata={**meta, "chunk_index": i, "char_len": len(c)})
        for i, c in enumerate(chunks)
    ]
