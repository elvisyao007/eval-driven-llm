"""Dirty-data ingestion pipeline (Track B).

Orchestrates: walk -> parse (tolerant) -> normalize -> dedup -> chunk -> audit.
Emits passages.jsonl in the SAME schema the eval corpus loader reads, so dirty
documents flow straight into index -> retrieve -> eval. Also writes audit.json:
what was parsed, what failed, what was normalized, what was deduped — the
auditability requirement that turns a black box into something a customer's IT
can trust and maintain (failure-mapping: IT 部門不配合 / 安全性担忧).

Default assumption is enterprise reality, not a clean notebook corpus.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..eval.ids import stable_doc_id
from . import chunk as _chunk
from . import dedup as _dedup
from . import normalize as _normalize
from .parse import parse_file


@dataclass
class IngestReport:
    n_files: int = 0
    n_parsed_ok: int = 0
    n_failed: int = 0
    failures: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    normalizations: dict = field(default_factory=dict)  # change -> count
    n_chunks_before_dedup: int = 0
    n_dropped_duplicates: int = 0
    n_chunks_out: int = 0


def ingest_corpus(corpus_dir, out_path, owner: str = "default",
                  max_chars: int = 512, overlap: int = 64,
                  near_threshold: int = 3):
    rep = IngestReport()
    raw_chunks: list[_chunk.Chunk] = []

    files = sorted(p for p in Path(corpus_dir).rglob("*") if p.is_file())
    rep.n_files = len(files)

    for p in files:
        result = parse_file(p)
        if not result.ok:
            rep.n_failed += 1
            rep.failures.append({"file": result.path, "error": result.error})
            continue  # one bad file does not kill the batch
        rep.n_parsed_ok += 1
        for w in result.warnings:
            rep.warnings.append({"file": result.path, "warning": w})
        for doc in result.docs:
            norm, changes = _normalize.normalize(doc.text)
            for k, v in changes.items():
                rep.normalizations[k] = rep.normalizations.get(k, 0) + (v if isinstance(v, int) else 1)
            if not norm:
                continue
            meta = {**doc.metadata, "owner": owner, "title": doc.title}
            raw_chunks.extend(_chunk.chunk_text(norm, max_chars, overlap, meta))

    rep.n_chunks_before_dedup = len(raw_chunks)
    keep, dropped = _dedup.find_duplicates([c.text for c in raw_chunks], near_threshold)
    rep.n_dropped_duplicates = len(dropped)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i in keep:
            c = raw_chunks[i]
            title = c.metadata.get("title", "")
            rec = {"doc_id": stable_doc_id(title, c.text), "title": title,
                   "text": c.text, **{k: v for k, v in c.metadata.items() if k != "title"}}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rep.n_chunks_out = len(keep)

    audit = out.parent / "audit.json"
    audit.write_text(json.dumps(asdict(rep), ensure_ascii=False, indent=2), "utf-8")
    return rep, str(out), str(audit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir")
    ap.add_argument("--out", default="data/corpus/_ingested/passages.jsonl")
    ap.add_argument("--owner", default="default", help="ACL owner for these docs")
    ap.add_argument("--max-chars", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    args = ap.parse_args()
    rep, out, audit = ingest_corpus(
        args.corpus_dir, args.out, owner=args.owner,
        max_chars=args.max_chars, overlap=args.overlap)
    print(f"wrote {out}")
    print(f"audit {audit}")
    for k, v in asdict(rep).items():
        if not isinstance(v, list) or v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
