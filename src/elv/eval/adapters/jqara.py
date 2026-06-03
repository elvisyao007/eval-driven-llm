"""JQaRA -> frozen corpus + golden set (Track A, ADR-0004).

JQaRA (hotchpotch/JQaRA) rows pair a question with a candidate Wikipedia passage
and a relevance `label`. Rows are grouped by `q_id`; passages with label==1 are
the relevant set for that question. License: CC BY-SA 4.0 (verify before
redistributing derived data — commit hashes + a build script, not the corpus).

Assumed columns (verify against the live dataset card; change in one place):
    q_id, question, answers, title, text, label
The dataset ships no passage id, so we synthesize a stable id from content
(elv.eval.ids.stable_doc_id), which also dedups identical passages.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ids import content_hash, stable_doc_id

COL_QID = "q_id"
COL_QUESTION = "question"
COL_TITLE = "title"
COL_TEXT = "text"
COL_LABEL = "label"


def build_frozen_set(split: str, version: str, out_root: str | Path) -> Path:
    """Materialize a frozen golden set + corpus from a JQaRA split.

    Requires `datasets` and network access to Hugging Face on the target
    machine. Writes:
        data/golden/jqara/<version>/queries.jsonl
        data/golden/jqara/<version>/manifest.json
        data/corpus/jqara/<version>/passages.jsonl
    """
    from datasets import load_dataset  # lazy import

    ds = load_dataset("hotchpotch/JQaRA", split=split)

    corpus: dict[str, dict] = {}     # doc_id -> {title, text}
    by_q: dict[str, dict] = {}       # q_id -> {query, relevant:set}

    for row in ds:
        doc_id = stable_doc_id(row[COL_TITLE], row[COL_TEXT])
        corpus.setdefault(doc_id, {"title": row[COL_TITLE], "text": row[COL_TEXT]})
        q = by_q.setdefault(
            str(row[COL_QID]), {"query": row[COL_QUESTION], "relevant": set()}
        )
        if int(row[COL_LABEL]) == 1:
            q["relevant"].add(doc_id)

    gdir = Path(out_root) / "data" / "golden" / "jqara" / version
    cdir = Path(out_root) / "data" / "corpus" / "jqara" / version
    gdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)

    with (cdir / "passages.jsonl").open("w", encoding="utf-8") as fh:
        for doc_id, d in corpus.items():
            fh.write(json.dumps({"doc_id": doc_id, **d}, ensure_ascii=False) + "\n")

    with (gdir / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for qid, q in by_q.items():
            fh.write(json.dumps(
                {"id": qid, "query": q["query"],
                 "relevant_doc_ids": sorted(q["relevant"])},
                ensure_ascii=False) + "\n")

    manifest = {
        "name": "jqara",
        "version": version,
        "split": split,
        "source": "hotchpotch/JQaRA (CC BY-SA 4.0)",
        "corpus_path": str(cdir / "passages.jsonl"),
        "n_queries": len(by_q),
        "n_passages": len(corpus),
        "doc_id_hashes": {
            d: content_hash(v["title"], v["text"]) for d, v in corpus.items()
        },
    }
    (gdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return gdir
