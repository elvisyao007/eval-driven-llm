"""Frozen golden-set loader.

A golden set is immutable once cut. The loader verifies the run is scoring
against the exact corpus the labels refer to, via the document hashes recorded
in the manifest (see data/golden/README.md, DECISIONS.md ADR-0004).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    relevant_doc_ids: set[str]
    reference_answer: str | None = None


@dataclass(frozen=True)
class GoldenSet:
    name: str
    version: str
    queries: list[GoldenQuery]
    doc_id_hashes: dict[str, str]  # doc_id -> content hash


def load(path: str | Path) -> GoldenSet:
    """Load a frozen golden set from data/golden/<name>/<version>/."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    queries: list[GoldenQuery] = []
    with (root / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries.append(
                GoldenQuery(
                    id=obj["id"],
                    query=obj["query"],
                    relevant_doc_ids=set(obj["relevant_doc_ids"]),
                    reference_answer=obj.get("reference_answer"),
                )
            )
    return GoldenSet(
        name=manifest["name"],
        version=manifest["version"],
        queries=queries,
        doc_id_hashes=manifest.get("doc_id_hashes", {}),
    )


def verify_corpus(golden: GoldenSet, actual_hashes: dict[str, str]) -> list[str]:
    """Return doc ids whose content hash does not match the frozen manifest.
    A non-empty result means the run is scoring against a different corpus than
    the labels assume → metrics are not comparable. Fail loudly."""
    mismatches = []
    for doc_id, expected in golden.doc_id_hashes.items():
        if actual_hashes.get(doc_id) != expected:
            mismatches.append(doc_id)
    return mismatches
