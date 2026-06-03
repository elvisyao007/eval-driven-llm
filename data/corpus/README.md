# Corpora

Source documents the system indexes. Two tracks (see ADR-0004):

- **Track A — JQaRA** (recognized Japanese retrieval-augmented QA dataset).
  Used for credible, repeatable retrieval metrics. Verify its license before
  redistributing anything derived from it.
- **Track B — dirty real PDFs** (public-domain Japanese documents: multi-column
  layouts, tables, ruby text, OCR noise). Exercises the dirty-data pipeline.
  Verify each document's license/source; prefer public-domain government
  material and record provenance.

Do not commit documents you do not have the right to redistribute. Prefer a
fetch script + hashes (see `data/golden/README.md`).


## Ingestion pipeline (Track B / dirty data)

`elv.ingest.loader` turns a directory of real documents into `passages.jsonl`
(the schema the eval corpus loader reads) plus `audit.json`. Stages:
parse (tolerant, multi-format, JP encoding fallback) -> normalize (NFKC, control/
zero-width strip, de-hyphenation; mojibake flagged not silently fixed) -> dedup
(exact + SimHash near-dup) -> structure-aware chunk (carries owner/ACL metadata
for per-user retrieval). A single bad file is logged and skipped, never crashes
the batch. Rebuild the demo fixtures with `make build-dirty-fixtures` and ingest
with `make ingest-dirty`.
