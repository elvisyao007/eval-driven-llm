# Golden evaluation sets

A golden set is **frozen and versioned**. Once a version is cut, it does not
change — otherwise metrics stop being comparable across runs.

## Layout

```
data/golden/<name>/<version>/
  queries.jsonl      # {id, query, relevant_doc_ids:[...], reference_answer?}
  manifest.json      # {name, version, created, source, doc_id_hashes, license}
```

## Rules

- Version with a date or semver tag; never edit a cut version, add a new one.
- `manifest.json` records the corpus document hashes so a run can verify it
  scored against the exact documents the labels refer to.
- **Do not commit copyrighted documents.** When the corpus license does not
  allow redistribution, commit labels + document ids/hashes + a fetch script
  instead of the documents (see ADR-0004).
- Track A (JQaRA) and Track B (dirty PDFs) each get their own named golden set.
