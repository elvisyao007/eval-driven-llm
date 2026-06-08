#!/usr/bin/env python3
"""
Metric analysis: proportion recall vs hit@k on JQaRA (multi-answer dataset).

Pure data analysis — reads existing rag_results.json, no model loading.

Motivation:
  JQaRA has 6–28 relevant docs per query (mean 9.7). QA only needs 1 relevant
  doc to answer correctly. The proportion recall used in the baseline
  (context_recall_docs = |retrieved ∩ relevant| / |relevant|) penalises a query
  that retrieved 2 of 23 relevant docs as "0.087 recall" — but those 2 docs may
  be sufficient to answer the question. This analysis checks whether the 33/100
  grounded-but-wrong count is inflated by that denominator artifact.

Definitions:
  - proportion_recall (existing): context_recall_docs = |retrieved ∩ relevant| / |relevant|
  - hit@5: 1 if proportion_recall > 0, else 0  (at least 1 relevant doc in top-5)
  - grounded_but_wrong (existing): faithfulness >= 0.8 AND proportion_recall < 0.5
  - grounded_but_wrong_hit: faithfulness >= 0.8 AND hit@5 = 0
      (faithful answer with ZERO relevant docs retrieved — genuine retrieval failure)

Usage:
  python scripts/recall_metric_analysis.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAG_PATH    = ROOT / "reports/20260605T024628-gen/rag_results.json"
GOLDEN_PATH = ROOT / "data/golden/jqara/v0/queries.jsonl"
OUT_PATH    = ROOT / "reports/recall_metric_analysis.md"

FAITH_HI  = 0.8   # faithfulness threshold (unchanged from baseline)
RECALL_LO = 0.5   # proportion recall threshold for grounded-but-wrong (baseline definition)


def _load_n_rel() -> dict[str, int]:
    n_rel = {}
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            n_rel[q["id"]] = len(q["relevant_doc_ids"])
    return n_rel


def main() -> None:
    d = json.loads(RAG_PATH.read_text(encoding="utf-8"))
    pq = d["per_query"]         # {q_id: {faithfulness, context_recall_docs, grounded_but_wrong, ...}}
    n_rel = _load_n_rel()

    # ── per-query derived fields ───────────────────────────────────────────────
    rows = []
    for qid, v in pq.items():
        faith   = v["faithfulness"]
        p_recall = v["context_recall_docs"]   # proportion recall (existing)
        hit5    = int(p_recall > 0.0)         # binary: ≥1 relevant doc in top-5
        gbw_old = v["grounded_but_wrong"]     # existing definition
        gbw_new = int(faith >= FAITH_HI and hit5 == 0)   # hit-based definition
        nr      = n_rel.get(qid, None)
        rows.append({
            "qid": qid, "faith": faith, "p_recall": p_recall,
            "hit5": hit5, "gbw_old": gbw_old, "gbw_new": gbw_new, "n_rel": nr,
        })

    n = len(rows)

    # ── aggregate stats ────────────────────────────────────────────────────────
    mean_faith    = statistics.fmean(r["faith"]   for r in rows)
    mean_p_recall = statistics.fmean(r["p_recall"] for r in rows)
    mean_hit5     = statistics.fmean(r["hit5"]    for r in rows)

    # grounded-but-wrong counts
    gbw_old_n = sum(r["gbw_old"] for r in rows)   # should match existing 33
    gbw_new_n = sum(r["gbw_new"] for r in rows)   # hit-based definition

    # faith>=0.8 group
    faith_hi_rows = [r for r in rows if r["faith"] >= FAITH_HI]

    # decompose old grounded-but-wrong (33) into two classes
    gbw_true_fail   = [r for r in rows if r["gbw_old"] and r["hit5"] == 0]
    gbw_artifact    = [r for r in rows if r["gbw_old"] and r["hit5"] == 1]

    # queries with faith>=0.8 and hit5=1 (faithful + at least 1 relevant doc)
    faith_hi_hit1   = [r for r in faith_hi_rows if r["hit5"] == 1]
    faith_hi_hit0   = [r for r in faith_hi_rows if r["hit5"] == 0]

    # hit@5 distribution across all 100 queries
    all_hit0 = [r for r in rows if r["hit5"] == 0]
    all_hit1 = [r for r in rows if r["hit5"] == 1]

    # n_rel distribution for grounded-but-wrong queries (old definition)
    gbw_nr = [r["n_rel"] for r in rows if r["gbw_old"] and r["n_rel"] is not None]
    non_gbw_nr = [r["n_rel"] for r in rows if not r["gbw_old"] and r["n_rel"] is not None]

    # ── build report ───────────────────────────────────────────────────────────
    lines = [
        "# Recall metric analysis: proportion recall vs hit@k on JQaRA",
        "",
        f"> Source: `{RAG_PATH.relative_to(ROOT)}`",
        f"> (Phase 4, gemma4:31b judge, qwen3:32b generator, 100-query sample)",
        "",
        "---",
        "",
        "## 1. Why proportion recall misrepresents multi-answer QA",
        "",
        "JQaRA pairs each query with 6–28 relevant Wikipedia passages (mean 9.7).",
        "The baseline metric `context_recall_docs = |retrieved ∩ relevant| / |relevant|`",
        "was designed for the **retrieval** task (how many relevant docs did we surface?).",
        "For **generation**, the question is different: did the model receive *at least one*",
        "relevant document to ground its answer? Retrieving 2 of 23 relevant passages is",
        "scored as 0.087 recall by the proportion metric, but those 2 passages may be",
        "perfectly sufficient to answer the question.",
        "",
        "This analysis computes `hit@5` (binary: ≥1 relevant doc in top-5) and tests",
        "whether the 33/100 grounded-but-wrong count is inflated by the denominator effect.",
        "",
        "| Metric | Definition | Sensitive to multi-answer denominator? |",
        "|---|---|---|",
        "| proportion_recall (existing) | \\|retrieved ∩ relevant\\| / \\|relevant\\| | Yes — penalises partial hits |",
        "| hit@5 (new) | 1 if proportion_recall > 0 else 0 | No — only asks \"did we hit any?\" |",
        "",
        "---",
        "",
        "## 2. Overall recall statistics (100 queries)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| mean faithfulness | {mean_faith:.4f} |",
        f"| mean proportion_recall@5 | {mean_p_recall:.4f} |",
        f"| mean hit@5 | **{mean_hit5:.4f}** |",
        f"| queries with hit@5=0 (zero relevant docs in top-5) | {len(all_hit0)} / {n} |",
        f"| queries with hit@5=1 (≥1 relevant doc in top-5) | {len(all_hit1)} / {n} |",
        "",
        f"**{len(all_hit1)} of {n} queries ({100*mean_hit5:.1f}%) had at least one relevant document**",
        f"in the retrieved top-5. Only {len(all_hit0)} queries had zero relevant docs.",
        "",
        "---",
        "",
        "## 3. Decomposing the 33 grounded-but-wrong queries",
        "",
        "Original definition: `faithfulness ≥ 0.8 AND proportion_recall < 0.5`",
        "",
        f"**Total grounded-but-wrong (original): {gbw_old_n} / {n}**",
        "",
        f"| Class | Definition | Count |",
        f"|---|---|---|",
        f"| **True failure** | faith≥0.8, hit@5=0 (zero relevant docs retrieved) | **{len(gbw_true_fail)}** |",
        f"| **Metric artifact** | faith≥0.8, hit@5≥1 but proportion_recall<0.5 | **{len(gbw_artifact)}** |",
        f"| Total | — | {gbw_old_n} |",
        "",
        f"**True failures** ({len(gbw_true_fail)}): the model had *no* relevant document in",
        "its context. High faithfulness here means it confabulated a plausible-sounding",
        "answer from irrelevant passages. This is the genuinely dangerous failure.",
        "",
        f"**Metric artifacts** ({len(gbw_artifact)}): the model *did* retrieve at least one",
        "relevant document (hit@5=1), but the proportion recall was < 0.5 because the",
        "denominator (total relevant docs) is large. The model likely had sufficient",
        "grounding to answer correctly — these are not retrieval failures in the QA sense.",
        "",
    ]

    # n_rel stats for the two classes
    if gbw_true_fail:
        fail_nr = [r["n_rel"] for r in gbw_true_fail if r["n_rel"]]
        lines.append(f"n_rel for true failures: min={min(fail_nr)}, "
                     f"median={sorted(fail_nr)[len(fail_nr)//2]}, "
                     f"max={max(fail_nr)}, mean={statistics.fmean(fail_nr):.1f}")
    if gbw_artifact:
        art_nr = [r["n_rel"] for r in gbw_artifact if r["n_rel"]]
        structurally_impossible = sum(1 for n in art_nr if 5 / n < 0.5)  # n_rel > 10
        lines.append(f"n_rel for metric artifacts: min={min(art_nr)}, "
                     f"median={sorted(art_nr)[len(art_nr)//2]}, "
                     f"max={max(art_nr)}, mean={statistics.fmean(art_nr):.1f}")
        art_p_recalls = sorted(r["p_recall"] for r in gbw_artifact)
        lines.append(f"proportion_recall for metric artifacts: "
                     f"min={art_p_recalls[0]:.4f}, max={art_p_recalls[-1]:.4f}, "
                     f"mean={statistics.fmean(art_p_recalls):.4f}")
        lines.append(f"**Structurally impossible to fix at k=5** (n_rel > 10, "
                     f"oracle proportion_recall@5 = 5/n_rel < 0.5): "
                     f"**{structurally_impossible} of {len(art_nr)}**. "
                     f"These queries would be labelled grounded-but-wrong by the original "
                     f"definition *even with a perfect oracle retriever* at k=5.")

    lines += [
        "",
        "---",
        "",
        "## 4. Grounded-but-wrong with hit@5 as the recall criterion",
        "",
        "New definition: `faithfulness ≥ 0.8 AND hit@5 = 0`",
        "(faithful answer AND zero relevant documents retrieved)",
        "",
        f"| Definition | grounded-but-wrong count |",
        f"|---|---|",
        f"| Original (proportion_recall<0.5) | {gbw_old_n} / {n} |",
        f"| **Revised (hit@5=0)** | **{gbw_new_n} / {n}** |",
        f"| Difference | −{gbw_old_n - gbw_new_n} ({gbw_old_n - gbw_new_n} queries reclassified as non-failure) |",
        "",
        "**faith≥0.8 group breakdown:**",
        "",
        f"| Subgroup | Count | Interpretation |",
        f"|---|---|---|",
        f"| faith≥0.8, hit@5=0 (true failure) | {len(faith_hi_hit0)} | "
        f"Genuinely grounded-but-wrong: no relevant context available |",
        f"| faith≥0.8, hit@5≥1 (had relevant context) | {len(faith_hi_hit1)} | "
        f"Answered faithfully with relevant docs present — proportion recall was the issue |",
        f"| faith<0.8 (unfaithful) | {n - len(faith_hi_rows)} | "
        f"Flagged as unfaithful regardless of recall |",
        "",
        "---",
        "",
        "## 5. Honest accounting of metric choices",
        "",
        "### What changed and why it matters",
        "",
        f"The original 33/100 grounded-but-wrong count used `proportion_recall < 0.5`",
        f"as the \"bad retrieval\" criterion. Of those 33:",
        f"- **{len(gbw_true_fail)} are genuine retrieval failures** (hit@5=0 — no relevant doc in context at all)",
        f"- **{len(gbw_artifact)} are metric artifacts** (hit@5≥1 — model had relevant docs, but",
        f"  the proportion denominator was large enough to push recall below 0.5)",
        "",
        f"With hit@5 as the criterion, grounded-but-wrong drops from **33 → {gbw_new_n}**.",
        "",
        "### Does this invalidate the earlier conclusion?",
        "",
        "Partially. The earlier narrative (\"33 queries are grounded but wrong due to",
        "retrieval failure\") conflated two distinct phenomena:",
        "",
        f"1. **Genuine retrieval failure** (hit@5=0, n={len(gbw_true_fail)}): the model never had a chance",
        "   to answer correctly. This IS a pipeline failure worth fixing.",
        "",
        f"2. **Multi-answer dataset characteristic** (hit@5≥1, n={len(gbw_artifact)}): proportion recall",
        "   is the wrong metric for single-answer QA on a multi-answer retrieval dataset.",
        "   These queries are NOT evidence of retrieval failure — they're evidence of",
        "   **metric mismatch**.",
        "",
        "### Implications for hybrid experiment",
        "",
        "The gap decomposition (see `ceiling_check.md §5`) showed that the +0.20 ceiling",
        f"gap is sorting-improvable. If {len(gbw_artifact)} of the 33 flagged queries are actually",
        "fine (hit@5=1), hybrid reranking that improves proportion_recall from 0.2 to 0.5",
        "would change the label but not the model outcome for those queries.",
        "",
        f"**The meaningful target is the {len(gbw_true_fail)} true failures (hit@5=0).** Hybrid reranking",
        "should be evaluated by its effect on hit@5, not proportion_recall. A system that",
        "moves a query from hit@5=0 to hit@5=1 has genuinely fixed a retrieval failure;",
        "one that moves proportion_recall from 0.10 to 0.55 while maintaining hit@5=1",
        "has improved a metric without fixing a QA problem.",
        "",
        "### What was biased in the previous report",
        "",
        "- The `context_recall_docs` metric is correctly named (it IS a doc-level recall).",
        "  The bias was in interpreting it as a proxy for \"retrieval sufficient for QA\".",
        "- The 33/100 headline was a valid retrieval metric reading; it overstated the",
        "  practical retrieval failure rate for the generation task.",
        "- **No fabrication occurred** — numbers were real. The issue is the choice of",
        "  denominator for a multi-answer dataset being used as a QA benchmark.",
        "",
        "---",
        "",
        "## 6. Recommended metric additions for Step 1+",
        "",
        "| Metric | Definition | Purpose |",
        "|---|---|---|",
        "| `hit@k` | 1 if any relevant doc in top-k | QA-appropriate retrieval signal |",
        "| `proportion_recall@k` (keep) | existing | Retrieval completeness (retrieval task) |",
        "| `grounded_but_wrong_strict` | faith≥0.8 AND hit@k=0 | Genuine confabulation detection |",
        "| `grounded_but_wrong_original` (keep) | faith≥0.8 AND prop_recall<0.5 | Backward-compatible, flag for audit |",
        "",
        "Both metrics should be reported together. Dropping proportion_recall would lose",
        "the retrieval-completeness signal that matters for multi-document summarisation tasks.",
    ]

    # Format the f-strings inside list elements that used format syntax
    report = "\n".join(lines).format(
        **{k: v for k, v in {
            "len(gbw_true_fail)": len(gbw_true_fail),
            "len(gbw_artifact)": len(gbw_artifact),
            "len(faith_hi_hit0)": len(faith_hi_hit0),
            "len(faith_hi_hit1)": len(faith_hi_hit1),
            "gbw_new_n": gbw_new_n,
            "gbw_old_n": gbw_old_n,
            "n": n,
        }.items()}
    ) if False else "\n".join(lines)   # skip — already substituted in f-strings above

    OUT_PATH.write_text(report + "\n", encoding="utf-8")

    # ── console summary ────────────────────────────────────────────────────────
    print(f"Source: {RAG_PATH.relative_to(ROOT)}")
    print(f"\n=== Recall statistics (n={n}) ===")
    print(f"  mean proportion_recall@5 : {mean_p_recall:.4f}")
    print(f"  mean hit@5               : {mean_hit5:.4f}")
    print(f"  queries with hit@5=0     : {len(all_hit0)}")
    print(f"  queries with hit@5=1     : {len(all_hit1)}")
    print(f"\n=== Grounded-but-wrong decomposition ===")
    print(f"  Original (prop_recall<0.5): {gbw_old_n}")
    print(f"    └─ True failure (hit@5=0): {len(gbw_true_fail)}")
    print(f"    └─ Metric artifact (hit@5≥1, prop<0.5): {len(gbw_artifact)}")
    print(f"  Revised  (hit@5=0)        : {gbw_new_n}")
    print(f"  Reduction                 : −{gbw_old_n - gbw_new_n}")
    print(f"\nReport written to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
