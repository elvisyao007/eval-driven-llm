#!/usr/bin/env python3
"""
Gap decomposition — pure data analysis, no model loading.

Decomposes the ceiling gap (oracle_recall@k − current@k) into:
  - sorting-improvable : relevant docs in candidates, outside top-k, but k can hold them
  - k-truncation-locked: n_rel > k, so even a perfect ranker can't exceed k/n_rel

All numbers come from:
  - data/golden/jqara/v0/queries.jsonl  (n_rel per query)
  - reports/20260605T002608-gen/rag_results.json  (per-query current for gen set)
  - frozen aggregate baselines in this script (from existing eval reports)

Deterministic: no randomness, no model. Re-runs produce identical output.

Usage:
  python scripts/gap_decompose.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── frozen baseline numbers from existing eval runs ───────────────────────────
RETRIEVAL_CURRENT_5  = 0.4256   # dense-only, full corpus, k=5,  1667 q (comparison.json)
RETRIEVAL_CURRENT_10 = 0.5738   # dense-only, full corpus, k=10, 1667 q
GEN_CURRENT_5        = 0.4062   # dense+rerank, full corpus, k=5, 100 q (rag_results.json)

# oracle numbers confirmed by ceiling_check.py
ORACLE_RET_5  = 0.6489
ORACLE_RET_10 = 0.8609
ORACLE_GEN_5  = 0.6113
ORACLE_GEN_10 = 0.8337


def _load_n_rel(golden_path: Path) -> dict[str, int]:
    """Return {q_id: n_relevant_docs} from the frozen golden set."""
    n_rel = {}
    with (golden_path / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            n_rel[q["id"]] = len(q["relevant_doc_ids"])
    return n_rel


def _load_gen_current(rag_path: Path) -> dict[str, float]:
    """Return {q_id: context_recall_docs} from the generation eval results."""
    d = json.loads(rag_path.read_text(encoding="utf-8"))
    return {qid: v["context_recall_docs"] for qid, v in d["per_query"].items()}


def _oracle(n: int, k: int) -> float:
    return min(n, k) / n


def _k_trunc(n: int, k: int) -> float:
    return max(0, n - k) / n


def _pct(vals: list, p: float) -> float:
    vals_s = sorted(vals)
    idx = max(0, min(len(vals_s) - 1, int(p / 100 * len(vals_s))))
    return vals_s[idx]


def analyze(
    q_ids: list[str],
    n_rel: dict[str, int],
    current_per_q: dict[str, float] | None,
    current_agg_5: float,
    current_agg_10: float | None,   # None = no k=10 data for this eval set
    label: str,
) -> dict:
    ns = [n_rel[qid] for qid in q_ids if qid in n_rel]
    n = len(ns)

    oracle5_per_q  = [_oracle(v, 5)  for v in ns]
    oracle10_per_q = [_oracle(v, 10) for v in ns]
    ktrunc5_per_q  = [_k_trunc(v, 5)  for v in ns]
    ktrunc10_per_q = [_k_trunc(v, 10) for v in ns]

    oracle5  = statistics.fmean(oracle5_per_q)
    oracle10 = statistics.fmean(oracle10_per_q)
    ktrunc5  = statistics.fmean(ktrunc5_per_q)   # = 1 - oracle5
    ktrunc10 = statistics.fmean(ktrunc10_per_q)  # = 1 - oracle10

    gap5  = oracle5  - current_agg_5
    gap10 = (oracle10 - current_agg_10) if current_agg_10 is not None else None

    # groups by n_rel
    grp_low  = [v for v in ns if v <= 5]   # oracle = 1.0 at k=5 → all gap is sorting
    grp_high = [v for v in ns if v > 5]    # oracle < 1.0 at k=5 → k=5 is binding

    oracle5_high = statistics.fmean([_oracle(v, 5) for v in grp_high]) if grp_high else None

    # per-query decomposition (gen set only, where we have per-query current)
    per_q_sort5 = None
    per_q_sort5_low = None
    per_q_sort5_high = None
    if current_per_q is not None:
        matched = [(n_rel[qid], current_per_q.get(qid, None)) for qid in q_ids
                   if qid in n_rel and qid in current_per_q]
        sort_per_q   = [max(0.0, _oracle(n_v, 5) - cur) for n_v, cur in matched if cur is not None]
        sort_low_q   = [max(0.0, _oracle(n_v, 5) - cur) for n_v, cur in matched
                        if cur is not None and n_v <= 5]
        sort_high_q  = [max(0.0, _oracle(n_v, 5) - cur) for n_v, cur in matched
                        if cur is not None and n_v > 5]
        per_q_sort5      = statistics.fmean(sort_per_q)  if sort_per_q  else 0.0
        per_q_sort5_low  = statistics.fmean(sort_low_q)  if sort_low_q  else 0.0
        per_q_sort5_high = statistics.fmean(sort_high_q) if sort_high_q else 0.0

    return {
        "label": label,
        "n_queries": n,
        # n_rel distribution
        "n_rel_min":  min(ns),
        "n_rel_p25":  _pct(ns, 25),
        "n_rel_p50":  _pct(ns, 50),
        "n_rel_p75":  _pct(ns, 75),
        "n_rel_p90":  _pct(ns, 90),
        "n_rel_max":  max(ns),
        "n_rel_mean": statistics.fmean(ns),
        # group split
        "n_low":       len(grp_low),
        "frac_low":    len(grp_low) / n,
        "n_high":      len(grp_high),
        "frac_high":   len(grp_high) / n,
        "oracle5_high": oracle5_high,
        # oracle / k-trunc
        "oracle5":  oracle5,
        "oracle10": oracle10,
        "ktrunc5":  ktrunc5,
        "ktrunc10": ktrunc10,
        # current (aggregate — used for gap)
        "current5":  current_agg_5,
        "current10": current_agg_10,
        # gaps
        "gap5":  gap5,
        "gap10": gap10,  # None if k=10 current not available
        "has_k10": current_agg_10 is not None,
        # three-way recall budget (sums to 1.0)
        "budget_achieved5":     current_agg_5,
        "budget_sorting_gap5":  gap5,    # = oracle5 − current5
        "budget_ktrunc5":       ktrunc5, # = 1 − oracle5
        # per-query sorting decomposition (gen set only)
        "per_q_sort5":      per_q_sort5,
        "per_q_sort5_low":  per_q_sort5_low,
        "per_q_sort5_high": per_q_sort5_high,
    }


def _section(r: dict, show_per_q: bool) -> list[str]:
    lines = []
    lines += [
        f"### {r['label']}",
        "",
        "**n_rel distribution (relevant docs per query):**",
        "",
        f"| stat | value |",
        f"|---|---|",
        f"| min | {r['n_rel_min']} |",
        f"| p25 | {r['n_rel_p25']} |",
        f"| p50 | {r['n_rel_p50']} |",
        f"| p75 | {r['n_rel_p75']} |",
        f"| p90 | {r['n_rel_p90']} |",
        f"| max | {r['n_rel_max']} |",
        f"| mean | {r['n_rel_mean']:.2f} |",
        "",
        f"Queries with n_rel ≤ 5 (oracle@5 = 1.0, k=5 not binding): "
        f"**{r['n_low']} ({r['frac_low']*100:.1f}%)**  ",
        f"Queries with n_rel > 5 (oracle@5 = 5/n < 1.0, k=5 is binding): "
        f"**{r['n_high']} ({r['frac_high']*100:.1f}%)**",
        f"— mean oracle@5 for the n>5 group: {r['oracle5_high']:.4f}",
        "",
        "**Recall budget breakdown at k=5 (three parts sum to 1.0):**",
        "",
        f"| Component | @k=5 | @k=10 |",
        f"|---|---|---|",
        f"| currently achieved (full-corpus baseline) | "
        f"{r['current5']:.4f} | "
        + (f"{r['current10']:.4f}" if r['has_k10'] else "n/a") + " |",
        f"| **sorting-improvable gap** (oracle − current) | "
        f"**{r['gap5']:.4f}** | "
        + (f"**{r['gap10']:.4f}**" if r['has_k10'] else "n/a") + " |",
        f"| k-truncation-locked (1.0 − oracle) | "
        f"{r['ktrunc5']:.4f} | "
        + (f"{r['ktrunc10']:.4f}" if r['has_k10'] else "n/a") + " |",
        f"| **sum** | "
        f"**{r['current5']+r['gap5']+r['ktrunc5']:.4f}** | "
        + (f"**{r['current10'] + r['gap10'] + r['ktrunc10']:.4f}**" if r['has_k10'] else "n/a") + " |",
        "",
        "The **sorting-improvable gap** is the maximum recall that any reranker",
        "working within the 100 JQaRA candidates could theoretically recover at k=5.",
        "The **k-truncation-locked** portion is structurally unreachable at k=5 regardless",
        "of ranking algorithm — it exists because the mean query has 9.7 relevant docs",
        "and k=5 can only surface 5 of them.",
        "",
        "**Effect of increasing k:**",
    ] + ([
        f"Raising k from 5 → 10 reduces k-truncation-locked from "
        f"{r['ktrunc5']:.4f} to {r['ktrunc10']:.4f} (−{r['ktrunc5']-r['ktrunc10']:.4f})",
        f"and raises oracle from {r['oracle5']:.4f} to {r['oracle10']:.4f} (+{r['oracle10']-r['oracle5']:.4f}).",
        f"The sorting gap also grows: {r['gap5']:.4f} → {r['gap10']:.4f}.",
        "Increasing k yields more potential gain than better reranking at k=5.",
    ] if r['has_k10'] else [
        f"oracle@10 = {r['oracle10']:.4f} vs oracle@5 = {r['oracle5']:.4f} "
        f"(+{r['oracle10']-r['oracle5']:.4f}); k-truncation-locked@10 = {r['ktrunc10']:.4f}.",
        "No k=10 current@5 baseline exists for this eval set (generation eval used k=5 only).",
    ]) + [
    ]

    if show_per_q and r.get("per_q_sort5") is not None:
        lines += [
            "",
            "**Per-query gap decomposition by n_rel group (gen set only; "
            "current from rag_results.json dense+rerank):**",
            "",
            f"| Query group | n queries | mean sorting gap@5 |",
            f"|---|---|---|",
            f"| n_rel ≤ 5 (oracle=1.0) | {r['n_low']} | "
            f"{r['per_q_sort5_low']:.4f} |",
            f"| n_rel > 5 (oracle<1.0) | {r['n_high']} | "
            f"{r['per_q_sort5_high']:.4f} |",
            f"| all queries | {r['n_queries']} | "
            f"{r['per_q_sort5']:.4f} |",
            "",
            "The n_rel≤5 group has a **larger** mean sorting gap (0.39 vs 0.12): missing",
            "even one of two relevant docs costs 0.5 recall to the oracle=1.0 ceiling.",
            "For n_rel>5 queries dense already sits close to its structural oracle (5/n),",
            "so the per-query sorting headroom is smaller.",
        ]

    return lines


def main() -> None:
    golden_path    = ROOT / "data/golden/jqara/v0"
    rag_result_path = ROOT / "reports/20260605T002608-gen/rag_results.json"
    ceiling_path   = ROOT / "reports/ceiling_check.md"

    # ── load data ──────────────────────────────────────────────────────────────
    n_rel = _load_n_rel(golden_path)
    gen_current_per_q = _load_gen_current(rag_result_path)

    all_q_ids = list(n_rel.keys())   # 1667 queries, original file order
    gen_q_ids = all_q_ids[:100]      # generation eval = first 100 queries

    # ── analysis ───────────────────────────────────────────────────────────────
    gen_r = analyze(
        gen_q_ids, n_rel, gen_current_per_q,
        current_agg_5=GEN_CURRENT_5, current_agg_10=None,
        label="Generation eval set — first 100 queries",
    )
    ret_r = analyze(
        all_q_ids, n_rel, None,
        current_agg_5=RETRIEVAL_CURRENT_5, current_agg_10=RETRIEVAL_CURRENT_10,
        label="Retrieval eval set — all 1667 queries",
    )

    # ── print to console ───────────────────────────────────────────────────────
    for r, show in [(gen_r, True), (ret_r, False)]:
        print(f"\n{'='*60}")
        print(f"  {r['label']}")
        print(f"{'='*60}")
        print(f"n_rel: p50={r['n_rel_p50']}, p90={r['n_rel_p90']}, mean={r['n_rel_mean']:.2f}")
        print(f"  n_rel ≤ 5: {r['n_low']} ({r['frac_low']*100:.1f}%)")
        print(f"  n_rel > 5: {r['n_high']} ({r['frac_high']*100:.1f}%), oracle5_high={r['oracle5_high']:.4f}")
        print(f"\nRecall budget @k=5:")
        print(f"  currently achieved   = {r['current5']:.4f}")
        print(f"  sorting-improvable   = {r['gap5']:.4f}  ← hybrid's theoretical max")
        print(f"  k-truncation-locked  = {r['ktrunc5']:.4f}  ← unreachable at k=5")
        print(f"  sum                  = {r['current5']+r['gap5']+r['ktrunc5']:.4f}")
        gap10_str = f"{r['gap10']:.4f}" if r['has_k10'] else "n/a"
        print(f"\n@k=10: oracle={r['oracle10']:.4f}, gap={gap10_str}, ktrunc={r['ktrunc10']:.4f}")
        if show and r.get("per_q_sort5"):
            print(f"\nPer-query sorting gap by n_rel group:")
            print(f"  n_rel ≤ 5 (oracle=1.0): {r['per_q_sort5_low']:.4f}")
            print(f"  n_rel > 5 (oracle<1.0): {r['per_q_sort5_high']:.4f}")

    # ── append section to ceiling_check.md ────────────────────────────────────
    section_header = "\n---\n\n## 5. Gap decomposition — sorting-improvable vs k-truncation-locked\n"
    section_intro = """
> This section decomposes the ceiling gap (oracle_recall@k − current@k) into its
> two structural components. No model is loaded; all numbers come from the golden
> set (n_rel distribution) and the frozen eval reports (current recall).

**Definitions:**
- **sorting-improvable** = oracle_recall@k − current@k. Relevant docs are inside
  the 100 JQaRA candidates but ranked below top-k by the current model. A better
  ranker (hybrid or otherwise) could theoretically recover this.
- **k-truncation-locked** = 1.0 − oracle_recall@k. For queries with n_rel > k,
  even a perfect ranker cannot exceed k/n_rel recall. This portion is structurally
  unreachable at the given k — increasing k is the only lever.

The three components sum to 1.0 (full potential recall per query).

"""
    new_lines = [section_header, section_intro]
    new_lines.extend("\n".join(_section(ret_r, False)))
    new_lines.append("\n")
    new_lines.extend("\n".join(_section(gen_r, True)))
    new_lines.append("\n")
    new_lines.append("\n**One-line summary:**\n")
    new_lines.append(
        f"Hybrid reranking within 100 candidates can theoretically recover at most "
        f"**{ret_r['gap5']:.4f}** recall@5 (the sorting-improvable gap). "
        f"A further **{ret_r['ktrunc5']:.4f}** is k-truncation-locked and "
        f"unreachable at k=5 regardless of algorithm. "
        f"Raising k to 10 reduces the locked portion to {ret_r['ktrunc10']:.4f}.\n"
    )

    existing = ceiling_path.read_text(encoding="utf-8")
    # Remove any existing §5 section before appending (idempotent re-run)
    marker = "\n---\n\n## 5."
    if marker in existing:
        existing = existing[:existing.index(marker)]

    ceiling_path.write_text(existing + "".join(new_lines), encoding="utf-8")
    print(f"\nappended §5 gap decomposition to {ceiling_path}")


if __name__ == "__main__":
    main()
