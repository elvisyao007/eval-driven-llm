# Recall metric analysis: proportion recall vs hit@k on JQaRA

> Source: `reports/20260605T024628-gen/rag_results.json`
> (Phase 4, gemma4:31b judge, qwen3:32b generator, 100-query sample)

---

## 1. Why proportion recall misrepresents multi-answer QA

JQaRA pairs each query with 6–28 relevant Wikipedia passages (mean 9.7).
The baseline metric `context_recall_docs = |retrieved ∩ relevant| / |relevant|`
was designed for the **retrieval** task (how many relevant docs did we surface?).
For **generation**, the question is different: did the model receive *at least one*
relevant document to ground its answer? Retrieving 2 of 23 relevant passages is
scored as 0.087 recall by the proportion metric, but those 2 passages may be
perfectly sufficient to answer the question.

This analysis computes `hit@5` (binary: ≥1 relevant doc in top-5) and tests
whether the 33/100 grounded-but-wrong count is inflated by the denominator effect.

| Metric | Definition | Sensitive to multi-answer denominator? |
|---|---|---|
| proportion_recall (existing) | \|retrieved ∩ relevant\| / \|relevant\| | Yes — penalises partial hits |
| hit@5 (new) | 1 if proportion_recall > 0 else 0 | No — only asks "did we hit any?" |

---

## 2. Overall recall statistics (100 queries)

| Metric | Value |
|---|---|
| mean faithfulness | 0.6662 |
| mean proportion_recall@5 | 0.4062 |
| mean hit@5 | **0.9800** |
| queries with hit@5=0 (zero relevant docs in top-5) | 2 / 100 |
| queries with hit@5=1 (≥1 relevant doc in top-5) | 98 / 100 |

**98 of 100 queries (98.0%) had at least one relevant document**
in the retrieved top-5. Only 2 queries had zero relevant docs.

---

## 3. Decomposing the 33 grounded-but-wrong queries

Original definition: `faithfulness ≥ 0.8 AND proportion_recall < 0.5`

**Total grounded-but-wrong (original): 33 / 100**

| Class | Definition | Count |
|---|---|---|
| **True failure** | faith≥0.8, hit@5=0 (zero relevant docs retrieved) | **0** |
| **Metric artifact** | faith≥0.8, hit@5≥1 but proportion_recall<0.5 | **33** |
| Total | — | 33 |

**True failures** (0): the model had *no* relevant document in
its context. High faithfulness here means it confabulated a plausible-sounding
answer from irrelevant passages. This is the genuinely dangerous failure.

**Metric artifacts** (33): the model *did* retrieve at least one
relevant document (hit@5=1), but the proportion recall was < 0.5 because the
denominator (total relevant docs) is large. The model likely had sufficient
grounding to answer correctly — these are not retrieval failures in the QA sense.

n_rel for metric artifacts: min=5, median=14, max=27, mean=16.0
proportion_recall for metric artifacts: min=0.0769, max=0.4545, mean=0.2665
**Structurally impossible to fix at k=5** (n_rel > 10, oracle proportion_recall@5 = 5/n_rel < 0.5): **28 of 33**. These queries would be labelled grounded-but-wrong by the original definition *even with a perfect oracle retriever* at k=5.

---

## 4. Grounded-but-wrong with hit@5 as the recall criterion

New definition: `faithfulness ≥ 0.8 AND hit@5 = 0`
(faithful answer AND zero relevant documents retrieved)

| Definition | grounded-but-wrong count |
|---|---|
| Original (proportion_recall<0.5) | 33 / 100 |
| **Revised (hit@5=0)** | **0 / 100** |
| Difference | −33 (33 queries reclassified as non-failure) |

**faith≥0.8 group breakdown:**

| Subgroup | Count | Interpretation |
|---|---|---|
| faith≥0.8, hit@5=0 (true failure) | 0 | Genuinely grounded-but-wrong: no relevant context available |
| faith≥0.8, hit@5≥1 (had relevant context) | 54 | Answered faithfully with relevant docs present — proportion recall was the issue |
| faith<0.8 (unfaithful) | 46 | Flagged as unfaithful regardless of recall |

---

## 5. Honest accounting of metric choices

### What changed and why it matters

The original 33/100 grounded-but-wrong count used `proportion_recall < 0.5`
as the "bad retrieval" criterion. Of those 33:
- **0 are genuine retrieval failures** (hit@5=0 — no relevant doc in context at all)
- **33 are metric artifacts** (hit@5≥1 — model had relevant docs, but
  the proportion denominator was large enough to push recall below 0.5)

With hit@5 as the criterion, grounded-but-wrong drops from **33 → 0**.

### Does this invalidate the earlier conclusion?

Partially. The earlier narrative ("33 queries are grounded but wrong due to
retrieval failure") conflated two distinct phenomena:

1. **Genuine retrieval failure** (hit@5=0, n=0): the model never had a chance
   to answer correctly. This IS a pipeline failure worth fixing.

2. **Multi-answer dataset characteristic** (hit@5≥1, n=33): proportion recall
   is the wrong metric for single-answer QA on a multi-answer retrieval dataset.
   These queries are NOT evidence of retrieval failure — they're evidence of
   **metric mismatch**.

### Implications for hybrid experiment

The gap decomposition (see `ceiling_check.md §5`) showed that the +0.20 ceiling
gap is sorting-improvable. If 33 of the 33 flagged queries are actually
fine (hit@5=1), hybrid reranking that improves proportion_recall from 0.2 to 0.5
would change the label but not the model outcome for those queries.

**The meaningful target is the 0 true failures (hit@5=0).** Hybrid reranking
should be evaluated by its effect on hit@5, not proportion_recall. A system that
moves a query from hit@5=0 to hit@5=1 has genuinely fixed a retrieval failure;
one that moves proportion_recall from 0.10 to 0.55 while maintaining hit@5=1
has improved a metric without fixing a QA problem.

### What was biased in the previous report

- The `context_recall_docs` metric is correctly named (it IS a doc-level recall).
  The bias was in interpreting it as a proxy for "retrieval sufficient for QA".
- The 33/100 headline was a valid retrieval metric reading; it overstated the
  practical retrieval failure rate for the generation task.
- **No fabrication occurred** — numbers were real. The issue is the choice of
  denominator for a multi-answer dataset being used as a QA benchmark.

---

## 6. Recommended metric additions for Step 1+

| Metric | Definition | Purpose |
|---|---|---|
| `hit@k` | 1 if any relevant doc in top-k | QA-appropriate retrieval signal |
| `proportion_recall@k` (keep) | existing | Retrieval completeness (retrieval task) |
| `grounded_but_wrong_strict` | faith≥0.8 AND hit@k=0 | Genuine confabulation detection |
| `grounded_but_wrong_original` (keep) | faith≥0.8 AND prop_recall<0.5 | Backward-compatible, flag for audit |

Both metrics should be reported together. Dropping proportion_recall would lose
the retrieval-completeness signal that matters for multi-document summarisation tasks.
