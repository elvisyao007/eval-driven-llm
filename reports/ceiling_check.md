# Ceiling check — Step 0 gate (EXPERIMENT_hybrid.md §1)

> **Scope**: all numbers below operate within JQaRA's fixed 100 candidates per
> query. `oracle_recall@k` is the maximum `context_recall@k` achievable by any
> perfect reranker working within those 100 candidates. It does **not** represent a
> first-stage retrieval improvement — JQaRA is a reranking benchmark.
>
> **These numbers are the upper bound for the hybrid reranking experiment,
> not for a first-stage dense retrieval improvement.**

**Embedder for rank distribution**: `ruri` (same model as the dense baseline — exact dense ranks)

---

## 1. Generation eval set — first 100 queries

| Metric | Value |
|---|---|
| n_queries | 100 |
| binary_ceiling (all relevant docs in 100 candidates) | 1.0000 (100/100) |
| **oracle_recall@5** (ceiling within 100 candidates) | **0.6113** |
| oracle_recall@10 (ceiling within 100 candidates) | 0.8337 |
| dense_recall@5 within 100 candidates (ruri-v3 ordering) | 0.4224 |
| dense_recall@10 within 100 candidates | 0.5797 |
| **current context_recall@5** (dense+rerank, full corpus) | **0.4062** |
| source | reports/20260605T002608-gen/rag_results.json (dense+rerank, k=5) |
| **gap@5 = oracle − current** | **+0.2051** |

### Rank distribution within 100 candidates — gen set

- first-relevant rank within 100 candidates: p50=1, p90=2, max=55, mean=2.3
- fraction with first-relevant in top-1: 0.8400
- fraction with first-relevant in top-5: 0.9700
- fraction with first-relevant in top-10: 0.9800
- **dense recall@5 within 100 candidates**: 0.4224
- **dense recall@10 within 100 candidates**: 0.5797


---

## 2. Retrieval eval set — all 1667 queries

| Metric | Value |
|---|---|
| n_queries | 1667 |
| binary_ceiling (all relevant docs in 100 candidates) | 1.0000 (1667/1667) |
| **oracle_recall@5** (ceiling within 100 candidates) | **0.6489** |
| oracle_recall@10 (ceiling within 100 candidates) | 0.8609 |
| dense_recall@5 within 100 candidates (ruri-v3 ordering) | 0.4368 |
| dense_recall@10 within 100 candidates | 0.5885 |
| **current recall@5** (dense-only, full corpus) | **0.4256** |
| current recall@10 (dense-only, full corpus) | 0.5738 |
| source | reports/20260604T042010/comparison.json (dense-only, ruri-v3, k=5/10) |
| **gap@5 = oracle − current** | **+0.2233** |
| gap@10 = oracle − current@10 | +0.2871 |

### Rank distribution within 100 candidates — retrieval set

- first-relevant rank within 100 candidates: p50=1, p90=2, max=88, mean=1.9
- fraction with first-relevant in top-1: 0.8356
- fraction with first-relevant in top-5: 0.9616
- fraction with first-relevant in top-10: 0.9754
- **dense recall@5 within 100 candidates**: 0.4368
- **dense recall@10 within 100 candidates**: 0.5885


---

## 3. Decision

| Eval set | oracle@5 | dense@5 (within 100) | current@5 (full corpus) | gap@5 | bucket |
|---|---|---|---|---|---|
| gen (100 q) | 0.6113 | 0.4224 | 0.4062 | +0.2051 | ≥0.15 |
| retrieval (1667 q) | 0.6489 | 0.4368 | 0.4256 | +0.2233 | ≥0.15 |

**Decision (rule from EXPERIMENT_hybrid.md §1):** 全量継続 (full experiment continues — A0 through H4/R0)

Decision thresholds:
- gap ≥ 0.15 → 全量継続 (full experiment A0–H4/R0)
- 0.05 ≤ gap < 0.15 → MVL only (A0/A1/A2/H1/H2)
- gap < 0.05 → stop hybrid; pivot to ceiling narrative + grounded-but-wrong

---

## 4. Interpretation

### What the numbers mean

- **binary_ceiling = 1.0**: by JQaRA dataset construction, every query's relevant
  docs are included in its fixed 100 candidates. This is a validity check, not a
  finding — it should always be 1.0 on JQaRA data.

- **oracle_recall@5 ≈ 0.61–0.65**: a perfect reranker working only within the 100
  JQaRA candidates can recover at most 61–65% of relevant docs at k=5. The ceiling
  is limited by the number of relevant docs per query (mean ≈ 9.7), not by candidate
  coverage. For queries with >5 relevant docs, even an oracle can only return 5/n.

- **dense_recall@5 within 100 ≈ 0.42–0.44**: ruri-v3 dense ordering of the 100
  candidates already achieves nearly the same recall@5 as the full-corpus baseline.
  The rank distribution (p50=1, p90=2) confirms that the first relevant doc ranks
  very highly in the dense ordering. The gap from oracle (0.19–0.21) is driven by
  queries with many relevant docs, not by poor ordering of individual relevant docs.

- **Gap@5 ≈ +0.20**: this is the maximum lift that ANY reranker (hybrid or otherwise)
  could achieve within the 100 JQaRA candidates at k=5. It is ≥ 0.15, so the
  experiment should continue under the gate rule.

### Critical nuance for interpreting hybrid results

The p50/p90 rank distribution shows the dense model already ranks the first
relevant doc at position 1 or 2 for most queries. The remaining gap to the
oracle (≈ 0.19) is **structural**: it comes from queries that have 6–28 relevant
docs, and k=5 can only surface 5 of them regardless of ranking quality.

Hybrid reranking (BM25 + dense) may recover some of this structural gap if BM25
surface relevant docs that dense alone missed (complementary signals). But the
denominator is fixed by JQaRA's label density, not by retrieval algorithm design.
Track delta against dense_recall@5_within_100 (≈ 0.42–0.44), not just against
the full-corpus baseline (0.41), to correctly attribute any improvement.

---

## 5. Gap decomposition — sorting-improvable vs k-truncation-locked

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

### Retrieval eval set — all 1667 queries

**n_rel distribution (relevant docs per query):**

| stat | value |
|---|---|
| min | 1 |
| p25 | 4 |
| p50 | 8 |
| p75 | 14 |
| p90 | 21 |
| max | 28 |
| mean | 9.72 |

Queries with n_rel ≤ 5 (oracle@5 = 1.0, k=5 not binding): **573 (34.4%)**  
Queries with n_rel > 5 (oracle@5 = 5/n < 1.0, k=5 is binding): **1094 (65.6%)**
— mean oracle@5 for the n>5 group: 0.4651

**Recall budget breakdown at k=5 (three parts sum to 1.0):**

| Component | @k=5 | @k=10 |
|---|---|---|
| currently achieved (full-corpus baseline) | 0.4256 | 0.5738 |
| **sorting-improvable gap** (oracle − current) | **0.2233** | **0.2871** |
| k-truncation-locked (1.0 − oracle) | 0.3511 | 0.1391 |
| **sum** | **1.0000** | **1.0000** |

The **sorting-improvable gap** is the maximum recall that any reranker
working within the 100 JQaRA candidates could theoretically recover at k=5.
The **k-truncation-locked** portion is structurally unreachable at k=5 regardless
of ranking algorithm — it exists because the mean query has 9.7 relevant docs
and k=5 can only surface 5 of them.

**Effect of increasing k:**
Raising k from 5 → 10 reduces k-truncation-locked from 0.3511 to 0.1391 (−0.2120)
and raises oracle from 0.6489 to 0.8609 (+0.2120).
The sorting gap also grows: 0.2233 → 0.2871.
Increasing k yields more potential gain than better reranking at k=5.
### Generation eval set — first 100 queries

**n_rel distribution (relevant docs per query):**

| stat | value |
|---|---|
| min | 1 |
| p25 | 5 |
| p50 | 9 |
| p75 | 15 |
| p90 | 23 |
| max | 27 |
| mean | 10.73 |

Queries with n_rel ≤ 5 (oracle@5 = 1.0, k=5 not binding): **31 (31.0%)**  
Queries with n_rel > 5 (oracle@5 = 5/n < 1.0, k=5 is binding): **69 (69.0%)**
— mean oracle@5 for the n>5 group: 0.4367

**Recall budget breakdown at k=5 (three parts sum to 1.0):**

| Component | @k=5 | @k=10 |
|---|---|---|
| currently achieved (full-corpus baseline) | 0.4062 | n/a |
| **sorting-improvable gap** (oracle − current) | **0.2051** | n/a |
| k-truncation-locked (1.0 − oracle) | 0.3887 | n/a |
| **sum** | **1.0000** | n/a |

The **sorting-improvable gap** is the maximum recall that any reranker
working within the 100 JQaRA candidates could theoretically recover at k=5.
The **k-truncation-locked** portion is structurally unreachable at k=5 regardless
of ranking algorithm — it exists because the mean query has 9.7 relevant docs
and k=5 can only surface 5 of them.

**Effect of increasing k:**
oracle@10 = 0.8337 vs oracle@5 = 0.6113 (+0.2224); k-truncation-locked@10 = 0.1663.
No k=10 current@5 baseline exists for this eval set (generation eval used k=5 only).

**Per-query gap decomposition by n_rel group (gen set only; current from rag_results.json dense+rerank):**

| Query group | n queries | mean sorting gap@5 |
|---|---|---|
| n_rel ≤ 5 (oracle=1.0) | 31 | 0.3866 |
| n_rel > 5 (oracle<1.0) | 69 | 0.1236 |
| all queries | 100 | 0.2051 |

The n_rel≤5 group has a **larger** mean sorting gap (0.39 vs 0.12): missing
even one of two relevant docs costs 0.5 recall to the oracle=1.0 ceiling.
For n_rel>5 queries dense already sits close to its structural oracle (5/n),
so the per-query sorting headroom is smaller.

**One-line summary:**
Hybrid reranking within 100 candidates can theoretically recover at most **0.2233** recall@5 (the sorting-improvable gap). A further **0.3511** is k-truncation-locked and unreachable at k=5 regardless of algorithm. Raising k to 10 reduces the locked portion to 0.1391.
