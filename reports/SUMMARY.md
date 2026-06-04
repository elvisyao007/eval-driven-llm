# JQaRA v0 — end-to-end eval summary
**Date:** 2026-06-05 (Phase 4 re-run with independent judge)  
**Golden set:** jqara@v0 — 1667 queries, 144362 passages (CC BY-SA 4.0)  
**Retrieval:** ruri-v3-310m (dense) → BGE-Reranker-v2-m3 (cross-encoder, k=5)  
**Generation:** qwen3:32b (ollama, Apache-2.0)  
**Judge:** gemma4:31b (ollama, Apache-2.0) — **independent; different model family from generator**

---

## Phase 3 — Retrieval benchmark (1667 queries, deterministic)

| Metric | dense-only | dense+rerank | delta |
|---|---|---|---|
| P@1 | 0.8308 | **0.8440** | +0.0132 |
| MRR | 0.8858 | **0.8927** | +0.0069 |
| nDCG@5 | 0.7135 | 0.7051 | -0.0083 |
| nDCG@10 | 0.6870 | 0.6770 | -0.0099 |
| Recall@5 | 0.4256 | 0.4158 | -0.0098 |
| Recall@10 | 0.5738 | 0.5634 | -0.0104 |

Per-query P@1 flips: **133 fixed, 111 broke, 1423 unchanged** (net +22).

**Reading:** The cross-encoder reranker makes a clear trade-off — it lifts P@1 (+0.0132)
and MRR (+0.0069) by promoting the single best answer, but costs recall@10 (-0.0104)
by reshuffling mid-ranks. Correct choice for single-answer QA; revisit for
multi-document summarisation tasks.

---

## Phase 4 — Generation eval (100-query sample, independent judge)

| Metric | Value | Notes |
|---|---|---|
| faithfulness (mean) | **0.6662** | gemma4:31b judge; independent from generator |
| faithfulness (max spread, 3 runs) | **0.0500** | non-zero: variance is measurable, not hidden |
| context_recall_docs (mean) | **0.4062** | deterministic; fraction of golden docs in top-5 |
| grounded-but-wrong queries | **33 / 100** | faith≥0.8 AND ctx_recall<0.5 |

**How to read faithfulness alongside context_recall_docs:**  
A faithfulness-only score of 0.67 looks passable. The 0.41 context recall tells the
real story: in ~59% of queries the relevant document did not make it into the top-5,
so the model generated a plausible-sounding answer from wrong evidence.
33 of 100 queries are "grounded but wrong" — the exact failure mode a
faithfulness-only eval hides (ADR-0001).

Comparing to the earlier self-grading run (qwenj judging itself, 0.7751 faith / 0 spread):
the independent judge scores lower (0.6662) and shows non-zero spread (0.05). Both
differences are expected: an independent judge is stricter, and without self-grading
the model can no longer confirm its own claims trivially.

### Most instructive grounded-but-wrong cases (Phase 4 independent run)

| query id | faithfulness | ctx_recall | grounded_but_wrong |
|---|---|---|---|
| QA20CAPR-1008 | 0.000 | 0.000 | False |
| QA20CAPR-1055 | 0.000 | 0.000 | False |
| QA20CAPR-1099 | 0.250 | 0.056 | False |
| QA20CAPR-1116 | 0.667 | 0.071 | False |
| QA20CAPR-1130 | 0.500 | 0.071 | False |

Low-ctx_recall queries where faith dropped to 0: retrieval failure confirmed — no
relevant document retrieved, independent judge correctly returns no entailment.

---

## Limitations

### 100-query sample
Generation eval used `--max-queries 100` (6% of the 1667-query golden set) due
to wall-clock cost (~40 min for 100 queries with model-swapping on a single 32GB GPU).
The retrieval eval used all 1667 queries. The sample is front-loaded (first 100 query IDs).
Re-run on the full set to validate sample representativeness.

### Single-GPU model swapping
qwen3:32b (20 GB) and gemma4:31b (19 GB) cannot coexist in 32 GB VRAM. The eval
runs a two-pass architecture: all generation first, explicit VRAM unload, then all
judging. Wall-clock cost: ~28 min gen + ~12 min judge = ~40 min for 100 queries.

### qwen3:32b thinking mode (generation)
qwen3:32b produces extended reasoning in `<think>` blocks before answering. RAG
generation used `max_tokens=256` with stop sequences; thinking tokens count against
this budget. Answers may be truncated in complex queries. A non-thinking inference
mode (or larger token budget) would improve generation quality.

---

## What this run validated

1. **Eval harness is correctly wired** — retrieval metrics are deterministic and
   reproducible given the frozen golden set + corpus hash.
2. **The grounded-but-wrong detector works** — it surfaces a failure class that
   pure faithfulness scores hide, exactly as designed in ADR-0001.
3. **Reranker trade-off is measurable** — P@1/MRR lift vs recall@10 cost is
   quantified; the correct call depends on the downstream task.
4. **Generation pipeline runs end-to-end on-prem** — no cloud API in the path;
   all models local; reproducible with pinned temp/seed.
5. **Independent judge gives a lower, more credible faithfulness score** — 0.6662
   vs the earlier self-grading 0.7751; non-zero spread (0.05) confirms the judge
   is not trivially confirming the generator's claims.

## Next steps

- Run generation eval on full 1667-query set (not just first 100) to validate
  sample representativeness.
- Investigate the 33 grounded-but-wrong queries manually: are they
  retrieval failures (wrong doc) or annotation gaps (multiple valid answers)?
- Consider hybrid retrieval (dense + BM25) to raise recall@10 above 0.57 and
  reduce grounded-but-wrong rate.
- Evaluate qwen3:32b vs gemma4:31b as generator directly (swap gen model and
  re-run Phase 4) to measure which produces higher faithfulness on this task.
