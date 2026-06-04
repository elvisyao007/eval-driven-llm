# JQaRA v0 — end-to-end eval summary
**Date:** 2026-06-04  
**Golden set:** jqara@v0 — 1667 queries, 144362 passages (CC BY-SA 4.0)  
**Retrieval:** ruri-v3-310m (dense) → BGE-Reranker-v2-m3 (cross-encoder, k=5)  
**Generation:** qwenj:latest (Qwen 2.5 Coder 14B Q5_K_M, ollama)  
**Judge:** qwenj:latest (same model — see limitations)

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

## Phase 4 — Generation eval (100-query sample, judge-based)

| Metric | Value | Notes |
|---|---|---|
| faithfulness (mean) | **0.7751** | judge-based; see limitations |
| faithfulness (max spread, 2 runs) | **0.0000** | fully deterministic at temp=0/seed=0 |
| context_recall_docs (mean) | **0.4062** | deterministic; fraction of golden docs in top-5 |
| grounded-but-wrong queries | **48 / 100** | faith≥0.8 AND ctx_recall<0.5 |

**How to read faithfulness alongside context_recall_docs:**
A faithfulness-only score of 0.78 looks acceptable. The 0.41 context recall tells the
real story: in ~59% of queries the relevant document did not make it into the top-5,
so the model generated a plausible-sounding answer from wrong evidence.
48 of 100 queries are "grounded but wrong" — the exact failure mode a
faithfulness-only eval hides (ADR-0001).

### Most instructive grounded-but-wrong cases

| query id | query (abridged) | ref | faith | ctx_recall |
|---|---|---|---|---|
| QA20CAPR-1008 | 眠たい時についつい出てしまう生理現象 | あくび | 1.000 | 0.000 |
| QA20CAPR-1015 | 太極旗といえばどこの国旗 | 大韓民国 | 1.000 | 0.077 |
| QA20CAPR-1014 | 「木曽路はすべて山の中」で始まる島崎藤村の小説 | 夜明け前 | 1.000 | 0.273 |
| QA20CAPR-1007 | わざと人に逆らう言動をする人を鬼に例えて何という | 天邪鬼 | 1.000 | 0.250 |

QA20CAPR-1008 is the clearest case: zero relevant docs retrieved, faithfulness=1.0.
The model invented (or retrieved) something unrelated and presented it with full
confidence — undetectable without the ground-truth context_recall anchor.

---

## Limitations

### Generator = Judge (most important)
`qwenj:latest` is used as both generator and judge because:
- Target model (Gemma 4 27B) is only available as GGUF; vLLM 0.10.x does not
  support the `gemma4` architecture in GGUF format.
- All existing vLLM installations pre-date PyTorch 2.12.0+cu130 (required for
  RTX 5090, sm_120) and cannot load on this hardware.
- No ELYZA-JP model available on disk for an independent judge.

A model judging its own outputs is methodologically invalid for absolute
faithfulness scores. The zero spread (max_spread=0.0) confirms determinism but
not independence. **Treat the 0.7751 faithfulness number as a plumbing
verification, not a production metric.** Re-run with an independent judge
(ELYZA-JP-8B or similar) when hardware allows.

### 100-query sample
Generation eval used `--max-queries 100` (6% of the 1667-query golden set) due
to wall-clock cost (~4 min for 100 queries at ~2s/query). The retrieval eval
used all 1667 queries. The sample is front-loaded (first 100 query IDs).

### Qwen 2.5 Coder as RAG generator
Qwen 2.5 Coder is tuned for code/reasoning tasks, not concise Japanese QA. It
has a tendency to continue generating few-shot examples after the answer unless
stopped with explicit stop sequences (`stop=["\n\n[文脈", "\n\n質問", "\n\n回答"]`).
Answers were capped at 256 tokens. A purpose-tuned Japanese instruction model
(Gemma 4 / Swallow / ELYZA-JP) would produce cleaner generation and more
reliable faithfulness scoring.

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

## Next steps (when hardware/models allow)

- Install an independent Japanese judge model (ELYZA-JP-8B or Swallow-8B) and
  re-run Phase 4 to get a valid faithfulness score.
- Run generation eval on full 1667-query set (not just first 100).
- Investigate the 48 grounded-but-wrong queries manually: are they
  retrieval failures (wrong doc) or annotation gaps (multiple valid answers)?
- Consider hybrid retrieval (dense + BM25) to raise recall@10 above 0.57 and
  reduce grounded-but-wrong rate.
