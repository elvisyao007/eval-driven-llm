# RAG eval — generation (faithfulness, judge-based)

## Run metadata

- date: 2026-06-04T18:23:33
- golden: jqara@v0
- embedder: ruri
- reranker: cross-encoder
- generator: openai
- judge: local
- gen_model: qwenj:latest
- judge_model: qwenj:latest
- runs_per_item: 2
- note: 

## Results

| Metric | Value |
|---|---|
| faithfulness (mean) | 0.7751 |
| faithfulness (max judge spread) | 0.0000 |
| context_recall_docs (mean, deterministic) | 0.4062 |
| grounded-but-wrong queries | 48 / 100 |

> Read faithfulness AND context_recall together. High faithfulness with low context recall = confidently grounded in the wrong documents — the failure a faithfulness-only score hides (ADR-0001).

## Lowest-context-recall queries (inspect by hand for sec 6)

| query id | faithfulness | ctx_recall | grounded_but_wrong |
|---|---|---|---|
| QA20CAPR-1008 | 1.000 | 0.000 | True |
| QA20CAPR-1055 | 0.500 | 0.000 | False |
| QA20CAPR-1099 | 0.000 | 0.056 | False |
| QA20CAPR-1130 | 1.000 | 0.071 | True |
| QA20CAPR-1116 | 0.800 | 0.071 | True |
