# eval-driven-llm — eval-driven reliable LLM systems

> An on-prem LLM system built **eval-first**: the evaluation harness is the
> backbone, the RAG pipeline is the current (replaceable) payload. The goal is
> not a demo that runs — it is a system that survives enterprise reality:
> dirty data, multiple users, data that cannot leave the building, and an
> objective, repeatable definition of "good enough to ship."

**Status:** Layer 1 (retrieval + eval harness) — in progress, build-in-public.

---

## What this is / who it's for (30-second read)

- **Problem.** Most RAG projects die between PoC and production. The usual
  reasons are not "the model is dumb": they are dirty data, no objective
  acceptance bar, security/data-residency constraints, and nobody who can
  maintain the thing afterwards. "It answered my question in the notebook" is
  not a production signal.
- **Approach.** Put the **evaluation methodology at the center** and treat
  everything else (which retriever, which vector store, which model, later
  which agent) as a swappable payload measured against a frozen acceptance bar.
  Run fully **on-prem** so the customer's data never leaves their hardware.
- **What "good" means here.** Not vibes. A frozen golden set, deterministic
  retrieval metrics, judge-based generation metrics with a **pinned** judge,
  and a versioned report you can re-run and diff. See
  [`docs/eval-report-template.md`](docs/eval-report-template.md).

## Why it is built the way it is

The design choices — why eval-first, why this retriever stack, why these
models, why this corpus split — are not in the code comments. They are
recorded as dated, reversible decisions in **[`DECISIONS.md`](DECISIONS.md)**,
which is the part of this repo a generator cannot fake.

## Architecture

![eval-driven-llm architecture](docs/architecture.svg)

<details><summary>text version (concentric rings)</summary>

```
            ┌─────────────────────────────────────────────┐
            │  payload (swappable, do not over-invest)     │
            │   RAG today → agent later. Frameworks/models │
            │   are config, not commitments.               │
            │  ┌───────────────────────────────────────┐   │
            │  │  carrier ring                          │   │
            │  │   on-prem deploy + PoC→prod hardening  │   │
            │  │   (data never leaves the customer)     │   │
            │  │  ┌─────────────────────────────────┐   │   │
            │  │  │  CORE                            │   │   │
            │  │  │   eval / acceptance methodology  │   │   │
            │  │  │   framework- & model-agnostic    │   │   │
            │  │  └─────────────────────────────────┘   │   │
            │  └───────────────────────────────────────┘   │
            └─────────────────────────────────────────────┘
```

</details>

Full diagram and data flow: [`docs/architecture.md`](docs/architecture.md).

## Default stack (current payload)

| Layer | Choice | Why (short) |
|---|---|---|
| Generation | qwen3:32b (default) / gemma4:31b (comparison) via Ollama | Apache-2.0, multilingual, fit a single 32 GB GPU; both now on-prem |
| Judge | gemma4:31b — **separate from generator** | Qwen judges Gemma output and vice versa; different families prevent self-grading |
| Retrieval (1st pass) | ruri-v3-310m (dense) — BGE-M3 (dense+sparse) as hybrid option | Japanese-OSS retrieval SOTA; self-hosted, zero data egress |
| Rerank (2nd pass) | BGE-Reranker-v2-m3 | two-stage retrieve → rerank for precision@1 |
| Serving / gateway | Ollama (OpenAI-compatible at `:11434`) | both models resident; swappable behind one interface |
| Vector store | Faiss (benchmark track) / Qdrant (realism track) | Qdrant carries metadata filtering for per-user permissions |
| Eval | in-house deterministic retrieval metrics + judge-based generation metrics (pinned independent judge) + frozen golden set | see [`DECISIONS.md`](DECISIONS.md) ADR-0001 |

Everything above is a config entry, not a hard dependency. The only thing that
is *not* meant to be swapped cheaply is the eval methodology.

## Eval results

Full run details: [`reports/SUMMARY.md`](reports/SUMMARY.md).

**JQaRA v0 — retrieval (1667 queries, deterministic)**

| Metric | dense-only | dense+rerank | delta |
|---|---|---|---|
| P@1 | 0.8308 | **0.8440** | +0.0132 |
| Recall@5 | 0.4256 | 0.4158 | -0.0098 |
| nDCG@10 | 0.6870 | 0.6770 | -0.0099 |

Reranker lifts P@1 at the cost of recall — correct trade-off for single-answer QA.

**JQaRA v0 — generation (100-query sample, independent judge)**

| Metric | Value |
|---|---|
| faithfulness (mean, gemma4:31b judge) | **0.6662** |
| faithfulness max spread (3 runs) | 0.0500 |
| proportion_recall@5 (mean, deterministic) | **0.4062** |
| oracle proportion_recall@5 ceiling (within 100 candidates) | **0.6113** |
| hit@5 (≥1 relevant doc in top-5) | **0.98** (98 / 100 queries) |
| grounded-but-wrong queries (original flag) | 33 / 100 — metric artifact (see ADR-0009) |

Generator: `qwen3:32b`. Judge: `gemma4:31b` (different model family — not self-grading).

**ADR-0009 correction.** `proportion_recall@5 = 0.4062` correctly measures
retrieval completeness for this reranking benchmark but is **not** a QA-failure
signal. JQaRA queries have 6–28 relevant documents (mean 9.7); the structural
ceiling at k=5 is **0.6113** — unreachable even with a perfect reranker, because
k=5 can surface at most 5 of ~10 relevant documents.

`hit@5 = 0.98`: 98 of 100 queries had at least one relevant document in the
top-5 context window. The 2 queries with hit@5=0 both had faithfulness=0.0 —
the judge correctly gave them no credit.

The **33 "grounded-but-wrong" queries** (faithfulness ≥ 0.8 AND
proportion_recall < 0.5) are **100% metric artifacts**: all 33 had hit@5=1
(relevant context was present). 28 of 33 have n_rel > 10, making proportion_recall
< 0.5 structurally inevitable at k=5 even for a perfect retriever. At the hit@5
threshold, grounded-but-wrong = **0 / 100**. The number is real; the original
interpretation ("retrieval fed wrong documents") was wrong.

Full diagnostic: [`reports/recall_metric_analysis.md`](reports/recall_metric_analysis.md) ·
[ADR-0008/0009](DECISIONS.md) · blog-03 (forthcoming on dev.to) ·
[eval-sanity](https://github.com/elvisyao007/eval-sanity)

## How to run

Two paths. The baseline path needs no GPU and no model downloads — it exists so
the eval loop is verifiable anywhere. The semantic path is the real benchmark.

```bash
# --- dirty-data ingestion (Track B): real docs -> passages.jsonl + audit ---
make ingest-dirty       # parse/normalize/dedup/chunk a messy demo corpus
                        # (tolerant batch, JP encoding fallback, audit log)

# --- baseline / plumbing validation (CPU, deterministic, no downloads) ---
make install-base
make build-toy          # freeze a tiny synthetic golden set
make eval-toy           # runs the full loop with a lexical-baseline embedder
                        # NOTE: validates the pipeline; NOT a model benchmark

# --- Track A: real semantic benchmark (needs a CUDA GPU + HF access) ---
make install            # installs ruri-v3 / faiss / datasets / reranker
make build-jqara        # freeze a JQaRA split -> golden set + corpus
make eval GOLDEN=data/golden/jqara/v0   # ruri-v3 dense retrieval
make compare GOLDEN=data/golden/jqara/v0  # dense-only vs dense+cross-encoder

# end-to-end generation eval (faithfulness), independent LOCAL judge:
# gen=qwen3:32b, judge=gemma4:31b — different model families, no self-grading
make rag-eval GOLDEN=data/golden/jqara/v0 \
     BASE_URL=http://localhost:11434/v1 GEN_MODEL=qwen3:32b \
     JUDGE_MODEL=gemma4:31b JUDGE_BASE_URL=http://localhost:11434/v1 MAX_QUERIES=100
```

Generation eval reports faithfulness (with judge variance) **next to** a
ground-truth-anchored context recall. High faithfulness + low context recall =
the answer is confidently grounded in the *wrong* documents — the failure mode a
faithfulness-only score hides. The judge is a separate local model from the
generator (it does not grade its own homework) and is never a cloud API.

`compare` emits a side-by-side report (dense-only vs dense+rerank) with
per-query P@1 flips — which queries the reranker fixed and which it broke. The
fixed/broke lists, not the headline delta, are where the analysis lives.

Reports (scores + metadata) are written to `reports/<timestamp>/`. The runner
verifies the corpus content-hashes against the frozen golden manifest and aborts
if they disagree, so a score is always traceable to the exact documents its
labels assume. See the `Makefile` for the exact commands.

## Repo layout

```
configs/        eval + model registry (frozen golden ref, pinned judge, seeds)
data/golden/    frozen, versioned golden evaluation sets (see README inside)
data/corpus/    source documents + licensing notes (see README inside)
src/elv/        the system
  ingest/         dirty-data ingestion pipeline
  index/          vector store abstraction (Faiss / Qdrant)
  retrieve/       two-stage retrieval (dense → rerank), hybrid hook
  generate/       vLLM / LiteLLM OpenAI-compatible client
  eval/           THE CORE: metrics + golden-set loader + runner
  agent/          reserved extension point, intentionally not implemented
docs/           architecture, eval-report template, ADRs
reports/        eval run outputs
tests/          pytest smoke gate (CI-style)
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Third-party models and datasets carry
their own licenses — verify each (e.g. Gemma terms, Llama community license for
ELYZA/Swallow, dataset licenses for any committed golden set) before
redistribution.
