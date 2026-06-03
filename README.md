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
| Generation | Gemma 4 (27B, quantized) + Gemma 3 / Swallow / ELYZA-JP for comparison | Apache-2.0, multilingual, fits a single 32 GB GPU; comparison models measure whether newer actually wins |
| Retrieval (1st pass) | ruri-v3-310m (dense) — BGE-M3 (dense+sparse) as hybrid option | Japanese-OSS retrieval SOTA; self-hosted, zero data egress |
| Rerank (2nd pass) | BGE-Reranker-v2-m3 | two-stage retrieve → rerank for precision@1 |
| Serving / gateway | vLLM (OpenAI-compatible) + LiteLLM | already part of the lab; swappable behind one interface |
| Vector store | Faiss (benchmark track) / Qdrant (realism track) | Qdrant carries metadata filtering for per-user permissions |
| Eval | in-house deterministic retrieval metrics + RAGAS/DeepEval generation metrics (pinned judge) + frozen golden set | see [`DECISIONS.md`](DECISIONS.md) ADR-0001 |

Everything above is a config entry, not a hard dependency. The only thing that
is *not* meant to be swapped cheaply is the eval methodology.

## Eval results

First run pending. Results land in [`reports/`](reports/) and follow the
structure in [`docs/eval-report-template.md`](docs/eval-report-template.md).
No numbers are published here until a run produces them — placeholder metrics
defeat the entire purpose of this repo.

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

# end-to-end generation eval (faithfulness), pinned LOCAL judge:
make rag-eval GOLDEN=data/golden/jqara/v0 \
     BASE_URL=http://localhost:8000/v1 GEN_MODEL=gemma4-27b-q JUDGE_MODEL=elyza-jp-8b
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

Intended: Apache-2.0 (permissive, commercial-friendly). Add a `LICENSE` file
before publishing. Note: third-party models and datasets carry their own
licenses — verify each (e.g. Gemma terms, Llama community license for
ELYZA/Swallow, dataset licenses for any committed golden set) before
redistribution.
