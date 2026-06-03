# CLAUDE.md — working guide for this repo

Read this and `DECISIONS.md` before changing anything. This file is terse on
purpose (it loads every session). Rationale lives in `DECISIONS.md` (ADRs).

## What this is
An on-prem, eval-first LLM system. The eval methodology is the backbone; the RAG
pipeline is the current, swappable payload. Optimize for a system that survives
enterprise reality, not a demo that runs once.

## Non-negotiable working principle: default to enterprise reality
Every component assumes dirty data, multiple users, security, maintainability,
and measurability — never an ideal/clean-notebook scenario. Concretely:
- ingestion tolerates malformed input (log + skip, never crash the batch),
  fixes encoding, dedups, and writes an audit trail
- retrieval/index must support per-user permission filtering (ACL metadata)
- eval ties to ground truth and, where possible, to business impact
- ship maintenance/handover docs and a stable API, not a black box

## Eval ethic (the whole point — do not violate)
- NEVER write fabricated, placeholder, or "example" metric numbers as if real.
  An unwired metric is reported as "run pending", not invented.
- Retrieval metrics are deterministic (recall@k/MRR/nDCG/P@1) — keep them so.
- Judge-based metrics use a PINNED judge (model/temp/seed in config) and report
  variance across runs. Read faithfulness ALONGSIDE ground-truth context recall:
  high faithfulness + low context recall = grounded in the wrong docs (flag it).
- Golden sets are frozen + versioned; verify corpus hashes before scoring.

## Locked architecture decisions (see DECISIONS.md for why)
- The CORE is eval; on-prem deploy is the carrier; RAG/agent/frameworks/models
  are swappable payload behind config — do NOT hard-code or over-invest in them.
- Deployment models are clean-license / local only (Gemma 4 / Swallow / ELYZA-JP
  via vLLM, OpenAI-compatible). No cloud APIs in the deployment path.
- Agent layer is reserved, NOT implemented (ADR-0005). Keep the extension point;
  don't build an agent product until a concrete use case defines it.
- Retrieval is two-stage (ruri-v3 dense → BGE-reranker-v2-m3). Qdrant is the
  customer-facing store (ACL filtering); Faiss is the benchmark store.

## Commit discipline
Small, meaningful commits with messages that say what each step solves. Never
one big dump — the git history is meant to show real iteration. Don't commit
copyrighted/large corpora or model weights; commit build scripts + hashes
(`data/golden/README.md`).

## How to run
See the `Makefile`. Baseline loop needs no GPU/downloads (`make install-base`,
`make eval-toy`, `make compare-toy`, `make rag-eval-toy`). Real runs need the GPU
+ HF (`make install`, `make build-jqara`, `make compare`, `make rag-eval`).
Run `make test` (pytest) after changes; keep it green.

## Out of scope (do not propose)
Markets already excluded: finance, medical, insurance, public sector; pure
IT/Web/SaaS firms; large enterprises; commoditized EC-support RAG. The sweet
spot is "data can't go to the cloud × company too small for the big vendors",
targeted by pain, not by industry.
