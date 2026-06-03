# Decision log

Architecture decisions for this system, recorded as dated, reversible entries.
Each one states the constraints, the options considered, why one was chosen,
why the others were rejected, what was actually tried, and the condition under
which the decision should be revisited.

Format is loosely [ADR](https://adr.github.io/). Newest at the bottom.

---

## ADR-0001 — Eval is the backbone, not a library call
*Status: accepted · 2026-01*

**Context / constraints.** The whole premise is "objective, repeatable
definition of good." The eval layer has to be defensible to a customer signing
a "no pass, no pay" agreement, and reproducible across time and across model
swaps. It cannot be a black box.

**Options.**
- A. `import ragas`, report its four numbers, done.
- B. A managed platform (LangSmith / Phoenix / Maxim).
- C. A thin in-house harness: deterministic retrieval metrics computed
  directly, generation metrics borrowed from RAGAS/DeepEval but with a
  **pinned** judge, all anchored to a **frozen golden set**, wrapped in pytest
  so it runs as a CI gate.

**Decision: C.**

**Why.**
- LLM-as-judge metrics (faithfulness, answer-relevancy) are useful but cannot,
  on their own, tell a *factually wrong* retrieved context from a correct one —
  a pipeline can score 0.95 faithfulness and still answer wrong if retrieval
  fed it stale/incorrect passages. So judge metrics are necessary but not
  sufficient; they must sit on top of a labeled ground truth, not replace it.
- Reproducibility comes from things a managed platform hides: the golden set is
  **frozen and versioned**, the judge model + temperature + seed are
  **pinned** in `configs/eval.yaml`, and retrieval metrics (recall@k, MRR,
  nDCG@k, P@1) are **deterministic** given labels — no judge variance there at
  all. A frozen golden set per cycle is the only way metrics stay comparable
  over time.
- RAGAS is the de-facto metric vocabulary (faithfulness / answer-relevancy /
  context-precision / context-recall) and is framework-agnostic, so we adopt
  its *definitions* without adopting it as the harness. DeepEval gives the
  pytest/CI-gate shape we want for "this score is a build gate, not a
  notebook." We take the shape from DeepEval, the metric vocabulary from RAGAS,
  and own the orchestration so nothing is hidden.

**Rejected.**
- A — surrenders judgment to a library; produces numbers we cannot defend in an
  acceptance meeting; no ground-truth anchor.
- B — couples acceptance evidence to a vendor and (for most) ships data out;
  conflicts directly with the on-prem premise.

**Revisit when.** A judge model good enough to reliably detect wrong-context
appears, or a self-hostable platform makes the in-house orchestration redundant
without leaking data.

---

## ADR-0002 — Two-stage retrieval, store chosen for permissions not just speed
*Status: accepted · 2026-01*

**Context / constraints.** Japanese-language documents, on-prem, and — per the
enterprise-reality default — **multiple users with different access rights**.
Retrieval quality matters, but so does "user A must not retrieve user B's
documents," which is an architecture decision, not an afterthought.

**Options considered (retriever).** Single dense retriever; hybrid
(dense+sparse); two-stage (retrieve → rerank).

**Decision.** Two-stage: dense first pass with **ruri-v3-310m**, rerank top
candidates with **BGE-Reranker-v2-m3**. Keep a **hybrid** path (**BGE-M3**,
which does dense + sparse + multi-vector in one model) available as a
config-level option for wide-recall first passes.

**Why.**
- ruri-v3 is current Japanese retrieval SOTA on JMTEB and self-hosts with zero
  data egress — which is the point of on-prem. BGE-M3 collapses dense+sparse
  into one model, so a hybrid option costs almost no extra infrastructure.
- Empirically (a published 2000-query, 6-config Japanese benchmark): adding
  sparse mainly lifts **Recall@10**, barely moves **P@1**, and can shuffle
  mid-ranks enough to *lower* P@3. Conclusion baked into the design: use hybrid
  to widen the candidate pool, then let the reranker do the precision work —
  do not expect hybrid alone to fix top-1. This is why rerank is not optional.

**Decision (vector store).** Interface-first: `index/store.py` abstracts the
store. **Faiss** for the benchmark track (fastest to stand up, no server).
**Qdrant** for the realism track and anything customer-facing — because Qdrant
carries **payload/metadata filtering**, which is how per-user permission
filtering is enforced at query time. Permissions are a retrieval concern here,
not a bolt-on.

**Rejected.** Single dense retriever (leaves precision@1 on the table for hard
Japanese queries). Faiss everywhere (no native metadata-filtered permissions →
fails the multi-user enterprise-reality requirement).

**Revisit when.** A Japanese retriever clearly beats ruri-v3 on *our* golden
set (public benchmark rank ≠ our-data rank — see ADR-0004), or corpus size
forces a different index type.

---

## ADR-0003 — Generation model: clean-license payload, JP models as comparison
*Status: accepted · 2026-01*

**Context / constraints.** On-prem deployment for Japanese customers; must run
on a single 32 GB GPU (RTX 5090); license/compliance story must be clean enough
that a security-conscious customer cannot object on data-sovereignty grounds.

**Decision.** Default payload = **Gemma 3 (27B, quantized)**. Carry
**Swallow** and **ELYZA-JP** as Japanese-specialized comparison models.
Use a **smaller/faster** model (Gemma 3 12B or ELYZA-JP-8B) during eval-harness
development for fast iteration loops; run the full multi-model comparison only
for the *published* eval report.

**Why.**
- Gemma 3 27B fits a single 32 GB GPU when quantized, is multilingual, and has
  a clean, well-understood license — the "model running on your hardware,
  trained by a Western lab" story is the easiest to defend in a Japanese
  enterprise security review.
- Swallow (Tokyo Science Univ. / AIST, Japanese-continued-pretraining on Llama)
  and ELYZA-JP optimize Japanese token efficiency and JP-task quality; they are
  the right *comparison* axis, and the model-vs-model gap is itself a content
  piece because the eval harness makes it measurable.
- **Chinese frontier models are excluded from the deployment layer** (data-
  sovereignty / compliance optics for Japanese on-prem) and confined to the
  content/research layer only. This is a standing project rule, not a per-repo
  call.

**Rejected.** A 70B model (does not fit comfortably alongside embedding +
reranker + KV cache on 32 GB without aggressive quantization that muddies the
eval). Cloud API generation (defeats on-prem). Chinese models in deployment
(compliance).

**Revisit when.** A clean-license model meaningfully beats Gemma 3 / Swallow on
our golden set within the 32 GB budget.

---

## ADR-0004 — Two-track eval corpus: recognized benchmark + dirty real PDFs
*Status: accepted · 2026-01*

**Context / constraints.** Need (a) a credible, repeatable score on day one to
prove the harness works, and (b) something that exercises the enterprise-
reality differentiators (dirty data, real layouts) that a clean academic set
never touches.

**Decision. Run both tracks, on purpose.**
- **Track A — recognized benchmark.** Use **JQaRA** (a public Japanese
  retrieval-augmented QA dataset). Gives deterministic, comparable retrieval
  metrics against a set other people also use → credible "the harness produces
  repeatable scores" evidence immediately.
- **Track B — dirty realism.** A small corpus of **messy, public-domain
  Japanese PDFs** (multi-column, tables, ruby text, OCR noise) with a
  **hand-labeled golden QA set**. Exercises the dirty-data ingestion pipeline
  and end-to-end generation eval.

**Why two.** The gap between Track A and Track B numbers *is* the lesson and a
standing content theme: public-benchmark rank does not predict your-data
performance — you must benchmark on representative data before trusting any
retriever/model choice. Showing both, and the gap, is more honest and more
useful than either alone.

**Licensing caution (do not skip).** JQaRA and any PDF corpus carry their own
licenses (Wikipedia-derived content is typically CC BY-SA; government PDFs vary
by ministry). Verify before committing any derived golden set to a public repo;
when in doubt, commit the *labels + document IDs/hashes* and a fetch script
rather than the documents themselves.

**Revisit when.** A real (NDA-covered) customer corpus exists — then Track B is
replaced per-engagement and never committed publicly.

---

## ADR-0005 — Agent layer: reserved, not implemented
*Status: accepted · 2026-01*

**Context / constraints.** Agent/orchestration is the larger wave, but the
open-source framework landscape turns over on a monthly cadence; committing to
a specific agent framework now risks building on something obsolete within a
quarter.

**Decision.** Reserve a tool-calling / agent extension point in the
architecture (near-zero cost) and leave `src/elv/agent/` **intentionally
unimplemented**. Agent work lives in the content/radar layer (hands-on tests,
write-ups) — not as a maintained product surface — until a concrete need
defines the requirements.

**Why.** Pre-wiring an extension point avoids a future rewrite; building an
agent product now means maintaining a fast-decaying dependency for no current
payoff. The eval core is exactly what makes a future agent layer *measurable*
(agent reliability rides on the same acceptance methodology), so the core is
the right thing to invest in first.

**Revisit when.** A concrete use case (from real inbound, not speculation)
makes the agent requirements specific.
