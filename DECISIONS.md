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

**Decision.** Default payload = **Gemma 4 (27B, quantized)**. Carry **Gemma 3**,
**Swallow** and **ELYZA-JP** as comparison models. Use a **smaller/faster** model
(Gemma 4 12B or ELYZA-JP-8B) during eval-harness development for fast iteration
loops; run the full multi-model comparison only for the *published* eval report.

**Why.**
- Gemma 4 27B fits a single 32 GB GPU when quantized, is multilingual (Japanese
  is a natively supported language), and ships under **Apache 2.0** — a cleaner,
  more permissive license than Gemma 3's custom terms, which strengthens the
  "model on your hardware, no usage strings attached" story in a Japanese
  enterprise security review.
- Gemma 3 is kept as a comparison baseline on purpose: "newer" is a benchmark
  claim, not a guarantee on *this* task. The harness measures whether Gemma 4
  actually beats Gemma 3 on our golden set (faithfulness, context handling) —
  the changelog does not get to decide that.
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

**Revisit when.** A clean-license model meaningfully beats Gemma 4 / Swallow on
our golden set within the 32 GB budget. (Updated 2026-06: default moved
Gemma 3 → Gemma 4 on its Apache-2.0 release; Gemma 3 retained as comparison.)

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

---

## ADR-0006 — Hybrid experiment Step 0: ceiling prior + field definitions
*Status: accepted · 2026-06-08*

**Context.** Before implementing hybrid retrieval configurations (A0–H4/R0 from
`EXPERIMENT_hybrid.md`), we need to verify that the context_recall gap is a
ranking problem (solvable by reranking within 100 candidates) rather than a
candidate-coverage problem (where relevant docs are absent from the fixed 100,
making any reranker helpless).

**Script location.** `scripts/ceiling_check.py`. Run with:
```
python scripts/ceiling_check.py --embedder ruri   # full run with rank distribution
python scripts/ceiling_check.py --no-rank-dist    # ceiling only, no model needed
```
Deterministic: re-runs produce identical numbers. Requires HuggingFace network
access for JQaRA dataset; ruri-v3-310m embedder downloads on first run (~620 MB).

**JQaRA relevance label (口径).**
JQaRA uses binary relevance labels (0 / 1). The field is `label` in the dataset
(`hotchpotch/JQaRA`, test split). A document is "relevant" if and only if
`int(row["label"]) == 1` — identical to the threshold used in `adapters/jqara.py`
to build `relevant_doc_ids` in the golden set, and therefore identical to the
definition used in all `context_recall` calculations. No separate threshold
decision is needed; there are no graded relevance levels.

**Field definitions (canonical — must be used consistently in all subsequent steps).**

- **binary_ceiling**: fraction of queries where ALL relevant docs appear in the
  JQaRA-assigned 100 candidates. For JQaRA test split this is always 1.0
  (dataset construction guarantees). Reported as a validity check only.

- **oracle_recall@k**: `mean over queries of min(|relevant|, k) / |relevant|`.
  Since `|relevant ∩ candidates| = |relevant|` for JQaRA (binary_ceiling = 1.0),
  this simplifies to `mean min(|relevant|, k) / |relevant|`. This is the maximum
  achievable context_recall@k within the 100 candidates under a perfect reranker.

- **dense_recall@k within 100**: actual recall@k when the 100 JQaRA candidates
  are ranked by ruri-v3 cosine similarity (restricted pool). Uses the same
  `recall_at_k` formula as `metrics_retrieval.py`. This is the correct baseline
  for attributing hybrid improvement: delta above this number comes from BM25
  complementing dense, not from switching from full-corpus to candidate-pool.

- **current recall@k (full corpus)**: recall@k from the frozen eval reports,
  full 144K corpus, dense-only or dense+rerank. Used to compute the gate gap.

- **gap@k**: oracle_recall@k − current_recall@k (full corpus baseline). The
  gate decision uses this number.

**Step 0 results (frozen).**

| Eval set | oracle@5 | dense@5 (within 100) | current@5 (full corpus) | gap@5 |
|---|---|---|---|---|
| gen (100 q) | 0.6113 | 0.4224 | 0.4062 | +0.2051 |
| retrieval (1667 q) | 0.6489 | 0.4368 | 0.4256 | +0.2233 |

Rank distribution within 100 candidates under ruri-v3 dense ordering:
p50=1, p90=2, max=55–88 (first relevant doc already ranks very high).

**Gate decision.** Both gaps are ≥ 0.15. Full hybrid experiment continues
(all configurations A0–H4/R0 from `EXPERIMENT_hybrid.md §2`).

**Critical nuance for Step 1+.** The p50 rank = 1 shows that the dense model
already places the first relevant doc at the top within the 100 candidates for
most queries. The gap to oracle (≈ 0.19) is **structural**: queries have 6–28
relevant docs and k=5 can only surface 5. Hybrid BM25+dense may recover
complementary relevant docs that dense alone missed; track delta against
dense_recall@5_within_100 (≈ 0.42–0.44), not just the full-corpus baseline
(0.41), to correctly attribute any improvement.

**Revisit when.** A customer corpus with different candidate label density
makes the structural ceiling analysis non-trivially different from JQaRA.

---

## ADR-0007 — Hybrid experiment: pinned parameters
*Status: accepted · 2026-06-08*

**Context.** Reproducibility requires pinning every free parameter before
running Step 1. Undeclared choices (k in RRF, tokenizer mode, reranker range)
become confounds that make results irreproducible and comparisons invalid.
Parameters must be declared here, not discovered post-hoc.

**Pinned parameters (immutable once Step 1 starts).**

| Parameter | Value | Why |
|---|---|---|
| RRF k constant | 60 | Standard literature default (Cormack et al. 2009); rank-only, so k trades off top-rank vs tail sensitivity. 60 is the safe middle ground and the most-cited value. |
| SudachiPy split mode | C (longest unit) | Mode C produces longest compound tokens — best coverage for Japanese IR; modes A/B fragment too aggressively and inflate term frequency noise. Confirmed available in the project venv. |
| MeCab dictionary | IPAdic | Most common baseline for Japanese NLP; available on-prem without extra license. Enables like-for-like comparison with SudachiPy in A3 without introducing a second vocabulary variable. |
| H4 reranker input range | top-20 of H1 or H2 fusion output | Cross-encoders are O(k) at inference; top-20 covers the oracle@k ceiling comfortably (p90 first-relevant rank ≤ 2) while keeping latency bounded. Applying to the full 100 would be ~5× slower for negligible tail gain. |
| H3 weighted-norm α | 0.5 (dense : sparse = 50 : 50) | Fixed in advance; must not be tuned on the test set — that would be test-set overfitting and make H3 numbers incomparable to H1/H2 (which are parameter-free). 0.5 is the neutral uninformative prior. |
| Bootstrap seed | 42 | Determinism; re-runs give identical CI bounds. |
| Bootstrap resamples | 10 000 | Standard for 95% CI precision at this sample size (1667 queries). |

**Verification gate (before Step 1 continues):**
- Confirm `sudachipy` and `sudachipy-dictionary-small` importable in `.venv`
- Confirm `mecab-python3` and `ipadic` importable in `.venv`
- Run `ollama list` to record exact reranker model tag (BGE-reranker-v2-m3 variant)

**Revisit when.** A configuration change forces one of these values to shift;
update this ADR with the new value and rationale before re-running.

---

## ADR-0008 — Hybrid experiment: archived, original motivation disproved
*Status: accepted · 2026-06-08*

**Context.** The hybrid retrieval experiment (EXPERIMENT_hybrid.md, ADR-0006/0007) was
designed to improve `context_recall` from 0.41 and fix the 33/100 grounded-but-wrong
queries identified in the Phase 4 generation eval. After completing Step 0 (ceiling
gate) and the subsequent metric analysis, the core motivation was found to be invalid.

**Full reasoning chain (sequential — each step built on the previous).**

1. *Original motivation.* Two related problems drove the experiment:
   - `context_recall_docs` (proportion recall) = 0.41 — appeared to show retrieval
     failing on ~59% of queries.
   - 33/100 grounded-but-wrong (faithfulness ≥ 0.8 AND proportion_recall < 0.5) —
     appeared to show the model confabulating correct-sounding answers from wrong docs.

2. *Step 0 ceiling gate (ADR-0006, `reports/ceiling_check.md`).* Computed oracle
   recall@5 within JQaRA's fixed 100 candidates. Gap = oracle − current = +0.20 at
   k=5, placing the experiment in the "≥0.15 → full experiment continues" bucket.
   This gate passed and appeared to justify proceeding.

3. *Gap decomposition (`reports/ceiling_check.md §5`).* Decomposed the +0.20 gap:
   - k-truncation-locked: 0.35 (structurally unreachable at k=5; mean query has 9.7
     relevant docs and k=5 holds only 5).
   - Sorting-improvable: 0.22 (could be recovered by better ranking within candidates).
   Rank distribution p50=1, p90=2 showed dense already ranks the first relevant doc
   near the top. The headroom was structural, not a ranking problem.

4. *Metric analysis (`reports/recall_metric_analysis.md`).* Computed `hit@5` (binary:
   ≥1 relevant doc in top-5) alongside proportion recall for the 100-query Phase 4
   sample. Results:
   - hit@5 = 0.98 (98/100 queries had at least one relevant doc in top-5).
   - Only 2 queries had hit@5=0; both had faithfulness=0.0 — the judge correctly
     gave them no credit. No grounded-but-wrong cases among them.
   - Of the 33 grounded-but-wrong queries: **0 true failures (hit@5=0), 33 metric
     artifacts (hit@5≥1, proportion_recall < 0.5 due to large denominator)**.
   - Root cause: 28 of the 33 queries have n_rel > 10. At k=5, oracle proportion
     recall = 5/n < 0.5 — the grounded-but-wrong label was **structurally impossible
     to remove even with a perfect retriever**. The 0.5 threshold was calibrated for
     a low-n_rel dataset and is wrong for JQaRA (mean n_rel = 9.7).

5. *Conclusion.* The two motivating problems were both metric artifacts, not pipeline
   defects:
   - `context_recall = 0.41` correctly measures retrieval completeness for a
     reranking benchmark with ~10 relevant docs per query. It is not a failure signal
     for QA sufficiency.
   - `33/100 grounded-but-wrong` is 100% explained by the mismatch between the 0.5
     threshold and JQaRA's multi-answer label density. The model had at least one
     relevant document for 98% of queries.
   Hybrid reranking could improve proportion_recall without changing hit@5 — it would
   move the metric label without fixing an actual QA failure.

**Decision.** Archive the hybrid experiment. Do not run configurations A0–H4/R0.
`EXPERIMENT_hybrid.md` and ADR-0006/0007 are retained as a record of a
**rational, evidence-driven cancellation** — the methodology (ceiling gate + metric
audit) worked as designed; it correctly surfaced a bad premise before machine time
was spent.

**What remains useful from the spec.**
- ADR-0007 pinned parameters (RRF k=60, SudachiPy mode C, MeCab IPAdic, α=0.5) remain
  valid if BM25 configurations are ever revisited for a different task.
- The H1/H2/A2 configurations (dense vs BM25-local IDF vs BM25-global IDF within
  100 candidates) could serve as illustration data for a blog post on local-IDF
  degradation in small candidate pools, if that topic becomes content-worthy.

**Revisit when.** A new dataset or task where (a) n_rel per query is small (≤ 3),
(b) hit@k and proportion_recall diverge less, and (c) retrieval failures genuinely
exist at hit@k=0 for a meaningful fraction of queries. Under those conditions the
hybrid architecture and the pinned parameters in ADR-0007 are ready to use.
