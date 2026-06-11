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

---

## ADR-0009 — blog-01/02 correction: grounded-but-wrong 33/100 is a metric artifact
*Status: accepted · 2026-06-08*

**Context.** blog-01 and blog-02 (both published on dev.to and Zenn) cite
`grounded-but-wrong = 33/100` as evidence of a retrieval failure mode — faithful
answers grounded in wrong documents. The metric analysis in commit 9600846
(`reports/recall_metric_analysis.md`) proved this number is a structural artifact,
not a signal of actual retrieval failure.

**What the analysis showed.**

- `context_recall_docs` is ID-based proportion recall: `|retrieved ∩ relevant| / |relevant|`
  (NonLLMContextRecall / IDBasedContextRecall in RAGAS terminology; no LLM involved).
- The 33 flagged queries have mean n_rel = 16.0; 28/33 have n_rel > 10.
- At k=5, oracle proportion recall = 5/n_rel < 0.5 for any n_rel > 10 — the
  `grounded_but_wrong_flag` threshold of 0.5 is **structurally impossible to clear
  even with a perfect retriever** for these queries.
- `hit@5` (binary: ≥1 relevant doc in top-5) = 98/100. The 2 queries with hit@5=0
  both had faithfulness=0.0 and were not in the 33. Zero true retrieval failures.
- All 33 grounded-but-wrong cases had hit@5=1; the model had relevant context
  available for every one of them.

**What is NOT affected.**

- The blog-02 conclusion comparing self-grading (qwen3 judges itself) vs independent
  judge (gemma4:31b) remains fully valid — that comparison used the same metric on
  both runs, so the delta (0.7751 → 0.6662 faithfulness, zero spread → 0.05 spread)
  is unaffected by the absolute metric choice.
- The faithfulness numbers themselves are unaffected.
- The retrieval benchmark (P@1, MRR, nDCG, recall@k from runner.py) is unaffected —
  those metrics are correct for the retrieval task.

**Correction plan (content, not code).**

1. Publish blog-03 (full diagnostic: hit@k vs proportion recall on multi-answer data)
   in English on dev.to first.
2. Add an update note at the top of blog-01 and blog-02 on both dev.to and Zenn,
   pointing readers to blog-03. Original text preserved; no silent edits.
3. Publish Japanese Zenn translation of blog-03 after the English version is live.

**No code changes required.** The `context_recall_docs` implementation is correctly
named and correctly implemented for retrieval evaluation. The issue is the
interpretation of the 0.5 threshold for a multi-answer reranking dataset used as a
QA benchmark. The threshold is a hyperparameter in `grounded_but_wrong_flag`
(`metrics_generation.py:89`), not a bug.

**Revisit when.** A future dataset with low n_rel per query (≤ 3) would make
proportion recall and hit@k converge, restoring the original interpretation.

---

## ADR-0010 — Model selection benchmark v1: qwen3 as judge, not competitor
*Status: accepted · 2026-06-11*

**Context.** The model-selection benchmark (`reports/model-selection-v1/`) evaluates
four deployment candidates: gemma4:31b, ELYZA-JP-8B, Swallow-8B-Instruct, and
Nemotron-Nano-9B-JP. A judge model is needed to score faithfulness and answer
accuracy. qwen3:32b is available on this machine and is the most capable local model.

**Options.**
- A. Use qwen3:32b as both a competitor and the primary judge.
- B. Exclude qwen3 from the competitor list and use it only as judge.
- C. Use gemma4:31b as judge (it is already a competitor).

**Decision: B.** qwen3:32b judges, does not compete.

**Why.**
1. **Deployment/content layer separation (standing project rule — see ADR-0003).**
   Chinese frontier models are excluded from the deployment layer on data-sovereignty
   and compliance grounds for Japanese on-prem customers. qwen3 therefore belongs to
   the *content/research layer* (radar, evaluation tooling) rather than the
   *deployment payload layer*. It is architecturally ineligible as a deployment
   candidate, so adding it to the competitor list would blur a meaningful boundary.

2. **Self-preference / self-grading bias.** LLM judges are known to favour outputs
   stylistically similar to their own training distribution. A judge scoring its own
   outputs (or outputs of closely related models) inflates scores in a way that is
   hard to detect and harder to defend to a customer. The established mitigation is to
   ensure judge ≠ any competitor — which B achieves and A violates.

3. **Judge capability.** qwen3:32b is the strongest available local model and therefore
   the best judge for quality signals. Demoting it to a competitor to make the lineup
   "fair" would make the judge weaker for no benefit.

**Rejected.**
- A — mixes judge and competitor roles; creates self-grading risk for a model that is
  also architecturally excluded from the deployment layer.
- C — gemma4 IS a competitor; a competitor judging itself or the field is the exact
  self-grading problem this rule is designed to prevent.

**Isolation guarantee.** gemma4:31b is used as a *cross-validation judge* on a 25-item
subset **solely to verify that qwen3's judgments are reliable** — it is not used to
compute gemma4's own main scores. gemma4's main scores are computed by qwen3. This
isolation is enforced in `run_benchmark.py` (phase2 uses qwen3 for all models;
phase3 cross-validates on a subset using gemma4 but writes a separate
`cross_validation.json`, not back into gemma4's judged results).

**Revisit when.** A local model from a compliance-clear vendor (Apache 2.0 or MIT,
non-Chinese origin) matches qwen3:32b judgment quality — then that model can serve as
primary judge without the deployment-layer conflict.

---

## ADR-0011 — Why 8B and 31B models are compared on the same table
*Status: accepted · 2026-06-11*

**Context.** The model-selection benchmark puts ELYZA-JP-8B and Swallow-8B-Instruct
(8B) alongside gemma4:31b (31B) in the same results table. A common objection is
that this is an unfair apples-to-oranges comparison that a 31B model will trivially
win.

**Decision.** Compare them on the same table, clearly labelled.

**Why.**
The benchmark answers a **constraint-under-deployment** question, not an
**absolute quality ranking**. The right frame is: *given a VRAM budget, a latency
target, and a Japanese quality threshold, which model do I deploy?* Under that frame:
- A customer with a single 10 GB GPU **cannot deploy** gemma4:31b regardless of its
  quality advantage; the relevant question for them is which 8B model wins.
- A customer with 20 GB and no latency constraint should probably deploy gemma4:31b
  if the quality gap justifies the extra VRAM.
- The quality gap itself must be *measured* — it is not given. An 8B Japanese
  specialist might narrow the gap enough that the VRAM savings outweigh it.

The decision table in `summary.md` makes the constraint mapping explicit:
"VRAM ≤ 10 GB → 8B family; VRAM ≤ 20 GB → gemma4:31b if quality gap > threshold."
This is more useful than a one-winner ranking.

**What the table does NOT claim.** It does not claim the comparison is architecturally
symmetric. Parameter count, quantization, and training distribution are all different.
The table explicitly lists parameters and quantization so readers can calibrate.

**Revisit when.** A new 8B model with comparable quality to 31B emerges (then the
VRAM constraint no longer distinguishes them meaningfully and the decision table
collapses to one recommendation).

---

## ADR-0012 — Cross-validation protocol: primary judge scores + subset agreement check
*Status: accepted · 2026-06-11*

**Context.** Any single-judge eval carries judge reliability risk: the judge may be
systematically biased, misconfigured, or simply wrong on a class of inputs. The
benchmark uses a single primary judge (qwen3:32b) for all models, which gives
consistent, comparable scores but provides no self-check on that judge's reliability.

**Options.**
- A. Run all models through two independent judges and average scores.
- B. Run primary judge for all, then run a second judge on a held-out subset; report
  agreement between the two judges as a reliability signal rather than blending scores.
- C. Rely on the primary judge alone.

**Decision: B.**

**Why.**
- A blends scores (A) would require two judge passes over all items, roughly doubling
  VRAM switching cost and total run time. It also raises the question of how to
  combine conflicting judgments; averaged scores obscure real disagreements.
- B keeps main scores clean and comparable (primary judge only), while exposing
  judge reliability explicitly as an observable. If primary and cross judge agree
  ≥ 80%, the primary judge's scores can be trusted. If they disagree substantially,
  that signals an unreliable evaluation and warrants investigation before publishing.
- C provides no signal on judge quality, making the scores undefendable if challenged.

**Protocol details (immutable for this benchmark run).**
- Primary judge: qwen3:32b — scores all items for all competitors.
- Cross judge: gemma4:31b — scores a 25-item random subset (seed 42) drawn from one
  competitor's judged results (excluding gemma4's own results to avoid self-grading).
- Agreement metrics reported: hit agreement rate (binary match), Cohen's κ (hit),
  faithfulness agreement rate (|Δ| < 0.2).
- Disagreements listed individually in `cross_validation.json` for manual inspection.

**Isolation invariant.** gemma4's cross-judge pass **must not feed back into any
competitor's main scores**, including its own. Cross-validation output lives only in
`results/cross_validation.json`; `results/*_judged.json` files are written exclusively
by the primary judge (qwen3).

**Revisit when.** The benchmark is re-run with a third independent judge, or a
well-studied judge-calibration dataset for Japanese QA becomes available to replace
the subset agreement approach with a calibrated reliability bound.

---

## ADR-0013 — Benchmark discriminability is a prerequisite for meaningful model selection
*Status: accepted · 2026-06-11*

**Context.** The v1 model-selection golden set produced 18/20 all-correct items (90%),
Cohen's κ=1.0 between primary and cross-validation judges, and a hit_rate range of
only 0.90–1.00 across models. A benchmark in this state cannot distinguish 8B from 31B
models; any "winner" chosen from such results would be arbitrary.

**The core problem.** When nearly all questions are answered correctly by all models,
two properties fail simultaneously:
1. **Selection is uninformative.** A score spread of 0.10 across four models cannot
   support a deployment decision; the uncertainty in the judge itself (prompt phrasing,
   temperature, single-run variance) exceeds the measured difference.
2. **Cross-validation is trivially perfect.** When every answer is correct, any two
   judges will agree 100% — not because the judges are calibrated, but because there is
   nothing to disagree about. κ=1.0 in a zero-variance setting is meaningless.

**Why v1 golden set had zero discriminability.** The v1 questions were factual recall
(絶対零度 = −273.15°C, 富士山 = 3776m, 光速 = 299,792 km/s). All four of these facts
are saturated in LLM training corpora. Models answer correctly without reading the
provided context. Context-grounded evaluation that models can bypass via memorisation is
not evaluation of grounding; it is evaluation of training data coverage.

**Decision.** A golden set must achieve partial discrimination (some models wrong,
some right) on at least ~40% of items before a selection report can be published.
All-correct rate ≥ 80% is a redesign signal, not an acceptable result.

**v2 difficulty design (applied principles — not a checklist to follow blindly).**

| Principle | Rationale |
|---|---|
| Fictional specs as context | Model cannot answer from training data; must read the context. |
| Multi-step reasoning chains | Single-step fact retrieval is insufficient; mistakes accumulate. |
| Completeness via `expected_points` | Partial answers score zero; rewards thoroughness, not breadth. |
| Dense similar values | Multiple plausible numbers in context; wrong number selection → fail. |
| Japanese language nuance (keigo, pronouns) | Tests JP-specific capability, not generic QA skill. |
| Negative/exclusion reasoning | "What is NOT included" requires full context scan, not keyword match. |

**Empirical result after v2 redesign (45 items).**

| Category | v1 (20 items) | v2 (45 items) |
|---|---|---|
| All models correct | 90% | 29% |
| Partial discrimination | 10% | 51% |
| All models wrong | 0% | 20% |
| Cohen's κ (cross-val) | 1.0 (trivial) | 0.920 (substantive) |
| hit_rate spread | 0.10 | 0.22 |

The v1 results are preserved in `results/` and `summary.md`. The correction is visible
by design — a benchmark correction that is hidden is a benchmark that cannot be trusted.

**What all-models-wrong items indicate.** 20% of v2 items were answered incorrectly by
all four models. This is expected and acceptable: items that even the strongest model
fails confirm the golden set is genuinely hard. An ideal discrimination curve has most
items in the partial zone, with some floor (all-wrong) and some ceiling (all-correct).
A floor above ~30% signals the golden set is too hard and produces noisy signals.

**Revisit when.** A new golden set is designed for a different task domain, or if the
all-wrong rate rises above 30% (indicating the floor is too high and scores become
noise-dominated rather than capability-dominated).

---

## ADR-0014 — DeepDoc 独立モジュール採用: 完全 RAGFlow スタックを起動しない理由

**日付**: 2026-06-12  
**ステータス**: 採用 (Phase 1 完了)

### 背景
China bridge コンテンツシリーズの技術基盤として、日本語 PDF の解析品質を評価する必要が生じた。RAGFlow には深層文書理解モジュール (DeepDoc) が含まれており、OCR・レイアウト認識・表構造認識 (TSR) を提供する。

### 制約
- Ollama (qwen3:32b / gemma4:31b) が常駐しているため、追加の重いサービスを同居させるとメモリ競合が起きる。
- Elasticsearch・Redis・MinIO は eval の目的には不要であり、起動コストとネットワーク依存を増やすだけ。
- セキュリティ・監査の観点でサービス境界を最小にする (CLAUDE.md の "enterprise reality" 原則)。

### 検討した選択肢

| 選択肢 | 却下理由 |
|--------|----------|
| 完全 RAGFlow スタック (Docker Compose) | ES/Redis/MinIO/Web UI が必須で起動コスト大、Ollama と競合 |
| RAGFlow の API サーバーのみ | 依然として ES + Redis が必要、SDK 版でも同様 |
| **DeepDoc 独立モジュール (採用)** | `deepdoc/vision` + `deepdoc/parser` のみ; サービス依存ゼロ |
| MinerU / PaddleOCR | v2 横評価のスコープ; Phase 1 は DeepDoc 単独で充分 |

### 採用した設計

- `git sparse-checkout` で `deepdoc/`, `common/`, `rag/` のみ取得。  
  (`/mnt/data/ragflow-deepdoc`)
- `common/settings.py` を最小スタブに差し替え (`PARALLEL_DEVICES=0`, `DOC_ENGINE_INFINITY=False`)。  
  完全 RAGFlow が使う `rag.utils.*` / `memory.utils.*` 等への依存を遮断。
- 独立 venv (`/mnt/data/eval-driven-llm/.venv-deepdoc`, Python 3.12) で現行 eval venv を汚染しない。
- `PYTHONPATH=/mnt/data/ragflow-deepdoc` で実行。GPU は CPU で動作確認後 Phase 2 で判断。

### 実際に試したこと
Phase 1 で以下を確認:
- `t_ocr.py --help`, `t_recognizer.py --help` — 両スクリプト動作  
- layout 認識 (sample 01: 4p, 03: 24p) — JPG 出力  
- TSR (sample 01: 4p) — HTML テーブル出力  
- OCR (sample 01: 4p, 03: 24p) — .txt 出力  
- CPU 速度: layout ~0.6–1.1s/p、OCR ~3.3–3.9s/p

### 再検討するとき
- Phase 2 でバッチ処理が必要になり GPU スピードアップを検証するとき。  
- MinerU / PaddleOCR との横評価 (v2) で別環境が必要になったとき。  
- DeepDoc の API が breaking change で sparse-checkout が保守困難になったとき。

---

## ADR-0015 — DeepDoc-v1 は単一ツール評価から始める (横評価は v2 スコープ)

**日付**: 2026-06-12  
**ステータス**: 採用 (Phase 1 完了)

### 背景
日本語 PDF 解析のベンチマークとして、DeepDoc / MinerU / PaddleOCR などの横評価が考えられる。

### 判断
v1 では DeepDoc 単独で解析パイプラインを動かして初期品質を把握する。横評価は v2 にスコープを限定する。

### 理由

1. **範囲冻结の原則** (CLAUDE.md): 一つの実験で複数の変数を変えると比較が汚れる。v1 は環境・依存・ベースラインを確立することが目的。
2. **スタック差異の大きさ**: MinerU は独自 venv + layout モデルのダウンロードが必要; PaddleOCR は CUDA 依存が強い。それぞれ独立したセットアップコストがある。
3. **先にベースラインを取る**: DeepDoc の出力フォーマット・品質・速度を把握しておかないと、横評価のスコア項目を設計できない。
4. **コストと判断の順序**: 横評価は v1 のメトリクス設計が確定してから実施する方が、評価観点がブレない。

### 再検討するとき
v1 Phase 2 で DeepDoc の品質スコアが確定したとき。そこから v2 の比較軸を設計する。
