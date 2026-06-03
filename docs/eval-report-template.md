# Eval report — <system version> / <golden set>@<version>

> Template. Copy into `reports/<timestamp>/report.md` for each run.
> Do not fill result tables with placeholder or imagined numbers — leave
> `TBD (run pending)` until a real run produces them. An invented metric is
> worse than a missing one in a repo whose whole point is honest evaluation.

**Run metadata**

| Field | Value |
|---|---|
| Date | `<ISO timestamp>` |
| System commit | `<git sha>` |
| Golden set | `<name>@<version>` (frozen) |
| Corpus | `<name>` + size + source |
| Generation model | `<model@quant>` |
| Embedding / reranker | `<embed>` / `<reranker>` |
| Judge model (pinned) | `<judge model @ temp / seed>` |
| Config hash | `<sha of configs/eval.yaml>` |

---

## 1. What we measure / why (scenario-specific)

- **Task type:** `<RAG retrieval / grounded QA / classification / ...>`
- **Metrics chosen + rationale:** state each metric and *why it matters for
  this scenario*. Example: faithfulness weighted high because the cost of a
  confident wrong answer is high in this use case.
- **Acceptance bar + basis:** the threshold each metric must clear to be
  "shippable," and *why that number* (tie to business impact where possible —
  see §5). This bar is agreed before the run, not after.

## 2. Test set

- Source / size / how constructed.
- Frozen version id; how it is versioned (see `data/golden/README.md`).
- Known coverage and known gaps (easy vs hard queries, edge cases included).

## 3. Method

- **Retrieval metrics (deterministic, no judge):** recall@k, MRR, nDCG@k, P@1
  computed against labeled relevant document ids.
- **Generation metrics (pinned judge):** faithfulness, answer-relevancy,
  context-precision/recall — judge model, temperature and seed pinned and
  recorded above.
- **Pipeline:** build golden → run system → score → regression-gate → (later)
  monitor. Each step reproducible from the `Makefile`.

## 4. Results

### 4a. Retrieval (deterministic)

| Metric | Baseline | Tuned | Δ |
|---|---|---|---|
| Recall@5 | TBD | TBD | — |
| Recall@10 | TBD | TBD | — |
| MRR | TBD | TBD | — |
| nDCG@10 | TBD | TBD | — |
| P@1 | TBD | TBD | — |

### 4b. Generation (judge-based, judge pinned)

| Metric | Baseline | Tuned | Δ | Judge-variance note |
|---|---|---|---|---|
| Faithfulness | TBD | TBD | — | run N times, report spread |
| Answer relevancy | TBD | TBD | — | |
| Context precision | TBD | TBD | — | |
| Context recall | TBD | TBD | — | |

### 4c. Cost / latency (production reality)

| Metric | Value |
|---|---|
| p50 / p95 latency | TBD |
| tokens (in/out) per query | TBD |
| GPU memory peak | TBD |

## 5. Business translation (the differentiator)

Translate at least one technical metric into a business quantity:
e.g. "retrieval recall X% → analyst finds the right document in N seconds
instead of M minutes → ~H hours/month saved." Without this section the report
is a lab artifact, not an acceptance document.

## 6. Interpretation + my judgment

- Which metric is doing the real work / which is overrated for this scenario.
- What I would choose and why; what I would *not* trust.
- Limitations of this evaluation (judge variance, golden-set coverage, corpus
  representativeness).
- Next step + what would change the conclusion.
