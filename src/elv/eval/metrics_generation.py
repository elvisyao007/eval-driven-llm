"""Generation metrics (judge-based).

Headline metric is faithfulness: is every claim in the answer supported by the
retrieved context? It is judge-based and therefore not deterministic the way
retrieval metrics are, so we (ADR-0001):
  1. inject a PINNED judge (model/temp/seed fixed in config), and
  2. repeat the judgment `runs_per_item` times and report the SPREAD — we
     measure judge variance instead of pretending a single number is exact.

Critical caveat encoded below: high faithfulness does NOT mean a correct answer.
A pipeline can be perfectly faithful to the WRONG retrieved context. That is why
faithfulness must always be read next to a ground-truth-anchored signal
(context_recall_docs here, retrieval recall in the report).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from .judge import Judge


@dataclass(frozen=True)
class JudgeConfig:
    """Pinned judge settings — anything affecting judge output lives here so a
    run is reproducible from config alone."""
    model: str
    base_url: str = ""
    temperature: float = 0.0
    seed: int = 0
    runs_per_item: int = 3


@dataclass
class GenerationSample:
    query: str
    answer: str
    contexts: list[str]
    retrieved_doc_ids: list[str] = field(default_factory=list)
    relevant_doc_ids: set[str] = field(default_factory=set)  # from golden truth


@dataclass(frozen=True)
class MetricResult:
    mean: float
    runs: tuple[float, ...]

    @property
    def spread(self) -> float:
        return (max(self.runs) - min(self.runs)) if self.runs else 0.0

    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.runs) if len(self.runs) > 1 else 0.0


def faithfulness(sample: GenerationSample, judge: Judge, runs_per_item: int = 3) -> MetricResult:
    """supported_claims / total_claims, averaged over the contexts; repeated
    `runs_per_item` times to surface judge variance.

    Reminder: a high value only means the answer is grounded in whatever was
    retrieved — not that the retrieval was correct (see module docstring)."""
    context = "\n\n".join(sample.contexts)
    runs: list[float] = []
    for _ in range(max(1, runs_per_item)):
        claims = judge.extract_claims(sample.query, sample.answer)
        if not claims:
            runs.append(0.0)
            continue
        supported = sum(1 for c in claims if judge.entails(context, c))
        runs.append(supported / len(claims))
    return MetricResult(mean=statistics.fmean(runs), runs=tuple(runs))


def context_recall_docs(sample: GenerationSample) -> float:
    """Deterministic, ground-truth-anchored: fraction of the golden relevant
    documents that actually made it into the generation context window.

    This is the anchor faithfulness is read against. Low recall + high
    faithfulness = the system is confidently grounded in the WRONG documents."""
    if not sample.relevant_doc_ids:
        return 0.0
    got = set(sample.retrieved_doc_ids) & sample.relevant_doc_ids
    return len(got) / len(sample.relevant_doc_ids)


def grounded_but_wrong_flag(faith: float, ctx_recall: float,
                            faith_hi: float = 0.8, recall_lo: float = 0.5) -> bool:
    """True when the answer looks faithful but retrieval missed the truth — the
    failure mode a faithfulness-only eval would hide."""
    return faith >= faith_hi and ctx_recall < recall_lo
