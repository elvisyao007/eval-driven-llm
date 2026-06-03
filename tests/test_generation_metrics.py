"""Generation-metric logic tests with controllable fake judges, so assertions
are exact. The real LocalLLMJudge path needs a served model and is config-
swapped on the target machine."""

from elv.eval import metrics_generation as mg


class _Judge:
    """Fake judge: fixed claim count, and `support` decides entailment."""
    def __init__(self, claims, support):
        self._claims = claims
        self._support = support  # callable(claim)->bool or bool

    def extract_claims(self, question, answer):
        return list(self._claims)

    def entails(self, context, claim):
        return self._support(claim) if callable(self._support) else self._support


def _sample(**kw):
    base = dict(query="q", answer="a", contexts=["ctx"],
                retrieved_doc_ids=[], relevant_doc_ids=set())
    base.update(kw)
    return mg.GenerationSample(**base)


def test_faithfulness_all_supported():
    j = _Judge(["c1", "c2", "c3"], True)
    r = mg.faithfulness(_sample(), j, runs_per_item=3)
    assert r.mean == 1.0 and r.spread == 0.0


def test_faithfulness_half_supported():
    j = _Judge(["c1", "c2"], lambda c: c == "c1")
    r = mg.faithfulness(_sample(), j, runs_per_item=1)
    assert r.mean == 0.5


def test_faithfulness_no_claims_is_zero():
    j = _Judge([], True)
    assert mg.faithfulness(_sample(), j, runs_per_item=2).mean == 0.0


def test_context_recall_docs_deterministic():
    s = _sample(retrieved_doc_ids=["A", "X"], relevant_doc_ids={"A", "B"})
    assert mg.context_recall_docs(s) == 0.5
    s2 = _sample(retrieved_doc_ids=["A", "B"], relevant_doc_ids={"A", "B"})
    assert mg.context_recall_docs(s2) == 1.0


def test_grounded_but_wrong_flag():
    assert mg.grounded_but_wrong_flag(0.95, 0.2) is True   # faithful, bad retrieval
    assert mg.grounded_but_wrong_flag(0.95, 0.9) is False  # faithful, good retrieval
    assert mg.grounded_but_wrong_flag(0.3, 0.2) is False   # not faithful


def test_deterministic_test_judge_is_reproducible():
    from elv.eval.judge import DeterministicTestJudge
    j = DeterministicTestJudge()
    s = _sample(answer="vLLMは推論を高速化する。", contexts=["vLLMは推論を高速化する技術"])
    a = mg.faithfulness(s, j, runs_per_item=2)
    b = mg.faithfulness(s, j, runs_per_item=2)
    assert a.mean == b.mean and a.spread == 0.0
