"""Judges for generation metrics.

A judge does two jobs for faithfulness: decompose an answer into atomic claims,
and decide whether a claim is entailed by the retrieved context. Both are LLM
tasks in production. The judge is an injected dependency so the metric code does
not care which judge is used (ADR-0001).

Prompts are constants here on purpose: the prompt IS part of the methodology and
must be reviewable, not buried in f-strings.
"""

from __future__ import annotations

from typing import Protocol, Sequence

CLAIM_PROMPT = (
    "次の質問と回答を読み、回答に含まれる事実主張を、検証可能な最小単位で、"
    "1行に1つずつ列挙してください。前置きや番号は不要です。\n\n"
    "質問: {question}\n回答: {answer}\n\n主張:"
)

ENTAIL_PROMPT = (
    "文脈だけを根拠に、次の主張が支持されるか判定してください。"
    "文脈に根拠があれば Yes、なければ No、判断できなければ Unclear と"
    "1語だけで答えてください。\n\n"
    "文脈:\n{context}\n\n主張: {claim}\n\n判定:"
)


class Judge(Protocol):
    def extract_claims(self, question: str, answer: str) -> list[str]: ...
    def entails(self, context: str, claim: str) -> bool: ...


class LocalLLMJudge:
    """Pinned local-model judge (on-prem; never a cloud API).

    The judge model is intentionally separate from the generation model so the
    judge is not grading its own homework. Pin model/temp/seed in config.
    """

    def __init__(self, model: str, base_url: str, temperature: float = 0.0,
                 seed: int = 0) -> None:
        from elv.generate.client import OpenAICompatibleGenerator
        self.llm = OpenAICompatibleGenerator(
            model=model, base_url=base_url, temperature=temperature, seed=seed,
            system="あなたは厳密な評価者です。指示された形式のみで答えます。",
            # thinking-capable models (Gemma 4, Qwen3) must have thinking disabled:
            # reasoning tokens consume the max_tokens budget before the answer is
            # emitted, leaving content empty and faithfulness = 0 for all queries.
            disable_thinking=True)

    def extract_claims(self, question: str, answer: str,
                       timeout: int | None = None) -> list[str]:
        # Stop sequences prevent the model from generating additional Q&A examples.
        # max_tokens caps output so long answers don't run forever.
        out = self.llm.generate(
            CLAIM_PROMPT.format(question=question, answer=answer),
            max_tokens=256, stop=["\n\n", "質問:", "回答:"],
            timeout=timeout)
        lines = [ln.strip("・-* 　\t") for ln in out.splitlines() if ln.strip()]
        # Drop any lines that look like continuation prompts the model hallucinated
        return [ln for ln in lines if not ln.startswith(("質問", "回答", "主張"))]

    def entails(self, context: str, claim: str) -> bool:
        # No stop token: with thinking disabled, gemma4/qwen3 emit the answer
        # directly without a trailing newline that stop=["\n"] would cut before.
        out = self.llm.generate(
            ENTAIL_PROMPT.format(context=context, claim=claim),
            max_tokens=16)
        return out.strip().lower().startswith("yes")


class DeterministicTestJudge:
    """Harness self-test ONLY. Deterministic lexical-overlap entailment.

    NOT an evaluator — it cannot judge meaning. It exists so the faithfulness
    pipeline (decompose -> judge -> aggregate -> variance) can be tested offline
    without a model. Any score it produces is a plumbing artifact, not a metric.
    """

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold

    def extract_claims(self, question: str, answer: str) -> list[str]:
        # split on Japanese/ASCII sentence enders; one claim per sentence
        import re
        parts = re.split(r"[。.!?！？\n]", answer)
        return [p.strip() for p in parts if p.strip()]

    def entails(self, context: str, claim: str) -> bool:
        def bigrams(t):
            t = (t or "").strip()
            return {t[i:i+2] for i in range(len(t)-1)} if len(t) >= 2 else set()
        cb, kb = bigrams(context), bigrams(claim)
        if not kb:
            return False
        return len(cb & kb) / len(kb) >= self.threshold


def build_judge(name: str, **kwargs) -> Judge:
    if name in ("test", "deterministic", "offline"):
        return DeterministicTestJudge()
    if name in ("local", "llm", "vllm"):
        return LocalLLMJudge(**kwargs)
    raise ValueError(f"unknown judge: {name}")
