"""Generation client — OpenAI-compatible (vLLM serving, LiteLLM gateway).

On-prem only: base_url points at locally served models (Gemma 4 / Swallow /
ELYZA-JP). No cloud endpoints in the deployment path (ADR-0003). Swapping the
model is a config change, not a code change. Uses the stdlib so no extra
dependency is pulled in for the HTTP call.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol, Sequence


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


RAG_SYSTEM = (
    "あなたは社内文書に基づいて回答するアシスタントです。"
    "与えられた文脈のみを根拠に、日本語で簡潔に回答してください。"
    "文脈に根拠がない場合は『分かりません』と答えてください。"
)


def build_rag_prompt(query: str, contexts: Sequence[str]) -> str:
    ctx = "\n\n".join(f"[文脈{i+1}] {c}" for i, c in enumerate(contexts))
    return f"{ctx}\n\n質問: {query}\n回答:"


class OpenAICompatibleGenerator:
    """Calls a local OpenAI-compatible /v1/chat/completions endpoint.

    temperature/seed are pinned so a run is reproducible from config; the real
    model still carries residual non-determinism, which the eval measures rather
    than hides (see metrics_generation / runs_per_item).
    """

    def __init__(self, model: str, base_url: str, temperature: float = 0.0,
                 seed: int = 0, system: str = RAG_SYSTEM, timeout: int = 120) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.system = system
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class TemplateGenerator:
    """Deterministic generator for harness self-test ONLY (no model).

    Produces a pseudo-answer from the contexts so the end-to-end pipeline can be
    exercised offline. Its output is not an answer anyone should evaluate — it
    exists to validate plumbing.
    """

    def generate(self, prompt: str) -> str:
        body = prompt.split("質問:")[0]
        first = body.split("[文脈1]")[-1].strip().split("\n")[0]
        return first[:120] if first else "分かりません"


def build_generator(name: str, **kwargs) -> Generator:
    if name in ("template", "test", "offline"):
        return TemplateGenerator()
    if name in ("openai", "vllm", "litellm", "local"):
        return OpenAICompatibleGenerator(**kwargs)
    raise ValueError(f"unknown generator: {name}")
