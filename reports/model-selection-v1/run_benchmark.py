"""
Model-selection benchmark — on-prem deployment candidates.

Two-phase design (required when gen + judge don't fit together in 32 GB VRAM):
  Phase 1: per-model generation → <results-dir>/<model>.json  (resumable)
  Phase 2: qwen3:32b judges all results in one pass
  Phase 3: gemma4:31b cross-judges a subset → agreement rate

Dimensions measured:
  - Quality:  faithfulness (judge-based) + hit_rate (judge-based answer accuracy)
  - Latency:  tokens/s cold-start vs warm (from Ollama eval_duration)
  - Resource: VRAM peak (nvidia-smi) + model size/quantization (ollama show)

Run v1:  python run_benchmark.py [--phase 1|2|3|all] [--max-queries N]
Run v2:  python run_benchmark.py --golden golden_qa_v2.jsonl --results-dir results_v2 --summary summary_v2.md

Results land in --results-dir — already-completed models are skipped (resume-safe).
Summary lands in --summary.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
# Defaults (v1); overridden by CLI flags for v2
RESULTS_DIR = SCRIPT_DIR / "results"
GOLDEN_FILE = SCRIPT_DIR / "golden_qa.jsonl"
SUMMARY_FILE = SCRIPT_DIR / "summary.md"

BASE_URL = "http://localhost:11434"
OLLAMA_API = f"{BASE_URL}/api"

# Judge (not a competitor — see DECISIONS.md ADR for why qwen3 judges)
JUDGE_MODEL = "qwen3:32b"
CROSS_JUDGE_MODEL = "gemma4:31b"

# Competitors — confirmed available in Ollama registry 2026-06-11
# Tags confirmed with: curl -sI https://registry.ollama.ai/v2/<ns>/<model>/manifests/latest
MODELS = [
    {
        "tag": "gemma4:31b",
        "label": "gemma4-31b",
        "description": "Gemma 4 31B Q4_K_M — primary deployment candidate",
        "is_thinking": True,  # needs think:false when used as judge
    },
    {
        "tag": "dsasai/llama3-elyza-jp-8b:latest",
        "label": "elyza-jp-8b",
        "description": "Llama-3-ELYZA-JP-8B — Japanese fine-tune of Meta Llama 3 8B",
        "is_thinking": False,
    },
    {
        "tag": "schroneko/llama-3.1-swallow-8b-instruct-v0.1:latest",
        "label": "swallow-8b",
        "description": "Llama-3.1-Swallow-8B-Instruct-v0.1 — Tokyo Tech Japanese continued pretraining",
        "is_thinking": False,
    },
    {
        "tag": "fuukeidaisuki/nvidia-nemotron-nano-9b-v2-japanese:latest",
        "label": "nemotron-nano-9b-jp",
        "description": "NVIDIA Nemotron Nano 9B v2 Japanese — community Ollama port (nemotron_h arch, Q4_K_M)",
        "is_thinking": True,  # nemotron_h supports thinking; disable for fair comparison
    },
]

CROSS_JUDGE_SUBSET_SIZE = 25


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_API}/{endpoint}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def model_available(tag: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "show", tag], capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def pull_model(tag: str) -> bool:
    print(f"  pulling {tag} …", flush=True)
    result = subprocess.run(["ollama", "pull", tag], timeout=1800)
    return result.returncode == 0


def unload_model(tag: str) -> None:
    try:
        _post("generate", {"model": tag, "keep_alive": 0}, timeout=30)
        print(f"  unloaded {tag}", flush=True)
    except Exception:
        pass


def model_info(tag: str) -> dict:
    result = subprocess.run(
        ["ollama", "show", tag], capture_output=True, text=True, timeout=30)
    lines = result.stdout.splitlines()
    info = {}
    # Parse only the model card section (before "Capabilities"); stop at blank line after model card.
    in_model = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "model":
            in_model = True
            continue
        if in_model and stripped.lower() in ("capabilities", "parameters", "license", "system", "metadata"):
            break
        if not in_model:
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            key = parts[0].lower()
            val = parts[-1]
            if key == "parameters":
                info["parameters"] = val
            elif key == "quantization":
                info["quantization"] = val
            elif stripped.lower().startswith("context length"):
                info["context_length"] = val
    return info


# ── VRAM measurement ──────────────────────────────────────────────────────────

def vram_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip()
        return int(out.split("\n")[0].strip())
    except Exception:
        return -1


# ── Generation (native Ollama API for timing) ─────────────────────────────────

RAG_SYSTEM = (
    "あなたは与えられた文脈に基づいて回答するアシスタントです。"
    "文脈のみを根拠に、日本語で簡潔に回答してください。"
    "文脈に根拠がない場合は「分かりません」と答えてください。"
)


def _build_prompt(query: str, context: str) -> str:
    return f"文脈:\n{context}\n\n質問: {query}\n回答:"


def generate_with_timing(tag: str, query: str, context: str,
                          is_thinking: bool = False) -> dict:
    """Return dict with answer, eval_count, eval_duration_ns, tokens_per_second."""
    prompt = _build_prompt(query, context)
    payload = {
        "model": tag,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 256,
        },
        "messages": [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    if is_thinking:
        payload["think"] = False

    # Use native /api/chat for timing metadata
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    wall_s = time.monotonic() - t0

    answer = raw.get("message", {}).get("content", "")
    eval_count = raw.get("eval_count", 0)
    eval_duration_ns = raw.get("eval_duration", 0)
    prompt_eval_count = raw.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = raw.get("prompt_eval_duration", 0)

    tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0
    return {
        "answer": answer,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "tokens_per_second": round(tps, 2),
        "wall_seconds": round(wall_s, 2),
    }


# ── Phase 1: generate all answers ────────────────────────────────────────────

def load_golden() -> list[dict]:
    items = []
    with open(GOLDEN_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def phase1_generate(models: list[dict], golden: list[dict], args) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    n = len(golden) if args.max_queries is None else min(args.max_queries, len(golden))
    subset = golden[:n]

    for model in models:
        tag = model["tag"]
        label = model["label"]
        out_file = RESULTS_DIR / f"{label}.json"

        if out_file.exists():
            print(f"[phase1] {label}: already done → skipping", flush=True)
            continue

        print(f"\n[phase1] {label} ({tag})", flush=True)

        # Ensure model is available
        if not model_available(tag):
            print(f"  pulling …", flush=True)
            ok = pull_model(tag)
            if not ok:
                print(f"  FAILED to pull {tag} — skipping", flush=True)
                (RESULTS_DIR / f"{label}.SKIPPED").write_text(
                    json.dumps({"reason": "pull failed", "tag": tag}))
                continue

        # Measure VRAM before load
        vram_before = vram_used_mib()
        time.sleep(2)

        # Cold-start: first request (model loads here)
        print(f"  cold-start timing …", flush=True)
        t_cold_start = time.monotonic()
        cold = generate_with_timing(tag, subset[0]["query"], subset[0]["context"],
                                     model.get("is_thinking", False))
        cold_wall = time.monotonic() - t_cold_start

        vram_after = vram_used_mib()
        vram_delta = vram_after - vram_before

        # Warm: subsequent requests for timing (use item[1] to avoid cache bias)
        warm_results = []
        for item in subset[1:4]:
            warm_results.append(generate_with_timing(
                tag, item["query"], item["context"], model.get("is_thinking", False)))
        warm_tps = statistics.fmean([r["tokens_per_second"] for r in warm_results if r["tokens_per_second"] > 0]) if warm_results else 0.0

        # Generate all answers
        print(f"  generating {n} answers …", flush=True)
        answers = []
        for i, item in enumerate(subset):
            if i % 5 == 0:
                print(f"    {i+1}/{n}", flush=True)
            result = generate_with_timing(
                tag, item["query"], item["context"], model.get("is_thinking", False))
            answers.append({
                "id": item["id"],
                "query": item["query"],
                "context": item["context"],
                "reference_answer": item["reference_answer"],
                "answer": result["answer"],
                "tokens_per_second": result["tokens_per_second"],
                "eval_count": result["eval_count"],
            })

        # Model info
        minfo = model_info(tag)

        result_doc = {
            "model": tag,
            "label": label,
            "description": model["description"],
            "model_info": minfo,
            "latency": {
                "cold_start_wall_s": round(cold_wall, 2),
                "cold_tokens_per_second": cold["tokens_per_second"],
                "warm_tokens_per_second": round(warm_tps, 2),
            },
            "vram": {
                "before_mib": vram_before,
                "after_mib": vram_after,
                "delta_mib": vram_delta,
            },
            "answers": answers,
        }
        out_file.write_text(json.dumps(result_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved → {out_file.name}", flush=True)

        # Unload before next model
        unload_model(tag)
        time.sleep(3)


# ── Judge helpers ─────────────────────────────────────────────────────────────

CLAIM_PROMPT = (
    "次の質問と回答を読み、回答に含まれる事実主張を検証可能な最小単位で"
    "1行に1つずつ列挙してください。前置きや番号は不要です。\n\n"
    "質問: {question}\n回答: {answer}\n\n主張:"
)

ENTAIL_PROMPT = (
    "文脈だけを根拠に、次の主張が支持されるか判定してください。"
    "文脈に根拠があれば Yes、なければ No と1語だけで答えてください。\n\n"
    "文脈:\n{context}\n\n主張: {claim}\n\n判定:"
)

HIT_PROMPT = (
    "参照回答と生成回答を比較し、生成回答が質問に対して正しい情報を含んでいるか判定してください。"
    "正しければ Yes、不正解または情報が欠けていれば No と1語だけで答えてください。\n\n"
    "質問: {question}\n参照回答: {reference}\n生成回答: {answer}\n\n判定:"
)

HIT_PROMPT_WITH_POINTS = (
    "参照回答と採点ポイントを参照し、生成回答がすべての採点ポイントを満たしているか判定してください。"
    "すべてのポイントを満たしていれば Yes、1つでも欠けていれば No と1語だけで答えてください。\n\n"
    "質問: {question}\n参照回答: {reference}\n"
    "採点ポイント（以下をすべて含むこと）: {points}\n"
    "生成回答: {answer}\n\n判定:"
)


def _llm_generate(model: str, prompt: str, system: str | None = None,
                   max_tokens: int = 32, is_thinking: bool = False) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload: dict = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 0, "num_predict": max_tokens},
        "messages": msgs,
    }
    if is_thinking:
        payload["think"] = False
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return raw.get("message", {}).get("content", "").strip()


def _judge_item(judge_tag: str, item: dict, is_thinking: bool) -> dict:
    """Compute faithfulness + hit_rate for one answer item."""
    judge_system = "あなたは厳密な評価者です。指示された形式のみで答えます。"

    # Skip empty answers
    if not item["answer"] or item["answer"].strip() == "分かりません":
        return {"faithfulness": 0.0, "hit_rate": 0.0, "n_claims": 0,
                "supported": 0, "hit": False}

    # Extract claims
    claim_out = _llm_generate(
        judge_tag,
        CLAIM_PROMPT.format(question=item["query"], answer=item["answer"]),
        system=judge_system,
        max_tokens=256, is_thinking=is_thinking)
    lines = [ln.strip("・-* 　\t") for ln in claim_out.splitlines() if ln.strip()]
    claims = [ln for ln in lines if not ln.startswith(("質問", "回答", "主張"))]

    if not claims:
        faith = 0.0
        n_claims = 0
        supported = 0
    else:
        # Entailment per claim
        verdicts = []
        for claim in claims:
            out = _llm_generate(
                judge_tag,
                ENTAIL_PROMPT.format(context=item["context"], claim=claim),
                system=judge_system,
                max_tokens=8, is_thinking=is_thinking)
            verdicts.append(out.lower().startswith("yes"))
        supported = sum(verdicts)
        faith = supported / len(claims)
        n_claims = len(claims)

    # Hit rate — use expected_points prompt if available (v2), else basic match (v1)
    expected_points = item.get("expected_points")
    if expected_points:
        points_str = "、".join(expected_points)
        hit_prompt = HIT_PROMPT_WITH_POINTS.format(
            question=item["query"],
            reference=item["reference_answer"],
            points=points_str,
            answer=item["answer"])
    else:
        hit_prompt = HIT_PROMPT.format(
            question=item["query"],
            reference=item["reference_answer"],
            answer=item["answer"])
    hit_out = _llm_generate(
        judge_tag, hit_prompt, system=judge_system,
        max_tokens=8, is_thinking=is_thinking)
    hit = hit_out.lower().startswith("yes")

    return {
        "faithfulness": round(faith, 4),
        "hit_rate": 1.0 if hit else 0.0,
        "n_claims": n_claims,
        "supported": supported,
        "hit": hit,
    }


# ── Phase 2: judge with qwen3:32b ─────────────────────────────────────────────

def phase2_judge(args) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    gen_files = sorted(RESULTS_DIR.glob("*.json"))
    if not gen_files:
        print("[phase2] No generation results found in results/. Run phase 1 first.")
        return

    print(f"\n[phase2] judging with {JUDGE_MODEL}", flush=True)
    judge_is_thinking = True  # qwen3 is a thinking model

    for gen_file in gen_files:
        if gen_file.stem.endswith("_judged"):
            continue
        judged_file = RESULTS_DIR / f"{gen_file.stem}_judged.json"
        if judged_file.exists():
            print(f"  {gen_file.stem}: already judged → skipping", flush=True)
            continue

        doc = json.loads(gen_file.read_text(encoding="utf-8"))
        answers = doc.get("answers", [])
        if not answers:
            print(f"  {gen_file.stem}: no answers → skipping", flush=True)
            continue

        print(f"  judging {gen_file.stem} ({len(answers)} items) …", flush=True)
        judged_answers = []
        for i, item in enumerate(answers):
            if i % 5 == 0:
                print(f"    {i+1}/{len(answers)}", flush=True)
            scores = _judge_item(JUDGE_MODEL, item, judge_is_thinking)
            judged_answers.append({**item, **scores, "judge": JUDGE_MODEL})

        # Aggregate
        faiths = [a["faithfulness"] for a in judged_answers]
        hits = [a["hit_rate"] for a in judged_answers]
        agg = {
            "faithfulness_mean": round(statistics.fmean(faiths), 4) if faiths else 0.0,
            "hit_rate_mean": round(statistics.fmean(hits), 4) if hits else 0.0,
            "n_items": len(judged_answers),
            "judge": JUDGE_MODEL,
        }
        judged_doc = {**doc, "aggregate_quality": agg, "answers": judged_answers}
        judged_file.write_text(
            json.dumps(judged_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved → {judged_file.name}", flush=True)

    # Unload judge
    unload_model(JUDGE_MODEL)


# ── Phase 3: cross-validation with gemma4:31b ────────────────────────────────

def phase3_cross_validate(args) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    cross_file = RESULTS_DIR / "cross_validation.json"
    if cross_file.exists():
        print(f"[phase3] already done → skipping", flush=True)
        return

    # Pick any one model's judged file (preferably not gemma4 judging itself)
    judged_files = sorted(RESULTS_DIR.glob("*_judged.json"))
    # Exclude gemma4's own results — gemma4 cannot cross-judge itself
    ref_files = [f for f in judged_files if "gemma4" not in f.stem]
    if not ref_files:
        # Fall back to any non-gemma4 file
        ref_files = judged_files
    if not ref_files:
        print("[phase3] No judged results found. Run phase 2 first.")
        return

    ref_file = ref_files[0]
    ref_doc = json.loads(ref_file.read_text(encoding="utf-8"))
    all_answers = ref_doc.get("answers", [])

    # Select subset for cross-validation
    rng = random.Random(42)
    subset_size = min(CROSS_JUDGE_SUBSET_SIZE, len(all_answers))
    subset = rng.sample(all_answers, subset_size)

    print(f"\n[phase3] cross-validating {subset_size} items from {ref_file.stem}", flush=True)
    print(f"  primary judge: {JUDGE_MODEL}", flush=True)
    print(f"  cross judge:   {CROSS_JUDGE_MODEL}", flush=True)
    print(f"  NOTE: {CROSS_JUDGE_MODEL} cross-judges to validate {JUDGE_MODEL} reliability only.", flush=True)
    print(f"  {CROSS_JUDGE_MODEL}'s own main scores are judged by {JUDGE_MODEL} (no self-scoring).", flush=True)

    cross_is_thinking = True  # gemma4 is a thinking model
    cross_results = []
    for i, item in enumerate(subset):
        if i % 5 == 0:
            print(f"    {i+1}/{subset_size}", flush=True)
        primary_faith = item.get("faithfulness", None)
        primary_hit = item.get("hit_rate", None)
        cross_scores = _judge_item(CROSS_JUDGE_MODEL, item, cross_is_thinking)
        cross_results.append({
            "id": item["id"],
            "primary_judge": JUDGE_MODEL,
            "cross_judge": CROSS_JUDGE_MODEL,
            "primary_faithfulness": primary_faith,
            "cross_faithfulness": cross_scores["faithfulness"],
            "primary_hit": primary_hit,
            "cross_hit": cross_scores["hit_rate"],
            "faith_agree": abs((primary_faith or 0) - cross_scores["faithfulness"]) < 0.2,
            "hit_agree": primary_hit == cross_scores["hit_rate"],
        })

    # Agreement statistics
    faith_agree_rate = statistics.fmean([r["faith_agree"] for r in cross_results])
    hit_agree_rate = statistics.fmean([r["hit_agree"] for r in cross_results])

    # Cohen's kappa for hit (binary)
    p_primary_yes = statistics.fmean([r["primary_hit"] or 0 for r in cross_results])
    p_cross_yes = statistics.fmean([r["cross_hit"] or 0 for r in cross_results])
    p_agree_expected = (p_primary_yes * p_cross_yes +
                        (1 - p_primary_yes) * (1 - p_cross_yes))
    kappa = ((hit_agree_rate - p_agree_expected) / (1 - p_agree_expected)
             if p_agree_expected < 1.0 else 1.0)

    # Disagreement cases
    disagreements = [r for r in cross_results if not r["hit_agree"]]

    summary = {
        "source_model_file": ref_file.stem,
        "primary_judge": JUDGE_MODEL,
        "cross_judge": CROSS_JUDGE_MODEL,
        "n_items": subset_size,
        "faithfulness_agreement_rate": round(faith_agree_rate, 4),
        "hit_agreement_rate": round(hit_agree_rate, 4),
        "cohen_kappa_hit": round(kappa, 4),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements[:10],
        "items": cross_results,
    }
    cross_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved → {cross_file.name}", flush=True)
    print(f"  hit agreement rate: {hit_agree_rate:.2%}, Cohen's κ={kappa:.3f}", flush=True)

    # Unload cross-judge
    unload_model(CROSS_JUDGE_MODEL)


# ── Discriminability analysis ─────────────────────────────────────────────────

def compute_discriminability(results_dir: Path) -> dict:
    """Per-question: how many models answered correctly. Returns analysis dict."""
    judged_files = sorted(results_dir.glob("*_judged.json"))
    per_q: dict[str, dict] = {}
    model_labels = []
    for jf in judged_files:
        if jf.stem.startswith("cross"):
            continue
        doc = json.loads(jf.read_text("utf-8"))
        label = doc["label"]
        model_labels.append(label)
        for a in doc.get("answers", []):
            qid = a["id"]
            if qid not in per_q:
                per_q[qid] = {"query": a["query"], "difficulty": a.get("difficulty", "?"), "hits": {}}
            per_q[qid]["hits"][label] = a.get("hit_rate", 0.0)

    n_models = len(model_labels)
    all_correct, all_wrong, partial = [], [], []
    for qid, data in per_q.items():
        n_correct = sum(1 for h in data["hits"].values() if h >= 1.0)
        data["n_correct"] = n_correct
        if n_correct == n_models:
            all_correct.append(qid)
        elif n_correct == 0:
            all_wrong.append(qid)
        else:
            partial.append(qid)

    return {
        "n_models": n_models,
        "model_labels": model_labels,
        "n_questions": len(per_q),
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "partial": partial,
        "per_q": per_q,
    }


# ── Summary generation ────────────────────────────────────────────────────────

def generate_summary(title: str = "v1", golden_file: Path | None = None) -> None:
    judged_files = sorted(RESULTS_DIR.glob("*_judged.json"))
    cross_file = RESULTS_DIR / "cross_validation.json"

    rows = []
    for jf in judged_files:
        if jf.stem.startswith("cross"):
            continue
        doc = json.loads(jf.read_text(encoding="utf-8"))
        agg_q = doc.get("aggregate_quality", {})
        lat = doc.get("latency", {})
        vram = doc.get("vram", {})
        minfo = doc.get("model_info", {})
        rows.append({
            "label": doc.get("label", jf.stem),
            "params": minfo.get("parameters", "—"),
            "quant": minfo.get("quantization", "—"),
            "faithfulness": agg_q.get("faithfulness_mean", "—"),
            "hit_rate": agg_q.get("hit_rate_mean", "—"),
            "n_items": agg_q.get("n_items", "—"),
            "judge": agg_q.get("judge", "—"),
            "cold_tps": lat.get("cold_tokens_per_second", "—"),
            "warm_tps": lat.get("warm_tokens_per_second", "—"),
            "cold_wall_s": lat.get("cold_start_wall_s", "—"),
            "vram_delta_mib": vram.get("delta_mib", "—"),
            "vram_after_mib": vram.get("after_mib", "—"),
        })

    cross = {}
    if cross_file.exists():
        cross = json.loads(cross_file.read_text(encoding="utf-8"))

    skipped = []
    for sf in RESULTS_DIR.glob("*.SKIPPED"):
        info = json.loads(sf.read_text())
        skipped.append({"label": sf.stem, **info})

    gf = golden_file or GOLDEN_FILE
    n_golden = sum(1 for _ in gf.open(encoding="utf-8"))

    disc = compute_discriminability(RESULTS_DIR)

    lines = [
        f"# Model Selection Benchmark {title} — Summary",
        "",
        "> All numbers from real runs on RTX 5090 32 GB. No placeholders.",
        f"> Judge: **{JUDGE_MODEL}** (not a competitor — see DECISIONS.md ADR for rationale).",
        f"> Cross-validation judge: **{CROSS_JUDGE_MODEL}** on {cross.get('n_items', '—')} item subset.",
        "",
        "## Quality / Latency / Resource (three-dimension table)",
        "",
        "| Model | Params | Quant | Faithfulness | Hit Rate | n | Judge | Cold TPS | Warm TPS | Cold Wall(s) | VRAM Δ(MiB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        faith = f"{r['faithfulness']:.4f}" if isinstance(r['faithfulness'], float) else r['faithfulness']
        hit = f"{r['hit_rate']:.4f}" if isinstance(r['hit_rate'], float) else r['hit_rate']
        lines.append(
            f"| {r['label']} | {r['params']} | {r['quant']} | {faith} | {hit} "
            f"| {r['n_items']} | {r['judge']} | {r['cold_tps']} | {r['warm_tps']} "
            f"| {r['cold_wall_s']} | {r['vram_delta_mib']} |"
        )

    lines += [
        "",
        f"> faithfulness = fraction of answer claims entailed by provided context (judge-based, 1 run).",
        f"> hit_rate = fraction of items where answer matches reference answer (judge-based).",
        "",
        "## Judge Agreement (cross-validation)",
        "",
        f"- Primary judge: **{cross.get('primary_judge', JUDGE_MODEL)}**",
        f"- Cross judge:   **{cross.get('cross_judge', CROSS_JUDGE_MODEL)}** (also a competitor; cross-judges **only to validate primary judge reliability**, not to score itself)",
        f"- Subset size:   {cross.get('n_items', '—')} items (random seed 42)",
        f"- Hit agreement rate: **{cross.get('hit_agreement_rate', '—')}**",
        f"- Cohen's κ (hit): **{cross.get('cohen_kappa_hit', '—')}**",
        f"- Faithfulness agreement rate (|Δ|<0.2): **{cross.get('faithfulness_agreement_rate', '—')}**",
        f"- Disagreements: {cross.get('n_disagreements', '—')} items",
        "",
    ]

    if cross.get("disagreements"):
        lines += ["### Disagreement cases (primary_hit ≠ cross_hit)", "",
                  "| id | primary_hit | cross_hit | primary_faith | cross_faith |",
                  "|---|---|---|---|---|"]
        for d in cross["disagreements"]:
            pf = f"{d['primary_faithfulness']:.3f}" if isinstance(d.get('primary_faithfulness'), float) else "—"
            cf = f"{d['cross_faithfulness']:.3f}" if isinstance(d.get('cross_faithfulness'), float) else "—"
            lines.append(f"| {d['id']} | {d['primary_hit']} | {d['cross_hit']} | {pf} | {cf} |")
        lines.append("")

    # Decision table — derive best-per-category from measured rows
    def _pick(rows_list, key, higher_is_better=True):
        valid = [(r['label'], r[key]) for r in rows_list if isinstance(r[key], float)]
        if not valid:
            return "—"
        return max(valid, key=lambda x: x[1])[0] if higher_is_better else min(valid, key=lambda x: x[1])[0]

    small_vram = [r for r in rows if isinstance(r['vram_delta_mib'], int) and r['vram_delta_mib'] <= 10240]
    best_small_quality = _pick(small_vram, 'hit_rate')
    best_small_speed = _pick(small_vram, 'warm_tps')
    best_overall_quality = _pick(rows, 'hit_rate')
    best_cold = _pick(rows, 'cold_wall_s', higher_is_better=False)

    lines += [
        "## Constraint → Recommended model (decision table)",
        "",
        "| Constraint | Recommended | Rationale |",
        "|---|---|---|",
        f"| VRAM ≤ 10 GB + best Japanese quality | **{best_small_quality or 'elyza-jp-8b / swallow-8b'}** | 8B models fit 5–10 GB; winner by hit_rate |",
        f"| VRAM ≤ 10 GB + fastest tokens/s | **{best_small_speed or 'elyza-jp-8b'}** | highest warm TPS among ≤10 GB models |",
        f"| VRAM ≤ 20 GB + highest quality | **{best_overall_quality or 'gemma4-31b'}** | highest hit_rate across all models; 31B needs ~20 GB |",
        f"| Fastest cold start (lowest TTFT) | **{best_cold or 'elyza-jp-8b'}** | smallest cold_start_wall_s |",
        "| Multilingual (JP + EN) required | **gemma4-31b** | natively multilingual; 8B JP fine-tunes may degrade EN |",
        "| Apache 2.0 license required | **gemma4-31b** | Apache 2.0; verify 8B model licenses before production |",
        "",
        "## Protocol integrity notes",
        "",
        f"- **{JUDGE_MODEL}** judges all competitors. It is NOT a competitor (deployment/content layer separation — see DECISIONS.md).",
        f"- **{CROSS_JUDGE_MODEL}** cross-judges a {cross.get('n_items', CROSS_JUDGE_SUBSET_SIZE)}-item subset to validate primary judge reliability. Its own main scores are judged by {JUDGE_MODEL}, never itself.",
        "- All latency and VRAM numbers from live Ollama runs on RTX 5090 32 GB.",
        "- Golden set: 20 context-grounded QA items, neutral tech/Japanese knowledge, no customer data.",
        f"- Golden set source: {GOLDEN_FILE.name} (committed to repo).",
        "",
    ]

    # Discriminability analysis section
    disc = compute_discriminability(RESULTS_DIR)
    n_q = disc["n_questions"]
    lines += ["## Discriminability Analysis", ""]
    if n_q == 0:
        lines += [f"Golden set: **{n_golden} items** (from `{gf.name}`)",
                  "", "> No judged results yet — run phases 2 and 3 first.", ""]
    else:
        n_all_correct = len(disc["all_correct"])
        n_partial = len(disc["partial"])
        n_all_wrong = len(disc["all_wrong"])
        lines += [
            f"Golden set: **{n_q} items** (from `{gf.name}`)",
            "",
            f"| Category | Count | % |",
            "|---|---|---|",
            f"| All models correct (zero discrimination) | {n_all_correct} | {n_all_correct/n_q*100:.0f}% |",
            f"| Partial discrimination (some models wrong) | {n_partial} | {n_partial/n_q*100:.0f}% |",
            f"| All models wrong | {n_all_wrong} | {n_all_wrong/n_q*100:.0f}% |",
            "",
        ]
        if disc["partial"]:
            lines += ["### Discriminating questions (partial correct)", "",
                      f"| ID | Difficulty | #Correct/{disc['n_models']} | Query (preview) |",
                      "|---|---|---|---|"]
            for qid in sorted(disc["partial"]):
                d = disc["per_q"][qid]
                lines.append(f"| {qid} | {d['difficulty']} | {d['n_correct']}/{disc['n_models']} | {d['query'][:55]} |")
            lines.append("")

    if skipped:
        lines += ["## Excluded models", ""]
        for s in skipped:
            lines.append(f"- **{s['label']}**: {s.get('reason', 'unknown')} (tag: {s.get('tag', '—')})")
        lines.append("")

    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {SUMMARY_FILE}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global RESULTS_DIR, GOLDEN_FILE, SUMMARY_FILE

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["1", "2", "3", "all"],
                    help="which phase(s) to run")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="limit to first N golden items (for quick smoke-tests)")
    ap.add_argument("--golden", default=None,
                    help="path to golden QA jsonl (default: golden_qa.jsonl)")
    ap.add_argument("--results-dir", default=None,
                    help="output directory for per-model JSONs (default: results/)")
    ap.add_argument("--summary", default=None,
                    help="path to write summary markdown (default: summary.md)")
    ap.add_argument("--title", default=None,
                    help="benchmark title label (e.g. 'v2') for summary heading")
    args = ap.parse_args()

    # Override module-level paths from CLI
    if args.golden:
        GOLDEN_FILE = SCRIPT_DIR / args.golden
    if args.results_dir:
        RESULTS_DIR = SCRIPT_DIR / args.results_dir
    if args.summary:
        SUMMARY_FILE = SCRIPT_DIR / args.summary

    title = args.title or GOLDEN_FILE.stem.replace("golden_qa", "").strip("_-") or "v1"
    if not title:
        title = "v1"

    golden = load_golden()
    print(f"golden set: {len(golden)} items ({GOLDEN_FILE})", flush=True)

    if args.phase in ("1", "all"):
        phase1_generate(MODELS, golden, args)

    if args.phase in ("2", "all"):
        phase2_judge(args)

    if args.phase in ("3", "all"):
        phase3_cross_validate(args)

    generate_summary(title=title, golden_file=GOLDEN_FILE)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
