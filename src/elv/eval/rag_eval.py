"""End-to-end RAG evaluation: retrieve -> generate -> judge.

Produces the other half of the eval story (retrieval is in runner.py): is the
generated answer faithful to what was retrieved, and — read alongside the
ground-truth-anchored context recall — is it grounded in the RIGHT documents.

Offline self-test:   --gen template --judge test   (validates plumbing only)
Real run (their box): --gen openai --judge local --base-url ... --model ...
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
from pathlib import Path

from . import metrics_generation as mg
from .judge import build_judge
from .runner import _build_retriever, _load_golden_and_corpus, per_query_p1


def _prewarm(generator, judge, short_timeout: int = 30, retries: int = 10) -> None:
    """Block until both endpoints respond — retry with short timeout so a cold
    model-load after corpus indexing is absorbed rather than silently missed."""
    import time
    for name, fn in [("generator", lambda: generator.generate("準備完了", max_tokens=8, timeout=short_timeout)),
                     ("judge", lambda: judge.extract_claims("テスト", "テスト", timeout=short_timeout))]:
        for attempt in range(retries):
            try:
                fn()
                print(f"  prewarm {name}: OK (attempt {attempt+1})", flush=True)
                break
            except Exception as exc:
                wait = min(30, 5 * (attempt + 1))
                print(f"  prewarm {name}: not ready ({exc.__class__.__name__}), "
                      f"retry {attempt+1}/{retries} in {wait}s…", flush=True)
                time.sleep(wait)


def run_generation_eval(golden, retriever, generator, judge, k=5, runs_per_item=3):
    import datetime as _dt2
    from elv.generate.client import build_rag_prompt

    _prewarm(generator, judge)
    faiths, recalls, flags, per_query = [], [], 0, {}
    n = len(golden.queries)
    for qi, q in enumerate(golden.queries):
        if qi % 10 == 0:
            print(f"  query {qi+1}/{n} ({_dt2.datetime.now():%H:%M:%S})", flush=True)
        ids = retriever.retrieve(q.query, k=k)
        contexts = [retriever._text_by_id.get(i, "") for i in ids]
        # Stop at paragraph boundary so model doesn't invent additional passages/questions.
        answer = generator.generate(
            build_rag_prompt(q.query, contexts),
            max_tokens=256, stop=["\n\n[文脈", "\n\n質問", "\n\n回答"])
        sample = mg.GenerationSample(
            query=q.query, answer=answer, contexts=contexts,
            retrieved_doc_ids=ids, relevant_doc_ids=q.relevant_doc_ids)
        f = mg.faithfulness(sample, judge, runs_per_item=runs_per_item)
        r = mg.context_recall_docs(sample)
        flag = mg.grounded_but_wrong_flag(f.mean, r)
        faiths.append(f.mean)
        recalls.append(r)
        flags += int(flag)
        per_query[q.id] = {"faithfulness": round(f.mean, 4), "faith_spread": round(f.spread, 4),
                           "context_recall_docs": round(r, 4), "grounded_but_wrong": flag}
    return {
        "faithfulness_mean": statistics.fmean(faiths) if faiths else 0.0,
        "faithfulness_max_spread": max((per_query[q]["faith_spread"] for q in per_query), default=0.0),
        "context_recall_docs_mean": statistics.fmean(recalls) if recalls else 0.0,
        "grounded_but_wrong_count": flags,
        "n_queries": len(golden.queries),
    }, per_query


def write_report(out_dir, metadata, agg, per_query):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    worst = sorted(per_query.items(), key=lambda kv: (kv[1]["context_recall_docs"], -kv[1]["faithfulness"]))[:5]
    lines = ["# RAG eval — generation (faithfulness, judge-based)", "",
             "## Run metadata", ""]
    lines += [f"- {k}: {v}" for k, v in metadata.items()]
    lines += ["", "## Results", "",
              "| Metric | Value |", "|---|---|",
              f"| faithfulness (mean) | {agg['faithfulness_mean']:.4f} |",
              f"| faithfulness (max judge spread) | {agg['faithfulness_max_spread']:.4f} |",
              f"| context_recall_docs (mean, deterministic) | {agg['context_recall_docs_mean']:.4f} |",
              f"| grounded-but-wrong queries | {agg['grounded_but_wrong_count']} / {agg['n_queries']} |",
              "",
              "> Read faithfulness AND context_recall together. High faithfulness "
              "with low context recall = confidently grounded in the wrong "
              "documents — the failure a faithfulness-only score hides (ADR-0001).",
              "",
              "## Lowest-context-recall queries (inspect by hand for sec 6)", "",
              "| query id | faithfulness | ctx_recall | grounded_but_wrong |",
              "|---|---|---|---|"]
    for qid, m in worst:
        lines.append(f"| {qid} | {m['faithfulness']:.3f} | {m['context_recall_docs']:.3f} | {m['grounded_but_wrong']} |")
    report = out / "rag_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "rag_results.json").write_text(json.dumps(
        {"metadata": metadata, "aggregate": agg, "per_query": per_query},
        ensure_ascii=False, indent=2), "utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--rerank", default="none")
    ap.add_argument("--gen", default="template", help="template|openai")
    ap.add_argument("--judge", default="test", help="test|local")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--judge-base-url", default=None,
                    help="separate endpoint for the judge; defaults to --base-url")
    ap.add_argument("--model", default="gemma4-27b-q")
    ap.add_argument("--judge-model", default="elyza-jp-8b")  # separate from gen
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="limit to first N queries (for representative sampling runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from elv.generate.client import build_generator

    golden, ids, texts = _load_golden_and_corpus(args.golden, args.corpus)
    retr = _build_retriever(args.embedder, args.rerank, ids, texts)
    gen = build_generator(args.gen, model=args.model, base_url=args.base_url)
    judge_url = args.judge_base_url or args.base_url
    judge = build_judge(args.judge, model=args.judge_model, base_url=judge_url)

    if args.max_queries is not None:
        from dataclasses import replace as dc_replace
        golden = dc_replace(golden, queries=golden.queries[:args.max_queries])
    agg, per_query = run_generation_eval(golden, retr, gen, judge, k=args.k, runs_per_item=args.runs)
    self_test = args.gen in ("template",) or args.judge in ("test",)
    meta = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "golden": f"{golden.name}@{golden.version}",
        "embedder": args.embedder, "reranker": args.rerank,
        "generator": args.gen, "judge": args.judge,
        "gen_model": args.model if args.gen == "openai" else "-",
        "judge_model": args.judge_model if args.judge == "local" else "-",
        "runs_per_item": args.runs,
        "note": ("HARNESS SELF-TEST — NOT an evaluation (template gen / test judge)"
                 if self_test else ""),
    }
    report = write_report(args.out or f"reports/{_dt.datetime.now():%Y%m%dT%H%M%S}-gen",
                          meta, agg, per_query)
    print(f"wrote {report}")
    for k, v in agg.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
