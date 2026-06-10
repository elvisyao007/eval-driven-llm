"""Eval runner — reproducible end-to-end evaluation, emits a report.

Pipeline (reproducible from the Makefile):
  load frozen golden -> load corpus -> embed+index -> retrieve -> score -> report

Modes:
  single   score one configuration (optionally with a reranker)
  compare  score dense-only vs dense+rerank and emit a side-by-side report with
           per-query P@1 flips (which queries the reranker fixed / broke)

Deterministic retrieval metrics run today. Generation metrics activate once the
pinned judge is wired (metrics_generation.py). No fabricated numbers are ever
written; unwired metrics are reported as "run pending".
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from . import metrics_retrieval
from .golden import GoldenSet, load, verify_corpus
from .ids import content_hash


def run_retrieval_eval(golden: GoldenSet, retrieve_fn, ks=(5, 10)) -> dict[str, float]:
    """Score a retrieval function against a frozen golden set. Deterministic."""
    runs = [(retrieve_fn(q.query), q.relevant_doc_ids) for q in golden.queries]
    return metrics_retrieval.aggregate(runs, ks=ks)


def per_query_p1(golden: GoldenSet, retrieve_fn) -> dict[str, int]:
    """P@1 per query id (1 if top-ranked doc is relevant)."""
    out = {}
    for q in golden.queries:
        ranked = retrieve_fn(q.query)
        out[q.id] = int(bool(ranked) and ranked[0] in q.relevant_doc_ids)
    return out


def _load_corpus(path):
    ids, texts, hashes = [], [], {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ids.append(o["doc_id"])
            texts.append((o.get("title", "") + "\n" + o.get("text", "")).strip())
            hashes[o["doc_id"]] = content_hash(o.get("title", ""), o.get("text", ""))
    return ids, texts, hashes


def _build_retriever(embedder_name, reranker_name, ids, texts):
    from elv.embed.embedder import build_embedder
    from elv.index.faiss_store import FaissStore
    from elv.rerank.reranker import build_reranker
    from elv.retrieve.pipeline import Retriever

    embedder = build_embedder(embedder_name)
    dim = embedder.encode_docs(["x"]).shape[1]
    reranker = build_reranker(reranker_name)
    retr = Retriever(embedder, FaissStore(dim=dim), reranker=reranker)
    retr.index_corpus(ids, texts)
    return retr


def _load_golden_and_corpus(golden_path, corpus_override=None):
    golden = load(golden_path)
    manifest = json.loads((Path(golden_path) / "manifest.json").read_text("utf-8"))
    corpus_path = corpus_override or manifest["corpus_path"]
    ids, texts, actual = _load_corpus(corpus_path)
    mism = verify_corpus(golden, actual)
    if mism:
        raise SystemExit(
            f"corpus/label mismatch on {len(mism)} docs — metrics not comparable")
    return golden, ids, texts


def write_report(out_dir, metadata, results):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Eval report (auto-stub)", "", "## Run metadata", ""]
    lines += [f"- {k}: {v}" for k, v in metadata.items()]
    lines += ["", "## Results — retrieval (deterministic)", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in sorted(results.items())]
    lines += [
        "", "## Results — generation (judge-based)", "",
        "> run pending: pinned judge not wired in this run.", "",
        "## Business translation (sec 5) / interpretation (sec 6)", "",
        "> author by hand — judgment is not auto-generated.",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_comparison_report(out_dir, metadata, baseline, tuned, p1_base, p1_tuned):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixed = sorted(q for q in p1_base if p1_base[q] == 0 and p1_tuned[q] == 1)
    broke = sorted(q for q in p1_base if p1_base[q] == 1 and p1_tuned[q] == 0)
    keys = sorted(set(baseline) | set(tuned))

    lines = ["# Eval comparison — dense-only vs dense+rerank", "",
             "## Run metadata", ""]
    lines += [f"- {k}: {v}" for k, v in metadata.items()]
    lines += ["", "## Results — retrieval (deterministic)", "",
              "| Metric | dense-only (baseline) | dense+rerank (tuned) | delta |",
              "|---|---|---|---|"]
    for k in keys:
        b, t = baseline.get(k, 0.0), tuned.get(k, 0.0)
        lines.append(f"| {k} | {b:.4f} | {t:.4f} | {t - b:+.4f} |")
    lines += ["", "## Per-query P@1 flips (the interpretation substance)", "",
              f"- fixed by rerank (0->1): {len(fixed)} {fixed}",
              f"- broken by rerank (1->0): {len(broke)} {broke}",
              f"- unchanged: {len(p1_base) - len(fixed) - len(broke)}", "",
              "> Net P@1 change is the headline; the fixed/broke lists are where "
              "the real story is — inspect those queries by hand for sec 6.", "",
              "## Generation (judge-based)", "",
              "> run pending: pinned judge not wired in this run."]
    report = out / "comparison.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(
        {"metadata": metadata, "baseline": baseline, "tuned": tuned,
         "fixed": fixed, "broke": broke}, ensure_ascii=False, indent=2), "utf-8")
    return report


def _run_sanity_gate(golden: GoldenSet, out_dir: str) -> None:
    """Write sanity.txt to out_dir; uses eval-sanity if installed, else skips."""
    out = Path(out_dir) / "sanity.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from eval_sanity import sanity_report  # type: ignore[import]
        sanity_report(golden, out=str(out))
    except ImportError:
        out.write_text("eval-sanity not installed — gate skipped\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="data/golden/<name>/<version>")
    ap.add_argument("--corpus", default=None, help="override corpus path")
    ap.add_argument("--embedder", default="hashing", help="hashing|ruri")
    ap.add_argument("--rerank", default="none",
                    help="none|lexical|cross-encoder (cross-encoder needs weights)")
    ap.add_argument("--compare", action="store_true",
                    help="run dense-only vs dense+rerank and emit a comparison")
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    golden, ids, texts = _load_golden_and_corpus(args.golden, args.corpus)
    ts = args.out or f"reports/{_dt.datetime.now():%Y%m%dT%H%M%S}"
    baseline_note = ("lexical/baseline plumbing run — NOT a model benchmark"
                     if args.embedder in ("hashing", "lexical", "dummy") else "")

    _run_sanity_gate(golden, ts)

    if args.compare:
        rr = args.rerank if args.rerank != "none" else "lexical"
        base = _build_retriever(args.embedder, "none", ids, texts)
        tune = _build_retriever(args.embedder, rr, ids, texts)
        base_fn = lambda q: base.retrieve(q, k=max(ks))   # noqa: E731
        tune_fn = lambda q: tune.retrieve(q, k=max(ks))   # noqa: E731
        meta = {
            "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "golden": f"{golden.name}@{golden.version}",
            "n_queries": len(golden.queries), "n_passages": len(ids),
            "embedder": args.embedder, "baseline_reranker": "none",
            "tuned_reranker": rr, "note": baseline_note,
        }
        report = write_comparison_report(
            ts, meta,
            run_retrieval_eval(golden, base_fn, ks),
            run_retrieval_eval(golden, tune_fn, ks),
            per_query_p1(golden, base_fn),
            per_query_p1(golden, tune_fn),
        )
        print(f"wrote {report}")
        return

    retr = _build_retriever(args.embedder, args.rerank, ids, texts)
    results = run_retrieval_eval(golden, lambda q: retr.retrieve(q, k=max(ks)), ks=ks)
    meta = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "golden": f"{golden.name}@{golden.version}",
        "n_queries": len(golden.queries), "n_passages": len(ids),
        "embedder": args.embedder, "reranker": args.rerank, "note": baseline_note,
    }
    report = write_report(ts, meta, results)
    print(f"wrote {report}")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
