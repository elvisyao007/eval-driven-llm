"""
DeepDoc Eval Phase 3.2 — 3×2 retrieval comparison (pdfplumber × DeepDoc × MinerU).

Extends Phase 3.1 (2×2) by adding Pipeline C (MinerU) to the evaluation:
  A×BM25   : pdfplumber plain text → BM25 (character bigram)
  A×dense  : pdfplumber plain text → ruri-v3 dense (cosine similarity)
  B×BM25   : DeepDoc structured → BM25
  B×dense  : DeepDoc structured → ruri-v3 dense
  C×BM25   : MinerU structured → BM25
  C×dense  : MinerU structured → ruri-v3 dense

Same golden set v2, same embedding model (ruri-v3-310m), same sanity gate.
MinerU parse quality JSON produced by scripts/mineru_parse_all.py.

Key questions (Phase 3.2):
  1. Does MinerU's layout parsing give dense retrieval advantage similar to DeepDoc?
  2. Is MinerU's chunk quality better/worse than DeepDoc for dense embedding?
  3. Which parser produces the highest hit@5 across both retrievers?

Run from /mnt/data/eval-driven-llm:
  .venv/bin/python scripts/retrieval_delta_v3.py

Sanity gate: oracle full-text coverage must be ≥60% in at least one pipeline.
"""
import json
import os
import re
import sys
import time
from typing import List, Dict, Any, Tuple

PARSE_DIR = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/parse_quality"
GOLDEN    = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v2/golden_set_v2.json"
OUT_DIR   = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v2"
SAMPLES   = ["02_mof_budget_fy2025", "03_stat_kakei_2023", "04_mof_fiscal_202510"]

EMBED_MODEL  = "cl-nagoya/ruri-v3-310m"
QUERY_PREFIX = "検索クエリ: "
DOC_PREFIX   = "検索文書: "


# ─────────────────────────────────────────── tokenizer (BM25)

def bigram_tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", "", text)
    return [text[i:i+2] for i in range(len(text) - 1)]


# ─────────────────────────────────────────── chunking

CHUNK_SIZE = 300
CHUNK_STEP = 150


def sliding_chunks(text: str, source: str, page: int = 0) -> List[Dict]:
    chunks = []
    for i in range(0, max(1, len(text) - CHUNK_SIZE + CHUNK_STEP), CHUNK_STEP):
        chunk = text[i:i + CHUNK_SIZE].strip()
        if len(chunk) > 20:
            chunks.append({"text": chunk, "source": source, "page": page, "start": i})
    if not chunks and text.strip():
        chunks.append({"text": text.strip(), "source": source, "page": page, "start": 0})
    return chunks


# ─────────────────────────────────────────── corpus builders

def build_corpus_plain(stems: List[str]) -> List[Dict]:
    corpus = []
    for stem in stems:
        path = f"{PARSE_DIR}/{stem}_plain.json"
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        source = data["source"]
        for item in data["chunks"]:
            text = item.get("text", "").strip()
            if len(text) < 10:
                continue
            corpus.extend(sliding_chunks(text, source, item.get("page", 0)))
    return corpus


def build_corpus_deepdoc(stems: List[str]) -> List[Dict]:
    corpus = []
    for stem in stems:
        path = f"{PARSE_DIR}/{stem}_deepdoc.json"
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        source = data["source"]
        for item in data.get("chunks", []):
            text = item.get("text", "").strip()
            if len(text) < 10:
                continue
            corpus.extend(sliding_chunks(text, source, item.get("page", 0)))
        for tbl in data.get("tables", []):
            text = tbl.get("text", "").strip()
            if len(text) > 10:
                corpus.extend(sliding_chunks(text, source, 0))
    return corpus


def build_corpus_mineru(stems: List[str]) -> List[Dict]:
    corpus = []
    for stem in stems:
        path = f"{PARSE_DIR}/{stem}_mineru.json"
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        source = data["source"]
        for item in data.get("chunks", []):
            text = item.get("text", "").strip()
            if len(text) < 10:
                continue
            corpus.extend(sliding_chunks(text, source, item.get("page", 0)))
    return corpus


# ─────────────────────────────────────────── BM25

def build_bm25(corpus: List[Dict]):
    from rank_bm25 import BM25Okapi
    tokenized = [bigram_tokenize(c["text"]) for c in corpus]
    valid = [(tok, c) for tok, c in zip(tokenized, corpus) if tok]
    if not valid:
        return None, []
    toks, valid_corpus = zip(*valid)
    return BM25Okapi(list(toks)), list(valid_corpus)


def bm25_retrieve(bm25, corpus: List[Dict], query: str, k: int = 5) -> List[Dict]:
    if bm25 is None:
        return []
    tokens = bigram_tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [corpus[i] for i in top_k_idx]


# ─────────────────────────────────────────── Dense (ruri-v3)

def build_dense_index(corpus: List[Dict], model) -> Tuple:
    import numpy as np
    texts = [DOC_PREFIX + c["text"] for c in corpus]
    print(f"    embedding {len(texts)} chunks with ruri-v3…", flush=True)
    t0 = time.time()
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False, batch_size=64)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)
    return embs.astype("float32"), corpus


def dense_retrieve(embs, corpus: List[Dict], model, query: str, k: int = 5) -> List[Dict]:
    import numpy as np
    q_emb = model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")[0]
    sims = embs @ q_emb
    top_k_idx = sims.argsort()[::-1][:k]
    return [corpus[i] for i in top_k_idx]


# ─────────────────────────────────────────── hit@k eval

def check_hit(retrieved: List[Dict], keywords: List[str]) -> bool:
    combined = " ".join(c["text"] for c in retrieved)
    return any(kw in combined for kw in keywords)


# ─────────────────────────────────────────── sanity gate (3 pipelines)

def sanity_gate(corpora: Dict[str, List[Dict]], golden: dict) -> Tuple[bool, dict]:
    print("\n=== Sanity Gate: Oracle full-text keyword search ===")
    n = len(golden["questions"])
    oracle_counts = {}
    missing_ids = []

    for name, corpus in corpora.items():
        all_text = " ".join(c["text"] for c in corpus)
        found = 0
        for q in golden["questions"]:
            if any(kw in all_text for kw in q["answer_keywords"]):
                found += 1
        oracle_counts[name] = found
        print(f"  Pipeline {name} oracle: {found}/{n} = {found/n:.1%}")

    for q in golden["questions"]:
        if not any(
            any(kw in " ".join(c["text"] for c in corpus) for kw in q["answer_keywords"])
            for corpus in corpora.values()
        ):
            missing_ids.append(q["id"])

    if missing_ids:
        print(f"  Questions not found in ANY corpus (oracle-failing): {missing_ids}")

    passed = any(v / n >= 0.60 for v in oracle_counts.values())
    print(f"  Sanity gate: {'PASS' if passed else 'FAIL'} (≥60% in at least one pipeline)")

    return passed, {
        "oracle_counts": {k: v for k, v in oracle_counts.items()},
        "oracle_ceilings": {k: round(v / n, 4) for k, v in oracle_counts.items()},
        "n": n,
        "missing_ids": missing_ids,
    }


# ─────────────────────────────────────────── main

def main():
    import warnings
    warnings.filterwarnings("ignore")

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    n_q = len(golden["questions"])
    print(f"Golden set v2: {n_q} questions")

    print("\nBuilding corpus A (plain pdfplumber)…")
    corpus_a = build_corpus_plain(SAMPLES)
    print(f"  {len(corpus_a)} sliding-window chunks")

    print("Building corpus B (DeepDoc structured)…")
    corpus_b = build_corpus_deepdoc(SAMPLES)
    print(f"  {len(corpus_b)} sliding-window chunks")

    print("Building corpus C (MinerU structured)…")
    corpus_c = build_corpus_mineru(SAMPLES)
    print(f"  {len(corpus_c)} sliding-window chunks")

    # Sanity gate (3 pipelines)
    gate_passed, gate_stats = sanity_gate(
        {"A (plain)": corpus_a, "B (DeepDoc)": corpus_b, "C (MinerU)": corpus_c},
        golden
    )
    if not gate_passed:
        print("\n[WARNING] Sanity gate FAILED — results may not be trustworthy.")

    # ── BM25 indices
    print("\nBuilding BM25 indices…")
    bm25_a, corpus_a_bm25 = build_bm25(corpus_a)
    bm25_b, corpus_b_bm25 = build_bm25(corpus_b)
    bm25_c, corpus_c_bm25 = build_bm25(corpus_c)
    print(f"  BM25-A: {len(corpus_a_bm25)}, BM25-B: {len(corpus_b_bm25)}, BM25-C: {len(corpus_c_bm25)} valid docs")

    # ── Dense: load ruri-v3
    print(f"\nLoading dense embedder: {EMBED_MODEL}…")
    from sentence_transformers import SentenceTransformer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ruri = SentenceTransformer(EMBED_MODEL)
    print("  model loaded")

    print("\nBuilding dense index A (plain)…")
    embs_a, _ = build_dense_index(corpus_a, ruri)
    print("Building dense index B (DeepDoc)…")
    embs_b, _ = build_dense_index(corpus_b, ruri)
    print("Building dense index C (MinerU)…")
    embs_c, _ = build_dense_index(corpus_c, ruri)

    # ── Per-question evaluation
    results = []
    print("\n=== Question-level results ===")
    header = (f"{'ID':<5} {'Diff':<12} {'BM25-A':>7} {'BM25-B':>7} {'BM25-C':>7} "
              f"{'Den-A':>7} {'Den-B':>7} {'Den-C':>7}")
    print(header)
    print("-" * len(header))

    for q in golden["questions"]:
        kws = q["answer_keywords"]
        qtext = q["question"]
        diff = q.get("difficulty", "?")

        top_bm25_a = bm25_retrieve(bm25_a, corpus_a_bm25, qtext, k=5)
        top_bm25_b = bm25_retrieve(bm25_b, corpus_b_bm25, qtext, k=5)
        top_bm25_c = bm25_retrieve(bm25_c, corpus_c_bm25, qtext, k=5)
        top_den_a  = dense_retrieve(embs_a, corpus_a, ruri, qtext, k=5)
        top_den_b  = dense_retrieve(embs_b, corpus_b, ruri, qtext, k=5)
        top_den_c  = dense_retrieve(embs_c, corpus_c, ruri, qtext, k=5)

        h_bm25_a = check_hit(top_bm25_a, kws)
        h_bm25_b = check_hit(top_bm25_b, kws)
        h_bm25_c = check_hit(top_bm25_c, kws)
        h_den_a  = check_hit(top_den_a,  kws)
        h_den_b  = check_hit(top_den_b,  kws)
        h_den_c  = check_hit(top_den_c,  kws)

        sym = lambda h: "✓" if h else "✗"
        print(f"{q['id']:<5} {diff:<12} {sym(h_bm25_a):>7} {sym(h_bm25_b):>7} {sym(h_bm25_c):>7} "
              f"{sym(h_den_a):>7} {sym(h_den_b):>7} {sym(h_den_c):>7}")

        results.append({
            "id": q["id"],
            "source": q["source"],
            "difficulty": diff,
            "question": qtext,
            "answer_keywords": kws,
            "hit_bm25_a": h_bm25_a, "hit_bm25_b": h_bm25_b, "hit_bm25_c": h_bm25_c,
            "hit_dense_a": h_den_a, "hit_dense_b": h_den_b, "hit_dense_c": h_den_c,
            "top5_bm25_a":  [c["text"][:80] for c in top_bm25_a],
            "top5_bm25_b":  [c["text"][:80] for c in top_bm25_b],
            "top5_bm25_c":  [c["text"][:80] for c in top_bm25_c],
            "top5_dense_a": [c["text"][:80] for c in top_den_a],
            "top5_dense_b": [c["text"][:80] for c in top_den_b],
            "top5_dense_c": [c["text"][:80] for c in top_den_c],
        })

    # ── Aggregate
    def hit5(key):
        return sum(r[key] for r in results) / n_q

    h_bm25_a = hit5("hit_bm25_a");  h_bm25_b = hit5("hit_bm25_b");  h_bm25_c = hit5("hit_bm25_c")
    h_den_a  = hit5("hit_dense_a"); h_den_b  = hit5("hit_dense_b"); h_den_c  = hit5("hit_dense_c")

    print("\n=== 3×2 Summary (hit@5) ===")
    print(f"{'Pipeline':<20} {'BM25':>8} {'Dense':>8}")
    print("-" * 38)
    print(f"{'A (plain)':<20} {h_bm25_a:>8.1%} {h_den_a:>8.1%}")
    print(f"{'B (DeepDoc)':<20} {h_bm25_b:>8.1%} {h_den_b:>8.1%}")
    print(f"{'C (MinerU)':<20} {h_bm25_c:>8.1%} {h_den_c:>8.1%}")
    print(f"{'Delta B−A':<20} {h_bm25_b-h_bm25_a:>+8.1%} {h_den_b-h_den_a:>+8.1%}")
    print(f"{'Delta C−A':<20} {h_bm25_c-h_bm25_a:>+8.1%} {h_den_c-h_den_a:>+8.1%}")
    print(f"{'Delta C−B':<20} {h_bm25_c-h_bm25_b:>+8.1%} {h_den_c-h_den_b:>+8.1%}")
    print()
    print(f"Sanity gate  : {'PASS' if gate_passed else 'FAIL'}")
    print(f"Oracle counts: {gate_stats['oracle_counts']}")

    # ── Save
    out = {
        "metadata": {
            "phase": "3.2",
            "golden_set": GOLDEN,
            "n_questions": n_q,
            "samples": SAMPLES,
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_step": CHUNK_STEP,
            "pipelines": {
                "A": "pdfplumber plain text",
                "B": "DeepDoc (layout + TSR)",
                "C": "MinerU 3.3.1 (pipeline backend, txt method, japan lang)",
            },
        },
        "sanity_gate": {"passed": gate_passed, **gate_stats},
        "hit5": {
            "bm25_a":  round(h_bm25_a, 4), "bm25_b":  round(h_bm25_b, 4), "bm25_c":  round(h_bm25_c, 4),
            "dense_a": round(h_den_a,  4),  "dense_b": round(h_den_b,  4),  "dense_c": round(h_den_c,  4),
            "delta_bm25_b_a":  round(h_bm25_b - h_bm25_a, 4),
            "delta_bm25_c_a":  round(h_bm25_c - h_bm25_a, 4),
            "delta_bm25_c_b":  round(h_bm25_c - h_bm25_b, 4),
            "delta_dense_b_a": round(h_den_b  - h_den_a,  4),
            "delta_dense_c_a": round(h_den_c  - h_den_a,  4),
            "delta_dense_c_b": round(h_den_c  - h_den_b,  4),
        },
        "corpus_sizes": {
            "plain_chunks":   len(corpus_a),
            "deepdoc_chunks": len(corpus_b),
            "mineru_chunks":  len(corpus_c),
        },
        "results": results,
    }

    out_path = f"{OUT_DIR}/retrieval_results_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
