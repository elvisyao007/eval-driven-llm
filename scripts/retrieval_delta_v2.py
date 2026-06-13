"""
DeepDoc Eval v2 — 2×2 retrieval comparison.

Phase 3.1: Tests all four pipeline×retriever combinations on golden_set_v2.json:
  A×BM25 : pdfplumber plain text → BM25 (character bigram)
  A×dense : pdfplumber plain text → ruri-v3 dense (cosine similarity, in-memory)
  B×BM25 : DeepDoc structured → BM25 (character bigram)
  B×dense : DeepDoc structured → ruri-v3 dense (cosine similarity, in-memory)

Key question: Does DeepDoc's +15% BM25 advantage (v1) persist under dense retrieval?
Dense embedding is less sensitive to chunk boundaries than BM25 → advantage may shrink.

Embedding: cl-nagoya/ruri-v3-310m (Japanese-first, already cached at /mnt/cache/hf,
already used in the main retrieval pipeline). ruri requires instruction prefixes:
  query: "検索クエリ: " + text
  doc:   "検索文書: " + text

Run from /mnt/data/eval-driven-llm:
  .venv/bin/python scripts/retrieval_delta_v2.py

Sanity gate: oracle full-text coverage must be ≥60% in at least one pipeline.
Oracle ceiling for v2 is expected ~87.5% (28/32), proving harder than v1 (100%).
"""
import json
import os
import re
import sys
import time
from typing import List, Dict, Any, Tuple

PARSE_DIR  = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/parse_quality"
GOLDEN     = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v2/golden_set_v2.json"
OUT_DIR    = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v2"
SAMPLES    = ["02_mof_budget_fy2025", "03_stat_kakei_2023", "04_mof_fiscal_202510"]

EMBED_MODEL   = "cl-nagoya/ruri-v3-310m"
QUERY_PREFIX  = "検索クエリ: "
DOC_PREFIX    = "検索文書: "


# ────────────────────────────────────────────────── tokenizer (BM25)

def bigram_tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", "", text)
    return [text[i:i+2] for i in range(len(text) - 1)]


# ────────────────────────────────────────────────── chunking

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


# ────────────────────────────────────────────────── BM25

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


# ────────────────────────────────────────────────── Dense (ruri-v3)

def build_dense_index(corpus: List[Dict], model) -> Tuple[Any, List[Dict]]:
    """Encode all corpus chunks and return (embeddings_array, corpus)."""
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


# ────────────────────────────────────────────────── hit@k eval

def check_hit(retrieved: List[Dict], keywords: List[str]) -> bool:
    combined = " ".join(c["text"] for c in retrieved)
    return any(kw in combined for kw in keywords)


# ────────────────────────────────────────────────── sanity gate

def sanity_gate(corpus_a: List[Dict], corpus_b: List[Dict], golden: dict) -> Tuple[bool, dict]:
    print("\n=== Sanity Gate: Oracle full-text keyword search ===")
    all_text_a = " ".join(c["text"] for c in corpus_a)
    all_text_b = " ".join(c["text"] for c in corpus_b)

    found_a, found_b = 0, 0
    issues = []
    for q in golden["questions"]:
        kws = q["answer_keywords"]
        hit_a = any(kw in all_text_a for kw in kws)
        hit_b = any(kw in all_text_b for kw in kws)
        if hit_a:
            found_a += 1
        if hit_b:
            found_b += 1
        if not hit_a and not hit_b:
            issues.append(q["id"])

    n = len(golden["questions"])
    print(f"Pipeline A oracle: {found_a}/{n} = {found_a/n:.1%}")
    print(f"Pipeline B oracle: {found_b}/{n} = {found_b/n:.1%}")
    if issues:
        print(f"Questions not found in EITHER corpus (expected oracle-failing): {issues}")

    oracle_ceiling = found_a / n  # A and B have same oracle since keywords must exist somewhere
    pass_a = found_a / n >= 0.6
    pass_b = found_b / n >= 0.6
    passed = pass_a or pass_b
    print(f"Sanity gate: {'PASS' if passed else 'FAIL'} (≥60% required in at least one pipeline)")
    return passed, {
        "oracle_a": found_a,
        "oracle_b": found_b,
        "n": n,
        "oracle_ceiling_a": round(found_a / n, 4),
        "oracle_ceiling_b": round(found_b / n, 4),
        "missing_ids": issues
    }


# ────────────────────────────────────────────────── main

def main():
    import warnings
    warnings.filterwarnings("ignore")

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    n_q = len(golden["questions"])
    print(f"Golden set v2: {n_q} questions")
    print(f"Oracle-failing IDs (expected): {golden['metadata'].get('oracle_failing_ids', [])}")

    print("\nBuilding corpus A (plain pdfplumber)…")
    corpus_a = build_corpus_plain(SAMPLES)
    print(f"  {len(corpus_a)} sliding-window chunks")

    print("Building corpus B (DeepDoc structured)…")
    corpus_b = build_corpus_deepdoc(SAMPLES)
    print(f"  {len(corpus_b)} sliding-window chunks")

    # Sanity gate
    gate_passed, gate_stats = sanity_gate(corpus_a, corpus_b, golden)
    if not gate_passed:
        print("\n[WARNING] Sanity gate FAILED — results may not be trustworthy.")

    # ── BM25 indices
    print("\nBuilding BM25 indices…")
    bm25_a, corpus_a_bm25 = build_bm25(corpus_a)
    bm25_b, corpus_b_bm25 = build_bm25(corpus_b)
    print(f"  BM25-A: {len(corpus_a_bm25)} valid docs")
    print(f"  BM25-B: {len(corpus_b_bm25)} valid docs")

    # ── Dense: load ruri-v3 and build embeddings
    print(f"\nLoading dense embedder: {EMBED_MODEL}…")
    from sentence_transformers import SentenceTransformer
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ruri = SentenceTransformer(EMBED_MODEL)
    print("  model loaded")

    print("\nBuilding dense index A (plain)…")
    embs_a, _ = build_dense_index(corpus_a, ruri)
    print("Building dense index B (DeepDoc)…")
    embs_b, _ = build_dense_index(corpus_b, ruri)

    # ── Per-question evaluation
    results = []
    print("\n=== Question-level results ===")
    header = f"{'ID':<5} {'Diff':<12} {'BM25-A':>7} {'BM25-B':>7} {'Den-A':>7} {'Den-B':>7}"
    print(header)
    print("-" * len(header))

    for q in golden["questions"]:
        kws = q["answer_keywords"]
        qtext = q["question"]
        diff = q.get("difficulty", "?")

        top_bm25_a = bm25_retrieve(bm25_a, corpus_a_bm25, qtext, k=5)
        top_bm25_b = bm25_retrieve(bm25_b, corpus_b_bm25, qtext, k=5)
        top_den_a  = dense_retrieve(embs_a, corpus_a, ruri, qtext, k=5)
        top_den_b  = dense_retrieve(embs_b, corpus_b, ruri, qtext, k=5)

        h_bm25_a = check_hit(top_bm25_a, kws)
        h_bm25_b = check_hit(top_bm25_b, kws)
        h_den_a  = check_hit(top_den_a,  kws)
        h_den_b  = check_hit(top_den_b,  kws)

        sym = lambda h: "✓" if h else "✗"
        print(f"{q['id']:<5} {diff:<12} {sym(h_bm25_a):>7} {sym(h_bm25_b):>7} {sym(h_den_a):>7} {sym(h_den_b):>7}")

        results.append({
            "id": q["id"],
            "source": q["source"],
            "difficulty": diff,
            "question": qtext,
            "answer_keywords": kws,
            "hit_bm25_a": h_bm25_a,
            "hit_bm25_b": h_bm25_b,
            "hit_dense_a": h_den_a,
            "hit_dense_b": h_den_b,
            "top5_bm25_a": [c["text"][:80] for c in top_bm25_a],
            "top5_bm25_b": [c["text"][:80] for c in top_bm25_b],
            "top5_dense_a": [c["text"][:80] for c in top_den_a],
            "top5_dense_b": [c["text"][:80] for c in top_den_b],
        })

    # ── Aggregate
    def hit5(key):
        return sum(r[key] for r in results) / n_q

    h_bm25_a = hit5("hit_bm25_a")
    h_bm25_b = hit5("hit_bm25_b")
    h_den_a  = hit5("hit_dense_a")
    h_den_b  = hit5("hit_dense_b")

    print("\n=== 2×2 Summary ===")
    print(f"{'Pipeline':<20} {'BM25':>8} {'Dense':>8}")
    print("-" * 38)
    print(f"{'A (plain)':<20} {h_bm25_a:>8.1%} {h_den_a:>8.1%}")
    print(f"{'B (DeepDoc)':<20} {h_bm25_b:>8.1%} {h_den_b:>8.1%}")
    print(f"{'Delta (B-A)':<20} {h_bm25_b-h_bm25_a:>+8.1%} {h_den_b-h_den_a:>+8.1%}")
    print()
    print(f"Sanity gate         : {'PASS' if gate_passed else 'FAIL'}")
    print(f"Oracle ceiling (A)  : {gate_stats['oracle_ceiling_a']:.1%}  ({gate_stats['oracle_a']}/{gate_stats['n']})")
    print(f"Oracle ceiling (B)  : {gate_stats['oracle_ceiling_b']:.1%}  ({gate_stats['oracle_b']}/{gate_stats['n']})")
    print(f"Missing (oracle-fail): {gate_stats['missing_ids']}")

    # ── Save
    out = {
        "metadata": {
            "golden_set": GOLDEN,
            "n_questions": n_q,
            "samples": SAMPLES,
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_step": CHUNK_STEP,
        },
        "sanity_gate": {"passed": gate_passed, **gate_stats},
        "hit5": {
            "bm25_a":  round(h_bm25_a, 4),
            "bm25_b":  round(h_bm25_b, 4),
            "dense_a": round(h_den_a, 4),
            "dense_b": round(h_den_b, 4),
            "delta_bm25":  round(h_bm25_b - h_bm25_a, 4),
            "delta_dense": round(h_den_b  - h_den_a,  4),
        },
        "corpus_sizes": {
            "plain_chunks": len(corpus_a),
            "deepdoc_chunks": len(corpus_b),
            "bm25_a_valid": len(corpus_a_bm25),
            "bm25_b_valid": len(corpus_b_bm25),
        },
        "results": results,
    }

    out_path = f"{OUT_DIR}/retrieval_results_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
