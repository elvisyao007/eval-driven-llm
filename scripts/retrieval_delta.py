"""
Retrieval delta: Pipeline A (plain) vs Pipeline B (DeepDoc).

Both pipelines:
  - Chunk text from the parsed output
  - Build a BM25 index (character bigram tokenization for Japanese)
  - Retrieve top-5 chunks for each question in golden_set.json
  - hit@5 = 1 if any of the answer_keywords appear in top-5 chunks

Japanese BM25 note: since Japanese has no word boundaries, we use
character bigrams as tokens. This is a common baseline approach for
Japanese IR without a full tokenizer dependency.

Run from /mnt/data/eval-driven-llm:
  PYTHONPATH=/mnt/data/ragflow-deepdoc CUDA_VISIBLE_DEVICES="" \
  .venv-deepdoc/bin/python scripts/retrieval_delta.py
"""
import json
import os
import re
import sys
from typing import List, Dict, Any

sys.path.insert(0, "/mnt/data/ragflow-deepdoc")

PARSE_DIR  = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/parse_quality"
GOLDEN     = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/retrieval_delta/golden_set.json"
OUT_DIR    = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/retrieval_delta"
SAMPLES    = ["02_mof_budget_fy2025", "03_stat_kakei_2023", "04_mof_fiscal_202510"]

# ------------------------------------------------------------------ tokenizer
def bigram_tokenize(text: str) -> List[str]:
    """Character bigram tokenizer for Japanese (no word boundary needed)."""
    text = re.sub(r"\s+", "", text)  # remove whitespace
    return [text[i:i+2] for i in range(len(text) - 1)]


# ------------------------------------------------------------------ chunking
CHUNK_SIZE = 300  # characters
CHUNK_STEP = 150  # overlap


def sliding_chunks(text: str, source: str, page: int = 0) -> List[Dict]:
    """Split long text into overlapping chunks."""
    chunks = []
    for i in range(0, max(1, len(text) - CHUNK_SIZE + CHUNK_STEP), CHUNK_STEP):
        chunk = text[i:i + CHUNK_SIZE].strip()
        if len(chunk) > 20:
            chunks.append({"text": chunk, "source": source, "page": page, "start": i})
    if not chunks and text.strip():
        chunks.append({"text": text.strip(), "source": source, "page": page, "start": 0})
    return chunks


def build_corpus_plain(stems: List[str]) -> List[Dict]:
    """Load plain-parsed chunks, apply sliding window chunking."""
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
            page = item.get("page", 0)
            corpus.extend(sliding_chunks(text, source, page))
    return corpus


def build_corpus_deepdoc(stems: List[str]) -> List[Dict]:
    """Load DeepDoc-parsed chunks + table text."""
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
            page = item.get("page", 0)
            corpus.extend(sliding_chunks(text, source, page))
        # Add table text as additional chunks
        for tbl in data.get("tables", []):
            text = tbl.get("text", "").strip()
            if len(text) > 10:
                corpus.extend(sliding_chunks(text, source, 0))
    return corpus


# ------------------------------------------------------------------ BM25
def build_bm25(corpus: List[Dict]):
    from rank_bm25 import BM25Okapi
    tokenized = [bigram_tokenize(c["text"]) for c in corpus]
    # Filter empty tokenizations
    valid = [(tok, c) for tok, c in zip(tokenized, corpus) if tok]
    if not valid:
        return None, []
    toks, valid_corpus = zip(*valid)
    bm25 = BM25Okapi(list(toks))
    return bm25, list(valid_corpus)


def retrieve_top_k(bm25, corpus: List[Dict], query: str, k: int = 5) -> List[Dict]:
    if bm25 is None:
        return []
    tokens = bigram_tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [corpus[i] for i in top_k_idx]


# ------------------------------------------------------------------ hit@k eval
def check_hit(retrieved: List[Dict], keywords: List[str]) -> bool:
    """True if any of the keyword strings appear in any of the top-k chunks."""
    combined = " ".join(c["text"] for c in retrieved)
    return any(kw in combined for kw in keywords)


# ------------------------------------------------------------------ sanity gate
def sanity_gate(corpus_a, corpus_b, golden):
    """
    Verify that at least one pipeline can find the answer for >50% of questions
    using a simple keyword search (not BM25) — confirms the answers exist in the corpus.
    This is the sanity check equivalent of model-selection v1 lesson: confirm
    the oracle can find the answers before testing the retrieval system.
    """
    print("\n=== Sanity Gate: Oracle keyword search ===")
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
    print(f"Pipeline A oracle: {found_a}/{n} = {found_a/n:.0%}")
    print(f"Pipeline B oracle: {found_b}/{n} = {found_b/n:.0%}")
    if issues:
        print(f"Questions not found in either corpus: {issues}")
    pass_a = found_a / n >= 0.6
    pass_b = found_b / n >= 0.6
    passed = pass_a or pass_b
    print(f"Sanity gate: {'PASS' if passed else 'FAIL'} (≥60% required in at least one pipeline)")
    return passed, {"oracle_a": found_a, "oracle_b": found_b, "n": n, "missing_ids": issues}


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)

    print("Building corpus A (plain)...")
    corpus_a = build_corpus_plain(SAMPLES)
    print(f"  {len(corpus_a)} chunks")

    print("Building corpus B (deepdoc)...")
    corpus_b = build_corpus_deepdoc(SAMPLES)
    print(f"  {len(corpus_b)} chunks")

    # Sanity gate
    gate_passed, gate_stats = sanity_gate(corpus_a, corpus_b, golden)
    if not gate_passed:
        print("\nWARNING: Sanity gate FAILED — metric may not be trustworthy.")

    print("\nBuilding BM25 indices...")
    bm25_a, corpus_a_valid = build_bm25(corpus_a)
    bm25_b, corpus_b_valid = build_bm25(corpus_b)
    print(f"  BM25-A: {len(corpus_a_valid)} valid docs")
    print(f"  BM25-B: {len(corpus_b_valid)} valid docs")

    # Evaluate
    results = []
    print("\n=== Question-level results ===")
    print(f"{'ID':<5} {'Source':<35} {'Hit-A':>6} {'Hit-B':>6}")
    print("-" * 55)

    for q in golden["questions"]:
        kws = q["answer_keywords"]
        top_a = retrieve_top_k(bm25_a, corpus_a_valid, q["question"], k=5)
        top_b = retrieve_top_k(bm25_b, corpus_b_valid, q["question"], k=5)
        hit_a = check_hit(top_a, kws)
        hit_b = check_hit(top_b, kws)
        print(f"{q['id']:<5} {q['source']:<35} {'✓' if hit_a else '✗':>6} {'✓' if hit_b else '✗':>6}")
        results.append({
            "id": q["id"],
            "source": q["source"],
            "question": q["question"],
            "answer_keywords": kws,
            "hit_a": hit_a,
            "hit_b": hit_b,
            "top5_a": [c["text"][:80] for c in top_a],
            "top5_b": [c["text"][:80] for c in top_b],
        })

    n = len(results)
    hit5_a = sum(r["hit_a"] for r in results) / n
    hit5_b = sum(r["hit_b"] for r in results) / n
    delta   = hit5_b - hit5_a

    print("\n=== Summary ===")
    print(f"Pipeline A (plain pdfplumber) hit@5 : {hit5_a:.0%}  ({sum(r['hit_a'] for r in results)}/{n})")
    print(f"Pipeline B (DeepDoc PdfParser) hit@5: {hit5_b:.0%}  ({sum(r['hit_b'] for r in results)}/{n})")
    print(f"Delta (B - A)                        : {delta:+.1%}")
    print(f"Sanity gate                          : {'PASS' if gate_passed else 'FAIL'}")

    out = {
        "sanity_gate": {"passed": gate_passed, **gate_stats},
        "hit5_pipeline_a": round(hit5_a, 4),
        "hit5_pipeline_b": round(hit5_b, 4),
        "delta": round(delta, 4),
        "n_questions": n,
        "corpus_a_chunks": len(corpus_a_valid),
        "corpus_b_chunks": len(corpus_b_valid),
        "results": results,
    }
    with open(f"{OUT_DIR}/retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_DIR}/retrieval_results.json")


if __name__ == "__main__":
    main()
