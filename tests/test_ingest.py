"""Ingestion unit tests: normalization, dedup, chunking, parser tolerance, and a
mini end-to-end ingest. No pdfplumber/fpdf2 needed (the PDF path is demoed via
scripts + make, not unit-tested, to keep test deps light)."""

from pathlib import Path

from elv.ingest import chunk as ck
from elv.ingest import dedup as dd
from elv.ingest import normalize as nm
from elv.ingest.loader import ingest_corpus
from elv.ingest.parse import parse_file


def test_normalize_folds_fullwidth_and_strips_control():
    out, ch = nm.normalize("ｖＬＬＭ\u200bは\x07高速")
    assert "vLLM" in out and "\u200b" not in out and "\x07" not in out
    assert ch.get("nfkc") and ch.get("stripped_control_chars", 0) >= 1


def test_normalize_flags_mojibake_not_fixes():
    out, ch = nm.normalize("壊れた\ufffd文字")
    assert ch.get("mojibake_suspected") == 1  # flagged, left for human review


def test_dedup_exact_and_keeps_first():
    keep, dropped = dd.find_duplicates(["A", "B", "A"])
    assert keep == [0, 1] and dropped == [(2, 0)]


def test_dedup_near_duplicate():
    # SimHash near-dup is conservative and most reliable on longer text; a near
    # copy (trailing edit) on a long doc falls within threshold, a clearly
    # different doc does not.
    base = "オンプレミスでLLMを安全に運用するための設計指針をまとめた技術文書である。" * 3
    near = base + "追記。"
    far = "本日の天気は晴れで、最高気温は二十度の見込みである。" * 3
    keep, dropped = dd.find_duplicates([base, near, far], near_threshold=3)
    assert len(keep) == 2 and dropped == [(1, 0)]


def test_chunk_respects_max_and_overlaps():
    text = "\n\n".join(f"段落{i}です。" * 10 for i in range(5))
    chunks = ck.chunk_text(text, max_chars=80, overlap=16)
    assert all(c.metadata["char_len"] <= 80 for c in chunks)
    assert chunks[0].metadata["chunk_index"] == 0


def test_parser_tolerates_unsupported(tmp_path):
    f = tmp_path / "x.xyz"; f.write_text("hi")
    res = parse_file(f)
    assert res.ok is False and "unsupported" in res.error


def test_parser_encoding_fallback(tmp_path):
    f = tmp_path / "j.txt"; f.write_bytes("日本語テキスト".encode("cp932"))
    res = parse_file(f)
    assert res.ok and "日本語" in res.docs[0].text


def test_end_to_end_ingest_tolerates_bad_file(tmp_path):
    corpus = tmp_path / "c"; corpus.mkdir()
    (corpus / "a.txt").write_text("クリーンな文書。", encoding="utf-8")
    (corpus / "dup.txt").write_text("クリーンな文書。", encoding="utf-8")  # exact dup
    (corpus / "bad.pdf").write_bytes(b"not a pdf")  # must be caught
    rep, out, audit = ingest_corpus(corpus, tmp_path / "out.jsonl")
    assert rep.n_failed == 1 and rep.n_parsed_ok == 2
    assert rep.n_dropped_duplicates == 1
    assert Path(out).exists() and Path(audit).exists()
