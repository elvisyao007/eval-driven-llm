"""
Parse all 5 Japanese PDF samples using DeepDoc PdfParser API.
Outputs structured JSON per sample to parse_quality/.

PdfParser.__call__ returns (text_string, tables) where:
  - text_string: \n\n-separated text blocks, each line may have
    position tag: "text@@page\tx0\ttop\tx1\tbottom##"
  - tables: list of (PIL.Image or None, html_string) tuples

Run from /mnt/data/eval-driven-llm:
  PYTHONPATH=/mnt/data/ragflow-deepdoc CUDA_VISIBLE_DEVICES="" \
  .venv-deepdoc/bin/python scripts/deepdoc_parse_all.py
"""
import json
import os
import re
import sys
import time
from html.parser import HTMLParser

sys.path.insert(0, "/mnt/data/ragflow-deepdoc")

SAMPLES_DIR = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/samples"
OUT_DIR     = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/parse_quality"

SAMPLES = [
    "01_nta_kakutei_r06.pdf",
    "02_mof_budget_fy2025.pdf",
    "03_stat_kakei_2023.pdf",
    "04_mof_fiscal_202510.pdf",
    "05_archives_scan.pdf",
]

# Regex for position tag embedded in each text line
# Format: @@page\tx0\ttop\tx1\tbottom##
_POS_RE = re.compile(r"@@(\d+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)##")


def strip_pos_tag(line):
    return _POS_RE.sub("", line).strip()


def parse_text_block(block):
    """Parse one \n\n-separated text block into text + position."""
    lines = block.strip().split("\n")
    clean_lines = []
    page, x0, top, x1, bottom = None, None, None, None, None
    for line in lines:
        m = _POS_RE.search(line)
        if m and page is None:
            page = int(m.group(1))
            x0, top, x1, bottom = float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        clean_lines.append(strip_pos_tag(line))
    text = " ".join(t for t in clean_lines if t)
    return {"text": text, "page": page, "x0": x0, "top": top, "x1": x1, "bottom": bottom}


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        t = data.strip()
        if t:
            self.parts.append(t)

    def get_text(self):
        return " ".join(self.parts)


def html_to_text(html):
    p = HTMLTextExtractor()
    p.feed(html)
    return p.get_text()


def parse_with_deepdoc(pdf_path, parser):
    t0 = time.time()
    text, tbls = parser(pdf_path, need_image=False, return_html=True)
    elapsed = time.time() - t0

    # Split text into chunks
    blocks = [b for b in text.split("\n\n") if b.strip()]
    chunks = []
    for block in blocks:
        parsed = parse_text_block(block)
        if parsed["text"]:
            chunks.append(parsed)

    # Extract table HTML/text
    tbl_list = []
    for item in tbls:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            _, content = item
            if isinstance(content, str) and content.strip():
                tbl_list.append({
                    "html": content,
                    "text": html_to_text(content),
                })

    return chunks, tbl_list, elapsed


def parse_with_plain(pdf_path):
    from deepdoc.parser import PlainParser
    t0 = time.time()
    parser = PlainParser()
    lines, _ = parser(pdf_path)
    elapsed = time.time() - t0
    chunks = [{"text": line, "page": 0} for line, _ in lines if line.strip()]
    return chunks, elapsed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    from deepdoc.parser import PdfParser
    parser = PdfParser()

    summary_rows = []

    for fname in SAMPLES:
        pdf_path = os.path.join(SAMPLES_DIR, fname)
        stem = fname.replace(".pdf", "")
        print(f"\n== Parsing {fname} ==")

        # DeepDoc
        try:
            chunks, tables, elapsed = parse_with_deepdoc(pdf_path, parser)
            print(f"  DeepDoc: {len(chunks)} chunks, {len(tables)} tables, {elapsed:.1f}s")
            out = {"source": fname, "parser": "deepdoc", "elapsed_s": round(elapsed, 2),
                   "chunk_count": len(chunks), "table_count": len(tables),
                   "chunks": chunks, "tables": tables}
            with open(f"{OUT_DIR}/{stem}_deepdoc.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            import traceback
            print(f"  DeepDoc ERROR: {e}")
            traceback.print_exc()
            chunks, tables, elapsed = [], [], 0

        # Plain
        try:
            plain_chunks, plain_elapsed = parse_with_plain(pdf_path)
            print(f"  Plain:   {len(plain_chunks)} chunks, {plain_elapsed:.1f}s")
            out_plain = {"source": fname, "parser": "plain", "elapsed_s": round(plain_elapsed, 2),
                         "chunk_count": len(plain_chunks), "chunks": plain_chunks}
            with open(f"{OUT_DIR}/{stem}_plain.json", "w", encoding="utf-8") as f:
                json.dump(out_plain, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  Plain ERROR: {e}")
            plain_chunks, plain_elapsed = [], 0

        summary_rows.append({
            "file": fname,
            "deepdoc_chunks": len(chunks),
            "deepdoc_tables": len(tables),
            "deepdoc_elapsed_s": round(elapsed, 2),
            "plain_chunks": len(plain_chunks),
            "plain_elapsed_s": round(plain_elapsed, 2),
        })

    with open(f"{OUT_DIR}/parse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("\n=== Parse Summary ===")
    print(f"{'File':<35} {'DD chunks':>10} {'DD tables':>10} {'DD sec':>8} {'Plain ch':>9} {'Plain sec':>10}")
    for r in summary_rows:
        print(f"{r['file']:<35} {r['deepdoc_chunks']:>10} {r['deepdoc_tables']:>10} "
              f"{r['deepdoc_elapsed_s']:>8.1f} {r['plain_chunks']:>9} {r['plain_elapsed_s']:>10.1f}")


if __name__ == "__main__":
    main()
