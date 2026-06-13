"""
Phase 3.2 — MinerU parser for 3 Japanese PDF samples (02/03/04).

Reads MinerU content_list.json output and converts to the same JSON schema
used by deepdoc_parse_all.py and pdfplumber plain-text parse.

Output schema (same as deepdoc/plain):
  {
    "source": "02_mof_budget_fy2025.pdf",
    "parser": "mineru",
    "elapsed_s": float,
    "chunk_count": int,
    "table_count": int,
    "chunks": [
      {"text": str, "page": int}
    ]
  }

Two modes:
  1. Convert pre-existing MinerU output (fast, default): reads from MINERU_TMP_DIR
  2. Full run (--run): calls .venv-mineru/bin/mineru to parse, then converts

Run from /mnt/data/eval-driven-llm:
  # Convert pre-existing output (after running MinerU separately):
  .venv/bin/python scripts/mineru_parse_all.py

  # Run MinerU + convert:
  .venv/bin/python scripts/mineru_parse_all.py --run

MinerU isolation: .venv-mineru is SEPARATE from .venv and .venv-deepdoc.
Do NOT import MinerU into the main venv.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from html.parser import HTMLParser
from typing import List, Dict, Optional

SAMPLES_DIR = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/samples"
OUT_DIR     = "/mnt/data/eval-driven-llm/reports/deepdoc-eval-v1/parse_quality"
MINERU_VENV = "/mnt/data/eval-driven-llm/.venv-mineru/bin/mineru"

SAMPLES = [
    "02_mof_budget_fy2025",
    "03_stat_kakei_2023",
    "04_mof_fiscal_202510",
]

# Default temp output paths from previous mineru runs
MINERU_TMP = {
    "02_mof_budget_fy2025": "/tmp/mineru-out/02_mof_budget_fy2025/txt",
    "03_stat_kakei_2023":   "/tmp/mineru-out03/03_stat_kakei_2023/txt",
    "04_mof_fiscal_202510": "/tmp/mineru-out04/04_mof_fiscal_202510/txt",
}


# ─────────────────────────────────────────── HTML table text extractor

class _TableTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells: List[str] = []
        self._current: List[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self._in_cell = True
            self._current = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            cell = " ".join(self._current).strip()
            if cell:
                self.cells.append(cell)
            self._in_cell = False
        elif tag == "tr":
            pass  # row separator — cells already accumulated

    def handle_data(self, data):
        if self._in_cell:
            self._current.append(data.strip())


def html_table_to_text(html: str) -> str:
    """Extract cell text from HTML table, return as tab-separated rows."""
    p = _TableTextExtractor()
    p.feed(html)
    return "\t".join(p.cells)


# ─────────────────────────────────────────── MinerU output conversion

def content_list_to_chunks(content_list: list) -> List[Dict]:
    """Convert MinerU content_list.json to our standard chunk list."""
    chunks = []
    for block in content_list:
        btype = block.get("type", "")
        page = block.get("page_idx", 0)
        # page_idx is 0-indexed in MinerU; keep as-is (consistent with deepdoc page numbering)

        if btype == "text":
            text = block.get("text", "").strip()
            if len(text) >= 10:
                chunks.append({"text": text, "page": page})

        elif btype == "table":
            # Build text from: caption + table cells + footnote
            parts = []
            for cap in block.get("table_caption", []):
                if cap.strip():
                    parts.append(cap.strip())
            html = block.get("table_body", "")
            if html:
                cell_text = html_table_to_text(html)
                if cell_text:
                    parts.append(cell_text)
            for fn in block.get("table_footnote", []):
                if fn.strip():
                    parts.append(fn.strip())
            text = " ".join(parts)
            if len(text) >= 10:
                chunks.append({"text": text, "page": page})

        # Skip: page_number, image, interline_equation, etc.

    return chunks


def convert_mineru_output(stem: str, txt_dir: str, elapsed_s: float) -> Dict:
    """Read content_list.json from txt_dir and produce our standard JSON."""
    path = os.path.join(txt_dir, f"{stem}_content_list.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"MinerU content_list not found: {path}")

    with open(path, encoding="utf-8") as f:
        content_list = json.load(f)

    chunks = content_list_to_chunks(content_list)
    table_count = sum(1 for b in content_list if b.get("type") == "table")

    return {
        "source": f"{stem}.pdf",
        "parser": "mineru",
        "elapsed_s": round(elapsed_s, 2),
        "chunk_count": len(chunks),
        "table_count": table_count,
        "chunks": chunks,
    }


# ─────────────────────────────────────────── MinerU runner (--run mode)

def run_mineru(pdf_path: str, out_dir: str) -> float:
    """Run MinerU CLI and return elapsed seconds."""
    cmd = [
        MINERU_VENV,
        "-p", pdf_path,
        "-o", out_dir,
        "-b", "pipeline",
        "-m", "txt",
        "-l", "japan",
        "-f", "false",
        "-t", "true",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    if result.returncode != 0:
        # Check if error message is in stdout (MinerU logs to stdout mostly)
        err = result.stderr or result.stdout
        raise RuntimeError(f"MinerU failed (rc={result.returncode}): {err[-400:]}")
    return elapsed


# ─────────────────────────────────────────── main

def main():
    parser = argparse.ArgumentParser(description="MinerU → parse quality JSON")
    parser.add_argument("--run", action="store_true",
                        help="Run MinerU CLI (needs .venv-mineru). Default: read existing output.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    for stem in SAMPLES:
        out_path = os.path.join(OUT_DIR, f"{stem}_mineru.json")

        if args.run:
            print(f"\n=== Running MinerU on {stem} ===")
            import tempfile
            tmp_root = tempfile.mkdtemp(prefix="mineru_")
            t0 = time.time()
            try:
                elapsed = run_mineru(
                    os.path.join(SAMPLES_DIR, f"{stem}.pdf"),
                    tmp_root,
                )
            except Exception as e:
                print(f"  [ERROR] MinerU run failed: {e}")
                continue
            txt_dir = os.path.join(tmp_root, stem, "txt")
        else:
            txt_dir = MINERU_TMP.get(stem)
            if not txt_dir or not os.path.isdir(txt_dir):
                print(f"[WARN] Pre-existing MinerU output not found for {stem}. "
                      f"Run with --run to parse, or set MINERU_TMP in the script.")
                continue
            elapsed = 0.0  # unknown; fill placeholder
            # Try to recover elapsed from existing conversion metadata
            if os.path.exists(out_path):
                try:
                    with open(out_path) as f:
                        prev = json.load(f)
                    elapsed = prev.get("elapsed_s", 0.0)
                except Exception:
                    pass

        print(f"\n=== Converting MinerU output: {stem} ===")
        try:
            result = convert_mineru_output(stem, txt_dir, elapsed)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  chunks: {result['chunk_count']}, tables: {result['table_count']}")
        print(f"  elapsed: {result['elapsed_s']}s")
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
