"""Generate the dirty-data fixtures used to demo/validate ingestion.

Commit this script, not the generated files (data/corpus/_dirty is gitignored).
Run: python scripts/build_dirty_fixtures.py   then   make ingest-dirty
"""
from pathlib import Path

d = Path("data/corpus/_dirty"); d.mkdir(parents=True, exist_ok=True)

(d / "fullwidth.txt").write_bytes(
    "ｖＬＬＭ\u200bは\x07推論を\u3000\u3000高速化する。\n\n\n\nＲＡＧの説明。".encode("utf-8"))
(d / "sjis.txt").write_bytes("オンプレミス運用ではデータが社外に出ない。".encode("cp932"))
(d / "page.html").write_bytes((
    "<html><head><style>.x{color:red}</style><script>var a=1;</script></head>"
    "<body><h1>ベクトル検索</h1><p>Faissは近傍探索を高速に行う。</p></body></html>"
).encode("utf-8"))
(d / "page_copy.html").write_bytes((d / "page.html").read_bytes())  # exact dup
(d / "near.txt").write_bytes("Faissは近傍探索をとても高速に行うライブラリ。".encode("utf-8"))
(d / "broken.pdf").write_bytes(b"%PDF-1.4 not a real pdf body \x00\x01")  # must not crash batch

try:
    from fpdf import FPDF
    pdf = FPDF(); pdf.add_page(); pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 8, "vLLM serves LLMs with PagedAttention for high throughput.\n\n"
                         "Quantization reduces GPU memory so a single GPU can run a large model.")
    pdf.output(str(d / "doc.pdf"))
except ImportError:
    print("fpdf2 not installed; skipping doc.pdf (install with .[track-a])")

print("dirty fixtures:", sorted(p.name for p in d.iterdir()))
