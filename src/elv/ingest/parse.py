"""Tolerant, pluggable parsers. A parser never crashes the batch: per-file
failures are caught and returned as ParseResult(ok=False, error=...) for the
audit log (the "PoC dies on real data" failure mode, ADR / failure-mapping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

_ENCODINGS = ("utf-8", "cp932", "shift_jis", "euc-jp", "latin-1")  # JP-aware order


@dataclass
class ParsedDoc:
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ParseResult:
    path: str
    ok: bool
    docs: list[ParsedDoc] = field(default_factory=list)
    error: str = ""
    warnings: list[str] = field(default_factory=list)


def _read_text_bytes(raw: bytes) -> tuple[str, str]:
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"  # last resort


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data)


def parse_file(path: str | Path) -> ParseResult:
    p = Path(path)
    res = ParseResult(path=str(p), ok=True)
    try:
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            res.docs, res.warnings = _parse_pdf(p)
        elif suffix in (".html", ".htm"):
            text, enc = _read_text_bytes(p.read_bytes())
            parser = _Text(); parser.feed(text)
            res.docs = [ParsedDoc(title=p.stem, text="\n".join(parser.parts),
                                  metadata={"source": str(p), "encoding": enc})]
        elif suffix in (".txt", ".md", ".markdown", ""):
            text, enc = _read_text_bytes(p.read_bytes())
            if enc not in ("utf-8",):
                res.warnings.append(f"decoded as {enc} (non-utf-8 source)")
            res.docs = [ParsedDoc(title=p.stem, text=text,
                                  metadata={"source": str(p), "encoding": enc})]
        else:
            res.ok = False; res.error = f"unsupported file type: {suffix}"
    except Exception as e:  # tolerant: one bad file must not kill the batch
        res.ok = False
        res.error = f"{type(e).__name__}: {e}"
    return res


def _parse_pdf(p: Path) -> tuple[list[ParsedDoc], list[str]]:
    import pdfplumber  # lazy import

    docs, warnings, total_chars = [], [], 0
    with pdfplumber.open(str(p)) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            total_chars += len(text)
            tables = page.extract_tables() or []
            tbl_text = "\n".join(
                "\n".join(" | ".join(c or "" for c in row) for row in t) for t in tables)
            body = (text + ("\n\n[表]\n" + tbl_text if tbl_text else "")).strip()
            if body:
                docs.append(ParsedDoc(
                    title=f"{p.stem} p.{i}", text=body,
                    metadata={"source": str(p), "page": i}))
        # no/low text layer => likely scanned => flag for OCR (see SKILL)
        if n_pages and total_chars < 20 * n_pages:
            warnings.append("low/no text layer — likely scanned; OCR fallback needed")
    return docs, warnings
