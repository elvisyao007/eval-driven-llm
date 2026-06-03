"""Text normalization for dirty real-world documents.

Conservative on purpose: it folds width/encoding noise and collapses whitespace,
but does NOT try to auto-fix mojibake (it flags it instead) — silently "fixing"
corrupted text is how wrong data enters a knowledge base unnoticed. Every change
is reported for the audit log.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_WS = re.compile(r"[ \t\u3000]+")
_BLANKS = re.compile(r"\n{3,}")
_LATIN_HYPHEN_EOL = re.compile(r"([A-Za-z])-\n([A-Za-z])")
# crude mojibake signal: replacement chars / lone surrogate-ish artifacts
_MOJIBAKE = re.compile(r"[\ufffd]")


def normalize(text: str) -> tuple[str, dict]:
    """Return (normalized_text, changes). `changes` feeds the audit log."""
    changes: dict[str, int | bool] = {}
    if text is None:
        return "", {"empty": True}

    mojibake_hits = len(_MOJIBAKE.findall(text))
    if mojibake_hits:
        changes["mojibake_suspected"] = mojibake_hits  # flag, do NOT silently fix

    before = text
    text = unicodedata.normalize("NFKC", text)  # fold full/half width, etc.
    if text != before:
        changes["nfkc"] = True

    n_ctrl = len(_CONTROL.findall(text)) + len(_ZERO_WIDTH.findall(text))
    if n_ctrl:
        text = _ZERO_WIDTH.sub("", _CONTROL.sub("", text))
        changes["stripped_control_chars"] = n_ctrl

    n_hyph = len(_LATIN_HYPHEN_EOL.findall(text))
    if n_hyph:
        text = _LATIN_HYPHEN_EOL.sub(r"\1\2", text)  # de-hyphenate Latin line breaks
        changes["dehyphenated"] = n_hyph

    collapsed = _BLANKS.sub("\n\n", _WS.sub(" ", text)).strip()
    if collapsed != text:
        changes["whitespace_collapsed"] = True
    return collapsed, changes
