"""
Quantify OCR errors in DeepDoc output for Japanese text.

Focuses on:
1. 令和 year notation errors (令→今, 和→和 variants)
2. Common kanji misreads in legal/tax documents
3. Per-page error rate for pages containing year references

Run from repo root after deepdoc_parse_all.py completes.
"""
import json
import re
import os

PARSE_DIR = "reports/deepdoc-eval-v1/parse_quality"
OUT_DIR = PARSE_DIR

SAMPLES = [
    "01_nta_kakutei_r06",
    "02_mof_budget_fy2025",
    "03_stat_kakei_2023",
    "04_mof_fiscal_202510",
    "05_archives_scan",
]

# Known systematic substitution errors from Phase 1 observation
# Format: (wrong, correct, description)
KNOWN_ERRORS = [
    ("今和", "令和", "year-notation: 令→今"),
    ("覆得", "所得", "legal-term: 所→覆 (income)"),
    ("覆", "所", "single-char: 所→覆"),
]

# Patterns to detect correct year references for denominator
REIWA_CORRECT = re.compile(r"令和\d+年")
REIWA_WRONG   = re.compile(r"今和\d+年")
# Any year reference (including wrong ones) for page-level denominator
YEAR_REF_ANY  = re.compile(r"[令今]和\d+年")

# Cell-level ground truth for sample 01 tax form (partial, manually verified)
# Format: (page_0indexed, cell_text_expected)
# These are values we can see in the PDF source (e-Tax public form)
TSR_GROUND_TRUTH = [
    # Page 0 - main fields of 確定申告書 R06 form A
    (0, "令和"),
    (0, "税務署長"),
    (0, "納税地"),
    (0, "個人番号"),
    (0, "生年月日"),
    (0, "氏名"),
    (0, "職業"),
    (0, "世帯主の氏名"),
    (0, "振替納税希望"),
    (0, "事業所得"),
    (0, "配当所得"),
    (0, "不動産所得"),
    (0, "給与所得"),
    (0, "公的年金等"),
    (0, "申告納税額"),
    # Page 1
    (1, "社会保険料控除"),
    (1, "生命保険料控除"),
    (1, "地震保険料控除"),
    (1, "寡婦、寡夫控除"),
    (1, "扶養控除"),
]


def fuzzy_match(ocr_text, expected):
    """Check if expected text appears in ocr_text with allowance for known substitutions."""
    if expected in ocr_text:
        return True
    # Apply known error substitutions in reverse (correct→wrong) to see if degraded form is present
    degraded = expected
    for wrong, correct, _ in KNOWN_ERRORS:
        degraded = degraded.replace(correct, wrong)
    return degraded in ocr_text


def analyze_ocr(stem):
    path = f"{PARSE_DIR}/{stem}_deepdoc.json"
    if not os.path.exists(path):
        print(f"  [SKIP] {stem}_deepdoc.json not found")
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    all_text = " ".join(c["text"] for c in chunks)

    result = {
        "file": stem,
        "total_chunks": len(chunks),
        "total_chars": len(all_text),
    }

    # 1. 令和 year notation error rate
    correct_years = len(REIWA_CORRECT.findall(all_text))
    wrong_years   = len(REIWA_WRONG.findall(all_text))
    all_years     = correct_years + wrong_years
    result["reiwa_correct"] = correct_years
    result["reiwa_wrong"]   = wrong_years
    result["reiwa_total"]   = all_years
    result["reiwa_error_rate"] = round(wrong_years / all_years, 3) if all_years > 0 else None

    # 2. Per-page year error rate
    pages_with_year = []
    pages = {}
    for c in chunks:
        pg = c.get("page", 0)
        pages.setdefault(pg, []).append(c["text"])

    for pg, texts in sorted(pages.items()):
        page_text = " ".join(texts)
        pg_correct = len(REIWA_CORRECT.findall(page_text))
        pg_wrong   = len(REIWA_WRONG.findall(page_text))
        if pg_correct + pg_wrong > 0:
            pages_with_year.append({
                "page": pg,
                "correct": pg_correct,
                "wrong": pg_wrong,
                "error_rate": round(pg_wrong / (pg_correct + pg_wrong), 3),
            })
    result["pages_with_year_refs"] = pages_with_year

    # 3. Known error term frequency
    errors_found = {}
    for wrong, correct, desc in KNOWN_ERRORS:
        n = all_text.count(wrong)
        if n > 0:
            errors_found[desc] = n
    result["known_errors"] = errors_found

    return result


def analyze_tsr(stem):
    """Compare TSR extracted cells against ground truth for sample 01."""
    path = f"{PARSE_DIR}/{stem}_deepdoc.json"
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tables = data.get("tables", [])
    chunks = data.get("chunks", [])

    # For sample 01 the main text flow is almost empty (complex form fonts);
    # the actual content is in the table objects. Merge all text sources.
    all_table_text = " ".join(t.get("text", "") for t in tables)

    all_text_by_page = {}
    # Populate with table text on page 0 as fallback
    if all_table_text:
        all_text_by_page[0] = [all_table_text]
        all_text_by_page[1] = [all_table_text]

    for c in chunks:
        pg = c.get("page", 0)
        all_text_by_page.setdefault(pg, []).append(c["text"])

    if not all_text_by_page:
        return None

    matched = 0
    partial = 0
    missed = 0
    details = []

    for (page, expected) in TSR_GROUND_TRUTH:
        page_texts = " ".join(all_text_by_page.get(page, []))
        if expected in page_texts:
            matched += 1
            status = "exact"
        elif fuzzy_match(page_texts, expected):
            partial += 1
            status = "fuzzy"
        else:
            missed += 1
            status = "miss"
        details.append({"page": page, "expected": expected, "status": status})

    total = len(TSR_GROUND_TRUTH)
    return {
        "file": stem,
        "gt_cells": total,
        "exact_match": matched,
        "fuzzy_match": partial,
        "miss": missed,
        "exact_rate": round(matched / total, 3),
        "exact_or_fuzzy_rate": round((matched + partial) / total, 3),
        "details": details,
    }


def main():
    print("=== OCR Quality Analysis ===\n")

    ocr_results = []
    for stem in SAMPLES:
        r = analyze_ocr(stem)
        if r:
            ocr_results.append(r)
            yr = r["reiwa_error_rate"]
            yr_str = f"{yr:.0%}" if yr is not None else "N/A (no year refs)"
            print(f"{stem}:")
            print(f"  Total chars: {r['total_chars']:,}")
            print(f"  令和 refs: correct={r['reiwa_correct']}, wrong(今和)={r['reiwa_wrong']}, error_rate={yr_str}")
            if r["known_errors"]:
                for k, v in r["known_errors"].items():
                    print(f"  Error '{k}': {v} occurrences")
            if r["pages_with_year_refs"]:
                for pg in r["pages_with_year_refs"]:
                    print(f"    Page {pg['page']}: {pg['wrong']}/{pg['correct']+pg['wrong']} wrong ({pg['error_rate']:.0%})")
            print()

    # TSR accuracy for sample 01 (has complex form table)
    print("\n=== TSR Accuracy (sample 01, ground-truth cells) ===\n")
    tsr_result = analyze_tsr("01_nta_kakutei_r06")
    if tsr_result:
        print(f"Ground-truth cells: {tsr_result['gt_cells']}")
        print(f"Exact match: {tsr_result['exact_match']} ({tsr_result['exact_rate']:.0%})")
        print(f"Fuzzy match (known subs): {tsr_result['fuzzy_match']}")
        print(f"Missed: {tsr_result['miss']}")
        print(f"Exact+Fuzzy rate: {tsr_result['exact_or_fuzzy_rate']:.0%}")
        print("\nDetail:")
        for d in tsr_result["details"]:
            print(f"  Page {d['page']} | {d['expected']!r:24s} → {d['status']}")

    # Save results
    out = {"ocr": ocr_results, "tsr": tsr_result}
    with open(f"{OUT_DIR}/quality_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUT_DIR}/quality_analysis.json")


if __name__ == "__main__":
    main()
