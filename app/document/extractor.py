"""
Batch extraction orchestrator for DART annual reports.

CLI:
    python -m app.document.extractor                       # all 5 companies × 3 years
    python -m app.document.extractor --company 삼성전자    # one company, all years
    python -m app.document.extractor --year 2024           # all companies, one year
    python -m app.document.extractor --company 삼성전자 --year 2024
    python -m app.document.extractor --force               # ignore cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from app import config
from app.document.parser import parse_report
from app.validation.schema import ExtractionResult, ParsedSection
from app.validation.rules import ReviewRequired

TARGET_COMPANIES = ["삼성전자", "SK하이닉스", "현대자동차", "NAVER", "LG전자"]
TARGET_YEARS = [2023, 2024, 2025]
_ALL_SECTIONS = ["사업의내용", "MD&A", "연구개발", "위험요소"]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_company_year(company: str, year: int) -> ExtractionResult:
    """
    Run parse_report for one company/year and wrap in ExtractionResult.
    Extraction errors are captured, not raised — caller gets an ExtractionResult
    with extraction_ok=False and an error message.
    """
    sections: list[ParsedSection] = []
    rcept_no = ""
    error: Optional[str] = None
    extraction_ok = True

    try:
        sections = parse_report(company, year)
        if sections:
            rcept_no = sections[0].rcept_no
    except ReviewRequired as exc:
        error = f"{exc.rule}: {exc.detail}"
        extraction_ok = False
    except Exception as exc:
        error = str(exc)
        extraction_ok = False

    found = {s.section for s in sections}
    failed = [s for s in _ALL_SECTIONS if s not in found]

    return ExtractionResult(
        company=company,
        year=year,
        rcept_no=rcept_no,
        sections=sections,
        total_chars=sum(s.char_count for s in sections),
        failed_sections=failed,
        extraction_ok=extraction_ok and bool(sections),
        error=error,
    )


def save_parsed(result: ExtractionResult) -> Path:
    """Serialize ExtractionResult to data/parsed/{company}_{year}.json."""
    path = config.PARSED_DIR / f"{result.company}_{result.year}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_parsed(company: str, year: int) -> Optional[ExtractionResult]:
    """Return cached ExtractionResult, or None if not found."""
    path = config.PARSED_DIR / f"{company}_{year}.json"
    if not path.exists():
        return None
    return ExtractionResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    companies: list[str] = TARGET_COMPANIES,
    years: list[int] = TARGET_YEARS,
    force: bool = False,
) -> list[ExtractionResult]:
    """
    Extract all company/year combinations, skip cached results unless force=True.
    Logs each result line; extraction failures are printed, not raised.
    """
    results: list[ExtractionResult] = []
    total = len(companies) * len(years)
    done = 0

    for company in companies:
        for year in years:
            done += 1
            label = f"[{done}/{total}] {company} {year}"

            if not force:
                cached = load_parsed(company, year)
                if cached is not None:
                    print(f"{label} → cached ({cached.total_chars:,} chars, "
                          f"{len(cached.sections)} sections)")
                    results.append(cached)
                    continue

            print(f"{label} → extracting ...", end=" ", flush=True)
            result = extract_company_year(company, year)
            save_parsed(result)

            if result.extraction_ok:
                detail = ", ".join(
                    f"{s.section}({s.char_count:,}자)" for s in result.sections
                )
                print(f"OK [{detail}]")
            else:
                print(f"FAILED: {result.error}")
                if result.failed_sections:
                    print(f"       missing sections: {result.failed_sections}")

            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[ExtractionResult]) -> None:
    total = len(results)
    ok = sum(1 for r in results if r.extraction_ok)
    total_chars = sum(r.total_chars for r in results)

    print("\n" + "=" * 60)
    print(f"Extraction complete: {ok}/{total} OK  |  {total_chars:,} total chars")

    failures = [r for r in results if not r.extraction_ok]
    if failures:
        print("\nFailed:")
        for r in failures:
            print(f"  {r.company} {r.year}: {r.error}")

    missing = [(r.company, r.year, r.failed_sections) for r in results if r.failed_sections]
    if missing:
        print("\nMissing target sections:")
        for company, year, failed in missing:
            print(f"  {company} {year}: {', '.join(failed)}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DART annual report sections")
    parser.add_argument("--company", help="Company name (default: all 5)")
    parser.add_argument("--year", type=int, help="Fiscal year (default: 2023-2025)")
    parser.add_argument("--force", action="store_true", help="Ignore disk cache and re-extract")
    args = parser.parse_args()

    companies = [args.company] if args.company else TARGET_COMPANIES
    years = [args.year] if args.year else TARGET_YEARS

    print(f"Extracting {len(companies)} company × {len(years)} year(s)...\n")
    results = run_batch(companies, years, force=args.force)
    print_summary(results)


if __name__ == "__main__":
    main()
