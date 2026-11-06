"""
Indexing pipeline: ParsedSection → Chunk → Embedding → Qdrant

CLI:
    python -m app.retrieval.indexer                        # index all companies × years
    python -m app.retrieval.indexer --company 삼성전자 --year 2024
    python -m app.retrieval.indexer --force                # re-index even if already present
"""
from __future__ import annotations

import argparse
from typing import Optional

from app.document.chunker import chunk_extraction_result
from app.document.extractor import TARGET_COMPANIES, TARGET_YEARS, load_parsed
from app.retrieval.embedding import embed_chunks
from app.retrieval.vector_store import collection_count, upsert_chunks
from app.validation.schema import ExtractionResult


def index_extraction_result(result: ExtractionResult, force: bool = False) -> int:
    """
    Chunk → embed → upsert one ExtractionResult.
    Skips if already indexed (use force=True to overwrite).
    Returns number of chunks written (0 if skipped).
    """
    filters = {"company": result.company, "year": result.year}
    if not force and collection_count(filters) > 0:
        print(f"  skip {result.company} {result.year} (already indexed; use --force to re-index)")
        return 0

    chunks = chunk_extraction_result(result)
    if not chunks:
        print(f"  skip {result.company} {result.year} (no chunks produced)")
        return 0

    vectors = embed_chunks(chunks, show_progress=False)
    n = upsert_chunks(chunks, vectors)
    return n


def index_all(
    companies: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
    force: bool = False,
) -> list[dict]:
    """
    Index all (company, year) combinations.
    Skips pairs without parsed data — run app.document.extractor first.
    """
    companies = companies or TARGET_COMPANIES
    years = years or TARGET_YEARS

    summary = []
    for company in companies:
        for year in years:
            result = load_parsed(company, year)
            if result is None:
                print(f"  no parsed data for {company} {year} — run app.document.extractor first")
                summary.append({"company": company, "year": year, "status": "missing", "chunks": 0})
                continue

            n = index_extraction_result(result, force=force)
            status = "indexed" if n > 0 else "skipped"
            if n > 0:
                print(f"  OK {company} {year}: {n} chunks")
            summary.append({"company": company, "year": year, "status": status, "chunks": n})

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index parsed documents into Qdrant")
    parser.add_argument("--company", type=str, default=None, help="Single company name")
    parser.add_argument("--year", type=int, default=None, help="Single year")
    parser.add_argument("--force", action="store_true", help="Re-index even if already present")
    args = parser.parse_args()

    companies = [args.company] if args.company else None
    years = [args.year] if args.year else None

    results = index_all(companies, years, force=args.force)
    total_chunks = sum(r["chunks"] for r in results)
    n_indexed = sum(1 for r in results if r["status"] == "indexed")
    n_missing = sum(1 for r in results if r["status"] == "missing")

    print(f"\n완료: {n_indexed}개 인덱싱, {n_missing}개 누락, 총 청크 {total_chunks}개")
