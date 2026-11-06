"""
Retrieval quality metrics for RAG narrative evaluation.

Keyword Hit Rate (KHR): fraction of expected keywords found in top-k retrieved chunks.
Used as a lightweight proxy for retrieval precision without expensive LLM judges.

A question passes when KHR >= 0.5 AND retrieved source URLs match expected domains.
"""
from __future__ import annotations

from typing import Any

from app.validation.schema import SearchResult


def keyword_coverage(texts: list[str], keywords: list[str]) -> dict[str, bool]:
    """Return {keyword: found_in_any_text} for each keyword (case-insensitive)."""
    combined = " ".join(texts).lower()
    return {kw: kw.lower() in combined for kw in keywords}


def keyword_hit_rate(texts: list[str], keywords: list[str]) -> float:
    """Fraction of keywords found in texts. Returns 1.0 when keywords list is empty."""
    if not keywords:
        return 1.0
    coverage = keyword_coverage(texts, keywords)
    return sum(coverage.values()) / len(coverage)


def source_accuracy(results: list[SearchResult], expected_domains: list[str]) -> bool:
    """True if at least one expected domain appears in any result's source_url."""
    if not expected_domains:
        return True
    return any(
        domain in result.source_url
        for domain in expected_domains
        for result in results
    )


def evaluate_retrieval(
    item: dict[str, Any],
    results: list[SearchResult],
) -> dict[str, Any]:
    """
    Evaluate retrieval quality for one narrative eval question.

    Passes when keyword_hit_rate >= 0.5 AND source URLs contain expected domains.
    Returns passed=None when results are empty (documents not indexed yet).
    """
    expected = item.get("expected", {})
    must_contain: list[str] = expected.get("must_contain", [])
    expected_sources: list[str] = expected.get("expected_sources", [])

    if not results:
        return {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "passed": None,
            "keyword_hit_rate": 0.0,
            "keyword_coverage": {kw: False for kw in must_contain},
            "source_ok": False,
            "num_results": 0,
            "error": "no results — documents not yet indexed (run app.retrieval.indexer first)",
        }

    texts = [r.text for r in results]
    khr = keyword_hit_rate(texts, must_contain)
    coverage = keyword_coverage(texts, must_contain)
    src_ok = source_accuracy(results, expected_sources)

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "passed": khr >= 0.5 and src_ok,
        "keyword_hit_rate": round(khr, 3),
        "keyword_coverage": coverage,
        "source_ok": src_ok,
        "num_results": len(results),
        "error": None,
    }
