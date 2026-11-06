"""
Context window management utilities for RAG.

Three responsibilities:
  deduplicate   — remove results with the same chunk_id (can occur with multi-query)
  trim_to_budget — cap total text length to avoid blowing up the LLM context window
  format_context — render results as a numbered block ready for a system/user prompt
"""
from __future__ import annotations

from app.validation.schema import SearchResult


def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate chunks (same chunk_id), keeping first occurrence."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if r.chunk_id not in seen:
            seen.add(r.chunk_id)
            out.append(r)
    return out


def trim_to_budget(
    results: list[SearchResult],
    max_chars: int = 3000,
) -> list[SearchResult]:
    """
    Select results whose combined text fits within max_chars.
    Always returns at least the first result (even if it already exceeds the budget),
    so the caller is never handed an empty list when results are available.
    """
    if not results:
        return []

    selected: list[SearchResult] = []
    total = 0
    for r in results:
        if total + len(r.text) > max_chars:
            if not selected:
                selected.append(r)  # guarantee at least one
            break
        selected.append(r)
        total += len(r.text)
    return selected


def format_context(results: list[SearchResult]) -> str:
    """
    Render search results as a numbered context block for LLM prompts.

    Each entry shows: index, company + year + section, source URL, and text.
    Multiple entries are separated by a horizontal rule.
    """
    if not results:
        return "(검색 결과 없음)"

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r.company} {r.year}년 {r.section}\n"
            f"출처: {r.source_url}\n"
            f"{r.text}"
        )
    return "\n\n---\n\n".join(parts)
