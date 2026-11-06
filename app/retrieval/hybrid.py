"""
Hybrid search reranking: dense cosine similarity + BM25 keyword matching.

Strategy: Reciprocal Rank Fusion (RRF)
  final_score = 1/(k + dense_rank + 1) + 1/(k + bm25_rank + 1)

RRF avoids score normalization — it only uses rank positions.
Dense search captures semantic similarity; BM25 captures exact term matches.
k=60 is the constant from the original RRF paper (Cormack et al., 2009).

Usage:
    from app.retrieval.hybrid import hybrid_rerank
    results = hybrid_rerank(query, dense_results, top_k=5)
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from app.validation.schema import SearchResult

RRF_K = 60  # constant from the original RRF paper


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Whitespace tokenizer. Works for Korean financial text because
    Korean compounds are separated by spaces in business writing.
    """
    return text.lower().split()


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def bm25_scores(query: str, texts: list[str]) -> list[float]:
    """Return raw BM25 scores for each text against the query."""
    if not texts:
        return []
    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return list(bm25.get_scores(_tokenize(query)))


def bm25_ranks(query: str, texts: list[str]) -> list[int]:
    """
    Return 0-indexed rank for each text in original order.
    rank[i] == r means r texts have a higher BM25 score than texts[i].
    """
    scores = bm25_scores(query, texts)
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(texts)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


# ---------------------------------------------------------------------------
# RRF scoring
# ---------------------------------------------------------------------------

def rrf_score(dense_rank: int, keyword_rank: int, k: int = RRF_K) -> float:
    """Reciprocal rank fusion: higher score = more relevant."""
    return 1.0 / (k + dense_rank + 1) + 1.0 / (k + keyword_rank + 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_rerank(
    query: str,
    results: list[SearchResult],
    top_k: int | None = None,
) -> list[SearchResult]:
    """
    Rerank search results by combining dense rank (from cosine scores) with
    BM25 keyword rank via Reciprocal Rank Fusion.

    Returns a new list sorted by RRF score (highest first), truncated to top_k.
    Each result's .score field is updated to the RRF score.
    """
    if not results:
        return results

    texts = [r.text for r in results]
    bm25_rank_list = bm25_ranks(query, texts)

    scored: list[tuple[float, SearchResult]] = []
    for dense_rank, (result, keyword_rank) in enumerate(zip(results, bm25_rank_list)):
        combined = rrf_score(dense_rank, keyword_rank)
        scored.append((combined, result))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [
        SearchResult(**{**r.model_dump(), "score": round(s, 6)})
        for s, r in scored
    ]

    return reranked[:top_k] if top_k is not None else reranked
