"""
Qdrant-backed vector store for RAG chunks.

Local persistence: data/qdrant/
Collection       : financial_docs
Vector dim       : 1536 (text-embedding-3-small)
Distance metric  : COSINE

Usage:
    from app.retrieval.vector_store import upsert_chunks, search_narrative

    upsert_chunks(chunks, vectors)
    results = search_narrative("삼성전자 반도체 사업 전략", company="삼성전자", year=2024)
"""
from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app import config
from app.retrieval.embedding import embed_texts
from app.validation.schema import Chunk, SearchResult

COLLECTION = "financial_docs"
VECTOR_DIM = 1536

_qdrant_client: Optional[QdrantClient] = None


# ---------------------------------------------------------------------------
# Client management (singleton; replaceable via _set_client for tests)
# ---------------------------------------------------------------------------

def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        config.QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(config.QDRANT_DIR))
    return _qdrant_client


def _set_client(client: Optional[QdrantClient]) -> None:
    global _qdrant_client
    _qdrant_client = client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_collection() -> None:
    client = _get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def _build_filter(filters: dict) -> Filter:
    return Filter(
        must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
    )


def _chunk_to_point(chunk: Chunk, vector: list[float]) -> PointStruct:
    return PointStruct(
        id=int(chunk.chunk_id, 16),
        vector=vector,
        payload=chunk.model_dump(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> int:
    """Insert or update chunks in the vector store. Returns number of points written."""
    if not chunks:
        return 0
    _ensure_collection()
    client = _get_client()
    points = [_chunk_to_point(c, v) for c, v in zip(chunks, vectors)]
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(
    query_vector: list[float],
    top_k: int = 5,
    filters: Optional[dict] = None,
) -> list[SearchResult]:
    """Search the vector store by cosine similarity. Optional key=value filters."""
    _ensure_collection()
    client = _get_client()
    qdrant_filter = _build_filter(filters) if filters else None

    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
    )

    return [
        SearchResult(
            chunk_id=hit.payload["chunk_id"],
            company=hit.payload["company"],
            year=hit.payload["year"],
            section=hit.payload["section"],
            text=hit.payload["text"],
            source_url=hit.payload["source_url"],
            score=hit.score,
        )
        for hit in response.points
    ]


def search_narrative(
    query: str,
    company: Optional[str] = None,
    year: Optional[int] = None,
    section: Optional[str] = None,
    top_k: int = 5,
    rerank: bool = False,
    max_context_chars: int = 3000,
) -> list[SearchResult]:
    """
    High-level search: embed query text, then search with optional metadata filters.

    When rerank=True, over-fetches (top_k * 3) from Qdrant and applies
    hybrid BM25+dense reranking (RRF) followed by deduplication and
    character-budget trimming.
    """
    vectors = embed_texts([query])
    query_vector = vectors[0]

    filters: dict = {}
    if company:
        filters["company"] = company
    if year:
        filters["year"] = year
    if section:
        filters["section"] = section

    fetch_k = top_k * 3 if rerank else top_k
    results = search(query_vector, top_k=fetch_k, filters=filters or None)

    if rerank and results:
        from app.retrieval.hybrid import hybrid_rerank
        from app.retrieval.context import deduplicate, trim_to_budget
        results = hybrid_rerank(query, results, top_k=top_k)
        results = deduplicate(results)
        results = trim_to_budget(results, max_chars=max_context_chars)

    return results


def collection_count(filters: Optional[dict] = None) -> int:
    """Return the number of indexed points, optionally filtered."""
    _ensure_collection()
    client = _get_client()
    qdrant_filter = _build_filter(filters) if filters else None
    result = client.count(
        collection_name=COLLECTION,
        count_filter=qdrant_filter,
        exact=True,
    )
    return result.count
