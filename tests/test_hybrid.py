"""Tests for app/retrieval/hybrid.py — BM25 reranking and RRF scoring."""
from __future__ import annotations

import pytest

from app.retrieval.hybrid import (
    RRF_K,
    bm25_ranks,
    bm25_scores,
    hybrid_rerank,
    rrf_score,
)
from app.validation.schema import SearchResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_result(
    chunk_id: str,
    text: str,
    score: float = 0.5,
    company: str = "삼성전자",
    year: int = 2024,
    section: str = "사업의내용",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        company=company,
        year=year,
        section=section,
        text=text,
        source_url="https://dart.fss.or.kr/test",
        score=score,
    )


# ---------------------------------------------------------------------------
# bm25_scores
# ---------------------------------------------------------------------------

class TestBm25Scores:
    def test_returns_one_score_per_text(self):
        texts = ["반도체 사업", "자동차 제조"]
        scores = bm25_scores("반도체", texts)
        assert len(scores) == 2

    def test_matching_text_scores_higher(self):
        # BM25Okapi IDF = log((N-df+0.5)/(df+0.5)); needs N >= 3 for IDF > 0 when df=1
        texts = ["반도체 HBM 메모리 사업", "자동차 전기차 판매", "소비재 화장품 뷰티"]
        scores = bm25_scores("반도체 HBM", texts)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_empty_texts_returns_empty(self):
        assert bm25_scores("반도체", []) == []

    def test_non_matching_query_gives_zero(self):
        texts = ["전혀 관련 없는 내용", "다른 내용"]
        scores = bm25_scores("완전히다른키워드ZZZZ", texts)
        assert all(s == 0.0 for s in scores)

    def test_scores_are_non_negative(self):
        texts = ["반도체 사업 내용", "디스플레이 제조"]
        scores = bm25_scores("반도체", texts)
        assert all(s >= 0.0 for s in scores)


# ---------------------------------------------------------------------------
# bm25_ranks
# ---------------------------------------------------------------------------

class TestBm25Ranks:
    def test_returns_one_rank_per_text(self):
        texts = ["A", "B", "C"]
        ranks = bm25_ranks("A", texts)
        assert len(ranks) == 3

    def test_all_ranks_unique(self):
        texts = ["반도체 HBM", "자동차 EV", "디스플레이 OLED"]
        ranks = bm25_ranks("반도체", texts)
        assert len(set(ranks)) == len(ranks)

    def test_matching_text_gets_rank_zero(self):
        texts = ["전혀 무관한 텍스트", "반도체 사업 내용입니다", "자동차 관련"]
        ranks = bm25_ranks("반도체", texts)
        assert ranks[1] == 0  # "반도체 사업 내용" should be rank 0

    def test_empty_texts_returns_empty(self):
        assert bm25_ranks("반도체", []) == []

    def test_rank_range(self):
        texts = ["A B C", "D E F", "G H I"]
        ranks = bm25_ranks("A", texts)
        assert set(ranks) == {0, 1, 2}


# ---------------------------------------------------------------------------
# rrf_score
# ---------------------------------------------------------------------------

class TestRrfScore:
    def test_top_in_both_gives_highest_score(self):
        # rank 0 in both → maximum score
        best = rrf_score(0, 0)
        # rank 1 in one → lower
        worse = rrf_score(0, 1)
        assert best > worse

    def test_symmetry(self):
        assert rrf_score(2, 5) == rrf_score(5, 2)

    def test_score_decreases_as_rank_increases(self):
        s0 = rrf_score(0, 0)
        s1 = rrf_score(1, 1)
        s10 = rrf_score(10, 10)
        assert s0 > s1 > s10

    def test_scores_are_positive(self):
        for r in range(10):
            assert rrf_score(r, r) > 0.0

    def test_uses_rrf_k_constant(self):
        expected = 2.0 / (RRF_K + 1)
        assert abs(rrf_score(0, 0) - expected) < 1e-9


# ---------------------------------------------------------------------------
# hybrid_rerank
# ---------------------------------------------------------------------------

class TestHybridRerank:
    def test_empty_results_returns_empty(self):
        assert hybrid_rerank("쿼리", []) == []

    def test_returns_same_count_by_default(self):
        results = [_make_result(f"{i:0>16x}", f"텍스트 {i}") for i in range(5)]
        reranked = hybrid_rerank("텍스트", results)
        assert len(reranked) == 5

    def test_top_k_limits_output(self):
        results = [_make_result(f"{i:0>16x}", f"텍스트 {i}") for i in range(10)]
        reranked = hybrid_rerank("텍스트", results, top_k=3)
        assert len(reranked) == 3

    def test_keyword_match_moves_up_after_rerank(self):
        # RRF blends dense rank + BM25 rank. With k=60, rank-0 in dense is strong,
        # so a keyword match won't necessarily reach position 0 if it's last in dense.
        # But a low-ranked keyword match (dense rank 4) should move up vs no-match docs.
        # Math: e (dense=4, bm25=0) scores 1/65+1/61≈0.0318, beats d (dense=3, bm25=4): 1/64+1/65≈0.0309
        results = [
            _make_result("a" * 16, "화장품 뷰티 판매 현황", score=0.95),
            _make_result("b" * 16, "자동차 전기차 생산", score=0.90),
            _make_result("c" * 16, "소비재 식품 제조", score=0.85),
            _make_result("d" * 16, "게임 엔터테인먼트", score=0.80),
            _make_result("e" * 16, "반도체 HBM 메모리 기술 개발", score=0.60),
        ]
        reranked = hybrid_rerank("반도체 HBM 메모리", results)
        original_pos = {r.chunk_id: i for i, r in enumerate(results)}
        new_pos = {r.chunk_id: i for i, r in enumerate(reranked)}
        # Keyword-matching doc (e) should move up from its original last position
        assert new_pos["e" * 16] < original_pos["e" * 16]

    def test_score_field_updated_to_rrf(self):
        results = [_make_result("a" * 16, "반도체 사업", score=0.9)]
        reranked = hybrid_rerank("반도체", results)
        # RRF score is around 1/(60+1)*2 ≈ 0.0328, not the original 0.9
        assert reranked[0].score != 0.9
        assert 0.0 < reranked[0].score < 0.1

    def test_original_results_not_mutated(self):
        results = [_make_result("a" * 16, "텍스트", score=0.9)]
        hybrid_rerank("텍스트", results, top_k=1)
        assert results[0].score == 0.9  # original unchanged

    def test_all_chunk_ids_preserved(self):
        results = [_make_result(f"{i:0>16x}", f"텍스트 {i}") for i in range(5)]
        reranked = hybrid_rerank("텍스트", results)
        original_ids = {r.chunk_id for r in results}
        reranked_ids = {r.chunk_id for r in reranked}
        assert original_ids == reranked_ids

    def test_single_result_returned_unchanged_id(self):
        results = [_make_result("a" * 16, "단일 결과")]
        reranked = hybrid_rerank("단일", results)
        assert len(reranked) == 1
        assert reranked[0].chunk_id == "a" * 16

    def test_top_k_larger_than_results_returns_all(self):
        results = [_make_result(f"{i:0>16x}", f"텍스트 {i}") for i in range(3)]
        reranked = hybrid_rerank("텍스트", results, top_k=10)
        assert len(reranked) == 3


# ---------------------------------------------------------------------------
# search_narrative with rerank=True (integration, in-memory Qdrant)
# ---------------------------------------------------------------------------

class TestSearchNarrativeWithRerank:
    def test_rerank_returns_top_k(self):
        """search_narrative(rerank=True) should return at most top_k results."""
        import app.retrieval.vector_store as vs_mod
        from app.retrieval.vector_store import upsert_chunks, search_narrative, VECTOR_DIM
        from qdrant_client import QdrantClient
        from unittest.mock import patch
        from app.validation.schema import Chunk

        def _make_chunk(chunk_id: str, text: str) -> Chunk:
            return Chunk(
                chunk_id=chunk_id, company="삼성전자", year=2024,
                doc_type="사업보고서", section="사업의내용",
                section_title="report.html", text=text,
                text_with_context=f"[삼성전자 | 2024 | 사업의내용]\n{text}",
                char_count=len(text), chunk_index=0, total_chunks=1,
                rcept_no="123", source_url="https://dart.fss.or.kr/test",
            )

        def _unit_vec(hot: int) -> list[float]:
            v = [0.0] * VECTOR_DIM
            v[hot % VECTOR_DIM] = 1.0
            return v

        client = QdrantClient(":memory:")
        vs_mod._set_client(client)
        try:
            chunks = [_make_chunk(f"{i:0>16x}", f"반도체 사업 텍스트 {i}번째 내용입니다.") for i in range(1, 7)]
            vecs = [_unit_vec(i) for i in range(6)]
            upsert_chunks(chunks, vecs)

            fake_query_vec = _unit_vec(0)
            with patch("app.retrieval.vector_store.embed_texts", return_value=[fake_query_vec]):
                results = search_narrative("반도체 사업", top_k=3, rerank=True)

            assert len(results) <= 3
        finally:
            vs_mod._set_client(None)
