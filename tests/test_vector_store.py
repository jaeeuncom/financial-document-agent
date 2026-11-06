"""Tests for app/retrieval/vector_store.py and app/retrieval/indexer.py."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from qdrant_client import QdrantClient

import app.retrieval.vector_store as vs_mod
from app.retrieval.vector_store import (
    COLLECTION,
    VECTOR_DIM,
    _ensure_collection,
    collection_count,
    search,
    search_narrative,
    upsert_chunks,
)
from app.validation.schema import Chunk, ExtractionResult, ParsedSection


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str,
    text: str,
    company: str = "삼성전자",
    year: int = 2024,
    section: str = "사업의내용",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        company=company,
        year=year,
        doc_type="사업보고서",
        section=section,
        section_title="report-001.html",
        text=text,
        text_with_context=f"[{company} | {year} | {section}]\n{text}",
        char_count=len(text),
        chunk_index=0,
        total_chunks=1,
        rcept_no="20250310000123",
        source_url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250310000123",
    )


def _unit_vec(hot_index: int) -> list[float]:
    """One-hot vector: 1.0 at hot_index % VECTOR_DIM, 0.0 elsewhere."""
    v = [0.0] * VECTOR_DIM
    v[hot_index % VECTOR_DIM] = 1.0
    return v


@pytest.fixture(autouse=True)
def fresh_client():
    """Replace global Qdrant client with a clean in-memory instance for each test."""
    client = QdrantClient(":memory:")
    vs_mod._set_client(client)
    yield client
    vs_mod._set_client(None)


# ---------------------------------------------------------------------------
# collection creation
# ---------------------------------------------------------------------------

class TestEnsureCollection:
    def test_creates_collection_if_missing(self, fresh_client):
        assert not fresh_client.collection_exists(COLLECTION)
        _ensure_collection()
        assert fresh_client.collection_exists(COLLECTION)

    def test_idempotent(self, fresh_client):
        _ensure_collection()
        _ensure_collection()  # must not raise
        assert fresh_client.collection_exists(COLLECTION)


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------

class TestUpsertChunks:
    def test_returns_count(self, fresh_client):
        chunks = [_make_chunk("a" * 16, "텍스트 A"), _make_chunk("b" * 16, "텍스트 B")]
        vectors = [_unit_vec(0), _unit_vec(1)]
        assert upsert_chunks(chunks, vectors) == 2

    def test_points_stored_in_collection(self, fresh_client):
        upsert_chunks([_make_chunk("c" * 16, "본문")], [_unit_vec(2)])
        assert collection_count() == 1

    def test_empty_list_returns_zero(self, fresh_client):
        assert upsert_chunks([], []) == 0

    def test_upsert_is_idempotent(self, fresh_client):
        chunk = _make_chunk("d" * 16, "중복 텍스트")
        vec = _unit_vec(3)
        upsert_chunks([chunk], [vec])
        upsert_chunks([chunk], [vec])  # same ID → overwrites, not duplicates
        assert collection_count() == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_closest_chunk(self, fresh_client):
        c1 = _make_chunk("e" * 16, "반도체 사업")
        c2 = _make_chunk("f" * 16, "자동차 사업")
        upsert_chunks([c1, c2], [_unit_vec(0), _unit_vec(1)])

        results = search(_unit_vec(0), top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "e" * 16

    def test_top_k_limits_results(self, fresh_client):
        chunks = [_make_chunk(f"{i:0>16x}", f"텍스트 {i}") for i in range(1, 6)]
        vectors = [_unit_vec(i) for i in range(5)]
        upsert_chunks(chunks, vectors)

        results = search(_unit_vec(0), top_k=3)
        assert len(results) == 3

    def test_filter_by_company(self, fresh_client):
        c_samsung = _make_chunk("1" * 16, "삼성 반도체", company="삼성전자")
        c_sk = _make_chunk("2" * 16, "SK 메모리", company="SK하이닉스")
        upsert_chunks([c_samsung, c_sk], [_unit_vec(0), _unit_vec(0)])

        results = search(_unit_vec(0), top_k=5, filters={"company": "SK하이닉스"})
        assert len(results) == 1
        assert results[0].company == "SK하이닉스"

    def test_filter_by_year(self, fresh_client):
        c24 = _make_chunk("3" * 16, "2024 사업", year=2024)
        c23 = _make_chunk("4" * 16, "2023 사업", year=2023)
        upsert_chunks([c24, c23], [_unit_vec(0), _unit_vec(0)])

        results = search(_unit_vec(0), top_k=5, filters={"year": 2024})
        assert all(r.year == 2024 for r in results)
        assert len(results) == 1

    def test_filter_by_company_and_year(self, fresh_client):
        chunks = [
            _make_chunk("5" * 16, "A", company="삼성전자", year=2024),
            _make_chunk("6" * 16, "B", company="삼성전자", year=2023),
            _make_chunk("7" * 16, "C", company="SK하이닉스", year=2024),
        ]
        upsert_chunks(chunks, [_unit_vec(0)] * 3)

        results = search(_unit_vec(0), top_k=5, filters={"company": "삼성전자", "year": 2024})
        assert len(results) == 1
        assert results[0].chunk_id == "5" * 16

    def test_empty_collection_returns_empty(self, fresh_client):
        assert search(_unit_vec(0), top_k=5) == []

    def test_result_contains_expected_fields(self, fresh_client):
        c = _make_chunk("8" * 16, "결과 필드 테스트")
        upsert_chunks([c], [_unit_vec(0)])
        result = search(_unit_vec(0), top_k=1)[0]

        assert result.chunk_id == "8" * 16
        assert result.company == "삼성전자"
        assert result.year == 2024
        assert result.section == "사업의내용"
        assert result.text == "결과 필드 테스트"
        assert "dart.fss.or.kr" in result.source_url

    def test_score_in_valid_range(self, fresh_client):
        c = _make_chunk("9" * 16, "점수 테스트")
        v = _unit_vec(0)
        upsert_chunks([c], [v])
        result = search(v, top_k=1)[0]
        # Cosine similarity after normalization is in [-1, 1]; Qdrant scales to [0, 1]
        assert -1.0 - 1e-6 <= result.score <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# collection_count
# ---------------------------------------------------------------------------

class TestCollectionCount:
    def test_zero_on_empty(self, fresh_client):
        assert collection_count() == 0

    def test_count_increases_after_upsert(self, fresh_client):
        upsert_chunks(
            [_make_chunk("a0" * 8, "A"), _make_chunk("b1" * 8, "B")],
            [_unit_vec(0), _unit_vec(1)],
        )
        assert collection_count() == 2

    def test_count_with_filter(self, fresh_client):
        upsert_chunks(
            [_make_chunk("c2" * 8, "삼성", company="삼성전자"),
             _make_chunk("d3" * 8, "SK", company="SK하이닉스")],
            [_unit_vec(0), _unit_vec(0)],
        )
        assert collection_count({"company": "삼성전자"}) == 1
        assert collection_count({"company": "SK하이닉스"}) == 1
        assert collection_count() == 2


# ---------------------------------------------------------------------------
# search_narrative
# ---------------------------------------------------------------------------

class TestSearchNarrative:
    def test_calls_embed_texts_with_query(self, fresh_client):
        fake_vec = _unit_vec(0)
        upsert_chunks([_make_chunk("e4" * 8, "반도체 주요 사업")], [fake_vec])

        with patch("app.retrieval.vector_store.embed_texts", return_value=[fake_vec]) as mock_embed:
            search_narrative("반도체 사업 현황")

        mock_embed.assert_called_once_with(["반도체 사업 현황"])

    def test_company_filter_applied(self, fresh_client):
        fake_vec = _unit_vec(0)
        upsert_chunks(
            [_make_chunk("f5" * 8, "삼성 사업", company="삼성전자"),
             _make_chunk("06" * 8, "SK 사업", company="SK하이닉스")],
            [fake_vec, fake_vec],
        )

        with patch("app.retrieval.vector_store.embed_texts", return_value=[fake_vec]):
            results = search_narrative("반도체", company="삼성전자")

        assert len(results) == 1
        assert results[0].company == "삼성전자"

    def test_section_filter_applied(self, fresh_client):
        fake_vec = _unit_vec(0)
        upsert_chunks(
            [_make_chunk("17" * 8, "사업내용", section="사업의내용"),
             _make_chunk("28" * 8, "MD&A 내용", section="MD&A")],
            [fake_vec, fake_vec],
        )

        with patch("app.retrieval.vector_store.embed_texts", return_value=[fake_vec]):
            results = search_narrative("사업", section="MD&A")

        assert all(r.section == "MD&A" for r in results)

    def test_no_filter_returns_all(self, fresh_client):
        fake_vec = _unit_vec(0)
        upsert_chunks(
            [_make_chunk("39" * 8, "텍스트 1", company="삼성전자"),
             _make_chunk("4a" * 8, "텍스트 2", company="SK하이닉스")],
            [fake_vec, fake_vec],
        )

        with patch("app.retrieval.vector_store.embed_texts", return_value=[fake_vec]):
            results = search_narrative("텍스트")

        assert len(results) == 2

    def test_empty_index_returns_empty(self, fresh_client):
        fake_vec = _unit_vec(0)
        with patch("app.retrieval.vector_store.embed_texts", return_value=[fake_vec]):
            results = search_narrative("쿼리")
        assert results == []

    def test_top_k_respected(self, fresh_client):
        fake_vec = _unit_vec(0)
        chunks = [_make_chunk(f"{i:0>16x}", f"텍스트 {i}") for i in range(1, 6)]
        upsert_chunks(chunks, [fake_vec] * 5)

        with patch("app.retrieval.vector_store.embed_texts", return_value=[fake_vec]):
            results = search_narrative("텍스트", top_k=2)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# indexer integration
# ---------------------------------------------------------------------------

def _make_extraction_result(company: str = "삼성전자", year: int = 2024) -> ExtractionResult:
    section = ParsedSection(
        company=company,
        year=year,
        doc_type="사업보고서",
        section="사업의내용",
        section_title="report.html",
        text="삼성전자 사업 개요 상세 내용입니다. " * 50,
        page_or_order=1,
        rcept_no="20250310000123",
        source_url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250310000123",
        char_count=800,
    )
    return ExtractionResult(
        company=company,
        year=year,
        rcept_no="20250310000123",
        sections=[section],
        total_chars=800,
        failed_sections=[],
        extraction_ok=True,
    )


class TestIndexer:
    def test_index_extraction_result_inserts_chunks(self, fresh_client):
        from app.retrieval.indexer import index_extraction_result
        from app.document.chunker import chunk_extraction_result

        result = _make_extraction_result()
        expected_n = len(chunk_extraction_result(result))
        fake_vecs = [_unit_vec(i) for i in range(expected_n)]

        with patch("app.retrieval.indexer.embed_chunks", return_value=fake_vecs):
            n = index_extraction_result(result)

        assert n == expected_n
        assert collection_count() == n

    def test_index_skips_if_already_indexed(self, fresh_client):
        from app.retrieval.indexer import index_extraction_result
        from app.document.chunker import chunk_extraction_result

        result = _make_extraction_result()
        fake_vecs = [_unit_vec(i) for i in range(len(chunk_extraction_result(result)))]

        with patch("app.retrieval.indexer.embed_chunks", return_value=fake_vecs):
            index_extraction_result(result)       # first: inserts
            n2 = index_extraction_result(result)  # second: skips

        assert n2 == 0

    def test_index_force_reindexes(self, fresh_client):
        from app.retrieval.indexer import index_extraction_result
        from app.document.chunker import chunk_extraction_result

        result = _make_extraction_result()
        chunks = chunk_extraction_result(result)
        fake_vecs = [_unit_vec(i) for i in range(len(chunks))]

        with patch("app.retrieval.indexer.embed_chunks", return_value=fake_vecs):
            index_extraction_result(result)                    # first
            n2 = index_extraction_result(result, force=True)  # force re-index

        assert n2 == len(chunks)
        assert collection_count() == len(chunks)  # upsert is idempotent

    def test_index_all_skips_missing_parsed_data(self, fresh_client):
        from app.retrieval.indexer import index_all

        with patch("app.retrieval.indexer.load_parsed", return_value=None):
            summary = index_all(["없는회사"], [2024])

        assert summary[0]["status"] == "missing"
        assert summary[0]["chunks"] == 0

    def test_index_all_indexes_available_data(self, fresh_client):
        from app.retrieval.indexer import index_all
        from app.document.chunker import chunk_extraction_result

        result = _make_extraction_result()
        fake_vecs = [_unit_vec(i) for i in range(len(chunk_extraction_result(result)))]

        with (
            patch("app.retrieval.indexer.load_parsed", return_value=result),
            patch("app.retrieval.indexer.embed_chunks", return_value=fake_vecs),
        ):
            summary = index_all(["삼성전자"], [2024])

        assert summary[0]["status"] == "indexed"
        assert summary[0]["chunks"] > 0
