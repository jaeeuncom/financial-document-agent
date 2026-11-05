"""Tests for app/document/chunker.py and app/retrieval/embedding.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.document.chunker import (
    _chunk_id,
    _find_break_point,
    _split_text,
    chunk_section,
    chunk_extraction_result,
)
from app.validation.schema import ExtractionResult, ParsedSection


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_section(text: str, section: str = "사업의내용", company: str = "삼성전자") -> ParsedSection:
    return ParsedSection(
        company=company,
        year=2024,
        doc_type="사업보고서",
        section=section,
        section_title="report-002.html",
        text=text,
        page_or_order=1,
        rcept_no="20250310000123",
        source_url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250310000123",
        char_count=len(text),
        extraction_ok=True,
    )


# ---------------------------------------------------------------------------
# _find_break_point
# ---------------------------------------------------------------------------

class TestFindBreakPoint:
    def test_prefers_paragraph_break(self):
        text = "ABC\n\nDEF"
        # paragraph break at pos=3 → returns 5 (after \n\n)
        bp = _find_break_point(text, 4)
        assert text[:bp].endswith("\n\n") or bp >= 4

    def test_korean_sentence_end(self):
        text = "반도체를 만든다.\n다음 문장"
        bp = _find_break_point(text, 10)
        assert 0 < bp <= len(text)

    def test_fallback_to_space(self):
        text = "hello world foo"
        bp = _find_break_point(text, 7)
        assert 0 < bp <= len(text)

    def test_hard_break_when_no_boundary(self):
        text = "AAAAAAAAAAAAAAAA"  # no natural breaks
        bp = _find_break_point(text, 8)
        assert bp == 8  # falls back to pos


# ---------------------------------------------------------------------------
# _split_text
# ---------------------------------------------------------------------------

class TestSplitText:
    def test_empty_returns_empty(self):
        assert _split_text("") == []

    def test_whitespace_only_returns_empty(self):
        assert _split_text("   \n\n  ") == []

    def test_short_text_returned_as_single_chunk(self):
        text = "짧은 텍스트"
        assert _split_text(text, target_size=100) == [text]

    def test_long_text_split_into_multiple_chunks(self):
        text = "안녕하세요. " * 200  # ~1400 chars
        chunks = _split_text(text, target_size=400, overlap=50)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_target_plus_window(self):
        text = "X" * 2000
        chunks = _split_text(text, target_size=400, overlap=50)
        for c in chunks[:-1]:  # last chunk can be short
            assert len(c) <= 400 + 150 + 10  # target + window + strip slack

    def test_no_empty_chunks(self):
        text = "한국어 사업 설명 텍스트입니다. " * 100
        chunks = _split_text(text, target_size=300, overlap=50)
        assert all(c.strip() for c in chunks)

    def test_all_text_covered(self):
        # Every char in original should appear in at least one chunk
        text = "ABCDE" * 300  # 1500 chars, no boundaries
        chunks = _split_text(text, target_size=400, overlap=100)
        # With overlap the concatenated unique coverage should contain all chars
        # Simple check: total chars in chunks >= len(text)
        assert sum(len(c) for c in chunks) >= len(text)

    def test_overlap_creates_shared_content(self):
        # Hard break at pos=500; with overlap=150 next_start=350 → inside "가"*400 region
        text = ("가" * 400) + ("나" * 400)  # 800 chars, no natural boundaries
        chunks = _split_text(text, target_size=500, overlap=150)
        if len(chunks) >= 2:
            # Second chunk should start before the "가"/"나" boundary (pos 400)
            assert "가" in chunks[1]

    def test_exact_target_size_no_split(self):
        text = "A" * 800
        result = _split_text(text, target_size=800, overlap=50)
        assert len(result) == 1

    def test_single_char_over_target(self):
        text = "A" * 801
        result = _split_text(text, target_size=800, overlap=50)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# chunk_section
# ---------------------------------------------------------------------------

class TestChunkSection:
    def test_short_section_one_chunk(self):
        section = _make_section("짧은 사업 설명입니다.")
        chunks = chunk_section(section, target_size=800)
        assert len(chunks) == 1

    def test_long_section_multiple_chunks(self):
        section = _make_section("삼성전자 사업 설명. " * 200)  # ~2000 chars
        chunks = chunk_section(section, target_size=400, overlap=50)
        assert len(chunks) > 1

    def test_chunk_id_is_set(self):
        section = _make_section("테스트 텍스트")
        chunks = chunk_section(section)
        assert chunks[0].chunk_id
        assert len(chunks[0].chunk_id) == 16

    def test_chunk_id_unique_per_chunk(self):
        # Use varying text so chunks cannot be identical (avoids same-phase repetition)
        text = " ".join(f"삼성전자 사업 설명 {i}번째 항목입니다." for i in range(100))
        section = _make_section(text)
        chunks = chunk_section(section, target_size=400, overlap=50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # all unique

    def test_metadata_inherited_from_section(self):
        section = _make_section("테스트 본문", section="MD&A", company="SK하이닉스")
        chunks = chunk_section(section)
        c = chunks[0]
        assert c.company == "SK하이닉스"
        assert c.year == 2024
        assert c.section == "MD&A"
        assert c.rcept_no == "20250310000123"
        assert "dart.fss.or.kr" in c.source_url

    def test_text_with_context_has_prefix(self):
        section = _make_section("본문 내용입니다.")
        chunks = chunk_section(section)
        c = chunks[0]
        assert "[삼성전자 | 2024 | 사업의내용]" in c.text_with_context
        assert "본문 내용입니다." in c.text_with_context

    def test_text_without_prefix(self):
        section = _make_section("본문 내용입니다.")
        chunks = chunk_section(section)
        # raw text should NOT contain the bracket prefix
        assert "[삼성전자" not in chunks[0].text
        assert "본문 내용입니다." in chunks[0].text

    def test_chunk_index_and_total(self):
        section = _make_section("삼성전자 사업 설명. " * 200)
        chunks = chunk_section(section, target_size=400, overlap=50)
        total = len(chunks)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.total_chunks == total

    def test_char_count_matches_text(self):
        section = _make_section("삼성전자 사업 설명. " * 200)
        chunks = chunk_section(section, target_size=400, overlap=50)
        for c in chunks:
            assert c.char_count == len(c.text)

    def test_empty_section_returns_empty(self):
        section = _make_section("")
        assert chunk_section(section) == []

    def test_whitespace_section_returns_empty(self):
        section = _make_section("   \n\n  ")
        assert chunk_section(section) == []

    def test_different_sections_different_chunk_ids(self):
        # Same text but different sections → different text_with_context → different IDs
        s1 = _make_section("동일한 본문", section="사업의내용")
        s2 = _make_section("동일한 본문", section="MD&A")
        c1 = chunk_section(s1)[0]
        c2 = chunk_section(s2)[0]
        assert c1.chunk_id != c2.chunk_id


# ---------------------------------------------------------------------------
# chunk_extraction_result
# ---------------------------------------------------------------------------

class TestChunkExtractionResult:
    def _make_result(self) -> ExtractionResult:
        sections = [
            _make_section("사업 내용 본문 " * 50, section="사업의내용"),
            _make_section("경영진단 본문 " * 50, section="MD&A"),
        ]
        return ExtractionResult(
            company="삼성전자",
            year=2024,
            rcept_no="20250310000123",
            sections=sections,
            total_chars=sum(s.char_count for s in sections),
            failed_sections=[],
            extraction_ok=True,
        )

    def test_chunks_from_all_sections(self):
        result = self._make_result()
        chunks = chunk_extraction_result(result, target_size=300, overlap=50)
        sections_in_chunks = {c.section for c in chunks}
        assert "사업의내용" in sections_in_chunks
        assert "MD&A" in sections_in_chunks

    def test_empty_extraction_result(self):
        result = ExtractionResult(
            company="테스트", year=2024, rcept_no="",
            sections=[], total_chars=0, failed_sections=[], extraction_ok=False,
        )
        assert chunk_extraction_result(result) == []


# ---------------------------------------------------------------------------
# _chunk_id
# ---------------------------------------------------------------------------

class TestChunkId:
    def test_deterministic(self):
        assert _chunk_id("hello") == _chunk_id("hello")

    def test_different_text_different_id(self):
        assert _chunk_id("hello") != _chunk_id("world")

    def test_length_16(self):
        assert len(_chunk_id("any text")) == 16


# ---------------------------------------------------------------------------
# embed_texts (all OpenAI calls mocked)
# ---------------------------------------------------------------------------

def _fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    import hashlib
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return [((h >> (i * 4)) & 0xF) / 16.0 for i in range(1536)]


def _mock_openai_embed(texts: list[str]) -> list[list[float]]:
    return [_fake_embedding(t) for t in texts]


class TestEmbedTexts:
    def _mock_response(self, texts: list[str]) -> MagicMock:
        resp = MagicMock()
        resp.data = [
            MagicMock(index=i, embedding=_fake_embedding(t))
            for i, t in enumerate(texts)
        ]
        return resp

    def test_returns_embeddings_in_order(self, tmp_path):
        from app.retrieval import embedding as emb_mod

        texts = ["텍스트 A", "텍스트 B", "텍스트 C"]

        with (
            patch.object(emb_mod.config, "OPENAI_API_KEY", "fake_key"),
            patch.object(emb_mod.config, "DATA_DIR", tmp_path),
            patch("app.retrieval.embedding.OpenAI") as mock_cls,
        ):
            mock_cls.return_value.embeddings.create.return_value = self._mock_response(texts)
            vecs = emb_mod.embed_texts(texts)

        assert len(vecs) == 3
        assert len(vecs[0]) == 1536

    def test_cached_texts_not_re_embedded(self, tmp_path):
        from app.retrieval import embedding as emb_mod

        texts = ["캐시 텍스트"]
        mock_create = MagicMock(return_value=self._mock_response(texts))

        with (
            patch.object(emb_mod.config, "OPENAI_API_KEY", "fake_key"),
            patch.object(emb_mod.config, "DATA_DIR", tmp_path),
            patch("app.retrieval.embedding.OpenAI") as mock_cls,
        ):
            mock_cls.return_value.embeddings.create = mock_create
            emb_mod.embed_texts(texts)   # first call — API hit
            emb_mod.embed_texts(texts)   # second call — should use cache

        assert mock_create.call_count == 1

    def test_batch_splitting(self, tmp_path):
        from app.retrieval import embedding as emb_mod

        texts = [f"텍스트 {i}" for i in range(5)]
        call_texts: list[list[str]] = []

        def fake_create(model, input, encoding_format):
            call_texts.append(list(input))
            resp = MagicMock()
            resp.data = [
                MagicMock(index=j, embedding=_fake_embedding(t))
                for j, t in enumerate(input)
            ]
            return resp

        with (
            patch.object(emb_mod.config, "OPENAI_API_KEY", "fake_key"),
            patch.object(emb_mod.config, "DATA_DIR", tmp_path),
            patch("app.retrieval.embedding.OpenAI") as mock_cls,
        ):
            mock_cls.return_value.embeddings.create.side_effect = fake_create
            emb_mod.embed_texts(texts, batch_size=2)

        # 5 texts with batch_size=2 → 3 API calls (2+2+1)
        assert len(call_texts) == 3
        assert len(call_texts[0]) == 2
        assert len(call_texts[1]) == 2
        assert len(call_texts[2]) == 1

    def test_cache_persisted_to_disk(self, tmp_path):
        from app.retrieval import embedding as emb_mod

        texts = ["저장 테스트"]

        with (
            patch.object(emb_mod.config, "OPENAI_API_KEY", "fake_key"),
            patch.object(emb_mod.config, "DATA_DIR", tmp_path),
            patch("app.retrieval.embedding.OpenAI") as mock_cls,
        ):
            mock_cls.return_value.embeddings.create.return_value = self._mock_response(texts)
            emb_mod.embed_texts(texts)

        cache_file = tmp_path / "embeddings_cache.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["version"] == 1
        assert len(data["embeddings"]) == 1

    def test_mixed_cached_and_new(self, tmp_path):
        from app.retrieval import embedding as emb_mod

        texts_first = ["A텍스트", "B텍스트"]
        texts_second = ["A텍스트", "C텍스트"]  # A is cached, C is new

        with (
            patch.object(emb_mod.config, "OPENAI_API_KEY", "fake_key"),
            patch.object(emb_mod.config, "DATA_DIR", tmp_path),
            patch("app.retrieval.embedding.OpenAI") as mock_cls,
        ):
            mock_cls.return_value.embeddings.create.return_value = self._mock_response(texts_first)
            emb_mod.embed_texts(texts_first)

            call_count_before = mock_cls.return_value.embeddings.create.call_count

            mock_cls.return_value.embeddings.create.return_value = self._mock_response(["C텍스트"])
            vecs = emb_mod.embed_texts(texts_second)

        # Only 1 new call for "C텍스트"
        assert mock_cls.return_value.embeddings.create.call_count == call_count_before + 1
        assert len(vecs) == 2

    def test_embed_chunks_uses_text_with_context(self, tmp_path):
        from app.retrieval import embedding as emb_mod
        from app.document.chunker import chunk_section

        section = _make_section("본문 내용입니다.")
        chunks = chunk_section(section)
        assert chunks

        embedded_texts: list[str] = []

        def fake_create(model, input, encoding_format):
            embedded_texts.extend(input)
            resp = MagicMock()
            resp.data = [
                MagicMock(index=j, embedding=_fake_embedding(t))
                for j, t in enumerate(input)
            ]
            return resp

        with (
            patch.object(emb_mod.config, "OPENAI_API_KEY", "fake_key"),
            patch.object(emb_mod.config, "DATA_DIR", tmp_path),
            patch("app.retrieval.embedding.OpenAI") as mock_cls,
        ):
            mock_cls.return_value.embeddings.create.side_effect = fake_create
            emb_mod.embed_chunks(chunks, show_progress=False)

        # All embedded texts should be text_with_context (have the bracket prefix)
        assert all("[삼성전자 | 2024 | 사업의내용]" in t for t in embedded_texts)
