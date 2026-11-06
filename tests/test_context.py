"""Tests for app/retrieval/context.py — deduplicate, trim_to_budget, format_context."""
from __future__ import annotations

import pytest

from app.retrieval.context import deduplicate, format_context, trim_to_budget
from app.validation.schema import SearchResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

DART_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250310000123"


def _make_result(
    chunk_id: str,
    text: str = "기본 텍스트",
    company: str = "삼성전자",
    year: int = 2024,
    section: str = "사업의내용",
    source_url: str = DART_URL,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        company=company,
        year=year,
        section=section,
        text=text,
        source_url=source_url,
        score=0.85,
    )


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_empty_list_returns_empty(self):
        assert deduplicate([]) == []

    def test_no_duplicates_returns_all(self):
        results = [_make_result("a" * 16), _make_result("b" * 16), _make_result("c" * 16)]
        assert len(deduplicate(results)) == 3

    def test_duplicate_chunk_id_removed(self):
        results = [
            _make_result("a" * 16, text="첫 번째"),
            _make_result("b" * 16, text="두 번째"),
            _make_result("a" * 16, text="중복"),  # duplicate of first
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 2

    def test_first_occurrence_kept(self):
        results = [
            _make_result("a" * 16, text="원본"),
            _make_result("a" * 16, text="중복"),
        ]
        deduped = deduplicate(results)
        assert deduped[0].text == "원본"

    def test_original_order_preserved(self):
        results = [_make_result(f"{i:0>16x}") for i in [3, 1, 2]]
        deduped = deduplicate(results)
        ids = [r.chunk_id for r in deduped]
        assert ids == [f"{i:0>16x}" for i in [3, 1, 2]]

    def test_all_duplicates_keeps_one(self):
        results = [_make_result("a" * 16)] * 5
        assert len(deduplicate(results)) == 1

    def test_does_not_mutate_input(self):
        results = [_make_result("a" * 16), _make_result("a" * 16)]
        original_len = len(results)
        deduplicate(results)
        assert len(results) == original_len


# ---------------------------------------------------------------------------
# trim_to_budget
# ---------------------------------------------------------------------------

class TestTrimToBudget:
    def test_empty_list_returns_empty(self):
        assert trim_to_budget([], max_chars=1000) == []

    def test_all_fit_within_budget_returns_all(self):
        results = [_make_result(f"{i:0>16x}", text="짧은 텍스트") for i in range(3)]
        trimmed = trim_to_budget(results, max_chars=10000)
        assert len(trimmed) == 3

    def test_exceeds_budget_truncates(self):
        results = [_make_result(f"{i:0>16x}", text="A" * 500) for i in range(5)]
        trimmed = trim_to_budget(results, max_chars=1200)  # fits 2 × 500 = 1000, 3rd would exceed
        assert len(trimmed) == 2

    def test_always_includes_first_result(self):
        # First result alone exceeds budget
        results = [_make_result("a" * 16, text="X" * 2000)]
        trimmed = trim_to_budget(results, max_chars=100)
        assert len(trimmed) == 1
        assert trimmed[0].chunk_id == "a" * 16

    def test_exact_budget_boundary(self):
        # Two results that together exactly hit the budget
        results = [
            _make_result("a" * 16, text="A" * 500),
            _make_result("b" * 16, text="B" * 500),
        ]
        trimmed = trim_to_budget(results, max_chars=1000)
        assert len(trimmed) == 2

    def test_one_over_boundary_stops_at_second(self):
        results = [
            _make_result("a" * 16, text="A" * 500),
            _make_result("b" * 16, text="B" * 500),
            _make_result("c" * 16, text="C" * 500),
        ]
        # max=1001: first (500) fits, second (500) fits (total=1000 ≤ 1001),
        # third would make 1500 > 1001 → stop
        trimmed = trim_to_budget(results, max_chars=1001)
        assert len(trimmed) == 2

    def test_original_order_preserved(self):
        results = [_make_result(f"{i:0>16x}", text="짧음") for i in range(5)]
        trimmed = trim_to_budget(results, max_chars=10000)
        for original, kept in zip(results, trimmed):
            assert original.chunk_id == kept.chunk_id

    def test_does_not_mutate_input(self):
        results = [_make_result("a" * 16, text="A" * 500)]
        trim_to_budget(results, max_chars=100)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_empty_returns_no_results_message(self):
        assert format_context([]) == "(검색 결과 없음)"

    def test_single_result_numbered_one(self):
        result = _make_result("a" * 16, text="반도체 사업 현황")
        output = format_context([result])
        assert "[1]" in output
        assert "반도체 사업 현황" in output

    def test_multiple_results_numbered_sequentially(self):
        results = [_make_result(f"{i:0>16x}", text=f"텍스트 {i}") for i in range(1, 4)]
        output = format_context(results)
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output

    def test_includes_company_and_year(self):
        result = _make_result("a" * 16, company="SK하이닉스", year=2023)
        output = format_context([result])
        assert "SK하이닉스" in output
        assert "2023" in output

    def test_includes_section(self):
        result = _make_result("a" * 16, section="MD&A")
        output = format_context([result])
        assert "MD&A" in output

    def test_includes_source_url(self):
        result = _make_result("a" * 16, source_url=DART_URL)
        output = format_context([result])
        assert DART_URL in output

    def test_includes_text_content(self):
        result = _make_result("a" * 16, text="고유한 사업 설명 내용")
        output = format_context([result])
        assert "고유한 사업 설명 내용" in output

    def test_multiple_results_separated_by_rule(self):
        results = [_make_result(f"{i:0>16x}") for i in range(2)]
        output = format_context(results)
        assert "---" in output

    def test_no_separator_for_single_result(self):
        result = _make_result("a" * 16)
        output = format_context([result])
        assert "---" not in output

    def test_all_texts_present_in_output(self):
        texts = ["첫 번째 내용", "두 번째 내용", "세 번째 내용"]
        results = [_make_result(f"{i:0>16x}", text=t) for i, t in enumerate(texts)]
        output = format_context(results)
        for t in texts:
            assert t in output
