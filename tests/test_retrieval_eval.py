"""Tests for app/evaluation/retrieval_eval.py, metrics.evaluate_narrative_result,
and narrative routing in app/evaluation/evaluator.py."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from app.evaluation.retrieval_eval import (
    evaluate_retrieval,
    keyword_coverage,
    keyword_hit_rate,
    source_accuracy,
)
from app.evaluation.metrics import evaluate_narrative_result, compute_summary
from app.validation.schema import SearchResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

DART_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250310000123"


def _make_result(
    text: str,
    source_url: str = DART_URL,
    company: str = "삼성전자",
    year: int = 2024,
    section: str = "사업의내용",
) -> SearchResult:
    return SearchResult(
        chunk_id="a" * 16,
        company=company,
        year=year,
        section=section,
        text=text,
        source_url=source_url,
        score=0.85,
    )


def _make_item(
    qid: str = "nar-001",
    question: str = "삼성전자의 2024년 주요 사업 분야는?",
    must_contain: list[str] | None = None,
    expected_sources: list[str] | None = None,
    should_need_review: bool = False,
    tool_params: dict | None = None,
) -> dict:
    return {
        "id": qid,
        "category": "narrative",
        "question": question,
        "tool_params": tool_params or {"company": "삼성전자", "year": 2024, "section": "사업의내용"},
        "expected": {
            "must_contain": must_contain or ["반도체", "디스플레이"],
            "expected_sources": expected_sources or ["dart.fss.or.kr"],
            "should_need_review": should_need_review,
        },
    }


# ---------------------------------------------------------------------------
# keyword_coverage
# ---------------------------------------------------------------------------

class TestKeywordCoverage:
    def test_all_keywords_found(self):
        texts = ["삼성전자는 반도체와 디스플레이를 만든다."]
        result = keyword_coverage(texts, ["반도체", "디스플레이"])
        assert result == {"반도체": True, "디스플레이": True}

    def test_no_keywords_found(self):
        texts = ["자동차 사업에 대한 설명"]
        result = keyword_coverage(texts, ["반도체", "AI"])
        assert result == {"반도체": False, "AI": False}

    def test_partial_match(self):
        texts = ["반도체 사업 현황"]
        result = keyword_coverage(texts, ["반도체", "디스플레이"])
        assert result["반도체"] is True
        assert result["디스플레이"] is False

    def test_case_insensitive(self):
        texts = ["HBM memory technology"]
        result = keyword_coverage(texts, ["hbm", "HBM", "Memory"])
        assert all(result.values())

    def test_empty_keywords_returns_empty_dict(self):
        assert keyword_coverage(["some text"], []) == {}

    def test_empty_texts_all_false(self):
        result = keyword_coverage([], ["반도체"])
        assert result == {"반도체": False}

    def test_keyword_found_across_multiple_texts(self):
        texts = ["반도체 현황", "디스플레이 사업"]
        result = keyword_coverage(texts, ["반도체", "디스플레이"])
        assert result == {"반도체": True, "디스플레이": True}


# ---------------------------------------------------------------------------
# keyword_hit_rate
# ---------------------------------------------------------------------------

class TestKeywordHitRate:
    def test_all_found_returns_one(self):
        assert keyword_hit_rate(["반도체 디스플레이"], ["반도체", "디스플레이"]) == 1.0

    def test_none_found_returns_zero(self):
        assert keyword_hit_rate(["자동차"], ["반도체", "AI"]) == 0.0

    def test_half_found_returns_half(self):
        rate = keyword_hit_rate(["반도체 사업"], ["반도체", "AI"])
        assert rate == 0.5

    def test_empty_keywords_returns_one(self):
        assert keyword_hit_rate(["임의 텍스트"], []) == 1.0

    def test_one_of_three_keywords(self):
        rate = keyword_hit_rate(["반도체"], ["반도체", "AI", "클라우드"])
        assert abs(rate - 1 / 3) < 1e-6


# ---------------------------------------------------------------------------
# source_accuracy
# ---------------------------------------------------------------------------

class TestSourceAccuracy:
    def test_matching_domain_returns_true(self):
        results = [_make_result("text", source_url=DART_URL)]
        assert source_accuracy(results, ["dart.fss.or.kr"]) is True

    def test_non_matching_domain_returns_false(self):
        results = [_make_result("text", source_url="https://example.com")]
        assert source_accuracy(results, ["dart.fss.or.kr"]) is False

    def test_empty_expected_domains_returns_true(self):
        results = [_make_result("text")]
        assert source_accuracy(results, []) is True

    def test_empty_results_returns_false(self):
        assert source_accuracy([], ["dart.fss.or.kr"]) is False

    def test_at_least_one_match_sufficient(self):
        results = [
            _make_result("A", source_url="https://example.com"),
            _make_result("B", source_url=DART_URL),
        ]
        assert source_accuracy(results, ["dart.fss.or.kr"]) is True


# ---------------------------------------------------------------------------
# evaluate_retrieval
# ---------------------------------------------------------------------------

class TestEvaluateRetrieval:
    def test_all_keywords_found_and_correct_source_passes(self):
        item = _make_item(must_contain=["반도체", "디스플레이"])
        results = [_make_result("반도체와 디스플레이 사업 내용")]
        result = evaluate_retrieval(item, results)

        assert result["passed"] is True
        assert result["keyword_hit_rate"] == 1.0
        assert result["source_ok"] is True

    def test_no_keywords_found_fails(self):
        item = _make_item(must_contain=["반도체", "AI"])
        results = [_make_result("자동차 사업에 대한 설명")]
        result = evaluate_retrieval(item, results)

        assert result["passed"] is False
        assert result["keyword_hit_rate"] == 0.0

    def test_empty_results_returns_passed_none(self):
        item = _make_item(must_contain=["반도체"])
        result = evaluate_retrieval(item, [])

        assert result["passed"] is None
        assert result["num_results"] == 0
        assert "not yet indexed" in result["error"]

    def test_khr_below_threshold_fails(self):
        # 1 of 4 keywords found → khr = 0.25 < 0.5
        item = _make_item(must_contain=["반도체", "AI", "클라우드", "HBM"])
        results = [_make_result("반도체 사업")]
        result = evaluate_retrieval(item, results)

        assert result["passed"] is False
        assert result["keyword_hit_rate"] == 0.25

    def test_wrong_source_fails_even_with_keywords(self):
        item = _make_item(must_contain=["반도체"], expected_sources=["dart.fss.or.kr"])
        results = [_make_result("반도체 사업", source_url="https://example.com")]
        result = evaluate_retrieval(item, results)

        assert result["passed"] is False
        assert result["source_ok"] is False
        assert result["keyword_hit_rate"] == 1.0  # keywords found but source wrong

    def test_result_includes_coverage_detail(self):
        item = _make_item(must_contain=["반도체", "AI"])
        results = [_make_result("반도체 사업 내용")]
        result = evaluate_retrieval(item, results)

        assert result["keyword_coverage"]["반도체"] is True
        assert result["keyword_coverage"]["AI"] is False

    def test_num_results_matches_input(self):
        item = _make_item(must_contain=[])
        results = [_make_result("A"), _make_result("B"), _make_result("C")]
        result = evaluate_retrieval(item, results)
        assert result["num_results"] == 3

    def test_half_keywords_exactly_at_threshold_passes(self):
        item = _make_item(must_contain=["반도체", "디스플레이"])
        results = [_make_result("반도체 사업")]  # only 1/2 → khr=0.5 → passes
        result = evaluate_retrieval(item, results)
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# evaluate_narrative_result (from metrics.py)
# ---------------------------------------------------------------------------

class TestEvaluateNarrativeResult:
    def test_keywords_in_answer_text_passes(self):
        item = _make_item(must_contain=["반도체", "디스플레이"])
        results = [_make_result("반도체와 디스플레이 관련 내용")]
        result = evaluate_narrative_result(
            item, results, answer_text="삼성전자는 반도체와 디스플레이 사업을 운영합니다."
        )
        assert result["passed"] is True
        assert result["keyword_hit_rate"] == 1.0

    def test_missing_keywords_in_answer_fails(self):
        item = _make_item(must_contain=["반도체", "디스플레이"])
        results = [_make_result("관련 내용")]
        result = evaluate_narrative_result(item, results, answer_text="자동차 사업 설명")
        assert result["passed"] is False

    def test_falls_back_to_search_results_when_answer_empty(self):
        item = _make_item(must_contain=["반도체"])
        results = [_make_result("반도체 사업 현황")]
        result = evaluate_narrative_result(item, results, answer_text="")
        # keyword found in search result text
        assert result["keyword_coverage"]["반도체"] is True

    def test_should_need_review_false_on_success(self):
        item = _make_item(must_contain=["반도체"], should_need_review=False)
        results = [_make_result("반도체")]
        result = evaluate_narrative_result(item, results, answer_text="반도체 사업")
        assert result["review_correct"] is True

    def test_source_check_uses_search_results(self):
        item = _make_item(must_contain=[], expected_sources=["dart.fss.or.kr"])
        results = [_make_result("내용", source_url="https://example.com")]
        result = evaluate_narrative_result(item, results, answer_text="내용")
        assert result["source_ok"] is False


# ---------------------------------------------------------------------------
# compute_summary with narrative results
# ---------------------------------------------------------------------------

class TestComputeSummaryWithNarrative:
    def test_avg_keyword_hit_rate_included(self):
        results = [
            {"passed": True, "source_ok": True, "keyword_hit_rate": 1.0},
            {"passed": False, "source_ok": True, "keyword_hit_rate": 0.5},
        ]
        summary = compute_summary(results)
        assert summary["avg_keyword_hit_rate"] == 75.0

    def test_avg_keyword_hit_rate_none_when_no_narrative(self):
        results = [
            {"passed": True, "source_ok": True},  # numeric result — no khr key
        ]
        summary = compute_summary(results)
        assert summary["avg_keyword_hit_rate"] is None

    def test_mixed_numeric_and_narrative(self):
        results = [
            {"passed": True, "source_ok": True},                        # numeric
            {"passed": True, "source_ok": True, "keyword_hit_rate": 0.8},   # narrative
            {"passed": False, "source_ok": False, "keyword_hit_rate": 0.0}, # narrative
        ]
        summary = compute_summary(results)
        assert summary["total"] == 3
        assert summary["passed"] == 2
        assert summary["avg_keyword_hit_rate"] == 40.0  # (0.8+0.0)/2*100


# ---------------------------------------------------------------------------
# _run_narrative in evaluator (mocked vector store)
# ---------------------------------------------------------------------------

class TestRunNarrative:
    def test_calls_search_narrative_with_tool_params(self):
        from app.evaluation.evaluator import _run_narrative

        item = _make_item(
            must_contain=["반도체"],
            tool_params={"company": "삼성전자", "year": 2024, "section": "사업의내용"},
        )
        fake_results = [_make_result("반도체 사업 내용")]

        with patch("app.retrieval.vector_store.search_narrative", return_value=fake_results) as mock_fn:
            _run_narrative(item)

        mock_fn.assert_called_once_with(
            query=item["question"],
            company="삼성전자",
            year=2024,
            section="사업의내용",
            top_k=5,
        )

    def test_passes_when_keywords_found(self):
        from app.evaluation.evaluator import _run_narrative

        item = _make_item(must_contain=["반도체"])
        fake_results = [_make_result("반도체 관련 사업 내용입니다.")]

        with patch("app.retrieval.vector_store.search_narrative", return_value=fake_results):
            result = _run_narrative(item)

        assert result["passed"] is True

    def test_passed_none_when_no_results(self):
        from app.evaluation.evaluator import _run_narrative

        item = _make_item(must_contain=["반도체"])

        with patch("app.retrieval.vector_store.search_narrative", return_value=[]):
            result = _run_narrative(item)

        assert result["passed"] is None
        assert "not yet indexed" in result["error"]

    def test_passed_false_on_exception(self):
        from app.evaluation.evaluator import _run_narrative

        item = _make_item(must_contain=["반도체"])

        with patch("app.retrieval.vector_store.search_narrative", side_effect=RuntimeError("연결 오류")):
            result = _run_narrative(item)

        assert result["passed"] is False
        assert "연결 오류" in result["error"]

    def test_run_eval_routes_narrative_to_run_narrative(self):
        from app.evaluation.evaluator import run_eval

        items = [_make_item(qid="nar-001", must_contain=["반도체"])]
        fake_results = [_make_result("반도체 사업")]

        with patch("app.retrieval.vector_store.search_narrative", return_value=fake_results):
            results = run_eval(items)

        assert results[0]["id"] == "nar-001"
        assert results[0]["passed"] is True

    def test_run_eval_dry_run_skips_narrative(self):
        from app.evaluation.evaluator import run_eval

        items = [_make_item(qid="nar-999")]
        results = run_eval(items, dry_run=True)
        assert results[0]["passed"] is None
        assert results[0]["error"] == "dry-run"
