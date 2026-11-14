"""Tests for Day 13: eval set expansion, report generator, evaluator --report flag."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.report import _accuracy_str, _per_category, generate_report
from app.evaluation.metrics import compute_summary

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _result(
    id: str = "fin-001",
    category: str = "numeric",
    question: str = "테스트 질문",
    passed: bool | None = True,
    source_ok: bool = True,
    error: str | None = None,
    keyword_hit_rate: float | None = None,
) -> dict:
    r: dict = {
        "id": id,
        "category": category,
        "question": question,
        "passed": passed,
        "source_ok": source_ok,
        "error": error,
    }
    if keyword_hit_rate is not None:
        r["keyword_hit_rate"] = keyword_hit_rate
    return r


def _summary(**overrides) -> dict:
    base = {
        "total": 1,
        "passed": 1,
        "accuracy": 100.0,
        "source_match_rate": 100.0,
        "avg_keyword_hit_rate": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# eval_set.json structure
# ---------------------------------------------------------------------------

class TestEvalSetStructure:
    def test_file_exists(self):
        assert EVAL_SET_PATH.exists()

    def test_at_least_30_questions(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        assert len(data) >= 30

    def test_all_have_required_fields(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        for item in data:
            assert "id" in item, f"missing id in {item}"
            assert "category" in item
            assert "question" in item
            assert "expected" in item

    def test_all_ids_unique(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        ids = [item["id"] for item in data]
        assert len(ids) == len(set(ids))

    def test_expected_has_should_need_review(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        for item in data:
            assert "should_need_review" in item["expected"], f"missing in {item['id']}"

    def test_categories_cover_required_types(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        cats = {item["category"] for item in data}
        assert "numeric" in cats
        assert "adversarial" in cats
        assert "narrative" in cats

    def test_adversarial_should_need_review_varied(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        adv = [item for item in data if item["category"] == "adversarial"]
        review_values = {item["expected"]["should_need_review"] for item in adv}
        assert True in review_values   # some must flag for review
        assert False in review_values  # some must NOT flag (counter-claim cases)

    def test_new_questions_present(self):
        data = json.loads(EVAL_SET_PATH.read_text())
        ids = {item["id"] for item in data}
        assert "fin-010" in ids
        assert "adv-004" in ids
        assert "adv-005" in ids
        assert "nar-006" in ids
        assert "cmp-002" in ids


# ---------------------------------------------------------------------------
# _per_category
# ---------------------------------------------------------------------------

class TestPerCategory:
    def test_groups_by_category(self):
        results = [
            _result("n1", "numeric", passed=True),
            _result("n2", "numeric", passed=False),
            _result("a1", "adversarial", passed=True),
        ]
        cats = _per_category(results)
        assert "numeric" in cats
        assert "adversarial" in cats
        assert cats["numeric"]["total"] == 2
        assert cats["adversarial"]["total"] == 1

    def test_counts_pass_fail_skip(self):
        results = [
            _result(passed=True),
            _result(passed=False),
            _result(passed=None),
        ]
        cats = _per_category(results)
        stats = cats["numeric"]
        assert stats["passed"] == 1
        assert stats["failed"] == 1
        assert stats["skipped"] == 1

    def test_numeric_before_narrative_in_order(self):
        results = [
            _result("n1", "narrative"),
            _result("n2", "numeric"),
        ]
        cats = _per_category(results)
        keys = list(cats.keys())
        assert keys.index("numeric") < keys.index("narrative")

    def test_unknown_category_included(self):
        results = [_result("x1", "custom_cat")]
        cats = _per_category(results)
        assert "custom_cat" in cats

    def test_empty_results_returns_empty(self):
        assert _per_category([]) == {}


# ---------------------------------------------------------------------------
# _accuracy_str
# ---------------------------------------------------------------------------

class TestAccuracyStr:
    def test_all_pass(self):
        assert _accuracy_str(5, 5, 0) == "100.0% (5/5)"

    def test_partial_pass(self):
        assert _accuracy_str(3, 5, 0) == "60.0% (3/5)"

    def test_all_skipped(self):
        assert _accuracy_str(0, 5, 5) == "— (all skipped)"

    def test_some_skipped(self):
        result = _accuracy_str(2, 5, 2)
        assert "2/3" in result  # 3 runnable

    def test_zero_total(self):
        assert _accuracy_str(0, 0, 0) == "— (all skipped)"


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_contains_title(self):
        report = generate_report([], _summary(total=0, passed=0))
        assert "Evaluation Report" in report

    def test_contains_generated_date(self):
        report = generate_report([], _summary(), generated_at="2026-01-01")
        assert "2026-01-01" in report

    def test_contains_total_questions(self):
        results = [_result(), _result("fin-002")]
        report = generate_report(results, _summary(total=2))
        assert "2" in report

    def test_contains_executive_summary_section(self):
        report = generate_report([], _summary())
        assert "Executive Summary" in report

    def test_contains_results_by_category_section(self):
        report = generate_report([_result()], _summary())
        assert "Results by Category" in report

    def test_category_summary_table_present(self):
        report = generate_report([_result()], _summary())
        assert "| numeric |" in report

    def test_failure_analysis_absent_when_no_failures(self):
        results = [_result(passed=True)]
        report = generate_report(results, _summary())
        assert "Failure Analysis" not in report

    def test_failure_analysis_present_when_failures_exist(self):
        results = [_result(passed=False, error="company_not_found")]
        report = generate_report(results, _summary(passed=0))
        assert "Failure Analysis" in report

    def test_failure_analysis_contains_error(self):
        results = [_result(passed=False, error="company_not_found: ABC")]
        report = generate_report(results, _summary(passed=0))
        assert "company_not_found" in report

    def test_skipped_note_absent_when_all_ran(self):
        results = [_result(passed=True)]
        report = generate_report(results, _summary())
        assert "Skipped Questions" not in report

    def test_skipped_note_present_when_skipped_exist(self):
        results = [_result(passed=None)]
        report = generate_report(results, _summary())
        assert "Skipped Questions" in report or "skipped" in report.lower()

    def test_how_to_run_section_present(self):
        report = generate_report([], _summary())
        assert "How to Run" in report

    def test_avg_keyword_hit_rate_included_when_set(self):
        report = generate_report(
            [_result(keyword_hit_rate=0.75)],
            _summary(avg_keyword_hit_rate=75.0),
        )
        assert "75.0" in report

    def test_avg_keyword_hit_rate_omitted_when_none(self):
        report = generate_report([], _summary(avg_keyword_hit_rate=None))
        assert "keyword hit rate" not in report.lower() or "75" not in report

    def test_pass_symbol_in_passed_rows(self):
        results = [_result(passed=True)]
        report = generate_report(results, _summary())
        assert "✓" in report

    def test_fail_symbol_in_failed_rows(self):
        results = [_result(passed=False)]
        report = generate_report(results, _summary(passed=0))
        assert "✗" in report

    def test_skip_symbol_in_skipped_rows(self):
        results = [_result(passed=None)]
        report = generate_report(results, _summary())
        assert "—" in report

    def test_long_question_truncated(self):
        long_q = "삼" * 100
        results = [_result(question=long_q)]
        report = generate_report(results, _summary())
        assert long_q not in report  # must have been truncated

    def test_long_error_truncated(self):
        long_err = "error_" * 20
        results = [_result(passed=False, error=long_err)]
        report = generate_report(results, _summary(passed=0))
        assert long_err not in report

    def test_empty_results_does_not_crash(self):
        report = generate_report([], {"total": 0, "passed": 0, "accuracy": 0,
                                      "source_match_rate": 0, "avg_keyword_hit_rate": None})
        assert isinstance(report, str)

    def test_multiple_categories_all_present(self):
        results = [
            _result("n1", "numeric"),
            _result("a1", "adversarial"),
            _result("nar1", "narrative"),
        ]
        report = generate_report(results, _summary(total=3))
        assert "Numeric" in report
        assert "Adversarial" in report
        assert "Narrative" in report


# ---------------------------------------------------------------------------
# evaluator --report flag (integration)
# ---------------------------------------------------------------------------

class TestEvaluatorReportFlag:
    def test_dry_run_with_report_writes_file(self, tmp_path):
        from app.evaluation.evaluator import run_eval, _load_eval_set

        questions = _load_eval_set()[:3]  # only first 3 for speed
        results = run_eval(questions, dry_run=True)
        summary = compute_summary([r for r in results if r.get("passed") is not None])
        report = generate_report(results, summary)

        out = tmp_path / "EVALUATION_REPORT.md"
        out.write_text(report, encoding="utf-8")

        assert out.exists()
        content = out.read_text()
        assert "Evaluation Report" in content

    def test_report_contains_all_question_ids(self, tmp_path):
        from app.evaluation.evaluator import run_eval, _load_eval_set

        questions = _load_eval_set()
        results = run_eval(questions, dry_run=True)
        summary = compute_summary([])
        report = generate_report(results, summary)

        for q in questions[:5]:  # spot-check first 5
            assert q["id"] in report
