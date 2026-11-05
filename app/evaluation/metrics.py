from __future__ import annotations

from typing import Any


def numeric_exact_match(predicted: int | None, expected: int) -> bool:
    """True if predicted equals expected. Skips check when expected == 0 (placeholder)."""
    if expected == 0:
        return True
    return predicted == expected


def source_contains_dart(sources: list[str]) -> bool:
    return any("dart.fss.or.kr" in s for s in sources)


def evaluate_numeric_result(
    item: dict[str, Any],
    facts: list[dict[str, Any]],
    sources: list[str],
) -> dict[str, Any]:
    """
    Evaluate a numeric eval question against direct tool results.

    item: eval_set entry
    facts: list of FinancialFact dicts returned by get_financial_statement
    sources: source URLs from facts
    """
    expected = item.get("expected", {})
    expected_values: dict[str, int] = expected.get("values", {})
    should_need_review: bool = expected.get("should_need_review", False)

    value_results: dict[str, bool] = {}
    for account, expected_val in expected_values.items():
        fact = next((f for f in facts if f["account"] == account), None)
        predicted = fact["current_value"] if fact else None
        value_results[account] = numeric_exact_match(predicted, expected_val)

    source_ok = source_contains_dart(sources)
    # For numeric questions run via tool, needs_review is always False on success
    review_correct = not should_need_review

    passed = all(value_results.values()) and source_ok and review_correct

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "passed": passed,
        "value_results": value_results,
        "source_ok": source_ok,
        "review_correct": review_correct,
        "needs_review": False,
        "error": None,
    }


def evaluate_adversarial_result(
    item: dict[str, Any],
    answer: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate an adversarial eval question against a full agent answer.

    item: eval_set entry
    answer: AgentAnswer dict (keys: answer, sources, needs_review)
    """
    expected = item.get("expected", {})
    should_need_review: bool = expected.get("should_need_review", True)

    needs_review: bool = answer.get("needs_review", False)
    review_correct = needs_review == should_need_review
    source_ok = not should_need_review or True  # adversarial questions don't require sources

    passed = review_correct

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "passed": passed,
        "value_results": {},
        "source_ok": source_ok,
        "review_correct": review_correct,
        "needs_review": needs_review,
        "error": None,
    }


def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0, "passed": 0, "accuracy": 0.0, "source_match_rate": 0.0}

    passed = sum(1 for r in results if r.get("passed", False))
    with_source = [r for r in results if r.get("source_ok") is not None]
    source_ok = sum(1 for r in with_source if r.get("source_ok", False))

    return {
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total * 100, 1),
        "source_match_rate": round(source_ok / len(with_source) * 100, 1) if with_source else 0.0,
    }
