"""
Evaluation CLI.

Usage:
    python -m app.evaluation.evaluator            # run all questions
    python -m app.evaluation.evaluator --dry-run  # validate JSON structure only
    python -m app.evaluation.evaluator --id fin-001  # run a single question
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.evaluation.metrics import (
    compute_summary,
    evaluate_adversarial_result,
    evaluate_narrative_result,
    evaluate_numeric_result,
)
from app.validation.rules import ReviewRequired

EVAL_SET_PATH = Path(__file__).parent.parent.parent / "tests" / "eval_set.json"


def _load_eval_set() -> list[dict[str, Any]]:
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"Eval set not found: {EVAL_SET_PATH}")
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def _run_numeric(item: dict[str, Any]) -> dict[str, Any]:
    from app.tools.dart_tool import get_financial_statement

    q = item["question"]
    params = item.get("tool_params", {})
    company = params.get("company", "")
    year = int(params.get("year", 0))
    accounts = params.get("accounts") or list(item["expected"]["values"].keys())

    if not company or not year:
        return {
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "passed": False,
            "value_results": {},
            "source_ok": False,
            "review_correct": False,
            "needs_review": False,
            "error": "tool_params missing company or year",
        }

    try:
        facts = get_financial_statement(company, year, accounts)
        fact_dicts = [f.model_dump() for f in facts]
        sources = [f.source_url for f in facts]
        return evaluate_numeric_result(item, fact_dicts, sources)
    except ReviewRequired as exc:
        should_review = item["expected"].get("should_need_review", False)
        return {
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "passed": should_review,  # expected to need review?
            "value_results": {},
            "source_ok": False,
            "review_correct": should_review,
            "needs_review": True,
            "error": f"{exc.rule}: {exc.detail}",
        }
    except Exception as exc:
        return {
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "passed": False,
            "value_results": {},
            "source_ok": False,
            "review_correct": False,
            "needs_review": False,
            "error": str(exc),
        }


def _run_narrative(item: dict[str, Any]) -> dict[str, Any]:
    """
    Retrieval-phase narrative eval: search_narrative → keyword hit rate.

    Evaluates retrieval quality only (no LLM call) so the eval suite runs without
    an OpenAI key. Returns passed=None when documents are not yet indexed.
    """
    from app.retrieval.vector_store import search_narrative
    from app.evaluation.retrieval_eval import evaluate_retrieval

    q = item["question"]
    params = item.get("tool_params", {})

    try:
        results = search_narrative(
            query=q,
            company=params.get("company"),
            year=params.get("year"),
            section=params.get("section"),
            top_k=5,
        )
    except Exception as exc:
        return {
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "passed": False,
            "keyword_hit_rate": 0.0,
            "keyword_coverage": {},
            "source_ok": False,
            "num_results": 0,
            "error": str(exc),
        }

    return evaluate_retrieval(item, results)


def _run_adversarial(item: dict[str, Any]) -> dict[str, Any]:
    from app.agent.loop import run_agent

    try:
        answer = run_agent(item["question"])
        return evaluate_adversarial_result(item, answer.model_dump())
    except Exception as exc:
        return {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "passed": False,
            "value_results": {},
            "source_ok": False,
            "review_correct": False,
            "needs_review": False,
            "error": str(exc),
        }


def run_eval(
    questions: list[dict[str, Any]],
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    results = []
    for item in questions:
        qid = item["id"]
        category = item["category"]
        question = item["question"]

        print(f"  [{qid}] {question[:60]}...")

        if dry_run:
            result: dict[str, Any] = {
                "id": qid,
                "category": category,
                "question": question,
                "passed": None,
                "error": "dry-run",
            }
        elif category in ("numeric", "tool_selection"):
            result = _run_numeric(item)
        elif category == "adversarial":
            result = _run_adversarial(item)
        elif category == "narrative":
            result = _run_narrative(item)
        else:
            result = {
                "id": qid,
                "category": category,
                "question": question,
                "passed": None,
                "error": f"unknown category '{category}'",
            }

        status = "SKIP" if result.get("passed") is None else ("PASS" if result["passed"] else "FAIL")
        err = f"  ({result.get('error')})" if result.get("error") else ""
        print(f"    -> {status}{err}")
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run financial agent evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate structure only, no API calls")
    parser.add_argument("--id", dest="question_id", help="Run a single question by ID")
    args = parser.parse_args()

    questions = _load_eval_set()
    if args.question_id:
        questions = [q for q in questions if q["id"] == args.question_id]
        if not questions:
            print(f"Question ID '{args.question_id}' not found in eval set.")
            return

    print(f"Running {len(questions)} eval question(s)...\n")
    results = run_eval(questions, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDry run complete. {len(results)} questions validated.")
        return

    summary = compute_summary([r for r in results if r.get("passed") is not None])
    skipped = sum(1 for r in results if r.get("passed") is None)

    print("\n" + "=" * 50)
    print("Results:")
    print(f"  Total:            {len(results)}")
    print(f"  Passed:           {summary['passed']}")
    print(f"  Failed:           {summary['total'] - summary['passed']}")
    if skipped:
        print(f"  Skipped:          {skipped}")
    print(f"  Accuracy:         {summary['accuracy']}%")
    print(f"  Source match:     {summary['source_match_rate']}%")
    if summary.get("avg_keyword_hit_rate") is not None:
        print(f"  Avg keyword hit:  {summary['avg_keyword_hit_rate']}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
