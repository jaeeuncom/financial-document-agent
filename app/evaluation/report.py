"""Generate a Markdown EVALUATION_REPORT.md from eval results."""
from __future__ import annotations

from datetime import date
from typing import Any, Optional


CATEGORY_ORDER = ["numeric", "comparison", "adversarial", "narrative", "mixed", "tool_selection"]


def _per_category(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Group results by category in canonical order.

    Returns an ordered dict where each key is a category name and each value has:
        total, passed, failed, skipped, items
    """
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    ordered: dict[str, dict[str, Any]] = {}
    for cat in CATEGORY_ORDER:
        if cat in by_cat:
            _add_cat(ordered, cat, by_cat.pop(cat))
    for cat, items in by_cat.items():  # any remaining categories
        _add_cat(ordered, cat, items)
    return ordered


def _add_cat(out: dict, cat: str, items: list[dict]) -> None:
    out[cat] = {
        "total": len(items),
        "passed": sum(1 for i in items if i.get("passed") is True),
        "failed": sum(1 for i in items if i.get("passed") is False),
        "skipped": sum(1 for i in items if i.get("passed") is None),
        "items": items,
    }


def _accuracy_str(passed: int, total: int, skipped: int) -> str:
    runnable = total - skipped
    if runnable == 0:
        return "— (all skipped)"
    return f"{round(passed / runnable * 100, 1)}% ({passed}/{runnable})"


def _truncate(s: str, n: int) -> str:
    return s[:n - 3] + "..." if len(s) > n else s


def generate_report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    generated_at: Optional[str] = None,
) -> str:
    """
    Generate a Markdown EVALUATION_REPORT from eval results.

    Parameters
    ----------
    results:
        List of result dicts from run_eval() — each has keys:
        id, category, question, passed (True/False/None), error, source_ok, ...
    summary:
        Dict from compute_summary() — keys: total, passed, accuracy,
        source_match_rate, avg_keyword_hit_rate.
    generated_at:
        ISO date string (default: today).

    Returns
    -------
    str
        Full Markdown string ready to write to docs/EVALUATION_REPORT.md.
    """
    ts = generated_at or date.today().isoformat()
    total = len(results)
    runnable = [r for r in results if r.get("passed") is not None]
    n_passed = sum(1 for r in runnable if r.get("passed") is True)
    n_failed = sum(1 for r in runnable if r.get("passed") is False)
    n_skipped = total - len(runnable)
    accuracy = round(n_passed / len(runnable) * 100, 1) if runnable else 0.0

    cats = _per_category(results)

    lines: list[str] = [
        "# Financial Document Agent — Evaluation Report",
        "",
        f"> Generated: {ts} · Questions: {total} · Accuracy (runnable): {accuracy}%",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total questions | {total} |",
        f"| Passed | {n_passed} |",
        f"| Failed | {n_failed} |",
        f"| Skipped / Not runnable | {n_skipped} |",
        f"| Overall accuracy | {accuracy}% |",
        f"| Source match rate | {summary.get('source_match_rate', 0.0)}% |",
    ]
    avg_khr = summary.get("avg_keyword_hit_rate")
    if avg_khr is not None:
        lines.append(f"| Avg keyword hit rate (narrative) | {avg_khr}% |")

    lines += [
        "",
        "---",
        "",
        "## Results by Category",
        "",
        "| Category | Total | Passed | Failed | Skipped | Accuracy |",
        "|----------|-------|--------|--------|---------|----------|",
    ]
    for cat, stats in cats.items():
        acc = _accuracy_str(stats["passed"], stats["total"], stats["skipped"])
        lines.append(
            f"| {cat} | {stats['total']} | {stats['passed']} "
            f"| {stats['failed']} | {stats['skipped']} | {acc} |"
        )

    lines.append("")

    for cat, stats in cats.items():
        lines += [
            f"### {cat.capitalize()} ({stats['total']} questions)",
            "",
            "| ID | Question | Result | Source | Error |",
            "|----|----------|--------|--------|-------|",
        ]
        for item in stats["items"]:
            p = item.get("passed")
            sym = "✓" if p is True else ("✗" if p is False else "—")
            src = "✓" if item.get("source_ok") else ("—" if p is None else "✗")
            err = _truncate(item.get("error") or "", 60)
            q = _truncate(item.get("question", ""), 55)
            lines.append(f"| {item.get('id', '?')} | {q} | {sym} | {src} | {err} |")
        lines.append("")

    # Failure analysis
    failures = [r for r in results if r.get("passed") is False]
    if failures:
        lines += [
            "---",
            "",
            "## Failure Analysis",
            "",
            "| ID | Category | Question | Error |",
            "|----|----------|----------|-------|",
        ]
        for r in failures:
            err = _truncate(r.get("error") or "(no message)", 70)
            q = _truncate(r.get("question", ""), 60)
            lines.append(
                f"| {r.get('id', '?')} | {r.get('category', '?')} | {q} | {err} |"
            )
        lines.append("")

    if n_skipped:
        lines += [
            "### Skipped Questions",
            "",
            "> These questions require external resources (indexed documents, OpenAI API key,",
            "> or DART API key) that were not available in this run.",
            "",
        ]

    lines += [
        "---",
        "",
        "## How to Run",
        "",
        "```bash",
        "# Full eval (requires DART_API_KEY and OPENAI_API_KEY in .env)",
        "python -m app.evaluation.evaluator --report",
        "",
        "# Dry run — validate JSON structure only",
        "python -m app.evaluation.evaluator --dry-run --report",
        "",
        "# Single question",
        "python -m app.evaluation.evaluator --id fin-001",
        "```",
        "",
        "---",
        "",
        "*Report generated by `app/evaluation/report.py`*",
    ]

    return "\n".join(lines)
