"""Multi-company / multi-year financial comparison utility.

Takes a list of FinancialFact dicts (already serialized via model_dump()) and
groups them by account name so the LLM can produce a structured comparison table.
"""
from __future__ import annotations

from collections import defaultdict


def compare_financial_data(facts: list[dict]) -> list[dict]:
    """
    Group financial facts by account for multi-company / multi-year comparison.

    Returns a list of:
      {
        "account": str,
        "entries": [
          {"company": str, "year": int, "current_value": int|None,
           "prior_value": int|None, "source_url": str},
          ...
        ]
      }
    Sorted alphabetically by account name.
    Facts missing an account name are skipped.
    """
    if not facts:
        return []

    by_account: dict[str, list[dict]] = defaultdict(list)
    for f in facts:
        account = f.get("account", "")
        if not account:
            continue
        by_account[account].append({
            "company": f.get("company"),
            "year": f.get("year"),
            "current_value": f.get("current_value"),
            "prior_value": f.get("prior_value"),
            "source_url": f.get("source_url"),
        })

    return [
        {"account": account, "entries": entries}
        for account, entries in sorted(by_account.items())
    ]
