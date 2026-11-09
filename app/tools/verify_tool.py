"""verify_source: re-fetch a FinancialFact from DART and compare the stored value."""
from __future__ import annotations

from typing import Callable, Optional

from app.validation.rules import ReviewRequired


def verify_source(
    fact: dict,
    *,
    get_facts_fn: Optional[Callable] = None,
) -> None:
    """
    Re-fetch a financial fact from DART and confirm the stored current_value matches.

    Raises ReviewRequired("value_mismatch") when the re-fetched value differs.
    Raises ReviewRequired("invalid_argument") when required keys are absent.
    Propagates any ReviewRequired raised by get_facts_fn (e.g. missing API key,
    company_not_found).

    Parameters
    ----------
    fact:
        A dict produced by FinancialFact.model_dump() — must contain
        "company", "year", "account", and "current_value".
    get_facts_fn:
        Callable matching get_financial_statement(company, year, accounts) signature.
        Defaults to the real DART implementation; override in tests.
    """
    company = fact.get("company")
    year = fact.get("year")
    account = fact.get("account")

    if not company or not year or not account:
        raise ReviewRequired(
            "invalid_argument",
            "fact is missing required field(s): company, year, or account",
        )

    if get_facts_fn is None:
        from app.tools.dart_tool import get_financial_statement
        get_facts_fn = get_financial_statement

    fresh_facts = get_facts_fn(company, int(year), accounts=[account])
    fresh_map: dict[str, Optional[int]] = {f.account: f.current_value for f in fresh_facts}

    stored_value = fact.get("current_value")
    fresh_value = fresh_map.get(account)

    if fresh_value != stored_value:
        raise ReviewRequired(
            "value_mismatch",
            (
                f"{company} {year} {account}: "
                f"stored={stored_value}, fresh={fresh_value}"
            ),
        )
