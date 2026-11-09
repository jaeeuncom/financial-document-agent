from __future__ import annotations

from app.validation.schema import FinancialFact

DOMAIN_NON_NEGATIVE = {"매출액", "자산총계", "자본금", "자본총계"}

# Tolerance for cross-period comparison (integer KRW amounts — exact match expected)
_CROSS_PERIOD_TOLERANCE = 0


class ReviewRequired(Exception):
    def __init__(self, rule: str, detail: str = "") -> None:
        self.rule = rule
        self.detail = detail
        super().__init__(f"{rule}: {detail}")


def check_missing(requested: list[str], facts: list[FinancialFact]) -> None:
    found = {f.account for f in facts}
    missing = [acc for acc in requested if acc not in found]
    if missing:
        raise ReviewRequired(
            "required_field_missing",
            f"accounts not found: {missing}",
        )


def check_cross_period(year_n: FinancialFact, year_n1: FinancialFact) -> None:
    """
    Verify that year N's current value equals year N+1's prior value.

    In DART, a report for year N+1 includes a 'frmtrm_amount' column which is
    the prior-year (year N) comparative figure.  This should equal the
    'thstrm_amount' in the year N report.  A discrepancy indicates a restatement
    and must be flagged for human review.

    Raises ValueError if company or account differs (caller mistake).
    Raises ReviewRequired("cross_period_mismatch") on a numeric discrepancy.
    Silently returns when either value is None (can't compare missing data).
    """
    if year_n.company != year_n1.company:
        raise ValueError(
            f"company mismatch in cross-period check: "
            f"'{year_n.company}' vs '{year_n1.company}'"
        )
    if year_n.account != year_n1.account:
        raise ValueError(
            f"account mismatch in cross-period check: "
            f"'{year_n.account}' vs '{year_n1.account}'"
        )

    if year_n.current_value is None or year_n1.prior_value is None:
        return  # missing data: skip

    if year_n.current_value != year_n1.prior_value:
        raise ReviewRequired(
            "cross_period_mismatch",
            (
                f"{year_n.company} {year_n.account}: "
                f"year {year_n.year} current={year_n.current_value} "
                f"≠ year {year_n1.year} prior={year_n1.prior_value} "
                "(possible restatement)"
            ),
        )


def check_domain(fact: FinancialFact) -> None:
    if fact.account not in DOMAIN_NON_NEGATIVE:
        return
    if fact.current_value is not None and fact.current_value < 0:
        raise ReviewRequired(
            "domain_negative_value",
            f"{fact.account} current_value is negative: {fact.current_value}",
        )
    if fact.prior_value is not None and fact.prior_value < 0:
        raise ReviewRequired(
            "domain_negative_value",
            f"{fact.account} prior_value is negative: {fact.prior_value}",
        )
