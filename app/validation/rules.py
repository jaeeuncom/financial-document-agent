from __future__ import annotations

from app.validation.schema import FinancialFact

DOMAIN_NON_NEGATIVE = {"매출액", "자산총계", "자본금", "자본총계"}


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
