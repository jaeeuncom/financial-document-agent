from __future__ import annotations

import pytest

from app.validation.schema import FinancialFact
from app.validation.rules import ReviewRequired, check_missing, check_domain


def _make_fact(account: str, current: int | None = 100, prior: int | None = 90) -> FinancialFact:
    return FinancialFact(
        company="테스트",
        corp_code="00000001",
        year=2024,
        account=account,
        fs_div="CFS",
        current_value=current,
        prior_value=prior,
        rcept_no="20240101000000",
        source_url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20240101000000",
    )


class TestReviewRequired:
    def test_has_rule_and_detail(self):
        exc = ReviewRequired("company_not_found", "ABC not in DART")
        assert exc.rule == "company_not_found"
        assert exc.detail == "ABC not in DART"

    def test_is_exception(self):
        with pytest.raises(ReviewRequired):
            raise ReviewRequired("test_rule")


class TestCheckMissing:
    def test_all_present_does_not_raise(self):
        facts = [_make_fact("매출액"), _make_fact("영업이익")]
        check_missing(["매출액", "영업이익"], facts)  # no exception

    def test_missing_account_raises(self):
        facts = [_make_fact("매출액")]
        with pytest.raises(ReviewRequired) as exc_info:
            check_missing(["매출액", "영업이익"], facts)
        assert exc_info.value.rule == "required_field_missing"
        assert "영업이익" in exc_info.value.detail

    def test_empty_requested_does_not_raise(self):
        check_missing([], [])

    def test_all_missing_raises(self):
        with pytest.raises(ReviewRequired) as exc_info:
            check_missing(["매출액", "영업이익"], [])
        assert exc_info.value.rule == "required_field_missing"


class TestCheckDomain:
    def test_positive_revenue_ok(self):
        check_domain(_make_fact("매출액", current=1000, prior=900))

    def test_negative_revenue_raises(self):
        with pytest.raises(ReviewRequired) as exc_info:
            check_domain(_make_fact("매출액", current=-100, prior=900))
        assert exc_info.value.rule == "domain_negative_value"

    def test_negative_prior_revenue_raises(self):
        with pytest.raises(ReviewRequired) as exc_info:
            check_domain(_make_fact("매출액", current=1000, prior=-50))
        assert exc_info.value.rule == "domain_negative_value"

    def test_negative_operating_profit_ok(self):
        # 영업이익 is not in DOMAIN_NON_NEGATIVE — negative is allowed
        check_domain(_make_fact("영업이익", current=-500, prior=200))

    def test_negative_assets_raises(self):
        with pytest.raises(ReviewRequired) as exc_info:
            check_domain(_make_fact("자산총계", current=-1))
        assert exc_info.value.rule == "domain_negative_value"

    def test_none_values_do_not_raise(self):
        check_domain(_make_fact("매출액", current=None, prior=None))
