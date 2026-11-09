"""Tests for Day 11: verify_source tool and check_cross_period validation."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.tools.verify_tool import verify_source
from app.validation.rules import ReviewRequired, check_cross_period
from app.validation.schema import FinancialFact


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fact(
    company: str = "삼성전자",
    year: int = 2024,
    account: str = "매출액",
    current: int | None = 300_000_000,
    prior: int | None = 258_000_000,
) -> FinancialFact:
    return FinancialFact(
        company=company,
        corp_code="00126380",
        year=year,
        account=account,
        fs_div="CFS",
        current_value=current,
        prior_value=prior,
        currency="KRW",
        rcept_no="20250310000123",
        source_url=f"https://dart.fss.or.kr/test/{company}/{year}",
    )


def _fact_dict(**overrides) -> dict:
    base = _fact().model_dump()
    base.update(overrides)
    return base


def _make_get_facts_fn(facts: list[FinancialFact]):
    """Return a mock get_facts_fn that returns the given facts."""
    def _fn(company, year, accounts=None):
        return facts
    return _fn


# ---------------------------------------------------------------------------
# verify_source
# ---------------------------------------------------------------------------

class TestVerifySource:
    def test_matching_value_does_not_raise(self):
        fact = _fact_dict(current_value=300_000_000)
        fresh = [_fact(current=300_000_000)]
        verify_source(fact, get_facts_fn=_make_get_facts_fn(fresh))  # no exception

    def test_mismatched_value_raises_value_mismatch(self):
        fact = _fact_dict(current_value=300_000_000)
        fresh = [_fact(current=999_000_000)]  # different value
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn(fresh))
        assert exc.value.rule == "value_mismatch"

    def test_mismatch_detail_contains_company_year_account(self):
        fact = _fact_dict(company="삼성전자", year=2024, account="매출액", current_value=100)
        fresh = [_fact(company="삼성전자", year=2024, account="매출액", current=200)]
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn(fresh))
        detail = exc.value.detail
        assert "삼성전자" in detail
        assert "2024" in detail
        assert "매출액" in detail

    def test_mismatch_detail_contains_stored_and_fresh(self):
        fact = _fact_dict(current_value=100)
        fresh = [_fact(current=200)]
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn(fresh))
        assert "stored=100" in exc.value.detail
        assert "fresh=200" in exc.value.detail

    def test_none_stored_vs_none_fresh_does_not_raise(self):
        fact = _fact_dict(current_value=None)
        fresh = [_fact(current=None)]
        verify_source(fact, get_facts_fn=_make_get_facts_fn(fresh))  # no exception

    def test_none_stored_vs_value_fresh_raises(self):
        fact = _fact_dict(current_value=None)
        fresh = [_fact(current=300_000_000)]
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn(fresh))
        assert exc.value.rule == "value_mismatch"

    def test_value_stored_vs_none_fresh_raises(self):
        fact = _fact_dict(current_value=300_000_000)
        fresh = [_fact(current=None)]
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn(fresh))
        assert exc.value.rule == "value_mismatch"

    def test_missing_company_raises_invalid_argument(self):
        fact = _fact_dict()
        del fact["company"]
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn([]))
        assert exc.value.rule == "invalid_argument"

    def test_empty_company_raises_invalid_argument(self):
        fact = _fact_dict(company="")
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn([]))
        assert exc.value.rule == "invalid_argument"

    def test_missing_year_raises_invalid_argument(self):
        fact = _fact_dict()
        del fact["year"]
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn([]))
        assert exc.value.rule == "invalid_argument"

    def test_missing_account_raises_invalid_argument(self):
        fact = _fact_dict()
        del fact["account"]
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn([]))
        assert exc.value.rule == "invalid_argument"

    def test_propagates_review_required_from_get_facts_fn(self):
        def _failing_fn(company, year, accounts=None):
            raise ReviewRequired("company_not_found", "not in DART")

        with pytest.raises(ReviewRequired) as exc:
            verify_source(_fact_dict(), get_facts_fn=_failing_fn)
        assert exc.value.rule == "company_not_found"

    def test_account_not_in_fresh_response_raises_value_mismatch(self):
        # DART returned data but account key is absent (edge case)
        fact = _fact_dict(account="영업이익", current_value=50_000_000)
        fresh = [_fact(account="매출액", current=300_000_000)]  # different account
        with pytest.raises(ReviewRequired) as exc:
            verify_source(fact, get_facts_fn=_make_get_facts_fn(fresh))
        assert exc.value.rule == "value_mismatch"

    def test_calls_get_facts_fn_with_correct_arguments(self):
        calls = []
        def _recording_fn(company, year, accounts=None):
            calls.append((company, year, accounts))
            return [_fact(company=company, year=year)]

        fact = _fact_dict(company="SK하이닉스", year=2023, account="매출액")
        verify_source(fact, get_facts_fn=_recording_fn)
        assert calls == [("SK하이닉스", 2023, ["매출액"])]


# ---------------------------------------------------------------------------
# check_cross_period
# ---------------------------------------------------------------------------

class TestCheckCrossPeriod:
    def test_matching_values_does_not_raise(self):
        # year 2023 current == year 2024 prior → consistent
        year_n = _fact(year=2023, current=258_000_000, prior=240_000_000)
        year_n1 = _fact(year=2024, current=300_000_000, prior=258_000_000)
        check_cross_period(year_n, year_n1)  # no exception

    def test_mismatch_raises_cross_period_mismatch(self):
        year_n = _fact(year=2023, current=258_000_000)
        year_n1 = _fact(year=2024, current=300_000_000, prior=260_000_000)  # ≠ 258M
        with pytest.raises(ReviewRequired) as exc:
            check_cross_period(year_n, year_n1)
        assert exc.value.rule == "cross_period_mismatch"

    def test_mismatch_detail_mentions_restatement(self):
        year_n = _fact(year=2023, current=100)
        year_n1 = _fact(year=2024, current=200, prior=999)
        with pytest.raises(ReviewRequired) as exc:
            check_cross_period(year_n, year_n1)
        assert "restatement" in exc.value.detail.lower() or "mismatch" in exc.value.detail.lower()

    def test_mismatch_detail_contains_both_years(self):
        year_n = _fact(year=2022, current=100)
        year_n1 = _fact(year=2023, current=200, prior=999)
        with pytest.raises(ReviewRequired) as exc:
            check_cross_period(year_n, year_n1)
        assert "2022" in exc.value.detail
        assert "2023" in exc.value.detail

    def test_none_current_in_year_n_skips_check(self):
        year_n = _fact(year=2023, current=None)
        year_n1 = _fact(year=2024, current=300_000_000, prior=258_000_000)
        check_cross_period(year_n, year_n1)  # no exception

    def test_none_prior_in_year_n1_skips_check(self):
        year_n = _fact(year=2023, current=258_000_000)
        year_n1 = _fact(year=2024, current=300_000_000, prior=None)
        check_cross_period(year_n, year_n1)  # no exception

    def test_both_none_skips_check(self):
        year_n = _fact(year=2023, current=None)
        year_n1 = _fact(year=2024, prior=None)
        check_cross_period(year_n, year_n1)  # no exception

    def test_company_mismatch_raises_value_error(self):
        year_n = _fact(company="삼성전자", year=2023)
        year_n1 = _fact(company="SK하이닉스", year=2024)
        with pytest.raises(ValueError):
            check_cross_period(year_n, year_n1)

    def test_account_mismatch_raises_value_error(self):
        year_n = _fact(account="매출액", year=2023)
        year_n1 = _fact(account="영업이익", year=2024)
        with pytest.raises(ValueError):
            check_cross_period(year_n, year_n1)

    def test_zero_values_match_does_not_raise(self):
        year_n = _fact(year=2023, current=0)
        year_n1 = _fact(year=2024, current=100, prior=0)
        check_cross_period(year_n, year_n1)  # no exception

    def test_different_account_types_work(self):
        year_n = _fact(account="영업이익", year=2023, current=-500_000, prior=200_000)
        year_n1 = _fact(account="영업이익", year=2024, current=800_000, prior=-500_000)
        check_cross_period(year_n, year_n1)  # no exception — values match

    def test_negative_operating_profit_mismatch_raises(self):
        year_n = _fact(account="영업이익", year=2023, current=-500_000)
        year_n1 = _fact(account="영업이익", year=2024, current=800_000, prior=-999_000)
        with pytest.raises(ReviewRequired) as exc:
            check_cross_period(year_n, year_n1)
        assert exc.value.rule == "cross_period_mismatch"
