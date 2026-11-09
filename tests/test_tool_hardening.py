"""Tests for Day 10: Pydantic input validation, retry with backoff, error codes."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest
from pydantic import ValidationError

from app.tools.dart_tool import (
    FinancialStatementQuery,
    _call_dart_api,
    get_financial_statement,
)
from app.validation.rules import ReviewRequired


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _api_resp(status: str, message: str = "test") -> MagicMock:
    m = MagicMock()
    m.json.return_value = {"status": status, "list": [], "message": message}
    m.raise_for_status = MagicMock()
    return m


# ---------------------------------------------------------------------------
# FinancialStatementQuery — input validation model
# ---------------------------------------------------------------------------

class TestFinancialStatementQuery:
    def test_valid_input(self):
        q = FinancialStatementQuery(company="삼성전자", year=2024)
        assert q.company == "삼성전자"
        assert q.year == 2024
        assert q.accounts is None

    def test_blank_company_raises(self):
        with pytest.raises(ValidationError):
            FinancialStatementQuery(company="", year=2024)

    def test_whitespace_only_company_raises(self):
        with pytest.raises(ValidationError):
            FinancialStatementQuery(company="   ", year=2024)

    def test_company_stripped(self):
        q = FinancialStatementQuery(company="  삼성전자  ", year=2024)
        assert q.company == "삼성전자"

    def test_year_below_range_raises(self):
        with pytest.raises(ValidationError):
            FinancialStatementQuery(company="삼성전자", year=1999)

    def test_year_at_minimum_accepted(self):
        q = FinancialStatementQuery(company="삼성전자", year=2000)
        assert q.year == 2000

    def test_year_above_range_raises(self):
        with pytest.raises(ValidationError):
            FinancialStatementQuery(company="삼성전자", year=2031)

    def test_year_at_maximum_accepted(self):
        q = FinancialStatementQuery(company="삼성전자", year=2030)
        assert q.year == 2030

    def test_empty_accounts_list_raises(self):
        with pytest.raises(ValidationError):
            FinancialStatementQuery(company="삼성전자", year=2024, accounts=[])

    def test_none_accounts_allowed(self):
        q = FinancialStatementQuery(company="삼성전자", year=2024, accounts=None)
        assert q.accounts is None

    def test_valid_accounts_list(self):
        q = FinancialStatementQuery(company="삼성전자", year=2024, accounts=["매출액"])
        assert q.accounts == ["매출액"]


# ---------------------------------------------------------------------------
# _call_dart_api — retry and error code logic
# ---------------------------------------------------------------------------

class TestCallDartApi:
    def test_000_returns_data(self):
        with patch("app.tools.dart_tool.requests.get", return_value=_api_resp("000")):
            data = _call_dart_api("http://test.com", {})
        assert data["status"] == "000"

    def test_013_raises_no_data_for_period(self):
        with patch("app.tools.dart_tool.requests.get", return_value=_api_resp("013", "no data")):
            with pytest.raises(ReviewRequired) as exc:
                _call_dart_api("http://test.com", {})
        assert exc.value.rule == "no_data_for_period"

    def test_other_error_raises_dart_api_error(self):
        with patch("app.tools.dart_tool.requests.get", return_value=_api_resp("010", "error")):
            with pytest.raises(ReviewRequired) as exc:
                _call_dart_api("http://test.com", {})
        assert exc.value.rule == "dart_api_error"

    def test_020_retries_before_success(self):
        responses = [_api_resp("020"), _api_resp("000")]
        with patch("app.tools.dart_tool.requests.get", side_effect=responses):
            with patch("app.tools.dart_tool.time.sleep"):
                data = _call_dart_api("http://test.com", {}, max_retries=3)
        assert data["status"] == "000"

    def test_020_exhausted_raises_dart_rate_limit(self):
        with patch("app.tools.dart_tool.requests.get", side_effect=[_api_resp("020")] * 3):
            with patch("app.tools.dart_tool.time.sleep"):
                with pytest.raises(ReviewRequired) as exc:
                    _call_dart_api("http://test.com", {}, max_retries=3)
        assert exc.value.rule == "dart_rate_limit"

    def test_sleep_called_between_retries(self):
        # Two 020s then success → sleep twice
        responses = [_api_resp("020"), _api_resp("020"), _api_resp("000")]
        with patch("app.tools.dart_tool.requests.get", side_effect=responses):
            with patch("app.tools.dart_tool.time.sleep") as mock_sleep:
                _call_dart_api("http://test.com", {}, max_retries=3)
        assert mock_sleep.call_count == 2

    def test_backoff_delays_are_exponential(self):
        responses = [_api_resp("020"), _api_resp("020"), _api_resp("000")]
        with patch("app.tools.dart_tool.requests.get", side_effect=responses):
            with patch("app.tools.dart_tool.time.sleep") as mock_sleep:
                _call_dart_api("http://test.com", {}, max_retries=3)
        # attempt 0 → sleep(1), attempt 1 → sleep(2)
        mock_sleep.assert_has_calls([call(1), call(2)])

    def test_013_does_not_retry(self):
        # 013 should raise immediately without retrying
        with patch("app.tools.dart_tool.requests.get", return_value=_api_resp("013")) as mock_get:
            with pytest.raises(ReviewRequired):
                _call_dart_api("http://test.com", {}, max_retries=3)
        assert mock_get.call_count == 1  # only one attempt

    def test_other_error_does_not_retry(self):
        with patch("app.tools.dart_tool.requests.get", return_value=_api_resp("010")) as mock_get:
            with pytest.raises(ReviewRequired):
                _call_dart_api("http://test.com", {}, max_retries=3)
        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# get_financial_statement — input validation bubbles up as ReviewRequired
# ---------------------------------------------------------------------------

class TestGetFinancialStatementValidation:
    def test_year_below_range_raises_invalid_argument(self):
        with pytest.raises(ReviewRequired) as exc:
            get_financial_statement("삼성전자", 1990)
        assert exc.value.rule == "invalid_argument"

    def test_year_above_range_raises_invalid_argument(self):
        with pytest.raises(ReviewRequired) as exc:
            get_financial_statement("삼성전자", 2031)
        assert exc.value.rule == "invalid_argument"

    def test_blank_company_raises_invalid_argument(self):
        with pytest.raises(ReviewRequired) as exc:
            get_financial_statement("", 2024)
        assert exc.value.rule == "invalid_argument"

    def test_whitespace_company_raises_invalid_argument(self):
        with pytest.raises(ReviewRequired) as exc:
            get_financial_statement("  ", 2024)
        assert exc.value.rule == "invalid_argument"

    def test_empty_accounts_raises_invalid_argument(self):
        with pytest.raises(ReviewRequired) as exc:
            get_financial_statement("삼성전자", 2024, accounts=[])
        assert exc.value.rule == "invalid_argument"

    def test_valid_input_passes_pydantic_then_raises_missing_api_key(self):
        from app.tools import dart_tool
        # Valid args pass Pydantic; missing DART key causes the next error
        with patch.object(dart_tool.config, "DART_API_KEY", None):
            with pytest.raises(ReviewRequired) as exc:
                get_financial_statement("삼성전자", 2024)
        assert exc.value.rule == "missing_api_key"

    def test_rate_limit_rule_on_020_exhaustion(self, tmp_path):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", side_effect=[_api_resp("020")] * 3),
            patch("app.tools.dart_tool.time.sleep"),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            with pytest.raises(ReviewRequired) as exc:
                get_financial_statement("삼성전자", 2024)
        assert exc.value.rule == "dart_rate_limit"

    def test_no_data_rule_on_013(self, tmp_path):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=_api_resp("013", "조회된 데이터가 없습니다")),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            with pytest.raises(ReviewRequired) as exc:
                get_financial_statement("삼성전자", 2024)
        assert exc.value.rule == "no_data_for_period"
