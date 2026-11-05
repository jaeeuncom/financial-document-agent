import io
import json
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from app.tools.dart_tool import _parse_amount, calculate_growth_rate
from app.validation.rules import ReviewRequired


# ---------------------------------------------------------------------------
# _parse_amount
# ---------------------------------------------------------------------------

class TestParseAmount:
    def test_comma_separated(self):
        assert _parse_amount("300,870,345,000,000") == 300_870_345_000_000

    def test_dash_returns_none(self):
        assert _parse_amount("-") is None

    def test_empty_returns_none(self):
        assert _parse_amount("") is None

    def test_simple_number(self):
        assert _parse_amount("12345") == 12345

    def test_whitespace_stripped(self):
        assert _parse_amount("  12,345  ") == 12345


# ---------------------------------------------------------------------------
# calculate_growth_rate
# ---------------------------------------------------------------------------

class TestCalculateGrowthRate:
    def test_positive_growth(self):
        assert calculate_growth_rate(110, 100) == 10.0

    def test_negative_growth(self):
        assert calculate_growth_rate(90, 100) == -10.0

    def test_division_by_zero(self):
        with pytest.raises(ReviewRequired) as exc_info:
            calculate_growth_rate(100, 0)
        assert exc_info.value.rule == "division_by_zero"

    def test_rounding_to_two_decimals(self):
        assert calculate_growth_rate(133, 100) == 33.0

    def test_negative_previous_uses_abs(self):
        # (-80 - -100) / |-100| * 100 = 20/100 * 100 = 20.0
        assert calculate_growth_rate(-80, -100) == 20.0


# ---------------------------------------------------------------------------
# _load_corp_codes / resolve_corp_code
# ---------------------------------------------------------------------------

def _make_corpcode_zip() -> bytes:
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<result>
    <list>
        <corp_code>00126380</corp_code>
        <corp_name>삼성전자</corp_name>
        <stock_code>005930</stock_code>
        <modify_date>20240101</modify_date>
    </list>
    <list>
        <corp_code>00164779</corp_code>
        <corp_name>SK하이닉스</corp_name>
        <stock_code>000660</stock_code>
        <modify_date>20240101</modify_date>
    </list>
    <list>
        <corp_code>99999999</corp_code>
        <corp_name>비상장회사</corp_name>
        <stock_code></stock_code>
        <modify_date>20240101</modify_date>
    </list>
</result>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CORPCODE.xml", xml.encode("utf-8"))
    return buf.getvalue()


class TestCorpCodes:
    def _mock_resp(self) -> MagicMock:
        m = MagicMock()
        m.content = _make_corpcode_zip()
        m.raise_for_status = MagicMock()
        return m

    def test_listed_companies_included(self, tmp_path):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=self._mock_resp()),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            codes = dart_tool._load_corp_codes()

        assert "삼성전자" in codes
        assert codes["삼성전자"] == "00126380"
        assert "SK하이닉스" in codes

    def test_unlisted_company_excluded(self, tmp_path):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=self._mock_resp()),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            codes = dart_tool._load_corp_codes()

        assert "비상장회사" not in codes

    def test_cache_used_on_second_call(self, tmp_path):
        from app.tools import dart_tool

        mock_get = MagicMock(return_value=self._mock_resp())
        with (
            patch("app.tools.dart_tool.requests.get", mock_get),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            dart_tool._load_corp_codes()
            dart_tool._load_corp_codes()

        # requests.get should be called only once; second call uses cache
        assert mock_get.call_count == 1

    def test_resolve_known_company(self, tmp_path):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=self._mock_resp()),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            code = dart_tool.resolve_corp_code("삼성전자")

        assert code == "00126380"

    def test_resolve_unknown_company_raises(self, tmp_path):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=self._mock_resp()),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            with pytest.raises(ReviewRequired) as exc_info:
                dart_tool.resolve_corp_code("없는회사")
        assert exc_info.value.rule == "company_not_found"


# ---------------------------------------------------------------------------
# get_financial_statement
# ---------------------------------------------------------------------------

FAKE_FINANCIAL_DATA = {
    "status": "000",
    "message": "정상",
    "list": [
        {
            "rcept_no": "20250310000123",
            "corp_code": "00126380",
            "account_nm": "매출액",
            "thstrm_amount": "300,870,345,000,000",
            "frmtrm_amount": "258,935,676,000,000",
            "currency": "KRW",
            "fs_div": "CFS",
        },
        {
            # OFS row for same account — must be filtered out
            "rcept_no": "20250310000123",
            "corp_code": "00126380",
            "account_nm": "매출액",
            "thstrm_amount": "261,728,660,000,000",
            "frmtrm_amount": "234,100,000,000,000",
            "currency": "KRW",
            "fs_div": "OFS",
        },
        {
            "rcept_no": "20250310000123",
            "corp_code": "00126380",
            "account_nm": "영업이익",
            "thstrm_amount": "32,726,543,000,000",
            "frmtrm_amount": "6,566,976,000,000",
            "currency": "KRW",
            "fs_div": "CFS",
        },
        {
            "rcept_no": "20250310000123",
            "corp_code": "00126380",
            "account_nm": "당기순이익",
            "thstrm_amount": "34,468,282,000,000",
            "frmtrm_amount": "15,484,468,000,000",
            "currency": "KRW",
            "fs_div": "CFS",
        },
    ],
}


class TestGetFinancialStatement:
    def _mock_financial_resp(self) -> MagicMock:
        m = MagicMock()
        m.json.return_value = FAKE_FINANCIAL_DATA
        m.raise_for_status = MagicMock()
        return m

    def test_returns_cfs_only(self):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=self._mock_financial_resp()),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
        ):
            facts = dart_tool.get_financial_statement("삼성전자", 2024)

        assert all(f.fs_div == "CFS" for f in facts)
        assert len(facts) == 3

    def test_parses_amounts_correctly(self):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=self._mock_financial_resp()),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
        ):
            facts = dart_tool.get_financial_statement("삼성전자", 2024)

        revenue = next(f for f in facts if f.account == "매출액")
        assert revenue.current_value == 300_870_345_000_000
        assert revenue.prior_value == 258_935_676_000_000

    def test_source_url_uses_rcept_no(self):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=self._mock_financial_resp()),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
        ):
            facts = dart_tool.get_financial_statement("삼성전자", 2024)

        for f in facts:
            assert "20250310000123" in f.source_url
            assert "dart.fss.or.kr" in f.source_url

    def test_dart_api_error_raises(self):
        from app.tools import dart_tool

        m = MagicMock()
        m.json.return_value = {"status": "010", "message": "조회된 데이터가 없습니다."}
        m.raise_for_status = MagicMock()

        with (
            patch("app.tools.dart_tool.requests.get", return_value=m),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
        ):
            with pytest.raises(ReviewRequired) as exc_info:
                dart_tool.get_financial_statement("삼성전자", 2020)
        assert exc_info.value.rule == "dart_api_error"

    def test_raw_response_cached_to_disk(self, tmp_path):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool.requests.get", return_value=self._mock_financial_resp()),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            dart_tool.get_financial_statement("삼성전자", 2024)

        cache_file = tmp_path / "raw" / "dart" / "00126380_2024_single.json"
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text())
        assert cached["status"] == "000"

    def test_raw_cache_avoids_second_api_call(self, tmp_path):
        from app.tools import dart_tool

        mock_get = MagicMock(return_value=self._mock_financial_resp())
        with (
            patch("app.tools.dart_tool.requests.get", mock_get),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            dart_tool.get_financial_statement("삼성전자", 2024)
            dart_tool.get_financial_statement("삼성전자", 2024)

        assert mock_get.call_count == 1

    def test_account_variant_normalized(self, tmp_path):
        """'수익(매출액)' in DART response should be normalized to '매출액'."""
        from app.tools import dart_tool

        variant_data = {
            "status": "000",
            "message": "정상",
            "list": [
                {
                    "rcept_no": "20250310000999",
                    "corp_code": "00000001",
                    "account_nm": "수익(매출액)",
                    "thstrm_amount": "10,000,000,000",
                    "frmtrm_amount": "9,000,000,000",
                    "currency": "KRW",
                    "fs_div": "CFS",
                },
            ],
        }
        m = MagicMock()
        m.json.return_value = variant_data
        m.raise_for_status = MagicMock()

        with (
            patch("app.tools.dart_tool.requests.get", return_value=m),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00000001"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            facts = dart_tool.get_financial_statement("테스트회사", 2024, accounts=["매출액"])

        assert len(facts) == 1
        assert facts[0].account == "매출액"
        assert facts[0].current_value == 10_000_000_000

    def test_bs_accounts_retrieved(self, tmp_path):
        """Balance sheet accounts (자산총계 etc.) are retrievable."""
        from app.tools import dart_tool

        bs_data = {
            "status": "000",
            "message": "정상",
            "list": [
                {
                    "rcept_no": "20250310000123",
                    "corp_code": "00126380",
                    "account_nm": "자산총계",
                    "thstrm_amount": "500,000,000,000,000",
                    "frmtrm_amount": "450,000,000,000,000",
                    "currency": "KRW",
                    "fs_div": "CFS",
                },
                {
                    "rcept_no": "20250310000123",
                    "corp_code": "00126380",
                    "account_nm": "부채총계",
                    "thstrm_amount": "200,000,000,000,000",
                    "frmtrm_amount": "180,000,000,000,000",
                    "currency": "KRW",
                    "fs_div": "CFS",
                },
                {
                    "rcept_no": "20250310000123",
                    "corp_code": "00126380",
                    "account_nm": "자본총계",
                    "thstrm_amount": "300,000,000,000,000",
                    "frmtrm_amount": "270,000,000,000,000",
                    "currency": "KRW",
                    "fs_div": "CFS",
                },
            ],
        }
        m = MagicMock()
        m.json.return_value = bs_data
        m.raise_for_status = MagicMock()

        with (
            patch("app.tools.dart_tool.requests.get", return_value=m),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            facts = dart_tool.get_financial_statement(
                "삼성전자", 2024, accounts=["자산총계", "부채총계", "자본총계"]
            )

        accounts_found = {f.account for f in facts}
        assert accounts_found == {"자산총계", "부채총계", "자본총계"}

    def test_use_all_calls_different_endpoint(self, tmp_path):
        from app.tools import dart_tool

        m = MagicMock()
        m.json.return_value = FAKE_FINANCIAL_DATA
        m.raise_for_status = MagicMock()
        mock_get = MagicMock(return_value=m)

        with (
            patch("app.tools.dart_tool.requests.get", mock_get),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(dart_tool.config, "DART_API_KEY", "fake_key"),
            patch.object(dart_tool.config, "DATA_DIR", tmp_path),
        ):
            dart_tool.get_financial_statement("삼성전자", 2024, use_all=True)

        called_url = mock_get.call_args[0][0]
        assert "fnlttSinglAcntAll" in called_url


# ---------------------------------------------------------------------------
# alias resolution
# ---------------------------------------------------------------------------

class TestAliasResolution:
    def test_alias_resolves_to_canonical(self, tmp_path):
        from app.tools import dart_tool

        aliases = {"삼전": "삼성전자"}
        corp_codes = {"삼성전자": "00126380"}

        with (
            patch("app.tools.dart_tool._load_corp_codes", return_value=corp_codes),
            patch("app.tools.dart_tool._load_aliases", return_value=aliases),
        ):
            code = dart_tool.resolve_corp_code("삼전")

        assert code == "00126380"

    def test_unknown_alias_raises(self, tmp_path):
        from app.tools import dart_tool

        with (
            patch("app.tools.dart_tool._load_corp_codes", return_value={"삼성전자": "00126380"}),
            patch("app.tools.dart_tool._load_aliases", return_value={}),
        ):
            with pytest.raises(ReviewRequired) as exc_info:
                dart_tool.resolve_corp_code("모르는회사")

        assert exc_info.value.rule == "company_not_found"

    def test_exact_match_takes_priority_over_alias(self, tmp_path):
        from app.tools import dart_tool

        aliases = {"삼성전자": "다른회사"}  # alias exists but exact match should win
        corp_codes = {"삼성전자": "00126380"}

        with (
            patch("app.tools.dart_tool._load_corp_codes", return_value=corp_codes),
            patch("app.tools.dart_tool._load_aliases", return_value=aliases),
        ):
            code = dart_tool.resolve_corp_code("삼성전자")

        assert code == "00126380"


# ---------------------------------------------------------------------------
# evaluation metrics
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def test_numeric_exact_match_placeholder_zero(self):
        from app.evaluation.metrics import numeric_exact_match
        assert numeric_exact_match(999, 0) is True

    def test_numeric_exact_match_correct(self):
        from app.evaluation.metrics import numeric_exact_match
        assert numeric_exact_match(300_870_345_000_000, 300_870_345_000_000) is True

    def test_numeric_exact_match_wrong(self):
        from app.evaluation.metrics import numeric_exact_match
        assert numeric_exact_match(100, 200) is False

    def test_numeric_exact_match_none_predicted(self):
        from app.evaluation.metrics import numeric_exact_match
        assert numeric_exact_match(None, 200) is False

    def test_source_contains_dart(self):
        from app.evaluation.metrics import source_contains_dart
        assert source_contains_dart(["https://dart.fss.or.kr/dsaf001/main.do?rcpNo=123"]) is True
        assert source_contains_dart(["https://example.com"]) is False
        assert source_contains_dart([]) is False

    def test_compute_summary_all_pass(self):
        from app.evaluation.metrics import compute_summary
        results = [
            {"passed": True, "source_ok": True},
            {"passed": True, "source_ok": True},
        ]
        summary = compute_summary(results)
        assert summary["accuracy"] == 100.0
        assert summary["total"] == 2
        assert summary["passed"] == 2

    def test_compute_summary_partial(self):
        from app.evaluation.metrics import compute_summary
        results = [
            {"passed": True, "source_ok": True},
            {"passed": False, "source_ok": False},
        ]
        summary = compute_summary(results)
        assert summary["accuracy"] == 50.0

    def test_compute_summary_empty(self):
        from app.evaluation.metrics import compute_summary
        summary = compute_summary([])
        assert summary["total"] == 0
        assert summary["accuracy"] == 0.0
