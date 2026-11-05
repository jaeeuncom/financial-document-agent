from __future__ import annotations

import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

from app import config
from app.validation.schema import FinancialFact
from app.validation.rules import ReviewRequired, check_missing, check_domain

# Default accounts to retrieve when none are specified
DEFAULT_ACCOUNTS = ["매출액", "영업이익", "당기순이익"]

# Balance-sheet accounts available via get_financial_statement
BS_ACCOUNTS = ["자산총계", "부채총계", "자본총계"]

# Maps canonical account name -> list of DART API account_nm variants (company-specific naming)
ACCOUNT_VARIANTS: dict[str, list[str]] = {
    "매출액": ["매출액", "수익(매출액)", "영업수익", "매출"],
    "영업이익": ["영업이익", "영업이익(손실)"],
    "당기순이익": ["당기순이익", "당기순이익(손실)", "분기순이익"],
    "자산총계": ["자산총계"],
    "부채총계": ["부채총계"],
    "자본총계": ["자본총계"],
}

# Reverse lookup: DART account_nm -> canonical name
_VARIANT_TO_CANONICAL: dict[str, str] = {
    variant: canonical
    for canonical, variants in ACCOUNT_VARIANTS.items()
    for variant in variants
}


def _parse_amount(raw: str) -> Optional[int]:
    s = raw.strip()
    if s in ("-", "", "N/A"):
        return None
    return int(s.replace(",", "").replace(" ", ""))


def _load_aliases() -> dict[str, str]:
    aliases_path = config.DATA_DIR / "aliases.json"
    if aliases_path.exists():
        return json.loads(aliases_path.read_text(encoding="utf-8"))
    return {}


def _load_corp_codes() -> dict[str, str]:
    cache_path = config.DATA_DIR / "corp_codes.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if not config.DART_API_KEY:
        raise ReviewRequired("missing_api_key", "DART_API_KEY is not set in .env")

    url = f"{config.DART_BASE}/corpCode.xml"
    resp = requests.get(url, params={"crtfc_key": config.DART_API_KEY}, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_bytes = zf.read("CORPCODE.xml")

    # DART XML may be UTF-8 or EUC-KR; try UTF-8 first
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        root = ET.fromstring(xml_bytes.decode("euc-kr").encode("utf-8"))

    codes: dict[str, str] = {}
    for item in root.findall("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        if not stock_code:
            continue
        corp_name = (item.findtext("corp_name") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if corp_name and corp_code:
            codes[corp_name] = corp_code

    cache_path.write_text(
        json.dumps(codes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return codes


def resolve_corp_code(company: str) -> str:
    codes = _load_corp_codes()

    # 1. Exact match
    if company in codes:
        return codes[company]

    # 2. Alias lookup -> canonical name -> corp code
    aliases = _load_aliases()
    canonical = aliases.get(company)
    if canonical and canonical in codes:
        return codes[canonical]

    raise ReviewRequired(
        "company_not_found",
        f"'{company}' not found in DART listed companies",
    )


def _raw_cache_path(corp_code: str, year: int, use_all: bool) -> Path:
    suffix = "all" if use_all else "single"
    return config.DATA_DIR / "raw" / "dart" / f"{corp_code}_{year}_{suffix}.json"


def get_financial_statement(
    company: str,
    year: int,
    accounts: Optional[list[str]] = None,
    use_all: bool = False,
) -> list[FinancialFact]:
    if not accounts:
        accounts = DEFAULT_ACCOUNTS

    if not config.DART_API_KEY:
        raise ReviewRequired("missing_api_key", "DART_API_KEY is not set in .env")

    corp_code = resolve_corp_code(company)

    endpoint = "fnlttSinglAcntAll" if use_all else "fnlttSinglAcnt"
    url = f"{config.DART_BASE}/{endpoint}.json"

    # Serve from raw cache if available (avoids redundant API calls)
    cache_path = _raw_cache_path(corp_code, year, use_all)
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        resp = requests.get(
            url,
            params={
                "crtfc_key": config.DART_API_KEY,
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": "11011",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "000":
            raise ReviewRequired(
                "dart_api_error",
                f"status={data.get('status')}, message={data.get('message')}",
            )

        # Cache raw response for review evidence and re-use
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if data.get("status") != "000":
        raise ReviewRequired(
            "dart_api_error",
            f"status={data.get('status')}, message={data.get('message')}",
        )

    seen: set[str] = set()
    facts: list[FinancialFact] = []

    for item in data.get("list", []):
        if item.get("fs_div") != "CFS":
            continue

        account_nm = (item.get("account_nm") or "").strip()
        # Normalize variant name to canonical (e.g. "수익(매출액)" -> "매출액")
        canonical = _VARIANT_TO_CANONICAL.get(account_nm, account_nm)

        if canonical not in accounts:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)

        rcept_no = item.get("rcept_no", "")
        fact = FinancialFact(
            company=company,
            corp_code=corp_code,
            year=year,
            account=canonical,
            fs_div="CFS",
            current_value=_parse_amount(item.get("thstrm_amount", "-")),
            prior_value=_parse_amount(item.get("frmtrm_amount", "-")),
            currency=item.get("currency", "KRW"),
            rcept_no=rcept_no,
            source_url=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}",
        )
        check_domain(fact)
        facts.append(fact)

    check_missing(accounts, facts)
    return facts


def calculate_growth_rate(current: int, previous: int) -> float:
    if previous == 0:
        raise ReviewRequired("division_by_zero", "previous value is zero")
    return round((current - previous) / abs(previous) * 100, 2)
