"""
DART annual report document parser.

Primary path  : download document.xml (zip), parse HTML files, detect target sections.
Secondary path: parse_pdf() — PyMuPDF text + pdfplumber tables for upload / cross-validation.
"""
from __future__ import annotations

import io
import json
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import requests

from app import config
from app.validation.schema import ParsedSection
from app.validation.rules import ReviewRequired

# Canonical section name -> detection keywords (checked case-insensitively in first 1 500 chars)
TARGET_SECTIONS: dict[str, list[str]] = {
    "사업의내용": [
        "사업의 내용",
        "사업내용",
        "ii. 사업",
        "제 2 부",
        "제2부",
    ],
    "MD&A": [
        "이사의 경영진단 및 분석의견",
        "경영진단 및 분석",
        "이사의 경영진단",
        "경영진단의견",
    ],
    "연구개발": [
        "연구개발활동",
        "연구개발에 관한 사항",
        "기술개발현황",
        "연구개발 현황",
    ],
    "위험요소": [
        "사업위험",
        "위험요소",
        "투자위험",
        "주요 위험",
    ],
}


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip HTML tags; skip <script>/<style>; insert newlines at block elements."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "td", "th"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        lines: list[str] = []
        for part in self._parts:
            if part == "\n":
                if lines and lines[-1] != "":
                    lines.append("")
            else:
                lines.append(part)
        return "\n".join(lines).strip()


def _decode_html(html_bytes: bytes) -> str:
    for enc in ("utf-8", "euc-kr", "cp949", "utf-8-sig"):
        try:
            return html_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return html_bytes.decode("utf-8", errors="replace")


def _html_to_text(html_str: str) -> str:
    extractor = _TextExtractor()
    try:
        extractor.feed(html_str)
    except Exception:
        pass
    return extractor.get_text()


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _detect_section(text: str) -> Optional[str]:
    """
    Return the first matching canonical section name from TARGET_SECTIONS,
    or None. Only the first 1 500 characters are checked to reduce false positives
    from body text that *refers to* section names.
    """
    sample = text[:1500].lower()
    for section_name, keywords in TARGET_SECTIONS.items():
        if any(kw.lower() in sample for kw in keywords):
            return section_name
    return None


def _build_sections_from_files(
    file_texts: list[tuple[int, str, str]],
) -> list[tuple[str, str, int, str]]:
    """
    Group consecutive HTML files into named sections.

    file_texts : [(order, filename, text), ...]
    Returns    : [(section_name, combined_text, start_order, first_filename), ...]

    Files before the first detected section heading are silently skipped
    (they are typically cover page / table-of-contents material).
    When a new section keyword appears, the current section is closed and a
    new one opens; files with no keyword continue the current section.
    """
    current_section: Optional[str] = None
    current_texts: list[str] = []
    current_order = 0
    current_filename = ""
    sections: list[tuple[str, str, int, str]] = []

    for order, filename, text in file_texts:
        detected = _detect_section(text)

        if detected and detected != current_section:
            # Save the section we were building
            if current_section is not None and current_texts:
                sections.append((
                    current_section,
                    "\n\n".join(current_texts),
                    current_order,
                    current_filename,
                ))
            current_section = detected
            current_texts = [text]
            current_order = order
            current_filename = filename
        elif current_section is not None:
            current_texts.append(text)
        # else: before first section — skip

    if current_section is not None and current_texts:
        sections.append((
            current_section,
            "\n\n".join(current_texts),
            current_order,
            current_filename,
        ))

    return sections


# ---------------------------------------------------------------------------
# DART API helpers
# ---------------------------------------------------------------------------

def _get_rcept_no(corp_code: str, year: int) -> str:
    """
    Return the annual report rcept_no for corp_code / year.

    Tries the financial-statement raw cache first (data/raw/dart/).
    Falls back to the DART list API if cache is absent.
    """
    for suffix in ("single", "all"):
        cache = config.DATA_DIR / "raw" / "dart" / f"{corp_code}_{year}_{suffix}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            items = [i for i in data.get("list", []) if i.get("rcept_no")]
            if items:
                return items[0]["rcept_no"]

    if not config.DART_API_KEY:
        raise ReviewRequired("missing_api_key", "DART_API_KEY is not set in .env")

    # Annual reports for FY {year} are filed between {year}/01/01 and {year+1}/06/30
    url = f"{config.DART_BASE}/list.json"
    resp = requests.get(
        url,
        params={
            "crtfc_key": config.DART_API_KEY,
            "corp_code": corp_code,
            "bgn_de": f"{year}0101",
            "end_de": f"{year + 1}0630",
            "pblntf_ty": "A",
            "last_reprt_at": "Y",
        },
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()
    if data.get("status") != "000" or not data.get("list"):
        raise ReviewRequired(
            "rcept_no_not_found",
            f"No annual report found for corp_code={corp_code} year={year}",
        )
    # list is sorted newest-first by default
    return data["list"][0]["rcept_no"]


def _download_document_zip(rcept_no: str) -> bytes:
    """
    Download and return the document zip for rcept_no.
    Caches to data/raw/dart/docs/{rcept_no}.zip to avoid re-downloads.
    """
    cache_dir = config.DATA_DIR / "raw" / "dart" / "docs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{rcept_no}.zip"

    if cache_path.exists():
        return cache_path.read_bytes()

    if not config.DART_API_KEY:
        raise ReviewRequired("missing_api_key", "DART_API_KEY is not set in .env")

    url = f"{config.DART_BASE}/document.xml"
    resp = requests.get(
        url,
        params={"crtfc_key": config.DART_API_KEY, "rcept_no": rcept_no},
        timeout=120,
    )
    resp.raise_for_status()

    cache_path.write_bytes(resp.content)
    return resp.content


def _extract_html_files(zip_bytes: bytes) -> list[tuple[int, str, str]]:
    """
    Extract text from all HTML files inside the zip.
    Returns [(order, filename, text), ...] sorted by filename.
    Errors per file are logged inline; extraction continues.
    """
    results: list[tuple[int, str, str]] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith((".html", ".htm")))
        for order, name in enumerate(names):
            try:
                raw = zf.read(name)
                text = _html_to_text(_decode_html(raw))
                if text.strip():
                    results.append((order, name, text))
            except Exception as exc:
                results.append((order, name, f"[extraction_error: {exc}]"))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_report(company: str, year: int) -> list[ParsedSection]:
    """
    Download and parse the DART annual report for company/year.

    Returns a list of ParsedSection objects — one per detected target section.
    Raises ReviewRequired on unrecoverable errors.
    """
    from app.tools.dart_tool import resolve_corp_code  # avoid circular import

    corp_code = resolve_corp_code(company)
    rcept_no = _get_rcept_no(corp_code, year)
    source_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"

    zip_bytes = _download_document_zip(rcept_no)
    file_texts = _extract_html_files(zip_bytes)

    if not file_texts:
        raise ReviewRequired(
            "document_parse_failed",
            f"No HTML files in document zip: company={company} year={year} rcept_no={rcept_no}",
        )

    raw_sections = _build_sections_from_files(file_texts)
    sections: list[ParsedSection] = []

    for section_name, text, order, filename in raw_sections:
        sections.append(
            ParsedSection(
                company=company,
                year=year,
                doc_type="사업보고서",
                section=section_name,
                section_title=filename,
                text=text,
                page_or_order=order,
                rcept_no=rcept_no,
                source_url=source_url,
                char_count=len(text),
                extraction_ok=True,
            )
        )

    return sections


def parse_pdf(
    pdf_path: Path,
    company: str,
    year: int,
    rcept_no: str = "",
) -> list[ParsedSection]:
    """
    Secondary path: extract narrative sections from an uploaded or cached PDF.

    Uses PyMuPDF for text extraction; pdfplumber for table extraction (returned
    as metadata, not stored in ParsedSection.text — tables are for cross-validation).
    Raises ReviewRequired if pymupdf is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ReviewRequired(
            "missing_dependency",
            "pymupdf is not installed. Run: pip install pymupdf",
        )

    source_url = (
        f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
        if rcept_no
        else str(pdf_path)
    )

    doc = fitz.open(str(pdf_path))
    page_texts: list[tuple[int, str]] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            page_texts.append((page_num, text))

    doc.close()

    # Reuse the same section-building algorithm
    file_texts = [
        (order, f"page_{page_num}", text)
        for order, (page_num, text) in enumerate(page_texts)
    ]
    raw_sections = _build_sections_from_files(file_texts)

    sections: list[ParsedSection] = []
    for section_name, text, order, _ in raw_sections:
        sections.append(
            ParsedSection(
                company=company,
                year=year,
                doc_type="PDF",
                section=section_name,
                section_title=f"PDF page group {order}",
                text=text,
                page_or_order=order,
                rcept_no=rcept_no,
                source_url=source_url,
                char_count=len(text),
                extraction_ok=True,
            )
        )

    return sections
