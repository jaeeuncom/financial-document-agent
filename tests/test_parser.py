"""Tests for app/document/parser.py and app/document/extractor.py."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.document.parser import (
    _html_to_text,
    _detect_section,
    _build_sections_from_files,
    _extract_html_files,
)


# ---------------------------------------------------------------------------
# _html_to_text
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_strips_tags(self):
        html = "<p>삼성전자의 <b>매출액</b>은 300조입니다.</p>"
        text = _html_to_text(html)
        assert "삼성전자의" in text
        assert "매출액" in text
        assert "<p>" not in text
        assert "<b>" not in text

    def test_excludes_script(self):
        html = "<p>본문 내용</p><script>alert('xss')</script>"
        text = _html_to_text(html)
        assert "본문 내용" in text
        assert "xss" not in text
        assert "alert" not in text

    def test_excludes_style(self):
        html = "<style>.cls{color:red}</style><p>실제 내용</p>"
        text = _html_to_text(html)
        assert "실제 내용" in text
        assert "color" not in text

    def test_empty_html_returns_empty(self):
        assert _html_to_text("") == ""

    def test_plain_div(self):
        assert "안녕하세요" in _html_to_text("<div>안녕하세요</div>")

    def test_table_cell_text_extracted(self):
        html = "<table><tr><td>매출액</td><td>300조</td></tr></table>"
        text = _html_to_text(html)
        assert "매출액" in text
        assert "300조" in text


# ---------------------------------------------------------------------------
# _detect_section
# ---------------------------------------------------------------------------

class TestDetectSection:
    def test_detects_사업의내용(self):
        assert _detect_section("II. 사업의 내용\n본 섹션에서는 사업을 설명합니다.") == "사업의내용"

    def test_detects_mda(self):
        assert _detect_section("이사의 경영진단 및 분석의견\n당기 연결 매출액은") == "MD&A"

    def test_detects_연구개발(self):
        assert _detect_section("연구개발활동\n당사는 반도체 연구에 집중하고 있습니다.") == "연구개발"

    def test_detects_위험요소(self):
        assert _detect_section("사업위험\n반도체 업황 하락 위험이 있습니다.") == "위험요소"

    def test_returns_none_for_unrelated(self):
        assert _detect_section("회사의 개요\n당사는 1969년 설립된 전자 회사입니다.") is None

    def test_case_insensitive_keyword(self):
        # keywords in TARGET_SECTIONS are lowercased; input may be mixed case
        assert _detect_section("II. 사업의 내용") == "사업의내용"

    def test_only_checks_first_1500_chars(self):
        # keyword appears after 1500 chars → should NOT be detected
        prefix = "관련 없는 내용 " * 200   # ~1600 chars
        text = prefix + "이사의 경영진단 및 분석의견"
        assert _detect_section(text) is None


# ---------------------------------------------------------------------------
# _build_sections_from_files
# ---------------------------------------------------------------------------

class TestBuildSectionsFromFiles:
    def test_groups_consecutive_files(self):
        files = [
            (0, "p1.html", "회사의 개요 내용"),
            (1, "p2.html", "II. 사업의 내용\n본문"),
            (2, "p3.html", "계속된 사업 설명"),
            (3, "p4.html", "이사의 경영진단 및 분석의견"),
            (4, "p5.html", "계속된 MD&A"),
        ]
        result = _build_sections_from_files(files)
        section_names = [r[0] for r in result]
        assert "사업의내용" in section_names
        assert "MD&A" in section_names

    def test_combines_text_within_section(self):
        files = [
            (0, "a.html", "사업의 내용\n첫 번째 파일"),
            (1, "b.html", "두 번째 파일"),
        ]
        result = _build_sections_from_files(files)
        combined = result[0][1]
        assert "첫 번째 파일" in combined
        assert "두 번째 파일" in combined

    def test_empty_input_returns_empty(self):
        assert _build_sections_from_files([]) == []

    def test_no_section_match_returns_empty(self):
        files = [(0, "a.html", "회사의 개요만 있음")]
        assert _build_sections_from_files(files) == []

    def test_does_not_split_same_section(self):
        # Two consecutive files both matching 사업의내용 should not create two sections
        files = [
            (0, "a.html", "사업의 내용\n첫 부분"),
            (1, "b.html", "사업의 내용\n두 번째 부분"),
        ]
        result = _build_sections_from_files(files)
        section_names = [r[0] for r in result]
        assert section_names.count("사업의내용") == 1

    def test_start_order_is_first_file(self):
        files = [
            (5, "p5.html", "사업의 내용\n내용"),
            (6, "p6.html", "계속"),
        ]
        result = _build_sections_from_files(files)
        assert result[0][2] == 5  # start_order

    def test_all_four_sections_detected(self):
        files = [
            (0, "a.html", "사업의 내용\n내용1"),
            (1, "b.html", "이사의 경영진단 및 분석의견\n내용2"),
            (2, "c.html", "연구개발활동\n내용3"),
            (3, "d.html", "사업위험\n내용4"),
        ]
        result = _build_sections_from_files(files)
        section_names = {r[0] for r in result}
        assert section_names == {"사업의내용", "MD&A", "연구개발", "위험요소"}


# ---------------------------------------------------------------------------
# _extract_html_files
# ---------------------------------------------------------------------------

class TestExtractHtmlFiles:
    def _make_zip(self, files: dict[str, str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content.encode("utf-8"))
        return buf.getvalue()

    def test_extracts_html_files(self):
        zip_bytes = self._make_zip({
            "report-001.html": "<p>사업의 내용</p>",
            "report-002.html": "<p>이사의 경영진단 및 분석의견</p>",
        })
        result = _extract_html_files(zip_bytes)
        assert len(result) == 2
        texts = [r[2] for r in result]
        assert any("사업의 내용" in t for t in texts)

    def test_skips_non_html_files(self):
        zip_bytes = self._make_zip({
            "report.xml": "<root>XML 내용</root>",
            "report.html": "<p>HTML 내용</p>",
        })
        result = _extract_html_files(zip_bytes)
        assert len(result) == 1

    def test_empty_html_skipped(self):
        zip_bytes = self._make_zip({
            "empty.html": "   ",
            "real.html": "<p>내용 있음</p>",
        })
        result = _extract_html_files(zip_bytes)
        assert len(result) == 1

    def test_sorted_by_filename(self):
        zip_bytes = self._make_zip({
            "b.html": "<p>B</p>",
            "a.html": "<p>A</p>",
        })
        result = _extract_html_files(zip_bytes)
        filenames = [r[1] for r in result]
        assert filenames == sorted(filenames)


# ---------------------------------------------------------------------------
# parse_report (integration-style, all HTTP mocked)
# ---------------------------------------------------------------------------

def _make_report_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("report-001.html", "<p>회사의 개요 내용입니다.</p>".encode("utf-8"))
        zf.writestr(
            "report-002.html",
            "<h2>II. 사업의 내용</h2><p>반도체 사업을 영위합니다.</p>".encode("utf-8"),
        )
        zf.writestr(
            "report-003.html",
            "<h2>이사의 경영진단 및 분석의견</h2><p>당기 매출은 증가했습니다.</p>".encode("utf-8"),
        )
        zf.writestr(
            "report-004.html",
            "<h2>연구개발활동</h2><p>AI 반도체 연구에 투자했습니다.</p>".encode("utf-8"),
        )
        zf.writestr(
            "report-005.html",
            "<h2>사업위험</h2><p>글로벌 경기 침체 위험이 있습니다.</p>".encode("utf-8"),
        )
    return buf.getvalue()


class TestParseReport:
    def _mock_get(self, url: str, **kwargs) -> MagicMock:
        m = MagicMock()
        m.raise_for_status = MagicMock()
        if "document.xml" in url:
            m.content = _make_report_zip()
        elif "list.json" in url:
            m.json.return_value = {
                "status": "000",
                "list": [{"rcept_no": "20250310000123"}],
            }
        return m

    def test_returns_parsed_sections(self, tmp_path):
        from app.document import parser as parser_mod

        with (
            patch("app.document.parser.requests.get", side_effect=self._mock_get),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(parser_mod.config, "DART_API_KEY", "fake_key"),
            patch.object(parser_mod.config, "DATA_DIR", tmp_path),
        ):
            sections = parser_mod.parse_report("삼성전자", 2024)

        assert len(sections) >= 2
        section_names = {s.section for s in sections}
        assert "사업의내용" in section_names
        assert "MD&A" in section_names

    def test_section_metadata_populated(self, tmp_path):
        from app.document import parser as parser_mod

        with (
            patch("app.document.parser.requests.get", side_effect=self._mock_get),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(parser_mod.config, "DART_API_KEY", "fake_key"),
            patch.object(parser_mod.config, "DATA_DIR", tmp_path),
        ):
            sections = parser_mod.parse_report("삼성전자", 2024)

        for s in sections:
            assert s.company == "삼성전자"
            assert s.year == 2024
            assert s.rcept_no == "20250310000123"
            assert "dart.fss.or.kr" in s.source_url
            assert s.char_count > 0
            assert s.extraction_ok is True

    def test_document_zip_cached(self, tmp_path):
        from app.document import parser as parser_mod

        mock_get = MagicMock(side_effect=self._mock_get)

        with (
            patch("app.document.parser.requests.get", mock_get),
            patch("app.tools.dart_tool.resolve_corp_code", return_value="00126380"),
            patch.object(parser_mod.config, "DART_API_KEY", "fake_key"),
            patch.object(parser_mod.config, "DATA_DIR", tmp_path),
        ):
            parser_mod.parse_report("삼성전자", 2024)
            parser_mod.parse_report("삼성전자", 2024)  # second call should use cache

        # document.xml should be fetched only once
        doc_calls = [c for c in mock_get.call_args_list if "document.xml" in str(c)]
        assert len(doc_calls) == 1


# ---------------------------------------------------------------------------
# extractor save / load
# ---------------------------------------------------------------------------

class TestExtractorSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        from app.document import extractor as ext_mod
        from app.validation.schema import ExtractionResult, ParsedSection

        section = ParsedSection(
            company="삼성전자", year=2024, section="사업의내용",
            section_title="report-002.html", text="사업 설명 본문",
            page_or_order=1, rcept_no="20250310000123",
            source_url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250310000123",
            char_count=8, extraction_ok=True,
        )
        result = ExtractionResult(
            company="삼성전자", year=2024, rcept_no="20250310000123",
            sections=[section], total_chars=8,
            failed_sections=["MD&A", "연구개발", "위험요소"],
            extraction_ok=True,
        )

        with patch.object(ext_mod.config, "PARSED_DIR", tmp_path):
            path = ext_mod.save_parsed(result)
            loaded = ext_mod.load_parsed("삼성전자", 2024)

        assert path.exists()
        assert loaded is not None
        assert loaded.company == "삼성전자"
        assert loaded.year == 2024
        assert len(loaded.sections) == 1
        assert loaded.sections[0].section == "사업의내용"

    def test_load_missing_returns_none(self, tmp_path):
        from app.document import extractor as ext_mod

        with patch.object(ext_mod.config, "PARSED_DIR", tmp_path):
            result = ext_mod.load_parsed("없는회사", 2024)

        assert result is None
