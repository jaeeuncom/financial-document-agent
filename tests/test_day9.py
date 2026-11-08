"""Tests for Day 9: compare_tool, state persistence, multi-company calculate_node."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.tools.compare_tool import compare_financial_data
from app.agent.nodes import calculate_node
from app.agent.state import AgentState
from app.validation.rules import ReviewRequired
from app.validation.schema import AgentAnswer, FinancialFact


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> AgentState:
    base: AgentState = {
        "question": "삼성전자 2024년 매출액은?",
        "entities": {"companies": ["삼성전자"], "years": [2024], "accounts": ["매출액"]},
        "route": "financial",
        "tool_results": [],
        "retrieved_chunks": [],
        "computed": {},
        "answer": None,
        "needs_review": False,
    }
    base.update(overrides)
    return base


def _fact(company: str = "삼성전자", year: int = 2024, account: str = "매출액",
          current: int = 300_000_000, prior: int = 258_000_000) -> dict:
    return {
        "company": company,
        "corp_code": "00126380",
        "year": year,
        "account": account,
        "fs_div": "CFS",
        "current_value": current,
        "prior_value": prior,
        "currency": "KRW",
        "rcept_no": "20250310000123",
        "source_url": f"https://dart.fss.or.kr/test/{company}/{year}",
    }


def _financial_fact(**overrides) -> FinancialFact:
    base = _fact()
    base.update(overrides)
    return FinancialFact(**base)


def _llm_resp(parsed) -> MagicMock:
    r = MagicMock()
    r.choices[0].message.parsed = parsed
    return r


# ---------------------------------------------------------------------------
# compare_financial_data
# ---------------------------------------------------------------------------

class TestCompareFinancialData:
    def test_empty_input_returns_empty(self):
        assert compare_financial_data([]) == []

    def test_single_fact_returns_one_account_group(self):
        result = compare_financial_data([_fact()])
        assert len(result) == 1
        assert result[0]["account"] == "매출액"
        assert len(result[0]["entries"]) == 1

    def test_multiple_companies_same_account_grouped(self):
        facts = [
            _fact(company="삼성전자", year=2024, account="매출액"),
            _fact(company="SK하이닉스", year=2024, account="매출액"),
        ]
        result = compare_financial_data(facts)
        assert len(result) == 1
        assert len(result[0]["entries"]) == 2
        companies = {e["company"] for e in result[0]["entries"]}
        assert companies == {"삼성전자", "SK하이닉스"}

    def test_multiple_years_same_company_grouped(self):
        facts = [
            _fact(company="삼성전자", year=2024, account="영업이익"),
            _fact(company="삼성전자", year=2023, account="영업이익"),
        ]
        result = compare_financial_data(facts)
        assert len(result) == 1
        years = {e["year"] for e in result[0]["entries"]}
        assert years == {2024, 2023}

    def test_multiple_accounts_produce_separate_groups(self):
        facts = [
            _fact(account="매출액"),
            _fact(account="영업이익"),
        ]
        result = compare_financial_data(facts)
        assert len(result) == 2
        accounts = {r["account"] for r in result}
        assert accounts == {"매출액", "영업이익"}

    def test_sorted_alphabetically_by_account(self):
        facts = [_fact(account="영업이익"), _fact(account="매출액"), _fact(account="당기순이익")]
        result = compare_financial_data(facts)
        account_names = [r["account"] for r in result]
        assert account_names == sorted(account_names)

    def test_entries_contain_required_fields(self):
        result = compare_financial_data([_fact()])
        entry = result[0]["entries"][0]
        assert "company" in entry
        assert "year" in entry
        assert "current_value" in entry
        assert "prior_value" in entry
        assert "source_url" in entry

    def test_facts_without_account_are_skipped(self):
        facts = [{"company": "삼성전자", "year": 2024}]  # no "account" key
        result = compare_financial_data(facts)
        assert result == []


# ---------------------------------------------------------------------------
# calculate_node with comparison
# ---------------------------------------------------------------------------

class TestCalculateNodeComparison:
    def test_no_comparison_for_single_company_year(self):
        state = _state(tool_results=[_fact()])
        with patch("app.agent.nodes.calculate_growth_rate", return_value=16.19):
            result = calculate_node(state)
        assert result["computed"]["comparison"] == []

    def test_comparison_added_for_multiple_companies(self):
        state = _state(tool_results=[
            _fact(company="삼성전자"),
            _fact(company="SK하이닉스"),
        ])
        with patch("app.agent.nodes.calculate_growth_rate", return_value=10.0):
            result = calculate_node(state)
        assert len(result["computed"]["comparison"]) == 1  # one account group
        assert result["computed"]["comparison"][0]["account"] == "매출액"

    def test_comparison_added_for_multiple_years(self):
        state = _state(tool_results=[
            _fact(year=2024),
            _fact(year=2023),
        ])
        with patch("app.agent.nodes.calculate_growth_rate", return_value=10.0):
            result = calculate_node(state)
        assert len(result["computed"]["comparison"]) == 1

    def test_growth_rates_still_computed_alongside_comparison(self):
        state = _state(tool_results=[
            _fact(company="삼성전자", current=300, prior=258),
            _fact(company="SK하이닉스", current=100, prior=90),
        ])
        with patch("app.agent.nodes.calculate_growth_rate", return_value=11.1):
            result = calculate_node(state)
        assert len(result["computed"]["growth_rates"]) == 2
        assert len(result["computed"]["comparison"]) > 0

    def test_needs_review_entries_excluded_from_comparison(self):
        # Two valid companies → comparison computed; needs_review entry must be absent
        state = _state(tool_results=[
            _fact(company="삼성전자"),
            _fact(company="SK하이닉스"),
            {"status": "needs_review", "company": "없는회사", "year": 2024},
        ])
        with patch("app.agent.nodes.calculate_growth_rate", return_value=10.0):
            result = calculate_node(state)
        entries = result["computed"]["comparison"][0]["entries"]
        companies = [e["company"] for e in entries]
        assert "없는회사" not in companies
        assert "삼성전자" in companies
        assert "SK하이닉스" in companies


# ---------------------------------------------------------------------------
# State persistence (_save_run + run_graph)
# ---------------------------------------------------------------------------

class TestSaveRun:
    def test_creates_file_in_runs_dir(self, tmp_path):
        from app.agent.graph import _save_run

        transitions = [
            {"analyze": {"entities": {"companies": ["삼성전자"]}, "route": "financial"}},
            {"generate": {"answer": {"answer": "테스트", "sources": [], "needs_review": False}}},
        ]
        final_state = {
            "route": "financial",
            "entities": {"companies": ["삼성전자"]},
            "tool_results": [],
            "retrieved_chunks": [],
            "needs_review": False,
        }

        with patch("app.agent.graph.config.RUNS_DIR", tmp_path):
            _save_run("테스트 질문", transitions, final_state)

        files = list(tmp_path.glob("run_*.json"))
        assert len(files) == 1

    def test_saved_file_contains_expected_fields(self, tmp_path):
        from app.agent.graph import _save_run

        with patch("app.agent.graph.config.RUNS_DIR", tmp_path):
            _save_run("삼성전자 질문", [], {"route": "rag", "entities": {}, "tool_results": [], "retrieved_chunks": [], "needs_review": False})

        data = json.loads(next(tmp_path.glob("run_*.json")).read_text())
        assert data["question"] == "삼성전자 질문"
        assert data["route"] == "rag"
        assert "timestamp" in data
        assert "node_sequence" in data

    def test_node_sequence_recorded(self, tmp_path):
        from app.agent.graph import _save_run

        transitions = [
            {"analyze": {"route": "financial"}},
            {"financial": {"tool_results": []}},
            {"calculate": {"computed": {}}},
            {"generate": {"answer": {}}},
        ]
        with patch("app.agent.graph.config.RUNS_DIR", tmp_path):
            _save_run("q", transitions, {"route": "financial", "entities": {}, "tool_results": [], "retrieved_chunks": [], "needs_review": False})

        data = json.loads(next(tmp_path.glob("run_*.json")).read_text())
        assert data["node_sequence"] == ["analyze", "financial", "calculate", "generate"]

    def test_does_not_raise_on_unserializable_data(self, tmp_path):
        from app.agent.graph import _save_run

        # object() is not JSON-serializable → default=str should handle it
        transitions = [{"analyze": {"weird": object()}}]
        with patch("app.agent.graph.config.RUNS_DIR", tmp_path):
            _save_run("q", transitions, {"route": "rag", "entities": {}, "tool_results": [], "retrieved_chunks": [], "needs_review": False})

        files = list(tmp_path.glob("run_*.json"))
        assert len(files) == 1  # file still written

    def test_does_not_raise_if_dir_missing(self):
        from app.agent.graph import _save_run
        from pathlib import Path

        with patch("app.agent.graph.config.RUNS_DIR", Path("/nonexistent/path/runs")):
            # must not raise
            _save_run("q", [], {"route": "rag", "entities": {}, "tool_results": [], "retrieved_chunks": [], "needs_review": False})


class TestRunGraphPersistence:
    def test_run_graph_saves_file_to_runs_dir(self, tmp_path):
        from app.agent.graph import run_graph
        from app.agent.state import QuestionAnalysis

        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=["매출액"], route="financial"
        )
        fact = _financial_fact()
        final_answer = AgentAnswer(answer="테스트 답변", sources=[], needs_review=False)

        with patch("app.agent.nodes.OpenAI") as MockLLM:
            client = MockLLM.return_value
            client.beta.chat.completions.parse.side_effect = [
                _llm_resp(analysis),
                _llm_resp(final_answer),
            ]
            with patch("app.agent.nodes.get_financial_statement", return_value=[fact]):
                with patch("app.agent.nodes.calculate_growth_rate", return_value=10.0):
                    with patch("app.agent.graph.config.RUNS_DIR", tmp_path):
                        result = run_graph("삼성전자 2024년 매출액은?")

        assert isinstance(result, AgentAnswer)
        files = list(tmp_path.glob("run_*.json"))
        assert len(files) == 1

    def test_run_file_records_node_sequence(self, tmp_path):
        from app.agent.graph import run_graph
        from app.agent.state import QuestionAnalysis

        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=["매출액"], route="financial"
        )
        final_answer = AgentAnswer(answer="테스트", sources=[], needs_review=False)

        with patch("app.agent.nodes.OpenAI") as MockLLM:
            client = MockLLM.return_value
            client.beta.chat.completions.parse.side_effect = [
                _llm_resp(analysis),
                _llm_resp(final_answer),
            ]
            with patch("app.agent.nodes.get_financial_statement", return_value=[]):
                with patch("app.agent.graph.config.RUNS_DIR", tmp_path):
                    run_graph("삼성전자 2024년 매출액은?")

        data = json.loads(next(tmp_path.glob("run_*.json")).read_text())
        # financial route: analyze → financial → calculate → generate
        assert "analyze" in data["node_sequence"]
        assert "generate" in data["node_sequence"]
