"""Tests for app/agent/state.py, nodes.py, and graph.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agent.graph import (
    _route_after_analyze,
    _route_after_financial,
    _route_after_rag,
)
from app.agent.nodes import (
    analyze_question,
    calculate_node,
    financial_node,
    generate_answer,
    rag_node,
)
from app.agent.state import AgentState, QuestionAnalysis
from app.validation.rules import ReviewRequired
from app.validation.schema import AgentAnswer, FinancialFact, SearchResult


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


def _make_fact_dict(**overrides) -> dict:
    base = {
        "company": "삼성전자",
        "corp_code": "00126380",
        "year": 2024,
        "account": "매출액",
        "fs_div": "CFS",
        "current_value": 3008703000000,
        "prior_value": 2589355000000,
        "currency": "KRW",
        "rcept_no": "20250310000123",
        "source_url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250310000123",
    }
    base.update(overrides)
    return base


def _make_financial_fact(**overrides) -> FinancialFact:
    return FinancialFact(**_make_fact_dict(**overrides))


def _llm_resp(parsed) -> MagicMock:
    r = MagicMock()
    r.choices[0].message.parsed = parsed
    return r


# ---------------------------------------------------------------------------
# QuestionAnalysis model
# ---------------------------------------------------------------------------

class TestQuestionAnalysis:
    def test_financial_route_valid(self):
        a = QuestionAnalysis(companies=["삼성전자"], years=[2024], accounts=["매출액"], route="financial")
        assert a.route == "financial"

    def test_rag_route_valid(self):
        a = QuestionAnalysis(companies=[], years=[], accounts=[], route="rag")
        assert a.route == "rag"

    def test_both_route_valid(self):
        a = QuestionAnalysis(companies=["삼성전자"], years=[2024], accounts=[], route="both")
        assert a.route == "both"

    def test_invalid_route_raises(self):
        with pytest.raises(Exception):
            QuestionAnalysis(companies=[], years=[], accounts=[], route="unknown")


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

class TestRoutingFunctions:
    def test_financial_route_goes_to_financial(self):
        assert _route_after_analyze(_state(route="financial")) == "financial"

    def test_rag_route_goes_to_rag(self):
        assert _route_after_analyze(_state(route="rag")) == "rag"

    def test_both_route_goes_to_financial_first(self):
        assert _route_after_analyze(_state(route="both")) == "financial"

    def test_after_financial_goes_to_calculate_for_financial_route(self):
        assert _route_after_financial(_state(route="financial")) == "calculate"

    def test_after_financial_goes_to_rag_for_both_route(self):
        assert _route_after_financial(_state(route="both")) == "rag"

    def test_after_rag_goes_to_generate_for_rag_route(self):
        assert _route_after_rag(_state(route="rag")) == "generate"

    def test_after_rag_goes_to_calculate_for_both_route(self):
        assert _route_after_rag(_state(route="both")) == "calculate"


# ---------------------------------------------------------------------------
# analyze_question
# ---------------------------------------------------------------------------

class TestAnalyzeQuestion:
    def test_financial_route_extracted(self):
        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=["매출액"], route="financial"
        )
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = _llm_resp(analysis)
            result = analyze_question(_state())
        assert result["route"] == "financial"
        assert result["entities"]["companies"] == ["삼성전자"]
        assert result["entities"]["years"] == [2024]

    def test_rag_route_extracted(self):
        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=[], route="rag"
        )
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = _llm_resp(analysis)
            result = analyze_question(_state(question="삼성전자 사업 전략은?"))
        assert result["route"] == "rag"
        assert result["entities"]["accounts"] == []

    def test_both_route_extracted(self):
        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=["매출액"], route="both"
        )
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = _llm_resp(analysis)
            result = analyze_question(_state(question="매출 변화와 그 원인은?"))
        assert result["route"] == "both"

    def test_fallback_on_llm_error(self):
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.side_effect = RuntimeError("API error")
            result = analyze_question(_state())
        assert result["route"] == "rag"
        assert result["entities"]["companies"] == []
        assert result["needs_review"] is True

    def test_fallback_on_null_parsed(self):
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = _llm_resp(None)
            result = analyze_question(_state())
        assert result["route"] == "rag"
        assert result["needs_review"] is True


# ---------------------------------------------------------------------------
# financial_node
# ---------------------------------------------------------------------------

class TestFinancialNode:
    def test_calls_tool_and_returns_facts(self):
        fact = _make_financial_fact()
        with patch("app.agent.nodes.get_financial_statement", return_value=[fact]):
            result = financial_node(_state())
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["account"] == "매출액"
        assert result["tool_results"][0]["current_value"] == 3008703000000

    def test_handles_review_required(self):
        with patch("app.agent.nodes.get_financial_statement",
                   side_effect=ReviewRequired("company_not_found", "not found")):
            with patch("app.agent.nodes.create_review_request", return_value="/tmp/r.json"):
                result = financial_node(_state())
        assert result["tool_results"][0]["status"] == "needs_review"
        assert result["tool_results"][0]["rule"] == "company_not_found"

    def test_empty_companies_returns_empty(self):
        state = _state(entities={"companies": [], "years": [2024], "accounts": []})
        result = financial_node(state)
        assert result["tool_results"] == []

    def test_empty_years_returns_empty(self):
        state = _state(entities={"companies": ["삼성전자"], "years": [], "accounts": []})
        result = financial_node(state)
        assert result["tool_results"] == []

    def test_multiple_companies_calls_per_company(self):
        fact = _make_financial_fact()
        state = _state(entities={"companies": ["삼성전자", "SK하이닉스"], "years": [2024], "accounts": None})
        with patch("app.agent.nodes.get_financial_statement", return_value=[fact]) as mock:
            financial_node(state)
        assert mock.call_count == 2


# ---------------------------------------------------------------------------
# rag_node
# ---------------------------------------------------------------------------

class TestRagNode:
    def test_returns_retrieved_chunks(self):
        result_obj = SearchResult(
            chunk_id="a" * 16, company="삼성전자", year=2024,
            section="사업의내용", text="반도체 사업 현황",
            source_url="https://dart.fss.or.kr/test", score=0.85,
        )
        with patch("app.retrieval.vector_store.search_narrative", return_value=[result_obj]):
            result = rag_node(_state(route="rag",
                                     question="삼성전자 사업 전략",
                                     entities={"companies": ["삼성전자"], "years": [2024], "accounts": []}))
        assert len(result["retrieved_chunks"]) == 1
        assert result["retrieved_chunks"][0]["text"] == "반도체 사업 현황"

    def test_empty_results_returns_empty(self):
        with patch("app.retrieval.vector_store.search_narrative", return_value=[]):
            result = rag_node(_state())
        assert result["retrieved_chunks"] == []

    def test_uses_first_company_and_year(self):
        with patch("app.retrieval.vector_store.search_narrative", return_value=[]) as mock:
            rag_node(_state(entities={
                "companies": ["삼성전자", "SK하이닉스"],
                "years": [2024, 2023],
                "accounts": [],
            }))
        kwargs = mock.call_args.kwargs
        assert kwargs["company"] == "삼성전자"
        assert kwargs["year"] == 2024

    def test_no_company_passes_none(self):
        with patch("app.retrieval.vector_store.search_narrative", return_value=[]) as mock:
            rag_node(_state(entities={"companies": [], "years": [], "accounts": []}))
        kwargs = mock.call_args.kwargs
        assert kwargs["company"] is None
        assert kwargs["year"] is None


# ---------------------------------------------------------------------------
# calculate_node
# ---------------------------------------------------------------------------

class TestCalculateNode:
    def test_computes_growth_rate(self):
        state = _state(tool_results=[_make_fact_dict()])
        with patch("app.agent.nodes.calculate_growth_rate", return_value=16.19):
            result = calculate_node(state)
        rates = result["computed"]["growth_rates"]
        assert len(rates) == 1
        assert rates[0]["growth_rate_pct"] == 16.19
        assert rates[0]["account"] == "매출액"

    def test_skips_missing_prior_value(self):
        state = _state(tool_results=[_make_fact_dict(prior_value=None)])
        result = calculate_node(state)
        assert result["computed"]["growth_rates"] == []

    def test_skips_missing_current_value(self):
        state = _state(tool_results=[_make_fact_dict(current_value=None)])
        result = calculate_node(state)
        assert result["computed"]["growth_rates"] == []

    def test_skips_needs_review_entries(self):
        state = _state(tool_results=[{"status": "needs_review", "company": "삼성전자"}])
        result = calculate_node(state)
        assert result["computed"]["growth_rates"] == []

    def test_handles_division_by_zero(self):
        state = _state(tool_results=[_make_fact_dict(prior_value=0)])
        with patch("app.agent.nodes.calculate_growth_rate",
                   side_effect=ReviewRequired("division_by_zero", "prior is zero")):
            result = calculate_node(state)
        assert result["computed"]["growth_rates"] == []

    def test_empty_tool_results(self):
        result = calculate_node(_state(tool_results=[]))
        assert result["computed"]["growth_rates"] == []


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    def test_returns_agent_answer(self):
        answer = AgentAnswer(
            answer="삼성전자의 2024년 매출액은 300조 원입니다.",
            sources=["https://dart.fss.or.kr/test"],
            needs_review=False,
        )
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = _llm_resp(answer)
            result = generate_answer(_state(tool_results=[_make_fact_dict()]))
        assert "삼성전자" in result["answer"]["answer"]
        assert result["answer"]["needs_review"] is False

    def test_fallback_on_llm_error(self):
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.side_effect = RuntimeError("fail")
            result = generate_answer(_state())
        assert result["answer"]["needs_review"] is True

    def test_needs_review_flag_passed_in_context(self):
        answer = AgentAnswer(answer="검토 필요", sources=[], needs_review=True)
        state = _state(tool_results=[{
            "status": "needs_review", "company": "없는회사", "year": 2024,
            "rule": "company_not_found", "detail": "not found", "review_file": "/tmp/r.json",
        }])
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = _llm_resp(answer)
            result = generate_answer(state)
        # Verify LLM received user message mentioning needs_review
        call_args = MockLLM.return_value.beta.chat.completions.parse.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert "needs_review=true" in user_content


# ---------------------------------------------------------------------------
# run_graph (integration, all LLM and tool calls mocked)
# ---------------------------------------------------------------------------

class TestRunGraph:
    def test_financial_route_end_to_end(self):
        from app.agent.graph import run_graph

        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=["매출액"], route="financial"
        )
        fact = _make_financial_fact()
        final_answer = AgentAnswer(
            answer="삼성전자 2024년 매출액은 300조 원입니다.",
            sources=["https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250310000123"],
            needs_review=False,
        )

        with patch("app.agent.nodes.OpenAI") as MockLLM:
            client = MockLLM.return_value
            client.beta.chat.completions.parse.side_effect = [
                _llm_resp(analysis),   # analyze_question
                _llm_resp(final_answer),  # generate_answer
            ]
            with patch("app.agent.nodes.get_financial_statement", return_value=[fact]):
                with patch("app.agent.nodes.calculate_growth_rate", return_value=16.19):
                    result = run_graph("삼성전자 2024년 매출액은?")

        assert isinstance(result, AgentAnswer)
        assert result.needs_review is False
        assert "삼성전자" in result.answer

    def test_rag_route_end_to_end(self):
        from app.agent.graph import run_graph

        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=[], route="rag"
        )
        chunk = SearchResult(
            chunk_id="a" * 16, company="삼성전자", year=2024,
            section="사업의내용", text="반도체 전략...",
            source_url="https://dart.fss.or.kr/test", score=0.9,
        )
        final_answer = AgentAnswer(
            answer="삼성전자의 사업 전략은...",
            sources=["https://dart.fss.or.kr/test"],
            needs_review=False,
        )

        with patch("app.agent.nodes.OpenAI") as MockLLM:
            client = MockLLM.return_value
            client.beta.chat.completions.parse.side_effect = [
                _llm_resp(analysis),
                _llm_resp(final_answer),
            ]
            with patch("app.retrieval.vector_store.search_narrative", return_value=[chunk]):
                result = run_graph("삼성전자 사업 전략은?")

        assert isinstance(result, AgentAnswer)
        assert result.needs_review is False


class TestRunAgent:
    def test_no_api_key_returns_review(self):
        from app.agent.loop import run_agent
        import app.config as cfg

        original = cfg.OPENAI_API_KEY
        cfg.OPENAI_API_KEY = ""
        try:
            result = run_agent("삼성전자 매출액은?")
            assert result.needs_review is True
            assert "OPENAI_API_KEY" in result.answer
        finally:
            cfg.OPENAI_API_KEY = original
