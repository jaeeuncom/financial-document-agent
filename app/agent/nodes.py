"""LangGraph node functions for the financial document agent."""
from __future__ import annotations

import json

from openai import OpenAI

from app import config
from app.agent.prompts import ANALYZE_SYSTEM, GENERATE_SYSTEM
from app.agent.state import AgentState, QuestionAnalysis
from app.tools.dart_tool import calculate_growth_rate, get_financial_statement
from app.validation.review import create_review_request
from app.validation.rules import ReviewRequired
from app.validation.schema import AgentAnswer


def analyze_question(state: AgentState) -> dict:
    """Extract entities and route via structured LLM output."""
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.beta.chat.completions.parse(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": ANALYZE_SYSTEM},
                {"role": "user", "content": state["question"]},
            ],
            response_format=QuestionAnalysis,
        )
        analysis = response.choices[0].message.parsed
        if analysis is None:
            raise ValueError("no parsed response")
        return {
            "entities": {
                "companies": analysis.companies,
                "years": analysis.years,
                "accounts": analysis.accounts,
            },
            "route": analysis.route,
        }
    except Exception:
        return {
            "entities": {"companies": [], "years": [], "accounts": []},
            "route": "rag",
            "needs_review": True,
        }


def financial_node(state: AgentState) -> dict:
    """Fetch financial figures from DART for each company/year combination."""
    entities = state["entities"]
    companies = entities.get("companies", [])
    years = entities.get("years", [])
    accounts = entities.get("accounts") or None

    if not companies or not years:
        return {"tool_results": []}

    tool_results: list[dict] = []
    for company in companies:
        for year in years:
            try:
                facts = get_financial_statement(company, int(year), accounts)
                tool_results.extend(f.model_dump() for f in facts)
            except ReviewRequired as exc:
                review_path = create_review_request(
                    question=state["question"],
                    extracted={"company": company, "year": year, "accounts": accounts},
                    evidence={},
                    failed_rule=exc.rule,
                )
                tool_results.append({
                    "status": "needs_review",
                    "company": company,
                    "year": year,
                    "rule": exc.rule,
                    "detail": exc.detail,
                    "review_file": review_path,
                })

    return {"tool_results": tool_results}


def rag_node(state: AgentState) -> dict:
    """Search indexed annual report sections for narrative content."""
    from app.retrieval.vector_store import search_narrative

    entities = state["entities"]
    companies = entities.get("companies", [])
    years = entities.get("years", [])

    company = companies[0] if companies else None
    year = int(years[0]) if years else None

    results = search_narrative(
        query=state["question"],
        company=company,
        year=year,
        top_k=5,
        rerank=True,
    )
    return {"retrieved_chunks": [r.model_dump() for r in results]}


def calculate_node(state: AgentState) -> dict:
    """Compute growth rates and comparison table for all available financial facts."""
    from app.tools.compare_tool import compare_financial_data

    facts = [f for f in state.get("tool_results", []) if f.get("status") != "needs_review"]

    growth_rates: list[dict] = []
    for fact in facts:
        current = fact.get("current_value")
        prior = fact.get("prior_value")
        if current is None or prior is None:
            continue
        try:
            rate = calculate_growth_rate(int(current), int(prior))
            growth_rates.append({
                "company": fact.get("company"),
                "year": fact.get("year"),
                "account": fact.get("account"),
                "growth_rate_pct": rate,
            })
        except ReviewRequired:
            pass  # prior == 0: skip growth rate for this fact

    # Comparison table is useful when there are multiple companies or years
    unique_companies = {f.get("company") for f in facts}
    unique_years = {f.get("year") for f in facts}
    comparison = (
        compare_financial_data(facts)
        if len(unique_companies) > 1 or len(unique_years) > 1
        else []
    )

    return {"computed": {"growth_rates": growth_rates, "comparison": comparison}}


def generate_answer(state: AgentState) -> dict:
    """Synthesize final Korean answer from all collected data."""
    from app.retrieval.context import format_context
    from app.validation.schema import SearchResult

    facts = [f for f in state.get("tool_results", []) if f.get("status") != "needs_review"]
    errors = [f for f in state.get("tool_results", []) if f.get("status") == "needs_review"]
    growth_rates = state.get("computed", {}).get("growth_rates", [])
    comparison = state.get("computed", {}).get("comparison", [])

    chunks_raw = state.get("retrieved_chunks", [])
    try:
        search_results = [SearchResult(**c) for c in chunks_raw]
        narrative_text = format_context(search_results) if search_results else "(검색된 문서 없음)"
    except Exception:
        narrative_text = "(검색된 문서 없음)"

    source_urls = sorted({
        *(f["source_url"] for f in facts if f.get("source_url")),
        *(c["source_url"] for c in chunks_raw if c.get("source_url")),
    })

    has_errors = bool(errors) or state.get("needs_review", False)

    user_msg = (
        f"질문: {state['question']}\n\n"
        f"재무 데이터 (DART API):\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2) if facts else '(재무 데이터 없음)'}\n\n"
        f"성장률 계산:\n"
        f"{json.dumps(growth_rates, ensure_ascii=False, indent=2) if growth_rates else '(계산된 성장률 없음)'}\n\n"
        f"비교 분석 데이터:\n"
        f"{json.dumps(comparison, ensure_ascii=False, indent=2) if comparison else '(비교 데이터 없음)'}\n\n"
        f"사업보고서 서술 섹션:\n{narrative_text}\n\n"
        f"참고 가능한 출처 URL:\n"
        f"{chr(10).join(source_urls) if source_urls else '(없음)'}\n\n"
        f"검토 필요 항목: {'있음 — needs_review=true로 설정 필요' if has_errors else '없음'}"
    )

    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.beta.chat.completions.parse(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": GENERATE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format=AgentAnswer,
        )
        result = response.choices[0].message.parsed
        if result:
            return {"answer": result.model_dump()}
    except Exception:
        pass

    return {
        "answer": AgentAnswer(
            answer="응답을 생성할 수 없습니다. 사람의 검토가 필요합니다.",
            sources=[],
            needs_review=True,
        ).model_dump()
    }
