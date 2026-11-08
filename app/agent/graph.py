"""LangGraph StateGraph for the financial document agent.

Flow:
  financial  →  analyze → financial_node → calculate_node → generate_answer
  rag        →  analyze → rag_node → generate_answer
  both       →  analyze → financial_node → rag_node → calculate_node → generate_answer
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    analyze_question,
    calculate_node,
    financial_node,
    generate_answer,
    rag_node,
)
from app.agent.state import AgentState
from app.validation.schema import AgentAnswer


# ---------------------------------------------------------------------------
# Pure routing functions (read state["route"], return destination key)
# ---------------------------------------------------------------------------

def _route_after_analyze(state: AgentState) -> str:
    return "financial" if state.get("route") in ("financial", "both") else "rag"


def _route_after_financial(state: AgentState) -> str:
    return "rag" if state.get("route") == "both" else "calculate"


def _route_after_rag(state: AgentState) -> str:
    return "generate" if state.get("route") == "rag" else "calculate"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("analyze", analyze_question)
    builder.add_node("financial", financial_node)
    builder.add_node("rag", rag_node)
    builder.add_node("calculate", calculate_node)
    builder.add_node("generate", generate_answer)

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges("analyze", _route_after_analyze, {
        "financial": "financial",
        "rag": "rag",
    })
    builder.add_conditional_edges("financial", _route_after_financial, {
        "rag": "rag",
        "calculate": "calculate",
    })
    builder.add_conditional_edges("rag", _route_after_rag, {
        "generate": "generate",
        "calculate": "calculate",
    })
    builder.add_edge("calculate", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# Lazy singleton — built on first call so imports stay fast
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_graph(question: str) -> AgentAnswer:
    """Invoke the LangGraph agent and return a structured AgentAnswer."""
    initial: AgentState = {
        "question": question,
        "entities": {"companies": [], "years": [], "accounts": []},
        "route": "rag",
        "tool_results": [],
        "retrieved_chunks": [],
        "computed": {},
        "answer": None,
        "needs_review": False,
    }
    final = _get_graph().invoke(initial)
    answer_data = final.get("answer")
    if answer_data:
        return AgentAnswer(**answer_data)
    return AgentAnswer(
        answer="응답 생성에 실패했습니다.",
        sources=[],
        needs_review=True,
    )
