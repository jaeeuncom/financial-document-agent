"""Agent state and structured output models for the LangGraph pipeline."""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel


class QuestionAnalysis(BaseModel):
    """Structured output from the analyze_question node."""
    companies: list[str]
    years: list[int]
    accounts: list[str]
    route: Literal["financial", "rag", "both"]


class AgentState(TypedDict):
    question: str
    entities: dict           # {"companies": [...], "years": [...], "accounts": [...]}
    route: str               # "financial" | "rag" | "both"
    tool_results: list[dict] # FinancialFact dicts or {"status": "needs_review", ...}
    retrieved_chunks: list[dict]  # SearchResult dicts
    computed: dict           # {"growth_rates": [...]}
    answer: Optional[dict]   # AgentAnswer.model_dump()
    needs_review: bool
