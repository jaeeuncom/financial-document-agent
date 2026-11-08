"""Public entry point for the agent. Delegates to the LangGraph pipeline."""
from __future__ import annotations

from app import config
from app.agent.graph import run_graph
from app.validation.schema import AgentAnswer


def run_agent(question: str, max_steps: int = 6) -> AgentAnswer:
    """Run the financial document agent. max_steps is unused (LangGraph manages flow)."""
    if not config.OPENAI_API_KEY:
        return AgentAnswer(
            answer="OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.",
            sources=[],
            needs_review=True,
        )
    return run_graph(question)
