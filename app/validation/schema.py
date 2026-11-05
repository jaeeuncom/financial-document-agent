from typing import List, Optional

from pydantic import BaseModel


class FinancialFact(BaseModel):
    company: str
    corp_code: str
    year: int
    account: str
    fs_div: str  # "CFS" or "OFS"
    current_value: Optional[int]
    prior_value: Optional[int]
    currency: str = "KRW"
    rcept_no: str
    source_url: str


class AgentAnswer(BaseModel):
    answer: str
    sources: List[str]
    needs_review: bool = False
