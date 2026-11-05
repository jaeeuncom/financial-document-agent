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


class ParsedSection(BaseModel):
    """One narrative section extracted from a DART annual report or PDF."""
    company: str
    year: int
    doc_type: str = "사업보고서"
    section: str              # canonical: 사업의내용 | MD&A | 연구개발 | 위험요소
    section_title: str        # detected heading / filename
    text: str
    page_or_order: int        # file order in zip, or PDF page number
    rcept_no: str
    source_url: str
    char_count: int = 0
    extraction_ok: bool = True
    extraction_error: Optional[str] = None


class ExtractionResult(BaseModel):
    """Extraction outcome for one company + year."""
    company: str
    year: int
    rcept_no: str
    sections: List[ParsedSection]
    total_chars: int
    failed_sections: List[str]   # target sections that were not found
    extraction_ok: bool
    error: Optional[str] = None


class Chunk(BaseModel):
    """
    A single RAG chunk derived from one ParsedSection.

    text              : raw chunk text (stored, displayed, returned to user)
    text_with_context : '[company | year | section]\\n' + text — used for embedding
    chunk_id          : sha256(text_with_context)[:16] — cache key for embeddings
    """
    chunk_id: str
    company: str
    year: int
    doc_type: str
    section: str            # canonical: 사업의내용 | MD&A | 연구개발 | 위험요소
    section_title: str
    text: str
    text_with_context: str
    char_count: int
    chunk_index: int        # 0-based within the parent section
    total_chunks: int       # total chunks from the same section
    rcept_no: str
    source_url: str
