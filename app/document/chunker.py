"""
Section-aware text chunker for RAG.

Each chunk stays within a single ParsedSection (never crosses section boundaries).
Overlap is applied between consecutive chunks so that sentences near chunk edges
appear in both chunks and can be retrieved from either.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from app.validation.schema import Chunk, ExtractionResult, ParsedSection

# Default chunking parameters
DEFAULT_TARGET = 800   # target chunk size in characters
DEFAULT_OVERLAP = 100  # overlap between consecutive chunks
DEFAULT_MIN = 300      # minimum useful chunk size (enforces forward progress)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_id(text_with_context: str) -> str:
    """16-char hex prefix of SHA-256 — unique per (company, year, section, text)."""
    return hashlib.sha256(text_with_context.encode("utf-8")).hexdigest()[:16]


def _find_break_point(text: str, pos: int, window: int = 150) -> int:
    """
    Find the last natural-break position in text[max(0, pos-window) : pos+len(pat)].
    Priority: paragraph break → Korean sentence end → English sentence end → newline → space.
    Falls back to hard break at pos if no boundary is found.
    """
    search_start = max(0, pos - window)

    for pat in ("\n\n", "다.\n", "요.\n", "다. ", "요. ", ".\n", ". ", "\n", " "):
        idx = text.rfind(pat, search_start, pos + len(pat))
        if idx != -1:
            return idx + len(pat)

    return pos  # hard break


def _split_text(
    text: str,
    target_size: int = DEFAULT_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    min_size: int = DEFAULT_MIN,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Rules:
    - Each chunk is at most target_size + window (≈target_size + 150) characters.
    - Each chunk is at least target_size // 2 characters (except the final tail chunk).
    - Consecutive chunks share ≈overlap characters for context continuity.
    - Breaks are sought at natural sentence/paragraph boundaries.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + target_size

        if end >= len(text):
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        raw_break = _find_break_point(text, end)

        # Enforce minimum chunk size: if natural break is too early, use hard break
        if raw_break < start + (target_size // 2):
            raw_break = end

        chunk = text[start:raw_break].strip()
        if chunk:
            chunks.append(chunk)

        # Next chunk starts overlap chars before the break point
        next_start = raw_break - overlap
        # Guard: always advance (prevents infinite loop when overlap ≥ chunk size)
        start = next_start if next_start > start else raw_break

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_section(
    section: ParsedSection,
    target_size: int = DEFAULT_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    min_size: int = DEFAULT_MIN,
) -> list[Chunk]:
    """
    Split one ParsedSection into Chunk objects.

    Each chunk's text_with_context is prefixed with '[company | year | section]'
    so that the embedding captures entity context even without metadata filters.
    """
    raw_chunks = _split_text(section.text, target_size, overlap, min_size)
    if not raw_chunks:
        return []

    total = len(raw_chunks)
    context_prefix = f"[{section.company} | {section.year} | {section.section}]\n"

    chunks: list[Chunk] = []
    for i, chunk_text in enumerate(raw_chunks):
        twc = context_prefix + chunk_text
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(twc),
                company=section.company,
                year=section.year,
                doc_type=section.doc_type,
                section=section.section,
                section_title=section.section_title,
                text=chunk_text,
                text_with_context=twc,
                char_count=len(chunk_text),
                chunk_index=i,
                total_chunks=total,
                rcept_no=section.rcept_no,
                source_url=section.source_url,
            )
        )

    return chunks


def chunk_extraction_result(
    result: ExtractionResult,
    target_size: int = DEFAULT_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    min_size: int = DEFAULT_MIN,
) -> list[Chunk]:
    """Chunk all sections in an ExtractionResult. Skips failed sections."""
    all_chunks: list[Chunk] = []
    for section in result.sections:
        all_chunks.extend(chunk_section(section, target_size, overlap, min_size))
    return all_chunks
