"""
OpenAI text-embedding-3-small embedder with on-disk JSON cache.

Cache location : data/embeddings_cache.json
Cache key      : SHA-256 of the text string
Cache format   : {"version": 1, "embeddings": {"<hex>": [float, ...], ...}}

Usage:
    from app.retrieval.embedding import embed_texts, embed_chunks

    vectors = embed_texts(["text A", "text B"])          # list[list[float]]
    vectors = embed_chunks(chunks)                        # uses chunk.text_with_context
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from openai import OpenAI

from app import config
from app.validation.rules import ReviewRequired

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
_CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Cache helpers  (DATA_DIR read at call time — safe to patch in tests)
# ---------------------------------------------------------------------------

def _cache_path():
    return config.DATA_DIR / "embeddings_cache.json"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, list[float]]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _CACHE_VERSION:
            return {}
        return data.get("embeddings", {})
    except Exception:
        return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    _cache_path().write_text(
        json.dumps({"version": _CACHE_VERSION, "embeddings": cache}, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# OpenAI API call
# ---------------------------------------------------------------------------

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API for a single batch. Raises ReviewRequired if key missing."""
    if not config.OPENAI_API_KEY:
        raise ReviewRequired("missing_api_key", "OPENAI_API_KEY is not set in .env")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
    )
    # API guarantees response.data items are ordered by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    batch_size: int = 100,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Embed a list of texts using text-embedding-3-small.

    Previously-seen texts are served from the on-disk cache without an API call.
    New texts are embedded in batches of `batch_size` and immediately cached.

    Returns a list of float vectors in the same order as the input.
    """
    cache = _load_cache()
    result: list[Optional[list[float]]] = [None] * len(texts)

    # Identify what's already cached
    to_embed: list[tuple[int, str]] = []
    for i, text in enumerate(texts):
        h = _text_hash(text)
        if h in cache:
            result[i] = cache[h]
        else:
            to_embed.append((i, text))

    if not to_embed:
        return result  # type: ignore[return-value]

    if show_progress:
        print(f"Embedding {len(to_embed)} new text(s) in batches of {batch_size}...")

    cache_dirty = False
    for batch_start in range(0, len(to_embed), batch_size):
        batch = to_embed[batch_start : batch_start + batch_size]
        indices = [b[0] for b in batch]
        batch_texts = [b[1] for b in batch]

        if show_progress:
            end = min(batch_start + batch_size, len(to_embed))
            print(f"  [{batch_start + 1}–{end}] calling OpenAI ...")

        embeddings = _embed_batch(batch_texts)

        for idx, text, embedding in zip(indices, batch_texts, embeddings):
            h = _text_hash(text)
            cache[h] = embedding
            result[idx] = embedding
            cache_dirty = True

    if cache_dirty:
        _save_cache(cache)

    return result  # type: ignore[return-value]


def embed_chunks(
    chunks: list,  # list[Chunk]
    batch_size: int = 100,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of Chunk objects.
    Uses chunk.text_with_context (includes [company | year | section] prefix).
    """
    texts = [c.text_with_context for c in chunks]
    return embed_texts(texts, batch_size=batch_size, show_progress=show_progress)
