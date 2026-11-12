from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app import config


def create_review_request(
    question: str,
    extracted: dict,
    evidence: dict,
    failed_rule: str,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = config.REVIEW_DIR / f"review_{timestamp}.json"
    payload = {
        "run_id": timestamp,
        "question": question,
        "extracted": extracted,
        "evidence": evidence,
        "failed_rule": failed_rule,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return str(path)


def list_reviews(review_dir: Optional[Path] = None) -> list[dict]:
    """Return all review items sorted by created_at (oldest first)."""
    directory = review_dir if review_dir is not None else config.REVIEW_DIR
    items: list[dict] = []
    for path in sorted(directory.glob("review_*.json")):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return sorted(items, key=lambda x: x.get("created_at", ""))


def resolve_review(review_id: str, review_dir: Optional[Path] = None) -> dict:
    """
    Mark a review item as resolved.

    Parameters
    ----------
    review_id:
        The run_id value (e.g. '20240101T120000000000') or the full filename
        stem (e.g. 'review_20240101T120000000000').
    review_dir:
        Directory to search; defaults to config.REVIEW_DIR.

    Returns the updated review dict.
    Raises FileNotFoundError if no matching review is found.
    """
    directory = review_dir if review_dir is not None else config.REVIEW_DIR

    # Accept both "20240101T..." and "review_20240101T..."
    stem = review_id if review_id.startswith("review_") else f"review_{review_id}"
    path = directory / f"{stem}.json"

    if not path.exists():
        raise FileNotFoundError(f"Review not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "resolved"
    data["resolved_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data
