import json
from datetime import datetime, timezone
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
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return str(path)
