from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
REVIEW_DIR = DATA_DIR / "review_queue"

DATA_DIR.mkdir(exist_ok=True)
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw" / "dart").mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DART_API_KEY: str | None = os.getenv("DART_API_KEY")
DART_BASE: str = "https://opendart.fss.or.kr/api"
