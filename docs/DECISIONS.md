# Decision Log

## D-001: Primary data source for financial figures
**Date:** 2026-11-05
**Decision:** DART Open API is the sole primary source for all financial figures. PDF extraction is for cross-validation only.
**Reason:** DART is the authoritative regulatory filing system; it's structured and machine-readable. PDF extraction is unreliable for financial tables.

## D-002: corp_codes.json cache strategy
**Date:** 2026-11-05
**Decision:** Download the DART corp code zip once, filter to listed companies (non-empty stock_code), cache as JSON map {name: corp_code}.
**Reason:** Corp codes rarely change; a local cache avoids rate limits and startup latency. Invalidate manually when needed (delete data/corp_codes.json).

## D-003: account_nm exact-match strategy (Day 1)
**Date:** 2026-11-05
**Decision:** Use exact string match for account names in DART API responses.
**Reason:** Starting simple. Variants (e.g. "수익(매출액)" vs "매출액") will only be known after testing real API responses across all 5 companies. An alias map will be added in Day 2.

## D-004: CFS-only by default
**Date:** 2026-11-05
**Decision:** Keep only fs_div == "CFS" (consolidated) rows by default, as stated in README scope.
**Reason:** Consolidated statements represent the economic group; most analyst questions refer to consolidated figures.

## D-005: OPENAI_MODEL default
**Date:** 2026-11-05
**Decision:** Default model is gpt-4.1-mini as specified in README .env.example.
**Reason:** Balances capability and cost for a portfolio project; can be overridden via .env.
