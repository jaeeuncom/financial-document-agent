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

## D-006: HTML parser — stdlib html.parser over beautifulsoup4
**Date:** 2026-11-05
**Decision:** Use Python's stdlib `html.parser.HTMLParser` for stripping HTML tags from DART documents.
**Reason:** Avoids adding a heavy dependency. DART HTML files are well-formed enough that a simple tag-stripper suffices. beautifulsoup4 can be added later if more robust parsing is needed (e.g., CSS selector queries during chunking in Day 4).

## D-007: Section detection — keyword match on first 1 500 characters only
**Date:** 2026-11-05
**Decision:** `_detect_section` checks only the first 1 500 characters of each HTML file for section-heading keywords.
**Reason:** DART HTML files start with the section heading. Checking the full file body risks false positives from body text that *refers to* section names (e.g., "자세한 내용은 사업의 내용 참조"). 1 500 chars covers any preamble before the heading.

## D-008: Section grouping — file-level consecutive grouping
**Date:** 2026-11-05
**Decision:** Each DART document zip is treated as an ordered list of HTML files. When a file's first 1 500 chars match a target-section keyword, that file starts a new section. Subsequent files with no match extend the current section.
**Reason:** DART annual reports split sections across many small HTML files (~40–200). Detecting sections per-file and grouping consecutive files is simpler and more robust than concatenating all text and splitting on regex patterns.
**Known limitation:** If two different target sections share a file (unusual), only the first keyword match wins.

## D-009: rcept_no lookup — financial-statement cache first, then list API
**Date:** 2026-11-05
**Decision:** `_get_rcept_no` reads `data/raw/dart/{corp_code}_{year}_single.json` (created in Day 2) before calling the DART `list.json` API.
**Reason:** Avoids a redundant API call when the financial-statement cache is already warm (the common path). Falls back to `list.json` when the cache is missing.

## D-010: Document zip cache
**Date:** 2026-11-05
**Decision:** Downloaded document zips are cached at `data/raw/dart/docs/{rcept_no}.zip`.
**Reason:** Annual-report zips are 2–10 MB and change rarely. Caching avoids repeated downloads during development and reduces DART rate-limit pressure. The zip is also needed for cross-validation in Day 11.
