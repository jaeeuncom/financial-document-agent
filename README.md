# Financial Document AI Agent

An LLM agent that answers questions about Korean listed companies using **DART disclosures**.
Numbers come from **tools** (DART Open API), explanations come from **RAG** (business report text),
and every answer is **validated, source-cited, and evaluated**. Failed validation is never "fixed" by the LLM; it is escalated to a human reviewer.

> Purpose: portfolio project for the PwC Samil (삼일회계법인) AI Lab internship application.
> Goal: demonstrate LLM/RAG/Tool Calling/MCP engineering, data processing, agent design, and quality evaluation on a professional-knowledge domain.
> Timeline: 14 days, 4–6 h/day. Working language: code, comments, commit messages, docs in English. Chat explanations with the owner in Korean.

---

## 0. Instructions for Claude Code (read first)

You are implementing this project end to end. Follow these rules in every session.

1. **Read this README fully, then check "Current Status" (section 3) before doing anything.** Continue from the first unchecked item.
2. **Work in vertical slices.** Keep the pipeline runnable end to end at all times (question -> tool -> cited answer). Never leave the repo in a state where `python -m app.main "<question>"` is broken at the end of a session.
3. **Design principles (non-negotiable):**
   - Numbers = tools. The LLM must never produce financial figures from memory. All figures come from tool results.
   - Narrative = RAG. Do not vectorize financial statements; vectorize only narrative sections of business reports.
   - PDF table extraction is used only for cross-validating DART API values, never as the primary number source.
   - Validation failure -> preserve original data + validation result -> create a review request JSON. Never let the LLM silently repair or guess.
   - Every answer carries `sources` (DART URLs / document + page or section). No source -> re-search or human review.
4. **Evaluation starts early and grows daily.** Add 3–5 eval questions per day from Day 4 onward. Re-run the eval after every meaningful change and record the score in `docs/eval_log.md`.
5. **Log technical challenges as they happen** in `docs/TECHNICAL_CHALLENGES.md` (problem -> cause -> fix -> result). Required topics are listed in section 9.
6. **Do not commit secrets.** `.env` is gitignored; keep `.env.example` updated.
7. **Prefer complete, runnable files over snippets.** Add or update tests when you add a module. Keep functions small and typed (Pydantic models for all data crossing module boundaries).
8. **If a requirement is ambiguous or an external API behaves unexpectedly, record it in `docs/DECISIONS.md` and pick the simplest option**, then tell the owner in your final message of the session.
9. **At the end of each session**: update the checkboxes in section 3, summarize what changed, what was verified (commands run + results), and what is next.
10. **Never verify against the LLM's own output.** Ground-truth values for the eval set must be checked against the DART viewer (dart.fss.or.kr) by the owner or by comparing with the raw API JSON.

### Git rules (enforced every session)

11. **Commit when a work unit is complete and tests pass.** One commit per feature or per Day. Use conventional commit messages in English: `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`.
12. **Check before every commit.** Run `git status` and confirm that `.env`, `data/`, API key strings (`sk-`, `crtfc_key` values), and `study_notes/` are not staged. If any are found, do not commit — stop and notify the owner.
13. **Push once per Day, only when all tests pass.** Never push a broken state.
14. **No force push or history rewriting** (`rebase`, `reset --hard`) without the owner's explicit permission.
15. **Include in every session summary:** commit hash, list of changed files, and push result.

### Study notes rules (private learning documents)

16. **Write `study_notes/day-XX_<topic>.md` at the end of each Day or large feature.** Example: `study_notes/day-01_dart-tool-and-agent-loop.md`.
17. **`study_notes/` is gitignored and must never be committed or pushed.** If it appears in `git status` before a commit, stop immediately and notify the owner.
18. **Never write API keys, tokens, or `.env` contents in notes.**
19. **Language:** explanations in Korean; code, filenames, and technical terms in English.
20. **Audience:** yourself a few weeks later having forgotten the details — explain even unfamiliar concepts with analogies, as if reading for the first time.

    Note format (follow in order):
    1. 오늘 한 일 (3–5줄 요약)
    2. 왜 이렇게 설계했는지 (선택한 방식, 버린 대안, 이유)
    3. 핵심 개념 설명 (비유와 함께)
    4. 만든/수정한 파일과 각 파일의 역할
    5. 데이터가 흐르는 과정 (입력 → 처리 → 출력, 텍스트 다이어그램 포함)
    6. 직접 실행해 보는 방법과 기대 결과
    7. 겪은 문제와 해결 과정 (원인 → 해결 → 배운 점)
    8. 면접 예상 질문 3개와 답변 포인트
    9. 다음 단계에서 이어질 내용

---

## 1. Problem and Approach

**Problem.** Annual reports mix structured data (financial statements) and unstructured text (business description, risks, MD&A). Finding information and comparing figures across documents is slow. Plain LLMs hallucinate content, mis-extract numbers, and confuse figures across documents.

**Approach.** An agent that analyzes the question, picks the right tool or retrieval path, executes it, validates the result, and answers with evidence.

| Data type | Source of truth | Access |
|---|---|---|
| Financial figures (revenue, operating profit, assets, ...) | DART Open API (`fnlttSinglAcnt`, `fnlttSinglAcntAll`) | Tool calling |
| Narrative (business, risks, MD&A, R&D) | DART business report original text (`document.xml`) | RAG |
| Cross-check | Uploaded/downloaded PDF tables (PyMuPDF, pdfplumber) | Validation only |

**Scope (keep it small so ground truth stays manageable).**
- Companies (5): 삼성전자, SK하이닉스, 현대자동차, NAVER, LG전자
- Fiscal years: 2023–2025 annual reports (report code `11011`). Confirm 2025 availability via API first; otherwise use 2022–2024.
- Consolidated statements (`CFS`) by default.
- Excluded for now: banks/insurers (different account names), quarterly reports.

---

## 2. Architecture

```
User question (+ optional PDF upload)
        |
        v
[1] LangGraph Agent
    analyze_question -> route -> { financial_tool | rag_search | both }
                              -> calculate -> verify -> { answer | regenerate | human_review }
        |
        +--> Financial Tool  : DART API (structured figures)
        +--> Document Search : RAG over business report narrative (Chroma)
        +--> Calculation Tool: growth rate, ratios, period comparison
        +--> MCP Tools       : same tools exposed via MCP server (extension)
        |
        v
[2] Validation & Evaluation
    required fields, domain rules, cross-document consistency, source check, answer check
        |
        v
[3] Response  |  Human Review (JSON review request)
```

### Agent state (`app/agent/state.py`)

```python
class AgentState(TypedDict):
    question: str
    entities: dict          # {"companies": [...], "years": [...], "accounts": [...]}
    route: str              # "financial" | "rag" | "both"
    tool_results: list[dict]
    retrieved_chunks: list[dict]
    computed: dict
    validation: dict        # {"passed": bool, "failed_rules": [...]}
    answer: dict | None     # AgentAnswer
    retries: int
    needs_review: bool
```

### Tools

| Tool | Purpose |
|---|---|
| `get_financial_statement` | DART structured figures (current and prior year) with source URL |
| `search_financial_document` | RAG search over narrative sections, with metadata filters (company, year, section) |
| `calculate_growth_rate` | YoY change (%) |
| `compare_financial_data` | Multi-period / multi-company comparison table |
| `verify_source` | Compare extracted value with original document text/table |

The LLM chooses tools and interprets results; it does not compute or recall figures itself.

---

## 3. Current Status (update every session)

**The repository starts empty (only this README exists). Nothing is implemented yet. Build everything from Day 1.**

**Day 1 checklist (vertical slice):**
- [x] Project skeleton, `requirements.txt`, `.env.example`, `.gitignore`, `app/config.py`
- [x] `app/validation/schema.py` (`FinancialFact`, `AgentAnswer`)
- [x] `app/validation/rules.py` (`ReviewRequired`, missing-field and negative-value checks)
- [x] `app/validation/review.py` (review request JSON into `data/review_queue/`)
- [x] `app/tools/dart_tool.py` (corp code cache, `get_financial_statement`, `calculate_growth_rate`)
- [x] `app/agent/loop.py` (simple OpenAI function-calling loop, structured final answer)
- [x] `app/main.py` CLI: `python -m app.main "<question>"`
- [x] Unit tests for rules and DART parsing (HTTP mocked)

**Owner prerequisites (the owner provides these in `.env`; never ask for keys in chat, never print or commit them):**
`OPENAI_API_KEY`, `DART_API_KEY` (issued at opendart.fss.or.kr). If a key is missing, stop at the first step that needs it, finish everything that does not need it (code, mocked tests), and tell the owner exactly what to add.

**Next (in order):** follow the day plan in section 4, starting at Day 1.

**Day 1 acceptance checks (confirm before moving on):**
- [ ] "삼성전자의 2024년 매출액과 영업이익을 알려주고 전년 대비 변화를 설명해 줘" returns figures, growth rate, and DART source URL — **pending: requires .env keys**
- [ ] Unknown company name creates a JSON in `data/review_queue/` — **pending: requires .env keys**
- [ ] Samsung 2024 revenue matches the DART viewer value — **pending: owner to verify**

---

## 4. Day Plan

Format: **Goal -> Tasks -> Deliverable -> Done when**.

### Day 1–3: Thin end-to-end slice + evaluation seed

**Day 1 — Skeleton + DART tool + minimal agent (build from scratch)**

Goal: one question goes through an LLM tool-calling loop, hits one DART tool, and returns a structured, source-cited answer.

Create these files (complete and runnable, typed, no placeholders):

- `requirements.txt`: `openai>=1.60`, `pydantic>=2.6`, `requests>=2.31`, `python-dotenv>=1.0`, `pytest`
- `.env.example`: `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-4.1-mini`, `DART_API_KEY`
- `.gitignore`: `.env`, `.venv/`, `__pycache__/`, `data/` (keep `data/.gitkeep`)
- `app/config.py`: load `.env`; expose `ROOT`, `DATA_DIR`, `REVIEW_DIR` (create dirs), `OPENAI_API_KEY`, `OPENAI_MODEL`, `DART_API_KEY`, `DART_BASE = "https://opendart.fss.or.kr/api"`
- `app/validation/schema.py`
  - `FinancialFact`: `company, corp_code, year:int, account, fs_div ("CFS"/"OFS"), current_value:int|None, prior_value:int|None, currency="KRW", rcept_no, source_url`
  - `AgentAnswer`: `answer:str` (Korean, uses only tool-result figures), `sources:list[str]`, `needs_review:bool=False`
- `app/validation/rules.py`
  - `ReviewRequired(Exception)` with `.rule` and `.detail`
  - `check_missing(requested, facts)`: raise `required_field_missing` listing absent accounts
  - `check_domain(fact)`: raise `domain_negative_value` if 매출액/자산총계/자본금 is negative (current or prior)
- `app/validation/review.py`: `create_review_request(question, extracted, evidence, failed_rule) -> path`; writes `data/review_queue/review_<timestamp>.json` with `{question, extracted, evidence, failed_rule, created_at}`
- `app/tools/dart_tool.py`
  - `_load_corp_codes()`: download `{DART_BASE}/corpCode.xml?crtfc_key=...` (zip containing XML), keep **listed companies only** (non-empty `stock_code`), map `corp_name -> corp_code`, cache to `data/corp_codes.json`
  - `resolve_corp_code(company)`: exact name match, else raise `ReviewRequired("company_not_found")`
  - `get_financial_statement(company, year, accounts=None) -> list[FinancialFact]`: call `fnlttSinglAcnt.json` with `reprt_code=11011`; non-`000` status -> `ReviewRequired("dart_api_error")`; keep only `fs_div == "CFS"` rows whose `account_nm` is requested (default: 매출액, 영업이익, 당기순이익), first match per account; parse `thstrm_amount` / `frmtrm_amount` (strip commas, `-` -> None); `source_url = https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}`; then run `check_missing` and `check_domain`
  - `calculate_growth_rate(current, previous)`: `(current - previous) / abs(previous) * 100`, rounded to 2 decimals; previous == 0 -> `ReviewRequired("division_by_zero")`
- `app/agent/loop.py`
  - OpenAI client; two function tools: `get_financial_statement(company, year, accounts?)` and `calculate_growth_rate(current, previous)`
  - System prompt rules: figures only from tool results; calculations only via the calculation tool; if a tool returns `status=needs_review`, do not guess, say review is needed and set `needs_review=true`; amounts are in KRW, show readable 조/억 units plus the raw KRW value; `sources` only from tool `source_url`
  - Tool dispatcher catches `ReviewRequired`, calls `create_review_request`, and returns `{"status": "needs_review", "rule", "detail", "review_file"}` to the model
  - `run_agent(question, max_steps=6) -> AgentAnswer`; use `client.beta.chat.completions.parse(..., tools=TOOLS, response_format=AgentAnswer)`; when `message.tool_calls` is empty return `message.parsed`; exceeding max steps -> `needs_review=True`
- `app/main.py`: argparse CLI, prints answer, `[출처]` list, and a review warning if `needs_review`
- `tests/test_rules.py`, `tests/test_dart_tool.py`: rules and amount parsing/CFS filtering/growth rate with mocked `requests.get` (no network, no keys needed)

Done when (all three must be verified by actually running, not assumed):
1. `pytest -q` passes.
2. `python -m app.main "삼성전자의 2024년 매출액과 영업이익을 알려주고 전년 대비 변화를 설명해 줘"` prints figures, growth rate, and a DART source URL.
3. `python -m app.main "없는회사의 2024년 매출액은?"` creates a JSON file in `data/review_queue/` and the answer says review is needed.

Also ask the owner to compare Samsung 2024 revenue with the DART viewer. Note API quirks found (account name variants, missing years) in `docs/DECISIONS.md`.

**Day 2 — Robust financial tool + evaluation seed**
- Add company alias map (`data/aliases.json`: "삼전" -> "삼성전자", "SK hynix" -> "SK하이닉스", ...) and resolve in `resolve_corp_code`.
- Extend accounts: 자산총계, 부채총계, 자본총계 (BS), 매출액, 영업이익, 당기순이익 (IS). Handle account-name variants per company (e.g. "수익(매출액)").
- Add `fnlttSinglAcntAll` support for detailed statements (optional, behind a flag).
- Cache raw API responses to `data/raw/dart/{corp_code}_{year}.json` (also needed for review evidence).
- Create `tests/eval_set.json` with the first 10 questions (schema in section 7). Ground truth from the DART viewer.
- Write `app/evaluation/metrics.py` + `evaluator.py` for numeric exact-match and a CLI: `python -m app.evaluation.evaluator`.
- Unit tests for amount parsing, growth calculation, domain rules (`tests/test_tools.py`, `tests/test_rules.py`).
- Done when: eval runs on 10 questions and prints accuracy; tests pass.

**Day 3 — Narrative document collection**
- Add `app/document/parser.py`: download business report original via DART `document.xml` (zip), parse the XML/HTML, and split into sections (II. 사업의 내용, III. 재무에 관한 사항 > MD&A ("이사의 경영진단 및 분석의견"), risk factors, R&D).
- Add `app/document/extractor.py`: metadata per section `{company, year, doc_type, section, page_or_order, rcept_no, source_url}`; detect empty/failed extraction.
- Store parsed sections at `data/parsed/{company}_{year}.json`.
- PDF path (secondary): `parse_pdf()` with PyMuPDF for text and pdfplumber for tables, used for upload and cross-validation.
- Done when: 5 companies x 3 years parsed with section metadata; extraction failures are logged, not silent.

### Day 4–7: RAG on narrative text

**Day 4 — Chunking + embedding**
- `app/document/chunker.py`: section-aware chunking (≈600–1000 chars, overlap ≈100), never split across sections, keep heading in chunk text, attach metadata.
- `app/retrieval/embedding.py`: OpenAI `text-embedding-3-small`, batched, with on-disk cache.
- Add 5 eval questions (narrative questions with expected source section).

**Day 5 — Vector store + search**
- `app/retrieval/vector_store.py`: Chroma persistent store at `data/chroma/`, metadata filters (company, year, section).
- `app/tools/search_tool.py`: `search_financial_document(query, company, year, section, k)` returns chunks with metadata and scores.
- Retrieval metrics: Recall@K and Precision@K against labeled relevant sections.
- Add 5 eval questions.

**Day 6 — RAG pipeline**
- Query -> entity extraction -> filtered retrieval -> answer generation grounded in chunks only, with citation of `(company, year, section, source_url)`.
- Answer must say "not found in the provided documents" when retrieval is weak (similarity threshold + LLM grounding check).
- Add 5 eval questions.

**Day 7 — Evaluation checkpoint 1**
- Run full eval (≥ 25 questions). Record: numeric accuracy, Recall@K, Precision@K, source match, hallucination rate in `docs/eval_log.md`.
- Analyze failures; fix the top 2 issues; log them in `docs/TECHNICAL_CHALLENGES.md` (include the bad retrieval example).

### Day 8–10: LangGraph agent

**Day 8 — Graph skeleton**
- `app/agent/state.py`, `nodes.py`, `graph.py`, `prompts.py`.
- Nodes: `analyze_question` (structured entity extraction via Pydantic), `route`, `financial_node`, `rag_node`, `calculate_node`, `generate_answer`.
- Replace `app/agent/loop.py` with the graph, but keep CLI behavior identical.

**Day 9 — Conditional flow**
- Routing: numeric question -> financial tool; explanation/why/risk question -> RAG; mixed -> both (e.g., "매출 변화와 그 원인").
- Multi-company / multi-year questions: loop over entities, assemble `compare_financial_data` results.
- State persistence for debugging: write each run's state transitions to `data/runs/{timestamp}.json`.

**Day 10 — Tool calling hardening**
- Tool argument validation (Pydantic), retries with backoff for DART errors (status `020` rate limit, `013` no data), clear error objects.
- `calculate_growth_rate`, `compare_financial_data` implemented as pure functions with tests.
- Add 5 eval questions on tool selection (including questions that should NOT call a tool).

### Day 11–12: Verification and human review

**Day 11 — Validation layer**
- `app/validation/rules.py` implements the table in section 6.
- `verify_source` tool: re-fetch/parse original and compare with extracted value; mismatch -> `ReviewRequired`.
- Cross-document validation: DART API value vs PDF/table value, and current-year `thstrm` vs next-year report's `frmtrm` (prior-year comparative) — mismatches indicate restatement and must be flagged.

**Day 12 — Review flow**
- Graph edge `verify -> human_review` creating the review JSON (`{question, extracted, evidence, failed_rule}` + run id).
- Answer-level check: re-ask the model (separate call) whether the answer is supported by the tool results; failure -> regenerate once, then review.
- Optional: simple review CLI to list/resolve items in `data/review_queue/`.

### Day 13–14: Evaluation report and polish

**Day 13 — Final evaluation**
- Expand eval set to 30–50 questions (categories in section 7), including adversarial cases (unknown company, missing year, negative-value injection, mismatched figures).
- Generate `docs/EVALUATION_REPORT.md` with per-metric tables and a failure analysis.

**Day 14 — Integration and portfolio**
- Integration tests (`tests/test_e2e.py`), clean README (portfolio version), architecture diagram (`docs/architecture.png` or Mermaid), demo script for a screen recording.
- Finalize `docs/TECHNICAL_CHALLENGES.md`.

### Extensions (only after the required items are complete)

1. **MCP server** (`mcp_server/server.py`, MCP Python SDK): wrap the existing tool functions (`get_financial_statement`, `search_financial_document`, `calculate_growth_rate`) without duplicating logic. Document how to connect a client. Est. 0.5–1 day.
2. **FastAPI** (`app/main.py`, `app/api/endpoints.py`): `POST /ask`, `POST /documents`, `GET /reviews`. Optional minimal web UI.
3. **Cross-document validation** dashboards and log analysis.

---

## 5. Project Structure

```
financial-document-agent/
├── app/
│   ├── main.py                  # CLI now; FastAPI later
│   ├── config.py
│   ├── api/endpoints.py
│   ├── agent/                   # graph.py, state.py, nodes.py, prompts.py (loop.py = Day 1 version)
│   ├── document/                # parser.py, chunker.py, extractor.py
│   ├── retrieval/               # embedding.py, vector_store.py
│   ├── tools/                   # dart_tool.py, search_tool.py, calculation_tool.py, verification_tool.py
│   ├── validation/              # schema.py, rules.py, review.py
│   └── evaluation/              # dataset.py, metrics.py, evaluator.py
├── mcp_server/server.py
├── tests/
├── data/                        # raw/, parsed/, chroma/, runs/, review_queue/ (gitignored except samples)
├── docs/                        # DECISIONS.md, TECHNICAL_CHALLENGES.md, eval_log.md, EVALUATION_REPORT.md
├── requirements.txt
├── .env.example
└── README.md
```

Dependencies to add over time: `langgraph`, `chromadb`, `pymupdf`, `pdfplumber`, `pandas`, `fastapi`, `uvicorn`, `pytest`, `mcp`.

---

## 6. Validation Rules

| Stage | Failure | Action |
|---|---|---|
| Text extraction | Parse failure / empty section | Review request |
| Data validation | Required field missing | Review request |
| Domain validation | Negative revenue/assets, absurd magnitude change (e.g., > 1000% YoY, flag don't reject) | Review request |
| Cross-document | Values differ across documents (API vs PDF; `thstrm` vs next year's `frmtrm`) | Review request with both values |
| Source validation | Answer's evidence cannot be located | Re-search once, then review |
| Answer validation | Answer unsupported by tool results or off-question | Regenerate once, then review |

**Review request JSON** (stored in `data/review_queue/`):

```json
{
  "run_id": "...",
  "question": "...",
  "extracted": {"company": "삼성전자", "year": 2024, "account": "매출액", "value": 0},
  "evidence": {"source_url": "...", "raw_excerpt": "...", "other_value": null},
  "failed_rule": "cross_document_mismatch",
  "created_at": "..."
}
```

The original data and the validation result must be preserved unmodified.

---

## 7. Evaluation

Start simple: 30–50 hand-labeled questions, expected vs actual comparison. No heavy framework.

**`tests/eval_set.json` schema**

```json
{
  "id": "fin-001",
  "category": "numeric | narrative | mixed | tool_selection | adversarial",
  "question": "삼성전자의 2024년 매출액은?",
  "expected": {
    "values": {"매출액": 0},
    "must_contain": [],
    "expected_tool": "get_financial_statement",
    "expected_sources": ["dart.fss.or.kr"],
    "should_need_review": false
  },
  "ground_truth_note": "Checked against DART viewer on YYYY-MM-DD"
}
```

**Metrics**

| Area | Metric |
|---|---|
| Document extraction | Field extraction accuracy |
| Retrieval | Recall@K, Precision@K |
| Answer | Numeric exact-match / answer correctness |
| Evidence | Source match rate |
| Validation | Error detection rate on adversarial cases |
| Reliability | Wrong-answer (hallucination) rate; correct "needs review" rate |

Log every run in `docs/eval_log.md`: date, git commit, change made, scores. Aim for visible improvements across runs; this log feeds the Evaluation Report.

---

## 8. Commands

```bash
pip install -r requirements.txt
cp .env.example .env        # OPENAI_API_KEY, OPENAI_MODEL, DART_API_KEY

python -m app.main "삼성전자의 2024년 매출액과 영업이익을 알려주고 전년 대비 변화를 설명해 줘"
python -m app.evaluation.evaluator
pytest -q
```

DART API notes:
- Corp codes: `corpCode.xml` (zip) -> cached in `data/corp_codes.json` (listed companies only).
- Key endpoints: `fnlttSinglAcnt.json` (main accounts), `fnlttSinglAcntAll.json` (full statements), `list.json` (disclosure search), `document.xml` (original document zip).
- Report codes: `11011` annual, `11012` half-year, `11013` Q1, `11014` Q3.
- Amounts are strings in KRW with commas; `-` means no value. Respect rate limits; cache raw responses.

---

## 9. Portfolio Deliverables

1. **README** — purpose, stack, how to run (portfolio version of this file).
2. **Architecture** — diagram and agent workflow explanation.
3. **Demo** — video from question to tool execution to cited answer, including a review-request case.
4. **Evaluation Report** — dataset description, metrics, before/after improvements, failure analysis.
5. **Technical Challenges** (`docs/TECHNICAL_CHALLENGES.md`) — must include:
   - Difficulty extracting accurate numbers and tables from PDFs (and why DART API became the primary source)
   - RAG returning irrelevant documents (cause, metadata filtering/chunking fix, before/after Recall@K)
   - Agent picking the wrong tool or answering without evidence (cause, prompt/validation fix, before/after rates)

---

## 10. Priorities

**Must complete:** DART tool, narrative parsing, embedding + vector search, RAG QA, LangGraph agent, source citation, basic evaluation, validation + review requests.

**If time remains:** MCP server, cross-document validation, FastAPI + simple UI, automatic evaluation and log analysis.

Cut order if behind schedule: web UI -> MCP -> cross-document validation. Never cut evaluation or source citation.
