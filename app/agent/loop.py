from __future__ import annotations

import json

from openai import OpenAI

from app import config
from app.validation.schema import AgentAnswer
from app.validation.review import create_review_request
from app.tools.dart_tool import get_financial_statement, calculate_growth_rate
from app.validation.rules import ReviewRequired

# strict=True is required when using response_format (structured output) with beta.parse.
# Strict mode also requires additionalProperties=False and all properties listed in required.
# Optional parameters use anyOf with null instead of being omitted from required.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_statement",
            "strict": True,
            "description": (
                "Retrieve financial figures for a Korean listed company from the DART API. "
                "Returns current-year and prior-year values with a DART source URL. "
                "Use this for income statement accounts (매출액, 영업이익, 당기순이익) "
                "and balance sheet accounts (자산총계, 부채총계, 자본총계). "
                "Company name accepts aliases (e.g. '삼전', '현대차', '네이버', 'SK hynix')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "Korean company name as registered in DART (e.g. '삼성전자')",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Fiscal year of the annual report (e.g. 2024)",
                    },
                    "accounts": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "description": (
                            "Account names to retrieve (Korean). "
                            "Pass null to use defaults: 매출액, 영업이익, 당기순이익."
                        ),
                    },
                },
                "required": ["company", "year", "accounts"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_growth_rate",
            "strict": True,
            "description": (
                "Calculate year-over-year growth rate as a percentage. "
                "Must be used for ALL growth rate or change calculations — never compute manually."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current": {
                        "type": "number",
                        "description": "Current year value (integer KRW amount)",
                    },
                    "previous": {
                        "type": "number",
                        "description": "Previous year value (integer KRW amount)",
                    },
                },
                "required": ["current", "previous"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """You are a financial document analysis assistant for Korean listed companies.

Rules (non-negotiable):
1. ALL financial figures must come exclusively from tool results. Never produce numbers from memory.
2. ALL growth rate calculations must use the calculate_growth_rate tool. Never compute manually.
3. If any tool returns {"status": "needs_review"}, do NOT guess or estimate. State clearly in Korean that human review is required and set needs_review=true.
4. Format KRW amounts in readable Korean units: show both the Korean form (e.g. 300조 8,703억 원) and the raw value.
5. The sources list must contain ONLY source_url values returned by tools. Never fabricate URLs.
6. Answer entirely in Korean.
"""


def _dispatch(question: str, name: str, arguments_str: str) -> str:
    try:
        args = json.loads(arguments_str)
    except json.JSONDecodeError as exc:
        return json.dumps({"status": "error", "detail": f"invalid JSON: {exc}"})

    try:
        if name == "get_financial_statement":
            facts = get_financial_statement(
                company=args["company"],
                year=int(args["year"]),
                accounts=args.get("accounts"),
            )
            return json.dumps([f.model_dump() for f in facts], ensure_ascii=False)

        if name == "calculate_growth_rate":
            rate = calculate_growth_rate(
                current=int(args["current"]),
                previous=int(args["previous"]),
            )
            return json.dumps({"growth_rate_pct": rate})

        return json.dumps({"status": "error", "detail": f"unknown tool: {name}"})

    except ReviewRequired as exc:
        review_path = create_review_request(
            question=question,
            extracted=args,
            evidence={},
            failed_rule=exc.rule,
        )
        return json.dumps(
            {
                "status": "needs_review",
                "rule": exc.rule,
                "detail": exc.detail,
                "review_file": review_path,
            },
            ensure_ascii=False,
        )


def run_agent(question: str, max_steps: int = 6) -> AgentAnswer:
    if not config.OPENAI_API_KEY:
        return AgentAnswer(
            answer="OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.",
            sources=[],
            needs_review=True,
        )

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        response = client.beta.chat.completions.parse(
            model=config.OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            response_format=AgentAnswer,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            if message.parsed:
                return message.parsed
            return AgentAnswer(
                answer=message.content or "응답을 파싱할 수 없습니다.",
                sources=[],
                needs_review=True,
            )

        # Append assistant message with tool calls
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        # Execute each tool call and append results
        for tc in message.tool_calls:
            result = _dispatch(question, tc.function.name, tc.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return AgentAnswer(
        answer="최대 처리 단계를 초과했습니다. 사람의 검토가 필요합니다.",
        sources=[],
        needs_review=True,
    )
