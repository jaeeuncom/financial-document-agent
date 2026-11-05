import argparse
import sys

from openai import RateLimitError, AuthenticationError

from app.agent.loop import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Document Agent")
    parser.add_argument("question", help="Question about Korean listed companies")
    args = parser.parse_args()

    try:
        answer = run_agent(args.question)
    except AuthenticationError:
        print("[오류] OPENAI_API_KEY가 유효하지 않습니다. .env 파일을 확인하세요.", file=sys.stderr)
        sys.exit(1)
    except RateLimitError as exc:
        if "credit_balance_exhausted" in str(exc) or "insufficient_quota" in str(exc):
            print(
                "[오류] OpenAI 크레딧이 소진되었습니다.\n"
                "  충전: https://platform.openai.com/settings/organization/billing/",
                file=sys.stderr,
            )
        else:
            print(f"[오류] OpenAI API 요청 한도 초과: {exc}", file=sys.stderr)
        sys.exit(1)

    print(answer.answer)

    if answer.sources:
        print("\n[출처]")
        for source in answer.sources:
            print(f"  - {source}")

    if answer.needs_review:
        print(
            "\n[주의] 이 답변은 사람의 검토가 필요합니다. "
            "data/review_queue/ 디렉토리를 확인하세요."
        )


if __name__ == "__main__":
    main()
