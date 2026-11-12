import sys

from openai import RateLimitError, AuthenticationError

from app.agent.loop import run_agent
from app.validation.review import list_reviews, resolve_review


def _cmd_ask(question: str) -> None:
    try:
        answer = run_agent(question)
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


def _cmd_review_list() -> None:
    items = list_reviews()
    if not items:
        print("검토 대기 항목이 없습니다.")
        return
    print(f"{'ID':<26}  {'상태':<10}  {'규칙':<25}  질문")
    print("-" * 90)
    for item in items:
        run_id = item.get("run_id", "")[:24]
        status = item.get("status", "pending")
        rule = item.get("failed_rule", "")[:24]
        question = item.get("question", "")[:50]
        print(f"{run_id:<26}  {status:<10}  {rule:<25}  {question}")


def _cmd_review_resolve(review_id: str) -> None:
    try:
        data = resolve_review(review_id)
        print(f"완료: {data['run_id']} → resolved")
    except FileNotFoundError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print("사용법:", file=sys.stderr)
        print("  python -m app.main \"질문\"", file=sys.stderr)
        print("  python -m app.main review list", file=sys.stderr)
        print("  python -m app.main review resolve <review_id>", file=sys.stderr)
        sys.exit(1)

    if args[0] == "review":
        sub = args[1] if len(args) > 1 else ""
        if sub == "list":
            _cmd_review_list()
        elif sub == "resolve":
            if len(args) < 3:
                print("[오류] review_id를 지정하세요.", file=sys.stderr)
                sys.exit(1)
            _cmd_review_resolve(args[2])
        else:
            print(f"[오류] 알 수 없는 review 서브커맨드: '{sub}'", file=sys.stderr)
            sys.exit(1)
    else:
        question = " ".join(args)
        _cmd_ask(question)


if __name__ == "__main__":
    main()
