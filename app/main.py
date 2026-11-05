import argparse

from app.agent.loop import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial Document Agent")
    parser.add_argument("question", help="Question about Korean listed companies")
    args = parser.parse_args()

    answer = run_agent(args.question)

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
