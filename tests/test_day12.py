"""Tests for Day 12: verify_node, human_review_node, graph routing, review CLI."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agent.graph import _route_after_verify
from app.agent.nodes import human_review_node, verify_node
from app.agent.state import AgentState, VerifyResult
from app.validation.review import create_review_request, list_reviews, resolve_review


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> AgentState:
    base: AgentState = {
        "question": "삼성전자 2024년 매출액은?",
        "entities": {"companies": ["삼성전자"], "years": [2024], "accounts": ["매출액"]},
        "route": "financial",
        "tool_results": [],
        "retrieved_chunks": [],
        "computed": {},
        "answer": {"answer": "삼성전자 매출액은 300조입니다.", "sources": [], "needs_review": False},
        "needs_review": False,
        "verify_passed": True,
        "regenerate_count": 0,
    }
    base.update(overrides)
    return base


def _fact_dict(**overrides) -> dict:
    base = {
        "company": "삼성전자",
        "corp_code": "00126380",
        "year": 2024,
        "account": "매출액",
        "fs_div": "CFS",
        "current_value": 300_000_000,
        "prior_value": 258_000_000,
        "currency": "KRW",
        "rcept_no": "20250310000123",
        "source_url": "https://dart.fss.or.kr/test",
    }
    base.update(overrides)
    return base


def _llm_resp(parsed) -> MagicMock:
    r = MagicMock()
    r.choices[0].message.parsed = parsed
    return r


def _verify_result(supported: bool, reason: str = "test") -> VerifyResult:
    return VerifyResult(supported=supported, reason=reason)


# ---------------------------------------------------------------------------
# verify_node
# ---------------------------------------------------------------------------

class TestVerifyNode:
    def test_no_facts_no_chunks_passes_without_llm(self):
        state = _state(tool_results=[], retrieved_chunks=[])
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            result = verify_node(state)
        MockLLM.assert_not_called()
        assert result["verify_passed"] is True

    def test_supported_answer_sets_verify_passed_true(self):
        state = _state(tool_results=[_fact_dict()])
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = (
                _llm_resp(_verify_result(supported=True))
            )
            result = verify_node(state)
        assert result["verify_passed"] is True

    def test_unsupported_answer_sets_verify_passed_false(self):
        state = _state(tool_results=[_fact_dict()])
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = (
                _llm_resp(_verify_result(supported=False))
            )
            result = verify_node(state)
        assert result["verify_passed"] is False

    def test_first_failure_increments_regenerate_count_to_1(self):
        state = _state(tool_results=[_fact_dict()], regenerate_count=0)
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = (
                _llm_resp(_verify_result(supported=False))
            )
            result = verify_node(state)
        assert result["regenerate_count"] == 1

    def test_second_failure_increments_regenerate_count_to_2(self):
        state = _state(tool_results=[_fact_dict()], regenerate_count=1)
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = (
                _llm_resp(_verify_result(supported=False))
            )
            result = verify_node(state)
        assert result["regenerate_count"] == 2

    def test_success_does_not_set_regenerate_count(self):
        state = _state(tool_results=[_fact_dict()], regenerate_count=0)
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = (
                _llm_resp(_verify_result(supported=True))
            )
            result = verify_node(state)
        assert "regenerate_count" not in result

    def test_llm_error_defaults_to_passed(self):
        state = _state(tool_results=[_fact_dict()])
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.side_effect = RuntimeError("API error")
            result = verify_node(state)
        assert result["verify_passed"] is True

    def test_null_parsed_response_defaults_to_passed(self):
        state = _state(tool_results=[_fact_dict()])
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = _llm_resp(None)
            result = verify_node(state)
        assert result["verify_passed"] is True

    def test_needs_review_entries_excluded_from_facts_context(self):
        # needs_review entry should not reach LLM as a verifiable fact
        state = _state(tool_results=[
            {"status": "needs_review", "company": "없는회사"},
        ])
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            result = verify_node(state)
        # Only needs_review entries → no facts → skip LLM
        MockLLM.assert_not_called()
        assert result["verify_passed"] is True

    def test_retrieved_chunks_trigger_verification(self):
        # No facts but chunks present → LLM should be called
        state = _state(
            tool_results=[],
            retrieved_chunks=[{"text": "반도체 사업 현황", "source_url": "https://dart.fss.or.kr/test"}],
        )
        with patch("app.agent.nodes.OpenAI") as MockLLM:
            MockLLM.return_value.beta.chat.completions.parse.return_value = (
                _llm_resp(_verify_result(supported=True))
            )
            result = verify_node(state)
        MockLLM.assert_called_once()
        assert result["verify_passed"] is True


# ---------------------------------------------------------------------------
# _route_after_verify
# ---------------------------------------------------------------------------

class TestRouteAfterVerify:
    def test_verify_passed_routes_to_end(self):
        state = _state(verify_passed=True)
        assert _route_after_verify(state) == "end"

    def test_first_failure_routes_to_regenerate(self):
        state = _state(verify_passed=False, regenerate_count=1)
        assert _route_after_verify(state) == "regenerate"

    def test_second_failure_routes_to_human_review(self):
        state = _state(verify_passed=False, regenerate_count=2)
        assert _route_after_verify(state) == "human_review"

    def test_default_verify_passed_true_routes_to_end(self):
        # state without verify_passed key → defaults to True
        state = _state()
        del state["verify_passed"]
        assert _route_after_verify(state) == "end"

    def test_regenerate_count_0_with_failure_routes_to_regenerate(self):
        # Edge: verify_node failed but somehow regenerate_count stayed 0
        state = _state(verify_passed=False, regenerate_count=0)
        assert _route_after_verify(state) == "regenerate"


# ---------------------------------------------------------------------------
# human_review_node
# ---------------------------------------------------------------------------

class TestHumanReviewNode:
    def test_sets_needs_review_true(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            result = human_review_node(_state())
        assert result["needs_review"] is True

    def test_creates_review_file_in_review_dir(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            human_review_node(_state())
        files = list(tmp_path.glob("review_*.json"))
        assert len(files) == 1

    def test_review_file_has_correct_failed_rule(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            human_review_node(_state())
        data = json.loads(next(tmp_path.glob("review_*.json")).read_text())
        assert data["failed_rule"] == "answer_not_supported"

    def test_review_file_contains_question(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            human_review_node(_state(question="삼성전자 테스트 질문"))
        data = json.loads(next(tmp_path.glob("review_*.json")).read_text())
        assert data["question"] == "삼성전자 테스트 질문"

    def test_review_file_contains_regenerate_count(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            human_review_node(_state(regenerate_count=2))
        data = json.loads(next(tmp_path.glob("review_*.json")).read_text())
        assert data["evidence"]["regenerate_count"] == 2


# ---------------------------------------------------------------------------
# list_reviews / resolve_review
# ---------------------------------------------------------------------------

class TestListReviews:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert list_reviews(review_dir=tmp_path) == []

    def test_returns_all_pending_reviews(self, tmp_path):
        create_review_request.__wrapped__ = None  # ensure no wrapping
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            create_review_request("질문 1", {}, {}, "company_not_found")
            create_review_request("질문 2", {}, {}, "value_mismatch")
        items = list_reviews(review_dir=tmp_path)
        assert len(items) == 2

    def test_items_sorted_by_created_at(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            create_review_request("질문 A", {}, {}, "rule_a")
            create_review_request("질문 B", {}, {}, "rule_b")
        items = list_reviews(review_dir=tmp_path)
        timestamps = [i["created_at"] for i in items]
        assert timestamps == sorted(timestamps)

    def test_new_review_has_pending_status(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            create_review_request("질문", {}, {}, "some_rule")
        items = list_reviews(review_dir=tmp_path)
        assert items[0]["status"] == "pending"

    def test_skips_malformed_json(self, tmp_path):
        (tmp_path / "review_bad.json").write_text("not json")
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            create_review_request("좋은 질문", {}, {}, "rule")
        items = list_reviews(review_dir=tmp_path)
        assert len(items) == 1  # only the valid one


class TestResolveReview:
    def test_resolve_updates_status_to_resolved(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            create_review_request("질문", {}, {}, "rule")
        item = list_reviews(review_dir=tmp_path)[0]
        resolve_review(item["run_id"], review_dir=tmp_path)
        updated = list_reviews(review_dir=tmp_path)[0]
        assert updated["status"] == "resolved"

    def test_resolve_adds_resolved_at_timestamp(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            create_review_request("질문", {}, {}, "rule")
        item = list_reviews(review_dir=tmp_path)[0]
        resolve_review(item["run_id"], review_dir=tmp_path)
        updated = list_reviews(review_dir=tmp_path)[0]
        assert "resolved_at" in updated

    def test_resolve_with_full_stem_works(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            create_review_request("질문", {}, {}, "rule")
        item = list_reviews(review_dir=tmp_path)[0]
        full_stem = f"review_{item['run_id']}"
        resolve_review(full_stem, review_dir=tmp_path)
        updated = list_reviews(review_dir=tmp_path)[0]
        assert updated["status"] == "resolved"

    def test_resolve_nonexistent_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_review("nonexistent_id", review_dir=tmp_path)

    def test_resolve_returns_updated_dict(self, tmp_path):
        with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
            create_review_request("질문", {}, {}, "rule")
        item = list_reviews(review_dir=tmp_path)[0]
        result = resolve_review(item["run_id"], review_dir=tmp_path)
        assert result["status"] == "resolved"
        assert result["question"] == "질문"


# ---------------------------------------------------------------------------
# run_graph integration: verify cycle
# ---------------------------------------------------------------------------

class TestRunGraphVerifyCycle:
    def test_verify_pass_ends_normally(self):
        """Happy path: verify passes on first attempt."""
        from app.agent.graph import run_graph
        from app.agent.state import QuestionAnalysis
        from app.validation.schema import AgentAnswer, FinancialFact

        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=["매출액"], route="financial"
        )
        fact = FinancialFact(
            company="삼성전자", corp_code="00126380", year=2024,
            account="매출액", fs_div="CFS",
            current_value=300_000_000, prior_value=258_000_000,
            rcept_no="20250310000123",
            source_url="https://dart.fss.or.kr/test",
        )
        final_answer = AgentAnswer(
            answer="삼성전자 2024년 매출액은 300조입니다.",
            sources=["https://dart.fss.or.kr/test"],
            needs_review=False,
        )
        verify_ok = VerifyResult(supported=True, reason="데이터와 일치합니다.")

        with patch("app.agent.nodes.OpenAI") as MockLLM:
            client = MockLLM.return_value
            client.beta.chat.completions.parse.side_effect = [
                _llm_resp(analysis),      # analyze_question
                _llm_resp(final_answer),  # generate_answer
                _llm_resp(verify_ok),     # verify_node
            ]
            with patch("app.agent.nodes.get_financial_statement", return_value=[fact]):
                with patch("app.agent.nodes.calculate_growth_rate", return_value=16.19):
                    result = run_graph("삼성전자 2024년 매출액은?")

        assert result.needs_review is False
        assert "삼성전자" in result.answer

    def test_verify_fail_then_pass_on_retry(self):
        """First verify fails, regenerate, second verify passes."""
        from app.agent.graph import run_graph
        from app.agent.state import QuestionAnalysis
        from app.validation.schema import AgentAnswer, FinancialFact

        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=["매출액"], route="financial"
        )
        fact = FinancialFact(
            company="삼성전자", corp_code="00126380", year=2024,
            account="매출액", fs_div="CFS",
            current_value=300_000_000, prior_value=258_000_000,
            rcept_no="20250310000123",
            source_url="https://dart.fss.or.kr/test",
        )
        answer_v1 = AgentAnswer(answer="첫 번째 답변", sources=[], needs_review=False)
        answer_v2 = AgentAnswer(answer="두 번째 (수정된) 답변", sources=[], needs_review=False)
        verify_fail = VerifyResult(supported=False, reason="수치 불일치")
        verify_ok = VerifyResult(supported=True, reason="일치합니다")

        with patch("app.agent.nodes.OpenAI") as MockLLM:
            client = MockLLM.return_value
            client.beta.chat.completions.parse.side_effect = [
                _llm_resp(analysis),     # analyze_question
                _llm_resp(answer_v1),    # generate_answer (1st)
                _llm_resp(verify_fail),  # verify_node (1st — fail)
                _llm_resp(answer_v2),    # generate_answer (2nd)
                _llm_resp(verify_ok),    # verify_node (2nd — pass)
            ]
            with patch("app.agent.nodes.get_financial_statement", return_value=[fact]):
                with patch("app.agent.nodes.calculate_growth_rate", return_value=16.19):
                    result = run_graph("삼성전자 테스트")

        assert result.needs_review is False

    def test_verify_fail_twice_routes_to_human_review(self, tmp_path):
        """Both verify attempts fail → human_review_node runs → needs_review=True."""
        from app.agent.graph import run_graph
        from app.agent.state import QuestionAnalysis
        from app.validation.schema import AgentAnswer, FinancialFact

        analysis = QuestionAnalysis(
            companies=["삼성전자"], years=[2024], accounts=["매출액"], route="financial"
        )
        fact = FinancialFact(
            company="삼성전자", corp_code="00126380", year=2024,
            account="매출액", fs_div="CFS",
            current_value=300_000_000, prior_value=258_000_000,
            rcept_no="20250310000123",
            source_url="https://dart.fss.or.kr/test",
        )
        answer = AgentAnswer(answer="잘못된 답변", sources=[], needs_review=False)
        verify_fail = VerifyResult(supported=False, reason="수치 불일치")

        with patch("app.agent.nodes.OpenAI") as MockLLM:
            client = MockLLM.return_value
            client.beta.chat.completions.parse.side_effect = [
                _llm_resp(analysis),     # analyze_question
                _llm_resp(answer),       # generate_answer (1st)
                _llm_resp(verify_fail),  # verify_node (1st — fail)
                _llm_resp(answer),       # generate_answer (2nd)
                _llm_resp(verify_fail),  # verify_node (2nd — fail)
            ]
            with patch("app.agent.nodes.get_financial_statement", return_value=[fact]):
                with patch("app.agent.nodes.calculate_growth_rate", return_value=16.19):
                    with patch("app.validation.review.config.REVIEW_DIR", tmp_path):
                        result = run_graph("삼성전자 테스트")

        assert result.needs_review is True
        assert len(list(tmp_path.glob("review_*.json"))) == 1
