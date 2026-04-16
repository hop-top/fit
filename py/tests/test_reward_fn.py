"""Tests for fit.training.reward_fn — composable reward functions."""
from __future__ import annotations

import pytest

from fit.training.reward_fn import (
    CompositeReward,
    ExactMatchReward,
    LLMJudgeReward,
    RubricJudgeReward,
    UserSignalReward,
    _parse_score,
)


class TestExactMatchReward:
    def test_match_found(self) -> None:
        fn = ExactMatchReward("hello")
        assert fn("", "", "say hello world") == 1.0

    def test_no_match(self) -> None:
        fn = ExactMatchReward("xyz")
        assert fn("", "", "hello world") == 0.0

    def test_case_insensitive(self) -> None:
        fn = ExactMatchReward("Hello", case_sensitive=False)
        assert fn("", "", "hello") == 1.0

    def test_case_sensitive(self) -> None:
        fn = ExactMatchReward("Hello", case_sensitive=True)
        assert fn("", "", "hello") == 0.0


class TestRubricJudgeReward:
    def test_all_match(self) -> None:
        fn = RubricJudgeReward([("concise", 0.5), ("accurate", 0.5)])
        assert fn("", "", "This is concise and accurate") == 1.0

    def test_partial_match(self) -> None:
        fn = RubricJudgeReward([("concise", 0.5), ("detailed", 0.5)])
        score = fn("", "", "This is concise but brief")
        assert 0.0 < score < 1.0

    def test_no_match(self) -> None:
        fn = RubricJudgeReward([("xyzabc", 1.0)])
        assert fn("", "", "normal text") == 0.0

    def test_empty_rubrics(self) -> None:
        fn = RubricJudgeReward([])
        assert fn("", "", "anything") == 0.0

    def test_weighted(self) -> None:
        fn = RubricJudgeReward([(r"\balpha\b", 0.3), (r"\bbeta\b", 0.7)])
        score = fn("", "", "contains alpha but not the other")
        assert score == pytest.approx(0.3)


class TestLLMJudgeReward:
    def test_fallback_on_missing_adapter(self) -> None:
        fn = LLMJudgeReward()
        # Should not raise, returns fallback score
        score = fn("ctx", "adv", "out")
        assert 0.0 <= score <= 1.0


class TestUserSignalReward:
    def test_known_key(self) -> None:
        import hashlib

        output = "test output"
        key = hashlib.sha256(output.encode()).hexdigest()[:16]
        fn = UserSignalReward(scores={key: 0.9})
        assert fn("", "", output) == 0.9

    def test_unknown_key_default(self) -> None:
        fn = UserSignalReward(default=0.3)
        assert fn("", "", "unknown output") == 0.3

    def test_add_score(self) -> None:
        fn = UserSignalReward()
        fn.add_score("mykey", 0.8)
        assert fn._scores["mykey"] == 0.8


class TestCompositeReward:
    def test_weighted_average(self) -> None:
        fn = CompositeReward([
            (ExactMatchReward("a"), 0.5),
            (ExactMatchReward("b"), 0.5),
        ])
        # "has a" matches first fn only
        score = fn("", "", "has a")
        assert score == pytest.approx(0.5)

    def test_all_match(self) -> None:
        fn = CompositeReward([
            (ExactMatchReward("a"), 1.0),
            (ExactMatchReward("b"), 1.0),
        ])
        assert fn("", "", "a and b") == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeReward([])

    def test_zero_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CompositeReward([(ExactMatchReward("x"), 0.0)])


class TestParseScore:
    def test_score_prefix(self) -> None:
        assert _parse_score("Score: 0.8") == 0.8

    def test_rating_prefix(self) -> None:
        assert _parse_score("Rating: 0.9") == 0.9

    def test_fraction(self) -> None:
        assert _parse_score("4/5") == pytest.approx(0.8)

    def test_bare_float(self) -> None:
        assert _parse_score("The result is 0.7") == 0.7

    def test_no_number(self) -> None:
        assert _parse_score("no numbers here") == 0.5

    def test_score_above_one_scaled(self) -> None:
        assert _parse_score("Score: 8") == 0.8
