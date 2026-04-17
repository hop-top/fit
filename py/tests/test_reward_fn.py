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
    def test_fallback_on_missing_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fit.adapters.anthropic as anthropic_mod

        class MissingAnthropicAdapter:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise ImportError("anthropic adapter unavailable")

        monkeypatch.setattr(
            anthropic_mod, "AnthropicAdapter", MissingAnthropicAdapter
        )

        fn = LLMJudgeReward()
        score = fn("ctx", "adv", "out")
        assert score == pytest.approx(0.5)


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


class TestParseScoreClamping:
    """Regression: _parse_score must clamp all returns to [0.0, 1.0].

    Fraction branches (both bare and prefixed) divide numerator by
    denominator without clamping, so num > denom yields values > 1.0.
    """

    def test_bare_fraction_over_one_clamped(self) -> None:
        """'4/3' currently returns ~1.333; must be clamped to 1.0."""
        assert _parse_score("4/3") <= 1.0

    def test_prefixed_fraction_over_one_clamped(self) -> None:
        """'Score: 5/3' currently returns ~1.667; must be clamped to 1.0."""
        assert _parse_score("Score: 5/3") <= 1.0

    def test_zero_fraction_stays_zero(self) -> None:
        """'0/5' returns 0.0 — should remain 0.0 after clamping."""
        assert _parse_score("0/5") == pytest.approx(0.0)

    def test_all_returns_in_unit_range(self) -> None:
        """Every branch of _parse_score must return a value in [0.0, 1.0]."""
        cases = [
            "4/3",
            "Score: 5/3",
            "0/5",
            "Score: 0.8",
            "Rating: 0.9",
            "4/5",
            "The result is 0.7",
            "no numbers here",
            "Score: 8",
            "8",
            "1",
            "10",
            "0",
            "I'd give it a 7 out of 10",
        ]
        for text in cases:
            score = _parse_score(text)
            assert 0.0 <= score <= 1.0, (
                f"_parse_score({text!r}) = {score}, outside [0.0, 1.0]"
            )


class TestParseScoreBareInteger:
    """Regression: _parse_score must handle bare integers like '8'
    (not just decimals like '8.0'). The bare-number regex required
    a decimal part, so '8' fell through to default 0.5."""

    def test_bare_integer_eight(self) -> None:
        assert _parse_score("8") == pytest.approx(0.8)

    def test_bare_integer_one(self) -> None:
        assert _parse_score("1") == pytest.approx(1.0)

    def test_bare_integer_ten(self) -> None:
        assert _parse_score("10") == pytest.approx(1.0)

    def test_bare_integer_zero(self) -> None:
        assert _parse_score("0") == pytest.approx(0.0)

    def test_mixed_text_with_bare_integer(self) -> None:
        assert _parse_score("I'd give it a 7 out of 10") == pytest.approx(0.7)


class TestParseScoreDivisionAndClampingRegression:
    """_parse_score division-by-zero and unclamped val/10.0 branches.

    The prefixed/bare fraction branches divide by the denominator
    without guarding against zero, so "5/0" raises ZeroDivisionError.
    The val/10.0 branches (prefixed integer >1.0 and bare number >1.0)
    never clamp, so values >10 return scores >1.0.
    """

    def test_division_by_zero_prefixed_fraction(self) -> None:
        """'Score: 5/0' must not raise and must return a value in [0,1]."""
        score = _parse_score("Score: 5/0")
        assert 0.0 <= score <= 1.0

    def test_division_by_zero_bare_fraction(self) -> None:
        """'5/0' must not raise and must return a value in [0,1]."""
        score = _parse_score("5/0")
        assert 0.0 <= score <= 1.0

    def test_large_integer_above_one_scaled(self) -> None:
        """'Score: 15' -> 15/10 = 1.5; must be clamped to [0,1]."""
        score = _parse_score("Score: 15")
        assert 0.0 <= score <= 1.0, f"got {score}, outside [0.0, 1.0]"

    def test_bare_large_integer(self) -> None:
        """'15' -> 15/10 = 1.5; must be clamped to [0,1]."""
        score = _parse_score("15")
        assert 0.0 <= score <= 1.0, f"got {score}, outside [0.0, 1.0]"

    def test_score_one_hundred_clamped(self) -> None:
        """'Score: 100' -> 100/10 = 10.0; must be clamped to [0,1]."""
        score = _parse_score("Score: 100")
        assert 0.0 <= score <= 1.0, f"got {score}, outside [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Regression: UserSignalReward docstring mismatch
# ---------------------------------------------------------------------------


class TestUserSignalRewardDocstringRegression:
    """UserSignalReward's docstring mentions ``session_id`` and
    ``(session_id, step)`` as the keying mechanism, but the
    implementation actually hashes the output text with SHA-256.

    The docstring must not reference a keying scheme that does not
    match the code.
    """

    def test_docstring_does_not_mention_session_id(self) -> None:
        """UserSignalReward.__doc__ must not reference 'session_id'."""
        doc = UserSignalReward.__doc__ or ""
        assert "session_id" not in doc, (
            "Docstring still mentions 'session_id'; impl uses output text hashing"
        )

    def test_docstring_does_not_mention_session_step_tuple(self) -> None:
        """UserSignalReward.__doc__ must not reference '(session_id, step)'."""
        doc = UserSignalReward.__doc__ or ""
        assert "(session_id, step)" not in doc, (
            "Docstring still mentions '(session_id, step)'; "
            "impl uses SHA-256 of output text"
        )


# ---------------------------------------------------------------------------
# Regression: UserSignalReward inline comment mismatch
# ---------------------------------------------------------------------------


class TestUserSignalRewardInlineCommentRegression:
    """UserSignalReward.__call__ inline comment mentions 'context key'
    but the implementation only hashes ``output`` — no context-based
    lookup exists.

    The inline comment must not reference a lookup mechanism that is
    not implemented.
    """

    def test_call_source_does_not_mention_context_key(self) -> None:
        """UserSignalReward.__call__ source must not mention 'context key'."""
        import inspect

        source = inspect.getsource(UserSignalReward.__call__)
        assert "context key" not in source, (
            "Inline comment still mentions 'context key'; "
            "impl only hashes output text"
        )


# ---------------------------------------------------------------------------
# Regression: LLMJudge missing advice handling
# ---------------------------------------------------------------------------


class TestLLMJudgePassesAdvice:
    """adapter.call signature is call(prompt, advice) but __call__
    must pass both arguments. TypeError is swallowed if only prompt
    is passed, always returning 0.5."""

    def test_call_receives_two_args(self) -> None:
        """Adapter.call must receive (prompt, Advice), not just
        prompt."""
        from unittest.mock import MagicMock, patch

        from fit.types import Advice

        judge = LLMJudgeReward()
        mock_adapter = MagicMock()
        mock_adapter.call.return_value = (
            "Score: 0.9",
            {"model": "test"},
        )

        with patch(
            "fit.adapters.anthropic.AnthropicAdapter",
            return_value=mock_adapter,
        ):
            judge("ctx", "some advice", "output text")

        mock_adapter.call.assert_called_once()
        call_args = mock_adapter.call.call_args

        if len(call_args.args) == 1 and not call_args.kwargs:
            pytest.fail(
                "adapter.call received 1 positional arg "
                f"({call_args.args[0][:60]!r}...) but needs "
                "(prompt, Advice). "
                "LLMJudgeReward swallows the TypeError and "
                "returns 0.5."
            )

        assert len(call_args.args) >= 2
        assert isinstance(call_args.args[1], Advice)

    def test_returns_real_score_not_fallback(self) -> None:
        """When adapter works, judge must return the parsed score,
        not 0.5."""
        from unittest.mock import MagicMock, patch

        judge = LLMJudgeReward()

        real_module = __import__(
            "fit.adapters.anthropic",
            fromlist=["AnthropicAdapter"],
        )
        RealClass = real_module.AnthropicAdapter

        mock_adapter = MagicMock(spec=RealClass)
        mock_adapter.call.return_value = (
            "Score: 0.9",
            {"model": "test"},
        )

        with patch(
            "fit.adapters.anthropic.AnthropicAdapter",
            return_value=mock_adapter,
        ):
            judge("ctx", "advice", "output")

        mock_adapter.call.assert_called_once()
        call_args = mock_adapter.call.call_args
        assert len(call_args.args) >= 2, (
            "adapter.call called with "
            f"{len(call_args.args)} arg(s) but needs 2 "
            "(prompt, Advice). In production this raises "
            "TypeError, caught by except, returns fallback 0.5."
        )


# ---------------------------------------------------------------------------
# Regression: _parse_score handles "Rating: 4/5"
# ---------------------------------------------------------------------------


class TestParseScorePrefixedFraction:
    """_parse_score must handle "Rating: 4/5" as 0.8, not 0.4."""

    def test_rating_fraction_4_over_5(self) -> None:
        score = _parse_score("Rating: 4/5")
        assert score == pytest.approx(0.8, abs=0.01), (
            f"'Rating: 4/5' parsed as {score}, expected ~0.8. "
            "Prefix regex matches '4' before fraction regex."
        )

    def test_score_fraction_3_over_10(self) -> None:
        score = _parse_score("Score: 3/10")
        assert score == pytest.approx(0.3, abs=0.01), (
            f"'Score: 3/10' parsed as {score}, expected ~0.3."
        )

    def test_score_fraction_7_over_8(self) -> None:
        score = _parse_score("Score: 7/8")
        assert score == pytest.approx(0.875, abs=0.01), (
            f"'Score: 7/8' parsed as {score}, expected ~0.875."
        )


# ---------------------------------------------------------------------------
# Regression: Advice line within 100 chars
# ---------------------------------------------------------------------------


class TestLLMJudgeAdviceLineLength:
    """``LLMJudgeReward.__call__`` Advice construction line must stay
    within the 100-character project line-length limit.
    """

    def test_advice_line_within_limit(self) -> None:
        """Every source line in LLMJudgeReward.__call__ must be
        at most 100 characters wide."""
        import inspect
        import pathlib

        src_file = pathlib.Path(
            inspect.getfile(LLMJudgeReward)
        )
        source = src_file.read_text()
        long_lines = [
            (i + 1, line)
            for i, line in enumerate(source.splitlines())
            if "Advice(" in line and len(line.rstrip()) > 99
        ]
        assert not long_lines, (
            "Advice construction line exceeds "
            f"100-char limit: {long_lines}"
        )


# ---------------------------------------------------------------------------
# Regression: LLMJudge returns fallback on ValueError
# ---------------------------------------------------------------------------


class TestLLMJudgeFallbackOnValueError:
    """``LLMJudgeReward.__call__`` must catch ValueError (e.g. missing
    API key) in addition to ImportError and return the 0.5 neutral
    fallback instead of crashing.
    """

    def test_returns_fallback_on_missing_api_key(self) -> None:
        """Calling LLMJudgeReward when the adapter raises ValueError
        must return 0.5, not raise."""
        from unittest.mock import patch

        with patch(
            "fit.training.reward_fn.AnthropicAdapter",
            side_effect=ValueError("no api key"),
            create=True,
        ), patch(
            "fit.adapters.anthropic.AnthropicAdapter",
            side_effect=ValueError("no api key"),
        ):
            reward = LLMJudgeReward()
            score = reward("ctx", "adv", "out")
        assert score == 0.5, (
            "LLMJudgeReward raises ValueError instead "
            "of returning 0.5 neutral fallback"
        )
