"""Regression tests for PR #37 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# Bug 1: LLMJudgeReward test is environment-dependent
# File: test_reward_fn.py, line 59-63
# ---------------------------------------------------------------------------


class TestLLMJudgeRewardEnvironmentDependentRegression:
    """``test_fallback_on_missing_adapter`` constructs ``LLMJudgeReward()``
    and calls it without mocking. When ``anthropic`` is installed (e.g.
    via ``fit[adapters]``), the call attempts to build a real
    ``AnthropicAdapter`` and may make network calls or fail on a missing
    API key. The test must use ``monkeypatch`` to be deterministic.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PR #37 review: LLMJudgeReward test is environment-dependent,"
            " no monkeypatch"
        ),
    )
    def test_fallback_test_uses_monkeypatch(self) -> None:
        """The method signature of ``test_fallback_on_missing_adapter``
        must include a ``monkeypatch`` parameter, proving the adapter
        import is mocked. It currently does not."""
        from tests.test_reward_fn import TestLLMJudgeReward

        sig = inspect.signature(
            TestLLMJudgeReward.test_fallback_on_missing_adapter
        )
        params = list(sig.parameters.keys())
        assert "monkeypatch" in params, (
            "Bug confirmed: test_fallback_on_missing_adapter has no "
            "monkeypatch parameter — the test is environment-dependent "
            "and may attempt real network calls when anthropic is installed."
        )
