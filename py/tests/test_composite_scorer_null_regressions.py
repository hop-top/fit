"""Regression tests for CompositeScorer null-score propagation.

PR#16: CompositeScorer.score() must return score=None with
metadata.error="child_score_is_null" when any child scorer
returns score=None, instead of raising TypeError on the
None * float multiplication.
"""

from __future__ import annotations

from typing import Any

from fit.reward import CompositeScorer, RewardScorer
from fit.types import Reward


class NullScorer(RewardScorer):
    """Scorer that returns score=None (simulates failure)."""

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        return Reward(score=None, breakdown={"failed": 0.0})


class FixedScorer(RewardScorer):
    """Scorer that returns a fixed score."""

    def __init__(self, score: float, breakdown: dict[str, float] | None = None) -> None:
        self._score = score
        self._breakdown = breakdown or {}

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        return Reward(score=self._score, breakdown=self._breakdown)


def test_composite_returns_null_when_one_child_is_null():
    """If any child scorer returns score=None, composite must propagate None."""
    good = FixedScorer(0.9, {"accuracy": 0.9})
    bad = NullScorer()
    composite = CompositeScorer([good, bad], weights=[0.5, 0.5])

    result = composite.score("test", {})

    assert result.score is None
    assert result.metadata.get("error") == "child_score_is_null"


def test_composite_null_propagates_breakdown():
    """Composite must merge breakdowns even when propagating null score."""
    good = FixedScorer(0.8, {"accuracy": 0.8})
    bad = NullScorer()
    composite = CompositeScorer([good, bad], weights=[0.7, 0.3])

    result = composite.score("test", {})

    assert result.score is None
    assert "accuracy" in result.breakdown
    assert "failed" in result.breakdown


def test_composite_null_all_children():
    """When all children return None, composite must still return None."""
    bad_a = NullScorer()
    bad_b = NullScorer()
    composite = CompositeScorer([bad_a, bad_b])

    result = composite.score("test", {})

    assert result.score is None
    assert result.metadata.get("error") == "child_score_is_null"


def test_composite_no_null_still_works():
    """Without null scores, composite must compute weighted average as before."""
    a = FixedScorer(1.0, {"a": 1.0})
    b = FixedScorer(0.0, {"b": 0.0})
    composite = CompositeScorer([a, b], weights=[0.5, 0.5])

    result = composite.score("test", {})

    assert result.score == 0.5
    assert "error" not in result.metadata
