"""Regression tests for CompositeScorer bugs.

Bug 1 (weights length mismatch): __init__ did not validate that
weights length matched scorers length. zip() would silently drop
extras.

Bug 2 (breakdown merge): score() returned breakdown=rewards[0].breakdown,
dropping breakdown dimensions from subsequent scorers. Other ports
(TS/PHP/Rust) merge breakdowns from all scorers.
"""

from __future__ import annotations

from typing import Any

import pytest

from fit.reward import CompositeScorer, DimensionScorer, RewardScorer
from fit.types import Reward


class FixedScorer(RewardScorer):
    """Scorer that returns a fixed score with a specific breakdown."""

    def __init__(self, score: float, breakdown: dict[str, float]) -> None:
        self._score = score
        self._breakdown = breakdown

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        return Reward(score=self._score, breakdown=self._breakdown)


# --- Bug 1: weights/scorers length mismatch ---


def test_weights_longer_than_scorers_raises():
    """Regression: extra weights must raise ValueError, not be silently dropped.

    Before fix: zip(rewards, weights) in score() would silently truncate
    when weights had more entries than scorers, ignoring the extra weights.
    """
    scorers = [DimensionScorer("a"), DimensionScorer("b")]
    with pytest.raises(ValueError, match="mismatch"):
        CompositeScorer(scorers, weights=[0.5, 0.3, 0.2])


def test_weights_shorter_than_scorers_raises():
    """Regression: fewer weights must raise ValueError.

    Before fix: zip would pair only the matching count, leaving some
    scorers unweighted in the computation.
    """
    scorers = [DimensionScorer("a"), DimensionScorer("b"), DimensionScorer("c")]
    with pytest.raises(ValueError, match="mismatch"):
        CompositeScorer(scorers, weights=[0.5, 0.3])


def test_matching_weights_succeeds():
    """Equal-length weights and scorers must work without error."""
    scorers = [DimensionScorer("a"), DimensionScorer("b")]
    composite = CompositeScorer(scorers, weights=[0.6, 0.4])
    result = composite.score("test output", {})
    assert result.score > 0.0


def test_default_weights_succeed():
    """No weights provided: uniform distribution, no error."""
    scorers = [DimensionScorer("a"), DimensionScorer("b")]
    composite = CompositeScorer(scorers)
    result = composite.score("test output", {})
    assert result.score == pytest.approx(0.5)


# --- Bug 2: breakdown merge across all scorers ---


def test_breakdown_merges_all_scorer_dimensions():
    """Regression: breakdown must include dimensions from ALL scorers.

    Before fix: score() returned rewards[0].breakdown, dropping
    dimensions from subsequent scorers.
    """
    scorer_a = FixedScorer(0.8, {"accuracy": 0.8})
    scorer_b = FixedScorer(0.6, {"fluency": 0.6, "coherence": 0.7})
    composite = CompositeScorer([scorer_a, scorer_b], weights=[0.5, 0.5])

    result = composite.score("test output", {})

    # All dimensions from both scorers must be present
    assert "accuracy" in result.breakdown, "missing accuracy from scorer_a"
    assert "fluency" in result.breakdown, "missing fluency from scorer_b"
    assert "coherence" in result.breakdown, "missing coherence from scorer_b"


def test_breakdown_merge_values_preserved():
    """Merged breakdown values must match what each scorer returned."""
    scorer_a = FixedScorer(0.9, {"precision": 0.9})
    scorer_b = FixedScorer(0.7, {"recall": 0.7})
    composite = CompositeScorer([scorer_a, scorer_b], weights=[0.5, 0.5])

    result = composite.score("test output", {})

    assert result.breakdown["precision"] == pytest.approx(0.9)
    assert result.breakdown["recall"] == pytest.approx(0.7)


def test_single_scorer_breakdown_unchanged():
    """Single-scorer composite: breakdown must match the lone scorer."""
    scorer = FixedScorer(0.5, {"quality": 0.5, "safety": 1.0})
    composite = CompositeScorer([scorer])

    result = composite.score("test output", {})

    assert result.breakdown == {"quality": 0.5, "safety": 1.0}


def test_empty_scorers_no_breakdown():
    """Empty composite: breakdown must be empty dict."""
    composite = CompositeScorer([])

    result = composite.score("test output", {})

    assert result.breakdown == {}
