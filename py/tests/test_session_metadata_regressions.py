"""Regression tests for session failure reward metadata.

PR#16: session failure rewards must include metadata.error indicating
the failure type: "frontier_failure" when the adapter raises, and
"scorer_failure" when the scorer raises.
"""

from __future__ import annotations

from typing import Any

from fit.advisor import Advisor
from fit.reward import RewardScorer
from fit.session import Session
from fit.types import Advice, Reward


class CaptureAdvisor(Advisor):
    """Advisor that returns neutral advice."""

    def generate_advice(self, context: dict[str, Any]) -> Advice:
        return Advice(domain="test", steering_text="test", confidence=0.9)

    def model_id(self) -> str:
        return "capture-advisor"


class ErrorAdapter:
    """Adapter whose call() always raises."""

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        raise RuntimeError("frontier crashed")


class FakeAdapter:
    """Adapter that echoes the prompt."""

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        return prompt, {"provider": "fake"}


class ErrorScorer(RewardScorer):
    """Scorer whose score() always raises."""

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        raise RuntimeError("scorer crashed")


class FixedScorer(RewardScorer):
    """Scorer returning a fixed reward."""

    def score(self, output: str, context: dict[str, Any]) -> Reward:
        return Reward(score=0.8, breakdown={"accuracy": 0.8})


def test_frontier_failure_reward_has_error_metadata():
    """Frontier failure must set metadata.error='frontier_failure'."""
    advisor = CaptureAdvisor()
    adapter = ErrorAdapter()
    scorer = FixedScorer()
    session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

    output, reward, trace = session.run("test")

    assert reward.score is None
    assert reward.metadata.get("error") == "frontier_failure"


def test_scorer_failure_reward_has_error_metadata():
    """Scorer failure must set metadata.error='scorer_failure'."""
    advisor = CaptureAdvisor()
    adapter = FakeAdapter()
    scorer = ErrorScorer()
    session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

    output, reward, trace = session.run("test")

    assert reward.score is None
    assert reward.metadata.get("error") == "scorer_failure"


def test_success_reward_has_no_error_metadata():
    """Successful run must not have metadata.error."""
    advisor = CaptureAdvisor()
    adapter = FakeAdapter()
    scorer = FixedScorer()
    session = Session(advisor=advisor, adapter=adapter, scorer=scorer)

    output, reward, trace = session.run("test")

    assert reward.score is not None
    assert "error" not in reward.metadata
