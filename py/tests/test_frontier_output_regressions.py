"""PR#19 regression: spec/trace-format-v1.md requires frontier.output.

The session must inject adapter output into frontier_meta even when
the adapter omits it from its metadata dict.
"""

from __future__ import annotations

from typing import Any

from fit.advisor import Advisor
from fit.reward import RewardScorer
from fit.session import Session
from fit.types import Advice, Reward


class StubAdvisor(Advisor):
    def generate_advice(self, context: dict[str, Any]) -> Advice:
        return Advice(domain="test", steering_text="steer", confidence=0.5)

    def model_id(self) -> str:
        return "stub"


class StubScorer(RewardScorer):
    def score(self, output: str, context: dict[str, Any]) -> Reward:
        return Reward(score=0.8, breakdown={"accuracy": 0.8})


class MetaNoOutputAdapter:
    """Adapter that returns output but omits it from meta."""

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        return "hello", {"model": "test"}


class ErrorAdapter:
    """Adapter that always raises."""

    def call(self, prompt: str, advice: Advice) -> tuple[str, dict[str, Any]]:
        raise RuntimeError("boom")


def test_frontier_output_injected_when_adapter_omits():
    """Adapter returns output but meta has no 'output' key."""
    session = Session(
        advisor=StubAdvisor(),
        adapter=MetaNoOutputAdapter(),
        scorer=StubScorer(),
    )
    _output, _reward, trace = session.run("test")
    assert trace.frontier["output"] == "hello"


def test_frontier_output_injected_on_adapter_failure():
    """Even on adapter failure, frontier must contain output."""
    session = Session(
        advisor=StubAdvisor(),
        adapter=ErrorAdapter(),
        scorer=StubScorer(),
    )
    _output, _reward, trace = session.run("test")
    assert trace.frontier["output"] == ""
