from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .advisor import Advisor
from .reward import RewardScorer
from .types import Advice, Reward, Trace


@dataclass
class SessionConfig:
    mode: str = "one-shot"
    max_steps: int = 10
    reward_threshold: float = 1.0


class Session:
    """Orchestrates advisor → frontier → reward → trace cycle."""

    def __init__(
        self,
        advisor: Advisor,
        adapter: Any,
        scorer: RewardScorer,
        config: SessionConfig | None = None,
    ) -> None:
        self._advisor = advisor
        self._adapter = adapter
        self._scorer = scorer
        self._config = config or SessionConfig()

    def run(self, prompt: str, context: dict[str, Any] | None = None) -> tuple[str, Reward, Trace]:
        """Run a one-shot session. Returns (output, reward, trace)."""
        import uuid
        from datetime import datetime, timezone

        ctx = context or {}
        session_id = str(uuid.uuid4())

        # Advise
        try:
            advice = self._advisor.generate_advice({"prompt": prompt, **ctx})
        except Exception:
            advice = Advice(domain="unknown", steering_text="", confidence=0.0)

        # Frontier
        output, frontier_meta = self._adapter.call(prompt, advice)

        # Score
        try:
            reward = self._scorer.score(output, ctx)
        except Exception:
            reward = Reward(score=float("nan"), breakdown={})

        trace = Trace(
            id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input={"prompt": prompt, "context": ctx},
            advice=advice,
            frontier=frontier_meta,
            reward=reward,
        )

        return output, reward, trace
