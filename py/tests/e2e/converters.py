"""HuggingFace dataset -> fit trace schema converters.

Each converter yields trace dicts matching trace-format-v1.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterator


def trace_dict(
    *,
    prompt: str = "",
    domain: str = "general",
    advice_text: str = "Provide a helpful response",
    frontier_output: str = "",
    reward_score: float = 0.5,
    reward_breakdown: dict[str, float] | None = None,
    session_id: str | None = None,
    id: str | None = None,
    timestamp: str | None = None,
    frontier_model: str = "hf-source",
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a valid trace dict with sensible defaults."""
    return {
        "id": id or f"tr-{uuid.uuid4().hex[:8]}",
        "session_id": session_id or f"sess-{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "input": {
            "prompt": prompt,
            "context": context or {},
        },
        "advice": {
            "domain": domain,
            "steering_text": advice_text,
            "confidence": 0.8,
            "version": "1.0",
            "constraints": [],
            "metadata": {},
        },
        "frontier": {
            "model": frontier_model,
            "output": frontier_output,
        },
        "reward": {
            "score": reward_score,
            "breakdown": reward_breakdown or {"quality": reward_score},
        },
        "metadata": metadata or {},
    }


def no_robots_to_traces(
    dataset: Any,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Convert HuggingFaceH4/no_robots rows to trace dicts.

    Each row has: category (str), messages (list of {role, content}).
    """
    count = 0
    for row in dataset:
        if limit is not None and count >= limit:
            break

        category = row["category"]
        messages = row["messages"]
        prompt = messages[0]["content"]
        output = messages[-1]["content"]

        yield trace_dict(
            prompt=prompt,
            domain=category,
            advice_text=f"Respond as {category} expert",
            frontier_output=output,
            reward_score=0.8,
        )
        count += 1
