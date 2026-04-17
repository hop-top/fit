"""Shared e2e test fixtures and trace generators."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml


def generate_trace(
    *,
    id: str | None = None,
    session_id: str | None = None,
    domain: str = "general",
    prompt: str = "What is X?",
    advice_text: str = "Be concise and accurate",
    frontier_output: str = "X is a placeholder variable.",
    frontier_model: str = "test-model",
    reward_score: float = 0.85,
    reward_breakdown: dict[str, float] | None = None,
    timestamp: str | None = None,
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single trace dict matching trace-format-v1 schema."""
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
            "confidence": 0.85,
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


def generate_traces(
    n: int = 50,
    domain: str = "general",
    sessions: int = 5,
) -> list[dict[str, Any]]:
    """Generate n synthetic traces spread across `sessions` sessions.

    Each session gets roughly n // sessions traces.
    """
    traces: list[dict[str, Any]] = []
    session_ids = [f"sess-{uuid.uuid4().hex[:8]}" for _ in range(sessions)]

    for i in range(n):
        sid = session_ids[i % sessions]
        traces.append(
            generate_trace(
                id=f"tr-{i:04d}",
                session_id=sid,
                domain=domain,
                prompt=f"Question {i}: What is topic {i}?",
                advice_text=f"Advice for topic {i}",
                frontier_output=f"Answer {i}: topic {i} is well-known.",
                reward_score=round(0.5 + (i % 10) * 0.05, 2),
            )
        )
    return traces


def write_jsonl(path: Path, traces: list[dict[str, Any]]) -> Path:
    """Write traces to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    return path


def write_yaml_dir(path: Path, traces: list[dict[str, Any]]) -> Path:
    """Write traces as YAML cassette tree (session_id/step-NNN.yaml)."""
    path.mkdir(parents=True, exist_ok=True)

    # Group by session_id
    sessions: dict[str, list[dict[str, Any]]] = {}
    for t in traces:
        sid = t["session_id"]
        sessions.setdefault(sid, []).append(t)

    for sid, session_traces in sessions.items():
        session_dir = path / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        for step, t in enumerate(session_traces, 1):
            step_path = session_dir / f"step-{step:03d}.yaml"
            step_path.write_text(
                yaml.safe_dump(t, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
    return path


@pytest.fixture
def tmp_traces(tmp_path: Path) -> tuple[list[dict[str, Any]], Path]:
    """Generate 50 synthetic traces and return (traces, tmp_path)."""
    traces = generate_traces(50)
    return traces, tmp_path
