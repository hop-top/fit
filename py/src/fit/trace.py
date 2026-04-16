from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import Trace


class TraceWriter:
    """Writes xrr-compatible YAML cassettes."""

    def __init__(self, output_dir: str) -> None:
        self._dir = Path(output_dir)

    def write(self, trace: Trace, step: int = 1) -> Path:
        session_dir = self._dir / trace.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"step-{step:03d}.yaml"
        path.write_text(yaml.dump(_trace_to_dict(trace), default_flow_style=False, sort_keys=False))
        return path


class TraceReader:
    """Reads xrr-compatible YAML cassettes."""

    def __init__(self, output_dir: str) -> None:
        self._dir = Path(output_dir)

    def list_sessions(self) -> list[str]:
        if not self._dir.exists():
            return []
        return [d.name for d in sorted(self._dir.iterdir()) if d.is_dir()]

    def read(self, session_id: str, step: int = 1) -> dict[str, Any]:
        path = self._dir / session_id / f"step-{step:03d}.yaml"
        return yaml.safe_load(path.read_text())


def _trace_to_dict(trace: Trace) -> dict[str, Any]:
    return {
        "id": trace.id,
        "session_id": trace.session_id,
        "timestamp": trace.timestamp,
        "input": trace.input,
        "advice": {
            "domain": trace.advice.domain,
            "steering_text": trace.advice.steering_text,
            "confidence": trace.advice.confidence,
            "constraints": trace.advice.constraints,
            "metadata": trace.advice.metadata,
        },
        "frontier": trace.frontier,
        "reward": {
            "score": trace.reward.score,
            "breakdown": trace.reward.breakdown,
        },
        "metadata": trace.metadata,
    }
