"""Writes xrr-compatible traces from proxy requests."""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _system_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _extract_user_prompt(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _extract_system_content(messages: list[dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


def _extract_assistant_output(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


class BenchTraceWriter:
    """Writes xrr-compatible traces from proxy requests."""

    def __init__(
        self,
        output_dir: str = "./bench-traces",
        provider: str = "unknown",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._provider = provider

    def write_trace(
        self,
        request: dict,
        advice: str,
        response: dict,
        mode: str,
        timing_ms: int,
    ) -> str:
        """Write trace, return trace ID."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        trace_id = str(uuid.uuid4())
        messages = request.get("messages", [])
        sys_content = _extract_system_content(messages)
        sys_hash = _system_hash(sys_content)
        session_id = f"bench_{sys_hash[:16]}"

        usage = response.get("usage", {})
        model = response.get("model", "")

        # count user turns for context
        turn = sum(1 for m in messages if m.get("role") == "user")

        trace: dict[str, Any] = {
            "id": trace_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {
                "prompt": _extract_user_prompt(messages),
                "context": {
                    "mode": mode,
                    "system_hash": sys_hash,
                    "turn": turn,
                },
            },
            "advice": {
                "steering_text": advice,
                "domain": "benchmark",
                "confidence": 1.0,
            },
            "frontier": {
                "output": _extract_assistant_output(response),
                "model": model,
                "provider": self._provider,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            },
            "reward": {
                "score": None,
                "breakdown": {},
            },
            "metadata": {
                "benchmark_mode": mode,
                "timing_ms": timing_ms,
            },
        }

        out_path = self._output_dir / f"{trace_id}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(trace, f, default_flow_style=False, sort_keys=False)

        return trace_id
