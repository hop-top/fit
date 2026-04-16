"""Regression tests for YAML serialization in trace output.

Includes:
- yaml.dump() must not emit Python-specific tags (must use safe_dump)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fit.trace import TraceWriter, _trace_to_dict
from fit.types import Advice, Reward, Trace


def test_trace_writer_yaml_no_python_tags():
    """Regression: TraceWriter.write() must not emit !!python/ tags.

    Before fix: yaml.dump() could emit Python-specific tags like
    !!python/object: or !!python/tuple: for non-standard types.
    Using yaml.safe_dump() ensures only JSON-compatible types are
    written, which produces clean YAML that other languages can read.
    """
    advice = Advice(
        domain="test",
        steering_text="steer",
        confidence=0.9,
    )
    reward = Reward(score=0.95, breakdown={"accuracy": 1.0})
    trace = Trace(
        id="test-id",
        session_id="sess-1",
        timestamp="2026-04-15T10:00:00Z",
        input={"prompt": "hello", "context": {}},
        advice=advice,
        frontier={"model": "stub"},
        reward=reward,
        # Include a tuple to trigger !!python/tuple with yaml.dump()
        # but not with yaml.safe_dump()
        metadata={"coords": (1, 2, 3)},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TraceWriter(tmpdir)
        path = writer.write(trace, step=1)

        content = Path(path).read_text()

        # Must not contain any Python-specific YAML tags
        assert "!!python/" not in content, (
            f"YAML output contains Python-specific tags:\n{content}"
        )
