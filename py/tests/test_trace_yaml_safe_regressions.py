"""Regression tests for YAML serialization in trace output.

Includes:
- yaml.dump() must not emit Python-specific tags (must use safe_dump)
- write_text/read_text must use explicit UTF-8 encoding
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fit.trace import TraceWriter, TraceReader
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


def test_trace_utf8_roundtrip():
    """Regression: write_text/read_text must use explicit UTF-8.

    PR#11: Path.write_text() and Path.read_text() without encoding
    use platform default (may be ASCII/Latin-1 on some systems).
    Trace with non-ASCII characters must round-trip correctly.
    """
    advice = Advice(
        domain="internationalization",
        steering_text="steer",
        confidence=0.8,
    )
    reward = Reward(score=0.9, breakdown={"accuracy": 0.9})
    trace = Trace(
        id="utf8-test",
        session_id="sess-utf8",
        timestamp="2026-04-16T10:00:00Z",
        input={"prompt": "Was ist der Mehrwertsteuersatz in Deutschland?"},
        advice=advice,
        frontier={"model": "stub"},
        reward=reward,
        metadata={"note": "Umlaute: \u00e4\u00f6\u00fc \u00df"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TraceWriter(tmpdir)
        writer.write(trace, step=1)

        reader = TraceReader(tmpdir)
        loaded = reader.read("sess-utf8", step=1)

        assert loaded["input"]["prompt"] == "Was ist der Mehrwertsteuersatz in Deutschland?"
        assert loaded["metadata"]["note"] == "Umlaute: \u00e4\u00f6\u00fc \u00df"
