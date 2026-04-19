from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from fit.bench.tracing import BenchTraceWriter
from fit.training.tracer import TraceIngester


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    return tmp_path / "bench-traces"


def _sample_request() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "model": "gpt-4o",
    }


def _sample_response() -> dict:
    return {
        "choices": [
            {"message": {"content": "4", "role": "assistant"}}
        ],
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
        },
    }


class TestBenchTraceWriter:
    def test_creates_output_dir(self, trace_dir: Path) -> None:
        writer = BenchTraceWriter(output_dir=str(trace_dir))
        writer.write_trace(
            request=_sample_request(),
            advice="Be concise.",
            response=_sample_response(),
            mode="oneshot",
            timing_ms=150,
        )
        assert trace_dir.is_dir()

    def test_returns_trace_id(self, trace_dir: Path) -> None:
        writer = BenchTraceWriter(output_dir=str(trace_dir))
        trace_id = writer.write_trace(
            request=_sample_request(),
            advice="Be concise.",
            response=_sample_response(),
            mode="oneshot",
            timing_ms=150,
        )
        assert isinstance(trace_id, str)
        assert len(trace_id) == 36  # UUID format

    def test_trace_file_created(self, trace_dir: Path) -> None:
        writer = BenchTraceWriter(output_dir=str(trace_dir))
        trace_id = writer.write_trace(
            request=_sample_request(),
            advice="Be concise.",
            response=_sample_response(),
            mode="oneshot",
            timing_ms=150,
        )
        yaml_files = list(trace_dir.rglob("*.yaml"))
        assert len(yaml_files) == 1
        assert trace_id in yaml_files[0].name

    def test_yaml_parseable(self, trace_dir: Path) -> None:
        writer = BenchTraceWriter(output_dir=str(trace_dir))
        writer.write_trace(
            request=_sample_request(),
            advice="Be concise.",
            response=_sample_response(),
            mode="oneshot",
            timing_ms=150,
        )
        yaml_file = next(trace_dir.rglob("*.yaml"))
        with yaml_file.open() as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_trace_format_v1_fields(self, trace_dir: Path) -> None:
        writer = BenchTraceWriter(output_dir=str(trace_dir))
        writer.write_trace(
            request=_sample_request(),
            advice="Be concise.",
            response=_sample_response(),
            mode="oneshot",
            timing_ms=150,
        )
        yaml_file = next(trace_dir.rglob("*.yaml"))
        with yaml_file.open() as f:
            data = yaml.safe_load(f)

        # top-level required keys
        assert "id" in data
        assert "session_id" in data
        assert "timestamp" in data

        # input
        assert data["input"]["prompt"] == "What is 2+2?"
        assert data["input"]["context"]["mode"] == "oneshot"
        assert "system_hash" in data["input"]["context"]

        # advice
        assert data["advice"]["steering_text"] == "Be concise."
        assert data["advice"]["domain"] == "benchmark"
        assert data["advice"]["confidence"] == 1.0

        # frontier
        assert data["frontier"]["output"] == "4"
        assert data["frontier"]["model"] == "gpt-4o"
        assert data["frontier"]["usage"]["prompt_tokens"] == 20
        assert data["frontier"]["usage"]["completion_tokens"] == 5
        assert data["frontier"]["usage"]["total_tokens"] == 25

        # reward
        assert data["reward"]["score"] is None
        assert data["reward"]["breakdown"] == {}

        # metadata
        assert data["metadata"]["benchmark_mode"] == "oneshot"
        assert data["metadata"]["timing_ms"] == 150

    def test_session_id_derived_from_system_hash(
        self, trace_dir: Path,
    ) -> None:
        writer = BenchTraceWriter(output_dir=str(trace_dir))
        writer.write_trace(
            request=_sample_request(),
            advice="A",
            response=_sample_response(),
            mode="session",
            timing_ms=10,
        )
        writer.write_trace(
            request=_sample_request(),
            advice="B",
            response=_sample_response(),
            mode="session",
            timing_ms=10,
        )
        yaml_files = sorted(trace_dir.rglob("*.yaml"))
        assert len(yaml_files) == 2
        with yaml_files[0].open() as f:
            d0 = yaml.safe_load(f)
        with yaml_files[1].open() as f:
            d1 = yaml.safe_load(f)
        # same system prompt => same session_id
        assert d0["session_id"] == d1["session_id"]

    def test_no_system_message(self, trace_dir: Path) -> None:
        request = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt-4o",
        }
        writer = BenchTraceWriter(output_dir=str(trace_dir))
        trace_id = writer.write_trace(
            request=request,
            advice="advice",
            response=_sample_response(),
            mode="plan",
            timing_ms=5,
        )
        assert trace_id  # no crash

    def test_provider_from_config(self, trace_dir: Path) -> None:
        writer = BenchTraceWriter(
            output_dir=str(trace_dir), provider="anthropic",
        )
        writer.write_trace(
            request=_sample_request(),
            advice="x",
            response=_sample_response(),
            mode="oneshot",
            timing_ms=1,
        )
        yaml_file = next(trace_dir.rglob("*.yaml"))
        with yaml_file.open() as f:
            data = yaml.safe_load(f)
        assert data["frontier"]["provider"] == "anthropic"


class TestTraceIngesterCompat:
    """Verify BenchTraceWriter output loads via TraceIngester."""

    def test_load_yaml_dir(self, trace_dir: Path) -> None:
        writer = BenchTraceWriter(
            output_dir=str(trace_dir), provider="openai",
        )
        writer.write_trace(
            request=_sample_request(),
            advice="Be concise.",
            response=_sample_response(),
            mode="oneshot",
            timing_ms=150,
        )
        ingester = TraceIngester()
        ingester.load_yaml_dir(trace_dir)
        records = ingester.to_trace_records()

        assert len(records) == 1
        rec = records[0]
        assert rec.prompt == "What is 2+2?"
        assert rec.advice_text == "Be concise."
        assert rec.advice_domain == "benchmark"
        assert rec.advice_confidence == 1.0
        assert rec.frontier_output == "4"
        assert rec.frontier_model == "gpt-4o"
        assert rec.reward_score is None
        assert rec.metadata["benchmark_mode"] == "oneshot"
        assert rec.metadata["timing_ms"] == 150

    def test_multiple_traces_load(self, trace_dir: Path) -> None:
        writer = BenchTraceWriter(output_dir=str(trace_dir))
        for i in range(3):
            writer.write_trace(
                request=_sample_request(),
                advice=f"advice-{i}",
                response=_sample_response(),
                mode="session",
                timing_ms=i * 10,
            )
        ingester = TraceIngester()
        ingester.load_yaml_dir(trace_dir)
        assert ingester.count() == 3
