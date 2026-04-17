"""L0 smoke: ingest -> dataset -> dry-run pipeline with synthetic traces."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fit.training.dataset import DatasetBuilder, FitDataset
from fit.training.tracer import TraceIngester

from .conftest import generate_traces, write_jsonl, write_yaml_dir


class TestSmokeJSONL:
    """Pipeline smoke test via JSONL ingestion."""

    def test_smoke_pipeline_jsonl(
        self,
        tmp_traces: tuple[list[dict[str, Any]], Path],
    ) -> None:
        traces, tmp_path = tmp_traces
        jsonl_path = write_jsonl(tmp_path / "traces.jsonl", traces)

        ingester = TraceIngester()
        ingester.load_jsonl(jsonl_path)
        assert ingester.count() == 50

        records = ingester.to_trace_records()
        dataset = DatasetBuilder(records).build()
        assert len(dataset) > 0

        train, val = dataset.split(val_ratio=0.2)
        assert len(train) + len(val) == len(dataset)

        stats = dataset.reward_stats()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert stats["count"] == 50.0


class TestSmokeYAMLDir:
    """Pipeline smoke test via YAML cassette ingestion."""

    def test_smoke_pipeline_yaml_dir(
        self,
        tmp_traces: tuple[list[dict[str, Any]], Path],
    ) -> None:
        traces, tmp_path = tmp_traces
        yaml_dir = write_yaml_dir(tmp_path / "cassettes", traces)

        ingester = TraceIngester()
        ingester.load_yaml_dir(yaml_dir)
        assert ingester.count() == 50

        records = ingester.to_trace_records()
        dataset = DatasetBuilder(records).build()
        assert len(dataset) > 0

        train, val = dataset.split(val_ratio=0.2)
        assert len(train) + len(val) == len(dataset)


class TestSmokeMixedBatch:
    """Pipeline smoke test with mixed JSONL + YAML in same dir."""

    def test_smoke_pipeline_mixed_batch(
        self,
        tmp_traces: tuple[list[dict[str, Any]], Path],
    ) -> None:
        traces, tmp_path = tmp_traces

        # Split traces: half to JSONL, half to YAML
        half = len(traces) // 2
        jsonl_traces = traces[:half]
        yaml_traces = traces[half:]

        batch_dir = tmp_path / "mixed"
        batch_dir.mkdir(parents=True, exist_ok=True)

        write_jsonl(batch_dir / "batch.jsonl", jsonl_traces)
        write_yaml_dir(batch_dir / "cassettes", yaml_traces)

        ingester = TraceIngester()
        # load_batch auto-detects formats in a directory
        ingester.load_batch([batch_dir])

        # Should pick up both JSONL and YAML traces
        assert ingester.count() == 50

        records = ingester.to_trace_records()
        dataset = DatasetBuilder(records).build()
        assert len(dataset) == 50
