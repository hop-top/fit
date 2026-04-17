"""L1: ingest -> dataset from HuggingFaceH4/no_robots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fit.training.dataset import DatasetBuilder
from fit.training.tracer import TraceIngester

from .conftest import write_jsonl
from .converters import no_robots_to_traces, trace_dict


# --- T-0081: converter unit test (no HF download needed) ---

_FAKE_NO_ROBOTS_ROWS = [
    {
        "category": "Brainstorm",
        "messages": [
            {"role": "user", "content": f"Give me ideas for topic {i}"},
            {"role": "assistant", "content": f"Here are ideas for topic {i}..."},
        ],
    }
    for i in range(10)
]


class TestNoRobotsConverterSchema:
    """Validate converter output schema without network."""

    def test_no_robots_converter_schema(self) -> None:
        traces = list(no_robots_to_traces(_FAKE_NO_ROBOTS_ROWS, limit=10))
        assert len(traces) == 10

        required_keys = {
            "id", "session_id", "timestamp", "input",
            "advice", "frontier", "reward", "metadata",
        }
        for t in traces:
            assert required_keys.issubset(t.keys()), (
                f"Missing keys: {required_keys - t.keys()}"
            )
            assert t["advice"]["domain"] == "Brainstorm"
            assert t["reward"]["score"] == 0.8
            assert "prompt" in t["input"]
            assert "output" in t["frontier"]

    def test_trace_dict_defaults(self) -> None:
        t = trace_dict()
        assert t["id"].startswith("tr-")
        assert t["session_id"].startswith("sess-")
        assert t["advice"]["domain"] == "general"
        assert t["reward"]["score"] == 0.5

    def test_converter_respects_limit(self) -> None:
        traces = list(no_robots_to_traces(_FAKE_NO_ROBOTS_ROWS, limit=3))
        assert len(traces) == 3


# --- T-0082: full e2e with HF download ---

datasets = pytest.importorskip("datasets")


@pytest.mark.slow
class TestNoRobotsIngest:
    """E2E ingestion from real HuggingFaceH4/no_robots dataset."""

    def test_no_robots_ingest_and_split(self, tmp_path: Path) -> None:
        ds = datasets.load_dataset(
            "HuggingFaceH4/no_robots",
            split="train",
            streaming=True,
        )
        traces = list(no_robots_to_traces(ds, limit=1000))
        assert len(traces) == 1000

        jsonl_path = write_jsonl(tmp_path / "no_robots.jsonl", traces)

        ingester = TraceIngester()
        ingester.load_jsonl(jsonl_path)
        assert ingester.count() == 1000

        records = ingester.to_trace_records()
        dataset = DatasetBuilder(records).build()
        assert len(dataset) == 1000

        train, val = dataset.split(val_ratio=0.1)
        assert 85 <= len(val) <= 115  # ~100 +/- tolerance
        assert 885 <= len(train) <= 915  # ~900 +/- tolerance

        stats = dataset.reward_stats()
        # All rewards are 0.8 -> after normalization all become 0.5
        assert stats["count"] == 1000.0

        # Filter by domain "Brainstorm" -> subset
        filtered = ingester.filter(domain="Brainstorm")
        assert filtered.count() < 1000
        assert filtered.count() > 0

    def test_no_robots_category_distribution(self) -> None:
        ds = datasets.load_dataset(
            "HuggingFaceH4/no_robots",
            split="train",
            streaming=True,
        )
        traces = list(no_robots_to_traces(ds, limit=1000))

        domains = {t["advice"]["domain"] for t in traces}
        assert len(domains) >= 5, (
            f"Expected >= 5 categories, got {len(domains)}: {domains}"
        )
