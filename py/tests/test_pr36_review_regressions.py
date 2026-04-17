"""Regression tests for PR #36 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fit.training.tracer import TraceIngester


# ---------------------------------------------------------------------------
# Bug 1: JSON branch in load_batch() ingests dicts unconditionally
# File: tracer.py, lines 209-219
# ---------------------------------------------------------------------------


class TestJsonBranchIngestsNonTraceDictsRegression:
    """load_batch JSON path appends every dict via _parse_raw() without
    checking for ``input`` or ``frontier`` keys. Unlike the YAML path,
    non-trace dicts (config files, reward schemas) produce mostly-empty
    TraceRecords instead of being skipped.
    """

    def test_json_array_filters_non_trace_dicts(
        self, tmp_path: Path
    ) -> None:
        """JSON array with one config dict and one trace dict should
        yield count == 1 (only the trace)."""
        data = [
            {"app": "fit"},
            {
                "input": {"prompt": "x"},
                "frontier": {"model": "m", "output": "ok"},
            },
        ]
        p = tmp_path / "mixed.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        ingester = TraceIngester().load_batch([p])
        assert ingester.count() == 1, (
            "Bug confirmed: JSON branch ingested non-trace dict. "
            f"Expected 1 record, got {ingester.count()}."
        )

    def test_single_non_trace_json_object_yields_zero(
        self, tmp_path: Path
    ) -> None:
        """A top-level JSON object without input/frontier keys must
        produce count == 0."""
        data = {"app": "fit", "version": "1.0"}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        ingester = TraceIngester().load_batch([p])
        assert ingester.count() == 0, (
            "Bug confirmed: JSON branch ingested non-trace top-level dict. "
            f"Expected 0 records, got {ingester.count()}."
        )


# ---------------------------------------------------------------------------
# Bug 2: train_advisor.py example loads all *.json including non-trace fixtures
# File: train_advisor.py, lines 109-115
# ---------------------------------------------------------------------------


class TestTrainAdvisorGlobsNonTraceJsonRegression:
    """train_advisor.py globs ``*.json`` and feeds every match to
    ``load_batch``. Non-trace JSON files (e.g. advice-v1.json,
    reward-v1.json) are ingested as mostly-empty TraceRecords.
    This depends on Bug 1 being fixed (load_batch filtering).
    """

    def test_glob_json_with_mixed_files_loads_only_traces(
        self, tmp_path: Path
    ) -> None:
        """Simulates the example's glob + load_batch flow. A dir with
        one trace JSON and one config JSON should yield count == 1."""
        trace = {
            "input": {"prompt": "hello"},
            "frontier": {"model": "gpt-4", "output": "world"},
        }
        config = {"schema": "reward-v1", "weights": [0.5, 0.5]}

        (tmp_path / "trace-001.json").write_text(
            json.dumps(trace), encoding="utf-8"
        )
        (tmp_path / "reward-v1.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

        # Replicate the example's glob pattern
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 2, "Fixture setup: expected 2 JSON files"

        ingester = TraceIngester().load_batch(json_files)
        assert ingester.count() == 1, (
            "Bug confirmed: train_advisor glob picks up non-trace JSON. "
            f"Expected 1 trace record, got {ingester.count()}."
        )


# ---------------------------------------------------------------------------
# Bug 3: test_tracer.py docstring says "step-NNN" but tests validate
#         content-based filtering
# File: test_tracer.py, line 136
# ---------------------------------------------------------------------------


class TestLoadYamlDirDocstringAccuracyRegression:
    """``TestLoadYamlDirFiltersNonTraceRegressions`` docstring claims
    "load_yaml_dir must only ingest step-NNN trace files" but the
    actual tests (and implementation) filter by content (input/frontier
    keys), not by filename pattern. The docstring is misleading.
    """

    def test_docstring_does_not_mention_step_nnn(self) -> None:
        """Docstring must describe content-based filtering, not
        filename-pattern filtering."""
        from tests.test_tracer import (
            TestLoadYamlDirFiltersNonTraceRegressions,
        )

        doc = TestLoadYamlDirFiltersNonTraceRegressions.__doc__ or ""
        mentions_step = "step-NNN" in doc or "step-nnn" in doc.lower()
        assert not mentions_step, (
            "Bug confirmed: docstring mentions 'step-NNN' filename "
            "filtering but the tests and implementation use content-based "
            "filtering (input/frontier keys). Docstring is misleading."
        )
