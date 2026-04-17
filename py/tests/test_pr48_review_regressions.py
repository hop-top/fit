"""Regression tests for PR #48 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect
import json
import sqlite3
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Bug 1: _train_simplified doesn't call model.train()
# File: py/src/fit/training/grpo.py, line 216
# After loading model via AutoModelForCausalLM.from_pretrained(), there is
# no model.train() call. Models default to eval mode.
# ---------------------------------------------------------------------------


class TestTrainSimplifiedMissingModelTrainRegression:
    """``_train_simplified`` loads a model with ``from_pretrained`` but never
    calls ``model.train()`` before the optimization loop. PyTorch models
    default to eval mode after ``from_pretrained``, so dropout and batch-norm
    layers are inactive during training.
    """

    def test_model_train_called_after_from_pretrained(self) -> None:
        """Source of ``_train_simplified`` must contain ``model.train()``
        between ``from_pretrained`` and the optimization loop."""
        from fit.training.grpo import GRPOTrainer

        src = inspect.getsource(GRPOTrainer._train_simplified)
        pretrained_pos = src.find("from_pretrained")
        loop_pos = src.find("for epoch in")
        train_call_pos = src.find("model.train()")

        assert pretrained_pos != -1, "from_pretrained not found in source"
        assert loop_pos != -1, "optimization loop not found in source"
        assert train_call_pos != -1 and pretrained_pos < train_call_pos < loop_pos, (
            "Bug confirmed: model.train() does not appear between "
            "from_pretrained and the optimization loop — the model trains "
            "in eval mode, disabling dropout and batch-norm."
        )


# ---------------------------------------------------------------------------
# Bug 2: SQLite fallback silently ignores JSON decode errors
# File: py/src/fit/training/tracer.py, lines 185-186
# except (json.JSONDecodeError, ValueError): pass silently swallows corrupt
# JSON in fallback columns, producing empty TraceRecords.
# ---------------------------------------------------------------------------


class TestSqliteFallbackSilentJsonErrorRegression:
    """``load_sqlite`` falls back to per-column JSON parsing when no ``data``
    blob exists. Corrupt JSON in those columns is silently swallowed by a
    bare ``except (json.JSONDecodeError, ValueError): pass``, yielding
    TraceRecords with empty default values and no error signal.
    """

    def test_corrupt_json_in_fallback_column_signals_error(
        self, tmp_path: Path
    ) -> None:
        """A row with invalid JSON in the ``input`` column must either
        raise ``ValueError`` with context or populate the record with
        meaningful data — not silently produce empty defaults."""
        from fit.training.tracer import TraceIngester

        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE traces (id TEXT, input TEXT, frontier TEXT)"
        )
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?)",
            ("row-1", 'not valid json{"', '{"model": "test"}'),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()
        try:
            ingester.load_sqlite(db_path)
        except ValueError:
            # Acceptable: raising with context about the corrupt data
            return

        records = ingester.to_trace_records()
        assert len(records) == 1, "Expected exactly one record"
        record = records[0]
        # If it didn't raise, the record's prompt (from input.prompt) must
        # not be silently empty.
        assert record.prompt != "", (
            "Bug confirmed: corrupt JSON in the input column was silently "
            "swallowed — the record has empty defaults with no error signal."
        )


# ---------------------------------------------------------------------------
# Bug 3: serve_advisor JSON config without error wrapping
# File: py/examples/serve_advisor.py, line 57
# json.loads() for JSON config has no JSONDecodeError catch, unlike YAML
# which wraps errors with path context via ValueError.
# ---------------------------------------------------------------------------


class TestServeAdvisorJsonConfigErrorWrappingRegression:
    """``FileAdvisor._load_config`` wraps YAML parse errors in a
    ``ValueError`` that includes the file path, but the JSON path uses
    bare ``json.loads`` with no error handling. Invalid JSON surfaces as
    a raw ``json.JSONDecodeError`` without file-path context.
    """

    def test_invalid_json_config_raises_valueerror_with_path(
        self, tmp_path: Path
    ) -> None:
        """Invalid ``advisor.json`` must raise ``ValueError`` mentioning
        the file path, matching the YAML error-wrapping behaviour."""
        # Ensure serve_advisor's parent is importable
        examples_dir = str(
            Path(__file__).resolve().parent.parent / "examples"
        )
        if examples_dir not in sys.path:
            sys.path.insert(0, examples_dir)

        from serve_advisor import FileAdvisor

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        bad_json = model_dir / "advisor.json"
        bad_json.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(ValueError, match=str(bad_json)):
            FileAdvisor(model_dir)


# ---------------------------------------------------------------------------
# Bug 4: train_advisor resolves relative paths against script dir, not cwd
# File: py/examples/train_advisor.py, line 87
# Path(__file__).resolve().parent / traces_path resolves relative to the
# script's directory instead of the caller's cwd.
# ---------------------------------------------------------------------------


class TestTrainAdvisorRelativePathResolutionRegression:
    """``train_advisor.main`` resolves relative CLI ``--traces`` paths via
    ``Path(__file__).resolve().parent``, anchoring them to the script's
    own directory. CLI arguments should resolve relative to the caller's
    working directory (``Path.cwd()``).
    """

    def test_no_path_file_in_trace_resolution(self) -> None:
        """The source of ``main`` must not use ``Path(__file__)`` for
        resolving the traces path. Currently it does."""
        examples_dir = str(
            Path(__file__).resolve().parent.parent / "examples"
        )
        if examples_dir not in sys.path:
            sys.path.insert(0, examples_dir)

        from train_advisor import main

        src = inspect.getsource(main)
        assert "Path(__file__)" not in src, (
            "Bug confirmed: relative trace paths are resolved against "
            "the script directory (Path(__file__)) instead of the "
            "caller's working directory."
        )
