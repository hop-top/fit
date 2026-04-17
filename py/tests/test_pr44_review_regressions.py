"""Regression tests for PR #44 code review items (tracer).

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from fit.training.tracer import TraceIngester


# -- helpers -----------------------------------------------------------------

def _make_trace_blob(
    *,
    prompt: str = "test prompt",
    output: str = "test output",
    model: str = "gpt-4",
    score: float = 0.85,
    domain: str = "fitness",
    session_id: str = "sess-001",
    trace_id: str = "t-001",
    timestamp: str = "2025-01-15T10:00:00Z",
) -> dict:
    """Build a valid trace dict matching the expected nested schema."""
    return {
        "id": trace_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "input": {"prompt": prompt, "context": {"source": "test"}},
        "advice": {
            "steering_text": f"advice for {domain}",
            "domain": domain,
            "confidence": 0.9,
        },
        "frontier": {"output": output, "model": model},
        "reward": {"score": score, "breakdown": {"quality": 0.8}},
        "metadata": {"tenant": "acme"},
    }


def _create_sqlite_with_data_column(
    db_path: Path,
    rows: list[str],
    table: str = "traces",
) -> None:
    """Create a SQLite DB with a ``data TEXT`` column and insert rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(f"CREATE TABLE {table} (data TEXT)")
    for row in rows:
        conn.execute(
            f"INSERT INTO {table} (data) VALUES (?)", (row,)
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Bug 1: yaml.safe_load() without YAMLError catch
# ---------------------------------------------------------------------------


class TestYamlSafeLoadMissingYAMLErrorCatch:
    """load_yaml_dir calls yaml.safe_load without catching yaml.YAMLError.

    A malformed YAML file raises a raw ``yaml.YAMLError`` that escapes
    without any file-path context, making it impossible for callers to
    identify which file is corrupt.

    Expected: ``ValueError`` whose message includes the file path.
    Actual:   raw ``yaml.YAMLError``.
    """

    def test_malformed_yaml_raises_value_error_with_path(
        self, tmp_path: Path
    ) -> None:
        """Malformed YAML should raise ValueError containing the file path."""
        bad_file = tmp_path / "step-001.yaml"
        bad_file.write_text(":\n  - [invalid", encoding="utf-8")

        ingester = TraceIngester()

        with pytest.raises(ValueError, match=r"step-001\.yaml") as exc_info:
            ingester.load_yaml_dir(tmp_path)

        assert exc_info.type is ValueError, (
            "Bug confirmed: load_yaml_dir raises raw yaml.YAMLError "
            "instead of ValueError with file-path context"
        )


# ---------------------------------------------------------------------------
# Bug 2: SQLite fallback branch produces empty TraceRecords
# ---------------------------------------------------------------------------


class TestLoadSqliteJsonBlobHappyPath:
    """Verify the JSON-blob (``data`` column) path works correctly.

    These tests exercise the primary code path in ``load_sqlite`` where
    each row has a single ``data TEXT`` column containing a full JSON
    trace object.  They should pass without changes.
    """

    def test_loads_multiple_valid_traces(self, tmp_path: Path) -> None:
        """Three valid JSON blobs load into three correct TraceRecords."""
        blobs = [
            _make_trace_blob(
                trace_id=f"t-{i}",
                prompt=f"prompt {i}",
                output=f"output {i}",
                score=float(i) / 10,
            )
            for i in range(1, 4)
        ]
        db_path = tmp_path / "traces.db"
        _create_sqlite_with_data_column(
            db_path, [json.dumps(b) for b in blobs]
        )

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()

        assert len(records) == 3, (
            f"Expected 3 records, got {len(records)}"
        )
        for i, rec in enumerate(records, 1):
            assert rec.id == f"t-{i}", (
                f"Record {i} id mismatch: {rec.id}"
            )
            assert rec.prompt == f"prompt {i}", (
                f"Record {i} prompt mismatch: {rec.prompt}"
            )
            assert rec.frontier_output == f"output {i}", (
                f"Record {i} frontier_output mismatch: {rec.frontier_output}"
            )
            assert rec.reward_score == float(i) / 10, (
                f"Record {i} reward_score mismatch: {rec.reward_score}"
            )

    def test_non_trace_dict_still_loaded_via_json_blob(
        self, tmp_path: Path
    ) -> None:
        """JSON blob path does not filter by input/frontier keys.

        Unlike load_batch's JSON branch which checks for ``input`` or
        ``frontier`` keys, load_sqlite's JSON blob path feeds every row
        through ``_parse_raw``.  A non-trace dict should still produce a
        record (with empty/default fields).
        """
        valid_blob = _make_trace_blob()
        non_trace = {"foo": "bar", "baz": 42}

        db_path = tmp_path / "mixed.db"
        _create_sqlite_with_data_column(
            db_path,
            [json.dumps(valid_blob), json.dumps(non_trace)],
        )

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()

        assert len(records) == 2, (
            "Both rows should be loaded — JSON blob path does not "
            f"filter by input/frontier; got {len(records)}"
        )
        # The non-trace record should have empty defaults
        non_trace_rec = records[1]
        assert non_trace_rec.prompt == "", (
            "Non-trace dict should produce empty prompt"
        )

    def test_empty_table_returns_zero_records(
        self, tmp_path: Path
    ) -> None:
        """Empty traces table should not crash and return 0 records."""
        db_path = tmp_path / "empty.db"
        _create_sqlite_with_data_column(db_path, [])

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)

        assert ingester.count() == 0, (
            "Empty table should produce 0 records without crashing"
        )


class TestLoadSqliteFallbackBranch:
    """Fallback branch (no ``data`` column) produces empty TraceRecords.

    When the table has individual columns instead of a ``data`` JSON blob,
    ``load_sqlite`` falls through to ``SELECT * FROM table`` and builds
    ``raw = dict(zip(cols, row))`` with flat column names.

    ``_parse_raw()`` expects nested dicts under keys ``input``, ``advice``,
    ``frontier``, ``reward``.  Flat columns like ``prompt``, ``output``,
    ``score`` are ignored, producing records where every meaningful field
    is empty/default.

    Even when columns *are* named ``input``, ``advice``, etc. and contain
    JSON text, ``_parse_raw`` receives the raw JSON string (not a parsed
    dict), so ``isinstance(inp_raw, dict)`` fails and the value is
    discarded.
    """

    def test_nested_json_columns_produce_populated_records(
        self, tmp_path: Path
    ) -> None:
        """Columns named input/advice/frontier/reward with JSON text.

        The fallback path should parse JSON strings in these columns
        back into dicts so _parse_raw can extract fields.  Currently
        the raw JSON strings fail the isinstance(x, dict) check and
        get replaced with {}.
        """
        db_path = tmp_path / "nested.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE traces ("
            "  id TEXT,"
            "  session_id TEXT,"
            "  timestamp TEXT,"
            "  input TEXT,"
            "  advice TEXT,"
            "  frontier TEXT,"
            "  reward TEXT,"
            "  metadata TEXT"
            ")"
        )
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "t-100",
                "sess-100",
                "2025-03-01T12:00:00Z",
                json.dumps({
                    "prompt": "nested prompt",
                    "context": {"source": "test"},
                }),
                json.dumps({
                    "steering_text": "nested advice",
                    "domain": "nutrition",
                    "confidence": 0.95,
                }),
                json.dumps({
                    "output": "nested output",
                    "model": "gpt-4",
                }),
                json.dumps({
                    "score": 0.9,
                    "breakdown": {"quality": 0.85},
                }),
                json.dumps({"tenant": "acme"}),
            ),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()

        assert len(records) == 1, (
            f"Expected 1 record, got {len(records)}"
        )
        rec = records[0]
        assert rec.prompt == "nested prompt", (
            "Bug confirmed: fallback branch produces empty prompt; "
            f"got {rec.prompt!r}"
        )
        assert rec.advice_text == "nested advice", (
            "Bug confirmed: fallback branch produces empty advice_text; "
            f"got {rec.advice_text!r}"
        )
        assert rec.frontier_output == "nested output", (
            "Bug confirmed: fallback branch produces empty frontier_output; "
            f"got {rec.frontier_output!r}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Truly flat columns (prompt, output, score) are not "
            "mapped to nested TraceRecord schema — known limitation"
        ),
    )
    def test_truly_flat_columns_produce_populated_records(
        self, tmp_path: Path
    ) -> None:
        """Columns like prompt, output, score, domain — no nesting at all.

        The fallback should map well-known flat column names to the
        corresponding TraceRecord fields.  Currently every field ends
        up as its empty default.
        """
        db_path = tmp_path / "flat.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE traces ("
            "  prompt TEXT,"
            "  output TEXT,"
            "  score REAL,"
            "  domain TEXT"
            ")"
        )
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?)",
            ("flat prompt", "flat output", 0.77, "strength"),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()

        assert len(records) == 1, (
            f"Expected 1 record, got {len(records)}"
        )
        rec = records[0]
        assert rec.prompt == "flat prompt", (
            "Bug confirmed: flat column 'prompt' not mapped; "
            f"got {rec.prompt!r}"
        )
        assert rec.frontier_output == "flat output", (
            "Bug confirmed: flat column 'output' not mapped; "
            f"got {rec.frontier_output!r}"
        )
        assert rec.reward_score == 0.77, (
            "Bug confirmed: flat column 'score' not mapped; "
            f"got {rec.reward_score!r}"
        )


class TestLoadSqliteJsonBlobErrorHandling:
    """Edge cases and error handling for the JSON blob (data column) path."""

    def test_invalid_json_raises_value_error_with_context(
        self, tmp_path: Path
    ) -> None:
        """Corrupt JSON in data column should raise ValueError with context.

        Covered by PR #43 but included for completeness.
        """
        db_path = tmp_path / "bad_json.db"
        _create_sqlite_with_data_column(
            db_path, ["not valid json at all"]
        )

        ingester = TraceIngester()

        with pytest.raises(ValueError, match=r"row|table"):
            ingester.load_sqlite(db_path)

    def test_non_dict_json_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Array JSON in data column should raise ValueError mentioning dict."""
        db_path = tmp_path / "array.db"
        _create_sqlite_with_data_column(
            db_path, [json.dumps([1, 2, 3])]
        )

        ingester = TraceIngester()

        with pytest.raises(ValueError, match=r"dict") as exc_info:
            ingester.load_sqlite(db_path)

        assert exc_info.type is ValueError, (
            "Bug confirmed: non-dict JSON should raise ValueError "
            "mentioning 'dict'"
        )

    def test_mixed_valid_invalid_raises_on_bad_row(
        self, tmp_path: Path
    ) -> None:
        """Mix of valid and invalid rows should error on the bad row with index."""
        valid_1 = json.dumps(_make_trace_blob(trace_id="t-ok-1"))
        valid_2 = json.dumps(_make_trace_blob(trace_id="t-ok-2"))
        invalid = "{{broken json}}"

        db_path = tmp_path / "mixed_err.db"
        _create_sqlite_with_data_column(
            db_path, [valid_1, valid_2, invalid]
        )

        ingester = TraceIngester()

        with pytest.raises(ValueError, match=r"row\s+3") as exc_info:
            ingester.load_sqlite(db_path)

        assert exc_info.type is ValueError, (
            "Bug confirmed: invalid row should raise ValueError "
            "with row index context"
        )


# ---------------------------------------------------------------------------
# Bug 3: `format` parameter shadows built-in
# ---------------------------------------------------------------------------


class TestFormatParameterShadowsBuiltin:
    """load_batch uses ``format`` as a parameter name, shadowing the built-in.

    While not a runtime crash, shadowing ``format()`` is a style/lint issue
    and can cause subtle bugs if the function body ever needs the built-in.
    The parameter should be renamed to ``fmt`` or similar.

    Expected: parameter named ``fmt`` (or anything other than ``format``).
    Actual:   parameter named ``format``.
    """

    def test_load_batch_does_not_shadow_format_builtin(self) -> None:
        """The second parameter of load_batch should not be named 'format'."""
        sig = inspect.signature(TraceIngester.load_batch)
        param_names = list(sig.parameters.keys())

        assert "format" not in param_names, (
            "Bug confirmed: load_batch uses 'format' as parameter name, "
            f"shadowing the built-in; params: {param_names}"
        )
