"""Tests for fit.training.tracer — SQLite ingestion path."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fit.training.tracer import TraceIngester


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
        conn.execute(f"INSERT INTO {table} (data) VALUES (?)", (row,))
    conn.commit()
    conn.close()


class TestLoadSqliteTableValidation:
    """load_sqlite must validate the table parameter against injection."""

    def _make_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute(
            "INSERT INTO traces VALUES (?)", (json.dumps({"input": {}}),)
        )
        conn.commit()
        conn.close()
        return db_path

    def test_injection_attempt_raises(self, tmp_path: Path) -> None:
        db_path = self._make_db(tmp_path)
        with pytest.raises(ValueError, match="table"):
            TraceIngester().load_sqlite(
                db_path, table="traces; DROP TABLE traces--"
            )

    def test_table_with_space_raises(self, tmp_path: Path) -> None:
        db_path = self._make_db(tmp_path)
        with pytest.raises(ValueError, match="table"):
            TraceIngester().load_sqlite(db_path, table="bad name")

    def test_empty_table_raises(self, tmp_path: Path) -> None:
        db_path = self._make_db(tmp_path)
        with pytest.raises(ValueError, match="table"):
            TraceIngester().load_sqlite(db_path, table="")

    def test_valid_table_works(self, tmp_path: Path) -> None:
        db_path = self._make_db(tmp_path)
        ingester = TraceIngester().load_sqlite(db_path, table="traces")
        assert ingester.count() == 1


class TestSqliteJsonBlobNonDict:
    """SQLite JSON blob path must raise ValueError for non-dict values."""

    def test_array_blob_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute(
            "INSERT INTO traces (data) VALUES (?)",
            (json.dumps([1, 2, 3]),),
        )
        conn.commit()
        conn.close()
        with pytest.raises(ValueError, match=r"dict|JSON object|row"):
            TraceIngester().load_sqlite(str(db_path))

    def test_string_blob_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute(
            "INSERT INTO traces (data) VALUES (?)",
            (json.dumps("just a string"),),
        )
        conn.commit()
        conn.close()
        with pytest.raises(ValueError, match=r"dict|JSON object|row"):
            TraceIngester().load_sqlite(str(db_path))


class TestSqliteJsonBlobHappyPath:
    """JSON-blob (data column) path loads traces correctly."""

    def test_loads_multiple_valid_traces(self, tmp_path: Path) -> None:
        blobs = [
            _make_trace_blob(trace_id=f"t-{i}", prompt=f"p{i}", score=float(i)/10)
            for i in range(1, 4)
        ]
        db_path = tmp_path / "traces.db"
        _create_sqlite_with_data_column(db_path, [json.dumps(b) for b in blobs])
        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()
        assert len(records) == 3
        for i, rec in enumerate(records, 1):
            assert rec.id == f"t-{i}"

    def test_non_trace_dict_still_loaded(self, tmp_path: Path) -> None:
        """JSON blob path doesn't filter by input/frontier keys."""
        valid = _make_trace_blob()
        non_trace = {"foo": "bar", "baz": 42}
        db_path = tmp_path / "mixed.db"
        _create_sqlite_with_data_column(
            db_path, [json.dumps(valid), json.dumps(non_trace)]
        )
        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        assert ingester.count() == 2

    def test_empty_table_returns_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        _create_sqlite_with_data_column(db_path, [])
        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        assert ingester.count() == 0


class TestSqliteJsonBlobErrorHandling:
    """Error handling for JSON blob path."""

    def test_invalid_json_raises_with_context(self, tmp_path: Path) -> None:
        db_path = tmp_path / "bad.db"
        _create_sqlite_with_data_column(db_path, ["not valid json"])
        with pytest.raises(ValueError, match=r"row|table"):
            TraceIngester().load_sqlite(db_path)

    def test_non_dict_array_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "array.db"
        _create_sqlite_with_data_column(db_path, [json.dumps([1, 2, 3])])
        with pytest.raises(ValueError, match=r"dict"):
            TraceIngester().load_sqlite(db_path)

    def test_mixed_valid_invalid_raises_on_bad_row(self, tmp_path: Path) -> None:
        valid_1 = json.dumps(_make_trace_blob(trace_id="t-ok-1"))
        valid_2 = json.dumps(_make_trace_blob(trace_id="t-ok-2"))
        invalid = "{{broken json}}"
        db_path = tmp_path / "mixed_err.db"
        _create_sqlite_with_data_column(db_path, [valid_1, valid_2, invalid])
        with pytest.raises(ValueError, match=r"row\s+3"):
            TraceIngester().load_sqlite(db_path)


class TestSqliteFallbackBranch:
    """Fallback branch (no data column) with JSON text columns."""

    def test_nested_json_columns_produce_populated_records(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "nested.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE traces ("
            "  id TEXT, session_id TEXT, timestamp TEXT,"
            "  input TEXT, advice TEXT, frontier TEXT,"
            "  reward TEXT, metadata TEXT)"
        )
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "t-100", "sess-100", "2025-03-01T12:00:00Z",
                json.dumps({"prompt": "nested prompt", "context": {}}),
                json.dumps({"steering_text": "nested advice", "domain": "d", "confidence": 0.9}),
                json.dumps({"output": "nested output", "model": "gpt-4"}),
                json.dumps({"score": 0.9}),
                json.dumps({"tenant": "acme"}),
            ),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()
        assert len(records) == 1
        rec = records[0]
        assert rec.prompt == "nested prompt"
        assert rec.advice_text == "nested advice"
        assert rec.frontier_output == "nested output"

    @pytest.mark.xfail(
        strict=True,
        reason="Truly flat columns not mapped to nested schema — known limitation",
    )
    def test_truly_flat_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "flat.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE traces (prompt TEXT, output TEXT, score REAL, domain TEXT)"
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
        assert len(records) == 1
        assert records[0].prompt == "flat prompt"

    def test_corrupt_json_in_fallback_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (id TEXT, input TEXT, frontier TEXT)")
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?)",
            ("row-1", 'not valid json{"', '{"model": "test"}'),
        )
        conn.commit()
        conn.close()
        with pytest.raises(ValueError, match=r"input|column"):
            TraceIngester().load_sqlite(db_path)


class TestSqliteCursorStreaming:
    """load_sqlite must stream rows via cursor, not fetchall."""

    def test_json_blob_path_no_fetchall(self) -> None:
        import inspect
        source = inspect.getsource(TraceIngester.load_sqlite)
        blob_start = source.index("JSON blob column first")
        blob_end = source.index("Fallback: individual columns")
        blob_section = source[blob_start:blob_end]
        assert ".fetchall()" not in blob_section

    def test_fallback_path_no_fetchall(self) -> None:
        import inspect
        source = inspect.getsource(TraceIngester.load_sqlite)
        fallback_start = source.index("Fallback: individual columns")
        fallback_section = source[fallback_start:]
        assert ".fetchall()" not in fallback_section
