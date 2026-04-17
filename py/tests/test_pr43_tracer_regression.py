"""Regression tests for PR #43 code review items (tracer).

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fit.training.tracer import TraceIngester


# ---------------------------------------------------------------------------
# Bug: load_sqlite doesn't catch json.JSONDecodeError
# ---------------------------------------------------------------------------


class TestLoadSqliteMissingJSONDecodeErrorWrapping:
    """load_sqlite calls json.loads() without catching json.JSONDecodeError.

    Unlike load_jsonl(), which wraps decode errors with location context
    (file path, line number), load_sqlite lets a raw JSONDecodeError escape
    when a row contains invalid JSON in the ``data`` column.  Callers get
    no table or row context, making debugging difficult.

    Expected: ``ValueError`` whose message includes table/row context.
    Actual:   raw ``json.JSONDecodeError``.
    """

    def test_invalid_json_raises_value_error_with_context(
        self, tmp_path: Path
    ) -> None:
        """Corrupt JSON in data column should raise ValueError, not JSONDecodeError."""
        db_path = tmp_path / "bad.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute(
            "INSERT INTO traces (data) VALUES (?)",
            ("not json at all",),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()

        with pytest.raises(
            ValueError,
            match=r"row|table",
        ) as exc_info:
            ingester.load_sqlite(db_path)

        assert exc_info.type is ValueError, (
            "Bug confirmed: load_sqlite raises raw JSONDecodeError "
            "instead of ValueError with table/row context"
        )
