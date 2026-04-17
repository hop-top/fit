"""Regression tests for PR #39 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect


# ---------------------------------------------------------------------------
# Bug 1 & 2: load_sqlite uses .fetchall() instead of cursor streaming
# File: py/src/fit/training/tracer.py, lines 145 and 164
# ---------------------------------------------------------------------------


class TestLoadSqliteFetchallRegression:
    """``TraceIngester.load_sqlite`` calls ``.fetchall()`` on both the
    JSON-blob path (line 145) and the fallback individual-columns path
    (line 164). This loads entire tables into memory before iterating,
    which can cause OOM on large trace tables. Both paths should stream
    rows via cursor iteration instead.
    """

    def test_json_blob_path_does_not_use_fetchall(self) -> None:
        """The JSON-blob SELECT path must not call ``.fetchall()`` —
        it should iterate the cursor directly to avoid loading the
        full result set into memory."""
        from fit.training.tracer import TraceIngester

        source = inspect.getsource(TraceIngester.load_sqlite)
        # Isolate the JSON blob section
        blob_start = source.index("JSON blob column first")
        blob_end = source.index("Fallback: individual columns")
        blob_section = source[blob_start:blob_end]
        assert ".fetchall()" not in blob_section, (
            "Bug confirmed: JSON-blob path calls .fetchall() instead of "
            "streaming rows via cursor iteration"
        )

    def test_fallback_column_path_does_not_use_fetchall(self) -> None:
        """The fallback individual-columns SELECT path must not call
        ``.fetchall()`` — it should iterate the cursor directly."""
        from fit.training.tracer import TraceIngester

        source = inspect.getsource(TraceIngester.load_sqlite)
        # Isolate the fallback section
        fallback_start = source.index("Fallback: individual columns")
        fallback_section = source[fallback_start:]
        assert ".fetchall()" not in fallback_section, (
            "Bug confirmed: fallback column path calls .fetchall() instead "
            "of streaming rows via cursor iteration"
        )
