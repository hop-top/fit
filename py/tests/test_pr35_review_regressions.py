"""Regression tests for PR #35 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from fit.training.tracer import TraceIngester


# ---------------------------------------------------------------------------
# Bug 1: SQLite JSON blob path accepts non-dict JSON
# File: tracer.py, line 147
# ---------------------------------------------------------------------------


class TestSqliteJsonBlobNonDictRegression:
    """load_sqlite deserialises each row's ``data`` column via
    ``json.loads`` then passes the result to ``_parse_raw()``.
    ``_parse_raw`` expects a dict and calls ``.get()``, so a
    non-dict value (list, str, int) raises ``AttributeError``
    instead of a clear ``ValueError``.
    """

    def test_non_dict_json_blob_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """A JSON array stored in the data column must raise
        ValueError, not AttributeError."""
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

    def test_non_dict_string_blob_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """A plain JSON string stored in the data column must raise
        ValueError, not AttributeError."""
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


# ---------------------------------------------------------------------------
# Bug 2: Double blank line in grpo.py (E303)
# File: grpo.py, lines 193-194
# ---------------------------------------------------------------------------


class TestGrpoDoubleBlankLineRegression:
    """``_train_simplified`` has two consecutive blank lines after
    ``import random`` (lines 193-194), violating Ruff E303 which
    forbids more than one blank line inside a function body.
    """

    def test_no_consecutive_blank_lines_after_import_random(self) -> None:
        """Source of _train_simplified must not have consecutive
        blank lines after ``import random``."""
        from fit.training.grpo import GRPOTrainer

        source = inspect.getsource(GRPOTrainer._train_simplified)
        lines = source.splitlines()

        import_idx = None
        for i, line in enumerate(lines):
            if "import random" in line:
                import_idx = i
                break

        assert import_idx is not None, (
            "Could not find 'import random' in _train_simplified source"
        )

        # Check for consecutive blank lines after import random
        for i in range(import_idx + 1, len(lines) - 1):
            current_blank = lines[i].strip() == ""
            next_blank = lines[i + 1].strip() == ""
            assert not (current_blank and next_blank), (
                "Bug confirmed: consecutive blank lines at "
                f"lines {i} and {i + 1} (relative to function start) "
                "after 'import random' in _train_simplified. "
                "Ruff E303 violation."
            )
            # Only check the region right after import random
            if not current_blank:
                break


# ---------------------------------------------------------------------------
# Bug 3: Docstring claims xfail marker that doesn't exist
# File: test_tracer.py, lines 579-580
# ---------------------------------------------------------------------------


class TestPR34DocstringClaimsXfailRegression:
    """Guard against reintroducing ``xfail`` wording into
    ``TestPR34LoadBatchErrorMessageWordingRegression`` unless the
    class also has a matching ``@pytest.mark.xfail`` decorator.

    The regression check below intentionally verifies the general
    contract: either an xfail marker exists, or the docstring does
    not claim xfail.
    """

    def test_docstring_does_not_claim_xfail(self) -> None:
        """Docstring must not claim xfail if no marker is present."""
        from tests.test_tracer import (
            TestPR34LoadBatchErrorMessageWordingRegression,
        )

        doc = TestPR34LoadBatchErrorMessageWordingRegression.__doc__ or ""
        markers = getattr(
            TestPR34LoadBatchErrorMessageWordingRegression,
            "pytestmark",
            [],
        )
        marker_names = [m.name for m in markers]
        has_xfail_marker = "xfail" in marker_names
        claims_xfail = "xfail" in doc.lower()

        # Either the marker exists, or the docstring doesn't claim it
        assert has_xfail_marker or not claims_xfail, (
            "Bug confirmed: "
            "TestPR34LoadBatchErrorMessageWordingRegression docstring "
            "claims xfail(strict=True) but no @pytest.mark.xfail "
            "decorator exists on the class."
        )
