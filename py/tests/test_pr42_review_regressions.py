"""Regression tests for PR #42 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Bug 1: JSON branch silently ignores non-list/non-dict top-level values
# File: py/src/fit/training/tracer.py, line ~220
# ---------------------------------------------------------------------------


class TestJsonBareValueSilentlyIgnored:
    """``TraceIngester.load_batch()`` JSON branch deserialises the file
    with ``json.load()`` and then checks ``isinstance(raw, list)`` and
    ``isinstance(raw, dict)``. When the top-level value is a bare
    string, number, or null the code falls through both branches and
    returns silently with zero records, instead of raising
    ``ValueError`` as ``load_jsonl()`` does for non-dict records.
    """

    def test_bare_string_raises(self, tmp_path: Path) -> None:
        """A JSON file containing a bare string (``"hello"``) must
        raise ``ValueError`` — not silently succeed with 0 records."""
        from fit.training.tracer import TraceIngester

        p = tmp_path / "bare_string.json"
        p.write_text('"hello"', encoding="utf-8")

        with pytest.raises(ValueError):
            TraceIngester().load_batch([p])

    def test_bare_number_raises(self, tmp_path: Path) -> None:
        """A JSON file containing a bare number (``42``) must raise
        ``ValueError`` — not silently succeed with 0 records."""
        from fit.training.tracer import TraceIngester

        p = tmp_path / "bare_number.json"
        p.write_text("42", encoding="utf-8")

        with pytest.raises(ValueError):
            TraceIngester().load_batch([p])


# ---------------------------------------------------------------------------
# Bug 2: test_pr41 uses wrong type annotation for tmp_path
# File: py/tests/test_pr41_review_regressions.py, line 29
# ---------------------------------------------------------------------------


class TestTmpPathAnnotationRegression:
    """``test_pr41_review_regressions.TestToGgufReturnsPhantomPath
    .test_returned_path_exists_or_raises`` annotates the ``tmp_path``
    parameter as ``pytest.TempPathFactory``. The ``tmp_path`` fixture
    actually provides ``pathlib.Path``. The wrong annotation also
    keeps ``pytest`` imported solely for a type hint that is
    incorrect.
    """

    def test_tmp_path_annotation_is_path(self) -> None:
        """The ``tmp_path`` parameter of
        ``TestToGgufReturnsPhantomPath.test_returned_path_exists_or_raises``
        must be annotated as ``pathlib.Path``, not
        ``pytest.TempPathFactory``."""
        from tests.test_pr41_review_regressions import (
            TestToGgufReturnsPhantomPath,
        )

        sig = inspect.signature(
            TestToGgufReturnsPhantomPath
            .test_returned_path_exists_or_raises
        )
        ann = sig.parameters["tmp_path"].annotation
        # With `from __future__ import annotations`, annotations
        # are strings. Accept both the type and the string form.
        assert ann is Path or ann == "Path", (
            "Bug confirmed: tmp_path annotation is "
            f"{ann!r} instead of pathlib.Path"
        )
