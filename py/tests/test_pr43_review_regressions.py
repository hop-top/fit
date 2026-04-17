"""Regression tests for PR #43 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# Bug 1: TestPR34TrainSimplifiedDocstringAccuracyRegression docstring
#         claims xfail but class has no @pytest.mark.xfail decorator
# File: py/tests/test_grpo.py, line ~314
# ---------------------------------------------------------------------------


class TestDocstringClaimsXfailButNoMarkerTrainSimplified:
    """``TestPR34TrainSimplifiedDocstringAccuracyRegression`` docstring
    (test_grpo.py:313-314) states "This test is marked
    xfail(strict=True)" but the class has no ``@pytest.mark.xfail``
    decorator and no ``pytestmark`` attribute containing an xfail
    marker. The docstring is misleading — either the marker must be
    added or the docstring claim must be removed.
    """

    def test_xfail_marker_matches_docstring_claim(self) -> None:
        """If the docstring claims xfail, the class must carry the
        marker; if not, the docstring must not claim it."""
        from tests.test_grpo import (
            TestPR34TrainSimplifiedDocstringAccuracyRegression,
        )

        cls = TestPR34TrainSimplifiedDocstringAccuracyRegression
        docstring = cls.__doc__ or ""
        claims_xfail = "xfail" in docstring.lower()

        markers = getattr(cls, "pytestmark", [])
        has_xfail = any(
            getattr(m, "name", None) == "xfail" for m in markers
        )

        assert has_xfail or not claims_xfail, (
            "Bug confirmed: docstring claims xfail but class has no "
            "pytest.mark.xfail marker"
        )


# ---------------------------------------------------------------------------
# Bug 2: TestPR34RewardFnIgnoredInSimplifiedRegression docstring
#         claims xfail but class has no @pytest.mark.xfail decorator
# File: py/tests/test_grpo.py, line ~364
# ---------------------------------------------------------------------------


class TestDocstringClaimsXfailButNoMarkerRewardFn:
    """``TestPR34RewardFnIgnoredInSimplifiedRegression`` docstring
    (test_grpo.py:363-364) states "This test is marked
    xfail(strict=True)" but the class has no ``@pytest.mark.xfail``
    decorator and no ``pytestmark`` attribute containing an xfail
    marker. The docstring is misleading — either the marker must be
    added or the docstring claim must be removed.
    """

    def test_xfail_marker_matches_docstring_claim(self) -> None:
        """If the docstring claims xfail, the class must carry the
        marker; if not, the docstring must not claim it."""
        from tests.test_grpo import (
            TestPR34RewardFnIgnoredInSimplifiedRegression,
        )

        cls = TestPR34RewardFnIgnoredInSimplifiedRegression
        docstring = cls.__doc__ or ""
        claims_xfail = "xfail" in docstring.lower()

        markers = getattr(cls, "pytestmark", [])
        has_xfail = any(
            getattr(m, "name", None) == "xfail" for m in markers
        )

        assert has_xfail or not claims_xfail, (
            "Bug confirmed: docstring claims xfail but class has no "
            "pytest.mark.xfail marker"
        )


# ---------------------------------------------------------------------------
# Bug 3: TestPR34EpochLossesMisleadingRegression docstring
#         claims xfail but class has no @pytest.mark.xfail decorator
# File: py/tests/test_grpo.py, line ~397
# ---------------------------------------------------------------------------


class TestDocstringClaimsXfailButNoMarkerEpochLosses:
    """``TestPR34EpochLossesMisleadingRegression`` docstring
    (test_grpo.py:396-397) states "This test is marked
    xfail(strict=True)" but the class has no ``@pytest.mark.xfail``
    decorator and no ``pytestmark`` attribute containing an xfail
    marker. The docstring is misleading — either the marker must be
    added or the docstring claim must be removed.
    """

    def test_xfail_marker_matches_docstring_claim(self) -> None:
        """If the docstring claims xfail, the class must carry the
        marker; if not, the docstring must not claim it."""
        from tests.test_grpo import (
            TestPR34EpochLossesMisleadingRegression,
        )

        cls = TestPR34EpochLossesMisleadingRegression
        docstring = cls.__doc__ or ""
        claims_xfail = "xfail" in docstring.lower()

        markers = getattr(cls, "pytestmark", [])
        has_xfail = any(
            getattr(m, "name", None) == "xfail" for m in markers
        )

        assert has_xfail or not claims_xfail, (
            "Bug confirmed: docstring claims xfail but class has no "
            "pytest.mark.xfail marker"
        )


# ---------------------------------------------------------------------------
# Bug 4: test_to_gguf_no_model only catches ImportError, misses
#         NotImplementedError that to_gguf() can also raise
# File: py/tests/test_export.py, line ~64
# ---------------------------------------------------------------------------


class TestToGgufExceptClauseMissesNotImplementedError:
    """``TestModelExporter.test_to_gguf_no_model`` (test_export.py:62-65)
    wraps ``exporter.to_gguf()`` in ``except ImportError: pass`` but
    ``to_gguf()`` now also raises ``NotImplementedError``. The test is
    environment-dependent — it passes when gguf is not installed
    (``ImportError``) but blows up when gguf is installed and the
    ``NotImplementedError`` path is hit. The except clause must catch
    both exception types.
    """

    def test_except_clause_includes_not_implemented_error(self) -> None:
        """The except clause in ``test_to_gguf_no_model`` must catch
        ``NotImplementedError`` in addition to ``ImportError``."""
        from tests.test_export import TestModelExporter

        source = inspect.getsource(
            TestModelExporter.test_to_gguf_no_model
        )

        # Find the except clause(s)
        assert "NotImplementedError" in source, (
            "Bug confirmed: test_to_gguf_no_model except clause "
            "catches ImportError but not NotImplementedError"
        )
