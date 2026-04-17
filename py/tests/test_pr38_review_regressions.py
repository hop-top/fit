"""Regression tests for PR #38 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# Bug 1: Redundant trainer.save() in train_advisor.py
# File: py/examples/train_advisor.py, line 180
# ---------------------------------------------------------------------------


class TestRedundantTrainerSaveRegression:
    """``GRPOTrainer.train()`` already calls ``self.save()`` internally
    (both TRL and simplified paths). The explicit ``trainer.save(...)``
    call after ``result = trainer.train()`` in ``main()`` is redundant,
    causing duplicate disk I/O.
    """

    def test_no_redundant_save_after_train(self) -> None:
        """The ``main`` function must not call ``trainer.save()`` after
        ``trainer.train()`` since the latter already persists the model."""
        from examples.train_advisor import main

        source = inspect.getsource(main)
        # Find the train() call, then check no trainer.save() follows
        train_idx = source.index("result = trainer.train(")
        after_train = source[train_idx:]
        assert "trainer.save(" not in after_train, (
            "Bug confirmed: trainer.save() is called after trainer.train() "
            "which already saves internally — duplicate disk I/O"
        )


# ---------------------------------------------------------------------------
# Bug 2: test_pr35 docstring inaccurately describes the original bug
# File: py/tests/test_pr35_review_regressions.py, lines 119-126
# ---------------------------------------------------------------------------


class TestPR35DocstringStaleXfailClaimRegression:
    """``TestPR34DocstringClaimsXfailRegression`` in test_pr35 has a
    docstring that references the already-fixed xfail wording instead
    of describing the general contract being guarded (marker/docstring
    sync).
    """

    def test_docstring_does_not_reference_stale_xfail(self) -> None:
        """The class docstring of ``TestPR34DocstringClaimsXfailRegression``
        must describe the guard contract, not reference the already-fixed
        xfail wording."""
        from tests.test_pr35_review_regressions import (
            TestPR34DocstringClaimsXfailRegression,
        )

        docstring = TestPR34DocstringClaimsXfailRegression.__doc__ or ""
        assert "This test is marked xfail(strict=True)" not in docstring, (
            "Bug confirmed: docstring still references stale xfail claim "
            "that was already removed in a prior fix"
        )


# ---------------------------------------------------------------------------
# Bug 3: test_tracer.py test docstring still references step-NNN pattern
# File: py/tests/test_tracer.py, lines 339-345
# ---------------------------------------------------------------------------


class TestTracerDocstringStepPatternRegression:
    """``test_non_step_pattern_file_loaded`` in test_tracer has a docstring
    that references a ``step-NNN`` filename pattern, but the loader is
    intentionally content-based (input/frontier keys), not filename-based.
    The docstring is misleading.
    """

    def test_docstring_does_not_reference_step_pattern(self) -> None:
        """The docstring of ``test_non_step_pattern_file_loaded`` must not
        reference ``step-NNN`` since the loader uses content-based detection,
        not filename patterns."""
        from tests.test_tracer import TestPR30YamlDirDocstringRegression

        method = TestPR30YamlDirDocstringRegression.test_non_step_pattern_file_loaded
        docstring = method.__doc__ or ""
        assert "step-NNN" not in docstring, (
            "Bug confirmed: docstring still references step-NNN pattern "
            "but the loader is content-based, not filename-based"
        )
