"""Regression tests for PR #47 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Bug 1: _detect_format uses glob() not rglob() for JSONL/NDJSON in dirs
# ---------------------------------------------------------------------------


class TestDetectFormatNonRecursiveJsonlGlob:
    """``_detect_format()`` uses ``path.glob("*.jsonl")`` (non-recursive)
    for JSONL/NDJSON detection but ``path.rglob("*.y*ml")`` (recursive)
    for YAML.

    When a directory contains JSONL files only in subdirectories (none at
    root level), the YAML check with ``rglob`` finds nothing, falls
    through, the JSONL check with ``glob`` also finds nothing (because
    the files are nested), and the function returns the ``"yaml"`` default.

    Expected: ``"jsonl"`` when subdirectories contain ``.jsonl`` files.
    Actual:   ``"yaml"`` (falls through to default).
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "tracer.py:310 uses glob() instead of rglob() for JSONL — "
            "nested JSONL files are invisible, causing yaml default"
        ),
    )
    def test_nested_jsonl_detected_as_jsonl(self, tmp_path: Path) -> None:
        """Dir with JSONL only in a subdir should be detected as jsonl."""
        from fit.training.tracer import _detect_format

        subdir = tmp_path / "session_01"
        subdir.mkdir()
        (subdir / "traces.jsonl").write_text(
            '{"prompt":"hi","response":"hello"}\n', encoding="utf-8"
        )

        result = _detect_format(tmp_path)

        assert result == "jsonl", (
            "Bug confirmed: _detect_format uses glob() not rglob() for "
            "JSONL — nested .jsonl files mis-detected as yaml"
        )


# ---------------------------------------------------------------------------
# Bug 2: _train_simplified passes redundant labels= to model forward pass
# ---------------------------------------------------------------------------


class TestTrainSimplifiedRedundantLabels:
    """``GRPOTrainer._train_simplified`` passes ``labels=inputs["input_ids"]``
    to the model forward call at grpo.py:253, but ``outputs.loss`` is never
    used — a custom loss is computed from ``outputs.logits`` instead.

    The ``labels=`` kwarg triggers the model's internal cross-entropy loss
    computation, which is pure overhead: the result is discarded in favour
    of the manually computed policy-gradient loss.

    Expected: ``model(**inputs)`` without ``labels=``.
    Actual:   ``model(**inputs, labels=inputs["input_ids"])``.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "grpo.py:253 passes labels= to model forward but never uses "
            "outputs.loss — redundant loss computation"
        ),
    )
    def test_no_redundant_labels_in_forward_pass(self) -> None:
        """_train_simplified should not pass labels= to the model call."""
        from fit.training.grpo import GRPOTrainer

        source = inspect.getsource(GRPOTrainer._train_simplified)

        # Find the model forward-pass line
        for line in source.splitlines():
            if "model(**inputs" in line:
                assert "labels=" not in line, (
                    "Bug confirmed: _train_simplified passes redundant "
                    "labels= to model(**inputs) — loss is never used"
                )
                return

        pytest.fail("Could not locate model(**inputs line in source")


# ---------------------------------------------------------------------------
# Bug 3: train_advisor manually globs instead of delegating to load_batch
# ---------------------------------------------------------------------------


class TestTrainAdvisorManualGlob:
    """``train_advisor.main()`` manually globs ``*.jsonl``, ``*.json``, and
    checks for YAML separately (lines 106-117) instead of delegating to
    ``ingester.load_batch([traces_path])``.

    This manual globbing misses ``*.ndjson`` files, does not recurse into
    subdirectories, and duplicates logic already present in
    ``TraceIngester.load_batch``.

    Expected: directory branch delegates to ``load_batch``.
    Actual:   manual ``traces_path.glob("*.jsonl")`` etc.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "train_advisor.py:106-117 manually globs instead of "
            "delegating to load_batch — misses .ndjson, no recursion"
        ),
    )
    def test_no_manual_glob_in_directory_branch(self) -> None:
        """Directory handling should delegate to load_batch, not glob."""
        from examples.train_advisor import main

        source = inspect.getsource(main)

        # The else branch (directory handling) should not contain manual
        # globbing — it should call load_batch instead.
        assert 'traces_path.glob("*.jsonl")' not in source, (
            "Bug confirmed: train_advisor manually globs *.jsonl "
            "instead of delegating to ingester.load_batch"
        )
