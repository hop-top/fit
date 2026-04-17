"""Regression tests for PR #41 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Bug 1: to_gguf() returns path without creating artifact
# File: py/src/fit/training/export.py, lines 96-102
# ---------------------------------------------------------------------------


class TestToGgufReturnsPhantomPath:
    """``ModelExporter.to_gguf()`` logs a warning when ``gguf`` is
    importable but conversion is not actually implemented. It then
    returns the output path even though no file was created there.
    Callers receive a path that does not exist on disk, making the
    method silently report success with no artifact.
    """

    def test_returned_path_exists_or_raises(
        self, tmp_path: Path
    ) -> None:
        """If ``to_gguf`` returns a path, that path must exist as a
        file. Otherwise the method must raise ``NotImplementedError``
        to signal that conversion is not available."""
        from fit.training.export import ModelExporter

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        exporter = ModelExporter(str(model_dir))

        # Inject a fake ``gguf`` module so the import succeeds
        fake_gguf = types.ModuleType("gguf")
        fake_gguf.GGUFWriter = type("GGUFWriter", (), {})  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"gguf": fake_gguf}):
            result = None
            raised = False
            try:
                result = exporter.to_gguf(
                    str(tmp_path / "model.gguf")
                )
            except NotImplementedError:
                raised = True

        if raised:
            return  # acceptable: method signals missing impl

        assert result is not None and result.exists(), (
            "Bug confirmed: to_gguf returned a path that does not "
            "exist on disk — no GGUF artifact was created"
        )
