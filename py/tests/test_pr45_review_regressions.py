"""Regression tests for PR #45 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fit.training.tracer import TraceIngester


# ---------------------------------------------------------------------------
# Bug 1: load_batch single-file YAML path doesn't catch yaml.YAMLError
# ---------------------------------------------------------------------------


class TestLoadBatchSingleYamlMissingYAMLErrorCatch:
    """load_batch reads single YAML files via yaml.safe_load without catching
    yaml.YAMLError.

    ``load_yaml_dir()`` already wraps the error and includes the file path,
    but the inline single-file branch at tracer.py:213-214 does not.  A
    malformed YAML file raises a raw ``yaml.YAMLError`` that escapes without
    any file-path context.

    Expected: ``ValueError`` whose message includes the file path.
    Actual:   raw ``yaml.YAMLError``.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "load_batch single-file YAML branch calls yaml.safe_load "
            "without catching yaml.YAMLError — inconsistent with "
            "load_yaml_dir"
        ),
    )
    def test_malformed_single_yaml_raises_value_error_with_path(
        self, tmp_path: Path
    ) -> None:
        """Malformed single YAML should raise ValueError containing the path."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(":\n  - [invalid", encoding="utf-8")

        ingester = TraceIngester()

        with pytest.raises(ValueError, match=r"bad\.yaml") as exc_info:
            ingester.load_batch([bad_file])

        assert exc_info.type is ValueError, (
            "Bug confirmed: load_batch raises raw yaml.YAMLError "
            "instead of ValueError with file-path context"
        )


# ---------------------------------------------------------------------------
# Bugs 2-3: Unused `import tempfile` in test files
# ---------------------------------------------------------------------------


class TestUnusedTempfileImport:
    """Several test files import ``tempfile`` but never reference it.

    The module is imported at the top level yet no ``tempfile.*`` attribute
    access appears anywhere in the AST.  This is dead code that should be
    removed.

    Detection: ``ast.parse`` + ``ast.walk`` checking for ``Import`` of
    ``tempfile`` and absence of any ``Attribute`` node with
    ``value.id == 'tempfile'``.
    """

    @staticmethod
    def _has_unused_tempfile_import(source: str) -> bool:
        """Return True if *source* imports tempfile but never uses it."""
        tree = ast.parse(source)

        imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "tempfile":
                        imported = True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "tempfile":
                    imported = True

        if not imported:
            return False

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "tempfile"
            ):
                return False

        return True

    @pytest.mark.xfail(
        strict=True,
        reason="tempfile is imported but never used in the target file",
    )
    @pytest.mark.parametrize(
        "rel_path",
        [
            "test_pr43_tracer_regression.py",
            "test_pr44_review_regressions.py",
        ],
        ids=["pr43", "pr44"],
    )
    def test_tempfile_not_unused(self, rel_path: str) -> None:
        """tempfile import should be used or removed."""
        test_dir = Path(__file__).parent
        source = (test_dir / rel_path).read_text(encoding="utf-8")

        assert not self._has_unused_tempfile_import(source), (
            f"Bug confirmed: {rel_path} imports tempfile but never "
            "references tempfile.* — dead import"
        )
