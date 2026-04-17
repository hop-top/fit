"""Regression tests for PR #40 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import ast
import inspect
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Bug 1: Line too long in reward_fn.py (Advice construction)
# File: py/src/fit/training/reward_fn.py, line 88
# ---------------------------------------------------------------------------


class TestLLMJudgeAdviceLineLengthRegression:
    """``LLMJudgeReward.__call__`` builds an ``Advice(...)`` on a single
    line that exceeds the 100-character line-length limit. The line
    should be wrapped to comply with project style.
    """

    def test_advice_line_within_limit(self) -> None:
        """Every source line in ``LLMJudgeReward.__call__`` must be
        at most 100 characters wide."""
        from fit.training.reward_fn import LLMJudgeReward

        import pathlib

        src_file = pathlib.Path(
            inspect.getfile(LLMJudgeReward)
        )
        source = src_file.read_text()
        long_lines = [
            (i + 1, line)
            for i, line in enumerate(source.splitlines())
            if "Advice(" in line and len(line.rstrip()) > 99
        ]
        assert not long_lines, (
            "Bug confirmed: Advice construction line exceeds "
            f"100-char limit: {long_lines}"
        )


# ---------------------------------------------------------------------------
# Bug 2: LLMJudgeReward only catches ImportError, not ValueError
# File: py/src/fit/training/reward_fn.py, lines 92-94
# ---------------------------------------------------------------------------


class TestLLMJudgeValueErrorRegression:
    """``LLMJudgeReward.__call__`` wraps the adapter import in a
    ``try/except ImportError`` block. When ``anthropic`` is installed
    but ``ANTHROPIC_API_KEY`` is unset, ``AnthropicAdapter.__init__``
    raises ``ValueError``. The except clause does not catch it, so
    the call crashes instead of returning the 0.5 neutral fallback.
    """

    def test_returns_fallback_on_missing_api_key(self) -> None:
        """Calling ``LLMJudgeReward()`` when the adapter raises
        ``ValueError`` (missing API key) must return 0.5, not raise."""
        from fit.training.reward_fn import LLMJudgeReward

        with patch(
            "fit.training.reward_fn.AnthropicAdapter",
            side_effect=ValueError("no api key"),
            create=True,
        ), patch(
            "fit.adapters.anthropic.AnthropicAdapter",
            side_effect=ValueError("no api key"),
        ):
            reward = LLMJudgeReward()
            score = reward("ctx", "adv", "out")
        assert score == 0.5, (
            "Bug confirmed: LLMJudgeReward raises ValueError instead "
            "of returning 0.5 neutral fallback"
        )


# ---------------------------------------------------------------------------
# Bugs 3-5: Unused ``import pytest`` in prior regression test files
# Files: test_pr37, test_pr38, test_pr39_review_regressions.py
# ---------------------------------------------------------------------------


class TestUnusedPytestImportRegression:
    """Several prior PR regression test files import ``pytest`` at
    module level but never reference it (``pytest.mark``,
    ``pytest.raises``, etc.). This triggers F401 (unused import).
    """

    _FILES = (
        "tests.test_pr37_review_regressions",
        "tests.test_pr38_review_regressions",
        "tests.test_pr39_review_regressions",
    )

    @pytest.mark.parametrize("module_name", _FILES)
    def test_pytest_import_is_used(self, module_name: str) -> None:
        """If a module imports ``pytest``, at least one ``pytest.``
        attribute access must appear in the AST (excluding the import
        statement itself)."""
        import importlib

        mod = importlib.import_module(module_name)
        source = inspect.getsource(mod)
        tree = ast.parse(source)

        has_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pytest":
                        has_import = True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "pytest":
                    has_import = True

        if not has_import:
            return  # no import to check

        # Count ``pytest.`` attribute accesses
        uses = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "pytest"
            ):
                uses += 1

        assert uses > 0, (
            f"Bug confirmed: {module_name}.py imports pytest "
            "but never references pytest.* (F401 unused import)"
        )
