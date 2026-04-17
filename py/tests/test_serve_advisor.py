"""Tests for examples.serve_advisor — HTTP advisor service."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Regression: empty YAML config handled
# ---------------------------------------------------------------------------


class TestFileAdvisorEmptyYaml:
    """_load_config must normalize empty YAML to {}, not return None."""

    def test_empty_yaml_returns_dict(self, tmp_path: Path) -> None:
        """_load_config must normalize yaml.safe_load None to {}."""
        from examples.serve_advisor import FileAdvisor

        model_dir = tmp_path / "advisor"
        model_dir.mkdir()
        (model_dir / "config.yaml").write_text(
            "", encoding="utf-8"
        )

        advisor = FileAdvisor(str(model_dir))
        assert isinstance(advisor._domain, str)
        assert advisor._domain == "general"

    def test_empty_yaml_in_file_advisor(
        self, tmp_path: Path
    ) -> None:
        """FileAdvisor must handle empty YAML config without
        crashing."""
        from examples.serve_advisor import FileAdvisor

        model_dir = tmp_path / "advisor"
        model_dir.mkdir()
        (model_dir / "config.yaml").write_text(
            "", encoding="utf-8"
        )

        try:
            advisor = FileAdvisor(str(model_dir))
            assert advisor._domain is not None
        except AttributeError as exc:
            pytest.fail(
                f"Empty YAML config causes {exc}. "
                "normalize yaml.safe_load result to {} when None."
            )


# ---------------------------------------------------------------------------
# Regression: JSON array config normalized
# ---------------------------------------------------------------------------


class TestFileAdvisorJsonArrayConfig:
    """JSON config containing a non-dict type (e.g. ``[1,2,3]``) must
    be normalised to ``{}`` (domain == "general") or raise ValueError,
    not crash with AttributeError when calling ``.get()``.
    """

    def test_json_array_config_does_not_crash(
        self, tmp_path: Path
    ) -> None:
        """advisor.json with ``[1,2,3]`` must not raise
        AttributeError."""
        from examples.serve_advisor import FileAdvisor

        model_dir = tmp_path / "advisor"
        model_dir.mkdir()
        (model_dir / "advisor.json").write_text(
            json.dumps([1, 2, 3]), encoding="utf-8"
        )

        advisor = FileAdvisor(str(model_dir))
        assert advisor._domain == "general"


# ---------------------------------------------------------------------------
# Regression: malformed YAML raises ValueError
# ---------------------------------------------------------------------------


class TestFileAdvisorMalformedYaml:
    """``FileAdvisor._load_config`` must catch ``yaml.YAMLError``
    on malformed config and either raise ValueError with path
    context or fall back gracefully.
    """

    def test_malformed_yaml_config_raises_value_error_or_falls_back(
        self, tmp_path: Path
    ) -> None:
        """Malformed config.yaml should raise ValueError or fall
        back."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        bad_config = model_dir / "config.yaml"
        bad_config.write_text(
            ":\n  - [invalid", encoding="utf-8"
        )

        from examples.serve_advisor import FileAdvisor

        try:
            advisor = FileAdvisor(model_dir)
            assert advisor is not None
        except ValueError as exc:
            assert (
                str(model_dir) in str(exc)
                or "config.yaml" in str(exc)
            ), "ValueError raised but missing path context"


# ---------------------------------------------------------------------------
# Regression: malformed JSON raises ValueError
# ---------------------------------------------------------------------------


class TestFileAdvisorMalformedJson:
    """``FileAdvisor._load_config`` wraps YAML parse errors in a
    ``ValueError`` with the file path. The JSON path must do the
    same instead of letting raw ``json.JSONDecodeError`` escape.
    """

    def test_invalid_json_config_raises_valueerror_with_path(
        self, tmp_path: Path
    ) -> None:
        """Invalid ``advisor.json`` must raise ``ValueError``
        mentioning the file path."""
        examples_dir = str(
            Path(__file__).resolve().parent.parent / "examples"
        )
        if examples_dir not in sys.path:
            sys.path.insert(0, examples_dir)

        from serve_advisor import FileAdvisor

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        bad_json = model_dir / "advisor.json"
        bad_json.write_text("{invalid json", encoding="utf-8")

        with pytest.raises(
            ValueError, match=re.escape(str(bad_json))
        ):
            FileAdvisor(model_dir)


# -------------------------------------------------------------------
# Regression: non-dict JSON body crashes handler (PR #55)
# -------------------------------------------------------------------


class TestServeAdvisorNonDictBodyRegression:
    """``_read_body()`` returns whatever ``json.loads`` produces.

    A valid JSON array (``[1,2,3]``), string, or number passes
    through to ``generate_advice(ctx)`` which calls ``ctx.get(...)``
    — crashes with ``AttributeError`` because lists/ints have no
    ``.get()``.  The handler must validate ``ctx`` is a dict before
    proceeding.
    """

    def test_non_dict_json_body_returns_400(self) -> None:
        """do_POST must guard against non-dict JSON bodies.

        Inspect the source of do_POST for an ``isinstance(ctx, dict)``
        check (or equivalent) between ``_read_body()`` and
        ``generate_advice(ctx)``.  Currently missing — this test
        fails until the validation is added.
        """
        import inspect

        from examples.serve_advisor import _build_app

        source = inspect.getsource(_build_app)

        # Locate the do_POST method body
        post_start = source.find("def do_POST")
        assert post_start != -1, "do_POST not found in _build_app"
        post_body = source[post_start:]

        # Between _read_body() and generate_advice() there must be
        # a dict type check so non-dict payloads get a 400.
        read_idx = post_body.find("_read_body()")
        advice_idx = post_body.find("generate_advice(")
        assert read_idx != -1 and advice_idx != -1, (
            "Expected _read_body and generate_advice in do_POST"
        )

        between = post_body[read_idx:advice_idx]
        has_dict_check = (
            "isinstance" in between and "dict" in between
        ) or "not isinstance" in between

        assert has_dict_check, (
            "do_POST passes _read_body() result straight to "
            "generate_advice() with no dict type check — a JSON "
            "array or scalar crashes with AttributeError"
        )


# -------------------------------------------------------------------
# Regression: match=str(path) in pytest.raises is regex-unsafe
# -------------------------------------------------------------------


class TestPytestMatchPathRegexRegression:
    """``pytest.raises(match=str(path))`` treats the string as a
    regex.  Paths on Windows contain backslashes (``C:\\Users\\...``)
    which are regex escapes — the match silently fails or errors.

    All ``match=`` arguments derived from filesystem paths must use
    ``re.escape()`` so that backslashes (and other metacharacters)
    are treated as literals.
    """

    def test_match_uses_re_escape_for_paths(self) -> None:
        """Scan this test file for ``match=str(`` patterns.

        Any path passed to ``pytest.raises(match=...)`` via bare
        ``str()`` is a latent Windows bug — it must use
        ``re.escape()`` instead.  This test fails until every
        occurrence is fixed.
        """
        import ast
        import inspect

        src = inspect.getsource(
            sys.modules[__name__]
        )

        tree = ast.parse(src)
        violations: list[int] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword):
                continue
            if node.arg != "match":
                continue
            val = node.value
            # match=str(...) — bare str call, no re.escape
            if (
                isinstance(val, ast.Call)
                and isinstance(val.func, ast.Name)
                and val.func.id == "str"
            ):
                violations.append(node.lineno)

        assert not violations, (
            "pytest.raises(match=str(...)) found at line(s) "
            f"{violations}. Use re.escape() to avoid regex "
            "metacharacter issues with filesystem paths on "
            "Windows."
        )
