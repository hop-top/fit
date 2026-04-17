"""Tests for examples.serve_advisor — HTTP advisor service."""
from __future__ import annotations

import json
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

        with pytest.raises(ValueError, match=str(bad_json)):
            FileAdvisor(model_dir)
