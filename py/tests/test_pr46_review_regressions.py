"""Regression tests for PR #46 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fit.training.tracer import TraceIngester, _detect_format


# ---------------------------------------------------------------------------
# Bug 1: JSON branch in load_batch doesn't catch json.JSONDecodeError
# ---------------------------------------------------------------------------


class TestLoadBatchJsonMissingJSONDecodeErrorCatch:
    """load_batch JSON branch calls ``json.load(f)`` without catching
    ``json.JSONDecodeError``.

    The YAML branch wraps ``yaml.YAMLError`` into ``ValueError`` with the
    file path, and the JSONL branch wraps ``json.JSONDecodeError`` similarly,
    but the JSON branch at tracer.py:232-233 does not.  A malformed ``.json``
    file raises a raw ``json.JSONDecodeError`` that escapes without any
    file-path context.

    Expected: ``ValueError`` whose message includes the file path.
    Actual:   raw ``json.JSONDecodeError``.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "load_batch JSON branch does not catch json.JSONDecodeError; "
            "raw exception escapes without file-path context"
        ),
    )
    def test_malformed_json_raises_value_error_with_path(
        self, tmp_path: Path
    ) -> None:
        """Malformed .json file should raise ValueError containing the path."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json{", encoding="utf-8")

        ingester = TraceIngester()

        with pytest.raises(ValueError, match=r"bad\.json") as exc_info:
            ingester.load_batch([bad_file])

        assert exc_info.type is ValueError, (
            "Bug confirmed: load_batch raises raw json.JSONDecodeError "
            "instead of ValueError with file-path context"
        )


# ---------------------------------------------------------------------------
# Bug 2: serve_advisor yaml.safe_load without YAMLError catch
# ---------------------------------------------------------------------------


class TestServeAdvisorMissingYAMLErrorCatch:
    """``FileAdvisor._load_config`` calls ``yaml.safe_load()`` on a YAML
    config file without catching ``yaml.YAMLError``.

    When the config is malformed YAML, a raw ``yaml.YAMLError`` escapes
    the constructor instead of being wrapped into a ``ValueError`` with
    path context or triggering a graceful fallback.

    Expected: ``ValueError`` with path context, or graceful fallback to
              empty config.
    Actual:   raw ``yaml.YAMLError``.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "FileAdvisor._load_config does not catch yaml.YAMLError; "
            "raw exception escapes without file-path context"
        ),
    )
    def test_malformed_yaml_config_raises_value_error_or_falls_back(
        self, tmp_path: Path
    ) -> None:
        """Malformed config.yaml should raise ValueError or fall back."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        bad_config = model_dir / "config.yaml"
        bad_config.write_text(":\n  - [invalid", encoding="utf-8")

        from examples.serve_advisor import FileAdvisor

        # Should either raise ValueError with path context or fall back
        # gracefully.  Currently raises raw yaml.YAMLError.
        try:
            advisor = FileAdvisor(model_dir)
            # If we get here, a graceful fallback occurred -- acceptable.
            assert advisor is not None
        except ValueError as exc:
            assert str(model_dir) in str(exc) or "config.yaml" in str(exc), (
                "Bug confirmed: ValueError raised but missing path context"
            )


# ---------------------------------------------------------------------------
# Bug 3: JSONL branch calls load_jsonl on directories
# ---------------------------------------------------------------------------


class TestLoadBatchJsonlBranchOnDirectory:
    """``load_batch()`` JSONL branch calls ``self.load_jsonl(p)`` without
    checking whether ``p`` is a directory.

    If ``_detect_format(dir)`` returns ``"jsonl"`` (because the directory
    contains ``*.jsonl`` files but no ``*.yaml``/``*.yml``), ``load_jsonl``
    tries to open a directory as a file, raising ``IsADirectoryError``.

    Expected: directory is walked; contained ``.jsonl`` files are loaded.
    Actual:   ``IsADirectoryError``.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "load_batch JSONL branch does not check if path is a directory; "
            "passes directory directly to load_jsonl which crashes"
        ),
    )
    def test_jsonl_directory_loads_records_without_error(
        self, tmp_path: Path
    ) -> None:
        """Directory with .jsonl files should be handled, not crash."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        record = {
            "input": {"prompt": "hello"},
            "advice": {"text": "world"},
            "advice_domain": "test",
        }
        jsonl_file = trace_dir / "traces.jsonl"
        jsonl_file.write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

        ingester = TraceIngester()

        # Should not raise IsADirectoryError.
        ingester.load_batch([trace_dir])

        assert ingester.count() > 0, (
            "Bug confirmed: load_batch JSONL branch raises "
            "IsADirectoryError on directories instead of loading "
            "contained .jsonl files"
        )


# ---------------------------------------------------------------------------
# Bug 4: _detect_format doesn't check *.ndjson in directories
# ---------------------------------------------------------------------------


class TestDetectFormatMissingNdjsonGlob:
    """``_detect_format()`` checks ``path.glob("*.jsonl")`` for directories
    but not ``*.ndjson``, even though single-file detection treats ``.ndjson``
    as ``"jsonl"``.

    A directory containing only ``*.ndjson`` files (no ``.jsonl``, no
    ``.yaml``/``.yml``) falls through to the default ``"yaml"`` return.

    Expected: ``"jsonl"``.
    Actual:   ``"yaml"`` (default).
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "_detect_format does not glob *.ndjson in directories; "
            "directories with only .ndjson files default to 'yaml'"
        ),
    )
    def test_ndjson_directory_detected_as_jsonl(
        self, tmp_path: Path
    ) -> None:
        """Directory with only .ndjson files should be detected as jsonl."""
        ndjson_dir = tmp_path / "ndjson_traces"
        ndjson_dir.mkdir()

        record = {"input": {"prompt": "test"}, "advice_domain": "test"}
        ndjson_file = ndjson_dir / "traces.ndjson"
        ndjson_file.write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

        result = _detect_format(ndjson_dir)

        assert result == "jsonl", (
            "Bug confirmed: _detect_format returns 'yaml' for directories "
            "containing only .ndjson files instead of 'jsonl'"
        )
