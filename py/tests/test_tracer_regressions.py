"""Regression tests for fit.training.tracer — content filtering, error handling."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from fit.training.tracer import TraceIngester, _detect_format


SAMPLE_TRACE = {
    "id": "test-001",
    "session_id": "sess_test",
    "timestamp": "2026-04-15T12:00:00Z",
    "input": {"prompt": "What is X?", "context": {"k": "v"}},
    "advice": {
        "domain": "test-domain",
        "steering_text": "Be concise",
        "confidence": 0.8,
        "version": "1.0",
    },
    "frontier": {"model": "test-model", "output": "X is Y"},
    "reward": {"score": 0.9, "breakdown": {"accuracy": 0.9}},
    "metadata": {},
}


class TestLoadYamlDirContentFilter:
    """load_yaml_dir filters by content (input/frontier keys), not filename."""

    def test_non_step_pattern_file_loaded(self, tmp_path: Path) -> None:
        """Any YAML with trace keys is loaded regardless of filename."""
        session_dir = tmp_path / "sess_docstring"
        session_dir.mkdir()
        (session_dir / "other.yaml").write_text(
            yaml.safe_dump(SAMPLE_TRACE, default_flow_style=False),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 1

    def test_step_pattern_file_loaded(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_happy"
        session_dir.mkdir()
        (session_dir / "step-001.yaml").write_text(
            yaml.safe_dump(SAMPLE_TRACE, default_flow_style=False),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 1
        assert ingester.to_trace_records()[0].id == "test-001"


class TestLoadYamlDirFiltersNonTrace:
    """load_yaml_dir must filter YAML by trace content, not just extension."""

    def test_mixed_dir_only_loads_traces(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_test"
        session_dir.mkdir()
        (session_dir / "step-001.yaml").write_text(
            yaml.safe_dump(SAMPLE_TRACE, default_flow_style=False),
            encoding="utf-8",
        )
        (session_dir / "config.yaml").write_text(
            yaml.safe_dump({"app": "fit", "version": "1.0"}),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 1

    def test_dir_with_only_non_trace_yields_zero(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_empty"
        session_dir.mkdir()
        (session_dir / "schema.yml").write_text(
            yaml.safe_dump({"type": "object", "properties": {}}),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 0

    def test_non_trace_not_producing_empty_record(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_cfg"
        session_dir.mkdir()
        (session_dir / "config.yaml").write_text(
            yaml.safe_dump({"app": "fit"}), encoding="utf-8"
        )
        records = TraceIngester().load_yaml_dir(tmp_path).to_trace_records()
        assert len(records) == 0


class TestLoadYamlDirYmlExtension:
    """load_yaml_dir must ingest .yml files, not just .yaml."""

    def test_yml_files_ingested(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_yml"
        session_dir.mkdir()
        trace = {**SAMPLE_TRACE, "id": "yml-test-001"}
        (session_dir / "step-001.yml").write_text(
            yaml.safe_dump(trace), encoding="utf-8"
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() >= 1

    def test_yaml_and_yml_both_ingested(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_mixed"
        session_dir.mkdir()
        trace_a = {**SAMPLE_TRACE, "id": "yaml-001"}
        trace_b = {**SAMPLE_TRACE, "id": "yml-001"}
        (session_dir / "step-001.yaml").write_text(
            yaml.safe_dump(trace_a), encoding="utf-8"
        )
        (session_dir / "step-002.yml").write_text(
            yaml.safe_dump(trace_b), encoding="utf-8"
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 2


class TestLoadYamlDirMalformedError:
    """Malformed YAML must raise ValueError with file path context."""

    def test_malformed_yaml_raises_value_error_with_path(
        self, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "step-001.yaml"
        bad_file.write_text(":\n  - [invalid", encoding="utf-8")
        with pytest.raises(ValueError, match=r"step-001\.yaml"):
            TraceIngester().load_yaml_dir(tmp_path)


class TestLoadBatchNonTraceFilter:
    """load_batch must apply input/frontier filter for YAML files."""

    def test_single_yaml_non_trace_dict_skipped(self, tmp_path: Path) -> None:
        yml = tmp_path / "config.yaml"
        yml.write_text(yaml.safe_dump({"app": "fit"}), encoding="utf-8")
        ingester = TraceIngester().load_batch([yml])
        assert ingester.count() == 0

    def test_single_yaml_list_filters_non_trace(self, tmp_path: Path) -> None:
        yml = tmp_path / "mixed.yaml"
        yml.write_text(
            yaml.safe_dump(
                [{"app": "fit"}, {"input": {"prompt": "x"}}],
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_batch([yml])
        assert ingester.count() == 1

    def test_load_batch_and_load_yaml_dir_consistent(self, tmp_path: Path) -> None:
        yml = tmp_path / "nondir"
        yml.mkdir()
        cfg = yml / "config.yaml"
        cfg.write_text(yaml.safe_dump({"app": "fit", "version": "1.0"}), encoding="utf-8")
        dir_count = TraceIngester().load_yaml_dir(yml).count()
        batch_count = TraceIngester().load_batch([cfg]).count()
        assert dir_count == 0
        assert batch_count == 0


class TestParseRawNonDictFields:
    """_parse_raw must handle non-dict field values defensively."""

    def test_null_input_returns_defaults(self) -> None:
        from fit.training.tracer import _parse_raw
        rec = _parse_raw({"input": None, "frontier": {}})
        assert rec.prompt == ""
        assert rec.context == {}

    def test_string_advice_returns_defaults(self) -> None:
        from fit.training.tracer import _parse_raw
        rec = _parse_raw({"input": {}, "advice": "not a dict"})
        assert rec.advice_text == ""
        assert rec.advice_domain == "unknown"

    def test_null_frontier_and_reward_returns_defaults(self) -> None:
        from fit.training.tracer import _parse_raw
        rec = _parse_raw({"input": {}, "frontier": None, "reward": None})
        assert rec.frontier_output == ""
        assert rec.reward_score is None
        assert rec.reward_breakdown == {}


class TestConfidenceTypeCoercion:
    """_parse_raw must handle non-float confidence values."""

    def test_null_confidence_returns_zero(self) -> None:
        from fit.training.tracer import _parse_raw
        rec = _parse_raw({"input": {}, "advice": {"confidence": None}})
        assert rec.advice_confidence == 0.0

    def test_string_confidence_returns_zero(self) -> None:
        from fit.training.tracer import _parse_raw
        rec = _parse_raw({"input": {}, "advice": {"confidence": "high"}})
        assert rec.advice_confidence == 0.0

    def test_numeric_confidence_works(self) -> None:
        from fit.training.tracer import _parse_raw
        rec = _parse_raw({"input": {}, "advice": {"confidence": 0.8}})
        assert rec.advice_confidence == 0.8


class TestJsonlNonDictLineValidation:
    """load_jsonl must validate JSON-decoded values are dicts."""

    def test_array_line_raises(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text("[]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object"):
            TraceIngester().load_jsonl(jsonl)

    def test_string_line_raises(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text('"hello"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object"):
            TraceIngester().load_jsonl(jsonl)


class TestLoadBatchJsonNonDictValidation:
    """load_batch JSON path must validate list items are dicts."""

    def test_non_dict_items_raises(self, tmp_path: Path) -> None:
        j = tmp_path / "traces.json"
        j.write_text(json.dumps([{"input": {}}, 42]), encoding="utf-8")
        with pytest.raises(ValueError, match="dict"):
            TraceIngester().load_batch([j])

    def test_all_dicts_works(self, tmp_path: Path) -> None:
        j = tmp_path / "traces.json"
        j.write_text(
            json.dumps([{"input": {}}, {"frontier": {}}]), encoding="utf-8"
        )
        ingester = TraceIngester().load_batch([j])
        assert ingester.count() == 2

    def test_error_message_contains_dict(self, tmp_path: Path) -> None:
        j = tmp_path / "traces.json"
        j.write_text(json.dumps([{"input": {}}, 42]), encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            TraceIngester().load_batch([j])
        msg = str(exc_info.value).lower()
        assert "dict" in msg or "json object" in msg


class TestLoadBatchJsonFilterNonTrace:
    """load_batch JSON branch must filter non-trace dicts by input/frontier."""

    def test_array_filters_non_trace(self, tmp_path: Path) -> None:
        data = [
            {"app": "fit"},
            {"input": {"prompt": "x"}, "frontier": {"model": "m", "output": "ok"}},
        ]
        p = tmp_path / "mixed.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        ingester = TraceIngester().load_batch([p])
        assert ingester.count() == 1

    def test_single_non_trace_yields_zero(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"app": "fit", "version": "1.0"}), encoding="utf-8")
        ingester = TraceIngester().load_batch([p])
        assert ingester.count() == 0


class TestLoadBatchJsonBareValueRaises:
    """Bare string/number JSON in load_batch must raise ValueError."""

    def test_bare_string_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bare.json"
        p.write_text('"hello"', encoding="utf-8")
        with pytest.raises(ValueError):
            TraceIngester().load_batch([p])

    def test_bare_number_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bare.json"
        p.write_text("42", encoding="utf-8")
        with pytest.raises(ValueError):
            TraceIngester().load_batch([p])


class TestLoadBatchSingleYamlError:
    """Malformed single YAML in load_batch must raise ValueError with path."""

    def test_malformed_raises_with_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(":\n  - [invalid", encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.yaml"):
            TraceIngester().load_batch([bad])


class TestLoadBatchJsonDecodeError:
    """Malformed JSON in load_batch must raise ValueError with path."""

    def test_malformed_raises_with_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json{", encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.json"):
            TraceIngester().load_batch([bad])


class TestLoadBatchParamNotFormat:
    """load_batch parameter must be named fmt, not format (shadow)."""

    def test_param_name(self) -> None:
        import inspect
        sig = inspect.signature(TraceIngester.load_batch)
        params = list(sig.parameters.keys())
        assert "format" not in params, "load_batch shadows built-in format()"
        assert "fmt" in params


class TestLoadBatchJsonlDirectory:
    """JSONL branch in load_batch must handle directories."""

    def test_directory_loads_records(self, tmp_path: Path) -> None:
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        record = {"input": {"prompt": "hello"}, "advice_domain": "test"}
        (trace_dir / "traces.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        ingester = TraceIngester()
        ingester.load_batch([trace_dir])
        assert ingester.count() > 0


class TestDetectFormatNdjsonDirectory:
    """_detect_format must check *.ndjson in directories."""

    def test_ndjson_dir_detected_as_jsonl(self, tmp_path: Path) -> None:
        ndjson_dir = tmp_path / "ndjson"
        ndjson_dir.mkdir()
        (ndjson_dir / "traces.ndjson").write_text(
            json.dumps({"input": {"prompt": "test"}}) + "\n", encoding="utf-8"
        )
        assert _detect_format(ndjson_dir) == "jsonl"


class TestDetectFormatNestedJsonl:
    """_detect_format must use rglob for nested JSONL files."""

    def test_nested_jsonl_detected(self, tmp_path: Path) -> None:
        subdir = tmp_path / "session_01"
        subdir.mkdir()
        (subdir / "traces.jsonl").write_text(
            '{"prompt":"hi"}\n', encoding="utf-8"
        )
        assert _detect_format(tmp_path) == "jsonl"


class TestDetectFormatMixedDirectory:
    """Directories with mixed formats must load all traces via load_batch."""

    def test_mixed_yaml_and_jsonl(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess"
        session_dir.mkdir()
        yaml_trace = {**SAMPLE_TRACE, "id": "yaml-001"}
        (session_dir / "step-001.yaml").write_text(
            yaml.safe_dump(yaml_trace), encoding="utf-8"
        )
        jsonl_trace = {**SAMPLE_TRACE, "id": "jsonl-001"}
        (tmp_path / "traces.jsonl").write_text(
            json.dumps(jsonl_trace), encoding="utf-8"
        )
        ingester = TraceIngester().load_batch([tmp_path])
        assert ingester.count() == 2


class TestSqliteUriCrossPlatformRegression:
    """Regression: load_sqlite builds SQLite URI via f-string interpolation.

    On Windows, Path objects produce backslashes and drive letters
    (e.g. ``C:\\data\\traces.db``), making ``f"file:{path}?mode=ro"``
    an invalid URI. The fix is to use ``path.as_uri()`` (or
    ``path.resolve().as_uri()``) which emits a proper
    ``file:///...`` URI on every platform.
    """

    def test_load_sqlite_does_not_use_raw_fstring_uri(self) -> None:
        """Source must not contain ``f"file:{path}`` pattern."""
        import inspect

        source = inspect.getsource(TraceIngester.load_sqlite)

        assert 'f"file:{path}' not in source, (
            "load_sqlite uses raw f-string URI construction "
            '(f"file:{path}?mode=ro") which breaks on Windows '
            "paths with backslashes/drive letters. Use "
            "path.resolve().as_uri() instead."
        )
