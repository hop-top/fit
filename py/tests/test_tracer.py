"""Tests for fit.training.tracer — trace ingestion from any fit port."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from fit.training.tracer import TraceIngester, TraceRecord, _detect_format


FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "spec" / "fixtures"

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


class TestTraceRecord:
    def test_from_raw_trace(self) -> None:
        from fit.training.tracer import _parse_raw

        rec = _parse_raw(SAMPLE_TRACE)
        assert rec.id == "test-001"
        assert rec.session_id == "sess_test"
        assert rec.prompt == "What is X?"
        assert rec.advice_text == "Be concise"
        assert rec.advice_domain == "test-domain"
        assert rec.frontier_output == "X is Y"
        assert rec.reward_score == 0.9
        assert rec.reward_breakdown == {"accuracy": 0.9}

    def test_missing_fields_defaults(self) -> None:
        from fit.training.tracer import _parse_raw

        rec = _parse_raw({})
        assert rec.id == ""
        assert rec.prompt == ""
        assert rec.advice_domain == "unknown"
        assert rec.reward_score is None
        assert rec.reward_breakdown == {}


class TestLoadJsonl:
    def test_load_jsonl_file(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "traces.jsonl"
        lines = [json.dumps(SAMPLE_TRACE), json.dumps({**SAMPLE_TRACE, "id": "test-002"})]
        jsonl.write_text("\n".join(lines), encoding="utf-8")

        ingester = TraceIngester().load_jsonl(jsonl)
        assert ingester.count() == 2

    def test_load_jsonl_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            TraceIngester().load_jsonl("/nonexistent/file.jsonl")

    def test_load_jsonl_invalid_json(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            TraceIngester().load_jsonl(jsonl)

    def test_load_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "traces.jsonl"
        jsonl.write_text(f"\n{json.dumps(SAMPLE_TRACE)}\n\n", encoding="utf-8")
        assert TraceIngester().load_jsonl(jsonl).count() == 1


class TestLoadYamlDir:
    def test_load_yaml_cassettes(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_test"
        session_dir.mkdir()
        (session_dir / "step-001.yaml").write_text(
            yaml.safe_dump(SAMPLE_TRACE, default_flow_style=False), encoding="utf-8"
        )

        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 1
        assert ingester.to_trace_records()[0].id == "test-001"

    def test_load_yaml_dir_missing(self) -> None:
        with pytest.raises(NotADirectoryError):
            TraceIngester().load_yaml_dir("/nonexistent/dir")

    def test_load_from_fixture_dir(self) -> None:
        ingester = TraceIngester().load_yaml_dir(FIXTURE_DIR)
        records = ingester.to_trace_records()
        assert len(records) >= 1
        assert all(isinstance(r, TraceRecord) for r in records)

    def test_load_yml_cassettes(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_yml"
        session_dir.mkdir()
        (session_dir / "step-001.yml").write_text(
            yaml.safe_dump(SAMPLE_TRACE, default_flow_style=False),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 1
        assert ingester.to_trace_records()[0].id == "test-001"


class TestLoadSqlite:
    def test_load_json_blob(self, tmp_path: Path) -> None:
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute("INSERT INTO traces VALUES (?)", (json.dumps(SAMPLE_TRACE),))
        conn.commit()
        conn.close()

        ingester = TraceIngester().load_sqlite(db_path)
        assert ingester.count() == 1
        assert ingester.to_trace_records()[0].id == "test-001"

    def test_load_sqlite_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            TraceIngester().load_sqlite("/nonexistent/file.db")


class TestLoadYamlDirFiltersNonTraceRegressions:
    """Regression: load_yaml_dir must filter YAML by trace content, not just extension.

    PR #29 review item 2 — rglob('*.y*ml') picks up ALL yaml files,
    including configs/schemas. YAML mappings that do not look like traces
    (for example, those missing trace fields such as ``input``/``frontier``)
    must be skipped so they do not produce mostly-empty TraceRecords.
    """

    def test_mixed_dir_only_loads_traces(self, tmp_path: Path) -> None:
        """Dir with step-001.yaml + config.yaml should load ONLY the trace."""
        session_dir = tmp_path / "sess_test"
        session_dir.mkdir()

        # valid trace
        (session_dir / "step-001.yaml").write_text(
            yaml.safe_dump(SAMPLE_TRACE, default_flow_style=False),
            encoding="utf-8",
        )
        # non-trace config
        (session_dir / "config.yaml").write_text(
            yaml.safe_dump({"app": "fit", "version": "1.0"}),
            encoding="utf-8",
        )

        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 1, (
            f"Expected 1 (trace only), got {ingester.count()} — "
            "non-trace YAML files must be filtered"
        )

    def test_dir_with_only_non_trace_yields_zero(self, tmp_path: Path) -> None:
        """Dir with only a schema.yml should produce 0 records."""
        session_dir = tmp_path / "sess_empty"
        session_dir.mkdir()
        (session_dir / "schema.yml").write_text(
            yaml.safe_dump({"type": "object", "properties": {}}),
            encoding="utf-8",
        )

        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 0, (
            f"Expected 0 (no traces), got {ingester.count()} — "
            "non-trace YAML must be filtered"
        )

    def test_non_trace_not_producing_empty_record(self, tmp_path: Path) -> None:
        """Non-trace YAML must not produce a mostly-empty TraceRecord."""
        session_dir = tmp_path / "sess_cfg"
        session_dir.mkdir()
        (session_dir / "config.yaml").write_text(
            yaml.safe_dump({"app": "fit"}),
            encoding="utf-8",
        )

        records = TraceIngester().load_yaml_dir(tmp_path).to_trace_records()
        assert len(records) == 0


class TestLoadSqliteTableValidationRegressions:
    """Regression: load_sqlite must validate the table parameter.

    PR #29 review item 3 — table is interpolated into f-string SQL.
    Malicious or malformed table names enable SQL injection.
    These tests prove the bug (they FAIL on main).
    """

    def _make_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute(
            "INSERT INTO traces VALUES (?)",
            (json.dumps(SAMPLE_TRACE),),
        )
        conn.commit()
        conn.close()
        return db_path

    def test_injection_attempt_raises(self, tmp_path: Path) -> None:
        """Injection string should be rejected before hitting SQL.

        Currently fails because table is interpolated unsafely.
        Fix should raise ValueError; currently raises OperationalError
        or silently executes — either way, no ValueError is raised.
        """
        db_path = self._make_db(tmp_path)
        with pytest.raises(ValueError, match="table"):
            TraceIngester().load_sqlite(
                db_path, table="traces; DROP TABLE traces--"
            )

    def test_table_with_space_raises(self, tmp_path: Path) -> None:
        """Spaces in table name should be rejected before hitting SQL."""
        db_path = self._make_db(tmp_path)
        with pytest.raises(ValueError, match="table"):
            TraceIngester().load_sqlite(db_path, table="bad name")

    def test_empty_table_raises(self, tmp_path: Path) -> None:
        """Empty table name should be rejected before hitting SQL."""
        db_path = self._make_db(tmp_path)
        with pytest.raises(ValueError, match="table"):
            TraceIngester().load_sqlite(db_path, table="")

    def test_valid_table_works(self, tmp_path: Path) -> None:
        db_path = self._make_db(tmp_path)
        ingester = TraceIngester().load_sqlite(db_path, table="traces")
        assert ingester.count() == 1
        assert ingester.to_trace_records()[0].id == "test-001"


class TestLoadBatch:
    def test_auto_detect_json(self, tmp_path: Path) -> None:
        j = tmp_path / "traces.json"
        j.write_text(json.dumps([SAMPLE_TRACE]), encoding="utf-8")
        assert TraceIngester().load_batch([j]).count() == 1

    def test_auto_detect_jsonl(self, tmp_path: Path) -> None:
        j = tmp_path / "traces.jsonl"
        j.write_text(json.dumps(SAMPLE_TRACE), encoding="utf-8")
        assert TraceIngester().load_batch([j]).count() == 1

    def test_auto_detect_yaml_dir(self) -> None:
        ingester = TraceIngester().load_batch([FIXTURE_DIR])
        assert ingester.count() >= 1

    def test_skips_missing_paths(self, tmp_path: Path) -> None:
        # Should not raise, just skip
        assert TraceIngester().load_batch([tmp_path / "nonexistent"]).count() == 0


class TestFilter:
    def test_filter_by_domain(self, tmp_path: Path) -> None:
        traces = [
            {**SAMPLE_TRACE, "id": "a", "advice": {**SAMPLE_TRACE["advice"], "domain": "tax"}},
            {**SAMPLE_TRACE, "id": "b", "advice": {**SAMPLE_TRACE["advice"], "domain": "legal"}},
        ]
        tmp = tmp_path / "filter_test.jsonl"
        tmp.write_text("\n".join(json.dumps(t) for t in traces), encoding="utf-8")
        ingester = TraceIngester().load_jsonl(tmp).filter(domain="tax")
        assert ingester.count() == 1
        assert ingester.to_trace_records()[0].id == "a"

    def test_filter_by_tenant(self, tmp_path: Path) -> None:
        traces = [
            {**SAMPLE_TRACE, "id": "a", "metadata": {"tenant": "acme"}},
            {**SAMPLE_TRACE, "id": "b", "metadata": {"tenant": "globex"}},
        ]
        tmp = tmp_path / "tenant_test.jsonl"
        tmp.write_text("\n".join(json.dumps(t) for t in traces), encoding="utf-8")
        ingester = TraceIngester().load_jsonl(tmp).filter(tenant="acme")
        assert ingester.count() == 1

    def test_filter_non_mutating(self, tmp_path: Path) -> None:
        tmp = tmp_path / "nonmut.jsonl"
        tmp.write_text(json.dumps(SAMPLE_TRACE), encoding="utf-8")
        base = TraceIngester().load_jsonl(tmp)
        filtered = base.filter(domain="nonexistent")
        assert base.count() == 1
        assert filtered.count() == 0


class TestDetectFormat:
    def test_jsonl(self) -> None:
        assert _detect_format(Path("f.jsonl")) == "jsonl"
        assert _detect_format(Path("f.ndjson")) == "jsonl"

    def test_yaml(self) -> None:
        assert _detect_format(Path("f.yaml")) == "yaml"
        assert _detect_format(Path("f.yml")) == "yaml"

    def test_sqlite(self) -> None:
        assert _detect_format(Path("f.db")) == "sqlite"

    def test_json(self) -> None:
        assert _detect_format(Path("f.json")) == "json"

    def test_directory_with_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "step.yaml").touch()
        assert _detect_format(tmp_path) == "yaml"


class TestDetectFormatYml:
    """Regression: _detect_format must detect yaml format for dirs
    containing only .yml files (not just .yaml)."""

    def test_dir_with_yml_only(self, tmp_path: Path) -> None:
        (tmp_path / "step-001.yml").touch()
        assert _detect_format(tmp_path) == "yaml"

    def test_dir_with_yaml_and_yml(self, tmp_path: Path) -> None:
        (tmp_path / "step-001.yaml").touch()
        (tmp_path / "step-002.yml").touch()
        assert _detect_format(tmp_path) == "yaml"


class TestPR30YamlDirDocstringRegression:
    """PR #30 review item 4 — load_yaml_dir now correctly documents
    that it loads any YAML dict containing 'input' or 'frontier' keys.
    These tests verify the documented behavior: non-trace YAML
    (lacking both keys) is skipped, while any valid trace dict is
    loaded regardless of filename pattern.
    """

    def test_non_step_pattern_file_loaded(self, tmp_path: Path) -> None:
        """Any YAML file with valid trace content (input/frontier) is loaded.

        Filenames are irrelevant — the loader uses content-based detection.
        A file named 'other.yaml' with trace keys is correctly ingested.
        """
        session_dir = tmp_path / "sess_docstring"
        session_dir.mkdir()
        (session_dir / "other.yaml").write_text(
            yaml.safe_dump(SAMPLE_TRACE, default_flow_style=False),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 1, (
            "Expected 1 — any YAML with input/frontier keys is a trace."
        )

    def test_step_pattern_file_loaded(self, tmp_path: Path) -> None:
        """step-NNN.yaml with valid trace must always be loaded."""
        session_dir = tmp_path / "sess_happy"
        session_dir.mkdir()
        (session_dir / "step-001.yaml").write_text(
            yaml.safe_dump(SAMPLE_TRACE, default_flow_style=False),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 1
        assert ingester.to_trace_records()[0].id == "test-001"


class TestPR31LoadBatchNonTraceFilterRegression:
    """PR #31 review — load_batch() must apply input/frontier filter
    for single YAML files, consistent with load_yaml_dir()."""

    def test_single_yaml_non_trace_dict_skipped(self, tmp_path: Path) -> None:
        """Single YAML file with non-trace dict must be skipped.

        A file containing {"app": "fit"} (no input/frontier) should
        NOT be ingested — load_batch must apply the same filter as
        load_yaml_dir.
        """
        yml = tmp_path / "config.yaml"
        yml.write_text(yaml.safe_dump({"app": "fit"}), encoding="utf-8")

        ingester = TraceIngester().load_batch([yml])
        assert ingester.count() == 0, (
            f"Expected 0 (non-trace filtered), got {ingester.count()} — "
            "load_batch must skip YAML dicts without input/frontier keys"
        )

    def test_single_yaml_list_filters_non_trace(
        self, tmp_path: Path
    ) -> None:
        """YAML list of dicts: non-trace entries must be filtered.

        File contains [{"app": "fit"}, {"input": {"prompt": "x"}}].
        Only the second dict has input — should be the only record.
        """
        yml = tmp_path / "mixed.yaml"
        yml.write_text(
            yaml.safe_dump(
                [{"app": "fit"}, {"input": {"prompt": "x"}}],
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

        ingester = TraceIngester().load_batch([yml])
        assert ingester.count() == 1, (
            f"Expected 1 (only trace dict), got {ingester.count()} — "
            "load_batch must filter list items by input/frontier keys"
        )

    def test_load_batch_and_load_yaml_dir_consistent(
        self, tmp_path: Path
    ) -> None:
        """load_batch and load_yaml_dir must apply the same filter.

        Both code paths must skip non-trace YAML (no input/frontier).
        """
        yml = tmp_path / "nondir"
        yml.mkdir()
        cfg = yml / "config.yaml"
        cfg.write_text(
            yaml.safe_dump({"app": "fit", "version": "1.0"}),
            encoding="utf-8",
        )

        dir_count = TraceIngester().load_yaml_dir(yml).count()
        batch_count = TraceIngester().load_batch([cfg]).count()

        assert dir_count == 0
        assert batch_count == 0, (
            f"load_batch must also filter non-trace, got {batch_count}"
        )


class TestPR32ParseRawNonDictFieldsRegression:
    """Regression: _parse_raw() must handle non-dict field values.

    PR #32 review item — _parse_raw() defensively normalizes non-dict
    field values (None, str, int, etc.) to empty dicts before calling
    .get() on them. This prevents AttributeError crashes when trace
    data has input: null, advice: "string", frontier: null, etc.
    """

    def test_parse_raw_null_input_returns_defaults(self) -> None:
        """input: null must produce empty defaults, not crash."""
        from fit.training.tracer import _parse_raw

        rec = _parse_raw({"input": None, "frontier": {}})
        assert rec.prompt == ""
        assert rec.context == {}

    def test_parse_raw_string_advice_returns_defaults(self) -> None:
        """advice: "not a dict" must produce empty advice defaults."""
        from fit.training.tracer import _parse_raw

        rec = _parse_raw({"input": {}, "advice": "not a dict"})
        assert rec.advice_text == ""
        assert rec.advice_domain == "unknown"

    def test_parse_raw_null_frontier_and_reward_returns_defaults(self) -> None:
        """frontier: null and reward: null must produce empty defaults."""
        from fit.training.tracer import _parse_raw

        rec = _parse_raw({"input": {}, "frontier": None, "reward": None})
        assert rec.frontier_output == ""
        assert rec.frontier_model == ""
        assert rec.reward_score is None
        assert rec.reward_breakdown == {}


class TestPR33JsonlNonDictLineRegression:
    """Regression: load_jsonl() must validate JSON-decoded values are dicts.

    PR #33 review item 3 — json.loads() can return list/str for valid
    JSON (e.g. "[]", '"hello"'). These get passed to _parse_raw() which
    calls raw.get(...) on the non-dict, raising AttributeError.
    Fix should raise ValueError with a clear message.
    """

    def test_jsonl_array_line_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """A JSONL file with [] on a line must raise ValueError."""
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text("[]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object"):
            TraceIngester().load_jsonl(jsonl)

    def test_jsonl_string_line_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """A JSONL file with "hello" on a line must raise ValueError."""
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text('"hello"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object"):
            TraceIngester().load_jsonl(jsonl)


class TestPR33ConfidenceTypeRegression:
    """Regression: _parse_raw() must handle non-float confidence values.

    PR #33 review item 4 — advice.get("confidence", 0.0) returns None
    when confidence key exists with null value (dict.get skips default
    for existing keys). float(None) raises TypeError. Similarly
    float("high") raises ValueError. Fix should coerce to 0.0.
    """

    def test_parse_raw_null_confidence_returns_zero(self) -> None:
        """advice.confidence=null must produce advice_confidence=0.0."""
        from fit.training.tracer import _parse_raw

        rec = _parse_raw({"input": {}, "advice": {"confidence": None}})
        assert rec.advice_confidence == 0.0

    def test_parse_raw_string_confidence_returns_zero(self) -> None:
        """advice.confidence='high' must produce advice_confidence=0.0."""
        from fit.training.tracer import _parse_raw

        rec = _parse_raw({"input": {}, "advice": {"confidence": "high"}})
        assert rec.advice_confidence == 0.0

    def test_parse_raw_numeric_confidence_works(self) -> None:
        """advice.confidence=0.8 must still produce advice_confidence=0.8.

        This is the happy path — must not regress when fix is applied.
        """
        from fit.training.tracer import _parse_raw

        rec = _parse_raw({"input": {}, "advice": {"confidence": 0.8}})
        assert rec.advice_confidence == 0.8


class TestPR33LoadBatchJsonNonDictRegression:
    """Regression: load_batch() JSON path must validate list items are dicts.

    PR #33 review item 5 — the JSON branch (lines 197-199) iterates
    over list items without checking they're dicts. Non-dict items
    get passed to _parse_raw() which crashes with AttributeError.
    Fix should raise ValueError with a clear message.
    """

    def test_json_array_with_non_dict_items_raises(
        self, tmp_path: Path
    ) -> None:
        """JSON file with [dict, 42] must raise ValueError."""
        j = tmp_path / "traces.json"
        j.write_text(
            json.dumps([{"input": {}}, 42]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="dict"):
            TraceIngester().load_batch([j])

    def test_json_array_with_all_dicts_works(self, tmp_path: Path) -> None:
        """JSON file with all-dict list must load normally."""
        j = tmp_path / "traces.json"
        j.write_text(
            json.dumps([{"input": {}}, {"frontier": {}}]),
            encoding="utf-8",
        )
        ingester = TraceIngester().load_batch([j])
        assert ingester.count() == 2


class TestPR34LoadBatchErrorMessageWordingRegression:
    """Regression: load_batch() JSON error message must mention "dict" or
    "JSON object" — not just "an object".

    PR #34 review item 1 — the ValueError raised at tracer.py:206-209
    says "Expected each item ... to be an object" but the existing
    test (TestPR33LoadBatchJsonNonDictRegression) uses match="dict"
    which never matches. The word "dict" does not appear in the error.
    Either the error message should include "dict" (or "JSON object")
    or the existing test matcher is wrong.

    This regression test passes once the error message is corrected to
    include "dict" or "JSON object".
    """

    def test_error_message_contains_dict_or_json_object(
        self, tmp_path: Path
    ) -> None:
        """ValueError for non-dict JSON array item must mention
        "dict" or "JSON object" so tests can match reliably."""
        j = tmp_path / "traces.json"
        j.write_text(
            json.dumps([{"input": {}}, 42]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as exc_info:
            TraceIngester().load_batch([j])
        msg = str(exc_info.value).lower()
        assert "dict" in msg or "json object" in msg, (
            f"Error message must contain 'dict' or 'JSON object', "
            f"got: {exc_info.value!r}"
        )


# ---------------------------------------------------------------------------
# Regression: SQLite non-dict JSON blob raises ValueError
# ---------------------------------------------------------------------------


class TestSqliteJsonBlobNonDictRegression:
    """load_sqlite deserialises each row's ``data`` column via
    ``json.loads`` then passes the result to ``_parse_raw()``.
    ``_parse_raw`` expects a dict and calls ``.get()``, so a
    non-dict value (list, str, int) raises ``AttributeError``
    instead of a clear ``ValueError``.
    """

    def test_non_dict_json_blob_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """A JSON array stored in the data column must raise
        ValueError, not AttributeError."""
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute(
            "INSERT INTO traces (data) VALUES (?)",
            (json.dumps([1, 2, 3]),),
        )
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match=r"dict|JSON object|row"):
            TraceIngester().load_sqlite(str(db_path))

    def test_non_dict_string_blob_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """A plain JSON string stored in the data column must raise
        ValueError, not AttributeError."""
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute(
            "INSERT INTO traces (data) VALUES (?)",
            (json.dumps("just a string"),),
        )
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match=r"dict|JSON object|row"):
            TraceIngester().load_sqlite(str(db_path))


# ---------------------------------------------------------------------------
# Regression: JSON branch filters non-trace dicts
# ---------------------------------------------------------------------------


class TestJsonBranchIngestsNonTraceDictsRegression:
    """load_batch JSON path appends every dict via _parse_raw() without
    checking for ``input`` or ``frontier`` keys. Unlike the YAML path,
    non-trace dicts (config files, reward schemas) produce mostly-empty
    TraceRecords instead of being skipped.
    """

    def test_json_array_filters_non_trace_dicts(
        self, tmp_path: Path
    ) -> None:
        """JSON array with one config dict and one trace dict should
        yield count == 1 (only the trace)."""
        data = [
            {"app": "fit"},
            {
                "input": {"prompt": "x"},
                "frontier": {"model": "m", "output": "ok"},
            },
        ]
        p = tmp_path / "mixed.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        ingester = TraceIngester().load_batch([p])
        assert ingester.count() == 1, (
            "JSON branch ingested non-trace dict. "
            f"Expected 1 record, got {ingester.count()}."
        )

    def test_single_non_trace_json_object_yields_zero(
        self, tmp_path: Path
    ) -> None:
        """A top-level JSON object without input/frontier keys must
        produce count == 0."""
        data = {"app": "fit", "version": "1.0"}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        ingester = TraceIngester().load_batch([p])
        assert ingester.count() == 0, (
            "JSON branch ingested non-trace top-level dict. "
            f"Expected 0 records, got {ingester.count()}."
        )


# ---------------------------------------------------------------------------
# Regression: load_sqlite uses cursor streaming not fetchall
# ---------------------------------------------------------------------------


class TestLoadSqliteFetchallRegression:
    """``TraceIngester.load_sqlite`` calls ``.fetchall()`` on both the
    JSON-blob path and the fallback individual-columns path. This loads
    entire tables into memory before iterating, which can cause OOM on
    large trace tables. Both paths should stream rows via cursor
    iteration instead.
    """

    def test_json_blob_path_does_not_use_fetchall(self) -> None:
        """The JSON-blob SELECT path must not call ``.fetchall()`` --
        it should iterate the cursor directly to avoid loading the
        full result set into memory."""
        import inspect

        from fit.training.tracer import TraceIngester

        source = inspect.getsource(TraceIngester.load_sqlite)
        # Isolate the JSON blob section
        blob_start = source.index("JSON blob column first")
        blob_end = source.index("Fallback: individual columns")
        blob_section = source[blob_start:blob_end]
        assert ".fetchall()" not in blob_section, (
            "JSON-blob path calls .fetchall() instead of "
            "streaming rows via cursor iteration"
        )

    def test_fallback_column_path_does_not_use_fetchall(self) -> None:
        """The fallback individual-columns SELECT path must not call
        ``.fetchall()`` -- it should iterate the cursor directly."""
        import inspect

        from fit.training.tracer import TraceIngester

        source = inspect.getsource(TraceIngester.load_sqlite)
        # Isolate the fallback section
        fallback_start = source.index("Fallback: individual columns")
        fallback_section = source[fallback_start:]
        assert ".fetchall()" not in fallback_section, (
            "Fallback column path calls .fetchall() instead "
            "of streaming rows via cursor iteration"
        )


# ---------------------------------------------------------------------------
# Regression: bare string/number JSON raises ValueError
# ---------------------------------------------------------------------------


class TestJsonBareValueSilentlyIgnored:
    """``TraceIngester.load_batch()`` JSON branch deserialises the file
    with ``json.load()`` and then checks ``isinstance(raw, list)`` and
    ``isinstance(raw, dict)``. When the top-level value is a bare
    string, number, or null the code falls through both branches and
    returns silently with zero records, instead of raising
    ``ValueError`` as ``load_jsonl()`` does for non-dict records.
    """

    def test_bare_string_raises(self, tmp_path: Path) -> None:
        """A JSON file containing a bare string must
        raise ``ValueError`` -- not silently succeed with 0 records."""
        p = tmp_path / "bare_string.json"
        p.write_text('"hello"', encoding="utf-8")

        with pytest.raises(ValueError):
            TraceIngester().load_batch([p])

    def test_bare_number_raises(self, tmp_path: Path) -> None:
        """A JSON file containing a bare number must raise
        ``ValueError`` -- not silently succeed with 0 records."""
        p = tmp_path / "bare_number.json"
        p.write_text("42", encoding="utf-8")

        with pytest.raises(ValueError):
            TraceIngester().load_batch([p])


# ---------------------------------------------------------------------------
# Regression: invalid JSON in SQLite raises ValueError with context
# ---------------------------------------------------------------------------


class TestLoadSqliteMissingJSONDecodeErrorWrapping:
    """load_sqlite calls json.loads() without catching json.JSONDecodeError.

    Unlike load_jsonl(), which wraps decode errors with location context
    (file path, line number), load_sqlite lets a raw JSONDecodeError escape
    when a row contains invalid JSON in the ``data`` column.  Callers get
    no table or row context, making debugging difficult.

    Expected: ``ValueError`` whose message includes table/row context.
    Actual:   raw ``json.JSONDecodeError``.
    """

    def test_invalid_json_raises_value_error_with_context(
        self, tmp_path: Path
    ) -> None:
        """Corrupt JSON in data column should raise ValueError,
        not JSONDecodeError."""
        db_path = tmp_path / "bad.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE traces (data TEXT)")
        conn.execute(
            "INSERT INTO traces (data) VALUES (?)",
            ("not json at all",),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()

        with pytest.raises(
            ValueError,
            match=r"row|table",
        ) as exc_info:
            ingester.load_sqlite(db_path)

        assert exc_info.type is ValueError, (
            "load_sqlite raises raw JSONDecodeError "
            "instead of ValueError with table/row context"
        )


# ---------------------------------------------------------------------------
# Regression: malformed YAML raises ValueError with path
# ---------------------------------------------------------------------------


class TestYamlSafeLoadMissingYAMLErrorCatch:
    """load_yaml_dir calls yaml.safe_load without catching yaml.YAMLError.

    A malformed YAML file raises a raw ``yaml.YAMLError`` that escapes
    without any file-path context, making it impossible for callers to
    identify which file is corrupt.

    Expected: ``ValueError`` whose message includes the file path.
    Actual:   raw ``yaml.YAMLError``.
    """

    def test_malformed_yaml_raises_value_error_with_path(
        self, tmp_path: Path
    ) -> None:
        """Malformed YAML should raise ValueError containing
        the file path."""
        bad_file = tmp_path / "step-001.yaml"
        bad_file.write_text(":\n  - [invalid", encoding="utf-8")

        ingester = TraceIngester()

        with pytest.raises(
            ValueError, match=r"step-001\.yaml"
        ) as exc_info:
            ingester.load_yaml_dir(tmp_path)

        assert exc_info.type is ValueError, (
            "load_yaml_dir raises raw yaml.YAMLError "
            "instead of ValueError with file-path context"
        )


# -- helpers (SQLite test data builders) ------------------------------------


def _make_trace_blob(
    *,
    prompt: str = "test prompt",
    output: str = "test output",
    model: str = "gpt-4",
    score: float = 0.85,
    domain: str = "fitness",
    session_id: str = "sess-001",
    trace_id: str = "t-001",
    timestamp: str = "2025-01-15T10:00:00Z",
) -> dict:
    """Build a valid trace dict matching the expected nested schema."""
    return {
        "id": trace_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "input": {
            "prompt": prompt,
            "context": {"source": "test"},
        },
        "advice": {
            "steering_text": f"advice for {domain}",
            "domain": domain,
            "confidence": 0.9,
        },
        "frontier": {"output": output, "model": model},
        "reward": {
            "score": score,
            "breakdown": {"quality": 0.8},
        },
        "metadata": {"tenant": "acme"},
    }


def _create_sqlite_with_data_column(
    db_path: Path,
    rows: list[str],
    table: str = "traces",
) -> None:
    """Create a SQLite DB with a ``data TEXT`` column and insert rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(f"CREATE TABLE {table} (data TEXT)")
    for row in rows:
        conn.execute(
            f"INSERT INTO {table} (data) VALUES (?)", (row,)
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Regression: SQLite JSON blob happy path tests
# ---------------------------------------------------------------------------


class TestLoadSqliteJsonBlobHappyPath:
    """Verify the JSON-blob (``data`` column) path works correctly.

    These tests exercise the primary code path in ``load_sqlite`` where
    each row has a single ``data TEXT`` column containing a full JSON
    trace object.
    """

    def test_loads_multiple_valid_traces(self, tmp_path: Path) -> None:
        """Three valid JSON blobs load into three correct TraceRecords."""
        blobs = [
            _make_trace_blob(
                trace_id=f"t-{i}",
                prompt=f"prompt {i}",
                output=f"output {i}",
                score=float(i) / 10,
            )
            for i in range(1, 4)
        ]
        db_path = tmp_path / "traces.db"
        _create_sqlite_with_data_column(
            db_path, [json.dumps(b) for b in blobs]
        )

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()

        assert len(records) == 3, (
            f"Expected 3 records, got {len(records)}"
        )
        for i, rec in enumerate(records, 1):
            assert rec.id == f"t-{i}", (
                f"Record {i} id mismatch: {rec.id}"
            )
            assert rec.prompt == f"prompt {i}", (
                f"Record {i} prompt mismatch: {rec.prompt}"
            )
            assert rec.frontier_output == f"output {i}", (
                f"Record {i} frontier_output mismatch: "
                f"{rec.frontier_output}"
            )
            assert rec.reward_score == float(i) / 10, (
                f"Record {i} reward_score mismatch: "
                f"{rec.reward_score}"
            )

    def test_non_trace_dict_still_loaded_via_json_blob(
        self, tmp_path: Path
    ) -> None:
        """JSON blob path does not filter by input/frontier keys.

        Unlike load_batch's JSON branch which checks for ``input`` or
        ``frontier`` keys, load_sqlite's JSON blob path feeds every row
        through ``_parse_raw``.  A non-trace dict should still produce
        a record (with empty/default fields).
        """
        valid_blob = _make_trace_blob()
        non_trace = {"foo": "bar", "baz": 42}

        db_path = tmp_path / "mixed.db"
        _create_sqlite_with_data_column(
            db_path,
            [json.dumps(valid_blob), json.dumps(non_trace)],
        )

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()

        assert len(records) == 2, (
            "Both rows should be loaded -- JSON blob path does not "
            f"filter by input/frontier; got {len(records)}"
        )
        # The non-trace record should have empty defaults
        non_trace_rec = records[1]
        assert non_trace_rec.prompt == "", (
            "Non-trace dict should produce empty prompt"
        )

    def test_empty_table_returns_zero_records(
        self, tmp_path: Path
    ) -> None:
        """Empty traces table should not crash and return 0 records."""
        db_path = tmp_path / "empty.db"
        _create_sqlite_with_data_column(db_path, [])

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)

        assert ingester.count() == 0, (
            "Empty table should produce 0 records without crashing"
        )


# ---------------------------------------------------------------------------
# Regression: SQLite fallback branch with JSON columns
# ---------------------------------------------------------------------------


class TestLoadSqliteFallbackBranch:
    """Fallback branch (no ``data`` column) produces empty TraceRecords.

    When the table has individual columns instead of a ``data`` JSON
    blob, ``load_sqlite`` falls through to ``SELECT * FROM table`` and
    builds ``raw = dict(zip(cols, row))`` with flat column names.

    ``_parse_raw()`` expects nested dicts under keys ``input``,
    ``advice``, ``frontier``, ``reward``.  Flat columns like
    ``prompt``, ``output``, ``score`` are ignored, producing records
    where every meaningful field is empty/default.

    Even when columns *are* named ``input``, ``advice``, etc. and
    contain JSON text, ``_parse_raw`` receives the raw JSON string
    (not a parsed dict), so ``isinstance(inp_raw, dict)`` fails and
    the value is discarded.
    """

    def test_nested_json_columns_produce_populated_records(
        self, tmp_path: Path
    ) -> None:
        """Columns named input/advice/frontier/reward with JSON text.

        The fallback path should parse JSON strings in these columns
        back into dicts so _parse_raw can extract fields.
        """
        db_path = tmp_path / "nested.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE traces ("
            "  id TEXT,"
            "  session_id TEXT,"
            "  timestamp TEXT,"
            "  input TEXT,"
            "  advice TEXT,"
            "  frontier TEXT,"
            "  reward TEXT,"
            "  metadata TEXT"
            ")"
        )
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "t-100",
                "sess-100",
                "2025-03-01T12:00:00Z",
                json.dumps({
                    "prompt": "nested prompt",
                    "context": {"source": "test"},
                }),
                json.dumps({
                    "steering_text": "nested advice",
                    "domain": "nutrition",
                    "confidence": 0.95,
                }),
                json.dumps({
                    "output": "nested output",
                    "model": "gpt-4",
                }),
                json.dumps({
                    "score": 0.9,
                    "breakdown": {"quality": 0.85},
                }),
                json.dumps({"tenant": "acme"}),
            ),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()

        assert len(records) == 1, (
            f"Expected 1 record, got {len(records)}"
        )
        rec = records[0]
        assert rec.prompt == "nested prompt", (
            "Fallback branch produces empty prompt; "
            f"got {rec.prompt!r}"
        )
        assert rec.advice_text == "nested advice", (
            "Fallback branch produces empty advice_text; "
            f"got {rec.advice_text!r}"
        )
        assert rec.frontier_output == "nested output", (
            "Fallback branch produces empty frontier_output; "
            f"got {rec.frontier_output!r}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Truly flat columns (prompt, output, score) are not "
            "mapped to nested TraceRecord schema -- known limitation"
        ),
    )
    def test_truly_flat_columns_produce_populated_records(
        self, tmp_path: Path
    ) -> None:
        """Columns like prompt, output, score, domain -- no nesting.

        The fallback should map well-known flat column names to the
        corresponding TraceRecord fields.
        """
        db_path = tmp_path / "flat.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE traces ("
            "  prompt TEXT,"
            "  output TEXT,"
            "  score REAL,"
            "  domain TEXT"
            ")"
        )
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?, ?)",
            ("flat prompt", "flat output", 0.77, "strength"),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()
        ingester.load_sqlite(db_path)
        records = ingester.to_trace_records()

        assert len(records) == 1, (
            f"Expected 1 record, got {len(records)}"
        )
        rec = records[0]
        assert rec.prompt == "flat prompt", (
            "Flat column 'prompt' not mapped; "
            f"got {rec.prompt!r}"
        )
        assert rec.frontier_output == "flat output", (
            "Flat column 'output' not mapped; "
            f"got {rec.frontier_output!r}"
        )
        assert rec.reward_score == 0.77, (
            "Flat column 'score' not mapped; "
            f"got {rec.reward_score!r}"
        )


# ---------------------------------------------------------------------------
# Regression: SQLite error handling tests
# ---------------------------------------------------------------------------


class TestLoadSqliteJsonBlobErrorHandling:
    """Edge cases and error handling for the JSON blob
    (data column) path."""

    def test_invalid_json_raises_value_error_with_context(
        self, tmp_path: Path
    ) -> None:
        """Corrupt JSON in data column should raise ValueError
        with context."""
        db_path = tmp_path / "bad_json.db"
        _create_sqlite_with_data_column(
            db_path, ["not valid json at all"]
        )

        ingester = TraceIngester()

        with pytest.raises(ValueError, match=r"row|table"):
            ingester.load_sqlite(db_path)

    def test_non_dict_json_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """Array JSON in data column should raise ValueError
        mentioning dict."""
        db_path = tmp_path / "array.db"
        _create_sqlite_with_data_column(
            db_path, [json.dumps([1, 2, 3])]
        )

        ingester = TraceIngester()

        with pytest.raises(
            ValueError, match=r"dict"
        ) as exc_info:
            ingester.load_sqlite(db_path)

        assert exc_info.type is ValueError, (
            "Non-dict JSON should raise ValueError "
            "mentioning 'dict'"
        )

    def test_mixed_valid_invalid_raises_on_bad_row(
        self, tmp_path: Path
    ) -> None:
        """Mix of valid and invalid rows should error on the bad row
        with index."""
        valid_1 = json.dumps(_make_trace_blob(trace_id="t-ok-1"))
        valid_2 = json.dumps(_make_trace_blob(trace_id="t-ok-2"))
        invalid = "{{broken json}}"

        db_path = tmp_path / "mixed_err.db"
        _create_sqlite_with_data_column(
            db_path, [valid_1, valid_2, invalid]
        )

        ingester = TraceIngester()

        with pytest.raises(
            ValueError, match=r"row\s+3"
        ) as exc_info:
            ingester.load_sqlite(db_path)

        assert exc_info.type is ValueError, (
            "Invalid row should raise ValueError "
            "with row index context"
        )


# ---------------------------------------------------------------------------
# Regression: load_batch param named fmt not format
# ---------------------------------------------------------------------------


class TestFormatParameterShadowsBuiltin:
    """load_batch uses ``format`` as a parameter name, shadowing the
    built-in. The parameter should be renamed to ``fmt`` or similar.

    Expected: parameter named ``fmt`` (or anything other than
    ``format``).
    Actual:   parameter named ``format``.
    """

    def test_load_batch_does_not_shadow_format_builtin(
        self,
    ) -> None:
        """The second parameter of load_batch should not be named
        'format'."""
        import inspect

        sig = inspect.signature(TraceIngester.load_batch)
        param_names = list(sig.parameters.keys())

        assert "format" not in param_names, (
            "load_batch uses 'format' as parameter name, "
            f"shadowing the built-in; params: {param_names}"
        )


# ---------------------------------------------------------------------------
# Regression: malformed single YAML in load_batch
# ---------------------------------------------------------------------------


class TestLoadBatchSingleYamlMissingYAMLErrorCatch:
    """load_batch reads single YAML files via yaml.safe_load without
    catching yaml.YAMLError.

    ``load_yaml_dir()`` already wraps the error and includes the file
    path, but the inline single-file branch does not.  A malformed
    YAML file raises a raw ``yaml.YAMLError`` that escapes without
    any file-path context.

    Expected: ``ValueError`` whose message includes the file path.
    Actual:   raw ``yaml.YAMLError``.
    """

    def test_malformed_single_yaml_raises_value_error_with_path(
        self, tmp_path: Path
    ) -> None:
        """Malformed single YAML should raise ValueError containing
        the path."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(":\n  - [invalid", encoding="utf-8")

        ingester = TraceIngester()

        with pytest.raises(
            ValueError, match=r"bad\.yaml"
        ) as exc_info:
            ingester.load_batch([bad_file])

        assert exc_info.type is ValueError, (
            "load_batch raises raw yaml.YAMLError "
            "instead of ValueError with file-path context"
        )


# ---------------------------------------------------------------------------
# Regression: malformed JSON in load_batch
# ---------------------------------------------------------------------------


class TestLoadBatchJsonMissingJSONDecodeErrorCatch:
    """load_batch JSON branch calls ``json.load(f)`` without catching
    ``json.JSONDecodeError``.

    The YAML branch wraps ``yaml.YAMLError`` into ``ValueError`` with
    the file path, and the JSONL branch wraps ``json.JSONDecodeError``
    similarly, but the JSON branch does not.  A malformed ``.json``
    file raises a raw ``json.JSONDecodeError`` that escapes without
    any file-path context.

    Expected: ``ValueError`` whose message includes the file path.
    Actual:   raw ``json.JSONDecodeError``.
    """

    def test_malformed_json_raises_value_error_with_path(
        self, tmp_path: Path
    ) -> None:
        """Malformed .json file should raise ValueError containing
        the path."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json{", encoding="utf-8")

        ingester = TraceIngester()

        with pytest.raises(
            ValueError, match=r"bad\.json"
        ) as exc_info:
            ingester.load_batch([bad_file])

        assert exc_info.type is ValueError, (
            "load_batch raises raw json.JSONDecodeError "
            "instead of ValueError with file-path context"
        )


# ---------------------------------------------------------------------------
# Regression: JSONL branch handles directories
# ---------------------------------------------------------------------------


class TestLoadBatchJsonlBranchOnDirectory:
    """``load_batch()`` JSONL branch calls ``self.load_jsonl(p)``
    without checking whether ``p`` is a directory.

    If ``_detect_format(dir)`` returns ``"jsonl"`` (because the
    directory contains ``*.jsonl`` files but no ``*.yaml``/``*.yml``),
    ``load_jsonl`` tries to open a directory as a file, raising
    ``IsADirectoryError``.

    Expected: directory is walked; contained ``.jsonl`` files are
    loaded.
    Actual:   ``IsADirectoryError``.
    """

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
            "load_batch JSONL branch raises "
            "IsADirectoryError on directories instead of loading "
            "contained .jsonl files"
        )


# ---------------------------------------------------------------------------
# Regression: _detect_format includes *.ndjson
# ---------------------------------------------------------------------------


class TestDetectFormatMissingNdjsonGlob:
    """``_detect_format()`` checks ``path.glob("*.jsonl")`` for
    directories but not ``*.ndjson``, even though single-file
    detection treats ``.ndjson`` as ``"jsonl"``.

    A directory containing only ``*.ndjson`` files falls through to
    the default ``"yaml"`` return.

    Expected: ``"jsonl"``.
    Actual:   ``"yaml"`` (default).
    """

    def test_ndjson_directory_detected_as_jsonl(
        self, tmp_path: Path
    ) -> None:
        """Directory with only .ndjson files should be detected
        as jsonl."""
        ndjson_dir = tmp_path / "ndjson_traces"
        ndjson_dir.mkdir()

        record = {
            "input": {"prompt": "test"},
            "advice_domain": "test",
        }
        ndjson_file = ndjson_dir / "traces.ndjson"
        ndjson_file.write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

        result = _detect_format(ndjson_dir)

        assert result == "jsonl", (
            "_detect_format returns 'yaml' for directories "
            "containing only .ndjson files instead of 'jsonl'"
        )


# ---------------------------------------------------------------------------
# Regression: _detect_format uses rglob for JSONL
# ---------------------------------------------------------------------------


class TestDetectFormatNonRecursiveJsonlGlob:
    """``_detect_format()`` uses ``path.glob("*.jsonl")``
    (non-recursive) for JSONL/NDJSON detection but
    ``path.rglob("*.y*ml")`` (recursive) for YAML.

    When a directory contains JSONL files only in subdirectories
    (none at root level), the YAML check with ``rglob`` finds
    nothing, the JSONL check with ``glob`` also finds nothing
    (because the files are nested), and the function returns the
    ``"yaml"`` default.

    Expected: ``"jsonl"`` when subdirectories contain ``.jsonl``
    files.
    Actual:   ``"yaml"`` (falls through to default).
    """

    def test_nested_jsonl_detected_as_jsonl(
        self, tmp_path: Path
    ) -> None:
        """Dir with JSONL only in a subdir should be detected
        as jsonl."""
        subdir = tmp_path / "session_01"
        subdir.mkdir()
        (subdir / "traces.jsonl").write_text(
            '{"prompt":"hi","response":"hello"}\n',
            encoding="utf-8",
        )

        result = _detect_format(tmp_path)

        assert result == "jsonl", (
            "_detect_format uses glob() not rglob() for "
            "JSONL -- nested .jsonl files mis-detected as yaml"
        )


# ---------------------------------------------------------------------------
# Regression: corrupt JSON in fallback raises ValueError
# ---------------------------------------------------------------------------


class TestSqliteFallbackSilentJsonErrorRegression:
    """``load_sqlite`` falls back to per-column JSON parsing when no
    ``data`` blob exists. Corrupt JSON in those columns is silently
    swallowed by a bare ``except (json.JSONDecodeError, ValueError):
    pass``, yielding TraceRecords with empty default values and no
    error signal.
    """

    def test_corrupt_json_in_fallback_column_signals_error(
        self, tmp_path: Path
    ) -> None:
        """A row with invalid JSON in the ``input`` column must
        either raise ``ValueError`` with context or populate the
        record with meaningful data -- not silently produce empty
        defaults."""
        db_path = tmp_path / "traces.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE traces "
            "(id TEXT, input TEXT, frontier TEXT)"
        )
        conn.execute(
            "INSERT INTO traces VALUES (?, ?, ?)",
            (
                "row-1",
                'not valid json{"',
                '{"model": "test"}',
            ),
        )
        conn.commit()
        conn.close()

        ingester = TraceIngester()
        try:
            ingester.load_sqlite(db_path)
        except ValueError:
            # Acceptable: raising with context about corrupt data
            return

        records = ingester.records
        assert len(records) == 1, "Expected exactly one record"
        record = records[0]
        # If it didn't raise, the record's prompt must not be
        # silently empty.
        assert record.prompt != "", (
            "Corrupt JSON in the input column was silently "
            "swallowed -- the record has empty defaults with "
            "no error signal."
        )


# ---------------------------------------------------------------------------
# Regression: load_yaml_dir ingests .yml files
# ---------------------------------------------------------------------------


class TestYmlExtensionGlob:
    """load_yaml_dir must ingest .yml files, not just .yaml."""

    def test_yml_files_ingested_from_dir(
        self, tmp_path: Path
    ) -> None:
        session_dir = tmp_path / "sess_yml"
        session_dir.mkdir()
        trace = {
            "id": "yml-test-001",
            "session_id": "sess_yml",
            "timestamp": "2026-04-16T12:00:00Z",
            "input": {"prompt": "test"},
            "advice": {
                "domain": "test",
                "steering_text": "go",
            },
            "frontier": {"model": "m", "output": "ok"},
            "reward": {"score": 0.5},
        }
        (session_dir / "step-001.yml").write_text(
            yaml.safe_dump(trace), encoding="utf-8"
        )

        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() >= 1, (
            ".yml files not ingested. "
            f"Got {ingester.count()} records from dir with "
            ".yml cassette."
        )

    def test_yaml_and_yml_both_ingested(
        self, tmp_path: Path
    ) -> None:
        session_dir = tmp_path / "sess_mixed"
        session_dir.mkdir()
        trace_a = {
            "id": "yaml-001",
            "session_id": "s",
            "timestamp": "2026-04-16T12:00:00Z",
            "input": {"prompt": "a"},
            "advice": {"domain": "d"},
            "frontier": {"model": "m", "output": "a"},
            "reward": {},
        }
        trace_b = {**trace_a, "id": "yml-001"}
        (session_dir / "step-001.yaml").write_text(
            yaml.safe_dump(trace_a), encoding="utf-8"
        )
        (session_dir / "step-002.yml").write_text(
            yaml.safe_dump(trace_b), encoding="utf-8"
        )

        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() == 2, (
            "Only .yaml ingested, .yml missed. "
            f"Got {ingester.count()} records, expected 2."
        )


class TestDetectFormatMixedDirectoryRegression:
    """Directories with multiple trace formats (YAML + JSONL/JSON) should
    not silently skip valid traces. When a directory contains both YAML
    and JSONL files, passing it to load_batch must ingest ALL formats,
    not just the first one detected by _detect_format.
    """

    def test_mixed_yaml_and_jsonl_directory(self, tmp_path: Path) -> None:
        """Directory with both .yaml and .jsonl files must load all traces."""
        import yaml as _yaml

        # Create a YAML trace
        session_dir = tmp_path / "sess"
        session_dir.mkdir()
        yaml_trace = {
            "id": "yaml-001",
            "session_id": "s",
            "timestamp": "2026-04-17T00:00:00Z",
            "input": {"prompt": "yaml prompt"},
            "advice": {"domain": "d"},
            "frontier": {"model": "m", "output": "yaml out"},
            "reward": {},
        }
        (session_dir / "step-001.yaml").write_text(
            _yaml.safe_dump(yaml_trace), encoding="utf-8"
        )

        # Create a JSONL trace (at root level, not in session dir)
        jsonl_trace = {
            "id": "jsonl-001",
            "session_id": "s",
            "timestamp": "2026-04-17T00:00:00Z",
            "input": {"prompt": "jsonl prompt"},
            "advice": {"domain": "d"},
            "frontier": {"model": "m", "output": "jsonl out"},
            "reward": {},
        }
        (tmp_path / "traces.jsonl").write_text(
            json.dumps(jsonl_trace), encoding="utf-8"
        )

        ingester = TraceIngester().load_batch([tmp_path])
        assert ingester.count() == 2, (
            f"Expected 2 records (1 YAML + 1 JSONL), got {ingester.count()} — "
            "directory with mixed formats must load all supported formats"
        )
