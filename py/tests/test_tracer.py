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
