"""Regression tests for PR #27 code review items.

Each test proves the bug exists before the fix is applied.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from fit.training.tracer import TraceIngester


# ---------------------------------------------------------------------------
# Bug 1: load_yaml_dir only globs *.yaml, misses *.yml files
# File: tracer.py, line 96
# ---------------------------------------------------------------------------


class TestYmlExtensionGlob:
    """load_yaml_dir must ingest .yml files, not just .yaml."""

    def test_yml_files_ingested_from_dir(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sess_yml"
        session_dir.mkdir()
        trace = {
            "id": "yml-test-001",
            "session_id": "sess_yml",
            "timestamp": "2026-04-16T12:00:00Z",
            "input": {"prompt": "test"},
            "advice": {"domain": "test", "steering_text": "go"},
            "frontier": {"model": "m", "output": "ok"},
            "reward": {"score": 0.5},
        }
        (session_dir / "step-001.yml").write_text(
            yaml.safe_dump(trace), encoding="utf-8"
        )

        ingester = TraceIngester().load_yaml_dir(tmp_path)
        assert ingester.count() >= 1, (
            "Bug confirmed: .yml files not ingested. "
            f"Got {ingester.count()} records from dir with .yml cassette."
        )

    def test_yaml_and_yml_both_ingested(self, tmp_path: Path) -> None:
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
            "Bug confirmed: only .yaml ingested, .yml missed. "
            f"Got {ingester.count()} records, expected 2."
        )


# ---------------------------------------------------------------------------
# Bug 2: _parse_score mishandles "Rating: 4/5" — matches 4 before fraction
# File: reward_fn.py, line 140-157
# ---------------------------------------------------------------------------


class TestParseScoreFractionWithPrefix:
    """_parse_score must handle "Rating: 4/5" as 0.8, not 0.4."""

    def test_rating_fraction_4_over_5(self) -> None:
        from fit.training.reward_fn import _parse_score

        score = _parse_score("Rating: 4/5")
        assert score == pytest.approx(0.8, abs=0.01), (
            f"Bug confirmed: 'Rating: 4/5' parsed as {score}, expected ~0.8. "
            "Prefix regex matches '4' before fraction regex, returns 0.4."
        )

    def test_score_fraction_3_over_10(self) -> None:
        from fit.training.reward_fn import _parse_score

        score = _parse_score("Score: 3/10")
        assert score == pytest.approx(0.3, abs=0.01), (
            f"Bug confirmed: 'Score: 3/10' parsed as {score}, expected ~0.3."
        )

    def test_score_fraction_7_over_8(self) -> None:
        from fit.training.reward_fn import _parse_score

        score = _parse_score("Score: 7/8")
        assert score == pytest.approx(0.875, abs=0.01), (
            f"Bug confirmed: 'Score: 7/8' parsed as {score}, expected ~0.875."
        )


# ---------------------------------------------------------------------------
# Bug 3: push_to_hub passes raw bytes as path_or_fileobj
# File: export.py, line 182-186
# ---------------------------------------------------------------------------


class TestPushToHubBytesArgument:
    """push_to_hub must pass BytesIO, not raw bytes, to HfApi.upload_file."""

    def test_uses_bytes_io_not_raw_bytes(self, tmp_path: Path) -> None:
        from fit.training.export import ModelExporter
        from fit.training.grpo import TrainingResult

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        exporter = ModelExporter(str(model_dir))
        result = TrainingResult(
            model_path=str(model_dir),
            epochs_completed=1,
            final_loss=0.1,
            reward_stats={
                "mean": 0.5,
                "std": 0.0,
                "min": 0.5,
                "max": 0.5,
                "count": 1.0,
            },
        )

        captured_args: dict = {}

        class FakeHfApi:
            def upload_folder(self, **kwargs: object) -> None:
                pass

            def upload_file(self, **kwargs: object) -> None:
                captured_args.update(kwargs)

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(HfApi=FakeHfApi)}):
            exporter.push_to_hub("test/repo", result)

        assert "path_or_fileobj" in captured_args
        file_obj = captured_args["path_or_fileobj"]
        import io

        assert isinstance(file_obj, io.BytesIO), (
            f"Bug: path_or_fileobj is {type(file_obj).__name__}, "
            "expected BytesIO. Raw bytes cause upload_file to fail."
        )


# ---------------------------------------------------------------------------
# Bug 4: serve_advisor yaml.safe_load returns None for empty file
# File: serve_advisor.py, line 53-77
# ---------------------------------------------------------------------------


class TestServeAdvisorEmptyYamlConfig:
    """_load_config must normalize empty YAML to {}, not return None."""

    def test_empty_yaml_returns_dict(self, tmp_path: Path) -> None:
        """_load_config must normalize yaml.safe_load None to {}."""
        from examples.serve_advisor import FileAdvisor

        model_dir = tmp_path / "advisor"
        model_dir.mkdir()
        # Empty YAML file — yaml.safe_load returns None
        (model_dir / "config.yaml").write_text("", encoding="utf-8")

        # _load_config should normalize None -> {}
        advisor = FileAdvisor(str(model_dir))
        assert isinstance(advisor._domain, str)
        assert advisor._domain == "general"  # default value

    def test_empty_yaml_in_file_advisor(self, tmp_path: Path) -> None:
        """FileAdvisor must handle empty YAML config without crashing."""
        from examples.serve_advisor import FileAdvisor

        model_dir = tmp_path / "advisor"
        model_dir.mkdir()
        (model_dir / "config.yaml").write_text("", encoding="utf-8")

        try:
            advisor = FileAdvisor(str(model_dir))
            assert advisor._domain is not None
        except AttributeError as exc:
            pytest.fail(
                f"Bug confirmed: empty YAML config causes {exc}. "
                "normalize yaml.safe_load result to {} when None."
            )
