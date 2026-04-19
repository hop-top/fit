"""Tests for fit.bench.harness — benchmark wrappers."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fit.bench.harness import run_aider


RESULTS_KEYS = {"pass_rate_1", "pass_rate_2", "total", "correct_edit_format_pct"}


def _fake_results(total: int = 10, passed_1: int = 7, passed_2: int = 8) -> list[dict]:
    results = []
    for i in range(total):
        results.append({
            "language": "python",
            "testcase": f"test_{i}",
            "pass_1": i < passed_1,
            "pass_2": i < passed_2,
            "edit_format_correct": i < (total - 1),
        })
    return results


class TestRunAiderResultShape:
    @patch("fit.bench.harness.shutil.which", return_value="/usr/bin/aider")
    @patch("fit.bench.harness.subprocess.run")
    @patch("fit.bench.harness._collect_results")
    def test_returns_expected_keys(self, mock_collect, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        mock_collect.return_value = _fake_results()
        result = run_aider(endpoint="http://localhost:8000")
        assert set(result.keys()) == RESULTS_KEYS

    @patch("fit.bench.harness.shutil.which", return_value="/usr/bin/aider")
    @patch("fit.bench.harness.subprocess.run")
    @patch("fit.bench.harness._collect_results")
    def test_values_are_numeric(self, mock_collect, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        mock_collect.return_value = _fake_results()
        result = run_aider(endpoint="http://localhost:8000")
        for k, v in result.items():
            assert isinstance(v, (int, float)), f"{k}={v!r} not numeric"


class TestRunAiderRates:
    @patch("fit.bench.harness.shutil.which", return_value="/usr/bin/aider")
    @patch("fit.bench.harness.subprocess.run")
    @patch("fit.bench.harness._collect_results")
    def test_pass_rates(self, mock_collect, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        mock_collect.return_value = _fake_results(
            total=10, passed_1=7, passed_2=8,
        )
        result = run_aider(endpoint="http://localhost:8000")
        assert result["pass_rate_1"] == pytest.approx(70.0)
        assert result["pass_rate_2"] == pytest.approx(80.0)
        assert result["total"] == 10
        assert result["correct_edit_format_pct"] == pytest.approx(90.0)


class TestRunAiderLanguageFilter:
    @patch("fit.bench.harness.shutil.which", return_value="/usr/bin/aider")
    @patch("fit.bench.harness.subprocess.run")
    @patch("fit.bench.harness._collect_results")
    def test_languages_passed_to_subprocess(self, mock_collect, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        mock_collect.return_value = _fake_results(total=5, passed_1=3, passed_2=4)
        run_aider(
            endpoint="http://localhost:8000",
            languages=["python", "javascript"],
        )
        cmd = mock_run.call_args[0][0]
        assert "--languages" in cmd
        idx = cmd.index("--languages")
        assert cmd[idx + 1] == "python,javascript"


class TestRunAiderMissing:
    @patch("fit.bench.harness.shutil.which", return_value=None)
    def test_raises_file_not_found(self, mock_which):
        with pytest.raises(FileNotFoundError, match="aider"):
            run_aider(endpoint="http://localhost:8000")


class TestCollectResults:
    def test_parses_json_files(self, tmp_path):
        from fit.bench.harness import _collect_results

        data = _fake_results(total=3, passed_1=2, passed_2=3)
        results_file = tmp_path / ".aider.results.json"
        results_file.write_text(json.dumps(data))
        parsed = _collect_results(tmp_path)
        assert len(parsed) == 3
        assert parsed[0]["pass_1"] is True

    def test_empty_dir_returns_empty(self, tmp_path):
        from fit.bench.harness import _collect_results

        assert _collect_results(tmp_path) == []


# ── TAU-bench tests ────────────────────────────────────────────


def _make_fake_tau2() -> types.ModuleType:
    """Build a fake tau2 package with tau2.runner.run_domain."""
    tau2 = types.ModuleType("tau2")
    runner = types.ModuleType("tau2.runner")

    result = MagicMock()
    result.task_results = [
        MagicMock(reward=1.0),
        MagicMock(reward=0.0),
        MagicMock(reward=1.0),
    ]
    runner.run_domain = MagicMock(return_value=result)
    tau2.runner = runner
    return tau2


class TestRunTauResultShape:
    def test_returns_expected_keys(self):
        tau2 = _make_fake_tau2()
        with patch.dict(sys.modules, {"tau2": tau2, "tau2.runner": tau2.runner}):
            import fit.bench.harness as mod
            importlib.reload(mod)
            out = mod.run_tau(endpoint="http://localhost:8090")

        assert set(out.keys()) == {"passed", "total", "pass_rate"}

    def test_values_are_correct(self):
        tau2 = _make_fake_tau2()
        with patch.dict(sys.modules, {"tau2": tau2, "tau2.runner": tau2.runner}):
            import fit.bench.harness as mod
            importlib.reload(mod)
            out = mod.run_tau(endpoint="http://localhost:8090")

        assert out["passed"] == 2
        assert out["total"] == 3
        assert out["pass_rate"] == pytest.approx(2 / 3)


class TestRunTauDomain:
    def test_default_domain_is_retail(self):
        tau2 = _make_fake_tau2()
        with patch.dict(sys.modules, {"tau2": tau2, "tau2.runner": tau2.runner}):
            import fit.bench.harness as mod
            importlib.reload(mod)
            mod.run_tau(endpoint="http://localhost:8090")

        assert tau2.runner.run_domain.call_args[1]["domain"] == "retail"

    def test_domain_selection(self):
        tau2 = _make_fake_tau2()
        with patch.dict(sys.modules, {"tau2": tau2, "tau2.runner": tau2.runner}):
            import fit.bench.harness as mod
            importlib.reload(mod)
            mod.run_tau(endpoint="http://localhost:8090", domain="airline")

        assert tau2.runner.run_domain.call_args[1]["domain"] == "airline"


class TestRunTauParams:
    def test_task_ids_forwarded(self):
        tau2 = _make_fake_tau2()
        with patch.dict(sys.modules, {"tau2": tau2, "tau2.runner": tau2.runner}):
            import fit.bench.harness as mod
            importlib.reload(mod)
            mod.run_tau(endpoint="http://localhost:8090", task_ids=[1, 3, 5])

        assert tau2.runner.run_domain.call_args[1]["task_ids"] == [1, 3, 5]

    def test_concurrency_forwarded(self):
        tau2 = _make_fake_tau2()
        with patch.dict(sys.modules, {"tau2": tau2, "tau2.runner": tau2.runner}):
            import fit.bench.harness as mod
            importlib.reload(mod)
            mod.run_tau(endpoint="http://localhost:8090", max_concurrency=8)

        assert tau2.runner.run_domain.call_args[1]["max_concurrency"] == 8


class TestRunTauMissing:
    def test_import_error_when_missing(self):
        with patch.dict(sys.modules, {"tau2": None, "tau2.runner": None}):
            import fit.bench.harness as mod
            importlib.reload(mod)
            with pytest.raises(ImportError, match="tau2"):
                mod.run_tau(endpoint="http://localhost:8090")


class TestRunTauEdgeCases:
    def test_zero_tasks_returns_zero_rate(self):
        tau2 = _make_fake_tau2()
        tau2.runner.run_domain.return_value.task_results = []
        with patch.dict(sys.modules, {"tau2": tau2, "tau2.runner": tau2.runner}):
            import fit.bench.harness as mod
            importlib.reload(mod)
            out = mod.run_tau(endpoint="http://localhost:8090")

        assert out["total"] == 0
        assert out["pass_rate"] == 0.0
        assert out["passed"] == 0
