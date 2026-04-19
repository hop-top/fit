"""Tests for fit bench CLI entry points."""

from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from fit.bench.cli import build_parser, main


# ── parser construction ──────────────────────────────────────────


class TestBuildParser:
    def test_returns_parser(self):
        p = build_parser()
        assert p is not None

    def test_help_does_not_raise(self):
        p = build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--help"])
        assert exc.value.code == 0

    def test_no_args_prints_help(self, capsys):
        with pytest.raises(SystemExit):
            main(["fit-bench"])


# ── serve ────────────────────────────────────────────────────────


class TestServe:
    def test_defaults(self):
        p = build_parser()
        ns = p.parse_args(["serve", "--upstream", "openai"])
        assert ns.command == "serve"
        assert ns.mode == "plan"
        assert ns.upstream == "openai"
        assert ns.port == 8781

    def test_custom_args(self):
        p = build_parser()
        ns = p.parse_args([
            "serve",
            "--mode", "session",
            "--advisor", "/tmp/advisor.yaml",
            "--upstream", "anthropic",
            "--port", "9090",
        ])
        assert ns.mode == "session"
        assert ns.advisor == "/tmp/advisor.yaml"
        assert ns.upstream == "anthropic"
        assert ns.port == 9090

    def test_serve_outputs_config_to_stderr(self, capsys):
        main(["fit-bench", "serve", "--upstream", "openai"])
        err = capsys.readouterr().err
        assert "plan" in err
        assert "openai" in err
        assert "8781" in err


# ── run-swe ──────────────────────────────────────────────────────


class TestRunSwe:
    def test_defaults(self):
        p = build_parser()
        ns = p.parse_args(["run-swe", "--endpoint", "http://localhost:8781"])
        assert ns.command == "run-swe"
        assert ns.endpoint == "http://localhost:8781"
        assert ns.dataset == "princeton-nlp/SWE-bench_Lite"
        assert ns.output == "./predictions.jsonl"
        assert ns.instance_ids is None
        assert ns.max_workers == 4

    def test_custom_args(self):
        p = build_parser()
        ns = p.parse_args([
            "run-swe",
            "--endpoint", "http://localhost:8781",
            "--dataset", "custom/ds",
            "--output", "/tmp/out.jsonl",
            "--instance-ids", "a", "b",
            "--max-workers", "8",
        ])
        assert ns.dataset == "custom/ds"
        assert ns.output == "/tmp/out.jsonl"
        assert ns.instance_ids == ["a", "b"]
        assert ns.max_workers == 8

    @patch("fit.bench.cli.harness.run_swebench")
    def test_calls_harness(self, mock_run, capsys):
        mock_run.return_value = {"resolved": 5, "total": 10, "resolution_pct": 50.0}
        main([
            "fit-bench", "run-swe",
            "--endpoint", "http://localhost:8781",
        ])
        mock_run.assert_called_once_with(
            endpoint="http://localhost:8781",
            dataset="princeton-nlp/SWE-bench_Lite",
            output_path="./predictions.jsonl",
            instance_ids=None,
            max_workers=4,
        )
        out = json.loads(capsys.readouterr().out)
        assert out["resolved"] == 5

    @patch("fit.bench.cli.harness.run_swebench")
    def test_passes_instance_ids(self, mock_run, capsys):
        mock_run.return_value = {"resolved": 1, "total": 1, "resolution_pct": 100.0}
        main([
            "fit-bench", "run-swe",
            "--endpoint", "http://x",
            "--instance-ids", "id1", "id2",
        ])
        _, kwargs = mock_run.call_args
        assert kwargs["instance_ids"] == ["id1", "id2"]


# ── run-tau ──────────────────────────────────────────────────────


class TestRunTau:
    def test_defaults(self):
        p = build_parser()
        ns = p.parse_args(["run-tau", "--endpoint", "http://localhost:8781"])
        assert ns.command == "run-tau"
        assert ns.domain == "retail"
        assert ns.user_sim_model == "gpt-4o-mini"
        assert ns.max_concurrency == 4

    @patch("fit.bench.cli.harness.run_tau")
    def test_calls_harness(self, mock_run, capsys):
        mock_run.return_value = {"passed": 3, "total": 5, "pass_rate": 0.6}
        main([
            "fit-bench", "run-tau",
            "--endpoint", "http://localhost:8781",
            "--domain", "airline",
            "--user-sim-model", "gpt-4o",
            "--max-concurrency", "2",
        ])
        mock_run.assert_called_once_with(
            endpoint="http://localhost:8781",
            domain="airline",
            user_sim_model="gpt-4o",
            max_concurrency=2,
            task_ids=None,
        )
        out = json.loads(capsys.readouterr().out)
        assert out["passed"] == 3


# ── run-aider ────────────────────────────────────────────────────


class TestRunAider:
    def test_defaults(self):
        p = build_parser()
        ns = p.parse_args(["run-aider", "--endpoint", "http://localhost:8781"])
        assert ns.command == "run-aider"
        assert ns.exercises_dir == "vendor/polyglot-benchmark"
        assert ns.edit_format == "diff"
        assert ns.languages is None
        assert ns.tries == 2
        assert ns.threads == 10

    @patch("fit.bench.cli.harness.run_aider")
    def test_calls_harness(self, mock_run, capsys):
        mock_run.return_value = {
            "pass_rate_1": 50.0,
            "pass_rate_2": 70.0,
            "total": 10,
            "correct_edit_format_pct": 90.0,
        }
        main([
            "fit-bench", "run-aider",
            "--endpoint", "http://localhost:8781",
            "--languages", "python", "go",
            "--tries", "3",
            "--threads", "5",
        ])
        mock_run.assert_called_once_with(
            endpoint="http://localhost:8781",
            exercises_dir="vendor/polyglot-benchmark",
            edit_format="diff",
            languages=["python", "go"],
            tries=3,
            threads=5,
        )
        out = json.loads(capsys.readouterr().out)
        assert out["pass_rate_1"] == 50.0


# ── setup ────────────────────────────────────────────────────────


class TestSetup:
    @patch("fit.bench.cli.setup.check_environment")
    def test_calls_check_and_outputs_json(self, mock_check, capsys):
        mock_check.return_value = {
            "docker": "ok",
            "swebench": "missing",
            "tau2": "ok",
            "aider": "ok",
            "polyglot_exercises": "missing",
        }
        main(["fit-bench", "setup"])
        mock_check.assert_called_once()
        out = json.loads(capsys.readouterr().out)
        assert out["docker"] == "ok"
        assert out["swebench"] == "missing"

    @patch("fit.bench.cli.setup.check_environment")
    def test_prints_table_to_stderr(self, mock_check, capsys):
        mock_check.return_value = {
            "docker": "ok",
            "swebench": "missing",
        }
        main(["fit-bench", "setup"])
        err = capsys.readouterr().err
        assert "docker" in err
        assert "ok" in err


# ── error handling ───────────────────────────────────────────────


class TestErrors:
    @patch("fit.bench.cli.harness.run_swebench")
    def test_harness_error_exits_nonzero(self, mock_run):
        mock_run.side_effect = ImportError("swebench not installed")
        with pytest.raises(SystemExit) as exc:
            main(["fit-bench", "run-swe", "--endpoint", "http://x"])
        assert exc.value.code != 0
