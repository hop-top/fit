"""Tests for fit.bench.setup environment checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fit.bench.setup import check_environment


EXPECTED_KEYS = {
    "docker",
    "swebench",
    "tau2",
    "aider",
    "polyglot_exercises",
}

VALID_STATUSES = {"ok", "missing", "error"}


class TestCheckEnvironmentStructure:
    def test_returns_dict(self):
        result = check_environment()
        assert isinstance(result, dict)

    def test_contains_all_expected_keys(self):
        result = check_environment()
        assert set(result.keys()) == EXPECTED_KEYS

    def test_values_are_valid_statuses(self):
        result = check_environment()
        for key, val in result.items():
            assert val in VALID_STATUSES, f"{key}={val!r} not in {VALID_STATUSES}"


class TestDockerCheck:
    @patch("fit.bench.setup.subprocess.run")
    def test_docker_ok(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = check_environment()
        assert result["docker"] == "ok"

    @patch("fit.bench.setup.subprocess.run")
    def test_docker_not_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        result = check_environment()
        assert result["docker"] == "error"

    @patch("fit.bench.setup.subprocess.run", side_effect=FileNotFoundError)
    def test_docker_not_installed(self, mock_run):
        result = check_environment()
        assert result["docker"] == "missing"


class TestImportChecks:
    @patch("fit.bench.setup.importlib.import_module")
    def test_swebench_importable(self, mock_import):
        mock_import.return_value = MagicMock()
        result = check_environment()
        assert result["swebench"] == "ok"

    @patch(
        "fit.bench.setup.importlib.import_module",
        side_effect=ImportError("no module"),
    )
    def test_swebench_missing(self, mock_import):
        result = check_environment()
        assert result["swebench"] == "missing"

    @patch("fit.bench.setup.importlib.import_module")
    def test_tau2_importable(self, mock_import):
        mock_import.return_value = MagicMock()
        result = check_environment()
        assert result["tau2"] == "ok"

    @patch(
        "fit.bench.setup.importlib.import_module",
        side_effect=ImportError("no module"),
    )
    def test_tau2_missing(self, mock_import):
        result = check_environment()
        assert result["tau2"] == "missing"


class TestAiderCheck:
    @patch("fit.bench.setup.shutil.which", return_value="/usr/local/bin/aider")
    def test_aider_on_path(self, mock_which):
        result = check_environment()
        assert result["aider"] == "ok"

    @patch("fit.bench.setup.shutil.which", return_value=None)
    def test_aider_not_on_path(self, mock_which):
        result = check_environment()
        assert result["aider"] == "missing"


class TestPolyglotExercisesCheck:
    @patch("fit.bench.setup.Path.is_dir", return_value=True)
    def test_exercises_dir_exists(self, mock_isdir):
        result = check_environment()
        assert result["polyglot_exercises"] == "ok"

    @patch("fit.bench.setup.Path.is_dir", return_value=False)
    def test_exercises_dir_missing(self, mock_isdir):
        result = check_environment()
        assert result["polyglot_exercises"] == "missing"
