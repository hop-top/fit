"""Smoke tests for bench-ab shell scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


class TestBenchABExecutable:
    def test_bench_ab_is_executable(self):
        path = SCRIPTS_DIR / "bench-ab.sh"
        assert path.exists(), f"missing {path}"
        assert os.access(path, os.X_OK), f"{path} not executable"

    def test_bench_ab_lite_is_executable(self):
        path = SCRIPTS_DIR / "bench-ab-lite.sh"
        assert path.exists(), f"missing {path}"
        assert os.access(path, os.X_OK), f"{path} not executable"


class TestBenchABUsage:
    def test_no_args_prints_usage(self):
        result = subprocess.run(
            [str(SCRIPTS_DIR / "bench-ab.sh")],
            capture_output=True, text=True,
        )
        assert result.returncode == 2
        assert "usage:" in result.stderr.lower()

    def test_invalid_suite_errors(self):
        result = subprocess.run(
            [str(SCRIPTS_DIR / "bench-ab.sh"), "nonexistent-suite"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "not found" in result.stderr.lower()
