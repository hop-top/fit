"""Tests for ben suite YAML specs — structure validation."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

SUITE_DIR = Path(__file__).resolve().parent.parent / "src" / "fit" / "bench" / "suite"

REQUIRED_FIELDS = {"name", "candidates", "metrics", "scorer"}
EXPECTED_CANDIDATE_NAMES = {"baseline", "fit-steered"}

SUITE_FILES = sorted(SUITE_DIR.glob("*.yaml"))
if not SUITE_FILES:
    pytest.fail(f"No suite YAML files found in {SUITE_DIR}", pytrace=False)


@pytest.fixture(params=SUITE_FILES, ids=lambda p: p.stem)
def suite(request: pytest.FixtureRequest) -> dict:
    return yaml.safe_load(request.param.read_text())


class TestBenSuiteSpecs:
    def test_required_fields_present(self, suite: dict) -> None:
        missing = REQUIRED_FIELDS - suite.keys()
        assert not missing, f"missing fields: {missing}"

    def test_exactly_two_candidates(self, suite: dict) -> None:
        assert len(suite["candidates"]) == 2

    def test_candidate_names(self, suite: dict) -> None:
        names = {c["name"] for c in suite["candidates"]}
        assert names == EXPECTED_CANDIDATE_NAMES
