"""Regression tests for RemoteAdvisor version field bug.

Bug: generate_advice() ignored the optional "version" field from the
server response, always using the Advice default "1.0". The fix passes
data.get("version", "1.0") into the Advice constructor.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fit.advisor import RemoteAdvisor
from fit.types import Advice


def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def test_version_preserved_from_server_response():
    """Regression: version field from server must be passed to Advice.

    Before fix: generate_advice() never read data["version"], so Advice
    always got the default "1.0" even when the server returned a different
    version string.
    """
    server_data = {
        "domain": "math",
        "steering_text": "Check your arithmetic.",
        "confidence": 0.92,
        "constraints": ["be precise"],
        "metadata": {"source": "v2-model"},
        "version": "2.3",
    }

    advisor = RemoteAdvisor(endpoint="http://localhost:9999")
    mock_resp = _make_mock_response(server_data)

    with patch.object(advisor, "_endpoint", "http://localhost:9999"):
        # Patch the httpx module that generate_advice imports locally
        import fit.advisor as advisor_mod

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            advice = advisor.generate_advice({"prompt": "2+2"})

    assert advice.version == "2.3", (
        f"version should be '2.3' from server, got '{advice.version}'"
    )


def test_version_defaults_to_1_when_missing():
    """Regression: missing version field must default to '1.0'.

    Before fix: even though the default was "1.0", the code didn't
    pass version at all, relying on the dataclass default. This worked
    for the default case but the explicit pass-through ensures
    consistency with the protocol.
    """
    server_data = {
        "domain": "math",
        "steering_text": "Think step by step.",
        "confidence": 0.85,
        # no "version" key
    }

    advisor = RemoteAdvisor(endpoint="http://localhost:9999")
    mock_resp = _make_mock_response(server_data)

    mock_httpx = MagicMock()
    mock_httpx.post.return_value = mock_resp

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        advice = advisor.generate_advice({"prompt": "solve x^2=4"})

    assert advice.version == "1.0", (
        f"version should default to '1.0', got '{advice.version}'"
    )


def test_version_with_numeric_value():
    """Server may return version as a number; ensure it's handled."""
    server_data = {
        "domain": "code",
        "steering_text": "Use type hints.",
        "confidence": 0.7,
        "version": 3,
    }

    advisor = RemoteAdvisor(endpoint="http://localhost:9999")
    mock_resp = _make_mock_response(server_data)

    mock_httpx = MagicMock()
    mock_httpx.post.return_value = mock_resp

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        advice = advisor.generate_advice({"prompt": "write a function"})

    # version from server must be coerced to str for type conformance
    assert advice.version == "3", (
        f"version should be '3' (str), got '{advice.version}'"
    )
    assert isinstance(advice.version, str), (
        f"version must be str, got {type(advice.version).__name__}"
    )


def test_all_fields_preserved():
    """Ensure the fix doesn't break other fields."""
    server_data = {
        "domain": "reasoning",
        "steering_text": "Break it down.",
        "confidence": 0.88,
        "constraints": ["step-by-step", "verify"],
        "metadata": {"model": "gpt-5"},
        "version": "1.5",
    }

    advisor = RemoteAdvisor(endpoint="http://localhost:9999")
    mock_resp = _make_mock_response(server_data)

    mock_httpx = MagicMock()
    mock_httpx.post.return_value = mock_resp

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        advice = advisor.generate_advice({"prompt": "test"})

    assert advice.domain == "reasoning"
    assert advice.steering_text == "Break it down."
    assert advice.confidence == pytest.approx(0.88)
    assert advice.constraints == ["step-by-step", "verify"]
    assert advice.metadata == {"model": "gpt-5"}
    assert advice.version == "1.5"
