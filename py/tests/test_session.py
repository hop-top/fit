import importlib

import pytest

from fit.session import Session, SessionConfig
from fit.advisor import RemoteAdvisor
from fit.reward import CompositeScorer
from fit.adapters.openai import OpenAIAdapter

_has_openai = importlib.util.find_spec("openai") is not None


@pytest.mark.skipif(not _has_openai, reason="openai not installed")
def test_session_one_shot():
    advisor = RemoteAdvisor(endpoint="http://localhost:9999")
    scorer = CompositeScorer.composite(["accuracy", "relevance"])
    adapter = OpenAIAdapter()
    session = Session(advisor=advisor, adapter=adapter, scorer=scorer)
    # Session.run will fail on advisor HTTP call — test config only
    assert session._config.mode == "one-shot"


def test_session_config_defaults():
    cfg = SessionConfig()
    assert cfg.mode == "one-shot"
    assert cfg.max_steps == 10
