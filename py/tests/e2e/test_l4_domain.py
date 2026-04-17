"""L4 domain-specific e2e tests: lex_glue converter + domain-filtered training."""
from __future__ import annotations

import pytest

from tests.e2e.conftest import generate_trace, write_jsonl
from tests.e2e.converters import lex_glue_to_traces, trace_dict


# ---------------------------------------------------------------------------
# T-0087 — lex_glue converter unit test (no network)
# ---------------------------------------------------------------------------

_MOCK_LEX_GLUE_ROWS = [
    {"text": "The applicant was detained without trial.", "label": [0, 2]},
    {"text": "Freedom of expression was restricted.", "label": [1]},
    {"text": "Property rights were violated.", "label": 3},
]


def test_lex_glue_domain_consistency():
    """All traces from lex_glue converter use legal-compliance domain."""
    traces = list(lex_glue_to_traces(_MOCK_LEX_GLUE_ROWS, task="ecthr_a"))

    assert len(traces) == 3
    for t in traces:
        assert t["advice"]["domain"] == "legal-compliance"
        assert "ecthr_a" in t["advice"]["steering_text"]
        assert t["metadata"]["source"] == "lex_glue"
        assert t["metadata"]["task"] == "ecthr_a"
        assert t["input"]["prompt"]  # non-empty
        assert t["frontier"]["output"].startswith("label=")

    # Multi-label row produces comma-separated labels
    assert "0,2" in traces[0]["frontier"]["output"]
    # Single-label (int) row produces single label
    assert traces[2]["frontier"]["output"] == "label=3"

    # Limit works
    limited = list(lex_glue_to_traces(_MOCK_LEX_GLUE_ROWS, limit=2))
    assert len(limited) == 2


# ---------------------------------------------------------------------------
# T-0088 — domain-filtered training + quality checks (slow/gpu)
# ---------------------------------------------------------------------------


def _legal_traces(n: int = 30) -> list[dict]:
    """Generate n legal-compliance traces with varying reward."""
    return [
        trace_dict(
            prompt=f"Legal question {i}",
            domain="legal-compliance",
            advice_text="Classify under ecthr_a labels",
            frontier_output=f"label={i % 5}",
            reward_score=round(0.6 + (i % 10) * 0.04, 2),
        )
        for i in range(n)
    ]


def _generic_traces(n: int = 30) -> list[dict]:
    """Generate n generic-domain traces with lower reward."""
    return [
        generate_trace(
            domain="general",
            prompt=f"Generic question {i}",
            reward_score=round(0.3 + (i % 10) * 0.02, 2),
        )
        for i in range(n)
    ]


@pytest.mark.slow
@pytest.mark.gpu
def test_domain_filtered_training(tmp_path):
    """Train on legal-compliance traces only; verify dataset filters."""
    from fit.training.dataset import DatasetBuilder
    from fit.training.tracer import TraceIngester

    traces = _legal_traces(40) + _generic_traces(20)
    jsonl = write_jsonl(tmp_path / "mixed.jsonl", traces)

    ingester = TraceIngester()
    ingester.load_jsonl(jsonl)
    filtered = ingester.filter(domain="legal-compliance")
    records = filtered.to_trace_records()

    assert len(records) == 40
    assert all(r.advice_domain == "legal-compliance" for r in records)

    ds = DatasetBuilder(records).build()
    assert len(ds) == 40
    stats = ds.reward_stats()
    assert stats["count"] == 40.0
    assert stats["mean"] > 0


@pytest.mark.slow
@pytest.mark.gpu
def test_domain_vs_generic_reward_separation(tmp_path):
    """Legal-domain traces should have higher mean reward than generic."""
    from fit.training.dataset import DatasetBuilder
    from fit.training.tracer import TraceIngester

    legal = _legal_traces(30)
    generic = _generic_traces(30)

    legal_jsonl = write_jsonl(tmp_path / "legal.jsonl", legal)
    generic_jsonl = write_jsonl(tmp_path / "generic.jsonl", generic)

    legal_ing = TraceIngester()
    legal_ing.load_jsonl(legal_jsonl)
    legal_ds = DatasetBuilder(
        legal_ing.to_trace_records()
    ).build(normalize_rewards=False)

    generic_ing = TraceIngester()
    generic_ing.load_jsonl(generic_jsonl)
    generic_ds = DatasetBuilder(
        generic_ing.to_trace_records()
    ).build(normalize_rewards=False)

    assert legal_ds.reward_stats()["mean"] > generic_ds.reward_stats()["mean"]
