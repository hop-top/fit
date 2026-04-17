"""L5 multi-turn e2e tests: oasst2 converter + episode training."""
from __future__ import annotations

from collections import defaultdict

import pytest

from tests.e2e.conftest import write_jsonl
from tests.e2e.converters import oasst2_to_traces, trace_dict


# ---------------------------------------------------------------------------
# Mock oasst2 rows — two trees with interleaved user/assistant turns
# ---------------------------------------------------------------------------

_TREE_A = "tree-aaaa-1111"
_TREE_B = "tree-bbbb-2222"

_MOCK_OASST2_ROWS = [
    # Tree A: 2 assistant turns
    {
        "message_tree_id": _TREE_A,
        "message_id": "m-a1",
        "parent_text": "What is gravity?",
        "text": "Gravity is a fundamental force.",
        "role": "prompter",
        "created_date": "2023-01-01T00:00:00Z",
    },
    {
        "message_tree_id": _TREE_A,
        "message_id": "m-a2",
        "parent_text": "What is gravity?",
        "text": "Gravity is a fundamental force of attraction.",
        "role": "assistant",
        "created_date": "2023-01-01T00:01:00Z",
    },
    {
        "message_tree_id": _TREE_A,
        "message_id": "m-a3",
        "parent_text": "Can you elaborate?",
        "text": "Can you elaborate?",
        "role": "prompter",
        "created_date": "2023-01-01T00:02:00Z",
    },
    {
        "message_tree_id": _TREE_A,
        "message_id": "m-a4",
        "parent_text": "Can you elaborate?",
        "text": "It follows the inverse square law.",
        "role": "assistant",
        "created_date": "2023-01-01T00:03:00Z",
    },
    # Tree B: 1 assistant turn
    {
        "message_tree_id": _TREE_B,
        "message_id": "m-b1",
        "parent_text": "",
        "text": "Explain photosynthesis",
        "role": "prompter",
        "created_date": "2023-02-01T00:00:00Z",
    },
    {
        "message_tree_id": _TREE_B,
        "message_id": "m-b2",
        "parent_text": "Explain photosynthesis",
        "text": "Photosynthesis converts light energy to chemical energy.",
        "role": "assistant",
        "created_date": "2023-02-01T00:01:00Z",
    },
]


# ---------------------------------------------------------------------------
# T-0089 — oasst2 converter unit test (no network)
# ---------------------------------------------------------------------------


def test_oasst2_session_grouping():
    """oasst2 converter groups by tree, sorts by date, yields assistants."""
    traces = list(oasst2_to_traces(_MOCK_OASST2_ROWS))

    # Only assistant turns: tree A has 2, tree B has 1
    assert len(traces) == 3

    # All use multi-turn domain
    for t in traces:
        assert t["advice"]["domain"] == "multi-turn"
        assert t["metadata"]["source"] == "oasst2"

    # Group by session_id
    sessions: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        sessions[t["session_id"]].append(t)

    assert len(sessions) == 2  # two trees = two sessions

    # Tree A session has 2 traces
    tree_a_session = f"sess-{_TREE_A[:8]}"
    assert tree_a_session in sessions
    assert len(sessions[tree_a_session]) == 2

    # Verify sort order within tree A (earlier created_date first)
    a_traces = sessions[tree_a_session]
    assert "attraction" in a_traces[0]["frontier"]["output"]
    assert "inverse square" in a_traces[1]["frontier"]["output"]

    # Tree B session has 1 trace
    tree_b_session = f"sess-{_TREE_B[:8]}"
    assert tree_b_session in sessions
    assert len(sessions[tree_b_session]) == 1

    # Limit works
    limited = list(oasst2_to_traces(_MOCK_OASST2_ROWS, limit=2))
    assert len(limited) == 2


# ---------------------------------------------------------------------------
# T-0090 — multi-turn episode training (slow/gpu)
# ---------------------------------------------------------------------------


def _multiturn_traces(n_sessions: int = 5, turns: int = 4) -> list[dict]:
    """Generate multi-turn session traces for training tests."""
    traces = []
    for s in range(n_sessions):
        sid = f"sess-mt-{s:04d}"
        for t in range(turns):
            traces.append(
                trace_dict(
                    prompt=f"Session {s} turn {t} question",
                    domain="multi-turn",
                    advice_text="Continue the conversation helpfully",
                    frontier_output=f"Session {s} turn {t} answer",
                    reward_score=round(0.5 + t * 0.1, 2),
                    session_id=sid,
                )
            )
    return traces


@pytest.mark.slow
@pytest.mark.gpu
def test_multiturn_session_grouping_preserved(tmp_path):
    """DatasetBuilder preserves session grouping from multi-turn traces."""
    from fit.training.dataset import DatasetBuilder
    from fit.training.tracer import TraceIngester

    traces = _multiturn_traces(n_sessions=5, turns=4)
    jsonl = write_jsonl(tmp_path / "mt.jsonl", traces)

    ingester = TraceIngester()
    ingester.load_jsonl(jsonl)
    records = ingester.to_trace_records()

    ds = DatasetBuilder(records).build(group_by_session=True)
    assert len(ds) == 20

    # Verify examples are grouped by session
    prev_sid = ""
    session_boundaries = 0
    for ex in ds:
        if ex.session_id != prev_sid:
            session_boundaries += 1
            prev_sid = ex.session_id
    assert session_boundaries == 5


@pytest.mark.slow
@pytest.mark.gpu
def test_multiturn_episode_reward_shaping(tmp_path):
    """Later turns in a session have higher rewards (reward shaping)."""
    from fit.training.dataset import DatasetBuilder
    from fit.training.tracer import TraceIngester

    traces = _multiturn_traces(n_sessions=3, turns=5)
    jsonl = write_jsonl(tmp_path / "mt_reward.jsonl", traces)

    ingester = TraceIngester()
    ingester.load_jsonl(jsonl)
    records = ingester.to_trace_records()

    ds = DatasetBuilder(records).build(
        normalize_rewards=False,
        group_by_session=True,
    )

    # Group examples by session and verify reward increases within session
    sessions: dict[str, list] = defaultdict(list)
    for ex in ds:
        sessions[ex.session_id].append(ex)

    for sid, examples in sessions.items():
        rewards = [e.reward for e in examples]
        # Rewards should be non-decreasing within a session
        for i in range(1, len(rewards)):
            assert rewards[i] >= rewards[i - 1], (
                f"Session {sid}: reward dropped at turn {i}"
            )


@pytest.mark.slow
@pytest.mark.gpu
def test_multiturn_full_pipeline(tmp_path):
    """Full pipeline: ingest -> build dataset -> split -> verify."""
    from fit.training.dataset import DatasetBuilder
    from fit.training.tracer import TraceIngester

    traces = _multiturn_traces(n_sessions=8, turns=6)
    jsonl = write_jsonl(tmp_path / "mt_full.jsonl", traces)

    ingester = TraceIngester()
    ingester.load_jsonl(jsonl)
    records = ingester.to_trace_records()
    assert len(records) == 48

    ds = DatasetBuilder(records).build(
        normalize_rewards=True,
        group_by_session=True,
    )
    assert len(ds) == 48

    train, val = ds.split(val_ratio=0.1, seed=42)
    assert len(train) + len(val) == 48
    assert len(val) >= 1

    # Reward stats are populated
    stats = train.reward_stats()
    assert stats["count"] > 0
    assert 0.0 <= stats["mean"] <= 1.0
    assert stats["min"] >= 0.0
    assert stats["max"] <= 1.0
