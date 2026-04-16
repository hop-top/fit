from fit.types import Advice, Reward, Trace


def test_advice_defaults():
    a = Advice(domain="tax", steering_text="cite sources", confidence=0.9)
    assert a.constraints == []
    assert a.metadata == {}
    assert a.version == "1.0"


def test_reward_score_range():
    r = Reward(score=0.75, breakdown={"accuracy": 0.8})
    assert 0.0 <= r.score <= 1.0


def test_trace_round_trip():
    a = Advice(domain="code", steering_text="minimal patches", confidence=0.8)
    r = Reward(score=0.9, breakdown={"accuracy": 1.0})
    t = Trace(
        id="t1", session_id="s1", timestamp="2026-01-01T00:00:00Z",
        input={"prompt": "fix bug"}, advice=a,
        frontier={"model": "stub", "provider": "test", "output": "ok", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        reward=r,
    )
    assert t.advice.domain == "code"
    assert t.reward.score == 0.9
