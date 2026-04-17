"""Tests for fit.training.dataset — trace to training example conversion."""
from __future__ import annotations

from pathlib import Path

import pytest

from fit.training.dataset import DatasetBuilder, FitDataset, TrainingExample
from fit.training.tracer import TraceRecord

FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "spec" / "fixtures"


def _record(
    id: str = "r1",
    session_id: str = "sess1",
    domain: str = "tax",
    reward: float | None = 0.9,
) -> TraceRecord:
    return TraceRecord(
        id=id,
        session_id=session_id,
        timestamp="2026-04-15T12:00:00Z",
        prompt="What is X?",
        context={"k": "v"},
        advice_text="Be concise",
        advice_domain=domain,
        advice_confidence=0.8,
        frontier_output="X is Y",
        frontier_model="test-model",
        reward_score=reward,
        reward_breakdown={"accuracy": reward or 0.0},
        metadata={},
    )


class TestTrainingExample:
    def test_fields(self) -> None:
        ex = TrainingExample(context="ctx", advice="adv", reward=0.9)
        assert ex.context == "ctx"
        assert ex.advice == "adv"
        assert ex.reward == 0.9
        assert ex.session_id == ""
        assert ex.metadata == {}


class TestFitDataset:
    def test_len_and_getitem(self) -> None:
        ds = FitDataset([TrainingExample("a", "b", 0.5), TrainingExample("c", "d", 0.8)])
        assert len(ds) == 2
        assert ds[0].context == "a"
        assert ds[1].reward == 0.8

    def test_iter(self) -> None:
        ds = FitDataset([TrainingExample("x", "y", 1.0)])
        assert [e.context for e in ds] == ["x"]

    def test_split_deterministic(self) -> None:
        examples = [TrainingExample(f"ctx{i}", f"adv{i}", float(i) / 10) for i in range(20)]
        train1, val1 = FitDataset(examples).split(val_ratio=0.2, seed=42)
        train2, val2 = FitDataset(examples).split(val_ratio=0.2, seed=42)
        assert len(train1) == len(train2)
        assert [e.context for e in train1] == [e.context for e in train2]

    def test_split_ratios(self) -> None:
        examples = [TrainingExample(f"ctx{i}", f"adv{i}", 0.5) for i in range(100)]
        train, val = FitDataset(examples).split(val_ratio=0.1, seed=42)
        assert len(val) == 10
        assert len(train) == 90

    def test_reward_stats(self) -> None:
        ds = FitDataset([
            TrainingExample("a", "b", 0.2),
            TrainingExample("c", "d", 0.8),
        ])
        stats = ds.reward_stats()
        assert stats["min"] == 0.2
        assert stats["max"] == 0.8
        assert stats["mean"] == pytest.approx(0.5)

    def test_reward_stats_empty(self) -> None:
        assert FitDataset([]).reward_stats()["mean"] == 0.0

    def test_reward_stats_empty_has_count_key(self) -> None:
        """Regression: empty dataset reward_stats() must include 'count' key.

        When empty, reward_stats() returns {"mean":0,"std":0,"min":0,"max":0}
        but omits "count" — inconsistent with the non-empty path which
        includes "count": float(n).
        """
        stats = FitDataset([]).reward_stats()
        assert "count" in stats, (
            f"reward_stats() for empty dataset missing 'count' key: {stats}"
        )
        assert stats["count"] == 0.0

    def test_reward_stats_nonempty_has_count_key(self) -> None:
        """Non-empty dataset must also include 'count' key (sanity check)."""
        ds = FitDataset([TrainingExample("a", "b", 0.5)])
        stats = ds.reward_stats()
        assert "count" in stats
        assert stats["count"] == 1.0

    def test_examples_property_is_copy(self) -> None:
        ds = FitDataset([TrainingExample("a", "b", 0.5)])
        ds.examples.append(TrainingExample("c", "d", 0.6))
        assert len(ds) == 1  # original unchanged


class TestDatasetBuilder:
    def test_build_basic(self) -> None:
        records = [_record(reward=0.9), _record(reward=0.5, domain="legal")]
        ds = DatasetBuilder(records).build(normalize_rewards=False)
        assert len(ds) == 2
        assert ds[0].reward == 0.9

    def test_skips_null_rewards(self) -> None:
        records = [_record(reward=0.9), _record(reward=None)]
        ds = DatasetBuilder(records).build()
        assert len(ds) == 1

    def test_normalize_rewards(self) -> None:
        records = [_record(reward=0.2), _record(reward=0.8)]
        ds = DatasetBuilder(records).build(normalize_rewards=True)
        rewards = [e.reward for e in ds]
        assert min(rewards) == pytest.approx(0.0)
        assert max(rewards) == pytest.approx(1.0)

    def test_normalize_constant_rewards(self) -> None:
        records = [_record(reward=0.5), _record(reward=0.5)]
        ds = DatasetBuilder(records).build(normalize_rewards=True)
        assert all(e.reward == 0.5 for e in ds)

    def test_group_by_session(self) -> None:
        records = [
            _record(id="a", session_id="sess2"),
            _record(id="b", session_id="sess1"),
        ]
        ds = DatasetBuilder(records).build(group_by_session=True)
        _ = [e.metadata.get("domain") for e in ds]  # just check ordering
        # Should be sorted by session_id
        sessions = [e.session_id for e in ds]
        assert sessions == sorted(sessions)

    def test_context_formatting(self) -> None:
        rec = _record()
        ds = DatasetBuilder([rec]).build()
        ctx = ds[0].context
        assert "Domain: tax" in ctx
        assert "Prompt: What is X?" in ctx
        assert "Output: X is Y" in ctx

    def test_empty_records(self) -> None:
        ds = DatasetBuilder([]).build()
        assert len(ds) == 0

    def test_from_fixture(self) -> None:
        from fit.training.tracer import TraceIngester

        ingester = TraceIngester().load_batch([FIXTURE_DIR])
        records = ingester.to_trace_records()
        ds = DatasetBuilder(records).build()
        assert len(ds) >= 1
        for ex in ds:
            assert ex.context
            assert isinstance(ex.reward, float)


class TestSplitValRatioZeroRegression:
    """Regression: FitDataset.split(val_ratio=0.0) must produce no val set.

    split() uses
        val_count = max(1, int(len(indices) * val_ratio))
    which means val_ratio=0.0 still allocates 1 validation example
    for any dataset with len >= 2. This violates expected semantics:
    val_ratio=0.0 should mean "no validation split".
    These tests prove the bug (they FAIL on current code).
    """

    def test_val_ratio_zero_produces_no_val_set(self) -> None:
        """val_ratio=0.0 must yield len(val) == 0.

        Current code gives len(val) == 1 due to max(1, ...)
        on line 37 of dataset.py.
        """
        examples = [TrainingExample(f"ctx{i}", f"adv{i}", float(i) / 10) for i in range(10)]
        train, val = FitDataset(examples).split(val_ratio=0.0)
        assert len(val) == 0, (
            f"Expected 0 val examples with val_ratio=0.0, got {len(val)} — "
            "max(1, ...) in split() forces >=1 val example"
        )

    def test_val_ratio_zero_all_in_train(self) -> None:
        """val_ratio=0.0 must keep all 10 examples in train.

        Current code gives len(train) == 9 because 1 goes to val.
        """
        examples = [TrainingExample(f"ctx{i}", f"adv{i}", float(i) / 10) for i in range(10)]
        train, val = FitDataset(examples).split(val_ratio=0.0)
        assert len(train) == 10, (
            f"Expected 10 train examples with val_ratio=0.0, got {len(train)} — "
            "max(1, ...) steals 1 example for val"
        )


class TestSplitValRatioValidationRegression:
    """Regression: FitDataset.split() must validate val_ratio bounds.

    split() does not validate val_ratio.
    Negative ratios or values >1 produce nonsensical splits.
    Fix should raise ValueError for out-of-range ratios.
    """

    def test_negative_val_ratio_raises_value_error(self) -> None:
        """split(val_ratio=-0.1) must raise ValueError."""
        examples = [
            TrainingExample(f"ctx{i}", f"adv{i}", 0.5) for i in range(10)
        ]
        with pytest.raises(ValueError, match="val_ratio"):
            FitDataset(examples).split(val_ratio=-0.1)

    def test_val_ratio_above_one_raises_value_error(self) -> None:
        """split(val_ratio=1.5) must raise ValueError."""
        examples = [
            TrainingExample(f"ctx{i}", f"adv{i}", 0.5) for i in range(10)
        ]
        with pytest.raises(ValueError, match="val_ratio"):
            FitDataset(examples).split(val_ratio=1.5)


class TestSplitValRatioOneRegression:
    """Regression: FitDataset.split(val_ratio=1.0) is allowed but silently
    clamped, producing an empty training set.

    val_ratio=1.0 means "put everything in validation,
    nothing in train", which is never useful and likely a caller mistake.
    split() should raise ValueError for val_ratio >= 1.0 (exclusive upper
    bound).  Currently the value is silently accepted and clamped.
    """

    def test_val_ratio_one_raises_value_error(self) -> None:
        """split(val_ratio=1.0) must raise ValueError."""
        examples = [
            TrainingExample(f"ctx{i}", f"adv{i}", 0.5) for i in range(5)
        ]
        with pytest.raises(ValueError, match="val_ratio"):
            FitDataset(examples).split(val_ratio=1.0)


# ---------------------------------------------------------------------------
# Regression: split doesn't steal examples for val
# ---------------------------------------------------------------------------


class TestSplitDoesNotStealForVal:
    """val_count = max(1, int(len * val_ratio)) means len=1 => val=1,
    train=0. The split must not produce an empty train set."""

    def test_single_example_split(self) -> None:
        """Dataset with 1 example should not produce empty train."""
        examples = [
            TrainingExample(
                context="ctx", advice="adv", reward=0.5
            )
        ]
        ds = FitDataset(examples)
        train, val = ds.split(val_ratio=0.1)

        assert len(train) >= 1, (
            "split() produces empty train set "
            "for single-example dataset. "
            f"train={len(train)}, val={len(val)}"
        )

    def test_two_example_split_both_nonempty(self) -> None:
        """Dataset with 2 examples must have both train and val
        non-empty."""
        examples = [
            TrainingExample(
                context=f"ctx{i}", advice=f"adv{i}", reward=0.5
            )
            for i in range(2)
        ]
        ds = FitDataset(examples)
        train, val = ds.split(val_ratio=0.1)

        assert len(val) > 0, "Val set must not be empty"
        assert len(train) > 0, "Train set must not be empty"

    def test_split_preserves_total(self) -> None:
        """train + val must equal original dataset size."""
        examples = [
            TrainingExample(
                context=f"ctx{i}",
                advice=f"adv{i}",
                reward=float(i),
            )
            for i in range(5)
        ]
        ds = FitDataset(examples)
        train, val = ds.split(val_ratio=0.2)
        assert len(train) + len(val) == len(ds)
