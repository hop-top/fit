"""Tests for fit.training.grpo — GRPO trainer config and basic operations."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fit.training.dataset import FitDataset, TrainingExample
from fit.training.grpo import GRPOConfig, GRPOTrainer, TrainingResult, _compute_reward_stats


def _make_dataset(n: int = 10) -> FitDataset:
    return FitDataset([
        TrainingExample(
            context=f"context {i}",
            advice=f"advice {i}",
            reward=0.5 + i * 0.05,
            session_id="sess1",
        )
        for i in range(n)
    ])


class TestGRPOConfig:
    def test_defaults(self) -> None:
        cfg = GRPOConfig()
        assert cfg.base_model == "Qwen/Qwen2-0.5B"
        assert cfg.learning_rate == 1e-5
        assert cfg.epochs == 3
        assert cfg.batch_size == 8
        assert cfg.reward_shaping == "linear"

    def test_custom(self) -> None:
        cfg = GRPOConfig(base_model="test/model", epochs=1)
        assert cfg.base_model == "test/model"
        assert cfg.epochs == 1


class TestTrainingResult:
    def test_fields(self) -> None:
        result = TrainingResult(
            model_path="/tmp/out",
            epochs_completed=3,
            final_loss=0.5,
            reward_stats={"mean": 0.8},
        )
        assert result.model_path == "/tmp/out"
        assert result.epochs_completed == 3
        assert result.final_loss == 0.5
        assert result.reward_stats["mean"] == 0.8

    def test_metadata_default(self) -> None:
        result = TrainingResult(
            model_path="/out",
            epochs_completed=1,
            final_loss=0.0,
            reward_stats={},
        )
        assert result.training_metadata == {}


class TestGRPOTrainer:
    def test_init(self) -> None:
        trainer = GRPOTrainer(GRPOConfig())
        assert trainer._config.epochs == 3

    def test_train_empty_dataset_raises(self) -> None:
        trainer = GRPOTrainer(GRPOConfig())
        with pytest.raises(ValueError, match="empty"):
            trainer.train(FitDataset([]))

    def test_save_creates_config(self, tmp_path: Path) -> None:
        trainer = GRPOTrainer(GRPOConfig(output_dir=str(tmp_path / "model")))
        trainer.save(str(tmp_path / "model"))

        config_path = tmp_path / "model" / "training_config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["base_model"] == "Qwen/Qwen2-0.5B"
        assert data["epochs"] == 3

    def test_reward_shaping_linear(self) -> None:
        trainer = GRPOTrainer(GRPOConfig(reward_shaping="linear"))
        assert trainer._shape_reward(0.7) == 0.7

    def test_reward_shaping_exponential(self) -> None:
        import math

        trainer = GRPOTrainer(GRPOConfig(reward_shaping="exponential"))
        shaped = trainer._shape_reward(1.0)
        assert shaped == pytest.approx(1.0)

    def test_reward_shaping_clipped(self) -> None:
        trainer = GRPOTrainer(GRPOConfig(reward_shaping="clipped"))
        assert trainer._shape_reward(1.5) == 1.0
        assert trainer._shape_reward(-0.5) == 0.0
        assert trainer._shape_reward(0.5) == 0.5


class TestComputeRewardStats:
    def test_basic(self) -> None:
        stats = _compute_reward_stats([0.2, 0.5, 0.8])
        assert stats["mean"] == pytest.approx(0.5)
        assert stats["min"] == 0.2
        assert stats["max"] == 0.8
        assert stats["count"] == 3.0

    def test_empty(self) -> None:
        stats = _compute_reward_stats([])
        assert stats["mean"] == 0.0
        assert stats["count"] == 0.0

    def test_single(self) -> None:
        stats = _compute_reward_stats([0.5])
        assert stats["mean"] == 0.5
        assert stats["std"] == 0.0
