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


class TestRewardFnIndexingRegression:
    """Regression: reward_fn closure must map completions to examples correctly.

    PR #29 review item 5 — the current code uses
    `enumerate(range(len(completions)))` with separate i/j indices.
    This test reproduces the exact indexing logic from grpo.py lines
    136-145 to guarantee that simplifying to `for i in range(len(completions))`
    produces identical results.

    When len(completions) > len(examples), completions wrap via modulo.
    """

    @staticmethod
    def _current_indexing_logic(
        completions: list[str],
        examples: list,
    ) -> list:
        """Reproduce the exact logic from grpo.py:138-144.

        Current code:
            for i, j in enumerate(range(len(completions))):
                examples[i % len(examples)] ...
                completions[j] if j < len(completions) else ""
        """
        return [
            (examples[i % len(examples)], completions[j] if j < len(completions) else "")
            for i, j in enumerate(range(len(completions)))
        ]

    @staticmethod
    def _simplified_indexing_logic(
        completions: list[str],
        examples: list,
    ) -> list:
        """Proposed simplified logic: drop enumerate, use single index."""
        return [
            (examples[i % len(examples)], completions[i])
            for i in range(len(completions))
        ]

    def test_more_completions_than_examples(self) -> None:
        """3 completions, 2 examples — modulo wraps correctly."""
        examples = ["ex0", "ex1"]
        completions = ["c0", "c1", "c2"]

        current = self._current_indexing_logic(completions, examples)
        simplified = self._simplified_indexing_logic(completions, examples)

        assert current == simplified
        assert current == [
            ("ex0", "c0"),  # i=0, 0%2=0
            ("ex1", "c1"),  # i=1, 1%2=1
            ("ex0", "c2"),  # i=2, 2%2=0 (wraps)
        ]

    def test_equal_completions_and_examples(self) -> None:
        """Same count — straightforward 1:1 mapping."""
        examples = ["ex0", "ex1"]
        completions = ["c0", "c1"]

        current = self._current_indexing_logic(completions, examples)
        simplified = self._simplified_indexing_logic(completions, examples)

        assert current == simplified
        assert current == [("ex0", "c0"), ("ex1", "c1")]

    def test_fewer_completions_than_examples(self) -> None:
        """1 completion, 2 examples — only first example used."""
        examples = ["ex0", "ex1"]
        completions = ["c0"]

        current = self._current_indexing_logic(completions, examples)
        simplified = self._simplified_indexing_logic(completions, examples)

        assert current == simplified
        assert current == [("ex0", "c0")]

    def test_single_example_many_completions(self) -> None:
        """All completions map to same example (modulo 1 always = 0)."""
        examples = ["ex0"]
        completions = ["c0", "c1", "c2", "c3"]

        current = self._current_indexing_logic(completions, examples)
        simplified = self._simplified_indexing_logic(completions, examples)

        assert current == simplified
        assert all(pair[0] == "ex0" for pair in current)

    def test_dead_code_bounds_check(self) -> None:
        """j is always < len(completions) — the guard `if j < len(completions)`
        never triggers. Verify simplified version (no guard) is equivalent."""
        examples = ["ex0", "ex1"]
        completions = ["c0", "c1", "c2"]

        # j = range(len(completions)) so j is always 0..len-1
        # the `if j < len(completions)` check is dead code
        for i, j in enumerate(range(len(completions))):
            assert j < len(completions), (
                f"j={j} should always be < {len(completions)} — "
                "the bounds check is dead code"
            )


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


class TestPR30ReviewRegressions:
    """Regression tests for PR #30 review items.

    Item 1: Unused TrainingExample import in grpo.py (line 17).
    Item 2: Unconditional numpy import in _train_simplified (line 189).
    """

    def test_grpo_does_not_reexport_training_example(self) -> None:
        """TrainingExample must not be accessible via fit.training.grpo.

        PR #30 review item 1: grpo.py imports TrainingExample from
        .dataset but never uses it (F401 lint). After the fix removes
        that import, accessing TrainingExample through the grpo module
        should fail.
        """
        import fit.training.grpo as grpo_mod

        assert not hasattr(grpo_mod, "TrainingExample"), (
            "TrainingExample leaked into fit.training.grpo namespace -- "
            "remove the unused import from grpo.py"
        )

    def test_simplified_training_no_numpy_import(self) -> None:
        """_train_simplified must not import numpy.

        PR #30 review item 2: _train_simplified had `import numpy as np`
        but numpy is not a declared dependency. Reading the source
        directly avoids needing torch in the test environment.
        """
        import inspect

        from fit.training.grpo import GRPOTrainer

        source = inspect.getsource(GRPOTrainer._train_simplified)

        assert "import numpy" not in source, (
            "_train_simplified still imports numpy -- "
            "remove the unused numpy dependency"
        )
        assert "import numpy as np" not in source, (
            "_train_simplified still imports numpy as np -- "
            "remove the unused numpy dependency"
        )
