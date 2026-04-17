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


class TestPR31ImportErrorMsgRegression:
    """PR #31 review — ImportError catch in train() must include the
    actual exception in the log message so users can diagnose which
    dependency is missing (trl, torch, or transformers).
    """

    def test_import_error_log_includes_exception_info(self) -> None:
        """The except block must log the exception variable, not a
        hardcoded 'trl not available' message.
        """
        import inspect

        from fit.training.grpo import GRPOTrainer

        source = inspect.getsource(GRPOTrainer.train)

        # The except block must capture the exception (e.g. `except ImportError as exc`)
        # and pass it to the logger call.
        assert "except ImportError as exc" in source, (
            "train() must capture ImportError as `exc` (or similar)"
        )
        # Verify exc is referenced after the except line
        except_idx = source.index("except ImportError as exc")
        after_except = source[except_idx:]
        # Find the logger call — it must reference exc
        logger_start = after_except.find("logger.")
        assert logger_start >= 0, "No logger call in except block"
        # Grab from logger call to the next `return` or method-level statement
        logger_section = after_except[logger_start:]
        assert "exc" in logger_section.split("return")[0], (
            "The logger call in the except block must include the "
            "exception variable so the actual missing dep is diagnosable."
        )


class TestPR34TrainSimplifiedDocstringAccuracyRegression:
    """Regression: _train_simplified docstring must not claim features
    that aren't implemented in the method body.

    PR #34 review item 2 — the docstring (grpo.py:184-189) says
    "KL penalty via beta" and "PPO-style clip range" but neither
    cfg.beta nor cfg.clip_range is referenced in the method body.
    The docstring is misleading. Fix must either implement those
    features or remove the claims.

    This test is marked xfail(strict=True): it PASSES once the
    docstring is corrected to match reality.
    """

    @pytest.mark.xfail(strict=True)
    def test_docstring_does_not_mention_unimplemented_features(self) -> None:
        """Docstring must not claim beta/clip_range if body doesn't use them."""
        import inspect

        from fit.training.grpo import GRPOTrainer

        source = inspect.getsource(GRPOTrainer._train_simplified)

        # Extract docstring
        docstring = GRPOTrainer._train_simplified.__doc__ or ""
        docstring_lower = docstring.lower()

        # Check method body (everything after the docstring)
        # Find end of docstring
        body = source
        if '"""' in body:
            parts = body.split('"""')
            if len(parts) >= 3:
                body = '"""'.join(parts[2:])  # everything after closing """

        body_lower = body.lower()

        # If docstring mentions beta, body must reference cfg.beta
        if "beta" in docstring_lower:
            assert "cfg.beta" in body or "self._config.beta" in body, (
                "Docstring mentions 'beta' but method body never uses "
                "cfg.beta — docstring is misleading"
            )

        # If docstring mentions clip range, body must reference cfg.clip_range
        if "clip" in docstring_lower or "clip range" in docstring_lower:
            assert "cfg.clip_range" in body or "self._config.clip_range" in body, (
                "Docstring mentions 'clip range' but method body never uses "
                "cfg.clip_range — docstring is misleading"
            )


class TestPR34RewardFnIgnoredInSimplifiedRegression:
    """Regression: _train_simplified must warn when reward_fn is provided
    but silently ignored.

    PR #34 review item 3 — when simplified mode runs (ImportError fallback),
    any user-supplied reward_fn is never consulted. Rewards always come
    from TrainingExample.reward. A warning should be logged so the user
    knows their custom reward function is being ignored.

    This test is marked xfail(strict=True): it PASSES once the warning
    is added to _train_simplified.
    """

    @pytest.mark.xfail(strict=True)
    def test_warning_logged_when_reward_fn_ignored(self) -> None:
        """Simplified mode must log a warning when reward_fn is provided."""
        import logging
        from unittest.mock import MagicMock, patch

        # Create a mock reward function
        mock_reward_fn = MagicMock(return_value=0.5)

        cfg = GRPOConfig(epochs=1, batch_size=2, use_trl=False)
        trainer = GRPOTrainer(cfg, reward_fn=mock_reward_fn)

        dataset = _make_dataset(4)

        with (
            patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "transformers": MagicMock(),
                    "transformers.AutoModelForCausalLM": MagicMock(),
                    "transformers.AutoTokenizer": MagicMock(),
                },
            ),
            patch.object(
                GRPOTrainer, "_train_simplified", wraps=trainer._train_simplified
            ) as _spy,
        ):
            # We can't easily run _train_simplified without real torch.
            # Instead, read the source and check for the warning pattern.
            import inspect

            source = inspect.getsource(GRPOTrainer._train_simplified)
            source_lower = source.lower()

            assert (
                "reward_fn" in source_lower
                and "warn" in source_lower
            ), (
                "_train_simplified must log a warning when self._reward_fn "
                "is provided but simplified mode is used"
            )


class TestPR34EpochLossesMisleadingRegression:
    """Regression: epoch_losses key must be renamed and must contain all
    batch losses, not a sliced subset.

    PR #34 review item 4 — in grpo.py:305, training_metadata['epoch_losses']
    is sliced by cfg.epochs but the list contains per-batch losses.
    The name implies per-epoch but contains per-batch, and the slice
    discards data when batches > epochs.

    This test is marked xfail(strict=True): it PASSES once the key is
    renamed to "batch_losses" and contains ALL batch losses.
    """

    @pytest.mark.xfail(strict=True)
    def test_losses_key_named_batch_losses_and_complete(self) -> None:
        """training_metadata must use 'batch_losses' (not 'epoch_losses')
        and contain ALL per-batch loss values."""
        import inspect
        import re

        from fit.training.grpo import GRPOTrainer

        source = inspect.getsource(GRPOTrainer._train_simplified)

        # The key should be "batch_losses", not "epoch_losses"
        assert '"batch_losses"' in source or "'batch_losses'" in source, (
            "training_metadata key must be 'batch_losses', not 'epoch_losses'"
        )

        # The slice `[-cfg.epochs:]` must not appear for the losses key
        # (it discards batch losses when batches > epochs)
        assert "[-cfg.epochs" not in source or '"epoch_losses"' not in source, (
            "epoch_losses must not be sliced by cfg.epochs — "
            "use batch_losses without slicing"
        )
