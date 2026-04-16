"""Regression tests for PR #26 code review bugs.

These tests prove the bugs exist in the current codebase. They should
FAIL until the corresponding fixes are applied.
"""
from __future__ import annotations

import statistics
from unittest.mock import MagicMock, patch

import pytest

from fit.types import Advice
from fit.training.dataset import FitDataset, TrainingExample
from fit.training.reward_fn import LLMJudgeReward


# ---------------------------------------------------------------------------
# Bug 1: LLMJudgeReward calls adapter.call(prompt) without Advice argument
# File: reward_fn.py, line ~87
# ---------------------------------------------------------------------------


class TestLLMJudgeRewardMissingAdvice:
    """adapter.call signature is call(prompt, advice) but __call__ only
    passes prompt. TypeError is swallowed, always returns 0.5."""

    def test_call_receives_two_args(self) -> None:
        """Adapter.call must receive (prompt, Advice), not just prompt."""
        judge = LLMJudgeReward()
        mock_adapter = MagicMock()
        mock_adapter.call.return_value = ("Score: 0.9", {"model": "test"})

        with patch(
            "fit.adapters.anthropic.AnthropicAdapter",
            return_value=mock_adapter,
        ):
            score = judge("ctx", "some advice", "output text")

        mock_adapter.call.assert_called_once()
        call_args = mock_adapter.call.call_args

        # Bug: code passes 1 arg (prompt only). Fix requires 2 args.
        if len(call_args.args) == 1 and not call_args.kwargs:
            pytest.fail(
                "Bug confirmed: adapter.call received 1 positional arg "
                f"({call_args.args[0][:60]!r}...) but needs "
                "(prompt, Advice). "
                "LLMJudgeReward swallows the TypeError and returns 0.5."
            )

        # After fix: should have 2 args, second is an Advice instance
        assert len(call_args.args) >= 2
        assert isinstance(call_args.args[1], Advice)

    def test_returns_real_score_not_fallback(self) -> None:
        """When adapter works, judge must return the parsed score, not 0.5.

        The real AnthropicAdapter.call has a 2-arg signature
        (prompt, advice). MagicMock accepts any args, so we use
        a spec-enforcing mock to reproduce the actual TypeError.
        """
        judge = LLMJudgeReward()

        # Use a spec that enforces the real 2-arg signature
        real_module = __import__(
            "fit.adapters.anthropic",
            fromlist=["AnthropicAdapter"],
        )
        RealClass = real_module.AnthropicAdapter

        # Create a spec'd mock that enforces the real call signature
        mock_adapter = MagicMock(spec=RealClass)
        mock_adapter.call.return_value = ("Score: 0.9", {"model": "test"})

        with patch(
            "fit.adapters.anthropic.AnthropicAdapter",
            return_value=mock_adapter,
        ):
            score = judge("ctx", "advice", "output")

        # Bug: code calls adapter.call(prompt) with 1 arg.
        # MagicMock without spec silently accepts this (no TypeError).
        # With spec, the real __call__ would enforce 2 args.
        # Either way the test proves: when call() is invoked,
        # it must receive Advice as second arg for the judge to
        # return a real score instead of falling back to 0.5.
        #
        # Without spec enforcement, the mock silently succeeds
        # and returns ("Score: 0.9", ...), so score parses to 0.9.
        # In production, the real adapter raises TypeError -> caught -> 0.5.
        #
        # We verify the call count and arity:
        mock_adapter.call.assert_called_once()
        call_args = mock_adapter.call.call_args
        assert len(call_args.args) >= 2, (
            "Bug confirmed: adapter.call called with "
            f"{len(call_args.args)} arg(s) but needs 2 "
            "(prompt, Advice). In production this raises TypeError, "
            "caught by except, returns fallback 0.5."
        )


# ---------------------------------------------------------------------------
# Bug 2: GRPO TRL reward_fn passes completions[j] for both advice and output
# File: grpo.py, line ~141-142
# ---------------------------------------------------------------------------


class TestGRPORewardFnAdviceArg:
    """reward_fn lambda passes completions[j] as advice instead of
    the example's actual advice text."""

    def test_advice_arg_is_example_advice_not_completion(self) -> None:
        """The advice parameter to _reward_fn must be the example's advice,
        not the generated completion."""
        from fit.training.grpo import GRPOTrainer, GRPOConfig

        recorded_calls: list[tuple[str, str, str]] = []

        class RecordingReward:
            """Records all (context, advice, output) calls."""

            def __call__(
                self, context: str, advice: str, output: str
            ) -> float:
                recorded_calls.append((context, advice, output))
                return 0.7

        example = TrainingExample(
            context="test context",
            advice="real advice from dataset",
            reward=0.5,
        )
        dataset = FitDataset([example])

        config = GRPOConfig(use_trl=True)
        trainer = GRPOTrainer(config=config, reward_fn=RecordingReward())

        # Reproduce the exact lambda logic from grpo.py:136-145
        examples = dataset.examples
        completions = ["generated completion text"]

        def reward_fn_lambda(
            completions: list[str], **kwargs: object
        ) -> list[float]:
            if trainer._reward_fn:
                return [
                    trainer._reward_fn(
                        examples[i % len(examples)].context,
                        completions[j] if j < len(completions) else "",
                        completions[j] if j < len(completions) else "",
                    )
                    for i, j in enumerate(range(len(completions)))
                ]
            return [0.5]

        reward_fn_lambda(completions)

        assert len(recorded_calls) == 1
        ctx, adv, out = recorded_calls[0]

        # Bug: advice arg equals the completion, not the example's advice
        if adv == completions[0]:
            pytest.fail(
                "Bug confirmed: advice arg is the completion "
                f"{adv!r}, not the example's advice "
                f"{examples[0].advice!r}. "
                "Line 141-142 passes completions[j] twice."
            )

        assert adv == "real advice from dataset"


# ---------------------------------------------------------------------------
# Bug 3: ONNX export input_names mismatch with attention_mask
# File: export.py, line ~121-127
# ---------------------------------------------------------------------------


class TestOnnxExportInputNamesMismatch:
    """_onnx_export_torch passes (input_ids, attention_mask) as inputs
    but input_names=["input_ids"] only. torch.onnx.export requires
    len(input_names) == len(inputs)."""

    def test_input_names_count_matches_inputs(self) -> None:
        """input_names must have as many entries as model inputs."""
        from fit.training.export import ModelExporter

        exporter = ModelExporter("/tmp/nonexistent-model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Simulate tokenizer returning both input_ids and attention_mask
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        captured_export: dict = {}

        def capture_export(*args: object, **kwargs: object) -> None:
            captured_export.update(kwargs)

        # Build mock torch and transformers modules
        mock_torch = MagicMock()
        mock_torch.onnx.export = capture_export
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = (
            mock_model
        )
        mock_transformers.AutoTokenizer.from_pretrained.return_value = (
            mock_tokenizer
        )

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            try:
                exporter._onnx_export_torch(
                    exporter._model_path / "model.onnx"
                )
            except TypeError:
                pass

        input_names = captured_export.get("input_names", [])

        # Bug: input_names=["input_ids"] (length 1) but 2 inputs are
        # passed: (input_ids, attention_mask)
        if len(input_names) == 1:
            pytest.fail(
                "Bug confirmed: input_names has 1 entry "
                f"({input_names}) but 2 inputs are passed "
                "(input_ids + attention_mask). "
                "torch.onnx.export requires matching lengths."
            )

        assert len(input_names) == 2


# ---------------------------------------------------------------------------
# Bug 4: FitDataset.split() produces empty train set for small datasets
# File: dataset.py, line ~37
# ---------------------------------------------------------------------------


class TestFitDatasetSplitEmptyTrain:
    """val_count = max(1, int(len * val_ratio)) means len=1 => val=1,
    train=0. len=2, val_ratio=0.1 => val=1, train=1 (ok).
    But len=1 always results in empty train."""

    def test_single_example_split(self) -> None:
        """Dataset with 1 example should not produce empty train set."""
        examples = [
            TrainingExample(
                context="ctx", advice="adv", reward=0.5
            )
        ]
        ds = FitDataset(examples)
        train, val = ds.split(val_ratio=0.1)

        # With 1 example: val_count = max(1, int(1*0.1)) = max(1,0) = 1
        # train gets 0 examples, val gets 1.
        assert len(train) >= 1, (
            "Bug confirmed: split() produces empty train set "
            "for single-example dataset. "
            f"train={len(train)}, val={len(val)}"
        )

    def test_two_example_split_both_nonempty(self) -> None:
        """Dataset with 2 examples must have both train and val non-empty."""
        examples = [
            TrainingExample(context=f"ctx{i}", advice=f"adv{i}", reward=0.5)
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
                context=f"ctx{i}", advice=f"adv{i}", reward=float(i)
            )
            for i in range(5)
        ]
        ds = FitDataset(examples)
        train, val = ds.split(val_ratio=0.2)
        assert len(train) + len(val) == len(ds)


# ---------------------------------------------------------------------------
# Bug 5: avg_loss wrong epoch slice
# File: grpo.py, line ~274-277
# ---------------------------------------------------------------------------


class TestAvgLossEpochSlice:
    """epoch_losses[-len(indices):] slices by number of examples, not
    number of batches. When batch_size > 1, len(indices) !=
    number of batch losses appended per epoch."""

    def test_avg_loss_uses_batch_count_not_example_count(self) -> None:
        """Simulate the avg_loss calculation from _train_simplified.

        With 10 examples, batch_size=4: 3 batches per epoch.
        epoch_losses gets 3 entries per epoch, but the slice
        [-len(indices):] uses len(indices)=10, pulling in all
        prior losses instead of just the current epoch's.
        """
        num_examples = 10
        batch_size = 4
        num_batches_per_epoch = (
            num_examples + batch_size - 1
        ) // batch_size  # 3

        # Simulate 3 epochs
        epoch_losses: list[float] = []
        for epoch in range(3):
            for b in range(num_batches_per_epoch):
                epoch_losses.append(float(epoch * 10 + b))

        # After 3 epochs: 9 losses total.
        # Last epoch losses = [20.0, 21.0, 22.0], avg=21.0

        indices = list(range(num_examples))

        # Buggy formula from grpo.py:274
        buggy_slice = epoch_losses[-len(indices) :]
        buggy_avg = statistics.mean(buggy_slice) if buggy_slice else 0.0

        # Correct: slice by number of batches in this epoch
        correct_slice = epoch_losses[-num_batches_per_epoch:]
        correct_avg = statistics.mean(correct_slice) if correct_slice else 0.0

        assert correct_avg == pytest.approx(21.0)

        if abs(buggy_avg - correct_avg) > 0.01:
            pytest.fail(
                "Bug confirmed: avg_loss uses len(indices)="
                f"{len(indices)} to slice epoch_losses, but should "
                f"use batch count={num_batches_per_epoch}. "
                f"buggy_avg={buggy_avg:.2f} != "
                f"correct_avg={correct_avg:.2f}"
            )
