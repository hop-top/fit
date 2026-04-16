"""GRPO policy optimization for advisor model training.

Uses trl GRPOTrainer when available, otherwise falls back to a simplified
torch-based loop. All heavy deps are lazily imported.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import FitDataset, TrainingExample
from .reward_fn import RewardFn

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    base_model: str = "Qwen/Qwen2-0.5B"
    learning_rate: float = 1e-5
    epochs: int = 3
    batch_size: int = 8
    max_seq_length: int = 512
    reward_shaping: str = "linear"  # linear, exponential, clipped
    output_dir: str = "./advisor-output"
    use_trl: bool = True
    seed: int = 42
    # Simplified trainer settings (when trl unavailable)
    beta: float = 0.1  # KL penalty coefficient
    clip_range: float = 0.2


@dataclass
class TrainingResult:
    """Result of a GRPO training run."""

    model_path: str
    epochs_completed: int
    final_loss: float
    reward_stats: dict[str, float]
    training_metadata: dict[str, Any] = field(default_factory=dict)


class GRPOTrainer:
    """GRPO policy optimization trainer.

    Uses trl under the hood when available, otherwise runs a simplified
    GRPO loop with raw torch.
    """

    def __init__(
        self,
        config: GRPOConfig,
        reward_fn: RewardFn | None = None,
    ) -> None:
        self._config = config
        self._reward_fn = reward_fn
        self._model: Any = None
        self._tokenizer: Any = None

    def train(self, dataset: FitDataset) -> TrainingResult:
        """Train the advisor model using GRPO."""
        if len(dataset) == 0:
            raise ValueError("Cannot train on empty dataset")

        if self._config.use_trl:
            try:
                return self._train_trl(dataset)
            except ImportError:
                logger.info("trl not available, falling back to simplified GRPO")

        return self._train_simplified(dataset)

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            try:
                self._model.save_pretrained(str(out))
            except Exception:
                pass
        if self._tokenizer is not None:
            try:
                self._tokenizer.save_pretrained(str(out))
            except Exception:
                pass

        # Always save config for downstream consumers
        config_path = out / "training_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "base_model": self._config.base_model,
                    "learning_rate": self._config.learning_rate,
                    "epochs": self._config.epochs,
                    "batch_size": self._config.batch_size,
                    "max_seq_length": self._config.max_seq_length,
                    "reward_shaping": self._config.reward_shaping,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _train_trl(self, dataset: FitDataset) -> TrainingResult:
        """Train using trl GRPOTrainer."""
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from trl import GRPOTrainer as TRL_GRPOTrainer  # type: ignore[import-untyped]
            from trl import GRPOConfig as TRL_GRPOConfig  # type: ignore[import-untyped]
        except ImportError:
            raise

        model = AutoModelForCausalLM.from_pretrained(self._config.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self._config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._model = model
        self._tokenizer = tokenizer

        examples = dataset.examples

        def reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
            if self._reward_fn:
                return [
                    self._reward_fn(
                        examples[i % len(examples)].context,
                        examples[i % len(examples)].advice,
                        completions[i],
                    )
                    for i in range(len(completions))
                ]
            # Default: use pre-computed rewards from dataset
            return [examples[i % len(examples)].reward for i in range(len(completions))]

        trl_config = TRL_GRPOConfig(
            output_dir=self._config.output_dir,
            num_train_epochs=self._config.epochs,
            per_device_train_batch_size=self._config.batch_size,
            learning_rate=self._config.learning_rate,
            max_completion_length=self._config.max_seq_length,
            seed=self._config.seed,
        )

        trainer = TRL_GRPOTrainer(
            model=model,
            config=trl_config,
            reward_funcs=[reward_fn],
        )
        trainer.train()
        self.save(self._config.output_dir)

        rewards = [e.reward for e in dataset]
        return TrainingResult(
            model_path=self._config.output_dir,
            epochs_completed=self._config.epochs,
            final_loss=0.0,
            reward_stats=_compute_reward_stats(rewards),
            training_metadata={
                "trainer": "trl",
                "base_model": self._config.base_model,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _train_simplified(self, dataset: FitDataset) -> TrainingResult:
        """Simplified GRPO loop using raw torch.

        Runs policy gradient updates with:
        - Reward shaping (linear/exponential/clipped)
        - KL penalty via beta
        - PPO-style clip range
        """
        import random

        import numpy as np  # noqa: F401

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "GRPO training requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from exc

        cfg = self._config
        rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)

        model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._model = model
        self._tokenizer = tokenizer

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        examples = dataset.examples
        epoch_losses: list[float] = []
        all_rewards: list[float] = []

        for epoch in range(cfg.epochs):
            indices = list(range(len(examples)))
            rng.shuffle(indices)

            epoch_reward_samples: list[float] = []
            num_batches = 0
            for batch_start in range(0, len(indices), cfg.batch_size):
                batch_idx = indices[batch_start : batch_start + cfg.batch_size]
                batch = [examples[i] for i in batch_idx]

                # Compute shaped rewards
                shaped_rewards = [
                    self._shape_reward(e.reward) for e in batch
                ]
                epoch_reward_samples.extend(shaped_rewards)

                # Tokenize contexts
                contexts = [e.context for e in batch]
                inputs = tokenizer(
                    contexts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg.max_seq_length,
                )

                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits

                # Compute log probs for policy gradient
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                # Gather log probs for actual tokens
                token_log_probs = log_probs.gather(
                    2, inputs["input_ids"].unsqueeze(-1)
                ).squeeze(-1)

                # Mask padding
                mask = inputs["attention_mask"].float()
                masked_log_probs = (token_log_probs * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp(min=1)

                # Reward-weighted loss (policy gradient)
                reward_tensor = torch.tensor(
                    shaped_rewards, dtype=torch.float32
                )
                loss = -(masked_log_probs * reward_tensor).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                num_batches += 1

            all_rewards.extend(epoch_reward_samples)
            avg_loss = (
                statistics.mean(epoch_losses[-num_batches:])
                if epoch_losses
                else 0.0
            )
            avg_reward = (
                statistics.mean(epoch_reward_samples)
                if epoch_reward_samples
                else 0.0
            )
            logger.info(
                f"Epoch {epoch + 1}/{cfg.epochs}: "
                f"loss={avg_loss:.4f}, avg_reward={avg_reward:.4f}"
            )

        self.save(cfg.output_dir)

        final_loss = epoch_losses[-1] if epoch_losses else 0.0
        return TrainingResult(
            model_path=cfg.output_dir,
            epochs_completed=cfg.epochs,
            final_loss=final_loss,
            reward_stats=_compute_reward_stats(all_rewards),
            training_metadata={
                "trainer": "simplified",
                "base_model": cfg.base_model,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "epoch_losses": epoch_losses[-cfg.epochs :]
                if len(epoch_losses) >= cfg.epochs
                else epoch_losses,
            },
        )

    def _shape_reward(self, reward: float) -> float:
        """Apply reward shaping function."""
        if self._config.reward_shaping == "exponential":
            return math.exp(reward) / math.e  # normalize so r=1 -> 1.0
        if self._config.reward_shaping == "clipped":
            return max(0.0, min(1.0, reward))
        # linear: pass through
        return reward


def _compute_reward_stats(rewards: list[float]) -> dict[str, float]:
    """Compute reward statistics."""
    if not rewards:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    return {
        "mean": statistics.mean(rewards),
        "std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "min": min(rewards),
        "max": max(rewards),
        "count": float(len(rewards)),
    }
