"""fit.training — advisor model training pipeline.

All submodules use lazy imports so core `fit` works without training deps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracer import TraceIngester, TraceRecord


def __getattr__(name: str) -> object:
    """Lazy-import training submodules on first access."""
    _modules = {
        "TraceIngester": ".tracer",
        "TraceRecord": ".tracer",
        "DatasetBuilder": ".dataset",
        "FitDataset": ".dataset",
        "TrainingExample": ".dataset",
        "GRPOConfig": ".grpo",
        "GRPOTrainer": ".grpo",
        "TrainingResult": ".grpo",
        "RewardFn": ".reward_fn",
        "ExactMatchReward": ".reward_fn",
        "RubricJudgeReward": ".reward_fn",
        "LLMJudgeReward": ".reward_fn",
        "UserSignalReward": ".reward_fn",
        "CompositeReward": ".reward_fn",
        "ModelExporter": ".export",
    }
    if name in _modules:
        import importlib

        mod = importlib.import_module(_modules[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CompositeReward",
    "DatasetBuilder",
    "ExactMatchReward",
    "FitDataset",
    "GRPOConfig",
    "GRPOTrainer",
    "LLMJudgeReward",
    "ModelExporter",
    "RewardFn",
    "RubricJudgeReward",
    "TraceIngester",
    "TraceRecord",
    "TrainingExample",
    "TrainingResult",
    "UserSignalReward",
]
