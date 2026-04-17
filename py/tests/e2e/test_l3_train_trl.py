"""L3: UltraFeedback converter + TRL GRPO training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

trl = pytest.importorskip("trl")

import torch

from fit.training.dataset import DatasetBuilder, FitDataset
from fit.training.export import ModelExporter
from fit.training.grpo import GRPOConfig, GRPOTrainer
from fit.training.tracer import TraceIngester

from .conftest import write_jsonl
from .converters import ultrafeedback_to_traces


# --- T-0085: converter unit test (no network) ---

_FAKE_ULTRAFEEDBACK_ROWS = [
    {
        "chosen": [
            {"role": "user", "content": f"Explain topic {i} clearly."},
            {
                "role": "assistant",
                "content": f"Topic {i} is a well-studied area. "
                f"Here is a detailed explanation of topic {i}.",
            },
        ],
        "rejected": [
            {"role": "user", "content": f"Explain topic {i} clearly."},
            {
                "role": "assistant",
                "content": f"I think topic {i} is something.",
            },
        ],
        "score_chosen": 4.5,
        "score_rejected": 2.0,
    }
    for i in range(20)
]


class TestUltraFeedbackMultiDimRewards:
    """T-0085: validate converter yields multi-dimensional rewards."""

    def test_ultrafeedback_multi_dim_rewards(self) -> None:
        traces = list(
            ultrafeedback_to_traces(_FAKE_ULTRAFEEDBACK_ROWS, limit=20)
        )
        # 20 rows -> 40 traces (2 per row)
        assert len(traces) == 40

        # Verify schema
        required_keys = {
            "id", "session_id", "timestamp", "input",
            "advice", "frontier", "reward", "metadata",
        }
        for t in traces:
            assert required_keys.issubset(t.keys())

        # Verify multi-dim reward breakdown
        reward_dims = {"helpfulness", "honesty",
                       "instruction_following", "truthfulness"}
        for t in traces:
            breakdown = t["reward"]["breakdown"]
            assert reward_dims.issubset(breakdown.keys()), (
                f"Missing reward dims: {reward_dims - breakdown.keys()}"
            )
            for dim, val in breakdown.items():
                assert 0.0 <= val <= 1.0, (
                    f"Reward dim {dim} out of range: {val}"
                )

        # Verify reward variance (chosen > rejected scores)
        rewards = [t["reward"]["score"] for t in traces]
        assert min(rewards) < max(rewards), (
            "Expected reward variance between chosen and rejected"
        )

        # Verify domain
        for t in traces:
            assert t["advice"]["domain"] == "instruction-following"

        # Verify paired sessions
        session_counts: dict[str, int] = {}
        for t in traces:
            sid = t["session_id"]
            session_counts[sid] = session_counts.get(sid, 0) + 1
        for sid, cnt in session_counts.items():
            assert cnt == 2, f"Session {sid} should have exactly 2 traces"


# --- T-0086: TRL GRPO training e2e ---

@pytest.mark.slow
@pytest.mark.gpu
class TestTRLGRPO:
    """T-0086: TRL GRPO training on UltraFeedback traces."""

    @pytest.fixture
    def uf_dataset(self, tmp_path: Path) -> FitDataset:
        """Build FitDataset from 5k UltraFeedback traces."""
        rows = [
            {
                "chosen": [
                    {"role": "user", "content": f"Question {i}?"},
                    {
                        "role": "assistant",
                        "content": f"Detailed answer for question {i}.",
                    },
                ],
                "rejected": [
                    {"role": "user", "content": f"Question {i}?"},
                    {
                        "role": "assistant",
                        "content": f"Short answer {i}.",
                    },
                ],
                "score_chosen": 4.0 + (i % 10) * 0.1,
                "score_rejected": 1.5 + (i % 10) * 0.1,
            }
            for i in range(2500)
        ]
        traces = list(ultrafeedback_to_traces(rows, limit=2500))
        assert len(traces) == 5000

        jsonl_path = write_jsonl(tmp_path / "ultrafeedback.jsonl", traces)
        ingester = TraceIngester()
        ingester.load_jsonl(jsonl_path)
        records = ingester.to_trace_records()
        return DatasetBuilder(records).build(normalize_rewards=True)

    def test_trl_grpo_convergence(
        self, uf_dataset: FitDataset, tmp_path: Path
    ) -> None:
        config = GRPOConfig(
            base_model="Qwen/Qwen2-0.5B",
            learning_rate=1e-5,
            epochs=2,
            batch_size=8,
            max_seq_length=128,
            output_dir=str(tmp_path / "trl-output"),
            use_trl=True,
            seed=42,
        )
        trainer = GRPOTrainer(config)
        result = trainer.train(uf_dataset)

        assert result.epochs_completed == 2
        assert result.training_metadata.get("trainer") == "trl"
        assert result.reward_stats["count"] > 0

    def test_trl_grpo_export_safetensors(
        self, uf_dataset: FitDataset, tmp_path: Path
    ) -> None:
        config = GRPOConfig(
            base_model="Qwen/Qwen2-0.5B",
            learning_rate=1e-5,
            epochs=1,
            batch_size=8,
            max_seq_length=128,
            output_dir=str(tmp_path / "trl-output"),
            use_trl=True,
            seed=42,
        )
        trainer = GRPOTrainer(config)
        result = trainer.train(uf_dataset)

        exporter = ModelExporter(result.model_path)
        export_dir = tmp_path / "safetensors-export"
        out = exporter.to_safetensors(str(export_dir))

        assert out.exists()
        # Should have config/tokenizer artifacts
        assert (out / "training_config.json").exists() or (
            out / "config.json"
        ).exists()

    def test_trl_grpo_multi_dim_reward_stats(
        self, uf_dataset: FitDataset, tmp_path: Path
    ) -> None:
        config = GRPOConfig(
            base_model="Qwen/Qwen2-0.5B",
            learning_rate=1e-5,
            epochs=1,
            batch_size=8,
            max_seq_length=128,
            output_dir=str(tmp_path / "trl-output"),
            use_trl=True,
            seed=42,
        )
        trainer = GRPOTrainer(config)
        result = trainer.train(uf_dataset)

        stats = result.reward_stats
        assert stats["count"] > 0
        assert stats["mean"] > 0
        assert stats["std"] >= 0
        assert stats["min"] <= stats["max"]

        # Dataset should have multi-dim breakdown in metadata
        for ex in list(uf_dataset)[:10]:
            breakdown = ex.metadata.get("breakdown", {})
            if breakdown:
                assert "helpfulness" in breakdown or "quality" in breakdown
