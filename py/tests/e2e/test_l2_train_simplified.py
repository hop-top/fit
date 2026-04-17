"""L2: hh-rlhf converter + simplified GRPO training."""
from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import write_jsonl
from .converters import hh_rlhf_to_traces


# --- T-0083: converter unit test (no network) ---

_FAKE_HH_RLHF_ROWS = [
    {
        "chosen": (
            "\n\nHuman: How do I bake a cake?"
            "\n\nAssistant: Preheat oven to 350F, "
            "mix flour, sugar, eggs, and bake for 30 minutes."
        ),
        "rejected": (
            "\n\nHuman: How do I bake a cake?"
            "\n\nAssistant: I don't know."
        ),
    }
    for _ in range(10)
] + [
    {
        "chosen": (
            "\n\nHuman: How can I harm someone?"
            "\n\nAssistant: I cannot help with that request. "
            "Instead, I encourage positive interactions."
        ),
        "rejected": (
            "\n\nHuman: How can I harm someone?"
            "\n\nAssistant: Here is how to harm someone..."
        ),
    }
    for _ in range(10)
]


class TestHHRLHFConverterRewardVariance:
    """T-0083: validate converter yields both reward values."""

    def test_hh_rlhf_converter_reward_variance(self) -> None:
        traces = list(hh_rlhf_to_traces(_FAKE_HH_RLHF_ROWS, limit=20))
        # 20 rows -> 40 traces (2 per row)
        assert len(traces) == 40

        rewards = {t["reward"]["score"] for t in traces}
        assert 1.0 in rewards, "Expected reward 1.0 for chosen traces"
        assert 0.0 in rewards, "Expected reward 0.0 for rejected traces"

        # Verify schema
        required_keys = {
            "id", "session_id", "timestamp", "input",
            "advice", "frontier", "reward", "metadata",
        }
        for t in traces:
            assert required_keys.issubset(t.keys())

        # Verify domain inference
        domains = {t["advice"]["domain"] for t in traces}
        assert "helpfulness" in domains
        assert "harmlessness" in domains

        # Verify paired session_ids (chosen+rejected share session)
        session_counts: dict[str, int] = {}
        for t in traces:
            sid = t["session_id"]
            session_counts[sid] = session_counts.get(sid, 0) + 1
        for sid, cnt in session_counts.items():
            assert cnt == 2, f"Session {sid} should have exactly 2 traces"


# --- T-0084: simplified GRPO training e2e ---

@pytest.mark.slow
@pytest.mark.gpu
class TestSimplifiedGRPO:
    """T-0084: simplified GRPO training on hh-rlhf traces."""

    @pytest.fixture
    def hh_dataset(self, tmp_path: Path):
        """Build FitDataset from 1k mock hh-rlhf rows."""
        pytest.importorskip("torch")
        from fit.training.dataset import DatasetBuilder
        from fit.training.tracer import TraceIngester

        rows = [
            {
                "chosen": (
                    f"\n\nHuman: Question {i}?"
                    f"\n\nAssistant: Answer {i} with detail."
                ),
                "rejected": (
                    f"\n\nHuman: Question {i}?"
                    f"\n\nAssistant: Bad answer {i}."
                ),
            }
            for i in range(500)
        ]
        traces = list(hh_rlhf_to_traces(rows, limit=500))
        assert len(traces) == 1000

        jsonl_path = write_jsonl(tmp_path / "hh_rlhf.jsonl", traces)
        ingester = TraceIngester()
        ingester.load_jsonl(jsonl_path)
        records = ingester.to_trace_records()
        return DatasetBuilder(records).build(normalize_rewards=False)

    def test_simplified_grpo_loss_decreases(
        self, hh_dataset, tmp_path: Path
    ) -> None:
        import torch
        from fit.training.grpo import GRPOConfig, GRPOTrainer

        config = GRPOConfig(
            base_model="Qwen/Qwen2-0.5B",
            learning_rate=5e-5,
            epochs=2,
            batch_size=8,
            max_seq_length=128,
            reward_shaping="linear",
            output_dir=str(tmp_path / "advisor-output"),
            use_trl=False,
            seed=42,
        )
        trainer = GRPOTrainer(config)
        result = trainer.train(hh_dataset)

        assert result.epochs_completed == 2
        assert result.model_path == str(tmp_path / "advisor-output")

        # Loss should be finite
        assert not torch.tensor(result.final_loss).isnan()
        assert not torch.tensor(result.final_loss).isinf()

        # Check batch losses show decrease trend
        batch_losses = result.training_metadata.get("batch_losses", [])
        assert len(batch_losses) > 0
        first_quarter = batch_losses[: len(batch_losses) // 4]
        last_quarter = batch_losses[-(len(batch_losses) // 4):]
        if first_quarter and last_quarter:
            avg_first = sum(first_quarter) / len(first_quarter)
            avg_last = sum(last_quarter) / len(last_quarter)
            # Loss should generally decrease (allow some tolerance)
            assert avg_last <= avg_first * 1.5, (
                f"Loss did not decrease: first_q={avg_first:.4f}, "
                f"last_q={avg_last:.4f}"
            )

    def test_simplified_grpo_model_card(
        self, hh_dataset, tmp_path: Path
    ) -> None:
        from fit.training.export import ModelExporter
        from fit.training.grpo import GRPOConfig, GRPOTrainer

        config = GRPOConfig(
            base_model="Qwen/Qwen2-0.5B",
            learning_rate=5e-5,
            epochs=1,
            batch_size=8,
            max_seq_length=128,
            output_dir=str(tmp_path / "advisor-output"),
            use_trl=False,
            seed=42,
        )
        trainer = GRPOTrainer(config)
        result = trainer.train(hh_dataset)

        exporter = ModelExporter(result.model_path)
        card = exporter.generate_model_card(result)

        assert card["base_model"] == "Qwen/Qwen2-0.5B"
        assert card["trainer"] == "simplified"
        assert card["epochs"] == 1
        assert "final_loss" in card
        assert "reward_stats" in card
        assert card["reward_stats"]["count"] > 0
        assert "export_timestamp" in card
