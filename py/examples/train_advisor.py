"""End-to-end advisor training example.

Usage:
    python -m examples.train_advisor --traces ../../spec/fixtures --output ./advisor-output
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fit advisor model from trace data"
    )
    parser.add_argument(
        "--traces",
        type=str,
        default="../../spec/fixtures",
        help="Path to trace directory or JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./advisor-output",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Base model for advisor",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter traces by domain",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without actual training (no torch required)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve trace path
    traces_path = Path(args.traces)
    if not traces_path.is_absolute():
        traces_path = Path(__file__).resolve().parent / traces_path
    traces_path = traces_path.resolve()

    if not traces_path.exists():
        logger.error("Trace path not found: %s", traces_path)
        sys.exit(1)

    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # -- Step 1: Ingest traces --
    logger.info("Step 1: Ingesting traces from %s", traces_path)
    from fit.training.tracer import TraceIngester

    ingester = TraceIngester()

    # Try JSONL first, then directory
    if traces_path.is_file():
        ingester.load_batch([traces_path])
    else:
        # Load all formats from directory
        jsonl_files = list(traces_path.glob("*.jsonl"))
        json_files = list(traces_path.glob("*.json"))
        yaml_dir = traces_path if any(traces_path.rglob("*.y*ml")) else None

        if jsonl_files:
            ingester.load_batch(jsonl_files)
        if json_files:
            ingester.load_batch(json_files)
        if yaml_dir:
            ingester.load_yaml_dir(yaml_dir)

    # Filter by domain if specified
    if args.domain:
        ingester = ingester.filter(domain=args.domain)

    records = ingester.to_trace_records()
    logger.info("  Loaded %d trace records", len(records))

    if not records:
        logger.error("No trace records found. Exiting.")
        sys.exit(1)

    # -- Step 2: Build dataset --
    logger.info("Step 2: Building training dataset")
    from fit.training.dataset import DatasetBuilder

    builder = DatasetBuilder(records)
    dataset = builder.build(normalize_rewards=True, group_by_session=True)
    logger.info("  Dataset size: %d examples", len(dataset))

    if len(dataset) == 0:
        logger.error("No valid training examples (all rewards null?). Exiting.")
        sys.exit(1)

    stats = dataset.reward_stats()
    logger.info(
        "  Reward stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
        stats["mean"],
        stats["std"],
        stats["min"],
        stats["max"],
    )

    # Split
    train_ds, val_ds = dataset.split(val_ratio=args.val_ratio, seed=42)
    logger.info("  Train: %d, Val: %d", len(train_ds), len(val_ds))

    # -- Step 3: Train (or dry-run) --
    if args.dry_run:
        logger.info("Step 3: Dry-run mode — skipping training")
        result = _dry_run_result(args, records, stats)
    else:
        logger.info("Step 3: Training advisor model")
        from fit.training.grpo import GRPOConfig, GRPOTrainer

        config = GRPOConfig(
            base_model=args.base_model,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=str(output_path),
        )
        trainer = GRPOTrainer(config)
        try:
            result = trainer.train(train_ds)
        except ImportError as exc:
            logger.error(
                "Training requires torch + transformers: %s\n"
                "Use --dry-run to test the pipeline without training.",
                exc,
            )
            sys.exit(1)

    # -- Step 4: Export --
    logger.info("Step 4: Generating model card")
    from fit.training.export import ModelExporter

    exporter = ModelExporter(str(output_path))
    card = exporter.generate_model_card(result)

    card_path = output_path / "model_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    logger.info("  Model card saved to %s", card_path)

    # -- Summary --
    _print_summary(records, dataset, train_ds, val_ds, result, output_path)


def _dry_run_result(
    args: argparse.Namespace,
    records,
    stats: dict[str, float],
) -> "TrainingResult":  # noqa: F821
    """Create a mock training result for dry-run mode."""
    from fit.training.grpo import TrainingResult

    return TrainingResult(
        model_path=args.output,
        epochs_completed=0,
        final_loss=0.0,
        reward_stats=stats,
        training_metadata={
            "trainer": "dry-run",
            "base_model": args.base_model,
            "trace_count": len(records),
        },
    )


def _print_summary(
    records,
    dataset,
    train_ds,
    val_ds,
    result,
    output_path: Path,
) -> None:
    """Print training summary."""
    print("\n" + "=" * 60)
    print("  FIT ADVISOR TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Traces loaded:       {len(records)}")
    print(f"  Training examples:   {len(dataset)}")
    print(f"  Train/Val split:     {len(train_ds)}/{len(val_ds)}")
    print(f"  Epochs:              {result.epochs_completed}")
    print(f"  Final loss:          {result.final_loss:.4f}")
    rs = result.reward_stats
    print(
        f"  Reward: mean={rs.get('mean', 0):.3f} "
        f"std={rs.get('std', 0):.3f} "
        f"min={rs.get('min', 0):.3f} "
        f"max={rs.get('max', 0):.3f}"
    )
    print(f"  Output:              {output_path}")
    print(f"  Model card:          {output_path / 'model_card.json'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
