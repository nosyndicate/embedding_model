#!/usr/bin/env python3
"""HuggingFace Trainer baseline for MLM training.

This script provides a reference implementation using HuggingFace Trainer
to compare against our custom training loop.

Usage:
    python scripts/train_hf_baseline.py \
        --data_dir ./data/fineweb_edu_100bt \
        --model_name bert-base-uncased \
        --max_steps 10000 \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --packing_mode doc_aware \
        --sampling_strategy epoch_offset \
        --output_dir ./baseline_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from embedding_trainer.data.collators.mlm import MLMCollator, MLMCollatorConfig
from embedding_trainer.data.datasets.pretokenized import (
    PreTokenizedConfig,
    PreTokenizedDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MetricsCallback(TrainerCallback):
    """Callback to track and save metrics."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.metrics_log: list[dict[str, Any]] = []
        self.start_time: float | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.start_time = time.time()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs:
            entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                **logs,
            }
            if self.start_time:
                entry["elapsed_seconds"] = time.time() - self.start_time
            self.metrics_log.append(entry)
            logger.info(f"Step {state.global_step}: {logs}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # Save metrics to JSON
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")


def compute_mlm_accuracy(eval_pred: Any) -> dict[str, float]:
    """Compute MLM accuracy metric."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # Only consider non-padding positions (labels != -100)
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0

    return {"mlm_accuracy": float(accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="HuggingFace MLM baseline training")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to pre-tokenized data directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Model name or path",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (1e-4 for pretraining, 5e-5 for fine-tuning)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (fraction of total steps for warmup)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Warmup steps (overrides warmup_ratio if set)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision",
    )
    parser.add_argument(
        "--packing_mode",
        type=str,
        default="flat",
        choices=["flat", "doc_aware"],
        help="Packing mode: 'flat' or 'doc_aware' (respects document boundaries)",
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="epoch_offset",
        choices=["fixed", "epoch_offset", "random", "shuffle_indices"],
        help="Sampling strategy for sequences",
    )
    parser.add_argument(
        "--num_offset_phases",
        type=int,
        default=4,
        help="Number of offset phases for epoch_offset strategy",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-tokenized dataset using proper data pipeline
    logger.info(f"Loading data from: {args.data_dir}")
    dataset_config = PreTokenizedConfig(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split="train",
        shuffle=True,
        seed=args.seed,
        packing_mode=args.packing_mode,
        sampling_strategy=args.sampling_strategy,
        num_offset_phases=args.num_offset_phases,
    )
    train_dataset = PreTokenizedDataset(dataset_config)
    logger.info(f"Dataset: {train_dataset}")
    logger.info(
        f"Packing mode: {args.packing_mode}, Sampling strategy: {args.sampling_strategy}"
    )
    logger.info(f"LR: {args.learning_rate}, Scheduler: {args.lr_scheduler_type}")

    if train_dataset.header is None:
        raise RuntimeError("Failed to read shard header from data")

    # Initialize model
    # Use a fresh model config to ensure compatibility with pre-tokenized data
    logger.info(f"Loading model: {args.model_name}")
    config = BertConfig.from_pretrained(args.model_name)
    config.vocab_size = train_dataset.header.vocab_size
    config.pad_token_id = train_dataset.header.pad_id

    model = BertForMaskedLM(config)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # Configure collator from shard header (reads special token IDs)
    collator_config = MLMCollatorConfig(
        pad_token_id=train_dataset.header.pad_id,
        cls_token_id=train_dataset.header.cls_id,
        sep_token_id=train_dataset.header.sep_id,
        mask_token_id=train_dataset.header.mask_id,
        vocab_size=train_dataset.header.vocab_size,
    )
    data_collator = MLMCollator(collator_config)

    # Training arguments
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": 0.01,
        "logging_steps": args.logging_steps,
        "save_steps": args.max_steps,  # Save at end
        "save_total_limit": 1,
        "seed": args.seed,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "dataloader_num_workers": 0,  # Use 0 for IterableDataset
        "remove_unused_columns": False,
        "report_to": "none",  # Disable wandb/tensorboard for baseline
    }

    # Use warmup_steps if provided, otherwise use warmup_ratio
    if args.warmup_steps is not None:
        training_args_kwargs["warmup_steps"] = args.warmup_steps
        logger.info(f"Using warmup_steps: {args.warmup_steps}")
    else:
        training_args_kwargs["warmup_ratio"] = args.warmup_ratio
        logger.info(
            f"Using warmup_ratio: {args.warmup_ratio} ({int(args.max_steps * args.warmup_ratio)} steps)"
        )

    training_args = TrainingArguments(**training_args_kwargs)

    # Metrics callback
    metrics_callback = MetricsCallback(output_dir)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[metrics_callback],
    )

    # Train
    logger.info("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed_time = time.time() - start_time

    # Log final results
    logger.info(f"Training completed in {elapsed_time:.2f}s")
    logger.info(f"Final loss: {train_result.training_loss:.4f}")
    logger.info(f"Steps per second: {args.max_steps / elapsed_time:.2f}")

    # Save final summary
    summary = {
        "model_name": args.model_name,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio if args.warmup_steps is None else None,
        "max_seq_length": args.max_seq_length,
        "packing_mode": args.packing_mode,
        "sampling_strategy": args.sampling_strategy,
        "num_offset_phases": args.num_offset_phases,
        "final_loss": train_result.training_loss,
        "elapsed_seconds": elapsed_time,
        "steps_per_second": args.max_steps / elapsed_time,
        "model_parameters": model.num_parameters(),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
