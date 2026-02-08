#!/usr/bin/env python3
"""Evaluation script for embedding models.

Supports both HuggingFace models and custom models implementing BaseEmbeddingModel.

Usage:
    python scripts/evaluate.py \
        --model_path ./baseline_results/checkpoint-10000 \
        --data_dir ./data/fineweb_edu_100bt \
        --split val \
        --task mlm \
        --batch_size 32 \
        --num_batches 1000 \
        --output_dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from embedding_trainer.data.datasets.pretokenized import (
    PreTokenizedConfig,
    PreTokenizedDataset,
)
from embedding_trainer.eval import EVAL_TASK_REGISTRY, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on various tasks"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to pre-tokenized data directory",
    )

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "hf", "custom"],
        help="Model type: auto (detect), hf (HuggingFace), custom",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default=None,
        help="For custom models: class path (e.g., 'mymodule.MyModel')",
    )

    # Data configuration
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # Evaluation configuration
    parser.add_argument(
        "--task",
        type=str,
        default="mlm",
        help="Evaluation task (default: mlm)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Limit number of batches (for quick testing)",
    )

    # Precision
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

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")

    # Setup mixed precision
    dtype = torch.float32
    if args.bf16 and device == "cuda":
        dtype = torch.bfloat16
        logger.info("Using BF16 mixed precision")
    elif args.fp16 and device == "cuda":
        dtype = torch.float16
        logger.info("Using FP16 mixed precision")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model, model_metadata = load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        model_class=args.model_class,
        device=device,
    )
    logger.info(f"Model loaded: {model_metadata}")

    # Load dataset
    logger.info(f"Loading data from: {args.data_dir}")
    dataset_config = PreTokenizedConfig(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split=args.split,
        shuffle=False,  # No shuffling for evaluation
    )
    dataset = PreTokenizedDataset(dataset_config)
    logger.info(f"Dataset: {dataset}")

    if dataset.header is None:
        raise RuntimeError("Failed to read shard header")

    # Get evaluation task
    if args.task not in EVAL_TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {args.task}. "
            f"Available: {list(EVAL_TASK_REGISTRY._registry.keys())}"
        )
    eval_task = EVAL_TASK_REGISTRY.build(args.task)
    logger.info(f"Evaluation task: {args.task}")

    # Get collator from task
    collator = eval_task.get_collator(dataset.header)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,  # IterableDataset
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    start_time = time.time()
    batch_metrics = []
    num_samples = 0

    autocast_context = (
        torch.autocast(device_type=device, dtype=dtype)
        if dtype != torch.float32
        else torch.no_grad()
    )

    with torch.no_grad(), autocast_context:
        for batch_idx, batch in enumerate(dataloader):
            if args.num_batches is not None and batch_idx >= args.num_batches:
                break

            metrics = eval_task.compute_batch_metrics(model, batch, device)
            batch_metrics.append(metrics)
            num_samples += batch["input_ids"].size(0)

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"Processed {batch_idx + 1} batches ({num_samples} samples)"
                )

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f}s")

    # Aggregate metrics
    final_metrics = eval_task.aggregate_metrics(batch_metrics)
    logger.info(f"Final metrics: {final_metrics}")

    # Build result
    result = {
        "model_path": str(args.model_path),
        "model_type": model_metadata.get("model_type", "unknown"),
        "task": args.task,
        "data_path": str(args.data_dir),
        "split": args.split,
        **final_metrics,
        "num_samples": num_samples,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Split: {args.split}")
    print(f"Samples: {num_samples}")
    print("-" * 50)
    print(f"Loss: {final_metrics['eval_loss']:.4f}")
    print(f"Perplexity: {final_metrics['perplexity']:.2f}")
    print(f"MLM Accuracy: {final_metrics['mlm_accuracy']:.4f}")
    print(f"Top-3 Accuracy: {final_metrics['top_3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {final_metrics['top_5_accuracy']:.4f}")
    print(f"Top-10 Accuracy: {final_metrics['top_10_accuracy']:.4f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
