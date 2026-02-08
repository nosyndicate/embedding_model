"""MLM evaluation task."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from embedding_trainer.data.base import CollatorProtocol
from embedding_trainer.data.collators.mlm import MLMCollator, MLMCollatorConfig
from embedding_trainer.data.datasets.pretokenized import ShardHeader
from embedding_trainer.eval.base import BaseEvalTask
from embedding_trainer.eval.registry import EVAL_TASK_REGISTRY


@EVAL_TASK_REGISTRY.register("mlm")
class MLMEvalTask(BaseEvalTask):
    """
    Evaluation task for Masked Language Modeling.

    Computes:
    - eval_loss: Cross-entropy loss on masked tokens
    - perplexity: exp(eval_loss)
    - mlm_accuracy: Exact match accuracy on masked positions
    - top_k_accuracy: Top-k accuracy for k=3, 5, 10
    """

    def get_collator(self, header: ShardHeader) -> CollatorProtocol:
        """Get MLM collator configured from shard header."""
        config = MLMCollatorConfig(
            pad_token_id=header.pad_id,
            cls_token_id=header.cls_id,
            sep_token_id=header.sep_id,
            mask_token_id=header.mask_id,
            vocab_size=header.vocab_size,
        )
        return MLMCollator(config)

    def compute_batch_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        device: str,
    ) -> dict[str, float]:
        """
        Compute MLM metrics for a single batch.

        Handles both HuggingFace models (with .loss, .logits attributes)
        and custom models returning ModelOutput.
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Extract loss and logits (handle both HF and custom output formats)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss.item()
        else:
            # Need to compute loss manually
            loss = self._compute_loss(outputs.logits, labels)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            raise ValueError("Model output must have 'logits' attribute")

        # Compute accuracy metrics
        mask = labels != -100
        num_masked = mask.sum().item()

        if num_masked == 0:
            return {
                "loss": 0.0,
                "num_masked": 0,
                "correct": 0,
                "top_3_correct": 0,
                "top_5_correct": 0,
                "top_10_correct": 0,
            }

        # Get predictions
        predictions = logits.argmax(dim=-1)
        correct = ((predictions == labels) & mask).sum().item()

        # Top-k accuracy
        top_3_correct = self._top_k_correct(logits, labels, mask, k=3)
        top_5_correct = self._top_k_correct(logits, labels, mask, k=5)
        top_10_correct = self._top_k_correct(logits, labels, mask, k=10)

        return {
            "loss": loss * num_masked,  # Weighted by num_masked for aggregation
            "num_masked": num_masked,
            "correct": correct,
            "top_3_correct": top_3_correct,
            "top_5_correct": top_5_correct,
            "top_10_correct": top_10_correct,
        }

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute cross-entropy loss manually."""
        loss_fn = nn.CrossEntropyLoss()
        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = loss_fn(logits_flat, labels_flat)
        return loss.item()

    def _top_k_correct(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        k: int,
    ) -> int:
        """Count how many masked tokens have correct label in top-k predictions."""
        # Get top-k predictions
        _, top_k_preds = logits.topk(k, dim=-1)  # (batch, seq, k)

        # Expand labels for comparison
        labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)

        # Check if true label is in top-k
        correct = (top_k_preds == labels_expanded).any(dim=-1)  # (batch, seq)
        correct = (correct & mask).sum().item()

        return correct

    def aggregate_metrics(
        self,
        batch_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate metrics from all batches."""
        total_loss = sum(m["loss"] for m in batch_metrics)
        total_masked = sum(m["num_masked"] for m in batch_metrics)
        total_correct = sum(m["correct"] for m in batch_metrics)
        total_top_3 = sum(m["top_3_correct"] for m in batch_metrics)
        total_top_5 = sum(m["top_5_correct"] for m in batch_metrics)
        total_top_10 = sum(m["top_10_correct"] for m in batch_metrics)

        if total_masked == 0:
            return {
                "eval_loss": 0.0,
                "perplexity": 1.0,
                "mlm_accuracy": 0.0,
                "top_3_accuracy": 0.0,
                "top_5_accuracy": 0.0,
                "top_10_accuracy": 0.0,
            }

        avg_loss = total_loss / total_masked
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
            "mlm_accuracy": total_correct / total_masked,
            "top_3_accuracy": total_top_3 / total_masked,
            "top_5_accuracy": total_top_5 / total_masked,
            "top_10_accuracy": total_top_10 / total_masked,
        }
