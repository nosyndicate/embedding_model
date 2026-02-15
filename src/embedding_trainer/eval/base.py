"""Base classes and dataclasses for evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from embedding_trainer.data.base import CollatorProtocol, ShardHeader


@dataclass
class EvalResult:
    """Output from evaluation."""

    model_path: str
    task: str
    eval_loss: float
    perplexity: float
    accuracy: float
    metrics: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    elapsed_seconds: float = 0.0


class BaseEvalTask(ABC):
    """Abstract base class for evaluation tasks."""

    @abstractmethod
    def get_collator(self, header: ShardHeader) -> CollatorProtocol:
        """
        Get the collator for this evaluation task.

        Args:
            header: Shard header containing tokenizer info

        Returns:
            A collator instance
        """
        ...

    @abstractmethod
    def compute_batch_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        device: str,
    ) -> dict[str, float]:
        """
        Compute metrics for a single batch.

        Args:
            model: The model to evaluate
            batch: A batch of data
            device: Device to run on

        Returns:
            Dictionary of metric names to values
        """
        ...

    @abstractmethod
    def aggregate_metrics(
        self,
        batch_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        """
        Aggregate metrics from all batches.

        Args:
            batch_metrics: List of per-batch metrics

        Returns:
            Final aggregated metrics
        """
        ...
