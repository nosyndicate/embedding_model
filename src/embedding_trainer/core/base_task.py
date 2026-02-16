"""Base class for training tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from embedding_trainer.core.types import TaskOutput
from embedding_trainer.data.base import PreTokenizedBatch


class BaseTask(ABC):
    """Abstract base class for training tasks.

    A task defines how to compute loss and metrics for a specific
    training objective (e.g., MLM, contrastive learning, etc.).
    """

    @abstractmethod
    def compute_loss(self, model: Any, batch: PreTokenizedBatch) -> TaskOutput:
        """Compute loss for a batch.

        Args:
            model: The model to compute loss for.
            batch: PreTokenizedBatch containing input_ids, attention_mask, and labels tensors.

        Returns:
            TaskOutput containing the loss and any computed metrics.
        """
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Get accumulated metrics since last reset.

        Returns:
            Dictionary of metric names to values.
        """
        ...
