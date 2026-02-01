"""Base class for training tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from embedding_trainer.core.types import TaskOutput


class BaseTask(ABC):
    """Abstract base class for training tasks.

    A task defines how to compute loss and metrics for a specific
    training objective (e.g., MLM, contrastive learning, etc.).
    """

    @abstractmethod
    def compute_loss(self, model: Any, batch: dict[str, Tensor]) -> TaskOutput:
        """Compute loss for a batch.

        Args:
            model: The model to compute loss for.
            batch: Dictionary containing batch tensors (input_ids, attention_mask, etc.)

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

    def reset_metrics(self) -> None:
        """Reset accumulated metrics. Override if task tracks metrics."""
        pass
