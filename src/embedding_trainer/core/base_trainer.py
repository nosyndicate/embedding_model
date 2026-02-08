"""Base class for trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from embedding_trainer.core.types import EvalOutput, TrainOutput


class BaseTrainer(ABC):
    """Abstract base class for trainers.

    A trainer orchestrates the training loop, handling:
    - Iteration over data
    - Forward/backward passes
    - Optimizer steps
    - Checkpointing
    - Callbacks
    """

    @abstractmethod
    def train(self) -> TrainOutput:
        """Run the training loop.

        Returns:
            TrainOutput containing training results and metrics.
        """
        ...

    @abstractmethod
    def evaluate(self) -> EvalOutput:
        """Run evaluation on the validation set.

        Returns:
            EvalOutput containing evaluation results and metrics.
        """
        ...

    @abstractmethod
    def save_checkpoint(self, path: str | Path) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save the checkpoint to.
        """
        ...

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to load the checkpoint from.
        """
        ...
