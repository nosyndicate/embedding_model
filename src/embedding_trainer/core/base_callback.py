"""Base class for training callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


class BaseCallback:
    """Base class for training callbacks.

    Callbacks allow injecting custom behavior at various points in the
    training loop. Override the methods you need.

    All methods are no-ops by default, so subclasses only need to
    implement the hooks they care about.
    """

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch.

        Args:
            trainer: The trainer instance.
            epoch: The current epoch number (0-indexed).
        """
        pass

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Called at the end of each epoch.

        Args:
            trainer: The trainer instance.
            epoch: The current epoch number (0-indexed).
        """
        pass

    def on_step_begin(self, trainer: Any, step: int) -> None:
        """Called at the beginning of each training step.

        Args:
            trainer: The trainer instance.
            step: The current global step number.
        """
        pass

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        """Called at the end of each training step.

        Args:
            trainer: The trainer instance.
            step: The current global step number.
            loss: The loss value for this step.
        """
        pass

    def on_evaluate_begin(self, trainer: Any) -> None:
        """Called at the beginning of evaluation.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_evaluate_end(self, trainer: Any, metrics: dict[str, float]) -> None:
        """Called at the end of evaluation.

        Args:
            trainer: The trainer instance.
            metrics: The evaluation metrics.
        """
        pass
