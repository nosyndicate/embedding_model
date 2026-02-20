"""Protocol definitions for the embedding trainer framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    pass


@dataclass
class ModelOutput:
    """Output from a model forward pass."""

    embeddings: Tensor | None = None
    logits: Tensor | None = None
    hidden_states: Tensor | None = None
    attentions: Tensor | None = None


@dataclass
class TaskOutput:
    """Output from a task's compute_loss method."""

    loss: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainOutput:
    """Output from training."""

    global_step: int
    train_loss: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalOutput:
    """Output from evaluation."""

    eval_loss: float
    metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for embedding models."""

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> ModelOutput:
        """Forward pass through the model."""
        ...

    def get_embeddings(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Get embeddings for the input."""
        ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for trainers."""

    def train(self) -> TrainOutput:
        """Run training loop."""
        ...

    def evaluate(self) -> EvalOutput:
        """Run evaluation."""
        ...


@runtime_checkable
class TaskProtocol(Protocol):
    """Protocol for training tasks."""

    def compute_loss(self, model: Any, batch: TypedDict) -> TaskOutput:
        """Compute loss for a batch."""
        ...


@runtime_checkable
class CallbackProtocol(Protocol):
    """Protocol for training callbacks."""

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        ...

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        ...

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        ...

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Called at the end of each epoch."""
        ...

    def on_step_begin(self, trainer: Any, step: int) -> None:
        """Called at the beginning of each training step."""
        ...

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        """Called at the end of each training step."""
        ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for loggers."""

    def log(self, message: str) -> None:
        """Log a message."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        ...
