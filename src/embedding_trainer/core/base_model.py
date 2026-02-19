"""Base class for embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn

from embedding_trainer.core.types import ModelOutput


class BaseEmbeddingModel(nn.Module, ABC):
    """Abstract base class for embedding models.

    Subclasses must implement:
    - forward(): Full forward pass returning ModelOutput
    - get_embeddings(): Extract embeddings from inputs
    - hidden_size: Property returning the embedding dimension
    """

    @abstractmethod
    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> ModelOutput:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length).
            attention_mask: Attention mask of shape (batch_size, seq_length).

        Returns:
            ModelOutput containing embeddings and optional loss/logits.
        """
        ...

    @abstractmethod
    def get_embeddings(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Get embeddings for the input.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_length).
            attention_mask: Attention mask of shape (batch_size, seq_length).

        Returns:
            Embeddings tensor of shape (batch_size, hidden_size) or
            (batch_size, seq_length, hidden_size) depending on implementation.
        """
        ...

    @abstractmethod
    def get_param_groups(self, **kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Return parameter groups for optimization.

        This can be used to apply different learning rates or weight decay to
        different parts of the model (e.g., backbone vs. pooling layer).

        Returns:
            A list of parameter group dictionaries, each containing a 'params' key
            with the parameters and optionally other keys like 'lr' or 'weight_decay'.
        """
        ...

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden size / embedding dimension of the model."""
        ...
