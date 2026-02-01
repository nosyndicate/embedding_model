"""Base collator with common utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from embedding_trainer.data.base import PreTokenizedBatch


@dataclass
class BaseCollatorConfig:
    """Base configuration for collators."""

    # BERT special token IDs
    pad_token_id: int = 0  # [PAD]
    cls_token_id: int = 101  # [CLS]
    sep_token_id: int = 102  # [SEP]
    mask_token_id: int = 103  # [MASK]
    vocab_size: int = 30522  # bert-base-uncased

    # Padding configuration
    pad_to_multiple_of: int = 8  # Tensor core efficiency


class BaseCollator(ABC):
    """Abstract base class for collators."""

    def __init__(self, config: BaseCollatorConfig) -> None:
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.cls_token_id = config.cls_token_id
        self.sep_token_id = config.sep_token_id
        self.mask_token_id = config.mask_token_id
        self.vocab_size = config.vocab_size
        self.pad_to_multiple_of = config.pad_to_multiple_of

        # Set of special tokens that should never be masked
        self.special_token_ids = {
            self.pad_token_id,
            self.cls_token_id,
            self.sep_token_id,
        }

    @abstractmethod
    def __call__(self, samples: list[dict[str, Any]]) -> PreTokenizedBatch:
        """Collate samples into a batch."""
        ...

    def _pad_sequences(
        self,
        sequences: list[torch.Tensor],
        padding_value: int = 0,
    ) -> torch.Tensor:
        """Pad sequences to the same length."""
        max_length = max(len(seq) for seq in sequences)

        # Round up to multiple for tensor core efficiency
        if self.pad_to_multiple_of > 1:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_length),
            padding_value,
            dtype=sequences[0].dtype,
        )

        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq

        return padded

    def _create_attention_mask(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """Create attention mask (1 for real tokens, 0 for padding)."""
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        return (input_ids != pad_token_id).long()

    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create mask for special tokens (1 for special, 0 for normal)."""
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            mask = mask | (input_ids == token_id)
        return mask
