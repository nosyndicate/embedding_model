"""Masked Language Model (MLM) collator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from embedding_trainer.data.base import PreTokenizedBatch
from embedding_trainer.data.collators.base import BaseCollator, BaseCollatorConfig
from embedding_trainer.data.registry import COLLATOR_REGISTRY


@dataclass
class MLMCollatorConfig(BaseCollatorConfig):
    """Configuration for MLM collator."""

    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8  # Probability of replacing with [MASK]
    random_replace_prob: float = 0.1  # Probability of replacing with random token
    # Remaining probability (0.1) keeps original token


@COLLATOR_REGISTRY.register("mlm")
class MLMCollator(BaseCollator):
    """
    Collator for Masked Language Modeling (MLM).

    Implements BERT-style masking:
    - 15% of tokens are selected for prediction
    - Of those: 80% replaced with [MASK], 10% with random token, 10% unchanged
    - Special tokens ([CLS], [SEP], [PAD]) are never masked
    """

    def __init__(self, config: MLMCollatorConfig | None = None) -> None:
        if config is None:
            config = MLMCollatorConfig()
        super().__init__(config)
        self.mlm_probability = config.mlm_probability
        self.mask_replace_prob = config.mask_replace_prob
        self.random_replace_prob = config.random_replace_prob

    def __call__(self, samples: list[dict[str, Any]]) -> PreTokenizedBatch:
        """
        Collate samples and apply MLM masking.

        Args:
            samples: List of dicts with "input_ids" key

        Returns:
            Batch with input_ids (masked), labels, and attention_mask
        """
        # Extract input_ids
        input_ids_list = [sample["input_ids"] for sample in samples]

        # Pad sequences
        input_ids = self._pad_sequences(input_ids_list, self.pad_token_id)

        # Create attention mask before masking
        attention_mask = self._create_attention_mask(input_ids)

        # Apply MLM masking
        input_ids, labels = self._mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_tokens(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MLM masking to input_ids.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_length)

        Returns:
            Tuple of (masked_input_ids, labels)
            Labels has -100 for non-masked positions (ignored by loss)
        """
        labels = input_ids.clone()
        input_ids = input_ids.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)

        # Get special tokens mask and set their probability to 0
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        # Also don't mask padding
        padding_mask = input_ids == self.pad_token_id
        probability_matrix.masked_fill_(padding_mask, 0.0)

        # Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (will be ignored in loss)
        labels[~masked_indices] = -100

        # Determine what to do with masked tokens
        # 80% -> [MASK]
        indices_replaced = (
            torch.bernoulli(
                torch.full(input_ids.shape, self.mask_replace_prob)
            ).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.mask_token_id

        # 10% -> random token
        random_mask = (
            torch.bernoulli(
                torch.full(
                    input_ids.shape,
                    self.random_replace_prob / (1 - self.mask_replace_prob),
                )
            ).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_tokens = torch.randint(
            0, self.vocab_size, input_ids.shape, dtype=input_ids.dtype
        )
        input_ids[random_mask] = random_tokens[random_mask]

        # 10% -> keep original (already done, since we cloned)

        return input_ids, labels
