"""Masked Language Model (MLM) collator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from embedding_trainer.data.base import LABEL_IGNORE_ID, PreTokenizedBatch
from embedding_trainer.data.collators.base import BaseCollator, BaseCollatorConfig
from embedding_trainer.data.registry import COLLATOR_REGISTRY


@dataclass
class MLMCollatorConfig(BaseCollatorConfig):
    """Configuration for MLM collator."""

    mlm_probability: float = 0.30  # Probability of replacing with [MASK]

    @staticmethod
    def from_tokenizer(tokenizer: PreTrainedTokenizerBase) -> MLMCollatorConfig:
        """Create config from a Hugging Face Tokenizer."""
        return MLMCollatorConfig(
            pad_token_id=tokenizer.pad_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            mask_token_id=tokenizer.mask_token_id,
            vocab_size=tokenizer.vocab_size,
        )


@COLLATOR_REGISTRY.register("mlm")
class MLMCollator(BaseCollator):
    """
    Collator for Masked Language Modeling (MLM).

    - 30% of tokens are selected for prediction,
        - See https://arxiv.org/pdf/2202.08005
    - Special tokens ([CLS], [SEP], [PAD]) are never masked
    """

    def __init__(self, config: MLMCollatorConfig) -> None:
        super().__init__(config)
        self.mlm_probability = config.mlm_probability

    def __call__(self, samples: list[dict[str, Tensor]]) -> PreTokenizedBatch:
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
        labels[~masked_indices] = LABEL_IGNORE_ID
        input_ids[masked_indices] = self.mask_token_id  # Default to [MASK]

        return input_ids, labels
