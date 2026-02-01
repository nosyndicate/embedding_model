"""ELECTRA-style collator for replaced token detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import torch

from embedding_trainer.data.collators.base import BaseCollator, BaseCollatorConfig
from embedding_trainer.data.registry import COLLATOR_REGISTRY


class ELECTRABatch(TypedDict):
    """Batch for ELECTRA training."""

    input_ids: torch.Tensor  # Original token IDs
    masked_input_ids: torch.Tensor  # Token IDs with masks for generator
    attention_mask: torch.Tensor  # Attention mask
    mlm_labels: torch.Tensor  # Labels for generator MLM (-100 for non-masked)
    is_masked: torch.Tensor  # Boolean mask of which positions were masked


@dataclass
class ELECTRACollatorConfig(BaseCollatorConfig):
    """Configuration for ELECTRA collator."""

    mlm_probability: float = 0.15


@COLLATOR_REGISTRY.register("electra")
class ELECTRACollator(BaseCollator):
    """
    Collator for ELECTRA training.

    ELECTRA uses a generator-discriminator setup:
    - Generator: MLM task on masked positions
    - Discriminator: Binary classification on whether each token is original or replaced

    This collator prepares data for both:
    - masked_input_ids: Input for generator (with [MASK] tokens)
    - mlm_labels: Labels for generator
    - is_masked: Mask indicating which positions the discriminator should classify
    """

    def __init__(self, config: ELECTRACollatorConfig | None = None) -> None:
        if config is None:
            config = ELECTRACollatorConfig()
        super().__init__(config)
        self.mlm_probability = config.mlm_probability

    def __call__(self, samples: list[dict[str, Any]]) -> ELECTRABatch:
        """
        Collate samples for ELECTRA training.

        Args:
            samples: List of dicts with "input_ids" key

        Returns:
            ELECTRABatch with all required tensors
        """
        # Extract input_ids
        input_ids_list = [sample["input_ids"] for sample in samples]

        # Pad sequences
        input_ids = self._pad_sequences(input_ids_list, self.pad_token_id)

        # Create attention mask
        attention_mask = self._create_attention_mask(input_ids)

        # Create masked version and labels
        masked_input_ids, mlm_labels, is_masked = self._create_electra_inputs(input_ids)

        return {
            "input_ids": input_ids,
            "masked_input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
            "is_masked": is_masked,
        }

    def _create_electra_inputs(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create inputs for ELECTRA training.

        Args:
            input_ids: Original token IDs

        Returns:
            Tuple of (masked_input_ids, mlm_labels, is_masked)
        """
        mlm_labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        # Don't mask padding
        padding_mask = input_ids == self.pad_token_id
        probability_matrix.masked_fill_(padding_mask, 0.0)

        # Sample which tokens to mask
        is_masked = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens
        mlm_labels[~is_masked] = -100

        # Replace masked tokens with [MASK] (ELECTRA always uses [MASK], no random/keep)
        masked_input_ids[is_masked] = self.mask_token_id

        return masked_input_ids, mlm_labels, is_masked
