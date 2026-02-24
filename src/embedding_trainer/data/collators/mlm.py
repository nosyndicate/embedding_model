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
    seed: int = 0
    epoch: int = 0

    @staticmethod
    def from_tokenizer(
        tokenizer: PreTrainedTokenizerBase,
        *,
        seed: int = 0,
        mlm_probability: float = 0.30,
        epoch: int = 0,
    ) -> MLMCollatorConfig:
        """Create config from a Hugging Face Tokenizer."""
        return MLMCollatorConfig(
            pad_token_id=tokenizer.pad_token_id,
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            mask_token_id=tokenizer.mask_token_id,
            vocab_size=tokenizer.vocab_size,
            seed=seed,
            mlm_probability=mlm_probability,
            epoch=epoch,
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
        self.seed = config.seed
        self.epoch = config.epoch

    def set_epoch(self, epoch: int) -> None:
        """Set epoch used by deterministic masking."""
        self.epoch = epoch

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
        sample_indices = self._get_sample_indices(samples)

        # Pad sequences
        input_ids = self._pad_sequences(input_ids_list, self.pad_token_id)

        # Create attention mask before masking
        attention_mask = self._create_attention_mask(input_ids)

        # Apply MLM masking
        input_ids, labels = self._mask_tokens(input_ids, sample_indices)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_tokens(
        self,
        input_ids: torch.Tensor,
        sample_indices: torch.Tensor | None = None,
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

        if sample_indices is None:
            sample_indices = torch.arange(
                input_ids.shape[0],
                device=input_ids.device,
                dtype=torch.int64,
            )
        else:
            sample_indices = sample_indices.to(
                device=input_ids.device, dtype=torch.int64
            )
            if (
                sample_indices.ndim != 1
                or sample_indices.shape[0] != input_ids.shape[0]
            ):
                raise ValueError(
                    "sample_indices must be a 1D tensor with one entry per batch row."
                )

        # Get special tokens mask
        special_tokens_mask = self._get_special_tokens_mask(input_ids)

        # Also don't mask padding.
        padding_mask = input_ids == self.pad_token_id
        eligible_mask = ~(special_tokens_mask | padding_mask)

        if self.mlm_probability <= 0.0:
            masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        elif self.mlm_probability >= 1.0:
            masked_indices = eligible_mask
        else:
            p = self._deterministic_uniform(
                sample_indices, input_ids.shape[1], input_ids.device
            )
            masked_indices = (p < self.mlm_probability) & eligible_mask

        # Set labels to -100 for non-masked tokens (will be ignored in loss)
        labels[~masked_indices] = LABEL_IGNORE_ID
        input_ids[masked_indices] = self.mask_token_id  # Default to [MASK]

        return input_ids, labels

    def _get_sample_indices(self, samples: list[dict[str, Tensor]]) -> torch.Tensor:
        sample_indices: list[int] = []
        for row_idx, sample in enumerate(samples):
            raw_idx = sample.get("sample_idx")
            if raw_idx is None:
                sample_indices.append(row_idx)
                continue
            if isinstance(raw_idx, torch.Tensor):
                if raw_idx.numel() != 1:
                    raise ValueError(
                        "sample_idx tensor must contain a single scalar value."
                    )
                sample_indices.append(int(raw_idx.item()))
                continue
            sample_indices.append(int(raw_idx))
        return torch.tensor(sample_indices, dtype=torch.int64)

    def _deterministic_uniform(
        self,
        sample_indices: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate deterministic pseudo-random values in [0, 1)."""
        modulus = 2_147_483_647  # 2^31 - 1, prime.

        sample_grid = sample_indices.unsqueeze(1)
        position_grid = torch.arange(
            seq_len, device=device, dtype=torch.int64
        ).unsqueeze(0)

        x = (
            sample_grid * 1_048_583
            + position_grid * 1_308_049
            + (self.seed % modulus) * 4_447_961
            + (self.epoch % modulus) * 22_695_477
            + 1_013_904_223
        ) % modulus
        x = (x * 48_271) % modulus
        return x.to(torch.float64) / float(modulus)
