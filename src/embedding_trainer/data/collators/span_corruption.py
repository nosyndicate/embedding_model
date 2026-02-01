"""Span corruption collator for T5-style denoising."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import torch

from embedding_trainer.data.collators.base import BaseCollator, BaseCollatorConfig
from embedding_trainer.data.registry import COLLATOR_REGISTRY


class SpanCorruptionBatch(TypedDict):
    """Batch for span corruption (T5-style) training."""

    input_ids: torch.Tensor  # Corrupted input with sentinel tokens
    attention_mask: torch.Tensor  # Attention mask for encoder
    labels: torch.Tensor  # Decoder targets (masked spans with sentinels)
    decoder_attention_mask: torch.Tensor  # Attention mask for decoder


@dataclass
class SpanCorruptionCollatorConfig(BaseCollatorConfig):
    """Configuration for span corruption collator."""

    noise_density: float = 0.15  # Fraction of tokens to corrupt
    mean_span_length: float = 3.0  # Average length of masked spans
    sentinel_start_id: int = 30000  # First sentinel token ID
    num_sentinels: int = 100  # Number of sentinel tokens available
    decoder_start_token_id: int = 0  # Decoder start token


@COLLATOR_REGISTRY.register("span_corruption")
class SpanCorruptionCollator(BaseCollator):
    """
    Collator for T5-style span corruption.

    Corrupts contiguous spans of tokens and replaces them with sentinel tokens.
    The decoder target contains the masked spans prefixed by their sentinel tokens.

    Example:
        Input:  "The quick brown fox jumps"
        Tokens: [The, quick, brown, fox, jumps]
        Corrupted: [The, <S0>, fox, <S1>]
        Target: [<S0>, quick, brown, <S1>, jumps]
    """

    def __init__(self, config: SpanCorruptionCollatorConfig | None = None) -> None:
        if config is None:
            config = SpanCorruptionCollatorConfig()
        super().__init__(config)
        self.noise_density = config.noise_density
        self.mean_span_length = config.mean_span_length
        self.sentinel_start_id = config.sentinel_start_id
        self.num_sentinels = config.num_sentinels
        self.decoder_start_token_id = config.decoder_start_token_id

    def __call__(self, samples: list[dict[str, Any]]) -> SpanCorruptionBatch:
        """
        Collate samples and apply span corruption.

        Args:
            samples: List of dicts with "input_ids" key

        Returns:
            SpanCorruptionBatch with corrupted inputs and decoder targets
        """
        batch_inputs = []
        batch_targets = []

        for sample in samples:
            input_ids = sample["input_ids"]
            corrupted, target = self._corrupt_spans(input_ids)
            batch_inputs.append(corrupted)
            batch_targets.append(target)

        # Pad inputs
        input_ids = self._pad_sequences(batch_inputs, self.pad_token_id)
        attention_mask = self._create_attention_mask(input_ids)

        # Pad targets
        labels = self._pad_sequences(batch_targets, -100)  # -100 for padding in loss
        decoder_attention_mask = (labels != -100).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask,
        }

    def _corrupt_spans(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply span corruption to a single sequence.

        Args:
            input_ids: Token IDs for a single sequence

        Returns:
            Tuple of (corrupted_input, decoder_target)
        """
        length = len(input_ids)

        # Calculate number of spans to corrupt
        num_to_mask = int(length * self.noise_density)
        if num_to_mask == 0:
            # Return with a dummy sentinel if nothing to mask
            sentinel = self.sentinel_start_id
            return input_ids, torch.tensor([sentinel], dtype=input_ids.dtype)

        # Sample span lengths from geometric distribution
        span_lengths = self._sample_span_lengths(num_to_mask)

        # Determine span starts
        span_starts = self._get_span_starts(length, span_lengths)

        if len(span_starts) == 0:
            sentinel = self.sentinel_start_id
            return input_ids, torch.tensor([sentinel], dtype=input_ids.dtype)

        # Build mask of positions to corrupt
        mask = torch.zeros(length, dtype=torch.bool)
        spans = []
        for start, span_len in zip(span_starts, span_lengths[: len(span_starts)]):
            end = min(start + span_len, length)
            mask[start:end] = True
            spans.append((start, end))

        # Build corrupted input (replace spans with sentinels)
        corrupted = []
        target = []
        in_span = False
        span_idx = 0
        current_span_end = -1

        for i, token in enumerate(input_ids):
            if mask[i]:
                if not in_span:
                    # Start of new span
                    in_span = True
                    sentinel = self.sentinel_start_id + (span_idx % self.num_sentinels)
                    corrupted.append(sentinel)
                    target.append(sentinel)
                    span_idx += 1
                    # Find end of this span
                    for start, end in spans:
                        if start <= i < end:
                            current_span_end = end
                            break
                target.append(token.item())
                if i == current_span_end - 1:
                    in_span = False
            else:
                corrupted.append(token.item())
                in_span = False

        corrupted_tensor = torch.tensor(corrupted, dtype=input_ids.dtype)
        target_tensor = torch.tensor(target, dtype=input_ids.dtype)

        return corrupted_tensor, target_tensor

    def _sample_span_lengths(self, num_tokens: int) -> list[int]:
        """Sample span lengths from geometric distribution."""
        lengths = []
        remaining = num_tokens
        while remaining > 0:
            # Geometric distribution with p = 1/mean_span_length
            p = 1.0 / self.mean_span_length
            span_len = max(1, int(torch.distributions.Geometric(p).sample().item()))
            span_len = min(span_len, remaining)
            lengths.append(span_len)
            remaining -= span_len
        return lengths

    def _get_span_starts(
        self,
        length: int,
        span_lengths: list[int],
    ) -> list[int]:
        """Get non-overlapping span start positions."""
        if length == 0 or not span_lengths:
            return []

        # Simple approach: distribute spans evenly
        num_spans = len(span_lengths)
        total_span = sum(span_lengths)

        if total_span >= length:
            # Too much to mask, reduce
            span_lengths = span_lengths[: max(1, length // 3)]
            num_spans = len(span_lengths)

        if num_spans == 0:
            return []

        # Random start positions with minimum gap
        available = length - sum(span_lengths)
        if available <= 0:
            return [0]

        gaps = torch.randint(0, max(1, available // (num_spans + 1)), (num_spans,))
        starts = []
        pos = 0
        for i, span_len in enumerate(span_lengths):
            pos += gaps[i].item()
            if pos + span_len <= length:
                starts.append(pos)
                pos += span_len
            else:
                break

        return starts
