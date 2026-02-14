"""Tests for the MLM collator."""

from __future__ import annotations

import pytest
import torch

from embedding_trainer.data.base import LABEL_IGNORE_ID
from embedding_trainer.data.collators.mlm import MLMCollator, MLMCollatorConfig
from embedding_trainer.data.registry import COLLATOR_REGISTRY

# ---------------------------------------------------------------------------
# Shared IDs
# ---------------------------------------------------------------------------
PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
MASK_ID = 3
VOCAB_SIZE = 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mlm_config() -> MLMCollatorConfig:
    return MLMCollatorConfig(
        pad_token_id=PAD_ID,
        cls_token_id=CLS_ID,
        sep_token_id=SEP_ID,
        mask_token_id=MASK_ID,
        vocab_size=VOCAB_SIZE,
    )


@pytest.fixture
def collator(mlm_config: MLMCollatorConfig) -> MLMCollator:
    return MLMCollator(mlm_config)


def make_sample(token_ids: list[int]) -> dict[str, torch.Tensor]:
    return {"input_ids": torch.tensor(token_ids, dtype=torch.long)}


# ---------------------------------------------------------------------------
# TestMLMCollatorConfig
# ---------------------------------------------------------------------------
class TestMLMCollatorConfig:
    def test_default_mlm_probability(self, mlm_config: MLMCollatorConfig) -> None:
        assert mlm_config.mlm_probability == 0.30

    def test_inherits_base_fields(self, mlm_config: MLMCollatorConfig) -> None:
        assert mlm_config.pad_token_id == PAD_ID
        assert mlm_config.cls_token_id == CLS_ID
        assert mlm_config.sep_token_id == SEP_ID
        assert mlm_config.mask_token_id == MASK_ID
        assert mlm_config.vocab_size == VOCAB_SIZE


# ---------------------------------------------------------------------------
# TestMLMCollatorCall
# ---------------------------------------------------------------------------
class TestMLMCollatorCall:
    def test_output_keys(self, collator: MLMCollator) -> None:
        batch = [make_sample([CLS_ID, 10, 20, SEP_ID])]
        result = collator(batch)
        assert set(result.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_output_shapes(self, collator: MLMCollator) -> None:
        batch = [
            make_sample([CLS_ID, 10, 20, SEP_ID]),
            make_sample([CLS_ID, 30, 40, 50, SEP_ID]),
        ]
        result = collator(batch)
        batch_size = 2
        seq_len = result["input_ids"].shape[1]
        assert result["input_ids"].shape == (batch_size, seq_len)
        assert result["attention_mask"].shape == (batch_size, seq_len)
        assert result["labels"].shape == (batch_size, seq_len)

    def test_padding_to_multiple_of_8(self, collator: MLMCollator) -> None:
        # Sequences of length 5 should be padded to 8
        batch = [make_sample([CLS_ID, 10, 20, 30, SEP_ID])]
        result = collator(batch)
        assert result["input_ids"].shape[1] == 8

    def test_attention_mask_correct(self, collator: MLMCollator) -> None:
        batch = [
            make_sample([CLS_ID, 10, SEP_ID]),  # length 3 → padded to 8
            make_sample([CLS_ID, 20, 30, 40, SEP_ID]),  # length 5 → padded to 8
        ]
        result = collator(batch)
        attn = result["attention_mask"]
        # First row: 3 real tokens, 5 padding
        assert attn[0, :3].sum().item() == 3
        assert attn[0, 3:].sum().item() == 0
        # Second row: 5 real tokens, 3 padding
        assert attn[1, :5].sum().item() == 5
        assert attn[1, 5:].sum().item() == 0

    def test_variable_length_sequences(self, collator: MLMCollator) -> None:
        batch = [
            make_sample([CLS_ID, 10, SEP_ID]),
            make_sample([CLS_ID, 20, 30, 40, 50, 60, SEP_ID]),
        ]
        result = collator(batch)
        # Max length 7, rounded up to 8
        assert result["input_ids"].shape == (2, 8)
        # Second sample's padding positions should be PAD_ID
        assert result["input_ids"][0, 3:].tolist() == [PAD_ID] * 5


# ---------------------------------------------------------------------------
# TestMaskTokens
# ---------------------------------------------------------------------------
class TestMaskTokens:
    def _make_input(self, token_ids: list[int]) -> torch.Tensor:
        return torch.tensor([token_ids], dtype=torch.long)

    def test_special_tokens_never_masked(self, collator: MLMCollator) -> None:
        # All-special-token row to verify none get masked
        input_ids = self._make_input([CLS_ID, SEP_ID, PAD_ID, PAD_ID])
        masked, labels = collator._mask_tokens(input_ids)
        # No position should become mask_token_id
        assert (masked == MASK_ID).sum().item() == 0
        # All labels should be LABEL_IGNORE_ID
        assert (labels == LABEL_IGNORE_ID).all().item()

    def test_labels_minus_100_for_unmasked(self, collator: MLMCollator) -> None:
        torch.manual_seed(0)
        input_ids = self._make_input([CLS_ID, 10, 20, 30, 40, SEP_ID, PAD_ID, PAD_ID])
        _, labels = collator._mask_tokens(input_ids)
        # Positions not replaced by [MASK] must have label LABEL_IGNORE_ID
        masked_positions = labels != LABEL_IGNORE_ID
        for pos in range(labels.shape[1]):
            if not masked_positions[0, pos]:
                assert labels[0, pos].item() == LABEL_IGNORE_ID

    def test_masked_positions_get_mask_token(self, collator: MLMCollator) -> None:
        # Force all eligible tokens to be masked
        high_prob_collator = MLMCollator(
            MLMCollatorConfig(
                pad_token_id=PAD_ID,
                cls_token_id=CLS_ID,
                sep_token_id=SEP_ID,
                mask_token_id=MASK_ID,
                vocab_size=VOCAB_SIZE,
                mlm_probability=1.0,
            )
        )
        input_ids = self._make_input([CLS_ID, 10, 20, SEP_ID])
        masked, _ = high_prob_collator._mask_tokens(input_ids)
        # Positions 1 and 2 (non-special) must become MASK_ID
        assert masked[0, 1].item() == MASK_ID
        assert masked[0, 2].item() == MASK_ID
        # Special tokens untouched
        assert masked[0, 0].item() == CLS_ID
        assert masked[0, 3].item() == SEP_ID

    def test_labels_preserve_original_for_masked(self, collator: MLMCollator) -> None:
        high_prob_collator = MLMCollator(
            MLMCollatorConfig(
                pad_token_id=PAD_ID,
                cls_token_id=CLS_ID,
                sep_token_id=SEP_ID,
                mask_token_id=MASK_ID,
                vocab_size=VOCAB_SIZE,
                mlm_probability=1.0,
            )
        )
        input_ids = self._make_input([CLS_ID, 10, 20, SEP_ID])
        _, labels = high_prob_collator._mask_tokens(input_ids)
        assert labels[0, 1].item() == 10
        assert labels[0, 2].item() == 20

    def test_masking_probability(self, collator: MLMCollator) -> None:
        torch.manual_seed(42)
        # Large sequence of eligible tokens to get stable statistics
        eligible_tokens = list(range(10, 90))  # 80 tokens, none are special
        input_ids = self._make_input([CLS_ID] + eligible_tokens + [SEP_ID])
        _, labels = collator._mask_tokens(input_ids)
        # Count masked positions among eligible tokens (positions 1..80)
        eligible_labels = labels[0, 1:81]
        n_masked = (eligible_labels != LABEL_IGNORE_ID).sum().item()
        ratio = n_masked / 80
        # Allow ±10% tolerance around 30%
        assert 0.15 <= ratio <= 0.50

    def test_zero_probability_masks_nothing(self) -> None:
        zero_prob_collator = MLMCollator(
            MLMCollatorConfig(
                pad_token_id=PAD_ID,
                cls_token_id=CLS_ID,
                sep_token_id=SEP_ID,
                mask_token_id=MASK_ID,
                vocab_size=VOCAB_SIZE,
                mlm_probability=0.0,
            )
        )
        input_ids = torch.tensor([[CLS_ID, 10, 20, 30, SEP_ID]], dtype=torch.long)
        masked, labels = zero_prob_collator._mask_tokens(input_ids)
        assert (masked == MASK_ID).sum().item() == 0
        assert (labels == LABEL_IGNORE_ID).all().item()

    def test_full_probability_masks_all_eligible(self) -> None:
        full_prob_collator = MLMCollator(
            MLMCollatorConfig(
                pad_token_id=PAD_ID,
                cls_token_id=CLS_ID,
                sep_token_id=SEP_ID,
                mask_token_id=MASK_ID,
                vocab_size=VOCAB_SIZE,
                mlm_probability=1.0,
            )
        )
        input_ids = torch.tensor(
            [[CLS_ID, 10, 20, 30, SEP_ID, PAD_ID, PAD_ID, PAD_ID]], dtype=torch.long
        )
        masked, labels = full_prob_collator._mask_tokens(input_ids)
        # Eligible positions (1, 2, 3) must be masked
        assert masked[0, 1].item() == MASK_ID
        assert masked[0, 2].item() == MASK_ID
        assert masked[0, 3].item() == MASK_ID
        # Special/padding positions untouched
        assert masked[0, 0].item() == CLS_ID
        assert masked[0, 4].item() == SEP_ID
        assert masked[0, 5].item() == PAD_ID


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_registered_as_mlm(self) -> None:
        assert "mlm" in COLLATOR_REGISTRY
        assert COLLATOR_REGISTRY.get("mlm") is MLMCollator
