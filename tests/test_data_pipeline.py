"""Tests for the data pipeline components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from embedding_trainer.data.collators.electra import ELECTRACollator
from embedding_trainer.data.collators.mlm import MLMCollator, MLMCollatorConfig
from embedding_trainer.data.collators.span_corruption import SpanCorruptionCollator
from embedding_trainer.data.datasets.pretokenized import (
    BERT_VERSION,
    HEADER_SIZE,
    MAGIC_NUMBER,
    PreTokenizedConfig,
    PreTokenizedDataset,
)
from embedding_trainer.data.loader import DataLoaderConfig, create_dataloader
from embedding_trainer.data.registry import COLLATOR_REGISTRY, DATASET_REGISTRY


def create_test_shard(path: Path, num_tokens: int = 10000) -> None:
    """Create a test shard file with random tokens."""
    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = MAGIC_NUMBER
    header[1] = BERT_VERSION
    header[2] = num_tokens
    header[3] = 30522  # vocab_size
    header[4] = 0  # pad_id
    header[5] = 101  # cls_id
    header[6] = 102  # sep_id
    header[7] = 103  # mask_id

    # Create tokens with [CLS] ... [SEP] pattern
    tokens = np.random.randint(1000, 30000, size=num_tokens, dtype=np.uint16)
    # Insert some [CLS] and [SEP] tokens
    for i in range(0, num_tokens, 512):
        if i < num_tokens:
            tokens[i] = 101  # [CLS]
        if i + 511 < num_tokens:
            tokens[i + 511] = 102  # [SEP]

    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


class TestRegistry:
    """Test the registry system."""

    def test_dataset_registry(self) -> None:
        assert "pretokenized" in DATASET_REGISTRY
        assert DATASET_REGISTRY.get("pretokenized") is PreTokenizedDataset

    def test_collator_registry(self) -> None:
        assert "mlm" in COLLATOR_REGISTRY
        assert "electra" in COLLATOR_REGISTRY
        assert "span_corruption" in COLLATOR_REGISTRY


class TestPreTokenizedDataset:
    """Test the pre-tokenized dataset."""

    def test_invalid_packing_mode_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid packing_mode"):
            PreTokenizedConfig(data_dir="/tmp", packing_mode="invalid")

    def test_invalid_sampling_strategy_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid sampling_strategy"):
            PreTokenizedConfig(data_dir="/tmp", sampling_strategy="invalid")

    def test_load_shards(self, tmp_path: Path) -> None:
        # Create test shards
        for i in range(3):
            split = "val" if i == 0 else "train"
            shard_path = tmp_path / f"fineweb_{split}_{i:06d}.bin"
            create_test_shard(shard_path, num_tokens=5000)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            split="train",
        )
        dataset = PreTokenizedDataset(config)

        assert dataset.num_shards == 2  # Only train shards
        assert dataset.header is not None
        assert dataset.header.vocab_size == 30522

    def test_iteration(self, tmp_path: Path) -> None:
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=2000)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            split="train",
        )
        dataset = PreTokenizedDataset(config)

        samples = list(dataset)
        # 2000 tokens / 512 = 3 full sequences (with remainder)
        assert len(samples) >= 3

        for sample in samples:
            assert "input_ids" in sample
            assert sample["input_ids"].shape == (512,)
            assert sample["input_ids"].dtype == torch.int64

    def test_set_epoch(self, tmp_path: Path) -> None:
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            shuffle=True,
        )
        dataset = PreTokenizedDataset(config)

        dataset.set_epoch(0)
        samples_epoch0 = [s["input_ids"][:10].tolist() for s in list(dataset)[:3]]

        dataset.set_epoch(1)
        samples_epoch1 = [s["input_ids"][:10].tolist() for s in list(dataset)[:3]]

        # With only one shard, shuffling doesn't change order
        # but the epoch should be tracked
        assert dataset._epoch == 1


class TestSamplingStrategies:
    """Test the sampling strategy options."""

    def test_fixed_strategy_same_across_epochs(self, tmp_path: Path) -> None:
        """sampling_strategy='fixed' produces identical sequences epoch 0 vs epoch 1."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=2000)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            sampling_strategy="fixed",
        )
        dataset = PreTokenizedDataset(config)

        dataset.set_epoch(0)
        samples_epoch0 = [s["input_ids"].tolist() for s in dataset]

        dataset.set_epoch(1)
        samples_epoch1 = [s["input_ids"].tolist() for s in dataset]

        assert samples_epoch0 == samples_epoch1

    def test_epoch_offset_strategy_different_across_epochs(
        self, tmp_path: Path
    ) -> None:
        """sampling_strategy='epoch_offset' produces different first tokens each epoch."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=5000)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            sampling_strategy="epoch_offset",
            num_offset_phases=4,
        )
        dataset = PreTokenizedDataset(config)

        first_tokens = []
        for epoch in range(4):
            dataset.set_epoch(epoch)
            samples = list(dataset)
            first_tokens.append(samples[0]["input_ids"][0].item())

        # Each epoch should have a different starting offset (thus different first token)
        assert len(set(first_tokens)) == 4

    def test_epoch_offset_wraps_after_num_phases(self, tmp_path: Path) -> None:
        """epoch 4 should produce same as epoch 0 (with num_offset_phases=4)."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=5000)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            sampling_strategy="epoch_offset",
            num_offset_phases=4,
        )
        dataset = PreTokenizedDataset(config)

        dataset.set_epoch(0)
        samples_epoch0 = [s["input_ids"].tolist() for s in dataset]

        dataset.set_epoch(4)
        samples_epoch4 = [s["input_ids"].tolist() for s in dataset]

        assert samples_epoch0 == samples_epoch4

    def test_random_strategy_different_across_epochs(self, tmp_path: Path) -> None:
        """sampling_strategy='random' produces different sequences each epoch."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=5000)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            sampling_strategy="random",
        )
        dataset = PreTokenizedDataset(config)

        dataset.set_epoch(0)
        samples_epoch0 = [s["input_ids"].tolist() for s in dataset]

        dataset.set_epoch(1)
        samples_epoch1 = [s["input_ids"].tolist() for s in dataset]

        # Sequences should be different (random positions)
        assert samples_epoch0 != samples_epoch1

    def test_random_strategy_deterministic_same_seed(self, tmp_path: Path) -> None:
        """Same seed+epoch = same sequences (deterministic)."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=5000)

        config1 = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            sampling_strategy="random",
            seed=42,
        )
        dataset1 = PreTokenizedDataset(config1)

        config2 = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            sampling_strategy="random",
            seed=42,
        )
        dataset2 = PreTokenizedDataset(config2)

        dataset1.set_epoch(0)
        samples1 = [s["input_ids"].tolist() for s in dataset1]

        dataset2.set_epoch(0)
        samples2 = [s["input_ids"].tolist() for s in dataset2]

        assert samples1 == samples2

    def test_shuffle_indices_strategy_different_order(self, tmp_path: Path) -> None:
        """sampling_strategy='shuffle_indices' shuffles sequence order."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=5000)

        config_fixed = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            sampling_strategy="fixed",
        )
        dataset_fixed = PreTokenizedDataset(config_fixed)
        samples_fixed = [s["input_ids"].tolist() for s in dataset_fixed]

        config_shuffle = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            sampling_strategy="shuffle_indices",
        )
        dataset_shuffle = PreTokenizedDataset(config_shuffle)
        dataset_shuffle.set_epoch(0)
        samples_shuffled = [s["input_ids"].tolist() for s in dataset_shuffle]

        # Same set of sequences, different order
        assert len(samples_fixed) == len(samples_shuffled)
        assert set(tuple(s) for s in samples_fixed) == set(
            tuple(s) for s in samples_shuffled
        )
        # Order should be different (shuffled)
        assert samples_fixed != samples_shuffled


class TestPackingModes:
    """Test the packing mode options."""

    def test_doc_aware_sequences_start_with_cls(self, tmp_path: Path) -> None:
        """packing_mode='doc_aware' - first token of each sequence is [CLS] (101)."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=5000)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            packing_mode="doc_aware",
        )
        dataset = PreTokenizedDataset(config)

        samples = list(dataset)
        for sample in samples:
            # First token should be [CLS] (101)
            assert sample["input_ids"][0].item() == 101

    def test_doc_aware_pads_short_sequences(self, tmp_path: Path) -> None:
        """Short final sequence is padded with [PAD] tokens (0)."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        # Create a small shard that will have short final sequence
        create_test_shard(shard_path, num_tokens=600)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            packing_mode="doc_aware",
        )
        dataset = PreTokenizedDataset(config)

        samples = list(dataset)
        # All sequences should be max_seq_length
        for sample in samples:
            assert sample["input_ids"].shape == (512,)

        # Check that padding exists in at least one sequence
        has_padding = any((sample["input_ids"] == 0).any().item() for sample in samples)
        assert has_padding

    def test_flat_mode_default_behavior(self, tmp_path: Path) -> None:
        """packing_mode='flat' behaves like original implementation."""
        shard_path = tmp_path / "fineweb_train_000000.bin"
        create_test_shard(shard_path, num_tokens=2000)

        config = PreTokenizedConfig(
            data_dir=str(tmp_path),
            max_seq_length=512,
            packing_mode="flat",
        )
        dataset = PreTokenizedDataset(config)

        samples = list(dataset)
        # 2000 tokens / 512 = 3 full sequences
        assert len(samples) >= 3

        for sample in samples:
            assert sample["input_ids"].shape == (512,)
