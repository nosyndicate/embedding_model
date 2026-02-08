"""Tests for FlatTokenDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from embedding_trainer.data.base import HEADER_SIZE, MAGIC_NUMBER, ROBERTA_VERSION
from embedding_trainer.data.datasets.flat_tokens import (
    FlatTokenConfig,
    FlatTokenDataset,
)
from embedding_trainer.data.registry import DATASET_REGISTRY


def create_shard(
    path: Path,
    num_tokens: int = 2048,
    *,
    magic: int = MAGIC_NUMBER,
    vocab_size: int = 50265,
    pad_id: int = 1,
    sep_id: int = 2,
) -> np.ndarray:
    """Create a test shard and return the token array written."""
    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = magic
    header[1] = ROBERTA_VERSION
    header[2] = num_tokens
    header[3] = vocab_size
    header[4] = pad_id
    header[5] = 0  # cls_id (unused for RoBERTa)
    header[6] = sep_id
    header[7] = 103  # mask_id

    tokens = np.random.default_rng(42).integers(
        100, vocab_size, size=num_tokens, dtype=np.uint16
    )
    # Sprinkle </s> separators every ~256 tokens
    for i in range(256, num_tokens, 256):
        tokens[i] = sep_id

    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())

    return tokens


# ---------------------------------------------------------------------------
# FlatTokenConfig
# ---------------------------------------------------------------------------


class TestFlatTokenConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = FlatTokenConfig(data_dir=str(tmp_path))
        assert cfg.max_seq_length == 512
        assert cfg.split == "train"

    def test_invalid_max_seq_length(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_seq_length must be >= 1"):
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=0)

        with pytest.raises(ValueError, match="max_seq_length must be >= 1"):
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=-5)


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------


class TestShardDiscovery:
    def test_missing_data_dir(self, tmp_path: Path) -> None:
        cfg = FlatTokenConfig(data_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            FlatTokenDataset(cfg)

    def test_no_shards_found(self, tmp_path: Path) -> None:
        cfg = FlatTokenConfig(data_dir=str(tmp_path), split="train")
        with pytest.raises(FileNotFoundError, match="No shards found"):
            FlatTokenDataset(cfg)

    def test_invalid_magic_number(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", magic=99999)
        cfg = FlatTokenConfig(data_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Invalid magic number"):
            FlatTokenDataset(cfg)

    def test_only_matching_split(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin")
        create_shard(tmp_path / "fineweb_val_000000.bin")

        ds = FlatTokenDataset(FlatTokenConfig(data_dir=str(tmp_path), split="train"))
        assert ds.num_shards == 1

    def test_discovers_multiple_shards_sorted(self, tmp_path: Path) -> None:
        for i in range(3):
            create_shard(tmp_path / f"fineweb_train_{i:06d}.bin")

        ds = FlatTokenDataset(FlatTokenConfig(data_dir=str(tmp_path)))
        assert ds.num_shards == 3


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_sequence_count_single_shard(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=2048)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=512)
        )
        assert len(ds) == 2048 // 512

    def test_remainder_tokens_dropped(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=1000)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=512)
        )
        # 1000 // 512 == 1 (488 remainder tokens dropped)
        assert len(ds) == 1

    def test_sequence_count_multiple_shards(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=1024)
        create_shard(tmp_path / "fineweb_train_000001.bin", num_tokens=2048)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=512)
        )
        assert len(ds) == (1024 // 512) + (2048 // 512)

    def test_getitem_out_of_range(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=1024)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=512)
        )

        with pytest.raises(IndexError):
            ds[len(ds)]

        with pytest.raises(IndexError):
            ds[-1]

    def test_getitem_returns_correct_tokens(self, tmp_path: Path) -> None:
        tokens = create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=2048)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=512)
        )

        for i in range(len(ds)):
            batch = ds[i]
            expected = tokens[i * 512 : (i + 1) * 512].astype(np.int64)
            assert torch.equal(batch["input_ids"], torch.from_numpy(expected))

    def test_getitem_returns_correct_shape_and_dtype(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=1024)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=128)
        )

        batch = ds[0]
        assert batch["input_ids"].shape == (128,)
        assert batch["input_ids"].dtype == torch.int64

    def test_getitem_cross_shard_boundary(self, tmp_path: Path) -> None:
        """Sequences near shard boundaries should map to the correct shard."""
        tokens_0 = create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=1024)
        tokens_1 = create_shard(tmp_path / "fineweb_train_000001.bin", num_tokens=1024)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=512)
        )

        # Last sequence in shard 0
        expected_last_0 = tokens_0[512:1024].astype(np.int64)
        assert torch.equal(ds[1]["input_ids"], torch.from_numpy(expected_last_0))

        # First sequence in shard 1
        expected_first_1 = tokens_1[0:512].astype(np.int64)
        assert torch.equal(ds[2]["input_ids"], torch.from_numpy(expected_first_1))


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_header(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin")
        ds = FlatTokenDataset(FlatTokenConfig(data_dir=str(tmp_path)))
        assert ds.header is not None
        assert ds.header.magic == MAGIC_NUMBER

    def test_total_tokens(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=1000)
        create_shard(tmp_path / "fineweb_train_000001.bin", num_tokens=2000)
        ds = FlatTokenDataset(FlatTokenConfig(data_dir=str(tmp_path)))
        assert ds.total_tokens == 3000

    def test_repr(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=1024)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=512)
        )
        r = repr(ds)
        assert "FlatTokenDataset" in r
        assert "num_shards=1" in r


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered(self) -> None:
        assert "flat_tokens" in DATASET_REGISTRY


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataLoaderIntegration:
    def test_works_with_dataloader(self, tmp_path: Path) -> None:
        create_shard(tmp_path / "fineweb_train_000000.bin", num_tokens=2048)
        ds = FlatTokenDataset(
            FlatTokenConfig(data_dir=str(tmp_path), max_seq_length=128)
        )

        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        assert batch["input_ids"].shape == (4, 128)
        assert batch["input_ids"].dtype == torch.int64
