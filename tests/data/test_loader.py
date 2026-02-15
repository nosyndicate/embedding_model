"""Tests for DataLoader factory helpers."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from embedding_trainer.data.loader import (
    DataLoaderConfig,
    create_dataloader,
    create_distributed_dataloader,
)


class DummyDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, size: int = 64) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {"input_ids": torch.tensor([idx], dtype=torch.int32)}


def collator(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {"input_ids": torch.cat([sample["input_ids"] for sample in samples], dim=0)}


class TestDataLoaderConfigValidation:
    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            DataLoaderConfig(batch_size=0)

    def test_invalid_num_workers_raises(self) -> None:
        with pytest.raises(ValueError, match="num_workers must be >= 0"):
            DataLoaderConfig(num_workers=-1)

    def test_invalid_prefetch_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="prefetch_factor must be >= 1"):
            DataLoaderConfig(num_workers=2, prefetch_factor=0)


class TestCreateDataLoader:
    def test_shuffle_true_uses_random_sampler(self) -> None:
        ds = DummyDataset()
        loader = create_dataloader(
            ds,
            collator,
            DataLoaderConfig(num_workers=0, shuffle=True, batch_size=4),
        )
        assert isinstance(loader.sampler, RandomSampler)

    def test_shuffle_false_uses_sequential_sampler(self) -> None:
        ds = DummyDataset()
        loader = create_dataloader(
            ds,
            collator,
            DataLoaderConfig(num_workers=0, shuffle=False, batch_size=4),
        )
        assert isinstance(loader.sampler, SequentialSampler)

    def test_num_workers_zero_ignores_prefetch_factor(self) -> None:
        ds = DummyDataset()
        loader = create_dataloader(
            ds,
            collator,
            DataLoaderConfig(num_workers=0, prefetch_factor=16, batch_size=4),
        )
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (4,)


class TestCreateDistributedDataLoader:
    def test_returns_loader_and_distributed_sampler(self) -> None:
        ds = DummyDataset(size=16)
        loader, sampler = create_distributed_dataloader(
            ds,
            collator,
            DataLoaderConfig(batch_size=2, num_workers=0, shuffle=True),
            rank=0,
            world_size=1,
        )

        assert isinstance(sampler, DistributedSampler)
        assert loader.sampler is sampler
        assert sampler.num_replicas == 1
        assert sampler.rank == 0
        assert sampler.drop_last is True

    def test_order_changes_across_epochs_when_set_epoch_called(self) -> None:
        ds = DummyDataset(size=32)
        loader, sampler = create_distributed_dataloader(
            ds,
            collator,
            DataLoaderConfig(
                batch_size=1,
                num_workers=0,
                shuffle=True,
                drop_last=False,
            ),
            rank=0,
            world_size=1,
        )

        sampler.set_epoch(0)
        order_epoch0 = [batch["input_ids"].item() for batch in loader]

        sampler.set_epoch(1)
        order_epoch1 = [batch["input_ids"].item() for batch in loader]

        assert order_epoch0 != order_epoch1

    def test_order_stays_same_without_set_epoch(self) -> None:
        ds = DummyDataset(size=32)
        loader, _sampler = create_distributed_dataloader(
            ds,
            collator,
            DataLoaderConfig(
                batch_size=1,
                num_workers=0,
                shuffle=True,
                drop_last=False,
            ),
            rank=0,
            world_size=1,
        )

        order_first_pass = [batch["input_ids"].item() for batch in loader]
        order_second_pass = [batch["input_ids"].item() for batch in loader]

        assert order_first_pass == order_second_pass
