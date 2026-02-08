"""DataLoader factory for the data pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from embedding_trainer.data.base import CollatorProtocol, DatasetProtocol


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader."""

    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = True


def create_dataloader(
    dataset: DatasetProtocol | Dataset,
    collator: CollatorProtocol | Callable[[list[dict[str, Any]]], Any],
    config: DataLoaderConfig | None = None,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset and collator.

    Args:
        dataset: Dataset to load from (IterableDataset or map-style)
        collator: Collator function to batch samples
        config: DataLoader configuration

    Returns:
        Configured DataLoader instance
    """
    if config is None:
        config = DataLoaderConfig()

    # Determine if this is an iterable dataset
    is_iterable = isinstance(dataset, IterableDataset)

    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "collate_fn": collator,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
        "drop_last": config.drop_last,
    }

    # prefetch_factor only valid when num_workers > 0
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    # Iterable datasets don't support shuffle (handled internally)
    if not is_iterable:
        loader_kwargs["shuffle"] = True

    return DataLoader(dataset, **loader_kwargs)


def create_distributed_dataloader(
    dataset: DatasetProtocol | IterableDataset,
    collator: CollatorProtocol | Callable[[list[dict[str, Any]]], Any],
    config: DataLoaderConfig | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for distributed training.

    For IterableDatasets like PreTokenizedDataset, distributed sharding
    is handled internally by the dataset. For map-style datasets,
    a DistributedSampler is used.

    Args:
        dataset: Dataset to load from
        collator: Collator function to batch samples
        config: DataLoader configuration
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Configured DataLoader for distributed training
    """
    if config is None:
        config = DataLoaderConfig()

    is_iterable = isinstance(dataset, IterableDataset)

    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "collate_fn": collator,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
        "drop_last": config.drop_last,
    }

    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    if not is_iterable:
        # Use DistributedSampler for map-style datasets
        from torch.utils.data.distributed import DistributedSampler

        sampler: DistributedSampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=config.drop_last,
        )
        loader_kwargs["sampler"] = sampler
    # For iterable datasets, distributed sharding is internal

    return DataLoader(dataset, **loader_kwargs)
