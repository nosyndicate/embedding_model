"""DataLoader factory for the data pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from embedding_trainer.data.base import CollatorProtocol, DatasetProtocol


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader."""

    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.num_workers > 0 and self.prefetch_factor < 1:
            raise ValueError(
                "prefetch_factor must be >= 1 when num_workers > 0, "
                f"got {self.prefetch_factor}"
            )


def _build_loader_kwargs(
    config: DataLoaderConfig,
    collator: CollatorProtocol | Callable[[list[dict[str, Any]]], Any],
) -> dict[str, Any]:
    """Build common DataLoader kwargs."""
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "collate_fn": collator,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and torch.cuda.is_available(),
        "drop_last": config.drop_last,
    }

    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    return loader_kwargs


def create_dataloader(
    dataset: DatasetProtocol | Dataset[Any],
    collator: CollatorProtocol | Callable[[list[dict[str, Any]]], Any],
    config: DataLoaderConfig | None = None,
    sampler: Sampler | None = None,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset and collator.

    Args:
        dataset: Map-style dataset to load from
        collator: Collator function to batch samples
        config: DataLoader configuration
        sampler: Optional sampler (mutually exclusive with shuffle)

    Returns:
        Configured DataLoader instance
    """
    if config is None:
        config = DataLoaderConfig()

    loader_kwargs = _build_loader_kwargs(config, collator)

    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = config.shuffle

    return DataLoader(dataset, **loader_kwargs)


def create_distributed_dataloader(
    dataset: DatasetProtocol | Dataset[Any],
    collator: CollatorProtocol | Callable[[list[dict[str, Any]]], Any],
    config: DataLoaderConfig | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DistributedSampler]:
    """
    Create a DataLoader for distributed training (map-style datasets only).

    Args:
        dataset: Dataset to load from
        collator: Collator function to batch samples
        config: DataLoader configuration
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Tuple of (DataLoader, DistributedSampler).
        The caller must call sampler.set_epoch(epoch) once per epoch for
        epoch-dependent shuffling.
    """
    if config is None:
        config = DataLoaderConfig()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
    )
    loader_kwargs = _build_loader_kwargs(config, collator)
    loader_kwargs["sampler"] = sampler

    return DataLoader(dataset, **loader_kwargs), sampler
