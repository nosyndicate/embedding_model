"""Base protocols and dataclasses for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, TypedDict, runtime_checkable

import torch


@dataclass
class ShardInfo:
    """Information about a data shard."""

    path: Path
    token_count: int
    split: str  # "train" or "val"

    @classmethod
    def from_header(cls, path: Path, header: dict[str, int], split: str) -> ShardInfo:
        """Create ShardInfo from shard header."""
        return cls(path=path, token_count=header["token_count"], split=split)


class PreTokenizedBatch(TypedDict, total=False):
    """Batch of pre-tokenized data."""

    input_ids: torch.Tensor  # (batch_size, seq_length)
    attention_mask: torch.Tensor  # (batch_size, seq_length)
    labels: torch.Tensor  # (batch_size, seq_length) for MLM


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for datasets."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset samples."""
        ...

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reproducible shuffling."""
        ...


@runtime_checkable
class CollatorProtocol(Protocol):
    """Protocol for collators."""

    def __call__(self, samples: list[dict[str, Any]]) -> PreTokenizedBatch:
        """Collate samples into a batch."""
        ...
