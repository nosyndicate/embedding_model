"""Base protocols and dataclasses for the data pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Required, TypedDict, runtime_checkable

import numpy as np
import torch

# Shard format constants
HEADER_SIZE = 256  # Number of int32 values in header
MAGIC_NUMBER = 20240520
BERT_VERSION = 2
ROBERTA_VERSION = 3

LABEL_IGNORE_ID = -100  # Used to ignore tokens in loss computation


@dataclass
class ShardHeader:
    """Header information from a shard file."""

    magic: int
    version: int
    token_count: int
    vocab_size: int
    pad_id: int
    cls_id: int
    sep_id: int
    mask_id: int

    @classmethod
    def from_array(cls, header: np.ndarray) -> ShardHeader:
        """Parse header from numpy array."""
        return cls(
            magic=int(header[0]),
            version=int(header[1]),
            token_count=int(header[2]),
            vocab_size=int(header[3]),
            pad_id=int(header[4]),
            cls_id=int(header[5]),
            sep_id=int(header[6]),
            mask_id=int(header[7]),
        )


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

    input_ids: Required[torch.Tensor]  # (batch_size, seq_length)
    attention_mask: Required[torch.Tensor]  # (batch_size, seq_length)
    labels: Required[torch.Tensor]  # (batch_size, seq_length) for MLM


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
