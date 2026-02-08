"""RoBERTa-style flat token dataset for loading pre-tokenized binary shards."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from embedding_trainer.data.base import (
    HEADER_SIZE,
    MAGIC_NUMBER,
    ShardHeader,
    ShardInfo,
)
from embedding_trainer.data.registry import DATASET_REGISTRY


@dataclass
class FlatTokenConfig:
    """Configuration for flat-token RoBERTa-style dataset."""

    data_dir: str
    max_seq_length: int = 512
    split: str = "train"

    def __post_init__(self) -> None:
        if self.max_seq_length < 1:
            raise ValueError(f"max_seq_length must be >= 1, got {self.max_seq_length}")


@DATASET_REGISTRY.register("flat_tokens")
class FlatTokenDataset(Dataset):
    """
    Map-style dataset that reads pre-tokenized binary shards as a flat token stream
    and slices them into fixed-length, non-overlapping windows.

    Each shard is a binary file with a header (magic number, token count, etc.)
    followed by a flat array of uint16 token IDs. Documents within a shard are
    separated by </s> tokens but are otherwise concatenated end-to-end.

    The dataset treats all shards as one logical token stream and divides it into
    ``total_tokens // max_seq_length`` sequences. Sequences may span document
    boundaries (RoBERTa FULL-SENTENCES approach). Remainder tokens at the end of
    each shard that don't fill a complete window are silently dropped.

    Shard data is accessed via memory-mapped arrays (numpy memmap), created lazily
    per-worker so that forked DataLoader workers each get their own file handles.
    """

    def __init__(self, config: FlatTokenConfig) -> None:
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.data_dir = Path(config.data_dir)
        self.split = config.split

        self._shards: list[ShardInfo] = []
        self._header: ShardHeader | None = None
        self._cumulative_seqs: list[int] = []
        self._total_seqs: int = 0

        # Lazy mmap cache (populated per-worker after fork)
        self._mmap_cache: dict[int, np.ndarray] = {}

        self._discover_shards()
        self._build_index()

    def _discover_shards(self) -> None:
        """Discover and validate shard files in data directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        pattern = f"fineweb_{self.split}_*.bin"
        shard_paths = sorted(self.data_dir.glob(pattern))

        if not shard_paths:
            raise FileNotFoundError(
                f"No shards found matching '{pattern}' in {self.data_dir}"
            )

        for path in shard_paths:
            header = self._read_header(path)
            if self._header is None:
                self._header = header
            self._shards.append(
                ShardInfo(
                    path=path,
                    token_count=header.token_count,
                    split=self.split,
                )
            )

    def _read_header(self, path: Path) -> ShardHeader:
        """Read and validate shard header."""
        with open(path, "rb") as f:
            header_bytes = f.read(HEADER_SIZE * 4)
            header = np.frombuffer(header_bytes, dtype=np.int32)

        parsed = ShardHeader.from_array(header)

        if parsed.magic != MAGIC_NUMBER:
            raise ValueError(
                f"Invalid magic number in {path}: {parsed.magic} != {MAGIC_NUMBER}"
            )

        return parsed

    def _build_index(self) -> None:
        """Build cumulative sequence counts for global-to-local index mapping.

        Figure out how many sequences each shard contributes and store the
        cumulative counts as a monotonic increase list. This list will be used
        for fast global-to-local index mapping by using binary search.

        Example:
            first shard contributes 100 sequences
            second shard contributes 150 sequences
            third shard contributes 150 sequences

            then:
                cumulative_seqs = [0, 100, 250, 400]

            if:
                idx = 120
                shard_idx = 1
                local_idx = 20

        Also, compute the total number of sequences across all shards. This is
        the number of samples the dataset will provide.
        """
        cumulative = 0
        self._cumulative_seqs = [0]
        for shard in self._shards:
            num_seqs = shard.token_count // self.max_seq_length
            cumulative += num_seqs
            self._cumulative_seqs.append(cumulative)
        self._total_seqs = cumulative

    def _get_mmap(self, shard_idx: int) -> np.ndarray:
        """Get or create memory-mapped array for a shard."""
        if shard_idx not in self._mmap_cache:
            shard = self._shards[shard_idx]
            self._mmap_cache[shard_idx] = np.memmap(
                shard.path,
                dtype=np.uint16,
                mode="r",
                offset=HEADER_SIZE * 4,
                shape=(shard.token_count,),
            )
        return self._mmap_cache[shard_idx]

    def __len__(self) -> int:
        return self._total_seqs

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx < 0 or idx >= self._total_seqs:
            raise IndexError(f"Index {idx} out of range [0, {self._total_seqs})")

        # Binary search for which shard contains this index
        shard_idx = bisect.bisect_right(self._cumulative_seqs, idx) - 1
        local_idx = idx - self._cumulative_seqs[shard_idx]

        tokens = self._get_mmap(shard_idx)

        start = local_idx * self.max_seq_length
        end = start + self.max_seq_length
        seq_tokens = tokens[start:end].copy()

        return {
            "input_ids": torch.from_numpy(seq_tokens.astype(np.int64)),
        }

    @property
    def header(self) -> ShardHeader | None:
        """Get the shard header (contains tokenizer info)."""
        return self._header

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all shards."""
        return sum(shard.token_count for shard in self._shards)

    @property
    def num_shards(self) -> int:
        """Number of shards in the dataset."""
        return len(self._shards)

    def __repr__(self) -> str:
        return (
            f"FlatTokenDataset("
            f"data_dir={self.data_dir}, "
            f"split={self.split}, "
            f"max_seq_length={self.max_seq_length}, "
            f"num_shards={self.num_shards}, "
            f"total_seqs={self._total_seqs}, "
            f"total_tokens={self.total_tokens:,})"
        )
