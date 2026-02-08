"""Pre-tokenized dataset for loading binary shards."""

from __future__ import annotations

import mmap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

from embedding_trainer.data.base import (
    BERT_VERSION,
    HEADER_SIZE,
    MAGIC_NUMBER,
    ShardHeader,
    ShardInfo,
)
from embedding_trainer.data.registry import DATASET_REGISTRY

# Re-export for backward compatibility
__all__ = [
    "ShardHeader",
    "HEADER_SIZE",
    "MAGIC_NUMBER",
    "BERT_VERSION",
    "PreTokenizedConfig",
    "PreTokenizedDataset",
]


@dataclass
class PreTokenizedConfig:
    """Configuration for pre-tokenized dataset."""

    data_dir: str
    max_seq_length: int = 512
    shuffle: bool = True
    stride: int | None = None  # Defaults to max_seq_length (no overlap)
    split: str = "train"
    seed: int = 42

    # Document boundary handling
    packing_mode: str = "flat"  # "flat" | "doc_aware"

    # Sequence sampling strategy (for flat mode)
    # - "fixed": Always start at offset 0 (current behavior)
    # - "epoch_offset": Rotate starting offset each epoch (4x data variety)
    # - "random": Sample random offsets (same count, random positions)
    # - "shuffle_indices": Shuffle non-overlapping sequence indices
    sampling_strategy: str = (
        "fixed"  # "fixed" | "epoch_offset" | "random" | "shuffle_indices"
    )
    num_offset_phases: int = 4  # for epoch_offset

    def __post_init__(self) -> None:
        if self.stride is None:
            self.stride = self.max_seq_length

        if self.packing_mode not in ("flat", "doc_aware"):
            raise ValueError(f"Invalid packing_mode: {self.packing_mode}")

        if self.sampling_strategy not in (
            "fixed",
            "epoch_offset",
            "random",
            "shuffle_indices",
        ):
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")


@DATASET_REGISTRY.register("pretokenized")
class PreTokenizedDataset(IterableDataset):
    """
    Dataset for loading pre-tokenized binary shards.

    Shards are memory-mapped for efficient reading. The dataset yields
    sequences of max_seq_length tokens, chunked from the continuous token
    stream with optional stride for overlapping sequences.

    Supports distributed training by assigning shards to workers based on rank.
    """

    def __init__(self, config: PreTokenizedConfig) -> None:
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.max_seq_length = config.max_seq_length
        self.stride = config.stride
        self.shuffle = config.shuffle
        self.split = config.split
        self.seed = config.seed

        self._epoch = 0
        self._shards: list[ShardInfo] = []
        self._header: ShardHeader | None = None

        self._discover_shards()

    def _discover_shards(self) -> None:
        """Discover and validate shard files in data directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Find all .bin files matching the split
        pattern = f"fineweb_{self.split}_*.bin"
        shard_paths = sorted(self.data_dir.glob(pattern))

        if not shard_paths:
            raise FileNotFoundError(
                f"No shards found matching '{pattern}' in {self.data_dir}"
            )

        # Validate and collect shard info
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
            header_bytes = f.read(HEADER_SIZE * 4)  # 256 int32s
            header = np.frombuffer(header_bytes, dtype=np.int32)

        parsed = ShardHeader.from_array(header)

        if parsed.magic != MAGIC_NUMBER:
            raise ValueError(
                f"Invalid magic number in {path}: {parsed.magic} != {MAGIC_NUMBER}"
            )

        return parsed

    def _mmap_shard(self, path: Path) -> np.ndarray:
        """Memory-map a shard file and return tokens as numpy array."""
        with open(path, "rb") as f:
            # Skip header
            f.seek(HEADER_SIZE * 4)
            # Memory-map the rest
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # Create numpy view of tokens (after header)
            tokens = np.frombuffer(mm, dtype=np.uint16, offset=HEADER_SIZE * 4)
            return tokens

    def _get_shards_for_worker(self) -> list[ShardInfo]:
        """Get shards assigned to current worker in distributed setting."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading
            return self._shards

        # Multi-worker: assign shards round-robin
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        return [
            shard
            for i, shard in enumerate(self._shards)
            if i % num_workers == worker_id
        ]

    def _get_shuffled_shards(self, shards: list[ShardInfo]) -> list[ShardInfo]:
        """Shuffle shards deterministically based on epoch and seed."""
        if not self.shuffle:
            return shards

        rng = np.random.default_rng(self.seed + self._epoch)
        indices = rng.permutation(len(shards))
        return [shards[i] for i in indices]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over sequences from all shards."""
        shards = self._get_shards_for_worker()
        shards = self._get_shuffled_shards(shards)

        for shard in shards:
            tokens = self._mmap_shard(shard.path)
            header = self._read_header(shard.path)

            if self.config.packing_mode == "flat":
                yield from self._iter_flat(tokens)
            elif self.config.packing_mode == "doc_aware":
                yield from self._iter_doc_aware(tokens, header)

    def _iter_flat(self, tokens: np.ndarray) -> Iterator[dict[str, Any]]:
        """Iterate over flat sequences with configurable sampling strategy."""
        seq_len = self.max_seq_length
        stride = self.stride
        strategy = self.config.sampling_strategy

        if strategy == "fixed":
            # Original behavior: always start at 0
            for i in range(0, len(tokens) - seq_len + 1, stride):
                yield self._make_sample(tokens, i, seq_len)

        elif strategy == "epoch_offset":
            # Rotate starting offset each epoch for data variety
            phase = self._epoch % self.config.num_offset_phases
            offset = (phase * stride) // self.config.num_offset_phases
            for i in range(offset, len(tokens) - seq_len + 1, stride):
                yield self._make_sample(tokens, i, seq_len)

        elif strategy == "random":
            # Sample random offsets (same count as fixed, but random positions)
            rng = np.random.default_rng(self.seed + self._epoch)
            max_start = len(tokens) - seq_len
            if max_start <= 0:
                return

            num_samples = max_start // stride + 1
            for _ in range(num_samples):
                start = rng.integers(0, max_start + 1)
                yield self._make_sample(tokens, start, seq_len)

        elif strategy == "shuffle_indices":
            # Shuffle non-overlapping sequence indices
            num_seqs = (len(tokens) - seq_len) // stride + 1
            if num_seqs <= 0:
                return

            rng = np.random.default_rng(self.seed + self._epoch)
            indices = rng.permutation(num_seqs)

            for idx in indices:
                start = idx * stride
                yield self._make_sample(tokens, start, seq_len)

    def _iter_doc_aware(
        self, tokens: np.ndarray, header: ShardHeader
    ) -> Iterator[dict[str, Any]]:
        """Iterate respecting document boundaries (CLS/SEP tokens)."""
        cls_id = header.cls_id
        sep_id = header.sep_id
        pad_id = header.pad_id
        seq_len = self.max_seq_length

        # Find all [CLS] positions (document starts)
        cls_positions = np.where(tokens == cls_id)[0].tolist()
        if not cls_positions:
            # No documents found, fall back to flat iteration
            yield from self._iter_flat(tokens)
            return

        cls_positions.append(len(tokens))  # End sentinel

        buffer: list[int] = []

        for i in range(len(cls_positions) - 1):
            doc_start = cls_positions[i]
            doc_end = cls_positions[i + 1]
            doc_tokens = tokens[doc_start:doc_end].tolist()

            # If single document exceeds max_seq_length, split it
            if len(doc_tokens) > seq_len:
                # Yield current buffer first if any
                if buffer:
                    yield self._make_padded_sample(buffer, seq_len, pad_id)
                    buffer = []

                # Split long document into chunks
                for chunk_start in range(0, len(doc_tokens), seq_len):
                    chunk = doc_tokens[chunk_start : chunk_start + seq_len]
                    if len(chunk) == seq_len:
                        yield self._make_padded_sample(chunk, seq_len, pad_id)
                    else:
                        # Last partial chunk goes into buffer
                        buffer = chunk
                continue

            # Check if adding document would exceed seq_len
            if len(buffer) + len(doc_tokens) > seq_len:
                # Yield current buffer and start new one
                yield self._make_padded_sample(buffer, seq_len, pad_id)
                buffer = doc_tokens
            else:
                # Pack document into buffer
                buffer.extend(doc_tokens)

        # Yield remaining buffer if not empty
        if buffer:
            yield self._make_padded_sample(buffer, seq_len, pad_id)

    def _make_sample(
        self, tokens: np.ndarray, start: int, length: int
    ) -> dict[str, Any]:
        """Create a sample dict from token slice."""
        seq_tokens = tokens[start : start + length].copy()
        return {
            "input_ids": torch.from_numpy(seq_tokens.astype(np.int64)),
        }

    def _make_padded_sample(
        self, tokens: list[int], seq_len: int, pad_id: int
    ) -> dict[str, Any]:
        """Create a padded sample dict from token list."""
        if len(tokens) < seq_len:
            tokens = tokens + [pad_id] * (seq_len - len(tokens))
        return {
            "input_ids": torch.tensor(tokens[:seq_len], dtype=torch.int64),
        }

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reproducible shuffling."""
        self._epoch = epoch

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
            f"PreTokenizedDataset("
            f"data_dir={self.data_dir}, "
            f"split={self.split}, "
            f"num_shards={self.num_shards}, "
            f"total_tokens={self.total_tokens:,})"
        )
