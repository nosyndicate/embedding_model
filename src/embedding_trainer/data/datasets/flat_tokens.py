"""RoBERTa-style flat token dataset for loading pre-tokenized binary shards."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

    # Sequence sampling strategy
    # - "fixed": Always start windows at offset 0 (default, deterministic)
    # - "epoch_offset": Rotate starting offset each epoch for data variety
    sampling_strategy: Literal["fixed", "epoch_offset"] = "fixed"
    num_offset_phases: int = 4  # Number of distinct offsets to cycle through

    def __post_init__(self) -> None:
        if self.max_seq_length < 1:
            raise ValueError(f"max_seq_length must be >= 1, got {self.max_seq_length}")
        if self.sampling_strategy not in ("fixed", "epoch_offset"):
            raise ValueError(
                f"Invalid sampling_strategy: {self.sampling_strategy!r}. "
                "Must be 'fixed' or 'epoch_offset'."
            )
        if self.num_offset_phases < 1:
            raise ValueError(
                f"num_offset_phases must be >= 1, got {self.num_offset_phases}"
            )


@DATASET_REGISTRY.register("flat_tokens")
class FlatTokenDataset(Dataset):
    """
    Map-style dataset that reads pre-tokenized binary shards as a flat token stream
    and slices them into fixed-length, non-overlapping windows.

    Each shard is a binary file with a header (magic number, token count, etc.)
    followed by a flat array of uint16 token IDs. Documents within a shard are
    separated by </s> tokens but are otherwise concatenated end-to-end.

    The dataset treats all shards as one logical token stream and divides it into
    ``total_tokens // max_seq_length`` sequences. Sequences may span both document
    and shard boundaries (RoBERTa FULL-SENTENCES approach). Only the remainder
    tokens at the very end of the global stream are dropped.

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
        self._cumulative_tokens: list[int] = []
        self._total_seqs: int = 0
        self._epoch: int = 0
        self._offset: int = 0

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
            else:
                # Validate format fields match (token_count may differ across shards)
                for field in (
                    "magic",
                    "version",
                    "vocab_size",
                    "pad_id",
                    "cls_id",
                    "sep_id",
                    "mask_id",
                ):
                    if getattr(header, field) != getattr(self._header, field):
                        raise ValueError(
                            f"Shard {path} has inconsistent {field}: "
                            f"{getattr(header, field)} != {getattr(self._header, field)}"
                        )
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
        """Build cumulative token counts for global-to-local index mapping.

        Stores cumulative token counts across shards so that a global token
        position can be mapped to a (shard, local_offset) pair via binary
        search.  Sequences are indexed over the global token stream, so a
        sequence may span a shard boundary.

        When ``sampling_strategy="epoch_offset"``, a per-epoch token offset is
        applied so that window boundaries shift each epoch:

            offset = (phase * max_seq_length) // num_offset_phases

        where ``phase = epoch % num_offset_phases``. The sequence count is
        recomputed on every :meth:`set_epoch` call.
        """
        if self.config.sampling_strategy == "epoch_offset":
            phase = self._epoch % self.config.num_offset_phases
            self._offset = (
                phase * self.max_seq_length
            ) // self.config.num_offset_phases
        else:
            self._offset = 0

        cumulative = 0
        self._cumulative_tokens = [0]
        for shard in self._shards:
            cumulative += shard.token_count
            self._cumulative_tokens.append(cumulative)

        total_tokens = cumulative - self._offset
        self._total_seqs = (
            total_tokens // self.max_seq_length if total_tokens > 0 else 0
        )

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

        global_start = self._offset + idx * self.max_seq_length

        # Binary search cumulative_tokens to find starting shard
        shard_idx = bisect.bisect_right(self._cumulative_tokens, global_start) - 1
        local_start = global_start - self._cumulative_tokens[shard_idx]

        tokens_in_curr_shard = self._get_mmap(shard_idx)
        remaining_in_shard = len(tokens_in_curr_shard) - local_start

        if remaining_in_shard >= self.max_seq_length:
            # Common case: entire sequence within one shard
            s, e = local_start, local_start + self.max_seq_length
            seq_tokens = tokens_in_curr_shard[s:e].copy()
        else:
            # Rare case: sequence spans shard boundary
            parts = [tokens_in_curr_shard[local_start:].copy()]
            needed = self.max_seq_length - remaining_in_shard
            next_shard = shard_idx + 1
            while needed > 0:
                tokens_in_next_shard = self._get_mmap(next_shard)
                take = min(needed, len(tokens_in_next_shard))
                parts.append(tokens_in_next_shard[:take].copy())
                needed -= take
                next_shard += 1
            seq_tokens = np.concatenate(parts)

        return {
            # Convert to int32 tensor for PyTorch (torch.uint16 has limited support)
            "input_ids": torch.from_numpy(seq_tokens.astype(np.int32)),
        }

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch, rebuilding the sequence index if needed.

        When ``sampling_strategy="epoch_offset"``, each epoch uses a different
        token offset so window boundaries shift and the model sees varied
        sequence groupings across epochs.
        """
        self._epoch = epoch
        self._build_index()

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
