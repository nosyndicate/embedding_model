"""Resumable sampler for deterministic, checkpoint-friendly data loading."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from embedding_trainer.data.base import DatasetProtocol


class ResumableSampler(Sampler[int]):
    """A sampler that produces deterministic permutations and can resume from a checkpoint.

    Each epoch's permutation is determined by ``seed + epoch``, so the same
    sequence is reproduced across runs.  The sampler tracks its position
    (epoch and index within that epoch) and exposes ``state_dict`` /
    ``load_state_dict`` so that training can resume from the exact same point.
    """

    def __init__(
        self,
        data_source: DatasetProtocol | Dataset[Any],
        seed: int = 0,
    ) -> None:
        self._num_samples = len(data_source)  # type: ignore[arg-type]
        self._seed = seed
        self._epoch = 0
        self._start_index = 0
        self._indices: list[int] = []
        self._generate_permutation()

    def _generate_permutation(self) -> None:
        """Generate a new random permutation for the current epoch."""
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        self._indices = torch.randperm(self._num_samples, generator=g).tolist()

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        # Yield remaining indices from the current epoch.
        # No state mutation here — advance() and start_new_epoch() handle that.
        for i in range(self._start_index, len(self._indices)):
            yield self._indices[i]

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self._epoch,
            "index": self._start_index,
            "seed": self._seed,
            "num_samples": self._num_samples,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        state_seed = state["seed"]
        state_num_samples = state["num_samples"]

        if state_seed != self._seed:
            raise ValueError(
                f"ResumableSampler.load_state_dict: seed mismatch "
                f"(checkpoint={state_seed}, current={self._seed})."
            )

        if state_num_samples != self._num_samples:
            raise ValueError(
                f"ResumableSampler.load_state_dict: num_samples mismatch "
                f"(checkpoint={state_num_samples}, current={self._num_samples})."
            )

        self._epoch = state["epoch"]
        self._start_index = state["index"]
        self._generate_permutation()

    def start_new_epoch(self) -> None:
        """Advance to the next epoch and regenerate the permutation."""
        self._epoch += 1
        self._start_index = 0
        self._generate_permutation()

    def advance(self, n: int) -> None:
        """Advance the position by *n* samples (call once per batch)."""
        self._start_index += n
