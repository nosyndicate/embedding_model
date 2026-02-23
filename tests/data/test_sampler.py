"""Tests for ResumableSampler."""

from __future__ import annotations

from embedding_trainer.data.sampler import ResumableSampler


class _FakeDataset:
    """Minimal dataset stub for sampler tests."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


class TestResumableSampler:
    def test_deterministic_permutation(self) -> None:
        """Same seed and dataset length produce identical index order."""
        ds = _FakeDataset(50)
        s1 = ResumableSampler(ds, seed=42)
        s2 = ResumableSampler(ds, seed=42)
        assert list(s1) == list(s2)

    def test_different_seed_gives_different_order(self) -> None:
        ds = _FakeDataset(50)
        s1 = ResumableSampler(ds, seed=0)
        s2 = ResumableSampler(ds, seed=1)
        assert list(s1) != list(s2)

    def test_len(self) -> None:
        ds = _FakeDataset(100)
        s = ResumableSampler(ds, seed=0)
        assert len(s) == 100

    def test_all_indices_yielded(self) -> None:
        n = 30
        ds = _FakeDataset(n)
        s = ResumableSampler(ds, seed=7)
        indices = list(s)
        assert sorted(indices) == list(range(n))

    def test_state_dict_roundtrip(self) -> None:
        """Saving and restoring state produces identical remaining indices."""
        ds = _FakeDataset(20)
        s = ResumableSampler(ds, seed=99)

        # Consume 5 items
        it = iter(s)
        for _ in range(5):
            next(it)
        s.advance(5)

        state = s.state_dict()

        # Create a new sampler and restore state
        s2 = ResumableSampler(ds, seed=99)
        s2.load_state_dict(state)

        # The remaining indices should be the same as continuing the original
        remaining_original = list(s)
        remaining_restored = list(s2)
        # Both should start from index 5 in epoch 0
        assert remaining_original == remaining_restored

    def test_resume_produces_identical_indices(self) -> None:
        """Full sequence from a resumed sampler matches straight-through."""
        n = 15
        ds = _FakeDataset(n)
        batch_size = 3
        stop_after = 3  # stop after 3 batches (9 samples)

        # Straight-through: collect all indices
        s_full = ResumableSampler(ds, seed=42)
        all_indices = list(s_full)

        # Interrupted run: collect first 9 indices
        s_partial = ResumableSampler(ds, seed=42)
        it = iter(s_partial)
        first_part = [next(it) for _ in range(stop_after * batch_size)]
        s_partial.advance(stop_after * batch_size)
        state = s_partial.state_dict()

        # Resume
        s_resumed = ResumableSampler(ds, seed=42)
        s_resumed.load_state_dict(state)
        second_part = list(s_resumed)

        assert first_part + second_part == all_indices

    def test_iter_does_not_mutate_state(self) -> None:
        """Exhausting __iter__ must not change epoch or start_index."""
        ds = _FakeDataset(10)
        s = ResumableSampler(ds, seed=0)

        state_before = s.state_dict().copy()
        _ = list(s)
        state_after = s.state_dict()

        assert state_before == state_after

    def test_epoch_advances_after_start_new_epoch(self) -> None:
        """start_new_epoch() increments epoch and resets index."""
        ds = _FakeDataset(10)
        s = ResumableSampler(ds, seed=0)

        assert s.state_dict()["epoch"] == 0

        # Exhaust epoch 0 and explicitly advance
        _ = list(s)
        s.start_new_epoch()

        assert s.state_dict()["epoch"] == 1
        assert s.state_dict()["index"] == 0

    def test_cross_epoch_permutations_differ(self) -> None:
        """Different epochs produce different permutations."""
        ds = _FakeDataset(50)

        s = ResumableSampler(ds, seed=0)
        epoch0 = list(s)  # exhausts epoch 0

        s.start_new_epoch()
        epoch1 = list(s)

        assert epoch0 != epoch1
        assert sorted(epoch0) == sorted(epoch1)
