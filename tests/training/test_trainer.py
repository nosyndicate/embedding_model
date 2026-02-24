"""Tests for SimpleTrainer checkpoint determinism behavior."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from embedding_trainer.core import BaseCallback, ModelOutput, TaskOutput
from embedding_trainer.core.base_task import BaseTask
from embedding_trainer.data.collators.mlm import MLMCollator, MLMCollatorConfig
from embedding_trainer.data.loader import DataLoaderConfig, create_dataloader
from embedding_trainer.data.sampler import ResumableSampler
from embedding_trainer.tasks.mlm import MaskedLanguageModelingTask
from embedding_trainer.training.trainer import SimpleTrainer


class _EpochDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, size: int = 16) -> None:
        self._size = size
        self.epoch_calls: list[int] = []

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([float(idx)], dtype=torch.float32),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "labels": torch.tensor([0], dtype=torch.long),
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch_calls.append(epoch)


class _EpochCollator:
    def __init__(self) -> None:
        self.epoch_calls: list[int] = []

    def __call__(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
            "attention_mask": torch.stack(
                [sample["attention_mask"] for sample in samples]
            ),
            "labels": torch.stack([sample["labels"] for sample in samples]),
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch_calls.append(epoch)


class _DummyTask(BaseTask):
    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        device: str = "cpu",
    ) -> TaskOutput:
        inputs = batch["input_ids"].to(device)
        predictions = model(inputs)
        loss = predictions.mean()
        return TaskOutput(loss=loss)

    def get_metrics(self) -> dict[str, float]:
        return {}


class _LossRecorder(BaseCallback):
    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_step_end(self, trainer: SimpleTrainer, step: int, loss: float) -> None:
        self.losses.append(loss)


class _SyntheticMLMDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, size: int = 128, seq_len: int = 16) -> None:
        self._size = size
        self._seq_len = seq_len
        self.epoch = 0

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        values = [((idx * 37 + i * 13 + self.epoch * 17) % 50) + 4 for i in range(14)]
        tokens = [1] + values + [2]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "sample_idx": torch.tensor(idx, dtype=torch.int64),
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _TinyMLMModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> ModelOutput:
        del attention_mask
        x = self.embedding(input_ids)
        x = self.dropout(x)
        logits = self.proj(x)
        return ModelOutput(logits=logits)


def _make_trainer(
    tmp_path: Path,
    *,
    dataset: _EpochDataset,
    collator: _EpochCollator,
    sampler: ResumableSampler,
) -> SimpleTrainer:
    loader = create_dataloader(
        dataset=dataset,
        collator=collator,
        config=DataLoaderConfig(batch_size=4, num_workers=0, shuffle=False),
        sampler=sampler,
    )
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return SimpleTrainer(
        model=model,
        task=_DummyTask(),
        loader=loader,
        optimizer=optimizer,
        device="cpu",
        max_steps=1,
        checkpoint_dir=tmp_path,
        save_every=None,
        sampler=sampler,
    )


def _make_mlm_trainer(
    checkpoint_dir: Path,
    *,
    max_steps: int,
    seed: int,
    num_workers: int,
    save_every: int | None = None,
    device: str = "cpu",
) -> tuple[SimpleTrainer, _LossRecorder]:
    dataset = _SyntheticMLMDataset()
    sampler = ResumableSampler(dataset, seed=seed)
    collator = MLMCollator(
        MLMCollatorConfig(
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            mask_token_id=3,
            vocab_size=64,
            mlm_probability=0.30,
            seed=seed,
            epoch=0,
        )
    )
    collator.set_epoch(sampler.epoch)
    loader = create_dataloader(
        dataset=dataset,
        collator=collator,
        config=DataLoaderConfig(
            batch_size=8,
            num_workers=num_workers,
            prefetch_factor=2,
            drop_last=True,
            shuffle=False,
            seed=seed,
        ),
        sampler=sampler,
    )
    model = _TinyMLMModel(vocab_size=64, hidden_size=16)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    recorder = _LossRecorder()
    trainer = SimpleTrainer(
        model=model,
        task=MaskedLanguageModelingTask(),
        loader=loader,
        optimizer=optimizer,
        device=device,
        max_steps=max_steps,
        callbacks=[recorder],
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        save_every=save_every,
        sampler=sampler,
    )
    return trainer, recorder


def _clone_state_dict(state: dict[str, object]) -> dict[str, object]:
    cloned: dict[str, object] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            cloned[key] = _clone_state_dict(value)
        elif isinstance(value, list):
            out: list[object] = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    out.append(item.detach().cpu().clone())
                elif isinstance(item, dict):
                    out.append(_clone_state_dict(item))
                else:
                    out.append(item)
            cloned[key] = out
        else:
            cloned[key] = value
    return cloned


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left.detach().cpu(), right.detach().cpu())
        return
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for li, ri in zip(left, right, strict=True):
            _assert_nested_equal(li, ri)
        return
    assert left == right


class TestSimpleTrainerCheckpoint:
    def test_load_checkpoint_restores_rng_states(self, tmp_path: Path) -> None:
        random.seed(111)
        np.random.seed(222)
        torch.manual_seed(333)

        dataset = _EpochDataset()
        collator = _EpochCollator()
        sampler = ResumableSampler(dataset, seed=7)
        trainer = _make_trainer(
            tmp_path,
            dataset=dataset,
            collator=collator,
            sampler=sampler,
        )

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()

        py_rng = random.Random()
        py_rng.setstate(python_state)
        expected_python = py_rng.random()

        np_rng = np.random.RandomState()
        np_rng.set_state(numpy_state)
        expected_numpy = float(np_rng.rand())

        torch_rng = torch.Generator()
        torch_rng.set_state(torch_state.clone())
        expected_torch = float(torch.rand((), generator=torch_rng))

        ckpt = tmp_path / "rng.pt"
        trainer.save_checkpoint(ckpt)

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        trainer.load_checkpoint(ckpt)
        assert random.random() == expected_python
        assert float(np.random.rand()) == expected_numpy
        assert float(torch.rand(())) == expected_torch

    def test_load_checkpoint_syncs_dataset_and_collator_epoch(
        self, tmp_path: Path
    ) -> None:
        dataset = _EpochDataset()
        collator = _EpochCollator()
        sampler = ResumableSampler(dataset, seed=21)
        sampler.start_new_epoch()
        sampler.start_new_epoch()

        trainer = _make_trainer(
            tmp_path,
            dataset=dataset,
            collator=collator,
            sampler=sampler,
        )
        ckpt = tmp_path / "epoch.pt"
        trainer.save_checkpoint(ckpt)

        resumed_dataset = _EpochDataset()
        resumed_collator = _EpochCollator()
        resumed_sampler = ResumableSampler(resumed_dataset, seed=21)
        resumed_trainer = _make_trainer(
            tmp_path,
            dataset=resumed_dataset,
            collator=resumed_collator,
            sampler=resumed_sampler,
        )

        resumed_trainer.load_checkpoint(ckpt)

        assert resumed_sampler.epoch == 2
        assert resumed_dataset.epoch_calls[-1] == 2
        assert resumed_collator.epoch_calls[-1] == 2

    def test_resume_matches_uninterrupted_with_multiworker(
        self, tmp_path: Path
    ) -> None:
        total_steps = 12
        split_step = 6
        seed = 2026
        num_workers = 2

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        full_dir = tmp_path / "full"
        full_dir.mkdir(parents=True, exist_ok=True)
        full_trainer, full_recorder = _make_mlm_trainer(
            full_dir,
            max_steps=total_steps,
            seed=seed,
            num_workers=num_workers,
            save_every=split_step,
        )
        full_trainer.train()
        ckpt = full_dir / f"step_{split_step}.pt"
        assert ckpt.exists()
        full_model = _clone_state_dict(full_trainer.model.state_dict())
        full_opt = _clone_state_dict(full_trainer.optimizer.state_dict())
        full_sched = _clone_state_dict(full_trainer.scheduler.state_dict())  # type: ignore[union-attr]
        full_sampler = full_trainer.sampler.state_dict()  # type: ignore[union-attr]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        resume_dir = tmp_path / "resume"
        resume_dir.mkdir(parents=True, exist_ok=True)
        stage2_trainer, stage2_recorder = _make_mlm_trainer(
            resume_dir,
            max_steps=total_steps,
            seed=seed,
            num_workers=num_workers,
        )
        stage2_trainer.load_checkpoint(ckpt)
        stage2_trainer.train()

        assert stage2_recorder.losses == full_recorder.losses[split_step:]
        _assert_nested_equal(stage2_trainer.model.state_dict(), full_model)
        _assert_nested_equal(stage2_trainer.optimizer.state_dict(), full_opt)
        _assert_nested_equal(
            stage2_trainer.scheduler.state_dict(),  # type: ignore[union-attr]
            full_sched,
        )
        assert stage2_trainer.sampler.state_dict() == full_sampler  # type: ignore[union-attr]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_resume_matches_uninterrupted_with_multiworker_cuda(
        self, tmp_path: Path
    ) -> None:
        total_steps = 12
        split_step = 6
        seed = 2026
        num_workers = 2
        device = "cuda"

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        full_dir = tmp_path / "full_cuda"
        full_dir.mkdir(parents=True, exist_ok=True)
        full_trainer, full_recorder = _make_mlm_trainer(
            full_dir,
            max_steps=total_steps,
            seed=seed,
            num_workers=num_workers,
            save_every=split_step,
            device=device,
        )
        full_trainer.train()
        ckpt = full_dir / f"step_{split_step}.pt"
        assert ckpt.exists()
        full_model = _clone_state_dict(full_trainer.model.state_dict())
        full_opt = _clone_state_dict(full_trainer.optimizer.state_dict())
        full_sched = _clone_state_dict(full_trainer.scheduler.state_dict())  # type: ignore[union-attr]
        full_sampler = full_trainer.sampler.state_dict()  # type: ignore[union-attr]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        resume_dir = tmp_path / "resume_cuda"
        resume_dir.mkdir(parents=True, exist_ok=True)
        stage2_trainer, stage2_recorder = _make_mlm_trainer(
            resume_dir,
            max_steps=total_steps,
            seed=seed,
            num_workers=num_workers,
            device=device,
        )
        stage2_trainer.load_checkpoint(ckpt)
        stage2_trainer.train()

        assert stage2_recorder.losses == full_recorder.losses[split_step:]
        _assert_nested_equal(stage2_trainer.model.state_dict(), full_model)
        _assert_nested_equal(stage2_trainer.optimizer.state_dict(), full_opt)
        _assert_nested_equal(
            stage2_trainer.scheduler.state_dict(),  # type: ignore[union-attr]
            full_sched,
        )
        assert stage2_trainer.sampler.state_dict() == full_sampler  # type: ignore[union-attr]
