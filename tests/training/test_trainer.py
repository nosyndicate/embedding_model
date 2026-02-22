"""Tests for SimpleTrainer checkpointing and loss accounting."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from embedding_trainer.core.types import TaskOutput
from embedding_trainer.training.trainer import SimpleTrainer


class ConstantLossTask:
    def __init__(self, loss_value: float = 2.0) -> None:
        self.loss_value = loss_value

    def compute_loss(
        self, model: torch.nn.Module, batch: object, device: str = "cpu"
    ) -> TaskOutput:
        del batch
        del device
        loss = model.weight * 0 + self.loss_value
        return TaskOutput(loss=loss)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


def _make_trainer(checkpoint_dir: str, max_steps: int) -> SimpleTrainer:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader([0, 1, 2, 3], batch_size=1, shuffle=False)
    return SimpleTrainer(
        model=model,
        task=ConstantLossTask(loss_value=2.0),
        loader=loader,
        optimizer=optimizer,
        device="cpu",
        max_steps=max_steps,
        checkpoint_dir=checkpoint_dir,
        save_every=2,
    )


def test_checkpoint_stores_cumulative_loss_fields(tmp_path) -> None:
    trainer = _make_trainer(str(tmp_path), max_steps=2)
    trainer.train()

    checkpoint = torch.load(
        tmp_path / "step_2.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["cumulative_loss_sum"] == 4.0
    assert checkpoint["cumulative_loss_steps"] == 2


def test_resume_uses_full_run_average_from_cumulative_state(tmp_path) -> None:
    first = _make_trainer(str(tmp_path), max_steps=2)
    first.train()

    resumed = _make_trainer(str(tmp_path), max_steps=4)
    resumed.load_checkpoint(tmp_path / "step_2.pt")
    output = resumed.train()

    assert output.global_step == 4
    assert output.train_loss == 2.0


def test_resume_from_legacy_checkpoint_without_cumulative_fields(tmp_path) -> None:
    first = _make_trainer(str(tmp_path), max_steps=2)
    first.train()

    legacy = torch.load(tmp_path / "step_2.pt", map_location="cpu", weights_only=True)
    legacy.pop("cumulative_loss_sum")
    legacy.pop("cumulative_loss_steps")
    legacy_path = tmp_path / "step_2_legacy.pt"
    torch.save(legacy, legacy_path)

    resumed = _make_trainer(str(tmp_path), max_steps=4)
    resumed.load_checkpoint(legacy_path)
    output = resumed.train()

    assert output.global_step == 4
    assert output.train_loss == 2.0


def test_non_resume_average_is_unchanged(tmp_path) -> None:
    trainer = _make_trainer(str(tmp_path), max_steps=4)
    output = trainer.train()
    assert output.global_step == 4
    assert output.train_loss == 2.0
