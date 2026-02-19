from __future__ import annotations

from pathlib import Path

import torch

from embedding_trainer.core import BaseCallback, BaseTask, EvalOutput, TrainOutput
from embedding_trainer.core.base_trainer import BaseTrainer


class SimpleTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        task: BaseTask,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        max_steps: int,
        callbacks: list[BaseCallback] | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        self.model = model
        self.task = task
        self.loader = loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.callbacks = callbacks or []
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

    def train(self) -> TrainOutput:
        self.model.train()
        for callback in self.callbacks:
            callback.on_train_begin(self)

        loss_sum = 0.0
        while self.global_step < self.max_steps:
            for batch in self.loader:
                if self.global_step >= self.max_steps:
                    break

                for callback in self.callbacks:
                    callback.on_step_begin(self, self.global_step)

                self.optimizer.zero_grad(set_to_none=True)
                task_output = self.task.compute_loss(
                    self.model,
                    batch,
                    device=self.device,
                )
                task_output.loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                loss_value = float(task_output.loss.detach().cpu())
                loss_sum += loss_value
                for callback in self.callbacks:
                    callback.on_step_end(self, self.global_step, loss_value)

                self.global_step += 1

            if len(self.loader) == 0:
                break

        for callback in self.callbacks:
            callback.on_train_end(self)

        avg_loss = loss_sum / max(self.global_step, 1)
        return TrainOutput(global_step=self.global_step, train_loss=avg_loss)

    def evaluate(self) -> EvalOutput:
        return EvalOutput(eval_loss=float("nan"))

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
