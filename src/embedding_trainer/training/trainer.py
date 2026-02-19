from __future__ import annotations

import math
import re
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
        eval_loader: torch.utils.data.DataLoader | None = None,
        eval_every: int | None = None,
        checkpoint_dir: str | Path | None = None,
        save_every: int | None = None,
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
        self.eval_loader = eval_loader
        self.eval_every = eval_every
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.save_every = save_every
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

                if (
                    self.eval_every is not None
                    and self.eval_loader is not None
                    and self.global_step % self.eval_every == 0
                ):
                    self.evaluate()

                if (
                    self.save_every is not None
                    and self.checkpoint_dir is not None
                    and self.global_step % self.save_every == 0
                ):
                    self.save_checkpoint(
                        self.checkpoint_dir / f"step_{self.global_step}.pt"
                    )

            if len(self.loader) == 0:
                break

        # Save final checkpoint
        if self.checkpoint_dir is not None:
            self.save_checkpoint(self.checkpoint_dir / f"step_{self.global_step}.pt")

        for callback in self.callbacks:
            callback.on_train_end(self)

        avg_loss = loss_sum / max(self.global_step, 1)
        return TrainOutput(global_step=self.global_step, train_loss=avg_loss)

    def evaluate(self) -> EvalOutput:
        if self.eval_loader is None:
            return EvalOutput(eval_loss=float("nan"))

        for callback in self.callbacks:
            callback.on_evaluate_begin(self)

        self.model.eval()
        loss_sum = 0.0
        total_masked = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                task_output = self.task.compute_loss(
                    self.model, batch, device=self.device
                )
                num_masked = int(task_output.metrics.get("num_masked", 1))
                loss_sum += float(task_output.loss.detach().cpu()) * num_masked
                total_masked += num_masked

        self.model.train()

        avg_loss = loss_sum / max(total_masked, 1)
        perplexity = math.exp(min(avg_loss, 100))
        metrics = {"perplexity": perplexity}

        for callback in self.callbacks:
            callback.on_evaluate_end(self, {"eval_loss": avg_loss, **metrics})

        return EvalOutput(eval_loss=avg_loss, metrics=metrics)

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    @staticmethod
    def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            return None
        pattern = re.compile(r"^step_(\d+)\.pt$")
        best: tuple[int, Path] | None = None
        for p in checkpoint_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                step = int(m.group(1))
                if best is None or step > best[0]:
                    best = (step, p)
        return best[1] if best is not None else None
