from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embedding_trainer.core import BaseCallback, BaseTask, EvalOutput, TrainOutput
from embedding_trainer.core.base_trainer import BaseTrainer
from embedding_trainer.data.sampler import ResumableSampler


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
        sampler: ResumableSampler | None = None,
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
        self.sampler = sampler
        self.global_step = 0
        self.cumulative_loss_sum = 0.0
        self.cumulative_loss_steps = 0

    def train(self) -> TrainOutput:
        self.model.train()
        for callback in self.callbacks:
            callback.on_train_begin(self)

        while self.global_step < self.max_steps:
            self._sync_data_epoch()
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
                self.cumulative_loss_sum += loss_value
                self.cumulative_loss_steps += 1
                for callback in self.callbacks:
                    callback.on_step_end(self, self.global_step, loss_value)

                self.global_step += 1
                if self.sampler is not None:
                    self.sampler.advance(batch["input_ids"].shape[0])

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
            else:
                # Epoch completed naturally (no break) — advance sampler epoch.
                if self.sampler is not None:
                    self.sampler.start_new_epoch()
                continue

            if len(self.loader) == 0:
                break

        # Save final checkpoint
        if self.checkpoint_dir is not None:
            self.save_checkpoint(self.checkpoint_dir / f"step_{self.global_step}.pt")

        for callback in self.callbacks:
            callback.on_train_end(self)

        avg_loss = self.cumulative_loss_sum / max(self.cumulative_loss_steps, 1)
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
            "cumulative_loss_sum": self.cumulative_loss_sum,
            "cumulative_loss_steps": self.cumulative_loss_steps,
            **self._capture_rng_state(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.sampler is not None:
            checkpoint["sampler_state_dict"] = self.sampler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.cumulative_loss_sum = float(checkpoint.get("cumulative_loss_sum", 0.0))
        if "cumulative_loss_steps" in checkpoint:
            self.cumulative_loss_steps = int(checkpoint["cumulative_loss_steps"])
        elif "cumulative_loss_sum" in checkpoint:
            self.cumulative_loss_steps = self.global_step
        else:
            self.cumulative_loss_steps = 0
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.sampler is not None and "sampler_state_dict" in checkpoint:
            self.sampler.load_state_dict(checkpoint["sampler_state_dict"])
        self._restore_rng_state(checkpoint)
        self._sync_data_epoch()

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

    def _capture_rng_state(self) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        state: dict[str, Any] = {
            "python_rng_state": random.getstate(),
            "numpy_rng_state": {
                "bit_generator": numpy_state[0],
                "state": numpy_state[1].tolist(),
                "pos": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, checkpoint: dict[str, Any]) -> None:
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])

        numpy_state = checkpoint.get("numpy_rng_state")
        if numpy_state is not None:
            if isinstance(numpy_state, dict):
                np_state = (
                    numpy_state["bit_generator"],
                    np.asarray(numpy_state["state"], dtype=np.uint32),
                    int(numpy_state["pos"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            else:
                # Backward-compatible fallback for tuple/list payloads.
                np_state = (
                    numpy_state[0],
                    np.asarray(numpy_state[1], dtype=np.uint32),
                    int(numpy_state[2]),
                    int(numpy_state[3]),
                    float(numpy_state[4]),
                )
            np.random.set_state(np_state)

        if "torch_rng_state" in checkpoint:
            torch_rng_state = checkpoint["torch_rng_state"]
            if isinstance(torch_rng_state, torch.Tensor):
                torch_rng_state = torch_rng_state.detach().to(
                    device="cpu", dtype=torch.uint8
                )
            torch.set_rng_state(torch_rng_state)

        if torch.cuda.is_available() and "torch_cuda_rng_state_all" in checkpoint:
            cuda_states = checkpoint["torch_cuda_rng_state_all"]
            normalized_states = []
            for state in cuda_states:
                if isinstance(state, torch.Tensor):
                    normalized_states.append(
                        state.detach().to(device="cpu", dtype=torch.uint8)
                    )
                else:
                    normalized_states.append(state)
            torch.cuda.set_rng_state_all(normalized_states)

    def _sync_data_epoch(self) -> None:
        if self.sampler is None:
            return
        epoch = self.sampler.epoch
        dataset = getattr(self.loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        collate_fn = getattr(self.loader, "collate_fn", None)
        if collate_fn is not None and hasattr(collate_fn, "set_epoch"):
            collate_fn.set_epoch(epoch)
