from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from embedding_trainer.core import BaseCallback, BaseTrainer, EvalOutput, TrainOutput
from embedding_trainer.data.collators import MLMCollatorConfig
from embedding_trainer.data.datasets import FlatTokenConfig
from embedding_trainer.data.loader import DataLoaderConfig, create_dataloader
from embedding_trainer.data.registry import COLLATOR_REGISTRY, DATASET_REGISTRY
from embedding_trainer.models.registry import MODEL_REGISTRY
from embedding_trainer.tasks.registry import TASK_REGISTRY
from embedding_trainer.utils import set_seed


class PrintLossCallback(BaseCallback):
    def __init__(self, log_every: int = 1) -> None:
        self.log_every = max(log_every, 1)

    def on_train_begin(self, trainer: SimpleTrainer) -> None:
        print(f"Starting training on {trainer.device}")

    def on_step_end(self, trainer: SimpleTrainer, step: int, loss: float) -> None:
        if step % self.log_every == 0:
            print(f"step={step} loss={loss:.4f}")

    def on_train_end(self, trainer: SimpleTrainer) -> None:
        print("Training finished.")


class SimpleTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        task: object,
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


def main() -> None:
    set_seed(42)

    dataset_class = DATASET_REGISTRY.get("flat_tokens")
    dataset = dataset_class(
        FlatTokenConfig(
            data_dir="./data/fineweb_edu_10bt",
            max_seq_length=512,
            split="train",
        )
    )
    header = dataset.header
    if header is None:
        raise ValueError("Dataset header is missing.")
    print(f"Vocab size: {header.vocab_size}, Pad ID: {header.pad_id}")

    collator_class = COLLATOR_REGISTRY.get("mlm")
    collator = collator_class(
        MLMCollatorConfig.from_tokenizer(
            AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
        )
    )

    task = TASK_REGISTRY.get("mlm")()
    model = MODEL_REGISTRY.get("base_embedding_model")(
        vocab_size=header.vocab_size, hidden_size=512, num_layers=4, num_heads=4
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loader = create_dataloader(
        dataset=dataset,
        collator=collator,
        config=DataLoaderConfig(
            batch_size=32,
            num_workers=0,
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        ),
    )

    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4)

    max_steps = 5000
    warmup_steps = 500

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = SimpleTrainer(
        model=model,
        task=task,
        loader=loader,
        optimizer=optimizer,
        device=device,
        max_steps=max_steps,
        callbacks=[PrintLossCallback(log_every=100)],
        scheduler=scheduler,
        max_grad_norm=1.0,
    )
    output = trainer.train()
    print(
        f"Done. global_step={output.global_step}, "
        f"avg_train_loss={output.train_loss:.4f}"
    )


if __name__ == "__main__":
    main()
