from __future__ import annotations

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
    ) -> None:
        self.model = model
        self.task = task
        self.loader = loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.callbacks = callbacks or []
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
                self.optimizer.step()

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = SimpleTrainer(
        model=model,
        task=task,
        loader=loader,
        optimizer=optimizer,
        device=device,
        max_steps=5000,
        callbacks=[PrintLossCallback(log_every=100)],
    )
    output = trainer.train()
    print(
        f"Done. global_step={output.global_step}, "
        f"avg_train_loss={output.train_loss:.4f}"
    )


if __name__ == "__main__":
    main()
