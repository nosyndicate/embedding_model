from __future__ import annotations

import math

import torch
from transformers import AutoTokenizer

from embedding_trainer.callbacks import CALLBACK_REGISTRY
from embedding_trainer.data import COLLATOR_REGISTRY, DATASET_REGISTRY
from embedding_trainer.data.collators import MLMCollatorConfig
from embedding_trainer.data.datasets import FlatTokenConfig
from embedding_trainer.data.loader import DataLoaderConfig, create_dataloader
from embedding_trainer.models import MODEL_REGISTRY
from embedding_trainer.tasks import TASK_REGISTRY
from embedding_trainer.training.trainer import SimpleTrainer
from embedding_trainer.utils import set_seed


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

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    # Cross-check tokenizer IDs against shard header
    assert tokenizer.vocab_size == header.vocab_size, (
        f"Tokenizer vocab_size ({tokenizer.vocab_size}) != shard header ({header.vocab_size})"
    )
    assert tokenizer.pad_token_id == header.pad_id, (
        f"Tokenizer pad_token_id ({tokenizer.pad_token_id}) != shard header ({header.pad_id})"
    )
    assert tokenizer.sep_token_id == header.sep_id, (
        f"Tokenizer sep_token_id ({tokenizer.sep_token_id}) != shard header ({header.sep_id})"
    )
    assert tokenizer.mask_token_id == header.mask_id, (
        f"Tokenizer mask_token_id ({tokenizer.mask_token_id}) != shard header ({header.mask_id})"
    )

    collator_class = COLLATOR_REGISTRY.get("mlm")
    collator = collator_class(MLMCollatorConfig.from_tokenizer(tokenizer))

    task = TASK_REGISTRY.get("mlm")()
    model = MODEL_REGISTRY.get("base_embedding_model")(
        vocab_size=header.vocab_size, hidden_size=512, num_layers=4, num_heads=4
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    callbacks = [
        CALLBACK_REGISTRY.get("print_loss")(log_every=100),
    ]

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

    optimizer = torch.optim.AdamW(model.get_param_groups(weight_decay=0.01), lr=1e-4)

    max_steps = 5000
    warmup_steps = 500

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = SimpleTrainer(
        model=torch.compile(model),
        task=task,
        loader=loader,
        optimizer=optimizer,
        device=device,
        max_steps=max_steps,
        callbacks=callbacks,
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
