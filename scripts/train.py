from __future__ import annotations

import logging
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
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

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()

    set_seed(cfg.training.seed)

    dataset_class = DATASET_REGISTRY.get(cfg.data.name)
    data_dir = cfg.data.data_dir
    if not Path(data_dir).is_absolute():
        data_dir = str(Path(orig_cwd) / data_dir)

    dataset = dataset_class(
        FlatTokenConfig(
            data_dir=data_dir,
            max_seq_length=cfg.data.max_seq_length,
            split=cfg.data.split,
        )
    )
    header = dataset.header
    if header is None:
        raise ValueError("Dataset header is missing.")
    logger.info(f"Vocab size: {header.vocab_size}, Pad ID: {header.pad_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer.tokenizer_name_or_path,
        use_fast=cfg.tokenizer.use_fast,
    )

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

    collator_class = COLLATOR_REGISTRY.get(cfg.collator.name)
    collator = collator_class(MLMCollatorConfig.from_tokenizer(tokenizer))

    task = TASK_REGISTRY.get(cfg.task.name)()
    model = MODEL_REGISTRY.get(cfg.model.name)(
        vocab_size=header.vocab_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    callbacks = []
    for _, cb_cfg in cfg.callbacks.items():
        cb_cls = CALLBACK_REGISTRY.get(cb_cfg.name)
        cb_kwargs = {k: v for k, v in cb_cfg.items() if k != "name"}
        callbacks.append(cb_cls(**cb_kwargs))

    loader = create_dataloader(
        dataset=dataset,
        collator=collator,
        config=DataLoaderConfig(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            prefetch_factor=cfg.training.prefetch_factor,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        ),
    )

    # Validation dataset and loader
    val_dataset = dataset_class(
        FlatTokenConfig(
            data_dir=data_dir,
            max_seq_length=cfg.data.max_seq_length,
            split=cfg.data.split,
        )
    )
    val_loader = create_dataloader(
        dataset=val_dataset,
        collator=collator,
        config=DataLoaderConfig(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            prefetch_factor=cfg.training.prefetch_factor,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            shuffle=False,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.get_param_groups(weight_decay=cfg.training.weight_decay),
        lr=cfg.training.lr,
    )

    max_steps = cfg.training.max_steps
    warmup_steps = cfg.training.warmup_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(orig_cwd) / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    compiled_model = torch.compile(model) if cfg.training.compile else model

    trainer = SimpleTrainer(
        model=compiled_model,
        task=task,
        loader=loader,
        optimizer=optimizer,
        device=device,
        max_steps=max_steps,
        callbacks=callbacks,
        scheduler=scheduler,
        max_grad_norm=cfg.training.max_grad_norm,
        eval_loader=val_loader,
        eval_every=cfg.training.eval_every,
        checkpoint_dir=checkpoint_dir,
        save_every=cfg.training.save_every,
    )

    # Resume from latest checkpoint if available
    latest = SimpleTrainer.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        logger.info(f"Resuming from checkpoint: {latest}")
        trainer.load_checkpoint(latest)

    output = trainer.train()
    logger.info(
        f"Done. global_step={output.global_step}, "
        f"avg_train_loss={output.train_loss:.4f}"
    )


if __name__ == "__main__":
    main()
