"""Tests for the MLM training task."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from embedding_trainer.core.base_model import BaseEmbeddingModel
from embedding_trainer.core.types import ModelOutput
from embedding_trainer.data.base import LABEL_IGNORE_ID, PreTokenizedBatch
from embedding_trainer.tasks import TASK_REGISTRY
from embedding_trainer.tasks.mlm import MaskedLanguageModelingTask


class DummyMLMModel(BaseEmbeddingModel):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.last_attention_mask: Tensor | None = None

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> ModelOutput:
        self.last_attention_mask = attention_mask
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size, dtype=torch.float32)
        embeddings = torch.zeros(batch_size, seq_len, self.hidden_size)
        return ModelOutput(embeddings=embeddings, logits=logits)

    def get_embeddings(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, self.hidden_size)

    def get_param_groups(self, **kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"params": self.parameters()}]

    @property
    def hidden_size(self) -> int:
        return 8


class NoLogitsModel(BaseEmbeddingModel):
    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> ModelOutput:
        embeddings = self.get_embeddings(input_ids, attention_mask)
        return ModelOutput(embeddings=embeddings, logits=None)

    def get_embeddings(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, self.hidden_size)

    def get_param_groups(self, **kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"params": self.parameters()}]

    @property
    def hidden_size(self) -> int:
        return 8


def _make_batch() -> PreTokenizedBatch:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long),
        "labels": torch.tensor(
            [[LABEL_IGNORE_ID, 2, LABEL_IGNORE_ID, 4], [5, LABEL_IGNORE_ID, 7, 8]],
            dtype=torch.long,
        ),
    }


class TestMaskedLanguageModelingTask:
    def test_instantiable(self) -> None:
        task = MaskedLanguageModelingTask()
        assert isinstance(task, MaskedLanguageModelingTask)

    def test_compute_loss_passes_attention_mask_and_returns_finite_loss(self) -> None:
        model = DummyMLMModel(vocab_size=16)
        task = MaskedLanguageModelingTask()
        batch = _make_batch()

        output = task.compute_loss(model, batch)

        assert model.last_attention_mask is not None
        assert torch.equal(model.last_attention_mask, batch["attention_mask"])
        assert torch.isfinite(output.loss).item()

    def test_zero_masked_tokens_returns_zero_loss(self) -> None:
        model = DummyMLMModel(vocab_size=16)
        task = MaskedLanguageModelingTask()
        batch = _make_batch()
        batch["labels"] = torch.full_like(batch["labels"], LABEL_IGNORE_ID)

        output = task.compute_loss(model, batch)

        assert not torch.isnan(output.loss).item()
        assert output.loss.item() == 0.0

    def test_compute_loss_accepts_int32_labels(self) -> None:
        model = DummyMLMModel(vocab_size=16)
        task = MaskedLanguageModelingTask()
        batch = _make_batch()
        batch["labels"] = batch["labels"].to(torch.int32)

        output = task.compute_loss(model, batch, device="cpu")

        assert torch.isfinite(output.loss).item()

    def test_missing_logits_raises_value_error(self) -> None:
        model = NoLogitsModel()
        task = MaskedLanguageModelingTask()
        batch = _make_batch()

        with pytest.raises(
            ValueError, match="Model output logits must not be None for MLM task"
        ):
            task.compute_loss(model, batch)

    def test_registered_as_mlm(self) -> None:
        assert "mlm" in TASK_REGISTRY
        assert TASK_REGISTRY.get("mlm") is MaskedLanguageModelingTask
