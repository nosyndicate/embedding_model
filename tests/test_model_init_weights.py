"""Tests for weight initialization of EmbeddingModel."""

from __future__ import annotations

import torch
from torch import nn

from embedding_trainer.models.model import EmbeddingModel


def _make_model() -> EmbeddingModel:
    return EmbeddingModel(vocab_size=100, hidden_size=64, num_layers=2, num_heads=4)


class TestInitWeights:
    def test_linear_weight_std(self) -> None:
        model = _make_model()
        for module in model.modules():
            if isinstance(module, nn.Linear):
                std = module.weight.std().item()
                assert abs(std - 0.02) < 0.01, (
                    f"Linear weight std {std:.4f} not close to 0.02"
                )

    def test_linear_bias_zero(self) -> None:
        model = _make_model()
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert torch.all(module.bias == 0), "Linear bias should be zero"

    def test_embedding_weight_std(self) -> None:
        model = _make_model()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                std = module.weight.std().item()
                assert abs(std - 0.02) < 0.01, (
                    f"Embedding weight std {std:.4f} not close to 0.02"
                )

    def test_layernorm_weight_ones(self) -> None:
        model = _make_model()
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                assert torch.all(module.weight == 1), "LayerNorm weight should be 1"

    def test_layernorm_bias_zero(self) -> None:
        model = _make_model()
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                assert torch.all(module.bias == 0), "LayerNorm bias should be 0"
