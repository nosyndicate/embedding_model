"""Tests for EmbeddingModel.get_param_groups weight-decay exclusions."""

from __future__ import annotations

import pytest
import torch.nn as nn

from embedding_trainer.models.model import EmbeddingModel


@pytest.fixture()
def model() -> EmbeddingModel:
    return EmbeddingModel(vocab_size=32, hidden_size=16, num_layers=1, num_heads=2)


class TestGetParamGroups:
    def test_no_weight_decay_returns_single_group(self, model: EmbeddingModel) -> None:
        groups = model.get_param_groups()
        assert len(groups) == 1
        assert "weight_decay" not in groups[0]
        assert len(groups[0]["params"]) == len(list(model.parameters()))

    def test_with_weight_decay_returns_two_groups(self, model: EmbeddingModel) -> None:
        groups = model.get_param_groups(weight_decay=0.01)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0

    def test_all_params_accounted_for(self, model: EmbeddingModel) -> None:
        groups = model.get_param_groups(weight_decay=0.1)
        grouped_ids = {id(p) for g in groups for p in g["params"]}
        all_ids = {id(p) for p in model.parameters()}
        assert grouped_ids == all_ids

    def test_no_duplicate_params(self, model: EmbeddingModel) -> None:
        groups = model.get_param_groups(weight_decay=0.1)
        all_params = [id(p) for g in groups for p in g["params"]]
        assert len(all_params) == len(set(all_params))

    def test_bias_params_excluded_from_decay(self, model: EmbeddingModel) -> None:
        groups = model.get_param_groups(weight_decay=0.1)
        no_decay_ids = {id(p) for p in groups[1]["params"]}
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                assert id(param) in no_decay_ids, f"{name} should be in no-decay group"

    def test_layernorm_params_excluded_from_decay(self, model: EmbeddingModel) -> None:
        groups = model.get_param_groups(weight_decay=0.1)
        no_decay_ids = {id(p) for p in groups[1]["params"]}

        # Collect all LayerNorm param names via isinstance check
        ln_param_names: list[str] = []
        for module_name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param_name, _ in module.named_parameters():
                    ln_param_names.append(f"{module_name}.{param_name}")

        assert len(ln_param_names) > 0, "Model should have LayerNorm parameters"

        for name in ln_param_names:
            param = dict(model.named_parameters())[name]
            assert id(param) in no_decay_ids, f"{name} should be in no-decay group"

    def test_embedding_params_excluded_from_decay(self, model: EmbeddingModel) -> None:
        groups = model.get_param_groups(weight_decay=0.1)
        no_decay_ids = {id(p) for p in groups[1]["params"]}

        param_dict = dict(model.named_parameters())
        emb_weight = param_dict["embedding.embedding.weight"]
        assert id(emb_weight) in no_decay_ids

    def test_linear_weights_in_decay_group(self, model: EmbeddingModel) -> None:
        groups = model.get_param_groups(weight_decay=0.1)
        decay_ids = {id(p) for p in groups[0]["params"]}

        param_dict = dict(model.named_parameters())
        for name in ("head.0.weight", "head.3.weight"):
            assert id(param_dict[name]) in decay_ids, f"{name} should be in decay group"
