"""Registry for datasets and collators."""

from __future__ import annotations

from embedding_trainer.utils.registry import Registry

DATASET_REGISTRY = Registry("dataset")
COLLATOR_REGISTRY = Registry("collator")
