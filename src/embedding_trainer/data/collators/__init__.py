"""Collator implementations."""

from embedding_trainer.data.collators.base import BaseCollator
from embedding_trainer.data.collators.electra import (
    ELECTRACollator,
    ELECTRACollatorConfig,
)
from embedding_trainer.data.collators.mlm import MLMCollator, MLMCollatorConfig
from embedding_trainer.data.collators.span_corruption import (
    SpanCorruptionCollator,
    SpanCorruptionCollatorConfig,
)

__all__ = [
    "BaseCollator",
    "MLMCollator",
    "MLMCollatorConfig",
    "ELECTRACollator",
    "ELECTRACollatorConfig",
    "SpanCorruptionCollator",
    "SpanCorruptionCollatorConfig",
]
