"""Models implementations."""

from embedding_trainer.models.model import BaseEmbeddingModel
from embedding_trainer.models.registry import MODEL_REGISTRY

__all__ = [
    "BaseEmbeddingModel",
    "MODEL_REGISTRY",
]
