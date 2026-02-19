"""Training callbacks."""

from embedding_trainer.callbacks.registry import CALLBACK_REGISTRY
from embedding_trainer.callbacks.simple_callbacks import PrintLossCallback

__all__ = ["PrintLossCallback", "CALLBACK_REGISTRY"]
