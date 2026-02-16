"""Training tasks."""

from embedding_trainer.tasks.mlm import MaskedLanguageModelingTask
from embedding_trainer.tasks.registry import TASK_REGISTRY

__all__ = ["MaskedLanguageModelingTask", "TASK_REGISTRY"]
