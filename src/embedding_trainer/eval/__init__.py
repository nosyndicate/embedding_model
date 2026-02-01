"""Evaluation module for embedding trainer."""

from embedding_trainer.eval.base import BaseEvalTask, EvalResult
from embedding_trainer.eval.model_loader import load_model
from embedding_trainer.eval.registry import EVAL_TASK_REGISTRY

# Import tasks to register them
from embedding_trainer.eval.tasks import MLMEvalTask

__all__ = [
    "BaseEvalTask",
    "EvalResult",
    "load_model",
    "EVAL_TASK_REGISTRY",
    "MLMEvalTask",
]
