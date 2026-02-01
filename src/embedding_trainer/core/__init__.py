"""Core abstractions for the embedding trainer framework."""

from embedding_trainer.core.base_callback import BaseCallback
from embedding_trainer.core.base_model import BaseEmbeddingModel
from embedding_trainer.core.base_task import BaseTask
from embedding_trainer.core.base_trainer import BaseTrainer
from embedding_trainer.core.precision import (
    BF16Precision,
    FP16Precision,
    FP32Precision,
    PrecisionContext,
    get_precision_context,
)
from embedding_trainer.core.types import (
    CallbackProtocol,
    EvalOutput,
    LoggerProtocol,
    ModelOutput,
    ModelProtocol,
    TaskOutput,
    TaskProtocol,
    TrainerProtocol,
    TrainOutput,
)

__all__ = [
    # Base classes
    "BaseCallback",
    "BaseEmbeddingModel",
    "BaseTask",
    "BaseTrainer",
    # Precision
    "BF16Precision",
    "FP16Precision",
    "FP32Precision",
    "PrecisionContext",
    "get_precision_context",
    # Types and protocols
    "CallbackProtocol",
    "EvalOutput",
    "LoggerProtocol",
    "ModelOutput",
    "ModelProtocol",
    "TaskOutput",
    "TaskProtocol",
    "TrainerProtocol",
    "TrainOutput",
]
