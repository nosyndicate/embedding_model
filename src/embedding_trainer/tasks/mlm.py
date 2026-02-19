import torch
from torch import nn

from embedding_trainer.core.base_task import BaseTask
from embedding_trainer.core.types import TaskOutput
from embedding_trainer.data.base import LABEL_IGNORE_ID, PreTokenizedBatch
from embedding_trainer.tasks.registry import TASK_REGISTRY


@TASK_REGISTRY.register("mlm")
class MaskedLanguageModelingTask(BaseTask):
    """Task for masked language modeling (MLM).

    This task trains a model to predict masked tokens in the input sequence.
    The model learns contextual representations by trying to fill in the blanks.

    The loss is typically computed using cross-entropy between the predicted token
    probabilities and the true token IDs at the masked positions.

    During training, we randomly mask some percentage of the input tokens and
    replace them with a special [MASK] token. The model then tries to predict
    the original token IDs for these masked positions.

    This task is commonly used for pretraining transformer-based language models,
    such as BERT. It helps the model learn bidirectional context and improves
    performance on downstream tasks.
    """

    def compute_loss(
        self,
        model: nn.Module,
        batch: PreTokenizedBatch,
        device: str = "cpu",
    ) -> TaskOutput:
        """Compute MLM loss for a batch.

        Args:
            model: The model to compute loss for.
            batch: Dictionary containing input_ids, attention_mask, and labels tensors.
        Returns:
            TaskOutput containing the loss and any computed metrics.
        """
        input_ids = batch["input_ids"]  # shape: (batch_size, seq_length)
        attention_mask = batch["attention_mask"]  # shape: (batch_size, seq_length)
        labels = batch["labels"]  # shape: (batch_size, seq_length)

        non_blocking = device == "cuda"
        input_ids = input_ids.to(device, non_blocking=non_blocking)
        attention_mask = attention_mask.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking, dtype=torch.long)

        # Forward pass through the model to get logits
        model_output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = model_output.logits  # shape: (batch_size, seq_length, vocab_size)
        if logits is None:
            raise ValueError("Model output logits must not be None for MLM task")

        # Compute loss only at masked positions (where labels != LABEL_IGNORE_ID)
        valid_positions = labels != LABEL_IGNORE_ID  # shape: (batch_size, seq_length)
        if not valid_positions.any():
            # Keep loss connected to model graph while producing an exact zero.
            loss = logits.sum() * 0.0
            return TaskOutput(loss=loss)

        loss_fct = nn.CrossEntropyLoss()
        valid_logits = logits[valid_positions]  # shape: (num_masked_tokens, vocab_size)
        valid_labels = labels[valid_positions]  # shape: (num_masked_tokens,)

        loss = loss_fct(valid_logits, valid_labels)
        num_masked = int(valid_positions.sum())
        return TaskOutput(loss=loss, metrics={"num_masked": float(num_masked)})

    def get_metrics(self) -> dict[str, float]:
        """Return task metrics accumulated since last reset."""
        return {}
