from torch import Tensor, nn

from embedding_trainer.core.base_model import BaseEmbeddingModel
from embedding_trainer.core.types import ModelOutput


class EmbeddingLayer(nn.Module):
    """
    A simple embedding layer with optional dropout and layer normalization.

    We don't add positional embeddings here since we plan to use RoPE (Rotary Positional Embeddings)
    in the transformer layers.

    Typically, dropout is applied before we send the embeddings to the next layer (e.g., a transformer layer).
    For encoding model, dropout helps prevent overfitting and improves generalization. But
    for decoder model, dropout rate is usually set to 0.

    Often, layer normalization is not used (at least in the original transformer paper).
    But adding layernorm before dropout can often stabilize training for deep models.
    This shows up more in
    - vision transformers
    - modern encoder variants

    We also have to consider the Pre-LayerNorm (Pre-LN) and Post-LayerNorm (Post-LN) configurations.
    If we use Pre-LN, then using layernorm here might be redundant.

    Another tricks is to do
        embedding * sqrt(d_model)
    This keeps the variance of the embeddings aligned with attention math.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.embedding(input_ids)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()

        # we need embedding layer
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size, embedding_dim=hidden_size
        )
        # we need several transformer layers
        # we need head

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> ModelOutput:
        hidden_states = self.embedding(input_ids)
        # TODO: pass through transformer layers
        # TODO: apply head
        return ModelOutput(last_hidden_state=hidden_states)
