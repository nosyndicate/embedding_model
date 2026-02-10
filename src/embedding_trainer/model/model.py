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
        # TODO: maybe try RMSNorm instead of LayerNorm
        self.norm = nn.LayerNorm(embedding_dim) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states: Tensor = self.embedding(input_ids)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """
        Forward pass through a single transformer layer using pre-layer normalization.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_length) or None

        Returns:
            Tensor of shape (batch_size, seq_length, hidden_size)
        """
        x = self.norm1(hidden_states)
        attn_output = self.attention(x, x, x, need_weights=False)
        hidden_states = hidden_states + self.dropout(attn_output)

        hidden_states = hidden_states + self.dropout(
            self.mlp(self.norm2(hidden_states))
        )
        return hidden_states


class EmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int
    ) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size, embedding_dim=hidden_size
        )
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        # we need head

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> ModelOutput:
        hidden_states = self.embedding(input_ids)
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        # TODO: apply head
        return ModelOutput(last_hidden_state=hidden_states)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size
