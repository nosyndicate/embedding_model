import math
from typing import Any

import torch
from torch import Tensor, nn

from embedding_trainer.core.base_model import BaseEmbeddingModel
from embedding_trainer.core.types import ModelOutput
from embedding_trainer.models.registry import MODEL_REGISTRY


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


def precompute_rope_cache(
    dim: int,
    max_seq_len: int,
    base: int = 10000,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute the frequencies for RoPE:
    Pairing is: (x[i], x[i + dim//2]) for i in [0, dim//2).

    m * theta_0, m * theta_1, ..., m * theta_{dim//2 - 1}

    frequences = 1 / (base ** (i / dim)) for i in [0, 2, 4, ..., dim-2]
    Args:
        dim: The dimension of the model (hidden size).
        max_seq_len: The maximum sequence length to precompute for.
        base: The base for computing the frequencies (default: 10000).
            Higher base means slower decay of frequencies, which can be beneficial for longer sequences.
        device: The device to store the precomputed frequencies on (e.g., 'cpu' or 'cuda').
        dtype: The data type for the precomputed frequencies (default: torch.float32 to avoid
            precision issue with sine/cosine computation).

    Returns:
        A tuple of tensors (cos, sin) each of shape (max_seq_len, dim // 2) containing the precomputed frequencies.
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE."

    # Compute theta for each dimension
    theta = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    )  # shape: (dim // 2,)

    # Create the position indices for all positions
    seq_index = torch.arange(
        max_seq_len, device=device, dtype=dtype
    )  # shape: (max_seq_len,)

    # Compute the rotation angle, m * theta, for all positions and dimensions
    index_theta = torch.outer(seq_index, theta)  # shape: (max_seq_len, dim // 2)

    # Compute the cosine and sine of the rotation angles
    cos = index_theta.cos()  # shape: (max_seq_len, dim // 2)
    sin = index_theta.sin()  # shape: (max_seq_len, dim // 2)
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to the input tensor.

    Using pairing  (x[i], x[i + dim//2]) for i in [0, dim//2), the RoPE transformation is:
    The rotation matrix is
        [[cos, -sin],
         [sin,  cos]]

    So we compute:
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x2 * cos + x1 * sin

    Args:
        x: Input tensor of shape (..., dim)
        cos: Precomputed cosine frequencies of shape (max_seq_len, dim // 2)
        sin: Precomputed sine frequencies of shape (max_seq_len, dim // 2)

    Returns:
        Tensor of shape (..., dim) with RoPE applied.
    """
    x1, x2 = x.chunk(2, dim=-1)  # shape: (..., dim//2), (..., dim//2)
    x1_rotated = x1 * cos - x2 * sin  # shape: (..., dim//2)
    x2_rotated = x2 * cos + x1 * sin  # shape: (..., dim//2)
    return torch.cat([x1_rotated, x2_rotated], dim=-1)  # shape: (..., dim)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.project = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )
        self.d_k = hidden_size // num_heads
        self.sqrt_d_k = math.sqrt(self.d_k)
        self.dropout = nn.Dropout(dropout)
        cos, sin = precompute_rope_cache(
            dim=self.d_k, max_seq_len=self.max_seq_len, device=device, dtype=dtype
        )
        self.register_buffer("cos_cache", cos, persistent=True)
        self.register_buffer("sin_cache", sin, persistent=True)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Compute self-attention with RoPE.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_length) or None
                1 for position to keep, 0 for position to mask

        Returns:
            Tensor of shape (batch_size, seq_length, hidden_size) after self-attention.
        """

        B, T, _ = hidden_states.size()
        qkv = self.project(hidden_states)  # shape: (B, T, hidden_size * 3)
        q, k, v = qkv.chunk(3, dim=-1)  # each shape: (B, T, hidden_size)

        # Reshape and transpose for multi-head attention
        q = q.contiguous().view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.contiguous().view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.contiguous().view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        if self.max_seq_len < T:
            raise ValueError(
                f"Sequence length {T} exceeds maximum supported {self.max_seq_len}"
            )

        q_rope = apply_rope(q, self.cos_cache[:T], self.sin_cache[:T])
        k_rope = apply_rope(k, self.cos_cache[:T], self.sin_cache[:T])

        attn_weights = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / self.sqrt_d_k

        if attention_mask is not None:
            key_padding = (attention_mask == 0)[:, None, None, :]
            attn_weights = attn_weights.masked_fill(key_padding, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)  # shape:(B, N_H, T, T)

        attn_output = torch.matmul(attn_weights, v)  # shape: (B, N_H, T, d_k)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        )  # shape: (B, T, hidden_size)
        attn_output = self.out_proj(attn_output)  # shape: (B, T, hidden_size)
        attn_output = self.dropout(attn_output)
        return attn_output


class TransformerLayer(nn.Module):
    """"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.attention = SelfAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
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
        attn_output = self.attention(x, attention_mask=attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)

        hidden_states = hidden_states + self.dropout(
            self.mlp(self.norm2(hidden_states))
        )
        return hidden_states


@MODEL_REGISTRY.register("base_embedding_model")
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
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> ModelOutput:
        hidden_states = self.embedding(input_ids)
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        logits = self.head(hidden_states)
        return ModelOutput(hidden_states=hidden_states, logits=logits)

    def get_embeddings(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """
        Get embeddings for input sequences.

        Args:
            input_ids: Tensor of shape (batch_size, seq_length) containing token IDs.
            attention_mask: Tensor of shape (batch_size, seq_length) or None
                1 for position to keep, 0 for position to mask
        Returns:
            Tensor of shape (batch_size, hidden_size) containing the pooled embeddings.
        """
        hidden_states = self.embedding(input_ids)
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # Masked mean pooling
        # attention_mask shape: (batch_size, seq_len)
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_sum = torch.sum(hidden_states * mask, dim=1)
        mask_sum = torch.sum(mask, dim=1).clamp(min=1e-9)
        embeddings: Tensor = masked_sum / mask_sum
        return embeddings

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def get_param_groups(self, **kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        weight_decay = kwargs.get("weight_decay")
        if weight_decay is not None:
            no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
            param_groups: list[dict[str, Any]] = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            return param_groups

        return [{"params": list(self.parameters())}]
