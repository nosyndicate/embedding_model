"""Tests for EmbeddingModel pooling behavior."""

import torch

from embedding_trainer.models.model import EmbeddingModel


def test_get_embeddings_ignores_padding() -> None:
    """
    Test that get_embeddings correctly ignores padding when attention_mask is provided.

    If masked pooling is working, the embedding for a sequence should be the same
    regardless of how many padding tokens are appended at the end.
    """
    torch.manual_seed(42)
    vocab_size = 100
    hidden_size = 32
    model = EmbeddingModel(
        vocab_size=vocab_size, hidden_size=hidden_size, num_layers=2, num_heads=4
    )
    model.eval()

    # Original sequence
    input_ids_orig = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    mask_orig = torch.ones_like(input_ids_orig)

    # Padded sequence (appended 0s)
    pad_id = 0
    input_ids_padded = torch.tensor(
        [[10, 20, 30, 40, pad_id, pad_id, pad_id, pad_id]], dtype=torch.long
    )
    mask_padded = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        emb_orig = model.get_embeddings(input_ids_orig, attention_mask=mask_orig)
        emb_padded = model.get_embeddings(input_ids_padded, attention_mask=mask_padded)

    # Check if they are close.
    # Current implementation (simple mean) will fail this as it averages over 8 tokens instead of 4.
    torch.testing.assert_close(emb_orig, emb_padded, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_get_embeddings_ignores_padding()
