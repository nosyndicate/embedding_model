"""Tests for RoPE (Rotary Positional Embeddings) implementation."""

from __future__ import annotations

import pytest
import torch

from embedding_trainer.models.model import apply_rope, precompute_rope_cache


class TestPrecomputeRopeCache:
    def test_output_shapes(self) -> None:
        dim, max_seq_len = 64, 128
        cos, sin = precompute_rope_cache(dim, max_seq_len)
        assert cos.shape == (max_seq_len, dim // 2)
        assert sin.shape == (max_seq_len, dim // 2)

    def test_values_in_range(self) -> None:
        cos, sin = precompute_rope_cache(dim=32, max_seq_len=64)
        assert cos.min() >= -1.0
        assert cos.max() <= 1.0
        assert sin.min() >= -1.0
        assert sin.max() <= 1.0

    def test_first_position_is_zero_angle(self) -> None:
        cos, sin = precompute_rope_cache(dim=16, max_seq_len=10)
        # Position 0: angle = 0 * theta = 0, so cos=1, sin=0
        torch.testing.assert_close(cos[0], torch.ones(8))
        torch.testing.assert_close(sin[0], torch.zeros(8))

    def test_different_base(self) -> None:
        cos_a, sin_a = precompute_rope_cache(dim=16, max_seq_len=10, base=10000)
        cos_b, sin_b = precompute_rope_cache(dim=16, max_seq_len=10, base=500)
        assert not torch.allclose(cos_a, cos_b)
        assert not torch.allclose(sin_a, sin_b)

    def test_odd_dim_raises(self) -> None:
        with pytest.raises(AssertionError, match="even"):
            precompute_rope_cache(dim=7, max_seq_len=10)


class TestApplyRope:
    def test_output_shape_matches_input(self) -> None:
        dim, seq_len = 16, 8
        x = torch.randn(2, seq_len, dim)
        cos, sin = precompute_rope_cache(dim, seq_len)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_zero_position_identity(self) -> None:
        dim = 16
        x = torch.randn(1, 1, dim)
        cos, sin = precompute_rope_cache(dim, max_seq_len=1)
        # At position 0, cos=1 and sin=0, so output == input
        out = apply_rope(x, cos, sin)
        torch.testing.assert_close(out, x)

    def test_rotation_preserves_norm(self) -> None:
        dim, seq_len = 32, 16
        x = torch.randn(4, seq_len, dim)
        cos, sin = precompute_rope_cache(dim, seq_len)
        out = apply_rope(x, cos, sin)
        input_norms = torch.norm(x, dim=-1)
        output_norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(input_norms, output_norms)

    def test_known_rotation(self) -> None:
        # 2D case: x = [1, 0], rotate by angle theta
        # Expected: [cos(theta), sin(theta)]
        dim = 2
        cos, sin = precompute_rope_cache(dim, max_seq_len=2)
        # Use position 1 (position 0 is identity)
        x = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
        c = cos[1:2].unsqueeze(0)  # (1, 1, 1)
        s = sin[1:2].unsqueeze(0)
        out = apply_rope(x, c, s)
        # x1=1, x2=0 -> x1_rot = 1*cos - 0*sin = cos, x2_rot = 0*cos + 1*sin = sin
        expected = torch.tensor([[[c.item(), s.item()]]])
        torch.testing.assert_close(out, expected)

    def test_different_positions_produce_different_outputs(self) -> None:
        dim = 16
        x = torch.randn(1, 1, dim)
        cos, sin = precompute_rope_cache(dim, max_seq_len=10)
        out_pos0 = apply_rope(x, cos[0:1].unsqueeze(0), sin[0:1].unsqueeze(0))
        out_pos5 = apply_rope(x, cos[5:6].unsqueeze(0), sin[5:6].unsqueeze(0))
        assert not torch.allclose(out_pos0, out_pos5)

    def test_inverse_rotation(self) -> None:
        dim, seq_len = 32, 8
        x = torch.randn(2, seq_len, dim)
        cos, sin = precompute_rope_cache(dim, seq_len)
        # Apply forward rotation
        rotated = apply_rope(x, cos, sin)
        # Apply inverse rotation (negate sin)
        recovered = apply_rope(rotated, cos, -sin)
        torch.testing.assert_close(recovered, x)

    def test_multi_dim_broadcasting(self) -> None:
        # Realistic shape: (batch, seq_len, num_heads, head_dim)
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 16
        x = torch.randn(batch, seq_len, num_heads, head_dim)
        cos, sin = precompute_rope_cache(head_dim, seq_len)
        # Reshape cos/sin to (1, seq_len, 1, head_dim//2) for broadcasting
        cos_b = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
        sin_b = sin.unsqueeze(0).unsqueeze(2)
        out = apply_rope(x, cos_b, sin_b)
        assert out.shape == x.shape
        # Verify norm preservation across all dims
        input_norms = torch.norm(x, dim=-1)
        output_norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(input_norms, output_norms)

    def test_odd_last_dim(self) -> None:
        # chunk(2, dim=-1) with odd last dim produces unequal chunks
        x = torch.randn(1, 1, 5)
        cos = torch.ones(1, 1, 2)
        sin = torch.zeros(1, 1, 2)
        # x.chunk(2, dim=-1) -> shapes (1,1,3) and (1,1,2), mismatched with cos/sin
        with pytest.raises((RuntimeError, ValueError)):
            apply_rope(x, cos, sin)

    def test_rotation_additivity(self) -> None:
        # Rotating by position a then position b == rotating by position a+b
        dim = 16
        x = torch.randn(1, 1, dim)
        cos, sin = precompute_rope_cache(dim, max_seq_len=20)

        a, b = 3, 5
        # Rotate by position a
        cos_a = cos[a : a + 1].unsqueeze(0)
        sin_a = sin[a : a + 1].unsqueeze(0)
        rotated_a = apply_rope(x, cos_a, sin_a)

        # Then rotate by position b
        cos_b = cos[b : b + 1].unsqueeze(0)
        sin_b = sin[b : b + 1].unsqueeze(0)
        rotated_ab = apply_rope(rotated_a, cos_b, sin_b)

        # Rotate directly by position a+b
        cos_ab = cos[a + b : a + b + 1].unsqueeze(0)
        sin_ab = sin[a + b : a + b + 1].unsqueeze(0)
        rotated_direct = apply_rope(x, cos_ab, sin_ab)

        torch.testing.assert_close(rotated_ab, rotated_direct, atol=1e-5, rtol=1e-5)

    def test_float16_norm_preservation(self) -> None:
        dim, seq_len = 32, 8
        x = torch.randn(2, seq_len, dim, dtype=torch.float16)
        cos, sin = precompute_rope_cache(dim, seq_len, dtype=torch.float16)
        out = apply_rope(x, cos, sin)
        input_norms = torch.norm(x.float(), dim=-1)
        output_norms = torch.norm(out.float(), dim=-1)
        torch.testing.assert_close(input_norms, output_norms, atol=1e-2, rtol=1e-2)
