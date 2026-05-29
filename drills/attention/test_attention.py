"""Correctness tests: our from-scratch attention vs PyTorch references.

    python test_attention.py            # plain run
    python -m pytest test_attention.py  # or via pytest
"""
import torch
import torch.nn.functional as F

from from_scratch import MultiHeadAttention, scaled_dot_product_attention


def test_sdpa_matches_torch():
    torch.manual_seed(0)
    B, H, L, d = 2, 4, 8, 16
    q, k, v = (torch.randn(B, H, L, d) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_causal_matches_torch():
    torch.manual_seed(0)
    B, H, L, d = 2, 4, 8, 16
    q, k, v = (torch.randn(B, H, L, d) for _ in range(3))
    ours, w = scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)
    # strictly-future positions must carry exactly zero weight
    future = torch.triu(torch.ones(L, L), diagonal=1).bool()
    assert (w[..., future] == 0).all()


def test_mha_shape_and_grad():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 10, 32, requires_grad=True)
    y = mha(x, is_causal=True)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


if __name__ == "__main__":
    test_sdpa_matches_torch()
    test_causal_matches_torch()
    test_mha_shape_and_grad()
    print("all attention drills passed ✓")
