"""Scaled dot-product & multi-head attention, from scratch.

No nn.MultiheadAttention, no F.scaled_dot_product_attention — the whole point
is to be able to derive and defend every line in an interview. See README.md
for the math and the stratified follow-up questions.

Requires: torch >= 2.0 (only for the reference comparison in the tests).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,                     # (..., Lq, d_k)
    k: torch.Tensor,                     # (..., Lk, d_k)
    v: torch.Tensor,                     # (..., Lk, d_v)
    mask: torch.Tensor | None = None,    # (..., Lq, Lk) bool, True = keep
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (output (..., Lq, d_v), attention weights (..., Lq, Lk))."""
    d_k = q.size(-1)
    # (..., Lq, Lk): similarity of every query against every key.
    # Scale by sqrt(d_k): Var(q·k) grows ~linearly in d_k, so without scaling
    # softmax saturates and gradients vanish. Dividing pulls the variance to ~1.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if is_causal:
        Lq, Lk = scores.size(-2), scores.size(-1)
        causal = torch.ones(Lq, Lk, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Fill with -inf (not 0) so these positions get exactly zero weight AFTER
    # softmax; filling 0 would still leak a non-zero probability.
    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Self-attention over (B, L, d_model). Splits d_model into n_heads
    subspaces, attends independently per head, concatenates, projects out."""

    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout_p = dropout_p

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        # (B, L, d_model) -> (B, n_heads, L, d_head)
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x, mask=None, is_causal=False):
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, Lq, Lk) -> broadcast over heads
        out, _ = scaled_dot_product_attention(
            q, k, v, mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        B, _, L, _ = out.shape
        # (B, n_heads, L, d_head) -> (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(out)
