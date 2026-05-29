import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) from scratch.

    GQA partitions query heads into `n_groups` groups. Each group shares
    a single key/value head. MQA is the degenerate case where n_kv_heads=1.

    Shapes legend (B=batch, S=seq_len, H=n_heads, Hkv=n_kv_heads, D=d_head):
        Q projections: (B, H,  S, D)   — one query head per "super-head"
        K projections: (B, Hkv, S, D)  — one KV head shared per group
        V projections: (B, Hkv, S, D)  — same grouping as K
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()

        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.d_model: int = d_model
        self.n_heads: int = n_heads          # total query heads
        self.n_kv_heads: int = n_kv_heads    # total KV heads (groups)
        self.n_groups: int = n_heads // n_kv_heads  # query heads per KV head
        self.d_head: int = d_model // n_heads
        self.dropout_p: float = dropout

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        # Projections — note K/V project to fewer heads than Q
        self.W_q: nn.Linear = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.W_k: nn.Linear = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.W_v: nn.Linear = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.W_o: nn.Linear = nn.Linear(n_heads * self.d_head, d_model, bias=bias)

        self.scale: float = 1.0 / math.sqrt(self.d_head)

        # Causal mask: upper-triangle of (S, S), True = masked
        mask: torch.Tensor = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )  # (max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, S, d_model)  — input hidden states
            attn_mask: optional (B, 1, 1, S) or (B, 1, S, S) boolean mask
                       (True/1 means *attend*, False/0 means *block*)
        Returns:
            output:    (B, S, d_model)
        """
        B, S, _ = x.shape  # (B, S, d_model)

        # ---------------------------------------------------------------
        # 1. Linear projections
        # ---------------------------------------------------------------
        q: torch.Tensor = self.W_q(x)  # (B, S, H * d_head)
        k: torch.Tensor = self.W_k(x)  # (B, S, Hkv * d_head)
        v: torch.Tensor = self.W_v(x)  # (B, S, Hkv * d_head)

        # ---------------------------------------------------------------
        # 2. Reshape into (B, n_heads, S, d_head)
        # ---------------------------------------------------------------
        q = q.view(B, S, self.n_heads, self.d_head)         # (B, S, H, D)
        q = q.transpose(1, 2)                                # (B, H, S, D)

        k = k.view(B, S, self.n_kv_heads, self.d_head)      # (B, S, Hkv, D)
        k = k.transpose(1, 2)                                # (B, Hkv, S, D)

        v = v.view(B, S, self.n_kv_heads, self.d_head)      # (B, S, Hkv, D)
        v = v.transpose(1, 2)                                # (B, Hkv, S, D)

        # ---------------------------------------------------------------
        # 3. Expand KV heads to match query heads (GQA core idea)
        #    Each group of `n_groups` query heads shares one KV head.
        #    repeat_interleave along dim=1: Hkv -> H
        # ---------------------------------------------------------------
        k = k.repeat_interleave(self.n_groups, dim=1)        # (B, H, S, D)
        v = v.repeat_interleave(self.n_groups, dim=1)        # (B, H, S, D)

        # ---------------------------------------------------------------
        # 4. Scaled dot-product attention (manual, no fused kernel)
        #    attn_weights = (Q @ K^T) / sqrt(d_head)         — (B, H, S, S)
        #    apply causal mask + optional padding mask
        #    attn_weights = softmax(attn_weights)
        #    output = attn_weights @ V                        — (B, H, S, D)
        # ---------------------------------------------------------------

        # QK^T: (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        attn_scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores * self.scale               # (B, H, S, S)

        # Causal mask: block future positions
        causal: torch.Tensor = self.causal_mask[:S, :S]      # (S, S), bool
        attn_scores = attn_scores.masked_fill(
            causal.unsqueeze(0).unsqueeze(0),  # broadcast -> (1, 1, S, S)
            float("-inf"),
        )

        # Optional padding / explicit attention mask (1 = attend, 0 = block)
        if attn_mask is not None:
            # attn_mask: (B, 1, 1, S) or (B, 1, S, S)
            attn_scores = attn_scores.masked_fill(
                ~attn_mask.bool(), float("-inf")
            )

        # Softmax along the key (last) dimension
        attn_weights: torch.Tensor = F.softmax(attn_scores, dim=-1)  # (B, H, S, S)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        # Weighted sum over values
        # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        attn_output: torch.Tensor = torch.matmul(attn_weights, v)

        # ---------------------------------------------------------------
        # 5. Concatenate heads and project output
        # ---------------------------------------------------------------
        attn_output = attn_output.transpose(1, 2)            # (B, S, H, D)
        attn_output = attn_output.contiguous().view(B, S, self.d_model)  # (B, S, d_model)

        output: torch.Tensor = self.W_o(attn_output)         # (B, S, d_model)
        return output


# ============================================================
# Quick smoke test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    B, S, D = 2, 16, 64
    N_HEADS, N_KV_HEADS = 8, 2        # 4 query heads share 1 KV head each

    gqa = GroupedQueryAttention(
        d_model=D,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        dropout=0.0,
        bias=False,
    )

    # MQA = GQA with n_kv_heads=1
    mqa = GroupedQueryAttention(
        d_model=D,
        n_heads=N_HEADS,
        n_kv_heads=1,       # single shared KV head
        dropout=0.0,
        bias=False,
    )

    # MHA = GQA with n_kv_heads == n_heads
    mha = GroupedQueryAttention(
        d_model=D,
        n_heads=N_HEADS,
        n_kv_heads=N_HEADS,  # no sharing — standard multi-head
        dropout=0.0,
        bias=False,
    )

    x = torch.randn(B, S, D)

    out_gqa = gqa(x)
    out_mqa = mqa(x)
    out_mha = mha(x)

    print(f"GQA  output shape: {out_gqa.shape}")  # (2, 16, 64)
    print(f"MQA  output shape: {out_mqa.shape}")  # (2, 16, 64)
    print(f"MHA  output shape: {out_mha.shape}")  # (2, 16, 64)

    # Gradient check
    out_gqa.sum().backward()
    print("Gradient flow OK — all parameters received gradients:",
          all(p.grad is not None for p in gqa.parameters()))
