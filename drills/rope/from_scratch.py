"""
Rotary Position Embedding (RoPE) — pure-PyTorch, from-scratch implementation.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Frequency precomputation
# ---------------------------------------------------------------------------

def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cos/sin lookup tables for RoPE.

    For each dimension-pair index i ∈ {0 … d/2−1} and sequence position m:
        θ_{m,i} = m · base^{−2i/d}

    The tables are *interleaved* so they are directly element-wise-multipliable
    against a flattened head vector:
        [cos θ₀, cos θ₀, cos θ₁, cos θ₁, …]

    Args:
        head_dim:     d — per-head dimension (must be even).
        max_seq_len:  precompute window size.
        base:         frequency base (default 10 000).
        dtype:        output dtype.
        device:       output device.

    Returns:
        cos_table : (max_seq_len, head_dim)
        sin_table : (max_seq_len, head_dim)
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # Per-pair inverse frequencies  θ_i = base^{−2i/d}
    # shape: (head_dim // 2,)
    pair_indices = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq: torch.Tensor = 1.0 / (base ** (pair_indices / head_dim))

    # Position vector  m = 0, 1, …, T−1
    # shape: (max_seq_len,)
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

    # Outer product → all (m, i) angles
    # shape: (max_seq_len, head_dim // 2)
    angles = torch.outer(positions, inv_freq)

    cos_half = torch.cos(angles)   # (max_seq_len, head_dim // 2)
    sin_half = torch.sin(angles)   # (max_seq_len, head_dim // 2)

    # Interleave each value twice to match full head_dim:
    #   [cos₀, cos₀, cos₁, cos₁, …]  →  (max_seq_len, head_dim)
    cos_table = torch.repeat_interleave(cos_half, repeats=2, dim=-1)
    sin_table = torch.repeat_interleave(sin_half, repeats=2, dim=-1)

    return cos_table.to(dtype), sin_table.to(dtype)


# ---------------------------------------------------------------------------
# 2. Pair-wise 90° rotation helper
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Treat the last dimension as consecutive pairs (x_{2i}, x_{2i+1})
    and rotate each pair by 90°.

        [x₀, x₁, x₂, x₃, …]  →  [−x₁, x₀, −x₃, x₂, …]

    Args:
        x: (…, d)

    Returns:
        (…, d)
    """
    x_even = x[..., 0::2]   # (…, d / 2)   picks x₀, x₂, x₄, …
    x_odd  = x[..., 1::2]   # (…, d / 2)   picks x₁, x₃, x₅, …
    # stack as [−x_odd, x_even] per pair, then flatten
    # (…, d / 2, 2) → (…, d)
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# 3. Core RoPE application
# ---------------------------------------------------------------------------

def apply_rope(
    x: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding.

    Per-pair formula at position m, pair index i:
        x'_{2i}   =  x_{2i}  · cos θ_{m,i}  −  x_{2i+1} · sin θ_{m,i}
        x'_{2i+1} =  x_{2i}  · sin θ_{m,i}  +  x_{2i+1} · cos θ_{m,i}

    Compact form:  x' = x ⊙ cos  +  rotate_half(x) ⊙ sin

    Args:
        x:          (batch, seq_len, num_heads, head_dim)
        cos_table:  (max_seq_len, head_dim)
        sin_table:  (max_seq_len, head_dim)
        positions:  (batch, seq_len) of integer position ids,
                    or None to use 0, 1, …, seq_len−1.

    Returns:
        (batch, seq_len, num_heads, head_dim) — same shape & dtype as x.
    """
    batch, seq_len, num_heads, head_dim = x.shape  # noqa: F841

    # ---- gather cos / sin for the relevant positions ----
    if positions is None:
        idx = torch.arange(seq_len, device=x.device)           # (seq_len,)
        cos = cos_table[idx].unsqueeze(0)                       # (1, seq_len, d)
        sin = sin_table[idx].unsqueeze(0)                       # (1, seq_len, d)
    else:
        cos = cos_table[positions]                              # (B, seq_len, d)
        sin = sin_table[positions]                              # (B, seq_len, d)

    # Broadcast across the heads dimension
    cos = cos.unsqueeze(2)                                      # (B, S, 1, d)
    sin = sin.unsqueeze(2)                                      # (B, S, 1, d)

    # ---- rotate ----
    return x * cos + rotate_half(x) * sin                       # (B, S, H, d)


# ---------------------------------------------------------------------------
# 4. Reusable nn.Module wrapper
# ---------------------------------------------------------------------------

class RoPE(nn.Module):
    """
    Drop-in RoPE module.

    Usage:
        rope = RoPE(head_dim=64, max_seq_len=4096)
        q_rot = rope(q)          # apply to queries
        k_rot = rope(k)          # apply to keys (same call)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        cos_table, sin_table = precompute_rope_frequencies(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            base=base,
        )
        # Buffers move with .to(device) / .cuda() but are not nn.Parameters
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (batch, seq_len, num_heads, head_dim)
            positions: (batch, seq_len) or None

        Returns:
            (batch, seq_len, num_heads, head_dim)
        """
        return apply_rope(x, self.cos_table, self.sin_table, positions)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_seq_len={self.max_seq_len}"


# ---------------------------------------------------------------------------
# 5. Smoke tests (run:  python rope.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, S, H, D = 2, 32, 8, 64          # batch, seq, heads, head_dim
    device = "cpu"

    rope = RoPE(head_dim=D, max_seq_len=512, base=10000.0).to(device)

    q = torch.randn(B, S, H, D, device=device)
    k = torch.randn(B, S, H, D, device=device)

    q_rot = rope(q)
    k_rot = rope(k)

    # ---- Test 1: RoPE is an orthogonal rotation → preserves L2 norms ----
    assert torch.allclose(
        q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5
    ), "FAIL: RoPE did not preserve query norms"
    assert torch.allclose(
        k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5
    ), "FAIL: RoPE did not preserve key norms"
    print("✓  Norm-preservation (rotation) verified.")

    # ---- Test 2: relative-position property ----
    # <ROPE(q,m), ROPE(k,n)> depends only on (m−n).
    # Equivalently, shifting both by +d leaves dot-products unchanged.
    SHIFT = 5
    base_pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)   # (B, S)

    q_p0 = rope(q, positions=base_pos)
    k_p0 = rope(k, positions=base_pos)

    q_pS = rope(q, positions=base_pos + SHIFT)
    k_pS = rope(k, positions=base_pos + SHIFT)

    dots_before = (q_p0 * k_p0).sum(dim=-1)      # (B, S)
    dots_after  = (q_pS * k_pS).sum(dim=-1)       # (B, S)
    assert torch.allclose(dots_before, dots_after, atol=1e-4), \
        "FAIL: relative-position property violated"
    print("✓  Relative-position shift-invariance verified.")

    # ---- Test 3: different absolute positions, same relative offset ----
    # <ROPE(q, 10), ROPE(k, 7)>  vs  <ROPE(q, 20), ROPE(k, 17)>
    # both have relative position 3
    pos_a = torch.full((B, 1), 10, device=device)
    pos_b = torch.full((B, 1),  7, device=device)
    pos_c = torch.full((B, 1), 20, device=device)
    pos_d = torch.full((B, 1), 17, device=device)

    dot_ab = (rope(q[:, 10:11], pos_a) * rope(k[:, 7:8], pos_b)).sum(dim=-1)
    dot_cd = (rope(q[:, 20:21], pos_c) * rope(k[:, 17:18], pos_d)).sum(dim=-1)
    assert torch.allclose(dot_ab, dot_cd, atol=1e-4), \
        "FAIL: same-relative-offset dot products differ"
    print("✓  Same-relative-offset dot-product equality verified.")

    # ---- Test 4: shape correctness ----
    assert q_rot.shape == (B, S, H, D), f"Shape mismatch: {q_rot.shape}"
    print(f"✓  Output shape correct: {tuple(q_rot.shape)}")

    print(f"\n{rope}")
    print("\nAll tests passed.")
