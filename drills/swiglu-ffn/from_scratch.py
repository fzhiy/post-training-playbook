"""SwiGLU Feed-Forward Block — pure PyTorch implementation from scratch.

SwiGLU(x) = (Swish(x @ W1) ⊙ (x @ W3)) @ W2

Reference: Shazeer (2020), "GLU Variants Improve Transformer"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    """Feed-forward block using the SwiGLU activation.

    SwiGLU combines the Swish activation with a Gated Linear Unit (GLU):
        gate  = Swish(x @ W1)          # (B, T, d_ff)
        value = x @ W3                  # (B, T, d_ff)
        out   = (gate * value) @ W2     # (B, T, d_model)

    Unlike a standard FFN (two weight matrices), SwiGLU requires three
    weight matrices because the gating path consumes an extra projection.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Follow the LLaMA convention: default d_ff = 8/3 * d_model, rounded
        # to the nearest multiple of 256 for hardware efficiency.
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 255) // 256) * 256

        self.d_model: int = d_model
        self.d_ff: int = d_ff

        # Three projection matrices (no activation fusion; explicit math)
        # W1: gate projection   (d_model → d_ff)
        # W3: value projection  (d_model → d_ff)
        # W2: down projection   (d_ff → d_model)
        self.W1: nn.Linear = nn.Linear(d_model, d_ff, bias=bias)
        self.W3: nn.Linear = nn.Linear(d_model, d_ff, bias=bias)
        self.W2: nn.Linear = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Swish / SiLU activation, implemented from scratch
    # ------------------------------------------------------------------
    @staticmethod
    def swish(x: Tensor) -> Tensor:
        """Swish(x) = x * sigmoid(x)   (element-wise).

        This is the same as SiLU.  We implement it explicitly rather than
        calling F.silu or F.swish so the math is fully transparent.
        """
        return x * torch.sigmoid(x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the SwiGLU block.

        Args:
            x: Input tensor of shape (B, T, d_model).
               B = batch size, T = sequence length, d_model = model width.

        Returns:
            Tensor of shape (B, T, d_model), same shape as input.
        """
        # ---- gate path ----
        gate: Tensor = self.W1(x)           # (B, T, d_model) → (B, T, d_ff)
        gate = self.swish(gate)             # (B, T, d_ff)  — Swish activation

        # ---- value path ----
        value: Tensor = self.W3(x)          # (B, T, d_model) → (B, T, d_ff)

        # ---- gated combination (element-wise multiply) ----
        gated: Tensor = gate * value        # (B, T, d_ff) ⊙ (B, T, d_ff) → (B, T, d_ff)

        # ---- output projection ----
        out: Tensor = self.W2(gated)        # (B, T, d_ff) → (B, T, d_model)
        out = self.dropout(out)             # (B, T, d_model)

        return out


# ======================================================================
# Quick sanity check (runs only when executed as a script)
# ======================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, d_model, d_ff = 2, 16, 512, 1376
    x: Tensor = torch.randn(B, T, d_model)

    block: SwiGLU = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=0.1)
    y: Tensor = block(x)

    print(f"Input shape : {tuple(x.shape)}")   # (2, 16, 512)
    print(f"Output shape: {tuple(y.shape)}")    # (2, 16, 512)
    print(f"Parameters  : {sum(p.numel() for p in block.parameters()):,}")
    assert y.shape == x.shape, "Shape mismatch!"

    # Verify Swish is correct against F.silu
    test: Tensor = torch.randn(4, 8)
    assert torch.allclose(SwiGLU.swish(test), F.silu(test), atol=1e-7), (
        "Swish implementation does not match F.silu"
    )
    print("All checks passed.")
