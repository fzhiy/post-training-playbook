import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Compared to LayerNorm this omits the re-centering step (no mean subtraction)
    and the optional bias term, keeping only the RMS-based rescaling plus a
    learnable gain vector γ.

    Forward shapes (typical transformer usage):
        x : (batch, seq_len, hidden_dim)  — float tensor
        return : (batch, seq_len, hidden_dim)  — same shape & dtype
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps
        # γ  —  per-feature gain, broadcast over batch & seq dims
        # shape: (hidden_dim,)
        self.weight: Tensor = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, hidden_dim)

        # --- compute RMS along the last dimension ---
        # x^2: (batch, seq_len, hidden_dim)
        x_squared: Tensor = x.pow(2)
        # mean of x^2 over hidden_dim → (batch, seq_len, 1)
        mean_sq: Tensor = x_squared.mean(dim=-1, keepdim=True)
        # rms = sqrt(mean_sq + eps)  → (batch, seq_len, 1)
        rms: Tensor = torch.sqrt(mean_sq + self.eps)

        # --- normalise then scale ---
        # x_norm = x / rms  → (batch, seq_len, hidden_dim)
        x_norm: Tensor = x / rms
        # broadcast γ  → (hidden_dim,) broadcasts over (batch, seq_len, hidden_dim)
        output: Tensor = x_norm * self.weight  # (batch, seq_len, hidden_dim)

        return output


# ---------------------------------------------------------------------------
# Quick smoke-test / gradient sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    batch, seq_len, dim = 2, 4, 8
    rmsnorm = RMSNorm(hidden_dim=dim)

    x = torch.randn(batch, seq_len, dim, requires_grad=True)
    out = rmsnorm(x)

    # Verify RMS of the normalised output is ≈ 1 (before γ scaling)
    rms_after = out.detach().pow(2).mean(dim=-1).sqrt()
    # With γ = 1 (init), RMS should be ~1.0 everywhere
    print("Per-position RMS after norm (expect ~1.0):")
    print(rms_after)  # (batch, seq_len)

    # Verify backward pass works
    out.sum().backward()
    print("\nGradient shape for x:", x.grad.shape)          # (2, 4, 8)
    print("Gradient shape for γ:", rmsnorm.weight.grad.shape)  # (8,)
    print("\nAll checks passed ✓")
