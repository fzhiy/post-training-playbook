import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for a frozen linear layer.

    Forward:
        y = W x + (alpha / r) * B @ A @ x

    where:
        W : (out_features, in_features)   -- frozen pretrained weights
        A : (rank, in_features)            -- low-rank "down-projection"
        B : (out_features, rank)           -- low-rank "up-projection"

    At init: A ~ Kaiming uniform, B = 0  =>  delta W = 0 (identity start).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.alpha: float = alpha
        self.scaling: float = alpha / rank  # (scalar) α / r

        # ---- frozen pretrained weight ----
        # shape: (out_features, in_features)
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype), requires_grad=False
        )
        # shape: (out_features,)
        self.bias: Optional[nn.Parameter]
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype), requires_grad=False
            )
        else:
            self.bias = None

        # ---- LoRA trainable parameters ----
        # shape: (rank, in_features)
        self.lora_A: nn.Parameter = nn.Parameter(torch.empty(rank, in_features, dtype=dtype))
        # shape: (out_features, rank)
        self.lora_B: nn.Parameter = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))

        # ---- initialise A (B stays zero so delta_W starts at zero) ----
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        self.merged: bool = False  # tracks whether weights are currently merged

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def compute_delta_weight(self) -> torch.Tensor:
        """
        Compute the low-rank weight update matrix.

        Returns
        -------
        delta_w : Tensor, shape (out_features, in_features)
            (alpha / r) * B @ A
        """
        # B: (out_features, rank)  @  A: (rank, in_features)
        #    -> (out_features, in_features)
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge_weights(self) -> None:
        """
        Fold LoRA into the frozen weight:  W ← W + (α/r)·B·A.
        Call once before inference to remove the extra matmul at runtime.
        """
        if self.merged:
            raise RuntimeError("Weights are already merged.")
        # delta_w: (out_features, in_features)
        delta_w: torch.Tensor = self.compute_delta_weight()
        # W: (out_features, in_features)  +=  delta_w: (out_features, in_features)
        self.weight.data.add_(delta_w)
        self.merged = True

    def unmerge_weights(self) -> None:
        """
        Reverse of merge:  W ← W − (α/r)·B·A.
        """
        if not self.merged:
            raise RuntimeError("Weights are not currently merged.")
        delta_w: torch.Tensor = self.compute_delta_weight()
        self.weight.data.sub_(delta_w)
        self.merged = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (*, in_features)   -- any leading batch dims

        Returns
        -------
        y : Tensor, shape (*, out_features)
        """
        # ---- frozen linear:  (*, in) -> (*, out) ----
        # W: (out, in)  @  x: (*, in)^T  ->  (*, out)
        result: torch.Tensor = F.linear(x, self.weight, self.bias)

        if not self.merged:
            # ---- LoRA path: (α/r) · B · A · x ----
            # x:        (*, in_features)
            # A @ x^T:  (rank, in) @ (in, *) -> (rank, *)
            # B @ (A@x): (out, rank) @ (rank, *) -> (out, *)
            # Transpose back: (*, out_features)
            #
            # In one shot using F.linear (which transposes weight internally):
            #   lora_A: (rank, in_features)   -> "weight" for down-proj
            #   lora_B: (out_features, rank)  -> "weight" for up-proj

            # Step 1: down-project   (*, in) -> (*, rank)
            # lora_down: shape (*, rank)
            lora_down: torch.Tensor = F.linear(x, self.lora_A)  # A @ x^T transposed

            # Step 2: up-project     (*, rank) -> (*, out)
            # lora_out: shape (*, out_features)
            lora_out: torch.Tensor = F.linear(lora_down, self.lora_B)  # B @ (A x)

            # Step 3: scale and add
            # result: (*, out_features) += (α/r) * lora_out
            result.add_(lora_out, alpha=self.scaling)

        return result


# ======================================================================
# Quick self-contained test / demo
# ======================================================================
if __name__ == "__main__":

    torch.manual_seed(42)

    batch: int = 4
    in_dim: int = 16
    out_dim: int = 8
    r: int = 2

    layer = LoRALinear(in_dim, out_dim, rank=r, alpha=4.0)

    # Fill frozen weight with something non-zero
    nn.init.kaiming_uniform_(layer.weight)

    x: torch.Tensor = torch.randn(batch, in_dim)  # (4, 16)

    # --- Forward (unmerged path) ---
    y_unmerged: torch.Tensor = layer(x)  # (4, 8)
    print(f"y_unmerged shape: {y_unmerged.shape}")
    print(f"y_unmerged:\n{y_unmerged}\n")

    # --- Manual recompute for sanity check ---
    # delta_w: (out, in)  = (α/r) * B @ A
    delta_w: torch.Tensor = layer.compute_delta_weight()
    # y_manual: (batch, out) = x @ (W + delta_w)^T + bias
    y_manual: torch.Tensor = F.linear(x, layer.weight + delta_w, layer.bias)
    print(f"Manual match: {torch.allclose(y_unmerged, y_manual, atol=1e-5)}\n")

    # --- Merge, then forward should give same result ---
    layer.merge_weights()
    assert layer.merged is True
    y_merged: torch.Tensor = layer(x)  # (4, 8)
    print(f"Merged match:  {torch.allclose(y_unmerged, y_merged, atol=1e-5)}\n")

    # --- Unmerge restores original frozen weight ---
    layer.unmerge_weights()
    y_unmerged2: torch.Tensor = layer(x)
    print(f"Round-trip:    {torch.allclose(y_unmerged, y_unmerged2, atol=1e-5)}")
