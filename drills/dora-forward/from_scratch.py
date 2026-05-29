import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA).

    Decomposes the adapted weight into magnitude and direction:
        W' = m * (W₀ + s·BA) / ‖W₀ + s·BA‖_c

    where:
        W₀ : frozen pretrained weight   (out, in)
        B  : LoRA up-projection         (out, r)
        A  : LoRA down-projection       (r, in)
        s  : scaling = lora_alpha / r
        m  : learnable magnitude        (out,)
        ‖·‖_c : per-row L2 norm

    At init: B = 0, so ΔW = 0 and output equals the pretrained model.
    The magnitude m is initialised to ‖W₀‖_c, preserving the pretrained
    magnitude–direction decomposition exactly.

    Reference: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation"
               arXiv:2402.09353 (2024).
    """

    def __init__(
        self,
        pretrained_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert r > 0, "LoRA rank must be a positive integer"

        self.in_features: int = pretrained_linear.in_features
        self.out_features: int = pretrained_linear.out_features
        self.r: int = r
        self.scaling: float = lora_alpha / r

        # ── Frozen pretrained parameters ──────────────────────────────
        self.weight: nn.Parameter = pretrained_linear.weight  # (out, in)
        self.weight.requires_grad_(False)

        if pretrained_linear.bias is not None:
            self.bias: Optional[nn.Parameter] = pretrained_linear.bias
            self.bias.requires_grad_(False)
        else:
            self.bias = None  # (out,) or None

        # ── LoRA low-rank factors ─────────────────────────────────────
        # A: project  in_features → r
        self.lora_A: nn.Parameter = nn.Parameter(torch.empty(r, self.in_features))
        # B: project  r → out_features
        self.lora_B: nn.Parameter = nn.Parameter(torch.empty(self.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)  # ΔW = 0 at init → pretrained output preserved

        # ── Learnable magnitude ───────────────────────────────────────
        # m₀ = ‖W₀‖_c  (per-row L2 norm of each output neuron's weights)
        with torch.no_grad():
            m_init: torch.Tensor = torch.norm(self.weight, p=2, dim=1)  # (out,)
        self.magnitude: nn.Parameter = nn.Parameter(m_init)

        # ── Dropout ───────────────────────────────────────────────────
        self.lora_dropout: nn.Module = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

        # ── Merge bookkeeping ─────────────────────────────────────────
        self.merged: bool = False
        self._original_weight: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (*, in_features)
        Returns:
            (*, out_features)
        """
        if self.merged:
            # Weight already contains the DoRA result; pure mat-mul.
            return F.linear(x, self.weight, self.bias)

        x = self.lora_dropout(x)

        # LoRA additive update
        delta_w: torch.Tensor = self.lora_B @ self.lora_A              # (out, in)

        # Adapted weight (pretrained + scaled low-rank update)
        adapted_w: torch.Tensor = self.weight + self.scaling * delta_w  # (out, in)

        # Per-row L2 norm  ‖W_adapted‖_c
        norm: torch.Tensor = torch.norm(                                # (out, 1)
            adapted_w, p=2, dim=1, keepdim=True
        )

        # Direction: unit-normalise each row
        direction: torch.Tensor = adapted_w / (norm + 1e-8)             # (out, in)

        # DoRA final weight  W' = m · direction
        dora_w: torch.Tensor = (                                        # (out, in)
            self.magnitude.unsqueeze(1) * direction                     # (out,1)*(out,in)
        )

        return F.linear(x, dora_w, self.bias)                          # (*, out)

    # ------------------------------------------------------------------ #
    #  Merge / Unmerge  (inference-time shortcut)                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def merge(self) -> None:
        """Bake LoRA + magnitude into self.weight for fast inference."""
        if self.merged:
            return
        self._original_weight = self.weight.data.clone()                # stash W₀

        delta_w: torch.Tensor = self.lora_B @ self.lora_A
        adapted_w: torch.Tensor = self.weight + self.scaling * delta_w
        norm: torch.Tensor = torch.norm(adapted_w, p=2, dim=1, keepdim=True)
        direction: torch.Tensor = adapted_w / (norm + 1e-8)

        self.weight.data = (self.magnitude.unsqueeze(1) * direction).detach()
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Restore the original frozen pretrained weight."""
        if not self.merged:
            return
        assert self._original_weight is not None
        self.weight.data.copy_(self._original_weight)
        self._original_weight = None
        self.merged = False

    # ------------------------------------------------------------------ #
    #  Repr                                                                #
    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, scaling={self.scaling:.4f}, merged={self.merged}"
        )


# ====================================================================== #
#  Quick smoke-test / study drill verification                            #
# ====================================================================== #
def _test_dora() -> None:
    torch.manual_seed(0)

    # ── Build a pretrained linear and wrap it ──────────────────────────
    pretrained = nn.Linear(in_features=16, out_features=8, bias=True)
    dora = DoRALinear(pretrained, r=4, lora_alpha=8.0, lora_dropout=0.1)
    dora.train()

    x: torch.Tensor = torch.randn(3, 16)                               # (batch, in)

    # 1.  At init the outputs must be identical (ΔW = 0).
    out_pretrained: torch.Tensor = pretrained(x)                        # (3, 8)
    out_dora_init: torch.Tensor = dora(x)                               # (3, 8)
    assert torch.allclose(out_pretrained, out_dora_init, atol=1e-5), (
        "Init mismatch: DoRA should equal pretrained at init"
    )
    print("✓ Test 1 passed — output matches pretrained at init")

    # 2.  Gradients flow through lora_A, lora_B, magnitude; NOT through weight.
    loss: torch.Tensor = out_dora_init.sum()
    loss.backward()
    assert dora.lora_A.grad is not None, "lora_A has no grad"
    assert dora.lora_B.grad is not None, "lora_B has no grad"
    assert dora.magnitude.grad is not None, "magnitude has no grad"
    assert dora.weight.grad is None, "frozen weight should have no grad"
    print("✓ Test 2 passed — grad flows to trainable params only")

    # 3.  One optimiser step changes the output.
    opt = torch.optim.Adam(
        [dora.lora_A, dora.lora_B, dora.magnitude], lr=1e-2
    )
    target = torch.randn(3, 8)
    for _ in range(50):
        opt.zero_grad()
        out = dora(x)
        mse = F.mse_loss(out, target)
        mse.backward()
        opt.step()

    out_after: torch.Tensor = dora(x)
    assert not torch.allclose(out_pretrained, out_after, atol=1e-4), (
        "Training had no effect"
    )
    print("✓ Test 3 passed — optimiser step updates output")

    # 4.  Merge / unmerge round-trip.
    out_unmerged: torch.Tensor = dora(x).detach()
    dora.merge()
    out_merged: torch.Tensor = dora(x).detach()
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5), (
        "Merged output diverges"
    )
    dora.unmerge()
    out_unmerged2: torch.Tensor = dora(x).detach()
    assert torch.allclose(out_unmerged, out_unmerged2, atol=1e-5), (
        "Unmerge did not restore correctly"
    )
    print("✓ Test 4 passed — merge/unmerge round-trip OK")

    # 5.  Parameter count sanity.
    total_params = sum(p.numel() for p in dora.parameters())
    trainable_params = sum(p.numel() for p in dora.parameters() if p.requires_grad)
    print(f"    Total params     : {total_params}")
    print(f"    Trainable params : {trainable_params}  "
          f"(= r·in + r·out + out = {4*16 + 4*8 + 8})")
    expected_trainable = 4 * 16 + 4 * 8 + 8  # lora_A + lora_B + magnitude
    assert trainable_params == expected_trainable, (
        f"Expected {expected_trainable} trainable, got {trainable_params}"
    )
    print("✓ Test 5 passed — trainable param count correct")

    print("\nAll tests passed ✔")


if __name__ == "__main__":
    _test_dora()
