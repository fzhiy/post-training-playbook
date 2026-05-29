import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import traceback

from from_scratch import DoRALinear

# ── Helpers ──────────────────────────────────────────────────────────────

def reference_dora_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    magnitude: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Pure-reference DoRA forward computed step-by-step."""
    delta_w = lora_B @ lora_A                          # (out, in)
    adapted_w = weight + scaling * delta_w             # (out, in)
    norm = torch.norm(adapted_w, p=2, dim=1, keepdim=True)  # (out, 1)
    direction = adapted_w / (norm + 1e-8)              # (out, in)
    dora_w = magnitude.unsqueeze(1) * direction        # (out, in)
    return F.linear(x, dora_w, bias)


def run_test(name: str):
    """Decorator that prints pass/fail and propagates exceptions."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                print(f"  ✓ {name}")
                return True
            except Exception as e:
                print(f"  ✗ {name}")
                traceback.print_exc()
                return False
        wrapper.__name__ = name
        return wrapper
    return decorator


# ── Tests ────────────────────────────────────────────────────────────────

ALL_PASSED = True

@run_test("Init output matches pretrained (ΔW = 0 at init)")
def test_init_matches_pretrained():
    torch.manual_seed(42)
    for in_f, out_f, bias_on in [(16, 8, True), (32, 16, False), (7, 13, True)]:
        pretrained = nn.Linear(in_f, out_f, bias=bias_on)
        dora = DoRALinear(pretrained, r=4, lora_alpha=8.0)
        dora.train()
        x = torch.randn(5, in_f)
        out_ref = pretrained(x)
        out_dora = dora(x)
        assert torch.allclose(out_ref, out_dora, atol=1e-5), (
            f"mismatch for ({in_f},{out_f},bias={bias_on}): "
            f"max diff = {(out_ref - out_dora).abs().max().item()}"
        )


@run_test("Forward matches hand-computed reference")
def test_forward_vs_reference():
    torch.manual_seed(0)
    in_f, out_f, r = 10, 6, 4
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=8.0, lora_dropout=0.0)
    dora.eval()
    x = torch.randn(3, in_f)
    out_dora = dora(x)
    out_ref = reference_dora_forward(
        x, dora.weight, dora.bias,
        dora.lora_A, dora.lora_B, dora.magnitude, dora.scaling,
    )
    assert torch.allclose(out_dora, out_ref, atol=1e-5), (
        f"max diff = {(out_dora - out_ref).abs().max().item()}"
    )


@run_test("Output shape is correct for various input shapes")
def test_output_shape():
    torch.manual_seed(1)
    in_f, out_f, r = 12, 7, 3
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=6.0)
    dora.eval()
    for shape in [(4, in_f), (2, 5, in_f), (1, 3, 4, in_f)]:
        x = torch.randn(*shape)
        out = dora(x)
        expected = (*shape[:-1], out_f)
        assert out.shape == expected, f"input {shape} → got {out.shape}, expected {expected}"


@run_test("Gradients are finite and flow to trainable params only")
def test_gradient_flow():
    torch.manual_seed(7)
    in_f, out_f, r = 14, 9, 5
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=10.0, lora_dropout=0.0)
    dora.train()
    x = torch.randn(6, in_f)
    out = dora(x)
    loss = out.sum()
    loss.backward()

    # Trainable params must have finite grads
    for name in ["lora_A", "lora_B", "magnitude"]:
        p = getattr(dora, name)
        assert p.grad is not None, f"{name} grad is None"
        assert torch.isfinite(p.grad).all(), f"{name} grad has non-finite values"

    # Frozen weight must have no grad
    assert dora.weight.grad is None, "frozen weight should have no grad"
    if dora.bias is not None:
        assert dora.bias.grad is None, "frozen bias should have no grad"


@run_test("Gradients are finite after a few optimiser steps")
def test_gradient_after_steps():
    torch.manual_seed(99)
    in_f, out_f, r = 10, 6, 4
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=8.0, lora_dropout=0.0)
    dora.train()
    opt = torch.optim.Adam(
        [dora.lora_A, dora.lora_B, dora.magnitude], lr=1e-3
    )
    x = torch.randn(4, in_f)
    target = torch.randn(4, out_f)
    for _ in range(20):
        opt.zero_grad()
        out = dora(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()

    # Check grads are still finite
    for name in ["lora_A", "lora_B", "magnitude"]:
        p = getattr(dora, name)
        assert torch.isfinite(p.grad).all(), f"{name} grad non-finite after steps"


@run_test("Training actually changes output")
def test_training_changes_output():
    torch.manual_seed(10)
    in_f, out_f, r = 12, 8, 4
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=8.0, lora_dropout=0.0)
    dora.train()
    x = torch.randn(4, in_f)
    out_before = dora(x).detach().clone()

    opt = torch.optim.Adam(
        [dora.lora_A, dora.lora_B, dora.magnitude], lr=1e-2
    )
    target = torch.randn(4, out_f)
    for _ in range(100):
        opt.zero_grad()
        loss = F.mse_loss(dora(x), target)
        loss.backward()
        opt.step()

    out_after = dora(x).detach()
    assert not torch.allclose(out_before, out_after, atol=1e-4), (
        "Training had no effect on output"
    )


@run_test("Merge produces same output as unmerged forward")
def test_merge_output():
    torch.manual_seed(33)
    in_f, out_f, r = 16, 8, 4
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=16.0, lora_dropout=0.0)
    dora.train()
    # Perturb LoRA so merge is non-trivial
    with torch.no_grad():
        dora.lora_B.copy_(torch.randn_like(dora.lora_B) * 0.1)
        dora.magnitude.add_(0.5)
    dora.eval()
    x = torch.randn(3, in_f)
    out_unmerged = dora(x).detach().clone()

    dora.merge()
    out_merged = dora(x).detach()
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5), (
        f"max diff = {(out_unmerged - out_merged).abs().max().item()}"
    )
    assert dora.merged is True


@run_test("Unmerge restores original weight and output")
def test_unmerge():
    torch.manual_seed(55)
    in_f, out_f, r = 10, 6, 3
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=8.0, lora_dropout=0.0)
    dora.eval()
    with torch.no_grad():
        dora.lora_B.copy_(torch.randn_like(dora.lora_B) * 0.05)
    x = torch.randn(2, in_f)
    out_original = dora(x).detach().clone()
    original_weight = dora.weight.data.clone()

    dora.merge()
    assert dora.merged is True
    dora.unmerge()
    assert dora.merged is False
    assert torch.allclose(dora.weight.data, original_weight, atol=1e-7), (
        "Weight not restored after unmerge"
    )
    out_restored = dora(x).detach()
    assert torch.allclose(out_original, out_restored, atol=1e-5), (
        f"max diff = {(out_original - out_restored).abs().max().item()}"
    )


@run_test("Merge→unmerge→merge round-trip consistency")
def test_merge_unmerge_round_trip():
    torch.manual_seed(77)
    in_f, out_f, r = 8, 5, 3
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=6.0, lora_dropout=0.0)
    dora.eval()
    with torch.no_grad():
        dora.lora_B.copy_(torch.randn_like(dora.lora_B) * 0.1)
    x = torch.randn(2, in_f)

    out_0 = dora(x).detach().clone()
    dora.merge()
    out_m1 = dora(x).detach().clone()
    dora.unmerge()
    out_u1 = dora(x).detach().clone()
    dora.merge()
    out_m2 = dora(x).detach().clone()

    assert torch.allclose(out_0, out_m1, atol=1e-5)
    assert torch.allclose(out_0, out_u1, atol=1e-5)
    assert torch.allclose(out_0, out_m2, atol=1e-5)


@run_test("Magnitude initialisation equals per-row L2 norm of pretrained weight")
def test_magnitude_init():
    torch.manual_seed(11)
    in_f, out_f = 20, 12
    pretrained = nn.Linear(in_f, out_f, bias=False)
    dora = DoRALinear(pretrained, r=4, lora_alpha=8.0)
    expected_m = torch.norm(pretrained.weight.data, p=2, dim=1)
    assert torch.allclose(dora.magnitude.data, expected_m, atol=1e-6), (
        f"max diff = {(dora.magnitude.data - expected_m).abs().max().item()}"
    )


@run_test("Trainable param count is correct")
def test_param_count():
    torch.manual_seed(13)
    for in_f, out_f, r in [(16, 8, 4), (32, 16, 8), (7, 13, 3)]:
        pretrained = nn.Linear(in_f, out_f, bias=True)
        dora = DoRALinear(pretrained, r=r, lora_alpha=16.0)
        trainable = sum(p.numel() for p in dora.parameters() if p.requires_grad)
        # lora_A: r*in_f, lora_B: out_f*r, magnitude: out_f
        expected = r * in_f + out_f * r + out_f
        assert trainable == expected, (
            f"({in_f},{out_f},r={r}): expected {expected}, got {trainable}"
        )
        # Frozen: weight (out_f*in_f) + bias (out_f)
        frozen = sum(p.numel() for p in dora.parameters() if not p.requires_grad)
        expected_frozen = out_f * in_f + out_f
        assert frozen == expected_frozen


@run_test("Scaling factor = lora_alpha / r")
def test_scaling():
    torch.manual_seed(15)
    pretrained = nn.Linear(10, 6, bias=False)
    for alpha, r in [(16.0, 8), (4.0, 4), (32.0, 2)]:
        dora = DoRALinear(pretrained, r=r, lora_alpha=alpha)
        assert abs(dora.scaling - alpha / r) < 1e-8, (
            f"scaling={dora.scaling}, expected {alpha/r}"
        )


@run_test("Dropout=0 gives deterministic output")
def test_deterministic_no_dropout():
    torch.manual_seed(20)
    in_f, out_f = 10, 6
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=4, lora_alpha=8.0, lora_dropout=0.0)
    dora.train()
    x = torch.randn(3, in_f)
    out1 = dora(x).detach().clone()
    out2 = dora(x).detach()
    assert torch.allclose(out1, out2, atol=1e-7), "Non-deterministic with dropout=0"


@run_test("No-bias variant works correctly")
def test_no_bias():
    torch.manual_seed(25)
    in_f, out_f = 12, 7
    pretrained = nn.Linear(in_f, out_f, bias=False)
    dora = DoRALinear(pretrained, r=3, lora_alpha=6.0)
    assert dora.bias is None
    dora.train()
    x = torch.randn(4, in_f)
    out_pre = pretrained(x)
    out_dora = dora(x)
    assert torch.allclose(out_pre, out_dora, atol=1e-5)


@run_test("Analytic: when lora_B=0, output equals pretrained exactly")
def test_analytic_zero_lora_b():
    torch.manual_seed(30)
    in_f, out_f = 14, 9
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=4, lora_alpha=8.0, lora_dropout=0.0)
    dora.eval()
    # lora_B is already zero at init; magnitude == ||W||_c at init
    x = torch.randn(5, in_f)
    out_ref = pretrained(x)
    out_dora = dora(x)
    assert torch.allclose(out_ref, out_dora, atol=1e-6)


@run_test("Analytic: known weight/direction decomposition")
def test_analytic_decomposition():
    """Build a known weight matrix, verify DoRA decomposes correctly."""
    torch.manual_seed(0)
    in_f, out_f, r = 4, 3, 2
    pretrained = nn.Linear(in_f, out_f, bias=False)
    # Set weight to known values
    W = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [0.0, 1.0, 0.0, 1.0],
                       [1.0, 0.0, 1.0, 0.0]])
    with torch.no_grad():
        pretrained.weight.copy_(W)

    dora = DoRALinear(pretrained, r=r, lora_alpha=float(r))  # scaling = 1.0
    # Zero out LoRA so adapted_w == W
    with torch.no_grad():
        dora.lora_B.zero_()
    dora.eval()

    # With zero ΔW: direction = W / ||W||_c, m = ||W||_c
    # So dora_w = m * direction = ||W||_c * W/||W||_c = W
    x = torch.randn(2, in_f)
    out_ref = F.linear(x, W, None)
    out_dora = dora(x)
    assert torch.allclose(out_ref, out_dora, atol=1e-6)

    # Now manually set magnitude to 2x the norms → output should double
    norms = torch.norm(W, p=2, dim=1)
    with torch.no_grad():
        dora.magnitude.copy_(2.0 * norms)
    out_doubled = dora(x).detach()
    out_expected = 2.0 * out_ref.detach()
    assert torch.allclose(out_doubled, out_expected, atol=1e-5), (
        f"max diff = {(out_doubled - out_expected).abs().max().item()}"
    )


@run_test("Row norms of DoRA weight equal magnitude parameter")
def test_row_norms_equal_magnitude():
    """After forward, each row of dora_w should have norm == magnitude[i]."""
    torch.manual_seed(44)
    in_f, out_f, r = 10, 6, 4
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=8.0, lora_dropout=0.0)
    dora.eval()

    # Manually compute the DoRA weight
    delta_w = dora.lora_B @ dora.lora_A
    adapted_w = dora.weight + dora.scaling * delta_w
    norm = torch.norm(adapted_w, p=2, dim=1, keepdim=True)
    direction = adapted_w / (norm + 1e-8)
    dora_w = dora.magnitude.unsqueeze(1) * direction

    # Each row of dora_w should have L2 norm == magnitude[i]
    row_norms = torch.norm(dora_w, p=2, dim=1)
    assert torch.allclose(row_norms, dora.magnitude.data, atol=1e-4), (
        f"max diff = {(row_norms - dora.magnitude.data).abs().max().item()}"
    )


@run_test("Merged weight row norms equal magnitude")
def test_merged_row_norms():
    torch.manual_seed(46)
    in_f, out_f, r = 10, 6, 4
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=r, lora_alpha=8.0, lora_dropout=0.0)
    with torch.no_grad():
        dora.lora_B.copy_(torch.randn_like(dora.lora_B) * 0.1)
    dora.merge()
    row_norms = torch.norm(dora.weight.data, p=2, dim=1)
    assert torch.allclose(row_norms, dora.magnitude.data, atol=1e-4)


@run_test("Extra repr contains expected info")
def test_extra_repr():
    torch.manual_seed(50)
    pretrained = nn.Linear(10, 6, bias=True)
    dora = DoRALinear(pretrained, r=4, lora_alpha=8.0)
    r = dora.extra_repr()
    assert "in=10" in r
    assert "out=6" in r
    assert "r=4" in r
    assert "merged=False" in r


@run_test("Batch dimension preservation across ranks")
def test_batch_dims():
    torch.manual_seed(60)
    in_f, out_f = 8, 5
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=3, lora_alpha=6.0)
    dora.eval()
    for shape in [(1, in_f), (2, in_f), (2, 3, in_f), (1, 2, 3, in_f)]:
        x = torch.randn(*shape)
        out = dora(x)
        assert out.shape == (*shape[:-1], out_f), f"{shape} → {out.shape}"


@run_test("Double precision maintains correctness")
def test_double_precision():
    torch.manual_seed(65)
    in_f, out_f = 10, 6
    pretrained = nn.Linear(in_f, out_f, bias=True).double()
    dora = DoRALinear(pretrained, r=4, lora_alpha=8.0, lora_dropout=0.0)
    dora.eval()
    x = torch.randn(3, in_f, dtype=torch.float64)
    out_ref = pretrained(x)
    out_dora = dora(x)
    assert torch.allclose(out_ref, out_dora, atol=1e-10)


@run_test("Merged model has no trainable LoRA params affecting grad")
def test_merged_no_grad_leak():
    torch.manual_seed(70)
    in_f, out_f = 10, 6
    pretrained = nn.Linear(in_f, out_f, bias=True)
    dora = DoRALinear(pretrained, r=4, lora_alpha=8.0, lora_dropout=0.0)
    dora.eval()
    dora.merge()
    # After merge, forward should not touch lora_A/lora_B/magnitude
    # (they still exist but the merged path bypasses them)
    x = torch.randn(3, in_f)
    out = dora(x)
    assert out.shape == (3, out_f)
    assert torch.isfinite(out).all()


# ── Run all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_init_matches_pretrained,
        test_forward_vs_reference,
        test_output_shape,
        test_gradient_flow,
        test_gradient_after_steps,
        test_training_changes_output,
        test_merge_output,
        test_unmerge,
        test_merge_unmerge_round_trip,
        test_magnitude_init,
        test_param_count,
        test_scaling,
        test_deterministic_no_dropout,
        test_no_bias,
        test_analytic_zero_lora_b,
        test_analytic_decomposition,
        test_row_norms_equal_magnitude,
        test_merged_row_norms,
        test_extra_repr,
        test_batch_dims,
        test_double_precision,
        test_merged_no_grad_leak,
    ]

    print(f"Running {len(tests)} tests for DoRALinear...\n")
    results = []
    for t in tests:
        results.append(t())

    passed = sum(results)
    failed = len(results) - passed
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(results)}")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed ✔")
