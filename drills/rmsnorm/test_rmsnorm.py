import torch
import torch.nn as nn
from from_scratch import RMSNorm


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Analytic RMSNorm reference: x / sqrt(mean(x^2) + eps) * weight."""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def _check(all_ok: bool, msg: str):
    if not all_ok:
        raise AssertionError(msg)


# ------------------------------------------------------------------ #
# 1  Output shape
# ------------------------------------------------------------------ #
def test_output_shape():
    for B, S, D in [(1, 1, 4), (2, 4, 8), (3, 10, 64), (1, 2048, 256)]:
        model = RMSNorm(hidden_dim=D)
        x = torch.randn(B, S, D)
        out = model(x)
        _check(out.shape == x.shape,
               f"Shape mismatch for (B={B},S={S},D={D}): got {out.shape}")
    print("  ✓ test_output_shape")


# ------------------------------------------------------------------ #
# 2  Forward correctness vs analytic reference
# ------------------------------------------------------------------ #
def test_forward_vs_analytic():
    torch.manual_seed(42)
    for dim in [1, 8, 32, 128, 512]:
        eps = 1e-6
        model = RMSNorm(hidden_dim=dim, eps=eps)
        x = torch.randn(2, 4, dim)
        out = model(x)
        ref = _reference_rms_norm(x, model.weight.detach(), eps)
        _check(torch.allclose(out, ref, atol=1e-5),
               f"Forward mismatch dim={dim}: max-diff {(out - ref).abs().max():.2e}")
    print("  ✓ test_forward_vs_analytic")


# ------------------------------------------------------------------ #
# 3  Forward correctness vs PyTorch nn.RMSNorm (≥ 2.4)
# ------------------------------------------------------------------ #
def test_forward_vs_pytorch_ref():
    try:
        TorchRMSNorm = torch.nn.RMSNorm
    except AttributeError:
        print("  ⚠ test_forward_vs_pytorch_ref SKIPPED (needs PyTorch ≥ 2.4)")
        return

    torch.manual_seed(123)
    for dim in [16, 64, 256]:
        eps = 1e-5
        model = RMSNorm(hidden_dim=dim, eps=eps)
        ref_model = TorchRMSNorm(dim, eps=eps)
        ref_model.weight.data.copy_(model.weight.data)

        x = torch.randn(3, 5, dim)
        out = model(x)
        ref_out = ref_model(x)
        _check(torch.allclose(out, ref_out, atol=1e-5),
               f"PyTorch ref mismatch dim={dim}: max-diff {(out - ref_out).abs().max():.2e}")
    print("  ✓ test_forward_vs_pytorch_ref")


# ------------------------------------------------------------------ #
# 4  RMS of output ≈ 1 with default γ = 1
# ------------------------------------------------------------------ #
def test_output_rms_is_one():
    torch.manual_seed(7)
    model = RMSNorm(hidden_dim=64)
    x = torch.randn(4, 16, 64)
    out = model(x)
    rms = out.detach().pow(2).mean(dim=-1).sqrt()
    _check(torch.allclose(rms, torch.ones_like(rms), atol=1e-4),
           f"RMS not ≈ 1: max-diff {(rms - 1).abs().max():.2e}")
    print("  ✓ test_output_rms_is_one")


# ------------------------------------------------------------------ #
# 5  Gradient shapes
# ------------------------------------------------------------------ #
def test_gradient_shapes():
    model = RMSNorm(hidden_dim=16)
    x = torch.randn(2, 3, 16, requires_grad=True)
    model(x).sum().backward()
    _check(x.grad.shape == x.shape,
           f"x.grad shape {x.grad.shape} != x shape {x.shape}")
    _check(model.weight.grad.shape == model.weight.shape,
           f"weight.grad shape {model.weight.grad.shape} != weight shape {model.weight.shape}")
    print("  ✓ test_gradient_shapes")


# ------------------------------------------------------------------ #
# 6  Gradients are finite
# ------------------------------------------------------------------ #
def test_gradients_finite():
    torch.manual_seed(99)
    for dim in [8, 32, 128]:
        model = RMSNorm(hidden_dim=dim)
        x = torch.randn(2, 4, dim, requires_grad=True)
        model(x).sum().backward()
        _check(torch.isfinite(x.grad).all(),
               f"x.grad non-finite for dim={dim}")
        _check(torch.isfinite(model.weight.grad).all(),
               f"weight.grad non-finite for dim={dim}")
    print("  ✓ test_gradients_finite")


# ------------------------------------------------------------------ #
# 7  Gradients match analytic reference (x and γ)
# ------------------------------------------------------------------ #
def test_gradients_vs_analytic():
    torch.manual_seed(42)
    eps = 1e-6
    dim = 16

    # --- model under test ---
    model = RMSNorm(hidden_dim=dim, eps=eps)
    x = torch.randn(2, 4, dim, requires_grad=True)
    out = model(x)
    out.sum().backward()

    # --- analytic reference (separate graph) ---
    w_ref = model.weight.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    ref_out = _reference_rms_norm(x_ref, w_ref, eps)
    ref_out.sum().backward()

    _check(torch.allclose(x.grad, x_ref.grad, atol=1e-5),
           f"x.grad mismatch: max-diff {(x.grad - x_ref.grad).abs().max():.2e}")
    _check(torch.allclose(model.weight.grad, w_ref.grad, atol=1e-5),
           "weight.grad mismatch with reference")

    # Verify weight grad via finite differences
    delta = 1e-4
    model2 = RMSNorm(hidden_dim=dim, eps=eps)
    model2.weight.data.copy_(model.weight.detach())
    x_det = x.detach().clone()
    grad_numerical = torch.zeros(dim)
    for i in range(dim):
        w_plus = model.weight.detach().clone()
        w_plus[i] += delta
        out_plus = _reference_rms_norm(x_det, w_plus, eps).sum()
        w_minus = model.weight.detach().clone()
        w_minus[i] -= delta
        out_minus = _reference_rms_norm(x_det, w_minus, eps).sum()
        grad_numerical[i] = (out_plus - out_minus) / (2 * delta)
    _check(torch.allclose(model.weight.grad, grad_numerical, atol=1e-3),
           f"Numerical grad mismatch: max-diff {(model.weight.grad - grad_numerical).abs().max():.2e}")
    print("  ✓ test_gradients_vs_analytic")


# ------------------------------------------------------------------ #
# 8  Zero input → zero output (eps prevents div-by-zero)
# ------------------------------------------------------------------ #
def test_zero_input():
    model = RMSNorm(hidden_dim=8)
    x = torch.zeros(1, 1, 8)
    out = model(x)
    _check(torch.isfinite(out).all(), "Non-finite output for zero input")
    _check(torch.allclose(out, torch.zeros_like(out), atol=1e-6),
           "Zero input should give zero output")
    print("  ✓ test_zero_input")


# ------------------------------------------------------------------ #
# 9  Non-unit weight scaling
# ------------------------------------------------------------------ #
def test_custom_weight():
    torch.manual_seed(0)
    dim = 32
    model = RMSNorm(hidden_dim=dim)
    model.weight.data = torch.randn(dim)
    x = torch.randn(2, 5, dim)
    out = model(x)
    ref = _reference_rms_norm(x, model.weight.detach(), model.eps)
    _check(torch.allclose(out, ref, atol=1e-5),
           f"Custom-weight mismatch: max-diff {(out - ref).abs().max():.2e}")
    print("  ✓ test_custom_weight")


# ------------------------------------------------------------------ #
# 10  Various eps values
# ------------------------------------------------------------------ #
def test_various_eps():
    torch.manual_seed(0)
    for eps in [1e-4, 1e-5, 1e-6, 1e-8]:
        model = RMSNorm(hidden_dim=32, eps=eps)
        x = torch.randn(2, 4, 32)
        out = model(x)
        ref = _reference_rms_norm(x, model.weight.detach(), eps)
        _check(torch.allclose(out, ref, atol=1e-5),
               f"Mismatch at eps={eps}")
    print("  ✓ test_various_eps")


# ------------------------------------------------------------------ #
# 11  Numerical stability: large and small inputs
# ------------------------------------------------------------------ #
def test_numerical_stability():
    torch.manual_seed(0)
    model = RMSNorm(hidden_dim=32)
    for scale in [1e-6, 1.0, 1e3, 1e6]:
        x = torch.randn(2, 4, 32) * scale
        out = model(x)
        _check(torch.isfinite(out).all(),
               f"Non-finite output at scale={scale}")
        ref = _reference_rms_norm(x, model.weight.detach(), model.eps)
        _check(torch.allclose(out, ref, atol=1e-5),
               f"Mismatch at scale={scale}: max-diff {(out - ref).abs().max():.2e}")
    print("  ✓ test_numerical_stability")


# ------------------------------------------------------------------ #
# 12  Half-precision (float16) basic sanity
# ------------------------------------------------------------------ #
def test_half_precision():
    torch.manual_seed(0)
    model = RMSNorm(hidden_dim=32).half()
    x = torch.randn(2, 4, 32, dtype=torch.float16)
    out = model(x)
    _check(out.dtype == torch.float16, "Dtype not preserved")
    _check(out.shape == x.shape, "Shape not preserved")
    _check(torch.isfinite(out).all(), "Non-finite output in fp16")
    print("  ✓ test_half_precision")


# ------------------------------------------------------------------ #
# 13  End-to-end: loss → backward → grad not zero
# ------------------------------------------------------------------ #
def test_end_to_end_learning():
    torch.manual_seed(0)
    dim = 16
    model = RMSNorm(hidden_dim=dim)
    target = torch.randn(2, 4, dim)

    out = model(torch.randn(2, 4, dim, requires_grad=True))
    loss = (out - target).pow(2).mean()
    loss.backward()

    _check(model.weight.grad is not None, "weight.grad is None")
    _check(model.weight.grad.abs().sum() > 0,
           "weight.grad is all zeros — gradient not flowing")
    _check(torch.isfinite(model.weight.grad).all(),
           "weight.grad contains non-finite values")
    print("  ✓ test_end_to_end_learning")


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Running RMSNorm tests …\n")
    test_output_shape()
    test_forward_vs_analytic()
    test_forward_vs_pytorch_ref()
    test_output_rms_is_one()
    test_gradient_shapes()
    test_gradients_finite()
    test_gradients_vs_analytic()
    test_zero_input()
    test_custom_weight()
    test_various_eps()
    test_numerical_stability()
    test_half_precision()
    test_end_to_end_learning()
    print("\n✅ All tests passed.")
