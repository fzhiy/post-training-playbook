import torch
import torch.nn.functional as F
import pytest
from from_scratch import SwiGLU

# Reference implementation for comparison
def reference_swiglu_forward(x, W1, W3, W2, b1, b3, b2):
    """Reference SwiGLU forward pass using PyTorch builtins."""
    gate = F.silu(F.linear(x, W1, b1))   # Swish is SiLU
    value = F.linear(x, W3, b3)
    gated = gate * value
    out = F.linear(gated, W2, b2)
    return out

def test_swish_function():
    """Test the custom swish implementation against F.silu."""
    for shape in [(1,), (3, 4), (2, 5, 6)]:
        x = torch.randn(shape, dtype=torch.float32)
        out_custom = SwiGLU.swish(x)
        out_ref = F.silu(x)
        assert torch.allclose(out_custom, out_ref, atol=1e-7), \
            f"Swish mismatch for shape {shape}"

def test_output_shape():
    """Verify output shape matches input shape for various configurations."""
    configs = [
        (512, None, 0.0, True),     # default d_ff
        (256, 1024, 0.1, False),    # explicit d_ff, no bias
        (128, 512, 0.0, True),
        (1024, 4096, 0.2, False),
    ]
    batch_size, seq_len = 2, 10

    for d_model, d_ff, dropout, bias in configs:
        model = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)
        x = torch.randn(batch_size, seq_len, d_model)
        y = model(x)
        assert y.shape == x.shape, \
            f"Shape mismatch for config (d_model={d_model}, d_ff={d_ff})"

def test_numerical_correctness():
    """Compare from_scratch output with reference implementation."""
    torch.manual_seed(42)
    d_model, d_ff = 32, 128
    batch_size, seq_len = 2, 5

    model = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=0.0, bias=True)

    # Extract weights and biases
    W1 = model.W1.weight.data
    W3 = model.W3.weight.data
    W2 = model.W2.weight.data
    b1 = model.W1.bias.data
    b3 = model.W3.bias.data
    b2 = model.W2.bias.data

    # Test with multiple inputs
    for _ in range(3):
        x = torch.randn(batch_size, seq_len, d_model)
        out_scratch = model(x)
        out_ref = reference_swiglu_forward(x, W1, W3, W2, b1, b3, b2)
        assert torch.allclose(out_scratch, out_ref, atol=1e-5), \
            f"Numerical mismatch at input shape {x.shape}"

def test_default_d_ff_calculation():
    """Verify the default d_ff rounding to nearest multiple of 256."""
    test_cases = [
        (512, 1365, 1536),   # 8/3 * 512 ≈ 1365.33 → round up to 1536
        (256, 682, 768),     # 8/3 * 256 ≈ 682.67  → round up to 768
        (1024, 2730, 2816),  # 8/3 * 1024 ≈ 2730.67 → round up to 2816
        (100, 266, 512),     # 8/3 * 100 ≈ 266.67  → round up to 512
    ]
    for d_model, _, expected_d_ff in test_cases:
        model = SwiGLU(d_model=d_model)  # Use default d_ff
        assert model.d_ff == expected_d_ff, \
            f"Default d_ff calculation wrong for d_model={d_model}"

def test_gradient_finiteness():
    """Ensure gradients are finite after backward pass."""
    d_model, d_ff = 64, 256
    model = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=0.1, bias=True)
    x = torch.randn(2, 3, d_model, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check gradients of all parameters
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite gradient in parameter {name}"

    # Check gradient of input
    assert torch.isfinite(x.grad).all(), "Non-finite gradient in input"

def test_dropout_behavior():
    """Test that dropout is applied (stochastic difference)."""
    torch.manual_seed(123)
    d_model, d_ff = 16, 32
    model = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=0.9, bias=True)
    model.train()  # Ensure dropout is active

    x = torch.randn(2, 4, d_model)
    y1 = model(x)
    y2 = model(x)  # Same input, different dropout mask
    # With high dropout, outputs should almost surely differ
    assert not torch.allclose(y1, y2, atol=1e-5), \
        "Dropout seems inactive (outputs identical in train mode)"

def test_eval_mode():
    """Ensure dropout is disabled in eval mode."""
    torch.manual_seed(123)
    d_model, d_ff = 16, 32
    model = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=0.9, bias=True)
    model.eval()  # Disable dropout

    x = torch.randn(2, 4, d_model)
    y1 = model(x)
    y2 = model(x)  # Same input, should be identical in eval mode
    assert torch.allclose(y1, y2, atol=1e-7), \
        "Outputs differ in eval mode (dropout should be off)"

def test_different_dtypes():
    """Test that model works with float32 (required for training)."""
    d_model, d_ff = 16, 32
    model = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=0.0, bias=True)
    x = torch.randn(1, 2, d_model, dtype=torch.float32)
    y = model(x)
    assert y.dtype == torch.float32, "Wrong dtype for float32 input"

if __name__ == "__main__":
    test_swish_function()
    test_output_shape()
    test_numerical_correctness()
    test_default_d_ff_calculation()
    test_gradient_finiteness()
    test_dropout_behavior()
    test_eval_mode()
    test_different_dtypes()
    print("All tests passed!")
