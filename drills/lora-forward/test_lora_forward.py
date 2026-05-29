import torch
import torch.nn.functional as F
import pytest
from from_scratch import LoRALinear


def test_output_shape():
    """Check that output shape matches (batch_size, out_features)."""
    batch, in_feat, out_feat = 4, 16, 10
    layer = LoRALinear(in_feat, out_feat, rank=4, alpha=8.0)
    x = torch.randn(batch, in_feat)
    y = layer(x)
    assert y.shape == (batch, out_feat)


def test_correctness_vs_reference():
    """Verify forward pass matches analytic ground truth."""
    torch.manual_seed(0)
    batch, in_feat, out_feat = 4, 16, 10
    rank, alpha = 4, 8.0
    layer = LoRALinear(in_feat, out_feat, rank=rank, alpha=alpha)

    # Set weights to known values
    torch.nn.init.normal_(layer.weight)
    torch.nn.init.normal_(layer.bias)

    x = torch.randn(batch, in_feat, requires_grad=True)
    y_pred = layer(x)

    # Compute reference output
    scaling = alpha / rank
    delta_w = scaling * (layer.lora_B @ layer.lora_A)
    y_ref = F.linear(x, layer.weight + delta_w, layer.bias)

    assert torch.allclose(y_pred, y_ref, atol=1e-5)


def test_finite_gradients():
    """Check that gradients are finite after backward pass."""
    layer = LoRALinear(8, 6, rank=3, alpha=6.0)
    x = torch.randn(2, 8, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert torch.all(torch.isfinite(layer.lora_A.grad))
    assert torch.all(torch.isfinite(layer.lora_B.grad))
    assert torch.all(torch.isfinite(x.grad))


def test_merge_unmerge():
    """Check merge/unmerge toggle works correctly."""
    layer = LoRALinear(8, 6, rank=3, alpha=6.0)
    x = torch.randn(2, 8)

    # Pre-merge forward
    y1 = layer(x)
    layer.merge_weights()
    y2 = layer(x)
    layer.unmerge_weights()
    y3 = layer(x)

    # Merged forward should match pre-merge
    assert torch.allclose(y1, y2, atol=1e-5)
    # Unmerged forward should also match
    assert torch.allclose(y1, y3, atol=1e-5)

    # Verify merged flag
    assert not layer.merged
    layer.merge_weights()
    assert layer.merged
    layer.unmerge_weights()
    assert not layer.merged

    # Should raise errors when calling merge/unmerge incorrectly
    with pytest.raises(RuntimeError):
        layer.unmerge_weights()
    layer.merge_weights()
    with pytest.raises(RuntimeError):
        layer.merge_weights()


def test_scaling_factor():
    """Ensure scaling factor alpha/r is applied correctly."""
    layer = LoRALinear(8, 6, rank=2, alpha=4.0)
    assert layer.scaling == 2.0  # alpha/r = 4/2 = 2


def test_parameters_require_grad():
    """LoRA parameters should be trainable, weight/bias frozen."""
    layer = LoRALinear(8, 6, rank=3, alpha=6.0, bias=True)
    assert not layer.weight.requires_grad
    assert not layer.bias.requires_grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


if __name__ == "__main__":
    test_output_shape()
    test_correctness_vs_reference()
    test_finite_gradients()
    test_merge_unmerge()
    test_scaling_factor()
    test_parameters_require_grad()
    print("All tests passed!")
