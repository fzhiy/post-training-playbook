import torch
import torch.nn as nn
import numpy as np
import sys
from from_scratch import (
    CausalMultiHeadAttention,
    TransformerDecoderBlock,
    MiniGPT,
    generate,
    _sample,
)

torch.manual_seed(42)
np.random.seed(42)

# ==================== Helper functions ====================
def check_close(a, b, atol=1e-5):
    """Check two tensors are close with given absolute tolerance."""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert torch.allclose(a, b, atol=atol), f"Max diff: {(a - b).abs().max().item():.2e}"
    assert torch.isfinite(a).all(), "Tensor contains non-finite values"
    assert torch.isfinite(b).all(), "Tensor contains non-finite values"

def check_shapes(tensor, expected_shape):
    """Check tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"

def check_finite_grad(model, input_ids, target_ids):
    """Verify all gradients are finite."""
    model.zero_grad()
    logits, _ = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

# ==================== Test 1: CausalMultiHeadAttention ====================
def test_causal_attention():
    print("Testing CausalMultiHeadAttention...")
    B, T, d_model, n_heads = 2, 5, 16, 4
    head_dim = d_model // n_heads
    
    attn = CausalMultiHeadAttention(d_model, n_heads)
    x = torch.randn(B, T, d_model)
    
    # Test forward without cache (prefill)
    output1, cache1 = attn(x)
    check_shapes(output1, (B, T, d_model))
    check_shapes(cache1[0], (B, n_heads, T, head_dim))
    check_shapes(cache1[1], (B, n_heads, T, head_dim))
    
    # Test forward with cache (decode step)
    x_step = torch.randn(B, 1, d_model)
    output2, cache2 = attn(x_step, kv_cache=cache1)
    check_shapes(output2, (B, 1, d_model))
    check_shapes(cache2[0], (B, n_heads, T + 1, head_dim))
    check_shapes(cache2[1], (B, n_heads, T + 1, head_dim))
    
    # Test that outputs are different (different random inputs)
    assert not torch.allclose(output1[:, -1, :], output2[:, 0, :]), "Output should differ"
    
    # Test causal masking correctness with simple analytic case
    # Create simple attention where we can verify masking manually
    attn_simple = CausalMultiHeadAttention(4, 2)
    # Set weights to identity-like for easy verification
    with torch.no_grad():
        # Force UNIFORM causal attention: zero Q and K so every score is equal,
        # making the causal softmax a plain average over positions 0..i. Identity
        # V and O then make each output the running mean of the one-hot inputs,
        # which is analytically checkable below.
        attn_simple.W_q.weight.zero_()
        attn_simple.W_k.weight.zero_()
        attn_simple.W_v.weight.copy_(torch.eye(4))
        attn_simple.W_o.weight.copy_(torch.eye(4))
        for proj in [attn_simple.W_q, attn_simple.W_k, attn_simple.W_v, attn_simple.W_o]:
            if proj.bias is not None:
                proj.bias.zero_()
    
    # Create input where position 0 = [1,0,0,0], position 1 = [0,1,0,0], etc.
    x_simple = torch.eye(4).unsqueeze(0)  # (1, 4, 4)
    output_simple, _ = attn_simple(x_simple)
    
    # With causal masking, output at position i should only depend on positions <= i
    # For position 0: only attends to itself -> output = [1,0,0,0]
    assert torch.allclose(output_simple[0, 0, :], torch.tensor([1., 0., 0., 0.]), atol=1e-6)
    # For position 1: attends to positions 0 and 1
    # Expected: average of [1,0,0,0] and [0,1,0,0] = [0.5, 0.5, 0, 0]
    expected_pos1 = torch.tensor([0.5, 0.5, 0., 0.])
    assert torch.allclose(output_simple[0, 1, :], expected_pos1, atol=1e-6), \
        f"Position 1 expected {expected_pos1}, got {output_simple[0, 1, :]}"
    
    print("✓ CausalMultiHeadAttention passed")

# ==================== Test 2: TransformerDecoderBlock ====================
def test_decoder_block():
    print("Testing TransformerDecoderBlock...")
    B, T, d_model, n_heads, d_ff = 2, 5, 16, 4, 32
    
    block = TransformerDecoderBlock(d_model, n_heads, d_ff)
    x = torch.randn(B, T, d_model)
    
    # Test forward without cache
    output1, cache1 = block(x)
    check_shapes(output1, (B, T, d_model))
    
    # Test forward with cache
    x_step = torch.randn(B, 1, d_model)
    output2, cache2 = block(x_step, kv_cache=cache1)
    check_shapes(output2, (B, 1, d_model))
    
    # Test that outputs are from different inputs
    assert not torch.allclose(output1[:, -1, :], output2[:, 0, :]), "Output should differ"
    
    # Test gradient flow
    block.zero_grad()
    loss = output1.sum()
    loss.backward()
    for param in block.parameters():
        assert param.grad is not None, "Gradient should exist"
        assert torch.isfinite(param.grad).all(), "Gradient should be finite"
    
    print("✓ TransformerDecoderBlock passed")

# ==================== Test 3: MiniGPT ====================
def test_mini_gpt():
    print("Testing MiniGPT...")
    vocab_size, d_model, n_heads, d_ff, n_layers = 100, 32, 4, 64, 2
    
    model = MiniGPT(vocab_size, d_model, n_heads, d_ff, n_layers)
    model.eval()
    
    B, T = 2, 8
    input_ids = torch.randint(0, vocab_size, (B, T))
    
    # Test forward without cache (prefill)
    logits1, caches1 = model(input_ids)
    check_shapes(logits1, (B, T, vocab_size))
    assert len(caches1) == n_layers
    for cache in caches1:
        check_shapes(cache[0], (B, n_heads, T, d_model // n_heads))
        check_shapes(cache[1], (B, n_heads, T, d_model // n_heads))
    
    # Test forward with cache (decode step)
    x_step = torch.randint(0, vocab_size, (B, 1))
    logits2, caches2 = model(x_step, kv_caches=caches1, position_offset=T)
    check_shapes(logits2, (B, 1, vocab_size))
    
    # Test that logits for the first T positions from full forward match cache forward
    # Run full sequence through model without cache
    full_input = torch.cat([input_ids, x_step], dim=1)
    logits_full, _ = model(full_input)
    
    # Compare the last position logits
    check_close(logits_full[:, -1, :], logits2[:, 0, :], atol=1e-5)
    
    # Test gradient flow
    model.zero_grad()
    target = torch.randint(0, vocab_size, (B, T))
    loss = nn.CrossEntropyLoss()(logits1.view(-1, vocab_size), target.view(-1))
    loss.backward()
    check_finite_grad(model, input_ids, target)
    
    print("✓ MiniGPT passed")

# ==================== Test 4: Generate function ====================
def test_generate():
    print("Testing generate...")
    vocab_size, d_model, n_heads, d_ff, n_layers = 50, 16, 2, 32, 2
    
    model = MiniGPT(vocab_size, d_model, n_heads, d_ff, n_layers)
    model.eval()
    
    B, prompt_len = 1, 5
    input_ids = torch.randint(0, vocab_size, (B, prompt_len))
    
    # Test greedy generation (temperature=0)
    torch.manual_seed(42)
    output_greedy = generate(model, input_ids, max_new_tokens=10, temperature=0.0)
    check_shapes(output_greedy, (B, prompt_len + 10))
    assert torch.isfinite(output_greedy).all(), "Output should be finite"
    
    # Test that generated tokens are in valid range
    assert (output_greedy >= 0).all() and (output_greedy < vocab_size).all(), \
        "Tokens should be in valid range"
    
    # Test deterministic behavior with same seed
    torch.manual_seed(123)
    output1 = generate(model, input_ids, max_new_tokens=5, temperature=0.0)
    torch.manual_seed(123)
    output2 = generate(model, input_ids, max_new_tokens=5, temperature=0.0)
    assert torch.equal(output1, output2), "Greedy generation should be deterministic"
    
    # Test sampling generation (temperature>0)
    torch.manual_seed(42)
    output_sample = generate(model, input_ids, max_new_tokens=10, temperature=1.0, top_k=5)
    check_shapes(output_sample, (B, prompt_len + 10))
    assert torch.isfinite(output_sample).all(), "Output should be finite"
    assert (output_sample >= 0).all() and (output_sample < vocab_size).all(), \
        "Tokens should be in valid range"
    
    # Test with EOS token
    torch.manual_seed(42)
    output_eos = generate(model, input_ids, max_new_tokens=20, temperature=0.0, eos_token_id=0)
    check_shapes(output_eos, (B, prompt_len + min(20, 20)))  # May stop early
    assert torch.isfinite(output_eos).all(), "Output should be finite"
    
    print("✓ Generate passed")

# ==================== Test 5: Reference comparison ====================
def test_reference_comparison():
    print("Testing against PyTorch reference implementation...")
    
    # Create simple reference attention implementation for comparison
    class ReferenceAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        def forward(self, x):
            B, T, _ = x.shape
            q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Full causal attention without cache
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))
            weights = torch.softmax(scores, dim=-1)
            attn = torch.matmul(weights, v)
            attn = attn.transpose(1, 2).contiguous().view(B, T, self.d_model)
            return self.W_o(attn)
    
    B, T, d_model, n_heads = 2, 6, 16, 4
    
    # Create both implementations with same weights
    ref_attn = ReferenceAttention(d_model, n_heads)
    our_attn = CausalMultiHeadAttention(d_model, n_heads)
    
    # Copy weights to ensure same parameters
    with torch.no_grad():
        our_attn.W_q.weight.copy_(ref_attn.W_q.weight)
        our_attn.W_k.weight.copy_(ref_attn.W_k.weight)
        our_attn.W_v.weight.copy_(ref_attn.W_v.weight)
        our_attn.W_o.weight.copy_(ref_attn.W_o.weight)
    
    x = torch.randn(B, T, d_model)
    
    # Get outputs from both
    ref_output = ref_attn(x)
    our_output, _ = our_attn(x)
    
    # They should be very close (small numerical differences allowed)
    check_close(ref_output, our_output, atol=1e-6)
    
    # Test that caching gives same result as without cache
    our_output_full, cache = our_attn(x)
    
    # Now process in two chunks with cache
    chunk1 = x[:, :3, :]
    chunk2 = x[:, 3:, :]
    
    output1, cache1 = our_attn(chunk1)
    output2, cache2 = our_attn(chunk2, kv_cache=cache1)
    
    # Combine outputs
    our_output_chunked = torch.cat([output1, output2], dim=1)
    
    # Should match full forward pass
    check_close(our_output_full, our_output_chunked, atol=1e-6)
    
    print("✓ Reference comparison passed")

# ==================== Test 6: Edge cases ====================
def test_edge_cases():
    print("Testing edge cases...")
    
    # Test with sequence length 1
    attn = CausalMultiHeadAttention(8, 2)
    x = torch.randn(1, 1, 8)
    output, cache = attn(x)
    check_shapes(output, (1, 1, 8))
    
    # Test with large batch
    x_large = torch.randn(10, 5, 8)
    output_large, _ = attn(x_large)
    check_shapes(output_large, (10, 5, 8))
    
    # Test MiniGPT with single token input
    model = MiniGPT(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    input_ids = torch.randint(0, 10, (1, 1))
    logits, caches = model(input_ids)
    check_shapes(logits, (1, 1, 10))
    
    # Test generation with max_new_tokens=0
    output = generate(model, input_ids, max_new_tokens=0)
    check_shapes(output, (1, 1))
    assert torch.equal(output, input_ids), "Should return original input"
    
    # Test generation with very small temperature (greedy-like)
    torch.manual_seed(42)
    output = generate(model, input_ids, max_new_tokens=3, temperature=0.001)
    check_shapes(output, (1, 4))
    assert torch.isfinite(output).all(), "Output should be finite"
    
    print("✓ Edge cases passed")

# ==================== Run all tests ====================
if __name__ == "__main__":
    print("Running tests for from_scratch module...")
    
    try:
        test_causal_attention()
        test_decoder_block()
        test_mini_gpt()
        test_generate()
        test_reference_comparison()
        test_edge_cases()
        
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
