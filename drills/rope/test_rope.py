"""
Comprehensive test suite for from_scratch.py (Rotary Position Embedding).

Run:  python test_rope.py
"""

import math
import torch
import torch.nn as nn
import pytest  # noqa – we use plain assert + manual runner so pytest is optional

from from_scratch import (
    precompute_rope_frequencies,
    rotate_half,
    apply_rope,
    RoPE,
)

# ── helpers ──────────────────────────────────────────────────────────────────

ATOL = 1e-5
FTOL = 1e-4  # slightly looser for operations involving many float ops


def _run(name: str):
    """Tiny decorator to print pass/fail for each test."""
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                fn(*a, **kw)
                print(f"  ✓  {name}")
            except Exception as e:
                print(f"  ✗  {name}: {e}")
                raise
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


# ── reference: analytic rotation via explicit matrix ─────────────────────────

def rope_rotation_matrix(m: int, d: int, base: float = 10000.0) -> torch.Tensor:
    """
    Build the d×d rotation matrix R_m for position m.
    Block-diagonal with 2×2 rotation blocks:
        R_i = [[cos θ_i, -sin θ_i],
               [sin θ_i,  cos θ_i]]
    where θ_i = m * base^{-2i/d}.
    """
    R = torch.eye(d)
    for i in range(d // 2):
        theta = m * (base ** (-2.0 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        R[2 * i, 2 * i] = c
        R[2 * i, 2 * i + 1] = -s
        R[2 * i + 1, 2 * i] = s
        R[2 * i + 1, 2 * i + 1] = c
    return R


# ═════════════════════════════════════════════════════════════════════════════
#  TEST BATTERY
# ═════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    passed = 0
    failed = 0
    all_tests = []

    # ------------------------------------------------------------------ #
    # 1. precompute_rope_frequencies — shape
    # ------------------------------------------------------------------ #
    @_run("precompute_rope_frequencies: shape")
    def test_freq_shape():
        for d in [32, 64, 128]:
            for T in [1, 128, 1024]:
                cos_t, sin_t = precompute_rope_frequencies(d, T)
                assert cos_t.shape == (T, d), f"cos shape {cos_t.shape} != ({T}, {d})"
                assert sin_t.shape == (T, d), f"sin shape {sin_t.shape} != ({T}, {d})"
    all_tests.append(test_freq_shape)

    # ------------------------------------------------------------------ #
    # 2. precompute_rope_frequencies — analytic values
    # ------------------------------------------------------------------ #
    @_run("precompute_rope_frequencies: analytic correctness")
    def test_freq_values():
        d, T, base = 16, 32, 10000.0
        cos_t, sin_t = precompute_rope_frequencies(d, T, base=base)
        for m in range(T):
            for i in range(d // 2):
                theta = m * (base ** (-2.0 * i / d))
                c_ref, s_ref = math.cos(theta), math.sin(theta)
                # Each pair is repeated twice
                assert torch.allclose(cos_t[m, 2 * i].float(), torch.tensor(c_ref), atol=ATOL)
                assert torch.allclose(cos_t[m, 2 * i + 1].float(), torch.tensor(c_ref), atol=ATOL)
                assert torch.allclose(sin_t[m, 2 * i].float(), torch.tensor(s_ref), atol=ATOL)
                assert torch.allclose(sin_t[m, 2 * i + 1].float(), torch.tensor(s_ref), atol=ATOL)
    all_tests.append(test_freq_values)

    # ------------------------------------------------------------------ #
    # 3. precompute_rope_frequencies — dtype propagation
    # ------------------------------------------------------------------ #
    @_run("precompute_rope_frequencies: dtype (float16, bfloat16)")
    def test_freq_dtype():
        for dt in [torch.float16, torch.bfloat16, torch.float32]:
            cos_t, sin_t = precompute_rope_frequencies(64, 16, dtype=dt)
            assert cos_t.dtype == dt, f"expected {dt}, got {cos_t.dtype}"
            assert sin_t.dtype == dt
    all_tests.append(test_freq_dtype)

    # ------------------------------------------------------------------ #
    # 4. precompute_rope_frequencies — head_dim odd raises
    # ------------------------------------------------------------------ #
    @_run("precompute_rope_frequencies: odd head_dim assertion")
    def test_freq_odd():
        try:
            precompute_rope_frequencies(33, 16)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass
    all_tests.append(test_freq_odd)

    # ------------------------------------------------------------------ #
    # 5. rotate_half — correctness against hand-crafted input
    # ------------------------------------------------------------------ #
    @_run("rotate_half: hand-crafted correctness")
    def test_rotate_half_manual():
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
        expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
        out = rotate_half(x)
        assert torch.allclose(out, expected, atol=ATOL), f"got {out}"
    all_tests.append(test_rotate_half_manual)

    # ------------------------------------------------------------------ #
    # 6. rotate_half — shape preservation
    # ------------------------------------------------------------------ #
    @_run("rotate_half: shape preservation for various dims")
    def test_rotate_half_shape():
        for shape in [(2, 8, 16), (1, 1, 64), (3, 10, 4, 32)]:
            x = torch.randn(*shape)
            out = rotate_half(x)
            assert out.shape == shape, f"shape {out.shape} != {shape}"
    all_tests.append(test_rotate_half_shape)

    # ------------------------------------------------------------------ #
    # 7. rotate_half — applying twice negates
    # ------------------------------------------------------------------ #
    @_run("rotate_half: rotate_half(rotate_half(x)) == -x")
    def test_rotate_half_double():
        x = torch.randn(4, 16, 64)
        assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=ATOL)
    all_tests.append(test_rotate_half_double)

    # ------------------------------------------------------------------ #
    # 8. apply_rope — output shape
    # ------------------------------------------------------------------ #
    @_run("apply_rope: output shape")
    def test_apply_rope_shape():
        B, S, H, D = 2, 16, 8, 64
        cos_t, sin_t = precompute_rope_frequencies(D, S)
        x = torch.randn(B, S, H, D)
        out = apply_rope(x, cos_t, sin_t)
        assert out.shape == x.shape, f"{out.shape} != {x.shape}"
    all_tests.append(test_apply_rope_shape)

    # ------------------------------------------------------------------ #
    # 9. apply_rope — correctness against matrix-multiply reference
    # ------------------------------------------------------------------ #
    @_run("apply_rope: analytic matrix-multiply reference (float64)")
    def test_apply_rope_matrix_ref():
        torch.manual_seed(0)
        B, S, H, D = 2, 8, 3, 16
        base = 10000.0
        cos_t, sin_t = precompute_rope_frequencies(D, S, base=base, dtype=torch.float64)
        x = torch.randn(B, S, H, D, dtype=torch.float64)

        out = apply_rope(x, cos_t, sin_t)

        # Build reference via explicit rotation matrices
        ref = torch.zeros_like(x)
        for m in range(S):
            R = rope_rotation_matrix(m, D, base).to(dtype=torch.float64)
            for b in range(B):
                for h in range(H):
                    ref[b, m, h] = R @ x[b, m, h]

        assert torch.allclose(out, ref, atol=ATOL), \
            f"max diff = {(out - ref).abs().max().item()}"
    all_tests.append(test_apply_rope_matrix_ref)

    # ------------------------------------------------------------------ #
    # 10. apply_rope — with explicit positions
    # ------------------------------------------------------------------ #
    @_run("apply_rope: explicit positions match default")
    def test_apply_rope_explicit_pos():
        B, S, H, D = 2, 16, 4, 32
        cos_t, sin_t = precompute_rope_frequencies(D, S)
        x = torch.randn(B, S, H, D)
        positions = torch.arange(S).unsqueeze(0).expand(B, -1)

        out_default = apply_rope(x, cos_t, sin_t)
        out_explicit = apply_rope(x, cos_t, sin_t, positions=positions)
        assert torch.allclose(out_default, out_explicit, atol=ATOL)
    all_tests.append(test_apply_rope_explicit_pos)

    # ------------------------------------------------------------------ #
    # 11. apply_rope — scrambled positions differ
    # ------------------------------------------------------------------ #
    @_run("apply_rope: scrambled positions produce different output")
    def test_apply_rope_scrambled():
        B, S, H, D = 1, 8, 1, 16
        cos_t, sin_t = precompute_rope_frequencies(D, S)
        x = torch.randn(B, S, H, D)
        pos_scrambled = torch.tensor([[7, 3, 0, 5, 1, 6, 2, 4]])
        out_default = apply_rope(x, cos_t, sin_t)
        out_scrambled = apply_rope(x, cos_t, sin_t, positions=pos_scrambled)
        assert not torch.allclose(out_default, out_scrambled, atol=1e-6), \
            "Scrambled positions should differ"
    all_tests.append(test_apply_rope_scrambled)

    # ------------------------------------------------------------------ #
    # 12. apply_rope — norm preservation (orthogonal rotation)
    # ------------------------------------------------------------------ #
    @_run("apply_rope: L2-norm preservation")
    def test_apply_rope_norm_preservation():
        B, S, H, D = 4, 32, 8, 64
        cos_t, sin_t = precompute_rope_frequencies(D, S)
        x = torch.randn(B, S, H, D)
        out = apply_rope(x, cos_t, sin_t)
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=ATOL)
    all_tests.append(test_apply_rope_norm_preservation)

    # ------------------------------------------------------------------ #
    # 13. Relative-position property: <R_m q, R_n k> depends only on m-n
    # ------------------------------------------------------------------ #
    @_run("relative-position: dot product depends only on m-n")
    def test_relative_position():
        torch.manual_seed(1)
        B, S, H, D = 2, 16, 4, 32
        cos_t, sin_t = precompute_rope_frequencies(D, S + 10)
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)

        base_pos = torch.arange(S).unsqueeze(0).expand(B, -1)
        SHIFT = 7

        q0 = apply_rope(q, cos_t, sin_t, positions=base_pos)
        k0 = apply_rope(k, cos_t, sin_t, positions=base_pos)
        dot0 = (q0 * k0).sum(dim=-1)  # (B, S, H)

        qS = apply_rope(q, cos_t, sin_t, positions=base_pos + SHIFT)
        kS = apply_rope(k, cos_t, sin_t, positions=base_pos + SHIFT)
        dotS = (qS * kS).sum(dim=-1)

        assert torch.allclose(dot0, dotS, atol=FTOL), \
            f"max diff = {(dot0 - dotS).abs().max().item()}"
    all_tests.append(test_relative_position)

    # ------------------------------------------------------------------ #
    # 14. Same relative offset at different absolute positions
    # ------------------------------------------------------------------ #
    @_run("relative-position: same offset at different absolutes")
    def test_same_offset():
        torch.manual_seed(2)
        D = 32
        cos_t, sin_t = precompute_rope_frequencies(D, 256)
        q = torch.randn(1, 1, 1, D)
        k = torch.randn(1, 1, 1, D)

        # relative offset = 3
        out1 = apply_rope(q, cos_t, sin_t, positions=torch.tensor([[10]]))
        out2 = apply_rope(k, cos_t, sin_t, positions=torch.tensor([[7]]))
        dot1 = (out1 * out2).sum()

        out3 = apply_rope(q, cos_t, sin_t, positions=torch.tensor([[50]]))
        out4 = apply_rope(k, cos_t, sin_t, positions=torch.tensor([[47]]))
        dot2 = (out3 * out4).sum()

        assert torch.allclose(dot1, dot2, atol=FTOL), \
            f"{dot1.item()} vs {dot2.item()}"
    all_tests.append(test_same_offset)

    # ------------------------------------------------------------------ #
    # 15. Grad check: gradients are finite and non-zero
    # ------------------------------------------------------------------ #
    @_run("apply_rope: finite and non-zero gradients")
    def test_finite_grads():
        B, S, H, D = 2, 16, 4, 32
        cos_t, sin_t = precompute_rope_frequencies(D, S)
        x = torch.randn(B, S, H, D, requires_grad=True)
        out = apply_rope(x, cos_t, sin_t)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient"
        assert torch.all(torch.isfinite(x.grad)), "Non-finite gradient"
        assert x.grad.abs().sum() > 0, "Zero gradient"
    all_tests.append(test_finite_grads)

    # ------------------------------------------------------------------ #
    # 16. Grad check via torch.autograd.gradcheck (double precision)
    # ------------------------------------------------------------------ #
    @_run("apply_rope: autograd.gradcheck (float64)")
    def test_gradcheck():
        torch.manual_seed(7)
        B, S, H, D = 1, 4, 2, 8
        cos_t, sin_t = precompute_rope_frequencies(D, S, dtype=torch.float64)
        cos_t.requires_grad_(False)
        sin_t.requires_grad_(False)

        x = torch.randn(B, S, H, D, dtype=torch.float64, requires_grad=True)

        def fn(inp):
            return apply_rope(inp, cos_t, sin_t)

        result = torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-5)
        assert result, "gradcheck failed"
    all_tests.append(test_gradcheck)

    # ------------------------------------------------------------------ #
    # 17. RoPE module: buffer registration
    # ------------------------------------------------------------------ #
    @_run("RoPE module: buffers registered")
    def test_rope_buffers():
        rope = RoPE(head_dim=64, max_seq_len=256)
        buf_names = {n for n, _ in rope.named_buffers()}
        assert "cos_table" in buf_names
        assert "sin_table" in buf_names
    all_tests.append(test_rope_buffers)

    # ------------------------------------------------------------------ #
    # 18. RoPE module: forward shape
    # ------------------------------------------------------------------ #
    @_run("RoPE module: forward output shape")
    def test_rope_module_shape():
        B, S, H, D = 2, 32, 8, 64
        rope = RoPE(head_dim=D, max_seq_len=512)
        x = torch.randn(B, S, H, D)
        out = rope(x)
        assert out.shape == (B, S, H, D), f"{out.shape}"
    all_tests.append(test_rope_module_shape)

    # ------------------------------------------------------------------ #
    # 19. RoPE module: .to(device) moves buffers
    # ------------------------------------------------------------------ #
    @_run("RoPE module: .to() moves buffers")
    def test_rope_to_device():
        rope = RoPE(head_dim=32, max_seq_len=64)
        rope = rope.to(dtype=torch.float16)
        assert rope.cos_table.dtype == torch.float16
        assert rope.sin_table.dtype == torch.float16
    all_tests.append(test_rope_to_device)

    # ------------------------------------------------------------------ #
    # 20. RoPE module: parameters are empty (no trainable weights)
    # ------------------------------------------------------------------ #
    @_run("RoPE module: no trainable parameters")
    def test_rope_no_params():
        rope = RoPE(head_dim=64, max_seq_len=128)
        params = list(rope.parameters())
        assert len(params) == 0, f"Expected 0 params, got {len(params)}"
    all_tests.append(test_rope_no_params)

    # ------------------------------------------------------------------ #
    # 21. RoPE module: gradient flows through
    # ------------------------------------------------------------------ #
    @_run("RoPE module: gradient flows through to input")
    def test_rope_module_grad():
        rope = RoPE(head_dim=32, max_seq_len=64)
        x = torch.randn(1, 8, 2, 32, requires_grad=True)
        out = rope(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))
    all_tests.append(test_rope_module_grad)

    # ------------------------------------------------------------------ #
    # 22. RoPE module: equivalence with apply_rope
    # ------------------------------------------------------------------ #
    @_run("RoPE module: equivalence with apply_rope function")
    def test_module_vs_function():
        B, S, H, D = 2, 16, 4, 32
        rope = RoPE(head_dim=D, max_seq_len=128, base=10000.0)
        cos_t, sin_t = precompute_rope_frequencies(D, 128, base=10000.0)
        x = torch.randn(B, S, H, D)
        out_module = rope(x)
        out_func = apply_rope(x, cos_t, sin_t)
        assert torch.allclose(out_module, out_func, atol=ATOL)
    all_tests.append(test_module_vs_function)

    # ------------------------------------------------------------------ #
    # 23. Identity at position 0 (cos=1, sin=0 → x unchanged)
    # ------------------------------------------------------------------ #
    @_run("apply_rope: position 0 is identity (cos=1, sin=0)")
    def test_position_zero_identity():
        D = 32
        cos_t, sin_t = precompute_rope_frequencies(D, 1)
        x = torch.randn(1, 1, 1, D)
        # At position 0: θ = 0 → cos=1, sin=0
        # rotate_half(x) * sin = 0, so output = x * 1 = x
        out = apply_rope(x, cos_t, sin_t, positions=torch.tensor([[0]]))
        assert torch.allclose(out, x, atol=ATOL), \
            f"max diff = {(out - x).abs().max().item()}"
    all_tests.append(test_position_zero_identity)

    # ------------------------------------------------------------------ #
    # 24. custom base parameter
    # ------------------------------------------------------------------ #
    @_run("precompute_rope_frequencies: custom base")
    def test_custom_base():
        D, T = 16, 4
        base = 500.0
        cos_t, sin_t = precompute_rope_frequencies(D, T, base=base)
        for m in range(T):
            for i in range(D // 2):
                theta = m * (base ** (-2.0 * i / D))
                assert torch.allclose(
                    cos_t[m, 2 * i].float(),
                    torch.tensor(math.cos(theta)),
                    atol=ATOL,
                )
    all_tests.append(test_custom_base)

    # ------------------------------------------------------------------ #
    # 25. extra_repr
    # ------------------------------------------------------------------ #
    @_run("RoPE module: extra_repr")
    def test_extra_repr():
        rope = RoPE(head_dim=64, max_seq_len=1024)
        r = repr(rope)
        assert "64" in r and "1024" in r
    all_tests.append(test_extra_repr)

    # ------------------------------------------------------------------ #
    # 26. Single-token (seq_len=1) edge case
    # ------------------------------------------------------------------ #
    @_run("apply_rope: single-token edge case")
    def test_single_token():
        D = 16
        cos_t, sin_t = precompute_rope_frequencies(D, 1)
        x = torch.randn(1, 1, 2, D)
        out = apply_rope(x, cos_t, sin_t)
        assert out.shape == x.shape
        # position 0 → identity
        assert torch.allclose(out, x, atol=ATOL)
    all_tests.append(test_single_token)

    # ------------------------------------------------------------------ #
    # 27. Orthogonality: dot(q, R_m^T R_n k) == dot(R_m q, R_n k)
    # ------------------------------------------------------------------ #
    @_run("RoPE: orthogonality — R^T R = I")
    def test_rotation_orthogonal():
        D = 32
        cos_t, sin_t = precompute_rope_frequencies(D, 128)
        for m in [0, 1, 5, 20, 99]:
            R = rope_rotation_matrix(m, D)
            # R^T @ R should be identity
            RtR = R.T @ R
            assert torch.allclose(RtR, torch.eye(D), atol=FTOL), \
                f"m={m}: max off-diag = {(RtR - torch.eye(D)).abs().max().item()}"
    all_tests.append(test_rotation_orthogonal)

    # ------------------------------------------------------------------ #
    # Run all
    # ------------------------------------------------------------------ #
    print(f"\nRunning {len(all_tests)} tests for from_scratch RoPE …\n")
    for t in all_tests:
        try:
            t()
            passed += 1
        except Exception:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(all_tests)} total")
    if failed:
        print("SOME TESTS FAILED")
        raise SystemExit(1)
    else:
        print("ALL TESTS PASSED ✓")


if __name__ == "__main__":
    main()
