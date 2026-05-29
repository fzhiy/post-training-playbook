"""Test suite for from_scratch.GroupedQueryAttention.

Run:  python test_gqa_mqa.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import unittest

from from_scratch import GroupedQueryAttention


# ──────────────────────────────────────────────────────────────────────
# Standalone reference forward (shares weights with the module under test)
# ──────────────────────────────────────────────────────────────────────
def _reference_gqa_forward(x, W_q, W_k, W_v, W_o, n_heads, n_kv_heads, d_head):
    """
    Pure-PyTorch reference GQA forward.
    Re-implements every step independently so bugs in the module are caught.
    """
    B, S, d_model = x.shape
    n_groups = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(d_head)

    q = W_q(x).view(B, S, n_heads, d_head).transpose(1, 2)      # (B,H,S,D)
    k = W_k(x).view(B, S, n_kv_heads, d_head).transpose(1, 2)   # (B,Hkv,S,D)
    v = W_v(x).view(B, S, n_kv_heads, d_head).transpose(1, 2)   # (B,Hkv,S,D)

    k = k.repeat_interleave(n_groups, dim=1)                      # (B,H,S,D)
    v = v.repeat_interleave(n_groups, dim=1)                      # (B,H,S,D)

    attn = (q @ k.transpose(-2, -1)) * scale                     # (B,H,S,S)
    causal = torch.triu(
        torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1
    )
    attn = attn.masked_fill(causal[None, None], float("-inf"))
    attn = F.softmax(attn, dim=-1)

    out = (attn @ v)                                              # (B,H,S,D)
    out = out.transpose(1, 2).contiguous().reshape(B, S, d_model)
    return W_o(out)


# ──────────────────────────────────────────────────────────────────────
# Test suite
# ──────────────────────────────────────────────────────────────────────
class TestGroupedQueryAttention(unittest.TestCase):

    D: int = 64
    N_HEADS: int = 8
    D_HEAD: int = D // N_HEADS
    B: int = 2
    S: int = 16

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.x = torch.randn(cls.B, cls.S, cls.D)

    # -- helpers --------------------------------------------------------
    def _make(self, n_kv_heads, bias=False, **kw):
        return GroupedQueryAttention(
            self.D, self.N_HEADS, n_kv_heads=n_kv_heads, bias=bias, **kw
        )

    # ================================================================== #
    #  1.  SHAPE CHECKS                                                   #
    # ================================================================== #
    def test_shape_gqa(self):
        out = self._make(2)(self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_shape_mqa(self):
        out = self._make(1)(self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_shape_mha(self):
        out = self._make(self.N_HEADS)(self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    def test_shape_varying_seq_len(self):
        mod = self._make(2, max_seq_len=128)
        for s in (1, 7, 32, 64, 128):
            out = mod(torch.randn(1, s, self.D))
            self.assertEqual(out.shape, (1, s, self.D), f"Failed at seq_len={s}")

    def test_shape_with_bias(self):
        out = self._make(2, bias=True)(self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))

    # ================================================================== #
    #  2.  GRADIENT CHECKS                                                #
    # ================================================================== #
    def test_grads_exist_and_finite(self):
        for n_kv in (1, 2, 4, self.N_HEADS):
            with self.subTest(n_kv_heads=n_kv):
                mod = self._make(n_kv, bias=False)
                mod(self.x).sum().backward()
                for name, p in mod.named_parameters():
                    self.assertIsNotNone(p.grad, f"None grad: {name}")
                    self.assertTrue(
                        torch.isfinite(p.grad).all(),
                        f"Non-finite grad: {name}",
                    )

    def test_grad_shapes_match_params(self):
        mod = self._make(2, bias=False)
        mod(self.x).sum().backward()
        for name, p in mod.named_parameters():
            self.assertEqual(
                p.grad.shape, p.shape,
                f"Shape mismatch: {name}  param={p.shape}  grad={p.grad.shape}",
            )

    def test_grads_with_bias(self):
        mod = self._make(2, bias=True)
        mod(self.x).sum().backward()
        for name, p in mod.named_parameters():
            self.assertIsNotNone(p.grad, f"None grad (bias=True): {name}")
            self.assertTrue(
                torch.isfinite(p.grad).all(),
                f"Non-finite grad (bias=True): {name}",
            )

    # ================================================================== #
    #  3.  NUMERICAL CORRECTNESS                                          #
    # ================================================================== #
    def test_vs_reference_gqa(self):
        """Module output must match hand-written reference for each n_kv."""
        for n_kv in (1, 2, 4):
            with self.subTest(n_kv_heads=n_kv):
                mod = self._make(n_kv)
                mod.eval()
                with torch.no_grad():
                    out_mod = mod(self.x)
                    out_ref = _reference_gqa_forward(
                        self.x,
                        mod.W_q, mod.W_k, mod.W_v, mod.W_o,
                        self.N_HEADS, n_kv, self.D_HEAD,
                    )
                self.assertTrue(
                    torch.allclose(out_mod, out_ref, atol=1e-5),
                    f"n_kv={n_kv}  max-diff="
                    f"{(out_mod - out_ref).abs().max().item():.2e}",
                )

    def test_vs_pytorch_multihead_attention(self):
        """
        When n_kv_heads == n_heads the module is standard MHA.
        Compare against nn.MultiheadAttention with the same weights.
        """
        n = self.N_HEADS
        mod = self._make(n, dropout=0.0)
        mod.eval()

        ref = nn.MultiheadAttention(
            self.D, n, bias=False, dropout=0.0, batch_first=True,
        )

        # Transfer weights from our module into the PyTorch reference
        with torch.no_grad():
            ref.in_proj_weight.copy_(
                torch.cat([mod.W_q.weight, mod.W_k.weight, mod.W_v.weight])
            )
            ref.out_proj.weight.copy_(mod.W_o.weight)
        ref.eval()

        # Float causal mask: 0 on lower-triangle, -inf on upper-triangle
        causal = torch.triu(
            torch.full((self.S, self.S), float("-inf")), diagonal=1
        )

        with torch.no_grad():
            out_mod = mod(self.x)
            out_ref, _ = ref(self.x, self.x, self.x, attn_mask=causal)

        self.assertTrue(
            torch.allclose(out_mod, out_ref, atol=1e-4),
            f"MHA mismatch  max-diff="
            f"{(out_mod - out_ref).abs().max().item():.2e}",
        )

    def test_double_precision_exact(self):
        """With float64, module and reference should agree to tighter tolerance."""
        x64 = self.x.double()
        mod = self._make(2)
        mod.double()
        mod.eval()
        with torch.no_grad():
            out_mod = mod(x64)
            out_ref = _reference_gqa_forward(
                x64,
                mod.W_q, mod.W_k, mod.W_v, mod.W_o,
                self.N_HEADS, 2, self.D_HEAD,
            )
        self.assertTrue(
            torch.allclose(out_mod, out_ref, atol=1e-10),
            f"float64 max-diff={( out_mod - out_ref).abs().max().item():.2e}",
        )

    # ================================================================== #
    #  4.  CAUSALITY                                                      #
    # ================================================================== #
    def test_causality_future_independent(self):
        """
        Perturbing tokens at positions >= k must not change outputs at 0..k-1
        thanks to the causal mask.
        """
        mod = self._make(2)
        mod.eval()
        k = self.S // 2

        x_perturbed = self.x.clone()
        x_perturbed[:, k:, :] = torch.randn(self.B, self.S - k, self.D)

        with torch.no_grad():
            out1 = mod(self.x)
            out2 = mod(x_perturbed)

        self.assertTrue(
            torch.allclose(out1[:, :k, :], out2[:, :k, :], atol=1e-6),
            f"Causality broken  max-diff in positions 0..{k-1}: "
            f"{(out1[:, :k] - out2[:, :k]).abs().max().item():.2e}",
        )

    # ================================================================== #
    #  5.  DETERMINISM                                                    #
    # ================================================================== #
    def test_deterministic_eval(self):
        mod = self._make(2)
        mod.eval()
        with torch.no_grad():
            a = mod(self.x)
            b = mod(self.x)
        self.assertTrue(torch.equal(a, b), "Non-deterministic in eval mode")

    # ================================================================== #
    #  6.  ATTENTION MASK                                                 #
    # ================================================================== #
    def test_attn_mask_changes_output(self):
        """Blocking the last key position must alter the output."""
        mod = self._make(2)
        mod.eval()

        mask = torch.ones(self.B, 1, 1, self.S, dtype=torch.bool)
        mask[:, :, :, -1] = False          # block last key

        with torch.no_grad():
            out_nomask = mod(self.x)
            out_masked = mod(self.x, attn_mask=mask)

        self.assertFalse(
            torch.equal(out_nomask, out_masked),
            "attn_mask had no effect — something is wrong",
        )

    # ================================================================== #
    # # 7.  WEIGHT SHAPES                                                #
    # ================================================================== #
    def test_weight_shapes(self):
        for n_kv in (1, 2, 4, 8):
            with self.subTest(n_kv_heads=n_kv):
                mod = self._make(n_kv)
                d = self.D_HEAD
                self.assertEqual(mod.W_q.weight.shape, (self.N_HEADS * d, self.D))
                self.assertEqual(mod.W_k.weight.shape, (n_kv * d, self.D))
                self.assertEqual(mod.W_v.weight.shape, (n_kv * d, self.D))
                self.assertEqual(mod.W_o.weight.shape, (self.D, self.D))

    # ================================================================== #
    #  8.  EDGE / CORNER CASES                                            #
    # ================================================================== #
    def test_single_head(self):
        mod = GroupedQueryAttention(32, n_heads=1, n_kv_heads=1)
        out = mod(torch.randn(1, 4, 32))
        self.assertEqual(out.shape, (1, 4, 32))

    def test_seq_len_one(self):
        out = self._make(2)(torch.randn(1, 1, self.D))
        self.assertEqual(out.shape, (1, 1, self.D))

    def test_gqa_real_grouping(self):
        """n_groups > 1 is the real GQA regime."""
        mod = self._make(4)           # 8 // 4 = 2 query-heads per KV-head
        self.assertEqual(mod.n_groups, 2)
        out = mod(self.x)
        self.assertEqual(out.shape, (self.B, self.S, self.D))
        out.sum().backward()
        for p in mod.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_identical_inputs_stable(self):
        """Identical tokens → well-defined (finite) output, no NaN."""
        mod = self._make(2)
        x_ident = torch.ones(1, 8, self.D)
        out = mod(x_ident)
        self.assertTrue(torch.isfinite(out).all(), "NaN/Inf with identical inputs")

    def test_parameter_count_gqa_less_than_mha(self):
        """GQA should have fewer KV parameters than full MHA."""
        gqa = self._make(2)
        mha = self._make(self.N_HEADS)
        gqa_kv = sum(p.numel() for n, p in gqa.named_parameters() if "W_k" in n or "W_v" in n)
        mha_kv = sum(p.numel() for n, p in mha.named_parameters() if "W_k" in n or "W_v" in n)
        self.assertLess(gqa_kv, mha_kv)


if __name__ == "__main__":
    unittest.main()
