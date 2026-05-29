"""
Tests for from_scratch decoding utilities.

Run:  python test_sampling.py
"""

import torch
import torch.nn.functional as F
import math

# ── import the module under test ─────────────────────────────────────
from from_scratch import apply_temperature, apply_top_k, apply_top_p, sample

torch.manual_seed(0)
ATOL = 1e-5


# ── helpers ───────────────────────────────────────────────────────────
def _report(name: str, passed: bool):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    assert passed, f"{name} failed!"


# ══════════════════════════════════════════════════════════════════════
#  1. apply_temperature
# ══════════════════════════════════════════════════════════════════════
def test_temperature_identity():
    """temperature == 1.0 must be a no-op."""
    logits = torch.randn(4, 50)
    out = apply_temperature(logits, 1.0)
    _report("temperature=1 identity",
            torch.allclose(out, logits, atol=ATOL))


def test_temperature_scaling():
    """Out must equal logits / temperature (analytic)."""
    logits = torch.randn(2, 20)
    for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
        out = apply_temperature(logits, temp)
        expected = logits / temp
        _report(f"temperature={temp} scaling",
                torch.allclose(out, expected, atol=ATOL))


def test_temperature_shape():
    logits = torch.randn(3, 100)
    out = apply_temperature(logits, 0.7)
    _report("temperature shape preserved", out.shape == logits.shape)


def test_temperature_gradient():
    """Gradients must flow and be finite."""
    logits = torch.randn(2, 30, requires_grad=True)
    out = apply_temperature(logits, 0.8)
    loss = out.sum()
    loss.backward()
    _report("temperature gradient finite",
            logits.grad is not None and torch.isfinite(logits.grad).all())


def test_temperature_invalid():
    for bad in [0.0, -1.0]:
        try:
            apply_temperature(torch.randn(2, 5), bad)
            _report(f"temperature={bad} raises", False)
        except ValueError:
            _report(f"temperature={bad} raises ValueError", True)


# ══════════════════════════════════════════════════════════════════════
#  2. apply_top_k
# ══════════════════════════════════════════════════════════════════════
def test_top_k_shape():
    logits = torch.randn(4, 50)
    out = apply_top_k(logits, k=5)
    _report("top-k shape preserved", out.shape == logits.shape)


def test_top_k_exactly_k_survive():
    """Exactly k logits per row must be finite (not -inf)."""
    batch, vocab = 4, 50
    logits = torch.randn(batch, vocab)
    for k in [1, 5, 10, 25]:
        out = apply_top_k(logits, k=k)
        finite_count = torch.isfinite(out).sum(dim=-1)  # per row
        expected = min(k, vocab)
        ok = (finite_count == expected).all()
        _report(f"top-k={k}: exactly {expected} survive per row", ok)


def test_top_k_survivors_match_topk():
    """The surviving values must equal the true top-k logits."""
    logits = torch.randn(3, 40)
    k = 7
    out = apply_top_k(logits, k=k)
    topk_vals, _ = torch.topk(logits, k=k, dim=-1)
    # Gather the finite (surviving) values from out per row
    for i in range(logits.shape[0]):
        finite_mask = torch.isfinite(out[i])
        surviving = out[i][finite_mask]
        expected = topk_vals[i]
        ok = torch.allclose(surviving.sort().values, expected.sort().values, atol=ATOL)
        _report(f"top-k={k} row {i} survivors match", ok)


def test_top_k_full_vocab():
    """k >= vocab_size should keep everything."""
    logits = torch.randn(2, 10)
    out = apply_top_k(logits, k=100)
    _report("top-k >= vocab: no filtering",
            torch.allclose(out, logits, atol=ATOL))


def test_top_k_k1_greedy():
    """k=1 keeps only the argmax logit."""
    logits = torch.randn(5, 20)
    out = apply_top_k(logits, k=1)
    for i in range(5):
        amax = logits[i].argmax()
        ok = bool(torch.isfinite(out[i]).sum() == 1) and bool(torch.isfinite(out[i, amax]))
        _report(f"top-k=1 row {i} keeps argmax", ok)


def test_top_k_gradient():
    logits = torch.randn(2, 30, requires_grad=True)
    out = apply_top_k(logits, k=5)
    loss = out.sum()
    loss.backward()
    _report("top-k gradient finite",
            logits.grad is not None and torch.isfinite(logits.grad).all())
    # Gradient should be 0 for masked positions
    with torch.no_grad():
        out_np = apply_top_k(logits.detach(), k=5)
        masked = ~torch.isfinite(out_np) | (out_np == -float("inf"))
        grad_zero_at_masked = (logits.grad[masked] == 0).all()
    _report("top-k gradient zero at masked positions", grad_zero_at_masked)


def test_top_k_invalid():
    try:
        apply_top_k(torch.randn(2, 5), k=0)
        _report("top-k=0 raises", False)
    except ValueError:
        _report("top-k=0 raises ValueError", True)


# ══════════════════════════════════════════════════════════════════════
#  3. apply_top_p
# ══════════════════════════════════════════════════════════════════════
def test_top_p_shape():
    logits = torch.randn(4, 50)
    out = apply_top_p(logits, top_p=0.9)
    _report("top-p shape preserved", out.shape == logits.shape)


def test_top_p_1_no_op():
    """top_p=1.0 should keep everything (cum_prob < 1.0 for all but last)."""
    logits = torch.randn(3, 20)
    out = apply_top_p(logits, top_p=1.0)
    _report("top-p=1.0 no filtering",
            torch.allclose(out, logits, atol=ATOL))


def test_top_p_filters_correctly():
    """Manually verify that the kept tokens cover ≥ top_p mass."""
    torch.manual_seed(123)
    logits = torch.randn(2, 50)
    top_p = 0.6

    out = apply_top_p(logits, top_p=top_p)
    probs = F.softmax(logits, dim=-1)

    for i in range(logits.shape[0]):
        kept_mask = torch.isfinite(out[i])
        kept_mass = probs[i][kept_mask].sum()
        ok = kept_mass.item() >= top_p - ATOL
        _report(f"top-p={top_p} row {i} kept mass {kept_mass:.4f} >= {top_p}", ok)


def test_top_p_min_tokens_to_keep():
    """min_tokens_to_keep guarantees at least that many survive."""
    logits = torch.randn(4, 100)
    # Very small top_p would normally keep 1 token, but we ask for 5
    out = apply_top_p(logits, top_p=0.001, min_tokens_to_keep=5)
    for i in range(4):
        n_survived = torch.isfinite(out[i]).sum().item()
        _report(f"top-p min_tokens_to_keep=5 row {i} survived={n_survived}",
                n_survived >= 5)


def test_top_p_kept_are_top_probability_tokens():
    """The kept tokens should be those with the highest probabilities."""
    torch.manual_seed(77)
    logits = torch.randn(1, 30)
    top_p = 0.5
    out = apply_top_p(logits, top_p=top_p)
    probs = F.softmax(logits, dim=-1)

    kept_idx = torch.isfinite(out[0]).nonzero(as_tuple=True)[0]
    removed_idx = ~torch.isfinite(out[0])
    # Every kept prob should be >= every removed prob (or equal due to ties)
    if kept_idx.numel() > 0 and removed_idx.sum() > 0:
        min_kept = probs[0][kept_idx].min()
        max_removed = probs[0][removed_idx].max()
        # Allow small tolerance for floating point
        _report("top-p: kept tokens have higher probs",
                min_kept.item() >= max_removed.item() - 1e-6)
    else:
        _report("top-p: kept tokens have higher probs", True)


def test_top_p_gradient():
    logits = torch.randn(2, 30, requires_grad=True)
    out = apply_top_p(logits, top_p=0.9)
    loss = out.sum()
    loss.backward()
    _report("top-p gradient finite",
            logits.grad is not None and torch.isfinite(logits.grad).all())
    # Gradient should be 0 for removed tokens
    with torch.no_grad():
        out_np = apply_top_p(logits.detach(), top_p=0.9)
        masked = ~torch.isfinite(out_np)
        grad_zero_at_masked = (logits.grad[masked] == 0).all()
    _report("top-p gradient zero at masked positions", grad_zero_at_masked)


def test_top_p_invalid():
    for bad in [0.0, -0.1, 1.5]:
        try:
            apply_top_p(torch.randn(2, 5), top_p=bad)
            _report(f"top-p={bad} raises", False)
        except ValueError:
            _report(f"top-p={bad} raises ValueError", True)


# ══════════════════════════════════════════════════════════════════════
#  4. sample  (end-to-end)
# ══════════════════════════════════════════════════════════════════════
def test_sample_shape():
    logits = torch.randn(4, 50)
    ids = sample(logits)
    _report("sample output shape", ids.shape == (4,))


def test_sample_values_in_range():
    logits = torch.randn(8, 100)
    ids = sample(logits, temperature=1.0, top_k=10, top_p=0.9)
    ok = (ids >= 0).all() and (ids < 100).all()
    _report("sample values in [0, vocab_size)", ok)


def test_sample_greedy_matches_argmax():
    """Very low temperature → deterministic argmax."""
    torch.manual_seed(999)
    logits = torch.randn(10, 50)
    greedy_ids = sample(logits, temperature=0.001)
    expected = logits.argmax(dim=-1)
    _report("greedy sampling matches argmax",
            torch.allclose(greedy_ids.float(), expected.float(), atol=ATOL))


def test_sample_deterministic_with_seed():
    """Same seed → same output."""
    logits = torch.randn(3, 20)
    torch.manual_seed(42)
    ids1 = sample(logits, temperature=0.8, top_k=5, top_p=0.9)
    torch.manual_seed(42)
    ids2 = sample(logits, temperature=0.8, top_k=5, top_p=0.9)
    _report("sample deterministic with same seed",
            torch.equal(ids1, ids2))


def test_sample_top_k_one():
    """top_k=1 → always pick the argmax."""
    torch.manual_seed(0)
    logits = torch.randn(20, 30)
    ids = sample(logits, temperature=1.0, top_k=1)
    expected = logits.argmax(dim=-1)
    _report("sample with top_k=1 → argmax", torch.equal(ids, expected))


def test_sample_top_p_skewed():
    """With a strongly skewed distribution, top_p should narrow choices."""
    torch.manual_seed(0)
    # Make one logit much larger
    logits = torch.full((1, 20), -10.0)
    logits[0, 7] = 10.0  # dominant
    # Even with generous top_p, should almost always pick 7
    hits = sum(
        sample(logits, temperature=1.0, top_p=0.95).item() == 7
        for _ in range(200)
    )
    _report(f"sample top-p skewed: picked 7 {hits}/200 times", hits > 190)


def test_sample_top_k_restricts_range():
    """Samples from top-k should lie within the top-k set."""
    torch.manual_seed(55)
    logits = torch.randn(5, 40)
    k = 8
    for _ in range(50):
        ids = sample(logits, temperature=1.0, top_k=k)
        for i in range(5):
            _, topk_idx = torch.topk(logits[i], k=k)
            ok = ids[i].item() in topk_idx.tolist()
            _report(f"sample top-k in top-k set row {i}", ok)


def test_sample_through_pipeline_gradient():
    """Gradients flow through temperature + top-k + top-p (not through multinomial).

    We test that the *logits* that reach softmax have finite gradients
    by manually running the pipeline without the final sampling step.
    """
    logits = torch.randn(2, 30, requires_grad=True)
    scaled = apply_temperature(logits, 0.7)
    filtered = apply_top_k(scaled, k=10)
    filtered = apply_top_p(filtered, top_p=0.9)
    probs = F.softmax(filtered, dim=-1)
    loss = probs.sum()
    loss.backward()
    _report("end-to-end gradient finite",
            logits.grad is not None and torch.isfinite(logits.grad).all())


def test_sample_output_dtype():
    logits = torch.randn(3, 20)
    ids = sample(logits)
    _report("sample output dtype is int64/long", ids.dtype == torch.long)


# ══════════════════════════════════════════════════════════════════════
#  5. Combined / regression checks
# ══════════════════════════════════════════════════════════════════════
def test_temperature_preserves_argmax():
    """Temperature scaling should not change argmax (just rescale)."""
    logits = torch.randn(5, 40)
    for temp in [0.3, 0.5, 1.0, 2.0, 5.0]:
        scaled = apply_temperature(logits, temp)
        _report(f"temp={temp} preserves argmax",
                (scaled.argmax(dim=-1) == logits.argmax(dim=-1)).all())


def test_top_k_preserves_order_among_survivors():
    """Among surviving logits, relative order must be preserved."""
    logits = torch.randn(3, 50)
    out = apply_top_k(logits, k=10)
    for i in range(3):
        finite_mask = torch.isfinite(out[i])
        original_vals = logits[i][finite_mask]
        out_vals = out[i][finite_mask]
        _report(f"top-k row {i} survivor order preserved",
                torch.allclose(original_vals, out_vals, atol=ATOL))


def test_top_p_preserves_order_among_survivors():
    """Among surviving logits, relative order must be preserved."""
    logits = torch.randn(3, 50)
    out = apply_top_p(logits, top_p=0.7)
    for i in range(3):
        finite_mask = torch.isfinite(out[i])
        original_vals = logits[i][finite_mask]
        out_vals = out[i][finite_mask]
        _report(f"top-p row {i} survivor order preserved",
                torch.allclose(original_vals, out_vals, atol=ATOL))


def test_combined_filters_are_stricter():
    """top-k AND top-p together should be at least as restrictive as either alone."""
    torch.manual_seed(321)
    logits = torch.randn(2, 100)
    k, p = 10, 0.5

    out_k = apply_top_k(logits, k=k)
    out_p = apply_top_p(logits, top_p=p)

    out_both = apply_top_p(apply_top_k(logits, k=k), top_p=p)

    for i in range(2):
        n_k = torch.isfinite(out_k[i]).sum().item()
        n_p = torch.isfinite(out_p[i]).sum().item()
        n_both = torch.isfinite(out_both[i]).sum().item()
        _report(f"row {i}: combined ({n_both}) <= top-k ({n_k})",
                n_both <= n_k + 1e-9)  # tiny tolerance for edge cases
        _report(f"row {i}: combined ({n_both}) <= top-p ({n_p})",
                n_both <= n_p + 1e-9)


# ══════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════
ALL_TESTS = [
    # temperature
    test_temperature_identity,
    test_temperature_scaling,
    test_temperature_shape,
    test_temperature_gradient,
    test_temperature_invalid,
    test_temperature_preserves_argmax,
    # top-k
    test_top_k_shape,
    test_top_k_exactly_k_survive,
    test_top_k_survivors_match_topk,
    test_top_k_full_vocab,
    test_top_k_k1_greedy,
    test_top_k_gradient,
    test_top_k_invalid,
    test_top_k_preserves_order_among_survivors,
    # top-p
    test_top_p_shape,
    test_top_p_1_no_op,
    test_top_p_filters_correctly,
    test_top_p_min_tokens_to_keep,
    test_top_p_kept_are_top_probability_tokens,
    test_top_p_gradient,
    test_top_p_invalid,
    test_top_p_preserves_order_among_survivors,
    # sample
    test_sample_shape,
    test_sample_values_in_range,
    test_sample_greedy_matches_argmax,
    test_sample_deterministic_with_seed,
    test_sample_top_k_one,
    test_sample_top_p_skewed,
    test_sample_top_k_restricts_range,
    test_sample_through_pipeline_gradient,
    test_sample_output_dtype,
    # combined
    test_combined_filters_are_stricter,
]


if __name__ == "__main__":
    print("=" * 60)
    print("Running from_scratch sampling tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        raise SystemExit(1)
    print("All tests passed!")
