"""Tests for the SimPO loss implementation.

Run:
    python test_simpo_loss.py        # or: python -m pytest test_simpo_loss.py

Tolerance: ~2e-2 (matching the drill suite convention).
"""
import math
import sys

import torch
import torch.nn.functional as F

from from_scratch import sequence_logp, simpo_loss


# ---------------------------------------------------------------------------
# 1. sequence_logp — shape and numerical correctness
# ---------------------------------------------------------------------------

def test_sequence_logp_shape():
    torch.manual_seed(0)
    batch, seq_len, vocab = 3, 10, 20
    logits = torch.randn(batch, seq_len, vocab)
    labels = torch.randint(0, vocab, (batch, seq_len))
    labels[:, -2:] = -100  # padding

    lp = sequence_logp(logits, labels)
    assert lp.shape == (batch,), f"Expected ({batch},), got {lp.shape}"


def test_sequence_logp_agrees_with_cross_entropy():
    """sequence_logp should match −CE(reduction='none') summed over non-padding tokens."""
    torch.manual_seed(1)
    batch, seq_len, vocab = 2, 8, 15
    logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq_len))
    labels[:, -3:] = -100

    lp = sequence_logp(logits, labels)

    # Reference via built-in cross-entropy (which is -log_softmax + gather)
    shifted_logits = logits[:, :-1, :].contiguous().view(-1, vocab)
    shifted_labels = labels[:, 1:].contiguous().view(-1)
    ce = F.cross_entropy(shifted_logits, shifted_labels,
                         ignore_index=-100, reduction='none')
    ce = ce.view(batch, -1)
    mask = (labels[:, 1:] != -100).float()
    ref_lp = -(ce * mask).sum(dim=-1)  # sum of log-probs

    assert torch.allclose(lp, ref_lp, atol=1e-5), \
        f"sequence_logp mismatch: max diff = {(lp - ref_lp).abs().max():.2e}"


def test_sequence_logp_gradient_flows():
    torch.manual_seed(2)
    batch, seq_len, vocab = 2, 6, 10
    logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq_len))

    lp = sequence_logp(logits, labels)
    lp.sum().backward()
    assert logits.grad is not None, "No gradient through sequence_logp"
    assert torch.isfinite(logits.grad).all(), "Non-finite gradients in sequence_logp"


# ---------------------------------------------------------------------------
# 2. simpo_loss — shape, finiteness, analytic correctness
# ---------------------------------------------------------------------------

def test_simpo_loss_shape():
    torch.manual_seed(3)
    batch = 5
    logps_w  = torch.randn(batch, requires_grad=True)
    logps_l  = torch.randn(batch, requires_grad=True)
    len_w    = torch.randint(5, 20, (batch,)).float()
    len_l    = torch.randint(5, 20, (batch,)).float()

    loss, r_w, r_l, margin = simpo_loss(logps_w, logps_l, len_w, len_l)

    assert loss.shape == torch.Size([]), f"loss should be scalar, got {loss.shape}"
    assert r_w.shape == (batch,),    f"reward_chosen shape: {r_w.shape}"
    assert r_l.shape == (batch,),    f"reward_rejected shape: {r_l.shape}"
    assert margin.shape == (batch,), f"margin shape: {margin.shape}"
    for name, t in [("loss", loss), ("r_w", r_w), ("r_l", r_l), ("margin", margin)]:
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"


def test_simpo_loss_analytic():
    """Check loss matches the closed-form expression across (beta, gamma) combinations."""
    torch.manual_seed(4)
    batch = 4
    logps_w = torch.randn(batch, requires_grad=True)
    logps_l = torch.randn(batch, requires_grad=True)
    len_w   = torch.tensor([10.0, 8.0, 15.0, 12.0])
    len_l   = torch.tensor([9.0, 11.0, 7.0, 14.0])

    for beta in [0.5, 1.0, 2.0]:
        for gamma in [0.0, 0.5, 1.0]:
            loss, r_w, r_l, margin = simpo_loss(
                logps_w, logps_l, len_w, len_l, beta=beta, gamma=gamma
            )

            # Analytic reference
            ref_r_w   = beta * logps_w / len_w
            ref_r_l   = beta * logps_l / len_l
            ref_margin = ref_r_w - ref_r_l
            ref_loss  = -F.logsigmoid(ref_margin - gamma).mean()

            assert torch.allclose(loss, ref_loss, atol=2e-2), \
                f"loss mismatch beta={beta} gamma={gamma}: {loss:.4f} vs {ref_loss:.4f}"
            assert torch.allclose(r_w, ref_r_w, atol=2e-2), \
                f"reward_chosen mismatch beta={beta}"
            assert torch.allclose(r_l, ref_r_l, atol=2e-2), \
                f"reward_rejected mismatch beta={beta}"
            assert torch.allclose(margin, ref_margin, atol=2e-2), \
                f"margin mismatch beta={beta}"


# ---------------------------------------------------------------------------
# 3. Monotonicity in γ: larger γ → larger loss (harder constraint)
# ---------------------------------------------------------------------------

def test_larger_gamma_increases_loss():
    """Increasing γ strictly increases the loss because σ(z - γ) decreases in γ."""
    torch.manual_seed(5)
    batch = 8
    logps_w = torch.randn(batch)
    logps_l = torch.randn(batch)
    len_w   = torch.full((batch,), 10.0)
    len_l   = torch.full((batch,), 10.0)

    losses = []
    for gamma in [0.0, 0.5, 1.0, 2.0]:
        loss, _, _, _ = simpo_loss(logps_w, logps_l, len_w, len_l, beta=1.0, gamma=gamma)
        losses.append(loss.item())

    for i in range(len(losses) - 1):
        assert losses[i] < losses[i + 1], \
            f"Expected loss to increase with γ, but losses[{i}]={losses[i]:.4f} " \
            f">= losses[{i+1}]={losses[i+1]:.4f}"


# ---------------------------------------------------------------------------
# 4. Length-normalisation: equal lengths degrade to plain average log-prob
# ---------------------------------------------------------------------------

def test_equal_lengths_reduce_to_average_logp():
    """When |y_w| == |y_l| == L, the SimPO reward difference equals
    β · (logps_w - logps_l) / L — same as dividing both by the common length."""
    torch.manual_seed(6)
    batch = 4
    L = 12
    logps_w = torch.randn(batch)
    logps_l = torch.randn(batch)
    len_both = torch.full((batch,), float(L))

    _, r_w, r_l, margin = simpo_loss(logps_w, logps_l, len_both, len_both,
                                     beta=2.0, gamma=0.0)

    # Expected: rewards are (beta/L) * logps_*
    expected_r_w = 2.0 * logps_w / L
    expected_r_l = 2.0 * logps_l / L
    expected_margin = expected_r_w - expected_r_l

    assert torch.allclose(r_w, expected_r_w, atol=1e-5), "Equal-length chosen reward mismatch"
    assert torch.allclose(r_l, expected_r_l, atol=1e-5), "Equal-length rejected reward mismatch"
    assert torch.allclose(margin, expected_margin, atol=1e-5), "Equal-length margin mismatch"


# ---------------------------------------------------------------------------
# 5. Gradient direction: loss should push chosen reward up
# ---------------------------------------------------------------------------

def test_gradient_pushes_chosen_logp_up():
    """∂L/∂(logps_chosen) < 0  (gradient ascent on reward_chosen)
    and ∂L/∂(logps_rejected) > 0  (gradient descent on reward_rejected)."""
    torch.manual_seed(7)
    batch = 6
    logps_w = torch.randn(batch, requires_grad=True)
    logps_l = torch.randn(batch, requires_grad=True)
    len_w = torch.full((batch,), 10.0)
    len_l = torch.full((batch,), 10.0)

    loss, _, _, _ = simpo_loss(logps_w, logps_l, len_w, len_l, beta=2.0, gamma=0.5)
    loss.backward()

    # To reduce loss, model should increase logps_w and decrease logps_l.
    # Hence ∂L/∂logps_w < 0 and ∂L/∂logps_l > 0.
    assert (logps_w.grad < 0).all(), \
        f"Expected all chosen gradients < 0, got {logps_w.grad.tolist()}"
    assert (logps_l.grad > 0).all(), \
        f"Expected all rejected gradients > 0, got {logps_l.grad.tolist()}"


# ---------------------------------------------------------------------------
# 6. No reference model required (sanity)
# ---------------------------------------------------------------------------

def test_no_reference_model_needed():
    """simpo_loss takes only policy log-probs; no reference arguments exist."""
    import inspect
    sig = inspect.signature(simpo_loss)
    params = set(sig.parameters.keys())
    for bad_name in ("reference", "ref", "logps_ref", "ref_chosen", "ref_rejected"):
        assert bad_name not in params, \
            f"simpo_loss should not have a '{bad_name}' parameter"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("sequence_logp: shape", test_sequence_logp_shape),
        ("sequence_logp: agrees with cross-entropy", test_sequence_logp_agrees_with_cross_entropy),
        ("sequence_logp: gradient flows", test_sequence_logp_gradient_flows),
        ("simpo_loss: shape & finiteness", test_simpo_loss_shape),
        ("simpo_loss: analytic correctness", test_simpo_loss_analytic),
        ("simpo_loss: larger gamma -> larger loss", test_larger_gamma_increases_loss),
        ("simpo_loss: equal lengths -> average logp diff", test_equal_lengths_reduce_to_average_logp),
        ("simpo_loss: gradient direction", test_gradient_pushes_chosen_logp_up),
        ("simpo_loss: no reference model", test_no_reference_model_needed),
    ]

    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASSED  {name}")
        except Exception as exc:
            print(f"FAILED  {name}: {exc}")
            failed += 1

    print()
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} test(s) failed.")
        sys.exit(1)
