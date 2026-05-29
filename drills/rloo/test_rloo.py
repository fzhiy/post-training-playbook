"""Tests for the RLOO drill.  Run: python test_rloo.py  (or pytest)"""
import torch

from from_scratch import rloo_advantages, rloo_loss


def test_leave_one_out_baseline():
    # K=3, hand-computable leave-one-out advantages
    r = torch.tensor([[1.0, 2.0, 3.0]])
    adv = rloo_advantages(r)
    # A0 = 1-(2+3)/2 = -1.5 ; A1 = 2-(1+3)/2 = 0 ; A2 = 3-(1+2)/2 = 1.5
    expected = torch.tensor([[-1.5, 0.0, 1.5]])
    assert torch.allclose(adv, expected, atol=1e-6), adv


def test_advantages_sum_to_zero_per_prompt():
    torch.manual_seed(0)
    r = torch.randn(4, 5)
    adv = rloo_advantages(r)
    # sum_i A_i = S - (K*S - S)/(K-1) = 0 for every prompt
    assert torch.allclose(adv.sum(dim=1), torch.zeros(4), atol=1e-5)


def test_loss_finite_and_grad():
    torch.manual_seed(0)
    logp = torch.randn(3, 4, requires_grad=True)
    r = torch.randn(3, 4)
    loss = rloo_loss(logp, r)
    assert torch.isfinite(loss)
    loss.backward()
    assert logp.grad is not None and torch.isfinite(logp.grad).all()


if __name__ == "__main__":
    test_leave_one_out_baseline()
    test_advantages_sum_to_zero_per_prompt()
    test_loss_finite_and_grad()
    print("all RLOO tests passed ✓")
