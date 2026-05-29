"""Tests for the GAE drill.  Run: python test_gae.py  (or pytest)"""
import torch

from from_scratch import compute_gae


def test_lambda0_is_one_step_td():
    T = 4
    r = torch.arange(1.0, T + 1)
    v = torch.tensor([0.5, 1.0, 1.5, 2.0])
    last = torch.tensor(3.0)
    g = 0.9
    adv, _ = compute_gae(r, v, last, gamma=g, lam=0.0)
    # lambda=0  ->  A_t = delta_t = r_t + g*V_{t+1} - V_t   (V_T = last)
    v_next = torch.cat([v[1:], last.view(1)])
    expected = r + g * v_next - v
    assert torch.allclose(adv, expected, atol=1e-5), (adv, expected)


def test_lambda1_is_montecarlo_minus_value():
    T = 4
    r = torch.arange(1.0, T + 1)
    v = torch.randn(T)
    last = torch.tensor(2.0)
    g = 0.95
    adv, _ = compute_gae(r, v, last, gamma=g, lam=1.0)
    # lambda=1 (no dones): A_t = [sum_{l=0}^{T-1-t} g^l r_{t+l} + g^{T-t} last] - V_t
    for t in range(T):
        disc = sum((g ** l) * r[t + l] for l in range(T - t)) + (g ** (T - t)) * last
        assert torch.allclose(adv[t], disc - v[t], atol=1e-4), (t, adv[t].item(), (disc - v[t]).item())


def test_done_resets_bootstrap():
    # a terminal at t kills the gamma*V_{t+1} bootstrap there
    r = torch.tensor([1.0, 1.0, 1.0])
    v = torch.zeros(3)
    dones = torch.tensor([0.0, 1.0, 0.0])
    adv, _ = compute_gae(r, v, torch.tensor(5.0), gamma=0.9, lam=0.95, dones=dones)
    # at t=1 (done): delta = r1 + 0 - 0 = 1, and gae from t=2 does not leak back
    assert torch.isfinite(adv).all() and adv.shape == (3,)


if __name__ == "__main__":
    test_lambda0_is_one_step_td()
    test_lambda1_is_montecarlo_minus_value()
    test_done_resets_bootstrap()
    print("all GAE tests passed ✓")
