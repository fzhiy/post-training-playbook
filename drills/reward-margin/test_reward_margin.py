"""Tests for the reward-margin / BT-loss drill.  Run: python test_reward_margin.py"""
import math

import torch

from from_scratch import bt_loss, reward_metrics


def test_equal_rewards_loss_is_log2():
    r = torch.zeros(8)
    # margin 0 -> -log sigmoid(0) = -log(0.5) = log 2
    assert torch.allclose(bt_loss(r, r), torch.tensor(math.log(2.0)), atol=1e-6)


def test_large_margin_loss_near_zero():
    cw = torch.full((4,), 10.0)
    rl = torch.full((4,), -10.0)
    assert bt_loss(cw, rl).item() < 1e-6


def test_metrics_and_grad():
    torch.manual_seed(0)
    cw = torch.randn(16, requires_grad=True)
    rl = torch.randn(16)
    m = reward_metrics(cw, rl)
    assert 0.0 <= m["accuracy"].item() <= 1.0
    assert torch.isfinite(m["loss"])
    m["loss"].backward()
    assert cw.grad is not None and torch.isfinite(cw.grad).all()


if __name__ == "__main__":
    test_equal_rewards_loss_is_log2()
    test_large_margin_loss_near_zero()
    test_metrics_and_grad()
    print("all reward-margin tests passed ✓")
