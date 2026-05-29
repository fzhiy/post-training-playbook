"""Bradley-Terry reward-model loss (pairwise preference), from scratch.

For a (chosen, rejected) pair with scalar reward scores r_w, r_l, the
Bradley-Terry model says P(w > l) = sigmoid(r_w - r_l); the reward model is
trained by maximizing that likelihood, i.e. minimizing

    loss = - log sigmoid(r_w - r_l).

See README.md for the math and follow-ups.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bt_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Mean Bradley-Terry pairwise loss. chosen_rewards, rejected_rewards: (B,)."""
    margin = chosen_rewards - rejected_rewards            # (B,)  r_w - r_l
    # logsigmoid is the numerically-stable log(sigmoid(.)) primitive
    return -F.logsigmoid(margin).mean()


def reward_metrics(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> dict:
    """Loss + interpretable diagnostics."""
    margin = chosen_rewards - rejected_rewards
    return {
        "loss": bt_loss(chosen_rewards, rejected_rewards),
        "margin": margin.mean(),                          # average reward gap r_w - r_l
        "accuracy": (margin > 0).float().mean(),          # fraction of pairs ranked correctly
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    cw = torch.randn(8, requires_grad=True)
    rl = torch.randn(8)
    m = reward_metrics(cw, rl)
    print("loss:", m["loss"].item(), "| margin:", m["margin"].item(), "| acc:", m["accuracy"].item())
