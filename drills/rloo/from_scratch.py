"""REINFORCE Leave-One-Out (RLOO), from scratch.

Critic-free policy gradient: for K sampled responses per prompt, each sample's
baseline is the MEAN reward of the OTHER K-1 samples — no value network.
See README.md for the math and the stratified follow-ups.
"""
from __future__ import annotations

import torch


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-one-out advantages.

    Args:
        rewards: (B, K) — K sampled responses per prompt, B prompts.
    Returns:
        (B, K) advantages: A_i = r_i - mean_{j != i}(r_j)
                                = r_i - (sum_k r_k - r_i) / (K - 1)
    """
    B, K = rewards.shape
    assert K >= 2, "RLOO needs at least 2 samples per prompt"
    total = rewards.sum(dim=1, keepdim=True)          # (B, 1)
    baseline = (total - rewards) / (K - 1)            # (B, K) leave-one-out mean
    return rewards - baseline


def rloo_loss(logprobs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """RLOO policy-gradient loss (to MINIMIZE).

    Args:
        logprobs: (B, K) — summed log pi_theta(response) per sample.
        rewards:  (B, K) — scalar reward per sample.
    Returns:
        scalar loss = - mean( stop_grad(A_i) * logprob_i )
    """
    adv = rloo_advantages(rewards).detach()           # baseline is not differentiated
    return -(adv * logprobs).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    r = torch.randn(2, 4)                              # 2 prompts, 4 samples each
    lp = torch.randn(2, 4, requires_grad=True)
    a = rloo_advantages(r)
    loss = rloo_loss(lp, r)
    print("advantages:\n", a)
    print("per-prompt advantage sums (≈0):", a.sum(dim=1))
    print("loss:", loss.item())
