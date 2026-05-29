"""Generalized Advantage Estimation (GAE), from scratch.

delta_t = r_t + gamma * V_{t+1} - V_t
A_t^GAE(gamma, lambda) = sum_{l>=0} (gamma*lambda)^l * delta_{t+l}
computed by the backward recursion  A_t = delta_t + gamma*lambda * A_{t+1}.
See README.md for the math and follow-ups.
"""
from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,        # (T,)
    values: torch.Tensor,         # (T,)  V(s_t)
    last_value: torch.Tensor,     # scalar tensor  V(s_T), bootstrap
    gamma: float = 0.99,
    lam: float = 0.95,
    dones: torch.Tensor | None = None,   # (T,) 0/1; 1 = terminal at t
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages (T,), returns (T,)=advantages+values)."""
    T = rewards.shape[0]
    if dones is None:
        dones = torch.zeros(T, dtype=rewards.dtype)
    adv = torch.zeros(T, dtype=rewards.dtype)
    gae = torch.zeros((), dtype=rewards.dtype)
    next_value = last_value
    for t in range(T - 1, -1, -1):                      # backward over time
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


if __name__ == "__main__":
    torch.manual_seed(0)
    T = 5
    r = torch.randn(T)
    v = torch.randn(T)
    adv, ret = compute_gae(r, v, last_value=torch.tensor(0.0), gamma=0.99, lam=0.95)
    print("advantages:", adv)
    print("returns   :", ret)
