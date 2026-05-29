from typing import Optional
import torch
import torch.nn as nn


def compute_ppo_clipped_surrogate_objective(
    action_logprobs: torch.Tensor,
    old_action_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    PPO clipped surrogate objective: L^CLIP = E_t[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

    Args:
        action_logprobs:      log π_θ(a_t|s_t)          shape: (batch_size,)
        old_action_logprobs:  log π_θ_old(a_t|s_t)      shape: (batch_size,)
        advantages:           A_t (GAE or otherwise)     shape: (batch_size,)
        clip_epsilon:         ε in original paper        scalar

    Returns:
        loss:     negative surrogate objective (minimize)  shape: scalar
        metrics:  diagnostic dict with detached tensors
    """
    # ---------- probability ratio r_t = π_θ / π_θ_old ----------
    log_ratio: torch.Tensor = action_logprobs - old_action_logprobs  # (batch_size,)
    ratio: torch.Tensor = torch.exp(log_ratio)                      # (batch_size,)
    # numerically-stable alternative when |log_ratio| is small:
    # ratio = 1.0 + log_ratio  (first-order Taylor — useful for logging)

    # ---------- unclipped surrogate ----------
    surrogate_unclipped: torch.Tensor = ratio * advantages           # (batch_size,)

    # ---------- clipped surrogate ----------
    ratio_clipped: torch.Tensor = torch.clamp(
        ratio,
        min=1.0 - clip_epsilon,
        max=1.0 + clip_epsilon,
    )                                                                # (batch_size,)
    surrogate_clipped: torch.Tensor = ratio_clipped * advantages     # (batch_size,)

    # ---------- element-wise min ----------
    surrogate_per_sample: torch.Tensor = torch.min(
        surrogate_unclipped,
        surrogate_clipped,
    )                                                                # (batch_size,)

    # ---------- scalar loss (mean over batch, negate for minimization) ----------
    loss: torch.Tensor = -surrogate_per_sample.mean()                # scalar

    # ---------- useful diagnostics ----------
    with torch.no_grad():
        approx_kl: torch.Tensor = ((ratio - 1.0) - log_ratio).mean()  # Schulman blog
        clip_frac: torch.Tensor = (
            (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()
        )

    metrics: dict[str, torch.Tensor] = {
        "ratio_mean": ratio.detach().mean(),
        "ratio_std": ratio.detach().std(),
        "surrogate_unclipped_mean": surrogate_unclipped.detach().mean(),
        "surrogate_clipped_mean": surrogate_clipped.detach().mean(),
        "loss": loss.detach(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
    }
    return loss, metrics


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    Generalized Advantage Estimation (GAE-Lambda).

    Args:
        rewards:  r_t              shape: (T,)          time-major
        values:   V(s_t)           shape: (T+1,)        last entry = V(s_T)
        dones:    episode-done flag shape: (T,)          1.0 if terminal, else 0.0
        gamma:    discount factor   scalar
        lam:      GAE lambda        scalar

    Returns:
        advantages: A_t             shape: (T,)
    """
    T: int = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    gae_accum: torch.Tensor = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        delta: torch.Tensor = (
            rewards[t]
            + gamma * values[t + 1] * (1.0 - dones[t])
            - values[t]
        )  # scalar
        gae_accum = delta + gamma * lam * (1.0 - dones[t]) * gae_accum  # scalar
        advantages[t] = gae_accum

    return advantages  # (T,)


# ---------------------------------------------------------------------------
# quick smoke-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    BATCH: int = 64

    # fake old-policy log-probs (frozen)
    old_logprobs: torch.Tensor = torch.randn(BATCH)  # (BATCH,)

    # fake current-policy log-probs (requires grad — this is what we optimize)
    cur_logprobs: torch.Tensor = old_logprobs.clone().detach().requires_grad_(True)  # (BATCH,)

    # fake advantages (whitened)
    raw_adv: torch.Tensor = torch.randn(BATCH)  # (BATCH,)
    advantages: torch.Tensor = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)  # (BATCH,)

    # forward
    loss, metrics = compute_ppo_clipped_surrogate_objective(
        action_logprobs=cur_logprobs,
        old_action_logprobs=old_logprobs,
        advantages=advantages,
        clip_epsilon=0.2,
    )

    # backward — verify gradients flow to cur_logprobs
    loss.backward()
    assert cur_logprobs.grad is not None, "no gradient — check computation graph"

    # print diagnostics
    print("=== PPO Clipped Surrogate — Smoke Test ===")
    for k, v in metrics.items():
        print(f"  {k:>28s}: {v.item():+.6f}")
    print(f"  {'cur_logprobs grad norm':>28s}: {cur_logprobs.grad.norm().item():.6f}")
    print("PASSED")
