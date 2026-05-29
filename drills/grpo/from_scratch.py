"""
From-scratch PyTorch implementation of GRPO (Group Relative Policy Optimization).
DeepSeek-Math style: no critic / value function; advantages come from normalising
rewards *within* each prompt-group of G sampled completions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. Group-Relative Advantage
# ---------------------------------------------------------------------------

def group_relative_advantage(
    rewards: Tensor,        # (batch_size,)  – scalar reward per sample
    group_size: int,        # G – completions generated per prompt
) -> Tensor:                # (batch_size,)
    """
    For every prompt we generated `group_size` completions.
    Normalise rewards *within* each group:
        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

    Assumes the first axis is laid out as:
        [prompt_0_comp_0, …, prompt_0_comp_{G-1},
         prompt_1_comp_0, …, prompt_1_comp_{G-1}, …]
    so that reshape(-1, group_size) groups correctly.
    """
    assert rewards.dim() == 1, f"Expected 1-D tensor, got {rewards.dim()}-D"
    assert rewards.shape[0] % group_size == 0, (
        f"batch_size ({rewards.shape[0]}) must be divisible by group_size ({group_size})"
    )

    # (num_prompts, G)
    groups = rewards.reshape(-1, group_size)

    # Per-group mean and std  –  (num_prompts, 1)
    group_mean = groups.mean(dim=-1, keepdim=True)
    group_std  = groups.std(dim=-1, keepdim=True, unbiased=False)

    # Normalised advantages – still (num_prompts, G), then flatten back
    advantages = (groups - group_mean) / (group_std + 1e-8)

    return advantages.reshape(-1)                # (batch_size,)


# ---------------------------------------------------------------------------
# 2. GRPO Clipped Surrogate Loss  (no critic)
# ---------------------------------------------------------------------------

def grpo_loss(
    logp_new:  Tensor,      # (batch_size, seq_len)  – log π_θ(a|s)
    logp_old:  Tensor,      # (batch_size, seq_len)  – log π_θ_old(a|s)
    advantages: Tensor,     # (batch_size,)          – group-normalised rewards
    logp_ref:  Tensor | None = None,  # (batch_size, seq_len)  – log π_ref(a|s)
    eps:  float = 0.2,      # PPO clipping epsilon
    beta: float = 0.04,     # KL penalty coefficient
) -> dict[str, Tensor]:
    """
    GRPO policy loss (clipped surrogate, no value function).

    Returns a dict with keys:
        loss          – total scalar loss to minimise
        pg_loss       – pure policy-gradient (surrogate) component
        kl            – per-token KL(π_θ ‖ π_ref), averaged
        approx_kl     – approximate KL from log-ratio (diagnostic)
        clip_fraction  – fraction of tokens where clipping was active
    """
    # ----- per-token log-ratio & per-sample ratio -----
    # We need a *per-sample* ratio, so aggregate over the sequence dim first.
    # This is the standard practice: sum log-probs over the response tokens.
    # (batch_size,)
    sum_logp_new = logp_new.sum(dim=-1)
    sum_logp_old = logp_old.sum(dim=-1)

    log_ratio = sum_logp_new - sum_logp_old          # (batch_size,)
    ratio     = log_ratio.exp()                       # r_t = π_new / π_old

    # ----- clipped surrogate -----
    pg_loss_1 = -advantages * ratio
    pg_loss_2 = -advantages * ratio.clamp(1.0 - eps, 1.0 + eps)
    pg_loss   = torch.max(pg_loss_1, pg_loss_2).mean()

    # ----- KL penalty (per-token, averaged) -----
    kl = torch.tensor(0.0, device=logp_new.device)
    if logp_ref is not None:
        # Per-token KL: KL(π_θ ‖ π_ref) ≈ logp_new - logp_ref  (k1 estimator, samples from π_θ)
        per_token_kl = logp_new - logp_ref            # (batch_size, seq_len)
        kl = per_token_kl.sum(dim=-1).mean()          # scalar

    # ----- diagnostics -----
    with torch.no_grad():
        approx_kl      = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction   = ((ratio - 1.0).abs() > eps).float().mean()

    loss = pg_loss + beta * kl

    return {
        "loss":          loss,
        "pg_loss":       pg_loss,
        "kl":            kl,
        "approx_kl":     approx_kl,
        "clip_fraction":  clip_fraction,
    }


# ---------------------------------------------------------------------------
# 3. Minimal training-loop demo  (random data, just to show wiring)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyperparameters
    NUM_PROMPTS  = 4       # distinct prompts
    GROUP_SIZE   = 8       # completions per prompt
    BATCH        = NUM_PROMPTS * GROUP_SIZE   # 32
    SEQ_LEN      = 16      # response tokens
    VOCAB        = 256
    LR           = 3e-4
    EPS          = 0.2
    BETA         = 0.04

    # Dummy policy (just a linear head over logits for illustration)
    policy     = torch.nn.Linear(VOCAB, VOCAB)       # maps one-hot → logits
    ref_policy = torch.nn.Linear(VOCAB, VOCAB)        # frozen reference
    ref_policy.load_state_dict(policy.state_dict())   # start identical
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    for step in range(200):
        # ---- fake data ----
        input_ids  = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))    # (B, T)
        one_hot    = F.one_hot(input_ids, VOCAB).float()           # (B, T, V)

        # Forward through policy & reference
        logits_new = policy(one_hot)                               # (B, T, V)
        logits_old = logits_new.detach()                           # frozen old policy
        logits_ref = ref_policy(one_hot)                           # (B, T, V)

        # Log-probs of the *taken* actions
        logp_new = F.log_softmax(logits_new, dim=-1).gather(
            -1, input_ids.unsqueeze(-1)).squeeze(-1)               # (B, T)
        logp_old = F.log_softmax(logits_old, dim=-1).gather(
            -1, input_ids.unsqueeze(-1)).squeeze(-1)               # (B, T)
        logp_ref = F.log_softmax(logits_ref, dim=-1).gather(
            -1, input_ids.unsqueeze(-1)).squeeze(-1)               # (B, T)

        # ---- fake rewards (pretend a reward model scored each completion) ----
        rewards = torch.randn(BATCH)                               # (B,)

        # ---- GRPO ----
        advantages = group_relative_advantage(rewards, GROUP_SIZE)  # (B,)
        info = grpo_loss(logp_new, logp_old, advantages, logp_ref, eps=EPS, beta=BETA)

        optimizer.zero_grad()
        info["loss"].backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"step {step:>3d}  |  loss {info['loss'].item():.4f}  "
                f"pg {info['pg_loss'].item():.4f}  kl {info['kl'].item():.4f}  "
                f"approx_kl {info['approx_kl'].item():.4f}  "
                f"clip_frac {info['clip_fraction'].item():.2%}"
            )
