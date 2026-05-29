"""SimPO (Simple Preference Optimization) loss, from scratch.

Reference: Meng et al. "SimPO: Simple Preference Optimization with a
Reference-Free Reward" (arXiv:2405.14734, NeurIPS 2024).

Key departures from DPO
-----------------------
1. **No reference model** — removes the π_ref forward pass entirely.
2. **Length-normalized reward** — divides the sequence log-prob by |y|
   (number of response tokens), so the reward is the *average* log-prob
   per token, not the raw sum.  Long sequences no longer get an unfair
   advantage just by being longer.
3. **Target-margin γ** — adds a hard margin γ > 0 into the objective so
   the model must separate chosen and rejected rewards by at least γ,
   not just be "a tiny bit better".

Loss (per sample):
    L = -log σ( β/|y_w| · log π(y_w|x)
               - β/|y_l| · log π(y_l|x)
               - γ )

where  β/|y| · log π(y|x)  is the length-normalised implicit reward
and γ is the target margin.

Requires: torch >= 2.0.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: sum log-probs over response tokens
# ---------------------------------------------------------------------------

def sequence_logp(
    logits: torch.Tensor,      # (batch, seq_len, vocab)
    labels: torch.Tensor,      # (batch, seq_len)  — response token ids; padding = -100
    ignore_index: int = -100,
) -> torch.Tensor:             # (batch,) — sum of log-probs over non-padding tokens
    """Compute per-sequence *sum* of log-probabilities for the response tokens.

    Uses standard next-token prediction: position i predicts token i+1,
    so logits are shifted left by one and labels right by one before
    computing log-softmax + gather.

    This is the *unnormalised* quantity Σ log π(y_t | y_<t, x).
    Divide by the response length to get SimPO's normalised reward.
    """
    # Shift for next-token prediction
    logits = logits[:, :-1, :]    # (batch, seq_len-1, vocab)
    labels = labels[:, 1:]        # (batch, seq_len-1)

    log_probs = F.log_softmax(logits, dim=-1)              # (batch, seq_len-1, vocab)
    # Gather the log-prob of the actual next token at each position
    per_token_lp = log_probs.gather(
        dim=-1,
        index=labels.clamp(min=0).unsqueeze(-1),           # clamp before gather (ignore_index is -100)
    ).squeeze(-1)                                          # (batch, seq_len-1)

    mask = (labels != ignore_index).float()               # (batch, seq_len-1)
    per_token_lp = per_token_lp * mask                    # zero out padding positions

    return per_token_lp.sum(dim=-1)                       # (batch,)


# ---------------------------------------------------------------------------
# Core SimPO loss
# ---------------------------------------------------------------------------

def simpo_loss(
    logps_chosen: torch.Tensor,    # (batch,) — Σ log π(y_w_t | ...) for each sample
    logps_rejected: torch.Tensor,  # (batch,) — Σ log π(y_l_t | ...) for each sample
    len_chosen: torch.Tensor,      # (batch,) — number of response tokens in y_w (float or int)
    len_rejected: torch.Tensor,    # (batch,) — number of response tokens in y_l
    beta: float = 2.0,
    gamma: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SimPO loss (reference-free, length-normalised).

    The implicit reward for a response y is:
        r(y | x) = β / |y| · log π(y | x)

    i.e. the length-averaged log-prob scaled by β.  No π_ref needed.

    The loss for a preference pair (y_w, y_l) is:
        L = -log σ( r(y_w | x) - r(y_l | x) - γ )

    Args:
        logps_chosen:   per-sample sum of log π over chosen tokens     (batch,)
        logps_rejected: per-sample sum of log π over rejected tokens   (batch,)
        len_chosen:     number of response tokens in chosen response   (batch,)
        len_rejected:   number of response tokens in rejected response (batch,)
        beta:           reward scaling temperature (paper default ≈ 2.0–2.5)
        gamma:          target margin (paper default ≈ 0.5–1.0)

    Returns:
        loss:               scalar mean loss
        reward_chosen:      length-normalised reward for chosen    (batch,)
        reward_rejected:    length-normalised reward for rejected  (batch,)
        reward_margin:      reward_chosen - reward_rejected        (batch,)
    """
    len_chosen = len_chosen.float()
    len_rejected = len_rejected.float()

    # Length-normalised implicit rewards: β · (1/|y|) · log π(y|x)
    # This is the per-token average log-prob scaled by β — the SimPO reward.
    reward_chosen   = beta * logps_chosen   / len_chosen    # (batch,)
    reward_rejected = beta * logps_rejected / len_rejected  # (batch,)

    # Reward margin (chosen advantage minus rejected advantage, before γ shift)
    reward_margin = reward_chosen - reward_rejected         # (batch,)

    # Loss: push the margin above γ — "chosen must beat rejected by at least γ"
    # -log σ(z - γ)  where z = reward_chosen - reward_rejected
    losses = -F.logsigmoid(reward_margin - gamma)           # (batch,)
    loss = losses.mean()                                    # scalar

    return loss, reward_chosen, reward_rejected, reward_margin


# ---------------------------------------------------------------------------
# Minimal smoke-test / usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    seq_len    = 20
    vocab_size = 256

    # Fake policy logits and token labels (padding positions marked -100)
    policy_chosen_logits   = torch.randn(batch_size, seq_len, vocab_size)
    policy_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
    chosen_labels   = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Simulate variable-length responses with padding at the end
    chosen_labels[:, -4:]   = -100
    rejected_labels[:, -6:] = -100

    # Compute per-sequence sum of log-probs
    lp_chosen   = sequence_logp(policy_chosen_logits,   chosen_labels)
    lp_rejected = sequence_logp(policy_rejected_logits, rejected_labels)

    # Response lengths = number of non-padding tokens in the shifted label
    len_chosen   = (chosen_labels[:, 1:] != -100).sum(dim=-1).float()
    len_rejected = (rejected_labels[:, 1:] != -100).sum(dim=-1).float()

    # Compute SimPO loss
    lp_chosen_grad   = lp_chosen.detach().requires_grad_(True)
    lp_rejected_grad = lp_rejected.detach().requires_grad_(True)

    loss, r_chosen, r_rejected, margin = simpo_loss(
        lp_chosen_grad, lp_rejected_grad, len_chosen, len_rejected,
        beta=2.0, gamma=0.5,
    )

    print(f"loss            = {loss.item():.4f}")
    print(f"reward chosen   = {r_chosen.tolist()}")
    print(f"reward rejected = {r_rejected.tolist()}")
    print(f"margin          = {margin.tolist()}")

    loss.backward()
    assert lp_chosen_grad.grad is not None, "No gradient for chosen log-probs"
    assert lp_rejected_grad.grad is not None, "No gradient for rejected log-probs"
    print("Gradients OK — loss is differentiable.")
