import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_log_probs_from_logits(
    logits: torch.Tensor,    # shape: (batch, seq_len, vocab)
    labels: torch.Tensor,    # shape: (batch, seq_len)
    ignore_index: int = -100,
) -> torch.Tensor:           # shape: (batch,)
    """Compute per-sequence sum of log-probabilities from model logits.

    Standard next-token cross-entropy: position i predicts token i+1,
    so we shift logits left by one and labels right by one.
    """
    # Shift so that position i predicts token at i+1
    logits = logits[:, :-1, :]        # (batch, seq_len-1, vocab)
    labels = labels[:, 1:]            # (batch, seq_len-1)

    # Per-token log-probabilities (before reduction)
    log_probs = F.log_softmax(logits, dim=-1)                    # (batch, seq_len-1, vocab)
    per_token_log_probs = log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)                       # (batch, seq_len-1, 1)
    ).squeeze(-1)                                                 # (batch, seq_len-1)

    # Mask out padding / ignored positions
    mask = (labels != ignore_index).float()                       # (batch, seq_len-1)
    per_token_log_probs = per_token_log_probs * mask             # (batch, seq_len-1)

    # Sum over sequence length → per-sequence log-prob
    seq_log_probs = per_token_log_probs.sum(dim=-1)              # (batch,)
    return seq_log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,     # shape: (batch,)
    policy_rejected_logps: torch.Tensor,   # shape: (batch,)
    reference_chosen_logps: torch.Tensor,  # shape: (batch,)
    reference_rejected_logps: torch.Tensor,# shape: (batch,)
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the DPO loss from pre-computed log-probabilities.

    The objective for each pair is:
        L = -log σ( β · [ log π_θ(y_w|x) - log π_ref(y_w|x)
                         - log π_θ(y_l|x) + log π_ref(y_l|x) ] )

    Args:
        policy_chosen_logps:      log π_θ(y_w | x)       shape: (batch,)
        policy_rejected_logps:    log π_θ(y_l | x)       shape: (batch,)
        reference_chosen_logps:   log π_ref(y_w | x)     shape: (batch,)
        reference_rejected_logps: log π_ref(y_l | x)     shape: (batch,)
        beta: temperature / KL penalty coefficient.
        label_smoothing: float in [0, 0.5). 0.0 = standard DPO.
            > 0 uses a conservative objective (see Rafailov et al. 2023 §5).

    Returns:
        loss:                     scalar mean loss
        chosen_rewards:           per-sample reward for chosen   (batch,)
        rejected_rewards:         per-sample reward for rejected (batch,)
        reward_margin:            chosen_reward - rejected_reward (batch,)
    """
    # ---- log-ratio under the policy minus log-ratio under the reference ----
    # This is the implicit reward margin:  β · [r(y_w) - r(y_l)]
    # where r(y) = log π_θ(y|x) - log π_ref(y|x)   (up to an x-dependent constant)
    chosen_log_ratio = policy_chosen_logps - reference_chosen_logps       # (batch,)
    rejected_log_ratio = policy_rejected_logps - reference_rejected_logps  # (batch,)
    logits_diff = beta * (chosen_log_ratio - rejected_log_ratio)           # (batch,)

    # ---- DPO loss with optional conservative label smoothing ----
    # When label_smoothing == 0:
    #   L = -log σ(logits_diff)
    # When label_smoothing > 0 (conservative / "robust" DPO):
    #   L = -(1 - ε) · log σ(z) - ε · log σ(-z)
    #     = -(1 - ε) · log σ(z) - ε · log(1 - σ(z))
    if label_smoothing == 0.0:
        losses = -F.logsigmoid(logits_diff)                     # (batch,)
    else:
        # ε-smoothed binary cross-entropy on the "preferred is better" label
        losses = (
            -(1.0 - label_smoothing) * F.logsigmoid(logits_diff)
            - label_smoothing * F.logsigmoid(-logits_diff)
        )                                                        # (batch,)

    loss = losses.mean()                                         # scalar

    # ---- Reward diagnostics (useful for logging / monitoring) ----
    # Under DPO the implicit reward is:  r(y|x) = β · (log π_θ - log π_ref)
    chosen_rewards = beta * chosen_log_ratio                     # (batch,)
    rejected_rewards = beta * rejected_log_ratio                 # (batch,)
    reward_margin = chosen_rewards - rejected_rewards            # (batch,)

    return loss, chosen_rewards, rejected_rewards, reward_margin


class DPOLossModule(nn.Module):
    """Convenience nn.Module wrapper that computes DPO loss end-to-end.

    Feed raw logits and labels for both the policy and reference models
    and get back a scalar loss + diagnostics.
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,       # (batch, seq_len, vocab)
        policy_rejected_logits: torch.Tensor,     # (batch, seq_len, vocab)
        reference_chosen_logits: torch.Tensor,    # (batch, seq_len, vocab)
        reference_rejected_logits: torch.Tensor,  # (batch, seq_len, vocab)
        chosen_labels: torch.Tensor,              # (batch, seq_len)
        rejected_labels: torch.Tensor,            # (batch, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert logits → per-sequence log-probabilities
        policy_chosen_logps = compute_log_probs_from_logits(
            policy_chosen_logits, chosen_labels
        )                                                  # (batch,)
        policy_rejected_logps = compute_log_probs_from_logits(
            policy_rejected_logits, rejected_labels
        )                                                  # (batch,)
        reference_chosen_logps = compute_log_probs_from_logits(
            reference_chosen_logits, chosen_labels
        )                                                  # (batch,)
        reference_rejected_logps = compute_log_probs_from_logits(
            reference_rejected_logits, rejected_labels
        )                                                  # (batch,)

        return dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )


# ---------------------------------------------------------------------------
# Minimal smoke-test / usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    vocab_size = 128

    # Fake logits and labels (padding tokens marked with -100)
    p_chosen_logits   = torch.randn(batch_size, seq_len, vocab_size)
    p_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
    r_chosen_logits   = torch.randn(batch_size, seq_len, vocab_size)
    r_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
    chosen_labels   = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Optionally mask some positions as padding
    chosen_labels[:, -3:]   = -100
    rejected_labels[:, -3:] = -100

    criterion = DPOLossModule(beta=0.1, label_smoothing=0.0)
    loss, ch_rw, rej_rw, margin = criterion(
        p_chosen_logits, p_rejected_logits,
        r_chosen_logits, r_rejected_logits,
        chosen_labels, rejected_labels,
    )

    print(f"loss            = {loss.item():.4f}")
    print(f"chosen reward   = {ch_rw.tolist()}")
    print(f"rejected reward = {rej_rw.tolist()}")
    print(f"margin          = {margin.tolist()}")

    # Verify gradients flow
    loss.backward()
    print("Gradients OK — loss is differentiable.")
