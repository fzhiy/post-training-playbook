"""SFT loss masking and masked cross-entropy, from scratch.

Covers the two core primitives every SFT training loop needs:

1. mask_labels_for_sft  — given token ids and the span(s) where the assistant
   speaks, return a labels tensor where prompt / user positions are replaced with
   ignore_index so they never contribute to the loss.

2. masked_ce_loss  — cross-entropy over a labels tensor that may contain
   ignore_index, with a choice of two normalisation conventions:
       "token"  : divide by the number of non-ignored tokens (standard)
       "sample" : divide by sequence length (some implementations; biases toward
                  longer samples, but keeps the loss scale stable when batching
                  sequences of very different lengths)

The clamp-then-mask pattern in masked_ce_loss is load-bearing:
  torch.gather / F.nll_loss with index -100 raises an index-out-of-bounds error
  on many backends. We clamp labels to [0, V-1] first so the gather is always
  in range, then zero out the contribution for ignored positions with a boolean
  mask. No approximation — the math is identical to the safe path.

See README.md for the math, motivation, and stratified follow-up questions.

Requires: torch >= 2.0
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Label masking
# ---------------------------------------------------------------------------

def mask_labels_for_sft(
    input_ids: torch.Tensor,                    # (L,) int
    assistant_spans: List[Tuple[int, int]],     # [(start, end), ...] half-open [start, end)
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return a labels tensor for SFT: keep only assistant tokens.

    Args:
        input_ids:        1-D token-id tensor of length L.
        assistant_spans:  List of (start, end) half-open intervals identifying
                          assistant turns.  A single-turn prompt would be
                          [(prompt_len, seq_len)].  Multi-turn would have one
                          interval per assistant turn, e.g.
                          [(a0_start, a0_end), (a1_start, a1_end), ...].
        ignore_index:     Value to fill masked positions (default -100, which
                          F.cross_entropy ignores by default).

    Returns:
        labels: (L,) tensor — same dtype as input_ids.  Non-assistant positions
                are set to ignore_index; assistant positions keep their token id.

    Design note:
        SFT trains the model to predict the *next* token.  The labels tensor is
        therefore the same as input_ids (each label[i] is the next-token target
        for position i-1), but with prompt / user turns masked out so the loss
        only flows through assistant output tokens.
    """
    labels = torch.full_like(input_ids, fill_value=ignore_index)
    for start, end in assistant_spans:
        labels[start:end] = input_ids[start:end]
    return labels


# ---------------------------------------------------------------------------
# Masked cross-entropy loss
# ---------------------------------------------------------------------------

def masked_ce_loss(
    logits: torch.Tensor,           # (B, L, V) or (L, V) float
    labels: torch.Tensor,           # (B, L)    or (L,)   int; ignore_index positions are masked
    ignore_index: int = -100,
    reduction: str = "token",       # "token" | "sample"
) -> torch.Tensor:
    """Cross-entropy loss that skips ignore_index positions.

    Two normalisation modes:
        "token"  : loss = sum_of_token_losses / num_non_ignored_tokens
                   Standard HuggingFace Trainer / TRL convention.  Gives equal
                   gradient weight to each *token*, regardless of how many tokens
                   each sample contributes.
        "sample" : loss = sum_of_token_losses / total_tokens (including ignored)
                   Some implementations (e.g. early OPT, certain RL trainers)
                   normalise by sequence length instead.  This biases toward
                   longer assistant turns but keeps the loss scale constant across
                   batches with variable-length padding.

    Clamp-then-mask pattern (see module docstring):
        We clamp labels to [0, V-1] before the gather so no out-of-bounds index
        is ever constructed, then multiply per-token losses by (labels != ignore_index)
        to zero out the masked positions.  This avoids a CUDA index-bounds error
        while remaining numerically identical to just skipping those positions.

    Args:
        logits:     Raw (un-normalised) model output.  Shape (B, L, V) or (L, V).
        labels:     Target token ids.  ignore_index positions are excluded from loss.
        ignore_index: Token id to skip (same convention as F.cross_entropy).
        reduction:  "token" or "sample".

    Returns:
        Scalar loss tensor.
    """
    if reduction not in ("token", "sample"):
        raise ValueError(f"reduction must be 'token' or 'sample', got {reduction!r}")

    # Flatten batch and sequence dims for uniform handling.
    if logits.dim() == 3:
        B, L, V = logits.shape
        logits_2d = logits.reshape(B * L, V)   # (N, V)
        labels_1d = labels.reshape(B * L)       # (N,)
    else:
        logits_2d = logits                      # (N, V)
        labels_1d = labels                      # (N,)
        N, V = logits_2d.shape

    N = logits_2d.size(0)

    # Boolean mask: True where the token should contribute to the loss.
    active = labels_1d != ignore_index          # (N,)

    # Clamp to valid vocab range before gather — see module docstring.
    safe_labels = labels_1d.clamp(min=0)        # (N,)

    # Per-token log-probabilities via log-softmax, then gather the target token.
    log_probs = F.log_softmax(logits_2d, dim=-1)                    # (N, V)
    nll = -log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1) # (N,)

    # Zero out ignored positions using the boolean mask.
    nll = nll * active.float()

    if reduction == "token":
        n_active = active.sum().clamp(min=1)    # avoid divide-by-zero on all-masked input
        return nll.sum() / n_active
    else:  # "sample"
        return nll.sum() / N
