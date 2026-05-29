"""
From-scratch decoding utilities: temperature scaling, top-k, top-p (nucleus) sampling.
Pure PyTorch — no HuggingFace dependencies.
"""

import torch
import torch.nn.functional as F


def apply_temperature(
    logits: torch.Tensor,  # shape: (batch_size, vocab_size)
    temperature: float,
) -> torch.Tensor:  # shape: (batch_size, vocab_size)
    """Scale logits by 1/temperature before softmax.

    As temperature → 0, distribution approaches argmax (greedy).
    As temperature → ∞, distribution approaches uniform.
    temperature = 1.0 is a no-op.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return logits / temperature


def apply_top_k(
    logits: torch.Tensor,  # shape: (batch_size, vocab_size)
    k: int,
    filter_value: float = -float("inf"),
) -> torch.Tensor:  # shape: (batch_size, vocab_size)
    """Zero out all logits except the k largest per batch element.

    For each row, find the k-th largest value as a threshold,
    then mask everything below it to filter_value.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    vocab_size: int = logits.shape[-1]
    k = min(k, vocab_size)  # clamp so k ≤ vocab_size

    # topk_values shape: (batch_size, k) — the k largest logits per row
    topk_values, _ = torch.topk(logits, k=k, dim=-1)

    # Keep the smallest value among the top-k as the threshold
    # topk_values[:, -1] shape: (batch_size,)
    thresholds: torch.Tensor = topk_values[:, -1].unsqueeze(-1)  # (batch_size, 1)

    # Mask: every logit below the threshold is set to -inf
    # shape: (batch_size, vocab_size)
    mask: torch.Tensor = logits < thresholds
    filtered_logits: torch.Tensor = logits.masked_fill(mask, filter_value)

    return filtered_logits


def apply_top_p(
    logits: torch.Tensor,  # shape: (batch_size, vocab_size)
    top_p: float,
    filter_value: float = -float("inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:  # shape: (batch_size, vocab_size)
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability mass ≥ top_p (after softmax).

    Tokens are sorted by descending probability.  We accumulate until the
    running sum reaches top_p, then mask out everything beyond that point.
    At least min_tokens_to_keep tokens survive.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    # Convert to probabilities; stable sort not needed for softmax
    # probs shape: (batch_size, vocab_size)
    probs: torch.Tensor = F.softmax(logits, dim=-1)

    # Sort probs descending per row
    # sorted_probs shape: (batch_size, vocab_size)
    # sorted_indices shape: (batch_size, vocab_size)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Cumulative sum of sorted probabilities
    # cum_probs shape: (batch_size, vocab_size)
    cum_probs: torch.Tensor = torch.cumsum(sorted_probs, dim=-1)

    # We want to REMOVE tokens whose cumulative sum *before* them already
    # exceeds top_p.  Shift cum_probs right by one so that each position
    # holds the cumulative sum *excluding* itself:
    #   [p1, p2, p3, ...]  →  [0, p1, p1+p2, ...]
    # Zero-pad on the left and drop the last element.
    cum_probs_shifted: torch.Tensor = F.pad(
        cum_probs[:, :-1],         # (batch_size, vocab_size - 1)
        pad=(1, 0),                # pad 1 zero on the left
        value=0.0,
    )                              # → (batch_size, vocab_size)

    # A token is removed if the cumulative sum of all tokens ranked *above*
    # it already meets or exceeds top_p.
    # shape: (batch_size, vocab_size)
    remove_mask: torch.Tensor = cum_probs_shifted >= top_p

    # Guarantee at least min_tokens_to_keep survive: unmask the first
    # min_tokens_to_keep positions in the sorted order.
    remove_mask[:, :min_tokens_to_keep] = False

    # Scatter the mask back into original (unsorted) token order.
    # We need the *inverse* permutation: for each position in the original
    # logit vector, was it marked for removal?
    # torch.argsort on sorted_indices gives us the inverse sort.
    # inv_indices shape: (batch_size, vocab_size)
    inv_indices: torch.Tensor = sorted_indices.argsort(dim=-1)

    # Gather mask into original order
    # original_mask shape: (batch_size, vocab_size)
    original_mask: torch.Tensor = torch.gather(
        remove_mask, dim=-1, index=inv_indices
    )

    filtered_logits: torch.Tensor = logits.masked_fill(original_mask, filter_value)
    return filtered_logits


def sample(
    logits: torch.Tensor,  # shape: (batch_size, vocab_size)
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:  # shape: (batch_size,)
    """Full decode pipeline: temperature → top-k → top-p → multinomial sample.

    Applies filters in the canonical order:
      1. Temperature scaling
      2. Top-k masking  (if k > 0)
      3. Top-p masking  (if p < 1.0)
      4. Sample one token per batch element via multinomial

    Returns the sampled token ids.
    """
    # --- Step 1: temperature ---
    scaled_logits: torch.Tensor = apply_temperature(logits, temperature)
    # scaled_logits: (batch_size, vocab_size)

    # --- Step 2: top-k ---
    if top_k > 0:
        scaled_logits = apply_top_k(scaled_logits, k=top_k)
        # scaled_logits: (batch_size, vocab_size)

    # --- Step 3: top-p ---
    if top_p < 1.0:
        scaled_logits = apply_top_p(scaled_logits, top_p=top_p)
        # scaled_logits: (batch_size, vocab_size)

    # --- Step 4: sample ---
    # Convert filtered logits to probabilities
    probs: torch.Tensor = F.softmax(scaled_logits, dim=-1)
    # probs: (batch_size, vocab_size)

    # Multinomial draws one index per row
    # next_token_ids: (batch_size,)
    next_token_ids: torch.Tensor = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token_ids


# ──────────────────────────────────────────────────────────────────────
# Quick smoke test / demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, vocab_size = 4, 50

    # Fake logits from a model's last layer
    raw_logits: torch.Tensor = torch.randn(batch_size, vocab_size)
    print(f"raw_logits shape: {raw_logits.shape}")

    # Deterministic greedy (temperature → 0 approximated by very small value)
    greedy_ids = sample(raw_logits, temperature=0.001)
    print(f"greedy ids:     {greedy_ids.tolist()}")

    # Top-k only (k=10)
    topk_ids = sample(raw_logits, temperature=1.0, top_k=10)
    print(f"top-k ids (10): {topk_ids.tolist()}")

    # Nucleus only (p=0.9)
    topp_ids = sample(raw_logits, temperature=1.0, top_p=0.9)
    print(f"top-p ids (0.9):{topp_ids.tolist()}")

    # Combined: temperature + top-k + top-p (typical real-world usage)
    combined_ids = sample(raw_logits, temperature=0.7, top_k=50, top_p=0.9)
    print(f"combined ids:   {combined_ids.tolist()}")

    # --- Verify filter shapes are preserved ---
    t = apply_temperature(raw_logits, 1.0)
    k = apply_top_k(raw_logits, k=5)
    p = apply_top_p(raw_logits, top_p=0.5)
    assert t.shape == raw_logits.shape
    assert k.shape == raw_logits.shape
    assert p.shape == raw_logits.shape
    print("All shape checks passed.")
