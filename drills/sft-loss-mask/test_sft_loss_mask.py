"""Correctness tests for SFT loss masking primitives.

    python test_sft_loss_mask.py            # plain run
    python -m pytest test_sft_loss_mask.py  # or via pytest
"""
import math

import torch
import torch.nn.functional as F

from from_scratch import mask_labels_for_sft, masked_ce_loss


# ---------------------------------------------------------------------------
# mask_labels_for_sft
# ---------------------------------------------------------------------------

def test_single_turn_prompt_masked():
    """Prompt tokens must be ignore_index; assistant tokens must keep their ids."""
    # 10 tokens: positions 0-4 are prompt, 5-9 are assistant response.
    input_ids = torch.arange(10, dtype=torch.long)
    labels = mask_labels_for_sft(input_ids, assistant_spans=[(5, 10)])

    assert (labels[:5] == -100).all(), "prompt positions should be -100"
    assert (labels[5:] == input_ids[5:]).all(), "assistant positions should keep token ids"


def test_single_turn_all_ignored_when_no_span():
    """Empty span list → all positions masked."""
    input_ids = torch.arange(6, dtype=torch.long)
    labels = mask_labels_for_sft(input_ids, assistant_spans=[])
    assert (labels == -100).all()


def test_multi_turn_correct_masking():
    """Multi-turn: interleaved user/assistant turns.

    Layout: [U0: 0-3] [A0: 4-6] [U1: 7-9] [A1: 10-14]
    Only A0 (4,7) and A1 (10,15) should survive.
    """
    L = 15
    input_ids = torch.arange(L, dtype=torch.long)
    assistant_spans = [(4, 7), (10, 15)]
    labels = mask_labels_for_sft(input_ids, assistant_spans=assistant_spans)

    # User / system turns must be masked.
    assert (labels[0:4] == -100).all(),  "U0 should be masked"
    assert (labels[7:10] == -100).all(), "U1 should be masked"

    # Assistant turns must keep original ids.
    assert (labels[4:7] == input_ids[4:7]).all(),   "A0 should be kept"
    assert (labels[10:15] == input_ids[10:15]).all(), "A1 should be kept"


def test_multi_turn_span_count():
    """Number of non-masked tokens equals the sum of assistant span lengths."""
    L = 20
    input_ids = torch.arange(L, dtype=torch.long)
    spans = [(3, 7), (11, 16)]
    labels = mask_labels_for_sft(input_ids, assistant_spans=spans)
    expected_active = sum(e - s for s, e in spans)
    assert (labels != -100).sum().item() == expected_active


# ---------------------------------------------------------------------------
# masked_ce_loss — basic correctness
# ---------------------------------------------------------------------------

def test_ignored_positions_do_not_contribute():
    """-100 positions must contribute zero to the loss.

    Strategy: construct logits where position 0 has a wildly wrong prediction
    (large loss) but label 0 = -100.  The returned loss must equal the loss
    computed only over the non-masked positions.
    """
    torch.manual_seed(42)
    V = 8
    # Position 0: label masked. Position 1: label kept.
    logits = torch.randn(1, 2, V)
    labels = torch.tensor([[-100, 3]])  # only position 1 contributes

    loss_ours = masked_ce_loss(logits, labels, reduction="token")

    # Reference: F.cross_entropy with ignore_index=-100 (token reduction).
    loss_ref = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )

    assert abs(loss_ours.item() - loss_ref.item()) < 2e-2, (
        f"masked loss {loss_ours.item():.6f} != ref {loss_ref.item():.6f}"
    )


def test_token_reduction_equals_pytorch():
    """Token-normalised loss must match F.cross_entropy with ignore_index=-100."""
    torch.manual_seed(7)
    B, L, V = 3, 12, 32
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    # mask ~40% of tokens
    mask = torch.rand(B, L) < 0.4
    labels[mask] = -100

    loss_ours = masked_ce_loss(logits, labels, reduction="token")
    loss_ref = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    assert abs(loss_ours.item() - loss_ref.item()) < 2e-2, (
        f"token loss {loss_ours.item():.6f} != ref {loss_ref.item():.6f}"
    )


def test_two_reductions_differ():
    """Token and sample reductions must differ when ignored tokens are present."""
    torch.manual_seed(99)
    B, L, V = 2, 10, 16
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    # mask first half of each sequence
    labels[:, :5] = -100

    loss_token = masked_ce_loss(logits, labels, reduction="token")
    loss_sample = masked_ce_loss(logits, labels, reduction="sample")

    assert loss_token.item() != loss_sample.item(), (
        "token and sample reductions should differ when some positions are masked"
    )


def test_no_ignored_tokens_reductions_close():
    """When nothing is masked, token and sample reduction differ only by L/n_active = 1."""
    torch.manual_seed(11)
    B, L, V = 2, 8, 16
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))  # no -100

    loss_token = masked_ce_loss(logits, labels, reduction="token")
    loss_sample = masked_ce_loss(logits, labels, reduction="sample")

    # With no masking, both divide by B*L so they should be numerically equal.
    assert abs(loss_token.item() - loss_sample.item()) < 2e-5, (
        "with no masking both reductions should be identical"
    )


def test_all_masked_returns_zero():
    """All-masked input: loss must be 0 (no tokens contribute)."""
    torch.manual_seed(0)
    V = 8
    logits = torch.randn(1, 4, V)
    labels = torch.full((1, 4), -100, dtype=torch.long)

    loss_token = masked_ce_loss(logits, labels, reduction="token")
    loss_sample = masked_ce_loss(logits, labels, reduction="sample")

    assert loss_token.item() == 0.0, "all-masked token loss should be 0"
    assert loss_sample.item() == 0.0, "all-masked sample loss should be 0"


def test_clamp_no_index_error():
    """Clamping labels before gather must not raise an index-out-of-bounds error."""
    V = 5
    logits = torch.randn(1, 3, V)
    # -100 would be out of bounds without the clamp.
    labels = torch.tensor([[-100, 2, -100]])
    loss = masked_ce_loss(logits, labels, reduction="token")
    assert torch.isfinite(loss)


def test_multi_turn_end_to_end():
    """Full pipeline: mask_labels_for_sft -> masked_ce_loss.

    Verify that a loss computed only over assistant spans equals the loss we
    get when we hand-select only those logit positions.
    """
    torch.manual_seed(5)
    L, V = 16, 20
    input_ids = torch.randint(0, V, (L,))
    logits = torch.randn(L, V)

    spans = [(4, 8), (12, 16)]
    labels = mask_labels_for_sft(input_ids, assistant_spans=spans)

    # Our pipeline.
    loss_ours = masked_ce_loss(logits, labels, reduction="token")

    # Reference: manually gather assistant-span tokens and compute CE directly.
    active_logits = torch.cat([logits[s:e] for s, e in spans])   # (n_active, V)
    active_labels = torch.cat([input_ids[s:e] for s, e in spans])
    loss_ref = F.cross_entropy(active_logits, active_labels, reduction="mean")

    assert abs(loss_ours.item() - loss_ref.item()) < 2e-2, (
        f"end-to-end loss {loss_ours.item():.6f} != ref {loss_ref.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_single_turn_prompt_masked()
    test_single_turn_all_ignored_when_no_span()
    test_multi_turn_correct_masking()
    test_multi_turn_span_count()
    test_ignored_positions_do_not_contribute()
    test_token_reduction_equals_pytorch()
    test_two_reductions_differ()
    test_no_ignored_tokens_reductions_close()
    test_all_masked_returns_zero()
    test_clamp_no_index_error()
    test_multi_turn_end_to_end()
    print("all sft-loss-mask drills passed ✓")
