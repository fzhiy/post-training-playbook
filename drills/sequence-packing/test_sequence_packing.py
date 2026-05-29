"""Correctness tests: sequence packing utilities.

    python test_sequence_packing.py            # plain run
    python -m pytest test_sequence_packing.py  # or via pytest
"""
import torch

from from_scratch import (
    build_block_diagonal_mask,
    pack_sequences,
    packed_attention_forward,
)

TOL = 2e-2   # tolerance used across all floating-point checks


# ---------------------------------------------------------------------------
# pack_sequences
# ---------------------------------------------------------------------------

def test_packed_ids_concat():
    """packed_ids is the exact concatenation of all input token lists."""
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    packed_ids, cu_seqlens, position_ids, loss_mask = pack_sequences(seqs)
    expected = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    assert torch.equal(packed_ids, expected), f"packed_ids mismatch: {packed_ids}"


def test_cu_seqlens_boundaries():
    """cu_seqlens[i] is the start of doc i; cu_seqlens[-1] equals total length."""
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    _, cu_seqlens, _, _ = pack_sequences(seqs)
    # Expected: [0, 3, 5, 9]
    expected = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    assert torch.equal(cu_seqlens, expected), f"cu_seqlens mismatch: {cu_seqlens}"
    assert int(cu_seqlens[-1]) == sum(len(s) for s in seqs)


def test_position_ids_reset_per_doc():
    """position_ids resets to 0 at the start of every new document."""
    seqs = [[10, 20, 30], [40, 50], [60]]
    _, _, position_ids, _ = pack_sequences(seqs)
    # Expected: [0,1,2, 0,1, 0]
    expected = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long)
    assert torch.equal(position_ids, expected), f"position_ids mismatch: {position_ids}"


def test_loss_mask_all_true():
    """By default every packed token participates in the loss."""
    seqs = [[1, 2], [3, 4, 5]]
    _, _, _, loss_mask = pack_sequences(seqs)
    assert loss_mask.all(), "Some tokens unexpectedly masked out of loss"


def test_single_sequence():
    """Edge case: a single document should behave like plain packing."""
    seqs = [[7, 8, 9]]
    packed_ids, cu_seqlens, position_ids, loss_mask = pack_sequences(seqs)
    assert torch.equal(packed_ids, torch.tensor([7, 8, 9], dtype=torch.long))
    assert torch.equal(cu_seqlens, torch.tensor([0, 3], dtype=torch.int32))
    assert torch.equal(position_ids, torch.tensor([0, 1, 2], dtype=torch.long))
    assert loss_mask.all()


def test_empty_pack():
    """Edge case: empty list of sequences returns zero-length tensors."""
    packed_ids, cu_seqlens, position_ids, loss_mask = pack_sequences([])
    assert packed_ids.numel() == 0
    assert torch.equal(cu_seqlens, torch.tensor([0], dtype=torch.int32))
    assert position_ids.numel() == 0
    assert loss_mask.numel() == 0


# ---------------------------------------------------------------------------
# build_block_diagonal_mask
# ---------------------------------------------------------------------------

def test_block_diagonal_same_doc_allowed():
    """Within each document every token can attend to every earlier token."""
    # Three docs: lengths [3, 2, 4]  → T=9
    seqs = [[0, 0, 0], [0, 0], [0, 0, 0, 0]]
    _, cu_seqlens, _, _ = pack_sequences(seqs)
    mask = build_block_diagonal_mask(cu_seqlens, total_len=9, causal=False)

    # Check the 3x3 first-doc block is all True
    assert mask[0:3, 0:3].all(), "First doc block should be all True"
    # Check the 2x2 second-doc block is all True
    assert mask[3:5, 3:5].all(), "Second doc block should be all True"
    # Check the 4x4 third-doc block is all True
    assert mask[5:9, 5:9].all(), "Third doc block should be all True"


def test_cross_doc_blocked():
    """Cross-document entries must be False so they get zero attention weight."""
    seqs = [[0, 0, 0], [0, 0], [0, 0, 0, 0]]
    _, cu_seqlens, _, _ = pack_sequences(seqs)
    mask = build_block_diagonal_mask(cu_seqlens, total_len=9, causal=False)

    # doc0 tokens (0-2) must NOT attend to doc1 tokens (3-4)
    assert not mask[0:3, 3:5].any(), "Doc0→Doc1 cross-doc attention should be blocked"
    # doc1 tokens must NOT attend to doc0 tokens
    assert not mask[3:5, 0:3].any(), "Doc1→Doc0 cross-doc attention should be blocked"
    # doc0 tokens must NOT attend to doc2 tokens
    assert not mask[0:3, 5:9].any(), "Doc0→Doc2 cross-doc attention should be blocked"


def test_causal_within_doc():
    """With causal=True, future tokens within the same doc must be blocked."""
    seqs = [[0, 0, 0, 0]]
    _, cu_seqlens, _, _ = pack_sequences(seqs)
    mask = build_block_diagonal_mask(cu_seqlens, total_len=4, causal=True)

    # Every position in the lower triangle must be allowed (True in mask).
    lower = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    # lower[i,j]=True implies mask[i,j]=True
    assert (mask[lower]).all(), "Lower-triangle positions should all be True in mask"

    # Upper-triangle (future) must all be False
    upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert not (mask & upper).any(), "Upper-triangle (future) should be fully False"


# ---------------------------------------------------------------------------
# packed_attention_forward — cross-doc attention weight is zero
# ---------------------------------------------------------------------------

def test_cross_doc_attention_weight_is_zero():
    """After softmax, attention weights across document boundaries must be 0."""
    torch.manual_seed(42)
    seqs = [[1, 2, 3], [4, 5, 6, 7]]   # doc0: 3 tokens, doc1: 4 tokens
    _, cu_seqlens, _, _ = pack_sequences(seqs)
    T = 7
    d_model, d_k, d_v = 16, 8, 8

    x = torch.randn(T, d_model)
    W_q = torch.randn(d_model, d_k) * 0.1
    W_k = torch.randn(d_model, d_k) * 0.1
    W_v = torch.randn(d_model, d_v) * 0.1

    _, weights = packed_attention_forward(x, W_q, W_k, W_v, cu_seqlens, causal=True)

    # Cross-doc slices: doc0 rows (0:3) attending to doc1 cols (3:7) and vice-versa
    cross_01 = weights[0:3, 3:7]
    cross_10 = weights[3:7, 0:3]

    assert (cross_01 == 0).all(), (
        f"Doc0→Doc1 weights should be exactly 0 (masked with -inf), max={cross_01.abs().max():.6f}"
    )
    assert (cross_10 == 0).all(), (
        f"Doc1→Doc0 weights should be exactly 0 (masked with -inf), max={cross_10.abs().max():.6f}"
    )


def test_weights_sum_to_one_within_doc():
    """Attention weights must sum to 1 over valid (non-masked) key positions."""
    torch.manual_seed(7)
    seqs = [[1, 2, 3], [4, 5]]
    _, cu_seqlens, _, _ = pack_sequences(seqs)
    T = 5
    d_model, d_k, d_v = 8, 4, 4

    x = torch.randn(T, d_model)
    W_q = torch.randn(d_model, d_k) * 0.1
    W_k = torch.randn(d_model, d_k) * 0.1
    W_v = torch.randn(d_model, d_v) * 0.1

    _, weights = packed_attention_forward(x, W_q, W_k, W_v, cu_seqlens, causal=True)

    # For causal packing each row i sums to 1 (softmax over non-masked entries).
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(T), atol=TOL), (
        f"Row sums deviate from 1: {row_sums}"
    )


# ---------------------------------------------------------------------------
# __main__ self-test (no pytest dependency)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_packed_ids_concat()
    test_cu_seqlens_boundaries()
    test_position_ids_reset_per_doc()
    test_loss_mask_all_true()
    test_single_sequence()
    test_empty_pack()
    test_block_diagonal_same_doc_allowed()
    test_cross_doc_blocked()
    test_causal_within_doc()
    test_cross_doc_attention_weight_is_zero()
    test_weights_sum_to_one_within_doc()
    print("all sequence-packing drills passed ✓")
