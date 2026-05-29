"""Sequence packing, from scratch.

No HuggingFace, no flash-attn kernels — the whole point is to be able to
derive and defend every line in an interview. See README.md for the math
and the stratified follow-up questions.

Requires: torch >= 2.0
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. pack_sequences
# ---------------------------------------------------------------------------

def pack_sequences(
    token_lists: list[list[int]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate variable-length sequences into one flat packed tensor.

    Args:
        token_lists: list of token-id lists, each representing one document.
        pad_id: unused in packing (kept for API symmetry with padded baselines).

    Returns:
        packed_ids   : (T,)   int64  — all tokens concatenated in order.
        cu_seqlens   : (N+1,) int32  — cumulative lengths; cu_seqlens[i] is the
                                        start offset of doc i; cu_seqlens[N] = T.
                                        Mirrors flash-attn's convention.
        position_ids : (T,)   int64  — per-document reset positions (0-based).
        loss_mask    : (T,)   bool   — True for every token that should
                                        contribute to the loss (all of them
                                        in the basic case; callers may refine).

    Why cu_seqlens and not a padding mask?
    - A padding mask of shape (B, L_max) forces all sequences to be padded to
      the longest one; wasted compute is O(L_max - L_i) per sequence.
    - cu_seqlens is O(N) metadata; the packed tensor has exactly T = sum(len)
      tokens — zero padding waste.
    - More importantly, a per-batch padding mask cannot block cross-document
      attention at the *token* level without becoming a full (T, T) mask and
      paying O(T^2) memory. cu_seqlens lets kernels (flash-attn varlen) build
      a block-diagonal attention pattern in O(T) space via index arithmetic.
    """
    packed_ids_list: list[int] = []
    position_ids_list: list[int] = []
    seqlens: list[int] = []

    for tokens in token_lists:
        n = len(tokens)
        packed_ids_list.extend(tokens)
        position_ids_list.extend(range(n))   # reset to 0 for each document
        seqlens.append(n)

    total_len = sum(seqlens)
    packed_ids = torch.tensor(packed_ids_list, dtype=torch.long)
    position_ids = torch.tensor(position_ids_list, dtype=torch.long)

    # cu_seqlens: cumulative sum, starting with 0 (N+1 elements).
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(seqlens, dtype=torch.int32).cumsum(0)

    loss_mask = torch.ones(total_len, dtype=torch.bool)
    return packed_ids, cu_seqlens, position_ids, loss_mask


# ---------------------------------------------------------------------------
# 2. build_block_diagonal_mask
# ---------------------------------------------------------------------------

def build_block_diagonal_mask(
    cu_seqlens: torch.Tensor,   # (N+1,) int32 cumulative lengths
    total_len: int,
    device: torch.device | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Build a (total_len, total_len) boolean attention mask.

    Entry [i, j] is True iff token i is allowed to attend to token j:
      - i and j belong to the same document (block-diagonal constraint).
      - if causal=True, also j <= i within the document.

    Cross-document entries are False, so they receive -inf before softmax
    and contribute exactly zero attention weight.

    Memory: O(T^2) — acceptable for small T in a drill; production uses
    flash-attn varlen kernels that never materialise this matrix.

    Args:
        cu_seqlens : cumulative sequence lengths, shape (N+1,).
        total_len  : T = cu_seqlens[-1].
        device     : target device; defaults to cu_seqlens.device.
        causal     : if True, additionally mask future tokens within each doc.

    Returns:
        mask : (T, T) bool tensor, True = attend, False = block.
    """
    if device is None:
        device = cu_seqlens.device

    # doc_ids[t] = which document token t belongs to.
    doc_ids = torch.zeros(total_len, dtype=torch.long, device=device)
    n_docs = len(cu_seqlens) - 1
    for doc_idx in range(n_docs):
        start = int(cu_seqlens[doc_idx])
        end = int(cu_seqlens[doc_idx + 1])
        doc_ids[start:end] = doc_idx

    # Two tokens may attend only if they share a document.
    # doc_ids[i] == doc_ids[j]  ↔ same document.
    same_doc = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)  # (T, T)

    if causal:
        # Within a document, only attend to earlier or same position.
        # We use global token indices here; since position resets per document,
        # we simply require i >= j (equivalent within the same-doc block).
        tril = torch.ones(total_len, total_len, dtype=torch.bool, device=device).tril()
        mask = same_doc & tril
    else:
        mask = same_doc

    return mask


# ---------------------------------------------------------------------------
# 3. packed_attention_forward
# ---------------------------------------------------------------------------

def packed_attention_forward(
    x: torch.Tensor,           # (T, d_model) — packed token embeddings
    W_q: torch.Tensor,         # (d_model, d_k)
    W_k: torch.Tensor,         # (d_model, d_k)
    W_v: torch.Tensor,         # (d_model, d_v)
    cu_seqlens: torch.Tensor,  # (N+1,) int32
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-head attention over a packed sequence with block-diagonal mask.

    This deliberately avoids nn.Linear and F.scaled_dot_product_attention so
    every operation can be reasoned about line-by-line. In production you would
    call flash_attn.flash_attn_varlen_func here instead.

    Returns:
        output  : (T, d_v) attended values.
        weights : (T, T)   attention weight matrix — cross-doc entries are 0.
    """
    T = x.size(0)
    d_k = W_q.size(1)

    Q = x @ W_q                # (T, d_k)
    K = x @ W_k                # (T, d_k)
    V = x @ W_v                # (T, d_v)

    scores = Q @ K.t() / math.sqrt(d_k)  # (T, T)

    mask = build_block_diagonal_mask(cu_seqlens, T, device=x.device, causal=causal)
    # Mask: True = keep, False = block (fill -inf)
    scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)  # (T, T)
    weights = torch.nan_to_num(weights, nan=0.0)  # guard all-masked rows (e.g. zero-length doc)
    output = weights @ V                     # (T, d_v)
    return output, weights
