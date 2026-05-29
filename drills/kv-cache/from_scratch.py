"""
From-scratch autoregressive decoding with KV-cache in pure PyTorch.
No HuggingFace utilities — every tensor op is explicit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with KV-cache
# ---------------------------------------------------------------------------
class CausalMultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product multi-head attention with a causal mask
    and an explicit key / value cache for autoregressive decoding.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.head_dim: int = d_model // n_heads

        # Linear projections — no bias, matching GPT-style
        self.W_q: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_k: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_v: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_o: nn.Linear = nn.Linear(d_model, d_model, bias=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns a boolean mask of shape (seq_len, seq_len) where
        True means *masked* (i.e. cannot attend).
        Upper-triangular (excluding diagonal) is True.
        """
        # (seq_len, seq_len)
        mask: torch.Tensor = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask  # (seq_len, seq_len)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                                          # (B, T, d_model)
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:        input tensor of shape (B, T_cur, d_model).
                      During *prefill* T_cur = full prompt length;
                      during *decode* T_cur = 1 (single new token).
            kv_cache: None on the first call (prefill); afterwards a
                      tuple (cached_k, cached_v) each of shape
                      (B, n_heads, T_past, head_dim).

        Returns:
            output:    (B, T_cur, d_model)
            new_cache: (cached_k, cached_v) updated with current step,
                       each of shape (B, n_heads, T_past + T_cur, head_dim).
        """
        B: int = x.size(0)
        T_cur: int = x.size(1)

        # --- project to Q, K, V ---
        # (B, T_cur, d_model) -> (B, T_cur, d_model)
        q: torch.Tensor = self.W_q(x)
        k: torch.Tensor = self.W_k(x)
        v: torch.Tensor = self.W_v(x)

        # --- reshape to (B, n_heads, T_cur, head_dim) ---
        q = q.view(B, T_cur, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_cur, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_cur, self.n_heads, self.head_dim).transpose(1, 2)
        # shapes: (B, n_heads, T_cur, head_dim)

        # --- append to KV-cache ---
        if kv_cache is not None:
            cached_k, cached_v = kv_cache  # each (B, n_heads, T_past, head_dim)
            k = torch.cat([cached_k, k], dim=2)  # (B, n_heads, T_past+T_cur, head_dim)
            v = torch.cat([cached_v, v], dim=2)  # (B, n_heads, T_past+T_cur, head_dim)

        new_cache: tuple[torch.Tensor, torch.Tensor] = (k, v)

        # --- compute attention scores ---
        T_kv: int = k.size(2)  # total key/value length so far

        # (B, n_heads, T_cur, head_dim) @ (B, n_heads, head_dim, T_kv)
        # -> (B, n_heads, T_cur, T_kv)
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        # --- apply causal mask ---
        # We need the mask covering T_cur query positions against T_kv key positions.
        # Query positions are the *last* T_cur rows of the full causal matrix.
        causal: torch.Tensor = self._causal_mask(T_kv, device=x.device)
        # Extract rows [T_kv - T_cur : T_kv] → (T_cur, T_kv)
        causal = causal[T_kv - T_cur : T_kv, :]       # (T_cur, T_kv)
        causal = causal.unsqueeze(0).unsqueeze(0)       # (1, 1, T_cur, T_kv)
        scores = scores.masked_fill(causal, float("-inf"))

        # --- softmax + weighted sum ---
        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)  # (B, n_heads, T_cur, T_kv)
        # (B, n_heads, T_cur, T_kv) @ (B, n_heads, T_kv, head_dim)
        # -> (B, n_heads, T_cur, head_dim)
        attn_out: torch.Tensor = torch.matmul(attn_weights, v)

        # --- merge heads ---
        attn_out = attn_out.transpose(1, 2).contiguous()        # (B, T_cur, n_heads, head_dim)
        attn_out = attn_out.view(B, T_cur, self.d_model)        # (B, T_cur, d_model)

        # --- output projection ---
        output: torch.Tensor = self.W_o(attn_out)               # (B, T_cur, d_model)

        return output, new_cache


# ---------------------------------------------------------------------------
# Transformer Decoder Block
# ---------------------------------------------------------------------------
class TransformerDecoderBlock(nn.Module):
    """
    Pre-norm Transformer decoder block:
        x → LayerNorm → CausalMHA → residual
          → LayerNorm → FFN          → residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.attn: CausalMultiHeadAttention = CausalMultiHeadAttention(d_model, n_heads)
        self.ln2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,                                          # (B, T, d_model)
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # --- self-attention sub-layer ---
        residual: torch.Tensor = x                                # (B, T, d_model)
        normed: torch.Tensor = self.ln1(x)                        # (B, T, d_model)
        attn_out, new_cache = self.attn(normed, kv_cache)         # (B, T, d_model)
        x = residual + attn_out                                   # (B, T, d_model)

        # --- feed-forward sub-layer ---
        residual = x                                              # (B, T, d_model)
        normed = self.ln2(x)                                      # (B, T, d_model)
        ffn_out: torch.Tensor = self.ffn(normed)                  # (B, T, d_model)
        x = residual + ffn_out                                    # (B, T, d_model)

        return x, new_cache


# ---------------------------------------------------------------------------
# Minimal GPT-style Decoder-Only Model
# ---------------------------------------------------------------------------
class MiniGPT(nn.Module):
    """Small decoder-only Transformer for demonstrating KV-cache decoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.n_layers: int = n_layers

        self.tok_emb: nn.Embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb: nn.Embedding = nn.Embedding(max_seq_len, d_model)
        self.layers: nn.ModuleList = nn.ModuleList(
            [TransformerDecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f: nn.LayerNorm = nn.LayerNorm(d_model)
        self.lm_head: nn.Linear = nn.Linear(d_model, vocab_size, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,                                     # (B, T)
        kv_caches: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids:      token ids, shape (B, T).
            kv_caches:      list of length n_layers; each element is either
                            None (first forward) or a (K, V) tuple.
            position_offset: added to position indices (equals number of
                             already-cached tokens).

        Returns:
            logits:     (B, T, vocab_size)
            new_caches: updated list of (K, V) caches, one per layer.
        """
        B: int = input_ids.size(0)
        T: int = input_ids.size(1)

        # position ids accounting for previously generated tokens
        positions: torch.Tensor = torch.arange(
            position_offset, position_offset + T, device=input_ids.device
        )  # (T,)

        # (B, T, d_model)
        x: torch.Tensor = self.tok_emb(input_ids) + self.pos_emb(positions)

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            cache_i: Optional[tuple[torch.Tensor, torch.Tensor]] = (
                None if kv_caches is None else kv_caches[i]
            )
            x, new_cache_i = layer(x, cache_i)           # (B, T, d_model)
            new_caches.append(new_cache_i)

        x = self.ln_f(x)                                  # (B, T, d_model)
        logits: torch.Tensor = self.lm_head(x)            # (B, T, vocab_size)

        return logits, new_caches


# ---------------------------------------------------------------------------
# Autoregressive Decoding Loop (greedy + sampling)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: MiniGPT,
    input_ids: torch.Tensor,                                 # (B, prompt_len)
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Autoregressively generate tokens using a KV-cache.

    1.  **Prefill**: run the full prompt through the model in one shot
        to populate the KV-cache.
    2.  **Decode**: feed one token at a time; each layer's cache grows
        by exactly one (K, V) step.

    Returns:
        generated: (B, prompt_len + n_generated) full token sequence.
    """
    model.eval()
    if max_new_tokens <= 0:
        return input_ids
    device: torch.device = input_ids.device
    B: int = input_ids.size(0)

    # ----- prefill phase -----
    # Process entire prompt at once; caches are built for all layers.
    logits, kv_caches = model(input_ids, kv_caches=None, position_offset=0)
    # logits: (B, prompt_len, vocab_size)
    # kv_caches: list[n_layers] of (K, V), each (B, n_heads, prompt_len, head_dim)

    # Next-token distribution from the *last* prompt position
    next_logits: torch.Tensor = logits[:, -1, :]             # (B, vocab_size)
    next_token: torch.Tensor = _sample(next_logits, temperature, top_k)  # (B,)
    generated: torch.Tensor = torch.cat(
        [input_ids, next_token.unsqueeze(1)], dim=1
    )  # (B, prompt_len + 1)

    position: int = input_ids.size(1)  # number of tokens already in cache

    # ----- decode phase (one token at a time) -----
    for _ in range(max_new_tokens - 1):
        # Feed only the newest token; cache supplies history.
        cur_tok: torch.Tensor = next_token.unsqueeze(1)      # (B, 1)

        logits, kv_caches = model(
            cur_tok,
            kv_caches=kv_caches,                              # growing each step
            position_offset=position,                         # absolute position
        )
        # logits: (B, 1, vocab_size)

        next_logits = logits[:, -1, :]                        # (B, vocab_size)
        next_token = _sample(next_logits, temperature, top_k) # (B,)

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        position += 1

        # early stop on EOS (batch-wise shortcut: stop if *all* hit EOS)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated


def _sample(
    logits: torch.Tensor,      # (B, vocab_size)
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply temperature scaling, optional top-k filtering, then sample.
    Returns token indices of shape (B,).
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)                           # greedy

    scaled: torch.Tensor = logits / temperature               # (B, vocab_size)

    if top_k is not None and top_k > 0:
        # Zero out everything outside the top-k per row
        topk_vals, _ = torch.topk(scaled, top_k, dim=-1)      # (B, top_k)
        threshold: torch.Tensor = topk_vals[:, -1:]           # (B, 1) smallest kept value
        scaled = torch.where(
            scaled < threshold,
            torch.full_like(scaled, float("-inf")),
            scaled,
        )

    probs: torch.Tensor = F.softmax(scaled, dim=-1)           # (B, vocab_size)
    next_token: torch.Tensor = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_token                                          # (B,)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    VOCAB: int = 256
    model: MiniGPT = MiniGPT(vocab_size=VOCAB, d_model=64, n_heads=4, d_ff=128, n_layers=2)
    model.eval()

    prompt: torch.Tensor = torch.randint(0, VOCAB, (1, 8))    # (B=1, prompt_len=8)
    print("prompt: ", prompt.tolist())

    out: torch.Tensor = generate(model, prompt, max_new_tokens=32, temperature=0.8, top_k=10)
    print("output: ", out.tolist())
    print("shape:  ", out.shape)  # (1, 8 + up to 32)
