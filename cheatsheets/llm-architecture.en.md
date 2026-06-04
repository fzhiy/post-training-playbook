# LLM Architecture Public Cheat Sheet

> Bilingual Cheat Sheet · Chinese Primary · English Terms Included
> Covers Transformer architecture, Attention variants, positional encoding, KV Cache, decoding strategies, MoE, Scaling Laws, and other core topics

---

## Part 1 · Core Concepts and Derivations

---

### 1.1 Transformer Basic Architecture

A standard **Decoder-only** block (GPT / LLaMA family) stacks the following components:

```
Input → LayerNorm → Multi-Head Self-Attention (causal) → + Residual
      → LayerNorm → Feed-Forward Network (FFN)          → + Residual
```

This is the **Pre-LN** (Pre-Norm) structure, where LayerNorm comes before each sub-layer. The original Transformer paper used **Post-LN** (after the sub-layer); Pre-LN is more stable to train in deep models.

**Encoder** (e.g. BERT) uses **bidirectional attention**; **Encoder-Decoder** (e.g. T5) adds a **cross-attention** layer in the Decoder. The current mainstream LLM architecture is decoder-only, for reasons including: autoregressive pretraining naturally suits generation tasks, better scaling behavior, and simpler architecture.

---

### 1.2 Self-Attention Computation

**Scaled Dot-Product Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**Steps**:

1. Input $X \in \mathbb{R}^{N \times d_{\text{model}}}$ is linearly projected to $Q = XW_Q$, $K = XW_K$, $V = XW_V$, each $\in \mathbb{R}^{N \times d_k}$
2. Compute attention scores $S = QK^\top / \sqrt{d_k} \in \mathbb{R}^{N \times N}$
3. Apply **causal mask** (fill upper triangle with $-\infty$)
4. Softmax to obtain weights $A = \text{softmax}(S)$
5. Output $O = AV \in \mathbb{R}^{N \times d_k}$

**Why scale by $\sqrt{d_k}$**: When $d_k$ is large, the variance of $QK^\top$ is approximately $d_k$ (assuming $Q, K$ components are independent, zero-mean, unit-variance), causing softmax inputs to be large and gradients to approach zero (saturation). Dividing by $\sqrt{d_k}$ normalizes the variance to $\sim 1$.

**Complexity**: time $O(N^2 d_k)$, space $O(N^2 + Nd_k)$ (the attention matrix is the main bottleneck).

---

### 1.3 Multi-Head Attention (MHA)

A single attention head can only learn one type of attention pattern. **Multi-Head Attention** runs $h$ heads in parallel, each learning in a different subspace:

$$
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

$$
\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

where $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$, typically $d_k = d_v = d_{\text{model}} / h$. Total parameter count is comparable to a single head.

---

### 1.4 GQA and MQA

| Method | Number of K/V heads | KV Cache size (relative to MHA) | Representative models |
|:---|:---|:---|:---|
| **MHA** | $h$ (= number of Q heads) | $1\times$ | GPT-3 |
| **MQA** (Multi-Query Attention) | $1$ | $1/h$ | PaLM, Falcon |
| **GQA** (Grouped-Query Attention) | $g$ ($1 < g < h$) | $g/h$ | LLaMA-2/3, Mistral |

**GQA in detail**: The $h$ Query heads are divided into $g$ groups, each group sharing one set of K/V heads. In practice the K/V projection matrices are compressed from $h \times d_k$ to $g \times d_k$, reducing KV Cache at inference to $g/h$.

To **uptrain** from an MHA checkpoint to GQA: average the K/V weights within each group as initialization, then continue training on a small amount of data (the original paper recommends roughly 5% of pretraining tokens).

---

### 1.4b MLA — Multi-head Latent Attention (DeepSeek-V2/V3)

GQA saves cache by reducing the number of KV heads, at the cost of expressivity. **MLA** (DeepSeek-V2, arXiv:2405.04434) takes a different route: it **jointly compresses K/V into a low-rank latent** $c^{KV}_t = W^{DKV} h_t$ ($d_c \ll d_{\text{model}}$), **caches only $c^{KV}_t$**, and up-projects at attention time:

$$k^{C}_t = W^{UK} c^{KV}_t,\qquad v^{C}_t = W^{UV} c^{KV}_t$$

**Absorption trick (key to inference):** the content term of the score can be rewritten as

$$(q^{C}_t)^\top k^{C}_s = (q^{C}_t)^\top W^{UK} c^{KV}_s = \big((W^{UK})^\top q^{C}_t\big)^\top c^{KV}_s$$

so $W^{UK}$ can be **absorbed into the Query projection** — at inference there is no need to reconstruct $k^C$ for every cached token; score directly against $c^{KV}$. Likewise $W^{UV}$ is absorbed into the output projection $W^O$. Thus **only $c^{KV}$** is cached (plus the RoPE key below) and all up-projections fold away.

**Decoupled RoPE:** RoPE is a position-dependent rotation; applying it to $k^C = W^{UK}c^{KV}$ wedges the rotation matrix between $W^{UQ}$ and $W^{UK}$, breaking the absorption above. MLA therefore splits the key into two parts:

- a **content part** $k^C$ (no RoPE, from $c^{KV}$, absorbable);
- a **RoPE part** $k^R_t = \mathrm{RoPE}(W^{KR} h_t)$ (position-carrying, **shared across all heads**, cached separately, small dim e.g. $d_R=64$).

The query is likewise split into $q^C$ (absorbed) and $q^R$ (RoPE-carrying). The final score is

$$S_{ts} = (q^{C}_t)^\top k^{C}_s + (q^{R}_t)^\top k^{R}_s$$

**Payoff:** only $d_c + d_R$ cached per token per layer (DeepSeek-V2: $512+64=576$), far below MHA's $2\,n_h d_h$ — cache on par with GQA (~2.25 groups) while retaining ≈MHA quality.

---

### 1.5 Positional Encoding

#### 1.5.1 Absolute Positional Encoding

**Sinusoidal** (original Transformer):

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

Added directly to the token embeddings. The **Learned** variant replaces this with a trainable embedding table. Drawback: maximum length is fixed, poor extrapolation.

#### 1.5.2 RoPE (Rotary Position Embedding)

Applies rotation matrices to Q and K so that their dot product depends only on the **relative position difference**:

$$
q_m' = R_m\, q_m, \quad k_n' = R_n\, k_n
$$

where $R_m$ is a 2D rotation matrix applied to the pair $(q_{2i}, q_{2i+1})$ at angle $m\theta_i$, with $\theta_i = 10000^{-2i/d}$.

**Key property**:

$$
\langle q_m',\, k_n' \rangle = \langle R_m q_m,\, R_n k_n \rangle = f(q, k,\, m - n)
$$

The attention score depends only on the relative position $m-n$. The rotation preserves vector norms and is numerically stable.

#### 1.5.3 ALiBi (Attention with Linear Biases)

Instead of modifying embeddings, a linear bias is added directly to the attention score:

$$
S_{ij} = q_i^\top k_j - m_h \cdot |i - j|
$$

$m_h$ takes fixed exponentially spaced values per head. Advantages: no extra parameters, good extrapolation. Drawbacks: prefix caching requires dynamically computing bias offsets based on cached tokens' absolute positions at attention time, adding implementation complexity (some inference frameworks do not support this).

---

### 1.6 RoPE Length Extrapolation

At training time the maximum length is $L_{\text{train}}$; at inference on longer sequences, unseen frequency components are encountered, causing distribution shift.

| Method | Core idea | Characteristics |
|:---|:---|:---|
| **Position Interpolation (PI)** | Scale position indices: $m \leftarrow m \cdot L_{\text{train}}/L_{\text{test}}$ | Simple, but loses high-frequency information |
| **NTK-aware Scaling** | Modify the base: $10000 \to \alpha \cdot 10000$, high-frequency components scale less, low-frequency scale more | Better than PI |
| **YaRN** | NTK + mixed interpolation + attention temperature adjustment | One of the current mainstream approaches |
| **Dynamic NTK** | Dynamically adjusts the base at inference time based on actual sequence length | No retraining required |

**Derivation intuition (why change the base, not scale positions):** RoPE's $i$-th dimension pair has frequency $\theta_i = \text{base}^{-2i/d}$ and wavelength $\lambda_i = 2\pi/\theta_i = 2\pi\,\text{base}^{2i/d}$ — low dims are high-frequency (short wavelength, local), high dims low-frequency (long wavelength, global).

- **PI** scales every position by $s = L_{\text{train}}/L_{\text{test}}$, equivalent to compressing **all** frequencies uniformly; but high-freq dims already have short wavelengths, so over-compressing them loses local resolution.
- **NTK-aware** instead enlarges the base (≈ $\text{base}\to\text{base}\cdot s^{d/(d-2)}$) so the **lowest-frequency dim** (longest wavelength, where extrapolation hurts most) is interpolated by exactly $s$ while the **highest-frequency dim is barely changed** — spreading interpolation pressure non-uniformly across dims and preserving local resolution.
- **YaRN** adds a per-wavelength ramp: dims whose wavelength $>$ context window are interpolated (NTK), dims $<$ keep extrapolating, with a linear ramp in between; plus an attention temperature factor $1/\sqrt{t}$ (YaRN sets $\sqrt{1/t}=0.1\ln s + 1$, so the factor grows $>1$ with $s$, sharpening the logits) to compensate for the logit softening / entropy increase from interpolation.

---

### 1.7 KV Cache

**Autoregressive generation**: when generating each new token, the K/V of all previous tokens are already computed. **KV Cache** stores historical K/V in GPU memory; a new token only needs to compute its own Q and attend to the cached K/V.

**Memory estimation** (per token):

$$
\text{KV Cache} = 2 \times L \times n_{\text{kv\_heads}} \times d_k \times \text{bytes\_per\_element}
$$

where $L$ = number of layers, $n_{\text{kv\_heads}}$ = number of K/V heads (with GQA this equals $g$), $d_k$ = head dimension, factor 2 accounts for K and V.

Example: 32 layers, $d_k = 128$, GQA $g = 8$, BF16 (2 bytes) → per token approximately $2 \times 32 \times 8 \times 128 \times 2 = 131072$ bytes $\approx 128$ KB. A sequence of length 4096 → approximately 0.5 GB (single sequence).

---

### 1.8 PagedAttention

Analogous to OS virtual memory paging: KV Cache is divided into fixed-size **blocks** (pages), with a **block table** managing the logical-to-physical mapping, allowing non-contiguous physical memory storage.

**Problem solved**: standard implementations require contiguous GPU memory for KV Cache, causing severe **memory fragmentation** (internal + external), leading to low GPU utilization. PagedAttention substantially improves memory utilization and is the core technology of the **vLLM** inference framework.

Relationship with **Continuous Batching** (iteration-level scheduling): Continuous batching solves request-level scheduling (requests of different lengths do not need to wait for the longest one to finish); PagedAttention solves memory management. The two are complementary.

---

### 1.9 FlashAttention

**Problem**: standard attention requires writing the $N \times N$ score matrix back to HBM (GPU High Bandwidth Memory), which is a memory-bound operation.

**Core idea**: **Tiling + Online Softmax**

1. Divide $Q, K, V$ into small **tiles**, loading them block by block into SRAM (on-chip high-speed cache)
2. Perform tiled attention computation within SRAM
3. Use the **online softmax** algorithm (Milakov & Gimelshein, 2018) to achieve a result **mathematically equivalent** to standard attention without storing the full $N \times N$ matrix

**IO complexity**: reduced from $O(N^2 d)$ to $O(N^2 d^2 / M)$, where $M$ is SRAM size. Memory footprint reduced from $O(N^2)$ to $O(N)$.

**Version history**: FlashAttention-2 optimized warp-level parallelism; FlashAttention-3 targets the Hopper architecture (H100) using asynchronous pipelines and FP8 Tensor Cores.

> **Key point**: FlashAttention is an **exact algorithm**, not an approximation.

### 1.9b Online Softmax recurrence (the heart of FlashAttention)

Tile $Q,K,V$ and process one K/V tile pair at a time in SRAM, never materializing the full $N\times N$ matrix. For a query row processing the $j$-th KV tile (local scores $S_j=qK_j^\top/\sqrt{d}$), maintain the state triple $(m_j,\ell_j,O_j)$:

$$m_j=\max\!\big(m_{j-1},\ \operatorname{rowmax}(S_j)\big)$$

$$\ell_j=e^{\,m_{j-1}-m_j}\,\ell_{j-1}+\operatorname{rowsum}\!\big(e^{\,S_j-m_j}\big)$$

$$O_j=\frac{e^{\,m_{j-1}-m_j}\,\ell_{j-1}\,O_{j-1}+e^{\,S_j-m_j}\,V_j}{\ell_j}$$

with $m_0=-\infty,\ \ell_0=0,\ O_0=0$; after sweeping all tiles, $O$ is the **exact** attention output. Only $m,\ell$ (each $O(N)$) and output $O$ ($O(Nd)$) need to be saved; the key insight is that the $O(N^2)$ attention score matrix is never materialized.

> 📝 **Rescaling trick:** when a new tile's local max exceeds the running $m$, the factor $e^{\,m_{j-1}-m_j}<1$ **scales down** the accumulated $\ell,O$ so the normalizer always matches the global max — exactly equivalent to a single global softmax. Sources: FlashAttention (Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)); the original online-softmax algorithm (Milakov & Gimelshein, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)).

---

### 1.10 FFN Layer

Standard FFN:

$$
\text{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2
$$

where $W_1 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$, $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$, $d_{\text{ffn}} = 4 d_{\text{model}}$ (an empirical choice from the original paper).

**SwiGLU variant** (LLaMA family):

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)\, W_2
$$

$d_{\text{ffn}}$ is typically set to $\frac{8}{3} d_{\text{model}}$ (rounded to a multiple of 128); because there are two gate matrices, the total parameter count is comparable to the standard $4\times$ FFN.

---

### 1.11 MoE (Mixture of Experts)

Replaces the FFN with $E$ **expert** FFNs; each token uses a **router** (linear layer + top-$k$ selection) to select $k$ experts (typically $k=2$), activating only the selected experts:

$$
y = \sum_{i \in \text{top-}k} g_i \cdot \text{Expert}_i(x), \quad g = \text{softmax}(\text{Router}(x))
$$

**Total parameter count** $\approx E \times$ single-expert parameters (much larger than dense), but **FLOPs/token** $\approx (k/E) \times$ dense FLOPs (where "dense" refers to an FFN with the same total parameter count), achieving "large parameter count, low compute."

**Load-balancing loss**: prevents **expert collapse** (all tokens routing to a few experts), typically via an auxiliary loss:

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot p_i
$$

$f_i$ = fraction of tokens assigned to expert $i$, $p_i$ = mean router assignment probability. Encourages $f_i, p_i$ to be uniform.

**Expert Capacity**: each expert has a capacity limit within a batch. Tokens that exceed the limit are **dropped** (that expert is skipped). During training, the capacity factor is typically 1.0–1.25.

### 1.11b MoE extensions: Expert Parallelism / DeepSeek-MoE / aux-loss-free balancing

**Expert Parallelism (EP)** — GShard ([arXiv:2006.16668](https://arxiv.org/abs/2006.16668)): when experts exceed one device's capacity, each device holds $E/P$ experts and tokens go through two **All-to-All** steps (dispatch tokens to the device holding their expert → compute locally → combine back). Overflow beyond the capacity limit: Switch drops it, GShard passes it through via the residual (gating zeroed). Usually combined with TP/DP.

**DeepSeek-MoE** ([arXiv:2401.06066](https://arxiv.org/abs/2401.06066)), two ideas:
- **Fine-grained segmentation**: split $N$ experts into $mN$ smaller ones (FFN inner dim $\div m$) and activate $K\to mK$ — params/FLOPs unchanged, but a larger combinatorial space and finer specialization.
- **Shared-expert isolation**: keep $K_s$ experts always active for all tokens (common knowledge), route the rest top-K (specialized), reducing knowledge redundancy across routed experts.

**Aux-loss-free balancing** ([arXiv:2408.15664](https://arxiv.org/abs/2408.15664); adopted by DeepSeek-V3, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)): give each expert a tunable **bias** $b_i$ **added to the router logit for top-K selection only** (it does not change the gating weight $g_i$); after each step update by load: $b_i-\gamma$ if overloaded, $b_i+\gamma$ if underloaded. Zero gradient interference, no $\alpha$ to tune.

---

### 1.12 Decoding Strategies

| Strategy | Principle | Characteristics |
|:---|:---|:---|
| **Greedy** | Take $\arg\max$ token at each step | Deterministic, prone to repetition degeneration |
| **Beam Search** | Maintain $k$ candidate sequences | Globally better, poor diversity, actually worse for open-ended generation |
| **Top-$k$** | Sample from the top-$k$ tokens by probability | Fixed $k$, does not adapt to the distribution shape |
| **Top-$p$ (Nucleus)** | Sample from the smallest set whose cumulative probability $\geq p$ | Adaptive candidate set size, current mainstream |
| **Temperature** | $\text{softmax}(z/T)$, $T < 1$ more deterministic, $T > 1$ more random | Typically used together with top-$p$ |
| **Min-$p$** | Filter tokens with probability $< p \cdot p_{\max}$ | More adaptive than top-$k$/top-$p$ |

**Speculative Decoding**: a small **draft model** generates $k$ candidate tokens in parallel; a large **verifier** then validates all of them in a single forward pass. Acceptance probability $\min(1,\; p_{\text{verifier}} / p_{\text{draft}})$; on rejection, resample from a corrected distribution. **The output distribution is equivalent to using the verifier directly** (lossless speedup).

**Equivalence proof (why speculative sampling is lossless):** let the draft distribution be $q$ and the target $p$. Propose $x\sim q$, accept with probability $\min(1,\,p(x)/q(x))$; on rejection, resample from $p_{\text{res}}(x)=\dfrac{(p(x)-q(x))_+}{\sum_y (p(y)-q(y))_+}$ (with $(\cdot)_+=\max(0,\cdot)$). The probability of emitting $x$ is:

$$\Pr[\text{out}=x] = \underbrace{q(x)\min\!\big(1,\tfrac{p(x)}{q(x)}\big)}_{\text{propose and accept}} + \underbrace{\Big(1-\textstyle\sum_y \min(q,p)\Big)}_{\Pr[\text{reject}]}\cdot p_{\text{res}}(x)$$

Since $q(x)\min(1,p/q)=\min(q(x),p(x))$ and $\sum_y(p-q)_+ = 1-\sum_y\min(p,q)=\Pr[\text{reject}]$, the second term equals $(p(x)-q(x))_+$. Adding:

$$\min(p,q)+(p-q)_+ = \min(p(x),q(x))+\max(0,\,p(x)-q(x)) = p(x).\quad\blacksquare$$

So the output follows the target $p$ exactly. With $k$ tokens proposed in parallel, apply this per position independently (accept up to the first rejection, then add one corrected sample — so each round emits at least 1 token).

---

### 1.13 Tokenization — BPE

**Byte Pair Encoding**:

1. Initial vocabulary = all bytes (or characters)
2. Iterate: count the most frequent adjacent symbol pair in the training corpus, merge it into a new symbol, add to vocabulary
3. Repeat until the vocabulary reaches the target size

**SentencePiece**: a language-agnostic BPE/Unigram implementation that operates directly on unicode bytes (used by the LLaMA family).

**Effect of vocabulary size**: Embedding layer parameter count = $|V| \times d_{\text{model}}$. A large vocabulary (e.g. LLaMA-3's 128K) is more friendly for multilingual text and code (Chinese tokens are more complete), but increases embedding memory and softmax computation. The LM head typically uses **weight tying** with the embedding to save parameters.

---

### 1.14 Scaling Laws

**Chinchilla Scaling Law** (Hoffmann et al., 2022):

$$
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty
$$

- $N$: number of parameters, $D$: number of training tokens, $L_\infty$: irreducible loss
- Given a FLOPs budget $C$, the optimal $N \propto D$ (both $\propto C^{0.5}$; empirically $D\approx20N$, i.e. ~20 tokens per parameter)

**Practical implication**: earlier large models (e.g. 175B parameters trained on 300B tokens) were "over-parameterized, under-trained." Training a smaller model on more tokens yields a better cost-performance ratio.

**Post-Chinchilla trend**: the LLaMA family and others have driven "inference-efficiency-oriented" training — continuing to train beyond the Chinchilla-optimal token count keeps inference FLOPs constant while performance keeps improving.

---

## Part 2 · PyTorch Code Snippets

---

### 2.1 Scaled Dot-Product Self-Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_k)
    mask:    (1, 1, seq_len, seq_len) or broadcastable
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, H, N, N)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V  # (B, H, N, d_k)
```

---

### 2.2 Multi-Head Attention (from scratch)

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        B, N, _ = x.shape
        # Project and reshape: (B, N, d_model) -> (B, H, N, d_k)
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1) @ V
        
        # Concat heads and project: (B, H, N, d_k) -> (B, N, d_model)
        out = attn.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(out)
```

---

### 2.3 Grouped-Query Attention (GQA)

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_q_heads // n_kv_heads  # Q heads per KV head
        self.d_k = d_model // n_q_heads
        
        self.W_q = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_q_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # Expand KV to match Q heads: (B, n_kv, N, d_k) -> (B, n_q, N, d_k)
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)
        
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1) @ V
        
        out = attn.transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(out)
```

---

### 2.4 Rotary Position Embedding (RoPE)

```python
import torch
import math

def precompute_rope_freqs(d_model, max_len, base=10000.0):
    """Precompute complex rotation frequencies."""
    # θ_i = base^(-2i/d), i = 0, 1, ..., d/2 - 1
    freqs = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
    t = torch.arange(max_len).float()           # (max_len,)
    freqs = torch.outer(t, freqs)                # (max_len, d/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # e^{i·m·θ}

def apply_rope(x, freqs):
    """
    x: (B, H, N, d_k) — real-valued
    freqs: (N, d_k/2) — complex
    """
    # View last dim as complex: (..., d_k) -> (..., d_k/2) complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Rotate: broadcast freqs to (1, 1, N, d_k/2)
    x_rotated = x_complex * freqs.unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(x_rotated).reshape_as(x).to(x.dtype)
```

---

### 2.5 KV Cache Autoregressive Inference

```python
class CausalLMWithKVCache:
    """Simplified autoregressive generation with KV cache."""
    
    def __init__(self, model):
        self.model = model
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_p=0.9):
        self.model.eval()
        kv_cache = None
        generated = prompt_ids.clone()  # (B, prompt_len)
        
        for step in range(max_new_tokens):
            if kv_cache is None:
                # Prefill: process entire prompt
                input_ids = generated
            else:
                # Decode: only new token(s)
                input_ids = generated[:, -1:]
            
            logits, kv_cache = self.model.forward_with_cache(input_ids, kv_cache)
            # logits: (B, seq_len, vocab_size) — take last position
            next_logits = logits[:, -1, :] / temperature
            
            next_token = top_p_sample(next_logits, p=top_p)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == self.model.eos_token_id).all():
                break
        
        return generated
```

---

### 2.6 Top-p (Nucleus) Sampling

```python
def top_p_sample(logits, p=0.9):
    """
    logits: (B, vocab_size)
    Returns: (B, 1) sampled token ids
    """
    probs = torch.softmax(logits, dim=-1)              # (B, V)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)     # (B, V)
    
    # Mask tokens outside the nucleus
    # Shift cumsum right by 1 so the first token above threshold is included
    mask = cum_probs - sorted_probs > p                 # tokens to exclude
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # re-normalize
    
    # Sample from the filtered distribution
    sampled = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1)
    # Map back to original token ids
    return sorted_indices.gather(-1, sampled)
```

---

### 2.7 Causal Mask Generation

```python
def make_causal_mask(seq_len, device):
    """
    Returns a boolean mask where True = allowed, False = masked out.
    Shape: (1, 1, seq_len, seq_len) for broadcasting with (B, H, N, N).
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool().unsqueeze(0).unsqueeze(0)

# Usage:
# mask = make_causal_mask(N, device)
# scores = scores.masked_fill(~mask, float('-inf'))
```

> ⚠️ **All-masked row → softmax NaN:** if an entire row is filled with $-\infty$ (e.g. a pure-padding query row, or a row in sliding-window / block attention that currently has no valid key), the softmax denominator $\sum e^{-\infty}=0$ gives $0/0=\text{NaN}$, which then poisons the whole batch via backprop. A standard causal mask never triggers this (the diagonal $i=i$ is always valid, so every row has ≥1 valid key); padding / sliding-window / block-sparse attention can. Fixes: guarantee at least one valid position per row, use a large finite negative (e.g. `-1e9`) instead of `-inf` for fully-masked rows, or zero out the row after softmax.

---

### 2.8 Simple MoE Layer

```python
import torch
import torch.nn as nn

class SimpleMoE(nn.Module):
    def __init__(self, d_model, d_ffn, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # Expert FFNs (each is a 2-layer MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn, bias=False),
                nn.SiLU(),
                nn.Linear(d_ffn, d_model, bias=False),
            ) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        # x: (B, N, d_model)
        B, N, D = x.shape
        x_flat = x.view(-1, D)                           # (B*N, D)
        
        # Router scores
        router_logits = self.router(x_flat)               # (B*N, E)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # normalize
        
        # Dispatch to experts
        output = torch.zeros_like(x_flat)                 # (B*N, D)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]               # (B*N,)
            weight = topk_probs[:, k]                     # (B*N,)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output
        
        return output.view(B, N, D)
```

> **Note**: The MoE implementation above is a naive version for pedagogical purposes. Production code uses fused kernels and capacity-aware dispatch to avoid Python loops.

---

### 2.9 Transformer Block (Pre-LN Decoder-Only)

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, n_kv_heads=None):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        self.ln1 = nn.RMSNorm(d_model)  # LLaMA uses RMSNorm
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ln2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.SiLU(),
            nn.Linear(d_ffn, d_model, bias=False),
        )
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)   # Pre-LN + residual
        x = x + self.ffn(self.ln2(x))                # Pre-LN + residual
        return x
```

---

## Part 3 · Interview Questions

---

### L1 · Foundational

---

<details>
<summary>Q1. What is the standard structure of a Transformer decoder block?</summary>

**A**: The Pre-LN structure proceeds as: LayerNorm → Multi-Head Causal Self-Attention → Residual → LayerNorm → FFN → Residual. The causal mask fills the upper triangle of the attention score matrix with $-\infty$, ensuring each token can only attend to itself and preceding tokens. The FFN typically expands the dimension by $4\times$ (or $8/3\times$ for SwiGLU) before projecting back.

> **Follow-up**: What are the pros and cons of Pre-LN vs Post-LN? Why do current LLMs generally use Pre-LN?
>
> *Hint: Pre-LN is more stable to train (smoother gradient flow), though some research suggests Post-LN has a slight edge in final performance. Pre-LN became mainstream because Post-LN is unstable in very deep models (>100 layers).*

</details>

---

<details>
<summary>Q2. What is the computation process and complexity of Self-Attention?</summary>

**A**: Input $X$ is projected by $W_Q, W_K, W_V$ to obtain $Q, K, V$; compute $\text{softmax}(QK^\top / \sqrt{d_k})V$. Time complexity $O(N^2 d)$, space complexity $O(N^2)$ (storing the attention matrix). $N$ is the sequence length, $d$ is the head dimension. The $N^2$ term is the bottleneck for long sequence processing.

> **Follow-up**: What is the role of the $\sqrt{d_k}$ scaling factor? What happens if it is omitted?
>
> *Hint: When $d_k$ is large, elements of $QK^\top$ have variance approximately $d_k$, causing softmax saturation and vanishing gradients. Dividing by $\sqrt{d_k}$ normalizes the variance to $\sim 1$.*

</details>

---

<details>
<summary>Q3. What is KV Cache? Why does it accelerate autoregressive generation?</summary>

**A**: During autoregressive generation, when generating each new token, the K/V vectors of all previous tokens are unchanged. KV Cache stores the already-computed K/V in GPU memory; a new token only needs to compute its own Q and attend to the cache, avoiding redundant computation of historical tokens. This reduces the computation per decode step after prefill from $O(n \cdot d)$ (recomputing all K/V) to $O(d)$ (projecting only the new token) + $O(n \cdot d)$ (attention, but no re-projection needed).

> **Follow-up**: How does KV Cache memory usage scale with sequence length and batch size? What compression methods exist?
>
> *Hint: memory = batch\_size × seq\_len × 2 × L × n\_kv\_heads × d\_k × bytes. Compression methods include MQA/GQA, KV Cache quantization (FP8/INT8), token pruning/eviction, etc.*

</details>

---

<details>
<summary>Q4. Compare Greedy, Beam Search, Top-$k$, Top-$p$, and Temperature sampling.</summary>

**A**:
- **Greedy**: take the highest-probability token at each step; deterministic, prone to repetition degeneration
- **Beam Search**: maintain $k$ candidates; suitable for generation with a definite target (e.g. translation); poor diversity in open-ended generation
- **Top-$k$**: sample from only the top-$k$ tokens by probability; fixed $k$ does not adapt to distribution shape
- **Top-$p$**: sample from the smallest set with cumulative probability $\geq p$; adaptive candidate set size
- **Temperature**: $T < 1$ sharpens the distribution (more deterministic), $T > 1$ flattens it (more random); commonly used together with top-$p$

> **Follow-up**: Why does Beam Search perform worse than sampling for open-ended generation?
>
> *Hint: Beam Search optimizes sequence probability, tending to select "safe" high-frequency tokens, resulting in outputs that lack diversity and naturalness. Diversity penalties partially mitigate this.*

</details>

---

<details>
<summary>Q5. How is Causal Mask implemented? Why must decoder-only models use it?</summary>

**A**: In the attention score matrix $S \in \mathbb{R}^{N \times N}$, positions where $j > i$ (the upper triangle) are set to $-\infty$; after softmax these positions have weight 0. This ensures that token $i$, when computing attention, can only see tokens at positions $\leq i$. Decoder-only models train with next-token prediction (computing loss at all positions in parallel); the causal mask ensures that each position's prediction does not leak future information, consistent with autoregressive behavior at inference time.

> **Follow-up**: How does a Prefix LM (e.g. T5 decoder, GLM) mix bidirectional and causal attention within the same sequence?
>
> *Hint: Tokens in the prefix portion use bidirectional attention (no mask) among themselves; the generation portion attends bidirectionally to the prefix but uses a causal mask for itself and subsequent tokens.*

</details>

---

<details>
<summary>Q6. What is the role of Residual Connection in the Transformer?</summary>

**A**: Three core roles:
1. **Gradient flow**: provides a bypass path around sub-layers, alleviating vanishing gradients in deep networks
2. **Feature reuse**: $h_l = h_{l-1} + F(h_{l-1})$, the model can selectively preserve or modify information
3. **Training stability**: combined with Pre-LN, enables stable training of models with hundreds of layers

> **Follow-up**: Are there tools for analyzing information flow in the residual stream across Transformer layers?
>
> *Hint: The Logit Lens method projects intermediate hidden states through the LM head into the vocabulary space, observing how the predicted distribution changes layer by layer.*

</details>

---

<details>
<summary>Q7. Why is the FFN dimension expansion ratio typically $4\times$?</summary>

**A**: $d_{\text{ffn}} = 4d_{\text{model}}$ is an empirical setting from the original Transformer paper, without a strict theoretical justification. Subsequent experiments have shown this ratio works well in most settings. The SwiGLU variant uses $\frac{8}{3}d_{\text{model}}$; because it has two gate matrices, the total parameter count is roughly equal to the standard $4\times$ FFN.

> **Follow-up**: What computational role might the FFN play in a Transformer? Is there research that treats the FFN as "knowledge storage"?
>
> *Hint: Some research (e.g. the key-value memory perspective) views the first layer of the FFN as a key matrix and the second as a value matrix, analogous to the retrieval mechanism of attention.*

</details>

---

### L2 · Intermediate

---

<details>
<summary>Q8. Why do we need Multi-Head Attention? What are its advantages over single-head attention?</summary>

**A**: A single attention head can only learn one type of "attention" pattern. $h$ heads each learn different relational patterns in different subspaces (e.g. syntactic dependencies, coreference, local n-grams, etc.), and are fused by projection through $W^O$ after concatenation. Total parameter count is unchanged (each head has $d_k = d_{\text{model}}/h$), but expressive power is greater.

> **Follow-up**: Do different heads actually learn different patterns? How feasible is head pruning?
>
> *Hint: Michel et al. (2019) found that a large number of heads can be removed at inference time with little performance degradation; Voita et al. (2019) analyzed the functional specialization of different heads via head importance analysis.*

</details>

---

<details>
<summary>Q9. What are the differences between MHA, MQA, and GQA? How is GQA converted from an MHA checkpoint?</summary>

**A**:
- **MHA**: $h$ Q heads each have their own K/V, largest KV Cache
- **MQA**: all Q heads share 1 set of K/V, KV Cache reduced by $h\times$, but may sacrifice quality
- **GQA**: $h$ Q heads divided into $g$ groups, each group sharing 1 set of K/V, a tradeoff between MHA and MQA

Converting from MHA to GQA: average the weights of every $h/g$ K/V heads to initialize the new K/V heads, then continue training (uptrain) on a small amount of data (e.g. 5% of the original pretraining volume).

> **Follow-up**: What attention variant does LLaMA-3 use? How is the GQA group count $g$ chosen?
>
> *Hint: LLaMA-3 uses GQA. Choosing $g$ is a tradeoff between inference efficiency and model quality, typically determined through small-scale experiments.*

</details>

---

<details>
<summary>Q10. What are the characteristics and differences between absolute positional encoding, RoPE, and ALiBi?</summary>

**A**:
- **Absolute positional encoding** (Sinusoidal/Learned): position information is added directly to the embedding, maximum length is fixed, poor extrapolation
- **RoPE**: rotations applied to Q/K so that attention scores depend only on relative position $m-n$; supports length extrapolation (PI/NTK/YaRN); used by the LLaMA family
- **ALiBi**: adds a linear bias $-m_h|i-j|$ to attention scores; no extra parameters, good extrapolation, but prefix caching requires dynamically computing bias offsets, adding implementation complexity (some inference frameworks do not support this)

> **Follow-up**: Does RoPE implement relative encoding by modifying embeddings or by modifying attention scores? What is its mathematical relationship to absolute positional encoding?
>
> *Hint: RoPE applies rotations at the embedding level (on Q/K), but the effect is equivalent to adding a relative-position-dependent bias to the attention score. The difference from absolute positional encoding is that it achieves relative positional encoding through rotational invariance.*

</details>

---

<details>
<summary>Q11. What is the principle behind BPE (Byte Pair Encoding)?</summary>

**A**: The initial vocabulary consists of all bytes/characters. Iteratively, the most frequent adjacent symbol pair in the corpus is merged into a new symbol and added to the vocabulary, until the target size is reached. Advantages: controllable vocabulary size, can handle OOV. Disadvantages: the same word may have multiple tokenizations (case, spacing variants), lower efficiency for non-English languages. SentencePiece is a language-agnostic BPE implementation operating directly on unicode bytes.

> **Follow-up**: How does tokenizer vocabulary size affect the model? What are the considerations behind GPT-2 (50K) vs LLaMA (32K) vs LLaMA-3 (128K)?
>
> *Hint: Effects include embedding layer parameter count ($|V| \times d_{\text{model}}$), information density per token, token efficiency for multilingual text and code, and softmax computation cost. A larger vocabulary is more multilingual-friendly but increases memory.*

</details>

---

<details>
<summary>Q12. What is the core idea of PagedAttention and what problem does it solve?</summary>

**A**: Analogous to OS virtual memory paging: KV Cache is divided into fixed-size blocks, with a block table managing logical-to-physical address mappings, allowing non-contiguous physical memory allocation. This solves the memory fragmentation problem caused by requiring contiguous memory for KV Cache in standard implementations (different request lengths lead to large amounts of internal and external fragmentation), substantially improving GPU memory utilization.

> **Follow-up**: What is the relationship between Continuous Batching and PagedAttention? What layer of problems does each solve?
>
> *Hint: Continuous batching solves the scheduling problem (iteration-level scheduling at the request level, so that the batch does not wait for the longest sequence to finish); PagedAttention solves the memory management problem (efficient KV Cache allocation). The two are complementary.*

</details>

---

<details>
<summary>Q13. What is the basic principle of MoE (Mixture of Experts)?</summary>

**A**: Replaces the FFN with $E$ expert FFNs. Each token uses a router (linear layer + softmax + top-$k$ selection) to select $k$ experts (typically $k=2$), and only the selected experts are activated for computation. Total parameters = $E \times$ expert parameters (much larger than dense), but FLOPs per token $\approx (k/E) \times$ dense FLOPs (where "dense" refers to an FFN with the same total parameter count). A load-balancing loss is added to prevent expert collapse (all tokens routing to a few experts).

> **Follow-up**: How do the memory and compute characteristics of MoE models differ between inference and training? What special requirements does MoE impose on tensor parallelism?
>
> *Hint: At inference, each token activates only $k$ experts, but all expert parameters must be loaded into memory (high memory requirement, low compute). During training, expert-level parallelism or capacity factor control is needed to manage load.*

</details>

---

<details>
<summary>Q14. What is the core conclusion of the Chinchilla Scaling Law?</summary>

**A**: Given a training FLOPs budget $C$, the optimal model parameter count $N$ and training token count $D$ should roughly satisfy $N \propto D$ (both $\propto C^{0.5}$; empirically $D\approx20N$, i.e. ~20 tokens per parameter). That is, many previous large models (e.g. 175B parameters trained on only 300B tokens) were "over-parameterized and under-trained." Training a smaller model on more data yields a better loss per FLOP.

> **Follow-up**: Does the Scaling Law still apply in the post-training (SFT/RLHF) stage?
>
> *Hint: Research on scaling laws for post-training is still ongoing. Some work has explored the relationship between SFT data volume and model size, but RLHF scaling behavior is more complex, influenced by reward model quality, KL constraints, and other factors.*

</details>

---

<details>
<summary>Q15. Estimate the KV Cache memory usage of a Transformer model.</summary>

**A**: Formula: $\text{KV Cache} = 2 \times L \times n_{\text{kv}} \times d_k \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes}$

where factor 2 accounts for K and V, $n_{\text{kv}}$ is the number of K/V heads (with GQA, $< h$), $d_k$ is the head dimension. Example with 32 layers, $n_{\text{kv}} = 8$, $d_k = 128$, BF16: approximately 128 KB per token, a sequence of 4096 tokens is approximately 0.5 GB (single sequence), and a batch of 32 is approximately 16 GB.

> **Follow-up**: What is the effect of KV Cache quantization (e.g. FP8/INT8) on generation quality? What implementation approaches exist?
>
> *Hint: INT8 quantization generally has very little impact on quality; FP8 reduces it further. Inference frameworks such as vLLM and TensorRT-LLM already include built-in KV Cache quantization support.*

</details>

---

<details>
<summary>Q16. What is Sliding Window Attention? What are its pros and cons?</summary>

**A**: Each token only attends to tokens within a distance of $\leq W$ (a local window), with complexity $O(NW)$. With multiple stacked layers the receptive field grows linearly with depth (theoretical receptive field $\approx L \times W$ after $L$ layers), analogous to CNN. Mistral-7B uses $W = 4096$.

Advantages: low complexity; at inference, a **rolling buffer KV Cache** (fixed-size circular buffer) can be used. Disadvantages: a single layer cannot build precise cross-window attention; performs poorly on needle-in-a-haystack retrieval tasks.

> **Follow-up**: How does Mistral's rolling buffer KV cache work?
>
> *Hint: The KV Cache uses a circular buffer of fixed size $W$; new tokens overwrite the oldest ones (at position $i \bmod W$), so memory does not grow with sequence length.*

</details>

---

<details>
<summary>Q17. What is the basic form of Neural Scaling Laws?</summary>

**A**:

$$
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty
$$


$N$ = parameter count, $D$ = number of training tokens, $L_\infty$ = irreducible loss (the entropy of the data itself). $\alpha$ and $\beta$ measure the rate of diminishing returns from parameters and data, respectively. Over a very wide range, loss maintains a power-law relationship with $N$ and $D$.

> **Follow-up**: Why can emergent abilities (e.g. chain-of-thought reasoning) not be directly predicted from scaling laws?
>
> *Hint: Scaling laws describe the smooth decrease in loss (perplexity); emergent abilities are specific task metrics (e.g. accuracy) that suddenly jump from near-random to well-above-random at a certain scale, which may be an artifact of metric choice (Wei et al., 2022; Schaeffer et al., 2023).*

</details>

---

<details>
<summary>Q18. How does the tokenizer vocabulary size affect model capability?</summary>

**A**: Embedding layer parameter count = $|V| \times d_{\text{model}}$ (typically weight-tied with the LM head). Large vocabulary: larger share of embedding parameters, higher softmax computation, but higher information density per token and more friendly for multilingual text and code. Small vocabulary: fewer embedding parameters, but Chinese and similar languages require more tokens to represent the same text (longer sequences, slower inference).

> **Follow-up**: Is there research on the scaling relationship between vocabulary size and model capability? How should the optimal vocabulary size be chosen?
>
> *Hint: Vocabulary size selection requires balancing multilingual coverage, inference efficiency, and embedding parameter overhead. The main motivation for LLaMA-3's expansion to 128K is to improve token efficiency for multilingual text and code.*

</details>

---

### L3 · Advanced

---

<details>
<summary>Q19. What problem does FlashAttention solve? Is it an exact algorithm or an approximation?</summary>

**A**: FlashAttention solves the **memory-bound** problem of standard attention — needing to write the $N \times N$ attention matrix back to HBM. The core is **tiling** + **online softmax**: Q/K/V are loaded in tiles into SRAM, tiled attention computation is performed within SRAM, and via the online softmax algorithm a result **mathematically equivalent** to standard attention is obtained without storing the full $N \times N$ matrix. FlashAttention is an **exact algorithm**, not an approximation. IO complexity is reduced from $O(N^2)$ to $O(N^2 d^2 / M)$ ($M$ is SRAM size), and memory from $O(N^2)$ to $O(N)$.

> **Follow-up**: How does online softmax guarantee mathematical equivalence without storing the full score matrix?
>
> *Hint: Online softmax maintains running-max and running-sum statistics; when processing each new block, it corrects previous blocks' contributions via rescaling, ultimately producing a result identical to global softmax (requiring an extra rescaling pass).*

</details>

---

<details>
<summary>Q20. What are the main methods for RoPE length extrapolation? What is each method's approach?</summary>

**A**:
1. **Position Interpolation (PI)**: linearly scale position indices $m \leftarrow m \cdot L_{\text{train}} / L_{\text{test}}$; simple but loses high-frequency information
2. **NTK-aware Scaling**: modify the base ($10000 \to \alpha \cdot 10000$), high-frequency dimensions scale less, low-frequency dimensions scale more
3. **YaRN**: NTK + non-uniform interpolation + attention temperature adjustment; better performance
4. **Dynamic NTK**: dynamically adjust the base at inference based on actual sequence length; no retraining required

Core challenge: unseen high-frequency rotation angles during training cause attention score distribution shift; extrapolation methods fundamentally balance preserving learned patterns and adapting to new lengths.

> **Follow-up**: What specific improvements does YaRN make over naive PI? Why is attention temperature adjustment needed?
>
> *Hint: YaRN uses different interpolation strategies for high-frequency and low-frequency dimensions, and adds an attention temperature factor $1/\sqrt{t}$ to compensate for changes in attention score magnitude caused by interpolation.*

</details>

---

<details>
<summary>Q21. What is the principle of Speculative Decoding? Why is the output distribution equivalent to using the large model directly?</summary>

**A**: A small draft model generates $k$ candidate tokens in parallel; a large verifier validates all of them in a single forward pass. Validation uses rejection sampling: acceptance probability $\min(1, p_{\text{verifier}}(x) / p_{\text{draft}}(x))$; on rejection, resample from a corrected distribution $\max(0, p_v - p_d)$ normalized. Mathematically it can be shown that the final token distribution at each position is exactly $p_{\text{verifier}}$.

Throughput gain comes from: the draft model is extremely fast (small model), and verifying $k$ tokens requires only one forward pass (parallel), whereas normal generation requires $k$ passes.

> **Follow-up**: Do the draft model and verifier need to share the same architecture? How does self-speculative decoding work?
>
> *Hint: They do not need the same architecture, but the vocabulary must be the same. Self-speculative decoding uses a portion of the same model's layers (early exit) or skips some layers (layer skipping) as the draft, avoiding the need for a separate model.*

</details>

---

<details>
<summary>Q22. What are the main technical approaches for extending LLM context length?</summary>

**A**:
1. **Positional encoding extrapolation**: PI/NTK/YaRN modify RoPE, requiring a small amount of continued training
2. **Sliding Window Attention**: local window $O(NW)$; stacking layers expands the receptive field
3. **Sparse Attention**: Longformer/BigBird mixed global + local + random attention
4. **Ring Attention**: splits long sequences across multiple devices and passes KV through all-to-all communication; theoretically supports unlimited length
5. **Attention sink**: retains KV of the first few tokens (StreamingLLM), addressing the "attention sink" phenomenon in sliding window attention

> **Follow-up**: What bandwidth requirements does Ring Attention have? How does it differ from Sequence Parallelism?
>
> *Hint: Ring Attention requires high-bandwidth interconnects between devices (e.g. NVLink); communication volume is proportional to window size. The difference from traditional Sequence Parallelism is that Ring Attention's partitioning is a pipeline-style transfer along the sequence dimension, rather than simple data parallelism.*

</details>

---

<details>
<summary>Q23. What are Expert Capacity and Token Dropping? What are their effects on training?</summary>

**A**: Each expert has a capacity limit within a batch (capacity factor × tokens/expert). Tokens that exceed the limit are **dropped** — that expert is skipped and the residual or zero output is used directly. During training the capacity factor is typically set to 1.0–1.25; setting it too large wastes computation, setting it too small causes token loss.

Token dropping introduces training noise and degrades gradient quality. Mitigation approaches include: increasing the capacity factor (at the cost of wasted computation), improving router design (e.g. load-balancing loss), and using expert choice routing (experts select tokens rather than tokens selecting experts).

> **Follow-up**: What is the difference between expert choice routing and token choice routing? What are the pros and cons of each?
>
> *Hint: Token choice (standard top-k): each token selects $k$ experts, which may cause load imbalance. Expert choice: each expert selects the top-$k$ tokens, naturally load-balanced, but some tokens may not be selected by any expert.*

</details>

---

<details>
<summary>Q24. How does attention computation with Causal Mask + KV Cache differ from training?</summary>

**A**: **During training**: the full sequence is processed in parallel; an $N \times N$ causal mask matrix is constructed and Q/K/V at all positions are computed in one pass. **During inference** (with KV Cache): the new token's Q is a $(1, d)$ vector, K/V Cache is an $(n, d)$ matrix ($n$ = current sequence length), attention computation is $(1, d) \times (d, n) = (1, n)$, and no explicit causal mask is needed (since K/V does not contain future tokens). The prefill stage (processing the prompt) is similar to training, using a causal mask and parallel computation.

> **Follow-up**: During prefill, if the prompt is very long (e.g. 100K tokens), where is the compute bottleneck? What optimizations exist?
>
> *Hint: The prefill bottleneck is $O(N^2)$ attention computation and $O(N)$ KV Cache writes. Optimizations include chunked prefill (processing in chunks interleaved with decode requests), FlashAttention to reduce memory and compute, and prefix caching (reusing KV Cache for requests with the same prefix).*

</details>

---

<details>
<summary>Q25. Compare the scaling behavior of dense Transformers vs MoE Transformers.</summary>

**A**: Dense models: increasing parameter count = increasing compute (FLOPs/token $\propto N$). MoE models: total parameters $N_{\text{total}} \gg N_{\text{active}}$ (active parameters), FLOPs/token $\propto N_{\text{active}}$. This means MoE can have a larger "knowledge capacity" under the same compute budget.

From a scaling law perspective: MoE loss is mainly determined by active parameters $N_{\text{active}}$ and data volume $D$ (similar to dense), but increasing the number of experts still yields gains when data is abundant (different experts can specialize in different domains of knowledge). However, MoE scaling efficiency is constrained by router quality, load balance, and expert utilization.

> **Follow-up**: What innovations does DeepSeek's MoE design (e.g. DeepSeek-V2/V3) introduce? What are the benefits of separating shared experts from routed experts?
>
> *Hint: DeepSeek-V2/V3 introduces a design with shared experts (all tokens pass through) + routed experts (selected by the router); shared experts handle general capabilities, routed experts handle domain knowledge. There are also fine-grained expert splitting (more but smaller experts) and other design choices.*

</details>

---

> **License**: This cheat sheet is for study and reference purposes only; do not use for commercial purposes. Content is compiled from publicly published research papers and technical blog posts.
>
> **Last Updated**: 2025


## Extended L3

<details>
<summary>Q26. How is the Online Softmax algorithm in FlashAttention implemented? Why can it produce exact results without storing the full attention matrix?</summary>

The core idea of Online Softmax is that softmax can be computed incrementally by maintaining a **running maximum** and a **correction factor**. After dividing Q and K into blocks (tiles), local $QK^\top$ scores are computed block by block; each block yields a local max $m_{\text{local}}$. Comparing with the current global max $m_{\text{global}}$, the previously accumulated $\sum \exp$ is multiplied by a correction factor $e^{m_{\text{global}} - m_{\text{new}}}$, and $m_{\text{global}}$ is then updated. In this way, the partial sum of each block can be correctly rescaled, and the final result is **element-wise identical** to computing the full softmax at once. The output $O$ accumulates in a similarly online weighted manner. Throughout the entire process, the $N \times N$ matrix never fully exists in HBM; only local tiles are processed in SRAM.

> **Follow-up**: If Online Softmax needs to retroactively correct previously accumulated $O$ when processing each tile, is the final output $O$ written back in one shot or corrected multiple times? How does FlashAttention avoid extra memory writes from these corrections?

</details>

<details>
<summary>Q27. Why is Pre-LN more stable than Post-LN for training deep Transformers? Analyze from the perspective of gradient propagation.</summary>

In Post-LN, LayerNorm comes after the residual: $\text{output} = \text{LN}(x + \text{SubLayer}(x))$. In deep networks, gradients on the residual path must pass through the LN at every layer; the LN Jacobian depends on the activation statistics of that layer, causing **inter-layer gradient scale coupling**. As depth increases, gradients in early layers tend to grow or decay exponentially (a variant of gradient explosion/vanishing), requiring careful warmup strategies. In Pre-LN, $\text{output} = x + \text{SubLayer}(\text{LN}(x))$, LN is at the sub-layer input side, and the residual path has **no LN blockage** — gradients can flow directly along the residual (similar to ResNet's identity shortcut), allowing deep models to train stably without warmup. The tradeoff is that Pre-LN may yield slightly lower final performance than carefully tuned Post-LN.

> **Follow-up**: DeepNorm (a variant proposed by Microsoft) stabilizes training on top of Post-LN by adjusting the residual scaling coefficient $\alpha$ and initialization. What is its core mathematical principle?

</details>

<details>
<summary>Q28. What are the different manifestations of router collapse in MoE? What mitigation methods exist beyond auxiliary loss?</summary>

Router collapse takes several forms: (1) **complete collapse** — almost all tokens select only one or two experts, leaving other experts with no gradient updates; (2) **partial imbalance** — most experts are utilized but a few "star experts" are overloaded; (3) **oscillatory cycling** — expert utilization alternates across training steps without converging. The auxiliary loss ($\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum f_i p_i$) is only the most basic approach. Other strategies include: **Expert-Choice Routing** — each expert actively selects its top-$k$ tokens (rather than tokens selecting experts), naturally guaranteeing load balance; **Random routing with noise** — adding tunable noise to router logits to prevent deterministic collapse; **dynamic capacity factor adjustment** — adaptive capacity based on training progress; and using a **smaller learning rate** or separate optimizer for router parameters to avoid excessive fluctuation in routing decisions.

> **Follow-up**: In Expert-Choice Routing, some tokens may be selected by zero or multiple experts simultaneously. What are the effects on model training and inference respectively, and how are they handled?

</details>

<details>
<summary>Q29. How do Copy-on-Write (CoW) and block sharing work in PagedAttention? In which scenarios is block sharing beneficial?</summary>

PagedAttention maintains a **block table** (logical-to-physical page mapping) for each request. When multiple requests need to share the same KV Cache prefix (e.g. a system prompt, or multiple branches of the same parent sequence in beam search), they can **share the same physical block**, avoiding duplicate storage. In this case the block's **reference count** > 1. When a request needs to modify (append new tokens to) a shared block's content, **Copy-on-Write** is triggered: a new physical block is allocated, the original block's content is copied, the original block's reference count is decremented, and the new data is written. This mechanism is most beneficial in: (1) **beam search**, where multiple beams share a prefix; (2) **parallel sampling** (generating multiple responses for the same prompt); (3) **prefix caching** (reusing the system prompt's KV across multi-turn conversations).

> **Follow-up**: How should PagedAttention's block size be chosen? What problems arise from a block size that is too small or too large? What is the typical block size in production systems?

</details>

<details>
<summary>Q30. Beyond GQA/MQA, what other KV Cache compression and eviction strategies exist? What are the tradeoffs of each?</summary>

Main approaches include: (1) **KV Cache quantization** — quantizing K/V from FP16 to INT8/INT4 or lower precision, reducing memory but requiring care about attention precision loss, with V quantization having a larger impact on output quality; (2) **Token eviction** — discarding KV of less important historical tokens based on a policy, e.g. H2O (Heavy Hitter Oracle) retains tokens with high attention scores, StreamingLLM retains "attention sinks" (first few tokens) + a sliding window; (3) **KV merging** — merging K/V vectors of adjacent tokens into one (e.g. via weighted average or PCA), trading precision for space; (4) **cross-layer KV sharing** — adjacent layers share a single set of K/V, reducing total cache volume. Each method is a three-way tradeoff of **precision, memory, and compute overhead**: quantization has the lowest compute overhead but is precision-bounded; eviction strategies are simple but may discard critical information; merging strategies are flexible but introduce additional computation.

> **Follow-up**: StreamingLLM observes the attention sink phenomenon — the first few tokens of the model receive high attention scores regardless of semantic relevance. What do you think is the underlying reason for this?

</details>

<details>
<summary>Q31. Where does SwiGLU's advantage over standard ReLU FFN come from? What is the design motivation for gated activation?</summary>

In the standard ReLU FFN $\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x)$, ReLU is a hard element-wise gate (output is 0 or linear), with a crude information pathway. SwiGLU introduces **multiplicative gating**: $\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2$, where $xW_3$ acts as the gate and $\odot$ is element-wise multiplication. This multiplicative interaction gives the FFN **richer feature combination capability** — the element-wise product of two linear transformations is equivalent to a bilinear operation, able to model more complex feature dependencies. Swish ($x \cdot \sigma(x)$) is itself a smooth, non-monotonic activation that provides smoother gradient flow than the sparsity of ReLU. Empirically, when total parameter count is held constant (reducing $d_{\text{ffn}}$ from $4d$ to approximately $\frac{8}{3}d$ to account for the additional gate matrix $W_3$), SwiGLU consistently outperforms ReLU/GELU FFN on multiple benchmarks.

> **Follow-up**: The GLU family (including ReGLU, GeGLU, SwiGLU) shares the same gating framework. Why was SwiGLU chosen as the mainstream variant (e.g. LLaMA, Mistral family) rather than the others?

</details>

<details>
<summary>Q32. In what scenarios does Speculative Decoding's speedup degrade significantly? What is the basic idea of self-speculative methods?</summary>

Speculative Decoding's speedup depends on the **acceptance rate** (the proportion of draft model predictions accepted by the large model). Scenarios where speedup degrades include: (1) **highly divergent distributions** — when the draft model and verifier distributions differ greatly (e.g. different architecture families), many tokens are rejected and speculation is nearly useless; (2) **highly random generation** — at higher temperature, the probability distribution over correct tokens is flat, making it harder for the draft model to match the verifier's sampling; (3) **structured/constrained output** — e.g. in JSON format generation, certain positions have only a few legal tokens and the draft model may not cover them; (4) **draft model too small** — quality is too low, causing $\mathbb{E}[\text{accepted tokens}]$ to approach 1. **Self-Speculative Decoding** (e.g. Medusa, EAGLE) avoids using an independent draft model and instead makes lightweight predictions on top of the large model itself: for example adding extra **prediction heads** after the final layers (Medusa), or using a small **draft decoder** that reuses the large model's hidden states (EAGLE), eliminating the memory and loading overhead of a separate small model.

> **Follow-up**: The Medusa method adds multiple prediction heads in parallel on top of the large model, each responsible for predicting the $k$-th future token. How are these heads trained? Why can a single head not simply be shared to predict multiple positions?

</details>

<details>
<summary>Q33. In the post-Chinchilla era, why has training models "beyond optimal" (far exceeding the Chinchilla-optimal token count) become mainstream? What is the logic of the "Inference-Optimal" training paradigm?</summary>

The Chinchilla Scaling Law optimizes **training loss given a FLOPs budget**. But in actual deployment, training is a one-time cost while **inference is an ongoing cost** — a model may serve billions of queries. A smaller model has lower FLOPs/decode token at inference, higher throughput, lower latency, and lower deployment cost. Therefore, if a smaller model is trained on far more tokens than the Chinchilla optimum, even though its training loss is worse than a larger model trained with the same FLOPs, the smaller model's **cost-efficiency at inference** may be significantly better — i.e. it can serve more requests under the same inference budget with limited performance loss. The LLaMA family pioneered this approach: models obtained with fewer parameters and more training tokens outperform larger models trained at the Chinchilla-optimal ratio in inference efficiency. The core logic is to shift the optimization objective from "**minimize training loss**" to "**achieve target performance at minimum inference cost**."

> **Follow-up**: If inference cost is also in the optimization objective, how would you define an "inference-optimal" model size choice from the perspective of training token count, model parameter count, and inference hardware constraints?

</details>

## §A Key Papers Timeline

- **2017-06 · Attention Is All You Need** — Vaswani et al., NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — Introduces the pure-attention Transformer: scaled dot-product + multi-head self-attention replace recurrence/convolution, enabling parallelism and direct long-range dependencies — the foundation of every later LLM.

- **2018-05 · Online Softmax** — Milakov & Gimelshein, arXiv preprint. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) — Stable single-pass softmax via running max/sum — the algorithmic basis for FlashAttention's tiling without materializing the N×N matrix.

- **2019-11 · Multi-Query Attention** — Shazeer, arXiv preprint. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) — Shares a single K/V head across all query heads (One Write-Head), shrinking the decode-time KV cache and memory bandwidth at slight quality cost — the ancestor of GQA/MLA.

- **2020-06 · GShard** — Lepikhin et al., ICLR 2021. [arXiv:2006.16668](https://arxiv.org/abs/2006.16668) — Introduces expert parallelism (all-to-all dispatch/combine) and the capacity factor for MoE, establishing the distributed sparse-expert training recipe.

- **2021-01 · Switch Transformers** — Fedus et al., JMLR 2022. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) — Simplifies MoE to top-1 routing + a load-balancing loss, scaling parameters to the trillions at constant FLOPs and establishing the sparse-expert engineering recipe.

- **2021-04 · RoFormer / RoPE** — Su et al., Neurocomputing 2024. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — Rotary position embedding: rotates Q/K in the complex plane by position so attention depends only on relative offset, naturally supporting extrapolation and long context — now the default scheme.

- **2021-08 · ALiBi** — Press et al., ICLR 2022. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) — Replaces position embeddings with a linear-distance attention bias, extrapolating from short training to long inference and inspiring later length-generalization work.

- **2022-03 · Chinchilla** — Hoffmann et al., NeurIPS 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) — Compute-optimal scaling law: under fixed compute, parameters and training tokens should scale at the same rate (empirically ~20 tokens per parameter), correcting the "just add parameters" bias and redefining pretraining budgets.

- **2022-05 · FlashAttention** — Dao et al., NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) — IO-aware exact attention: online-softmax tiling computes entirely in SRAM without materializing the N×N matrix, cutting memory from O(N²) to O(N).

- **2022-11 · Speculative Decoding** — Leviathan et al., ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) — A small draft model guesses several steps in parallel and the large model verifies in one pass; accept-reject sampling keeps the output distribution lossless while cutting decode latency.

- **2023-05 · Grouped-Query Attention** — Ainslie et al., EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) — Interpolates between MHA and MQA: each group of heads shares one K/V, trading off KV-cache size against quality at a tunable knob — the default in Llama-2/3.

- **2023-09 · YaRN** — Peng et al., ICLR 2024. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) — Applies band-wise NTK interpolation to RoPE plus attention-temperature scaling, efficiently extending the context window several-fold with minimal continued training.

- **2024-01 · DeepSeekMoE** — Dai et al., ACL 2024. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) — Fine-grained expert segmentation + shared-expert isolation, improving specialization and cutting routed-expert redundancy at constant compute.

- **2024-05 · DeepSeek-V2 (MLA)** — DeepSeek-AI, arXiv preprint. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) — Multi-head Latent Attention compresses K/V into a low-rank latent cache with an absorption trick and decoupled RoPE, slashing KV cache while preserving multi-head expressivity.

- **2024-08 · Auxiliary-Loss-Free Load Balancing** — Wang et al., arXiv preprint. [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) — Replaces the auxiliary loss with a per-expert bias (affecting routing selection only, not the gating weight) for zero-gradient-interference MoE balancing; adopted by DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)).
