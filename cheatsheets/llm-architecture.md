# LLM Architecture 公开速查手册

> Bilingual Cheat Sheet · 中文为主 · English Terms Included
> 覆盖 Transformer 结构、Attention 变体、位置编码、KV Cache、解码策略、MoE、Scaling Laws 等核心主题

---

## Part 1 · 核心概念与公式推导

---

### 1.1 Transformer 基本架构

标准 **Decoder-only** block（GPT / LLaMA 系列）由以下组件堆叠而成：

```
Input → LayerNorm → Multi-Head Self-Attention (causal) → + Residual
      → LayerNorm → Feed-Forward Network (FFN)          → + Residual
```

以上为 **Pre-LN**（Pre-Norm）结构，即 LayerNorm 在子层之前。原始 Transformer 论文使用 **Post-LN**（子层之后），Pre-LN 在深层模型中训练更稳定。

**Encoder**（如 BERT）使用 **bidirectional attention**；**Encoder-Decoder**（如 T5）在 Decoder 额外加 **cross-attention** 层。当前 LLM 主流为 decoder-only，原因包括：自回归预训练天然适配生成任务、scaling 表现好、架构简洁。

---

### 1.2 Self-Attention 计算

**Scaled Dot-Product Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**步骤**：

1. 输入 $X \in \mathbb{R}^{N \times d_{\text{model}}}$ 经线性投影得到 $Q = XW_Q$，$K = XW_K$，$V = XW_V$，各 $\in \mathbb{R}^{N \times d_k}$
2. 计算注意力分数 $S = QK^\top / \sqrt{d_k} \in \mathbb{R}^{N \times N}$
3. 施加 **causal mask**（上三角填 $-\infty$）
4. Softmax 得权重 $A = \text{softmax}(S)$
5. 输出 $O = AV \in \mathbb{R}^{N \times d_k}$

**$\sqrt{d_k}$ 缩放的作用**：当 $d_k$ 较大时，$QK^\top$ 的方差约为 $d_k$（假设 $Q, K$ 各分量独立、零均值、单位方差），导致 softmax 输入值过大，梯度趋近于零（饱和区）。除以 $\sqrt{d_k}$ 将方差归一化到 $\sim 1$。

**复杂度**：时间 $O(N^2 d_k)$，空间 $O(N^2 + Nd_k)$（attention 矩阵为主要瓶颈）。

---

### 1.3 Multi-Head Attention (MHA)

单个 attention head 只能学习一种关注模式。**Multi-Head Attention** 并行运行 $h$ 个 head，各自在不同子空间学习：

$$
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

$$
\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

其中 $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$，$W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$，通常 $d_k = d_v = d_{\text{model}} / h$。总参数量与单 head 相当。

---

### 1.4 GQA 与 MQA

| 方法 | K/V head 数 | KV Cache 大小（相对 MHA） | 代表模型 |
|:---|:---|:---|:---|
| **MHA** | $h$（= Q head 数） | $1\times$ | GPT-3 |
| **MQA** (Multi-Query Attention) | $1$ | $1/h$ | PaLM, Falcon |
| **GQA** (Grouped-Query Attention) | $g$（$1 < g < h$） | $g/h$ | LLaMA-2/3, Mistral |

**GQA 详解**：$h$ 个 Query head 分为 $g$ 组，每组共享一套 K/V head。实现时将 K/V 投影矩阵从 $h \times d_k$ 压缩为 $g \times d_k$，推理时 KV Cache 缩减为 $g/h$。

从 MHA checkpoint 做 **uptraining** 转换到 GQA：将每组内的 K/V 权重取均值作为初始化，再用少量数据继续训练（原论文建议约 5% 预训练 tokens）。

---

### 1.5 位置编码 (Positional Encoding)

#### 1.5.1 绝对位置编码

**Sinusoidal**（原始 Transformer）：

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

直接加到 token embedding 上。**Learned** 版本用可训练 embedding 表替代。缺点：最大长度固定，外推性差。

#### 1.5.2 RoPE (Rotary Position Embedding)

在 Q、K 上施加旋转矩阵，使点积仅依赖**相对位置差**：

$$
q_m' = R_m\, q_m, \quad k_n' = R_n\, k_n
$$

其中 $R_m$ 是对 $(q_{2i}, q_{2i+1})$ 子对施加角度 $m\theta_i$ 的 2D 旋转矩阵，$\theta_i = 10000^{-2i/d}$。

**关键性质**：

$$
\langle q_m',\, k_n' \rangle = \langle R_m q_m,\, R_n k_n \rangle = f(q, k,\, m - n)
$$

attention score 仅依赖相对位置 $m-n$。旋转操作不改变向量范数，数值稳定。

#### 1.5.3 ALiBi (Attention with Linear Biases)

不修改 embedding，直接在 attention score 上加线性偏置：

$$
S_{ij} = q_i^\top k_j - m_h \cdot |i - j|
$$

$m_h$ 按 head 取固定指数级值。优点：无需额外参数，外推性好。缺点：不支持 **prefix caching** 等优化。

---

### 1.6 RoPE 长度外推

训练时最大长度 $L_{\text{train}}$，推理时更长序列会遇到未见过的频率分量，导致分布偏移。

| 方法 | 核心思路 | 特点 |
|:---|:---|:---|
| **Position Interpolation (PI)** | 缩放位置索引：$m \leftarrow m \cdot L_{\text{train}}/L_{\text{test}}$ | 简单，但高频信息损失 |
| **NTK-aware Scaling** | 修改 base：$10000 \to \alpha \cdot 10000$，高频少缩放、低频多缩放 | 比 PI 效果好 |
| **YaRN** | NTK + 混合插值 + attention temperature 调整 | 当前主流方案之一 |
| **Dynamic NTK** | 推理时根据实际序列长度动态调整 base | 无需重新训练 |

---

### 1.7 KV Cache

**自回归生成**：每生成一个新 token，之前所有 token 的 K/V 已计算过。**KV Cache** 将历史 K/V 存储在 GPU 显存中，新 token 只需计算自身 Q 并与缓存的 K/V 做 attention。

**显存估算**（每 token）：

$$
\text{KV Cache} = 2 \times L \times n_{\text{kv\_heads}} \times d_k \times \text{bytes\_per\_element}
$$

其中 $L$ = 层数，$n_{\text{kv\_heads}}$ = K/V head 数（GQA 时 $= g$），$d_k$ = head dimension，factor 2 对应 K 和 V。

示例：32 层、$d_k = 128$、GQA $g = 8$、BF16（2 bytes）→ 每 token 约 $2 \times 32 \times 8 \times 128 \times 2 = 131072$ bytes $\approx 128$ KB。序列长 4096 → 约 0.5 GB（单序列）。

---

### 1.8 PagedAttention

类比 OS 虚拟内存分页：将 KV Cache 切成固定大小的 **block**（page），用 **block table** 管理逻辑-物理映射，允许非连续物理内存存储。

**解决的问题**：标准实现中 KV Cache 需要连续显存，导致严重的**内存碎片**（internal + external fragmentation），GPU 利用率低。PagedAttention 使显存利用率大幅提升，是 **vLLM** 推理框架的核心技术。

与 **Continuous Batching**（iteration-level scheduling）的关系：Continuous batching 解决请求级调度问题（不同请求长度不同，不必等最长请求完成）；PagedAttention 解决显存管理问题。二者互补。

---

### 1.9 FlashAttention

**问题**：标准 attention 需将 $N \times N$ 的 score 矩阵写回 HBM（GPU 高带宽内存），是 memory-bound 操作。

**核心思路**：**Tiling + Online Softmax**

1. 将 $Q, K, V$ 分成小 **tile**，逐块加载到 SRAM（片上高速缓存）
2. 在 SRAM 内完成分块的 attention 计算
3. 利用 **online softmax** 算法（Milakov & Gimelshein, 2018），在不存储完整 $N \times N$ 矩阵的情况下实现与标准 attention **数学等价**的结果

**IO 复杂度**：从 $O(N^2 d)$ 降至 $O(N^2 d^2 / M)$，其中 $M$ 为 SRAM 大小。显存从 $O(N^2)$ 降至 $O(N)$。

**版本演进**：FlashAttention-2 优化了 warp 级并行；FlashAttention-3 针对 Hopper 架构（H100）利用异步流水线和 FP8 Tensor Core。

> **关键**：FlashAttention 是**精确算法**，不是近似。

---

### 1.10 FFN 层

标准 FFN：

$$
\text{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2
$$

其中 $W_1 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$，$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$，$d_{\text{ffn}} = 4 d_{\text{model}}$（原始论文的经验选择）。

**SwiGLU 变体**（LLaMA 系列）：

$$
\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)\, W_2
$$

$d_{\text{ffn}}$ 通常取 $\frac{8}{3} d_{\text{model}}$（取整到 128 的倍数），因有两个 gate 矩阵，总参数量与标准 $4\times$ FFN 相当。

---

### 1.11 MoE (Mixture of Experts)

将 FFN 替换为 $E$ 个 **expert** FFN，每个 token 经 **router**（线性层 + top-$k$ 选择）选 $k$ 个 expert（通常 $k=2$），只激活选中的 expert：

$$
y = \sum_{i \in \text{top-}k} g_i \cdot \text{Expert}_i(x), \quad g = \text{softmax}(\text{Router}(x))
$$

**总参数量** ≈ $E \times$ 单 expert 参数（远大于 dense），但 **FLOPs/token** ≈ $(k/E) \times$ dense FLOPs，实现"大参数量、低计算量"。

**负载均衡 loss**：防止 **expert collapse**（所有 token 涌向少数 expert），通常加辅助 loss：

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot p_i
$$

$f_i$ = 分配到 expert $i$ 的 token 比例，$p_i$ = router 分配概率均值。鼓励 $f_i, p_i$ 均匀。

**Expert Capacity**：每个 expert 在一个 batch 内有容量上限。超出的 token 被 **drop**（跳过该 expert）。训练时 capacity factor 通常 1.0–1.25。

---

### 1.12 解码策略 (Decoding Strategies)

| 策略 | 原理 | 特点 |
|:---|:---|:---|
| **Greedy** | 每步取 $\arg\max$ token | 确定性，易重复退化 |
| **Beam Search** | 维护 $k$ 条候选序列 | 全局更优，多样性差，open-ended 生成效果反而更差 |
| **Top-$k$** | 从概率前 $k$ 的 token 中采样 | $k$ 固定，不自适应分布形状 |
| **Top-$p$ (Nucleus)** | 从累积概率 $\geq p$ 的最小集合中采样 | 自适应候选集大小，当前主流 |
| **Temperature** | $\text{softmax}(z/T)$，$T < 1$ 更确定，$T > 1$ 更随机 | 通常与 top-$p$ 联用 |
| **Min-$p$** | 过滤概率 $< p \cdot p_{\max}$ 的 token | 比 top-$k$/top-$p$ 更自适应 |

**Speculative Decoding**：用小 **draft model** 并行生成 $k$ 个候选 token，再用大 **verifier** 一次性验证。接受概率 $\min(1,\; p_{\text{verifier}} / p_{\text{draft}})$，拒绝时从修正分布重新采样。**输出分布与直接用 verifier 等价**（无损加速）。

---

### 1.13 Tokenization — BPE

**Byte Pair Encoding**：

1. 初始词表 = 所有字节（或字符）
2. 迭代：统计训练语料中频率最高的相邻符号对，合并为新符号，加入词表
3. 重复直到词表达到目标大小

**SentencePiece**：语言无关的 BPE/Unigram 实现，直接在 unicode 字节上操作（LLaMA 系列使用）。

**词表大小影响**：Embedding 层参数量 = $|V| \times d_{\text{model}}$。大词表（如 LLaMA-3 的 128K）对多语言和代码更友好（中文 token 更完整），但增加 embedding 显存和 softmax 计算量。LM head 通常与 embedding 权重 **weight tying** 以节省参数。

---

### 1.14 Scaling Laws

**Chinchilla Scaling Law**（Hoffmann et al., 2022）：

$$
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty
$$

- $N$：参数量，$D$：训练 token 数，$L_\infty$：不可约 loss
- 给定 FLOPs 预算 $C$，最优 $N \propto D$（约 1:1 比例）

**实践含义**：早期大模型（如 175B 参数、300B tokens）属于"过度参数化、训练不足"。用更小模型训练更多 token 可获得更优性价比。

**后 Chinchilla 趋势**：LLaMA 系列等推动"以推理效率为目标"的训练——即使超出 Chinchilla 最优 token 数继续训练，推理 FLOPs 不变但性能持续提升。

---

## Part 2 · PyTorch 代码片段

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

### 2.7 Causal Mask 生成

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

> **Note**: 上述 MoE 实现为教学用途的朴素版本，实际生产代码会用 fused kernels 和 capacity-aware dispatch 来避免 Python 循环。

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

## Part 3 · 面试题集 (Interview Questions)

---

### L1 · 基础 (Foundational)

---

<details>
<summary>Q1. Transformer decoder block 的标准结构是什么？</summary>

**A**: Pre-LN 结构依次为：LayerNorm → Multi-Head Causal Self-Attention → Residual → LayerNorm → FFN → Residual。其中 causal mask 将 attention score 矩阵的上三角填 $-\infty$，确保 token 只能看到自己及之前的 token。FFN 通常将维度扩展 4×（或 SwiGLU 的 $8/3\times$）再投影回来。

> **Follow-up**: Pre-LN 和 Post-LN 各有什么优缺点？为什么当前 LLM 普遍采用 Pre-LN？
>
> *提示：Pre-LN 训练更稳定（梯度流更平滑），但有研究认为 Post-LN 在最终性能上略有优势。Pre-LN 成为主流是因为在深层模型（>100层）中 Post-LN 训练不稳定。*

</details>

---

<details>
<summary>Q2. Self-Attention 的计算过程和复杂度是什么？</summary>

**A**: 输入 $X$ 经 $W_Q, W_K, W_V$ 投影得到 $Q, K, V$；计算 $\text{softmax}(QK^\top / \sqrt{d_k})V$。时间复杂度 $O(N^2 d)$，空间复杂度 $O(N^2)$（存储 attention 矩阵）。$N$ 为序列长度，$d$ 为 head dimension。$N^2$ 项是长序列处理的瓶颈。

> **Follow-up**: $\sqrt{d_k}$ 缩放因子的作用是什么？省略它会导致什么问题？
>
> *提示：当 $d_k$ 较大时，$QK^\top$ 的元素方差约为 $d_k$，导致 softmax 饱和，梯度消失。除以 $\sqrt{d_k}$ 使方差归一化到 $\sim 1$。*

</details>

---

<details>
<summary>Q3. 什么是 KV Cache？为什么能加速自回归生成？</summary>

**A**: 在自回归生成中，每生成一个新 token 时，之前所有 token 的 K/V 向量不会改变。KV Cache 将已计算的 K/V 存储在 GPU 显存中，新 token 只需计算自己的 Q 并与缓存做 attention，避免对历史 token 的重复计算。这将 prefill 之后每个 decode step 的计算量从 $O(n \cdot d)$（重新计算所有 K/V）降为 $O(d)$（仅计算新 token 的投影）+ $O(n \cdot d)$（attention，但无需重新投影）。

> **Follow-up**: KV Cache 的显存占用如何随序列长度和 batch size 变化？有哪些压缩方法？
>
> *提示：显存 = batch\_size × seq\_len × 2 × L × n\_kv\_heads × d\_k × bytes。压缩方法包括 MQA/GQA、KV Cache 量化（FP8/INT8）、token pruning/eviction 等。*

</details>

---

<details>
<summary>Q4. 比较 Greedy、Beam Search、Top-$k$、Top-$p$、Temperature 采样。</summary>

**A**:
- **Greedy**：每步取最高概率 token，确定性，易重复退化
- **Beam Search**：维护 $k$ 条候选，适合有确定目标的生成（翻译），open-ended 生成多样性差
- **Top-$k$**：只从概率前 $k$ 的 token 采样，$k$ 固定不自适应
- **Top-$p$**：从累积概率 $\geq p$ 的最小集合采样，自适应候选集大小
- **Temperature**：$T < 1$ 使分布更尖锐（更确定），$T > 1$ 更平坦（更随机），常与 top-$p$ 联用

> **Follow-up**: 为什么 Beam Search 在 open-ended 生成中反而比 sampling 效果差？
>
> *提示：Beam Search 优化的是序列概率，倾向于选择"安全"的高频 token，导致生成结果缺乏多样性和自然度。引入 diversity penalty 可部分缓解。*

</details>

---

<details>
<summary>Q5. Causal Mask 是如何实现的？为什么 decoder-only 模型必须使用它？</summary>

**A**: 在 attention score 矩阵 $S \in \mathbb{R}^{N \times N}$ 中，将 $j > i$ 的位置（上三角）设为 $-\infty$，softmax 后这些位置权重为 0。这确保了 token $i$ 在计算 attention 时只能看到位置 $\leq i$ 的 token。Decoder-only 模型用 next-token prediction 做训练（并行计算所有位置的 loss），causal mask 保证训练时每个位置的预测不泄露未来信息，与推理时的自回归行为一致。

> **Follow-up**: Prefix LM（如 T5 decoder、GLM）如何在同一序列中混合 bidirectional 和 causal attention？
>
> *提示：Prefix 部分的 token 之间使用双向 attention（无 mask），生成部分对 prefix 双向可见但对自身及之后使用 causal mask。*

</details>

---

<details>
<summary>Q6. Residual Connection 在 Transformer 中的作用是什么？</summary>

**A**: 三个核心作用：
1. **梯度流通**：提供绕过子层的直连路径，缓解深层网络梯度消失
2. **特征复用**：$h_l = h_{l-1} + F(h_{l-1})$，模型可选择性地保留或修改信息
3. **训练稳定性**：与 Pre-LN 结合使得百层以上模型可稳定训练

> **Follow-up**: 有没有工具可以分析 Transformer 各层残差流（residual stream）中的信息流动？
>
> *提示：Logit Lens 方法将中间层的 hidden state 通过 LM head 投影到词表空间，观察每层的预测分布变化。*

</details>

---

<details>
<summary>Q7. FFN 中的维度扩展比为什么通常是 $4\times$？</summary>

**A**: $d_{\text{ffn}} = 4d_{\text{model}}$ 来自原始 Transformer 论文的经验设定，没有严格的理论推导。后续实验表明该比例在多数设置下有效。SwiGLU 变体使用 $\frac{8}{3}d_{\text{model}}$，因有两个 gate 矩阵，总参数量与标准 $4\times$ FFN 大致相当。

> **Follow-up**: FFN 在 Transformer 中可能扮演什么样的计算角色？有没有研究将 FFN 视为"知识存储"？
>
> *提示：有研究（如 key-value memory 视角）将 FFN 的第一层视为 key 矩阵、第二层视为 value 矩阵，类比 attention 的检索机制。*

</details>

---

### L2 · 进阶 (Intermediate)

---

<details>
<summary>Q8. 为什么需要 Multi-Head Attention？与单 head 相比有什么优势？</summary>

**A**: 单 head attention 只能学习一种"关注"模式。$h$ 个 head 各自在不同子空间学习不同的关系模式（如句法依存、指代消解、局部 n-gram 等），concat 后通过 $W^O$ 投影融合。总参数量不变（每个 head 的 $d_k = d_{\text{model}}/h$），但表达能力更强。

> **Follow-up**: 不同 head 是否真的学到了不同的模式？Head pruning 的可行性如何？
>
> *提示：Michel et al. (2019) 发现可以在推理时去掉大量 head 而性能下降很小；Voita et al. (2019) 通过 head importance 分析了不同 head 的功能分化。*

</details>

---

<details>
<summary>Q9. MHA、MQA、GQA 的区别是什么？GQA 如何从 MHA checkpoint 做转换？</summary>

**A**:
- **MHA**：$h$ 个 Q head 对应 $h$ 套 K/V，KV Cache 最大
- **MQA**：所有 Q head 共享 1 套 K/V，KV Cache 缩减 $h$ 倍，但可能损失质量
- **GQA**：$h$ 个 Q head 分 $g$ 组，每组共享 1 套 K/V，是 MHA 和 MQA 的折衷

从 MHA 转换到 GQA：将每 $h/g$ 个 K/V head 的权重取均值作为新 K/V head 初始化，再用少量数据（如原预训练量的 5%）继续训练（uptrain）。

> **Follow-up**: LLaMA-3 使用了什么 attention 变体？GQA 的分组数 $g$ 如何选择？
>
> *提示：LLaMA-3 使用 GQA。$g$ 的选择是在推理效率和模型质量之间的 trade-off，通常通过小规模实验确定。*

</details>

---

<details>
<summary>Q10. 绝对位置编码、RoPE、ALiBi 各自的特点和区别？</summary>

**A**:
- **绝对位置编码**（Sinusoidal/Learned）：位置信息直接加到 embedding，最大长度固定，外推差
- **RoPE**：对 Q/K 施加旋转使 attention score 仅依赖相对位置 $m-n$，支持长度外推（PI/NTK/YaRN），LLaMA 系列采用
- **ALiBi**：在 attention score 上加线性偏置 $-m_h|i-j|$，无额外参数，外推好，但不支持 prefix caching

> **Follow-up**: RoPE 是通过修改 embedding 还是修改 attention score 来实现相对编码的？它与绝对位置编码在数学上有什么联系？
>
> *提示：RoPE 在 embedding 层面（Q/K 上）施加旋转，但从效果看等价于在 attention score 上加了依赖相对位置的 bias。与绝对位置编码的区别在于它通过旋转不变性实现了相对位置编码。*

</details>

---

<details>
<summary>Q11. BPE (Byte Pair Encoding) 的原理是什么？</summary>

**A**: 初始词表 = 所有字节/字符。迭代地统计语料中频率最高的相邻符号对，将其合并为新符号加入词表，直到达到目标大小。优点：词表大小可控，能处理 OOV。缺点：同一词可能有多种 tokenization 方案（大小写、空格变体），对非英文语言效率较低。SentencePiece 是语言无关的 BPE 实现，直接在 unicode 字节上操作。

> **Follow-up**: Tokenizer 的 vocabulary size 对模型有什么影响？GPT-2 (50K) vs LLaMA (32K) vs LLaMA-3 (128K) 各有什么考量？
>
> *提示：影响包括 embedding 层参数量（$|V| \times d_{\text{model}}$）、每 token 的信息密度、多语言和代码的 token 效率、softmax 计算开销。大词表对多语言更友好但增加显存。*

</details>

---

<details>
<summary>Q12. PagedAttention 的核心思想和解决了什么问题？</summary>

**A**: 类比 OS 的虚拟内存分页：将 KV Cache 切成固定大小的 block，用 block table 管理逻辑-物理地址映射，允许非连续物理显存分配。解决了标准实现中 KV Cache 需要连续显存导致的内存碎片问题（请求长度不同导致大量内部和外部碎片），大幅提高 GPU 显存利用率。

> **Follow-up**: Continuous batching 和 PagedAttention 是什么关系？它们分别解决什么层面的问题？
>
> *提示：Continuous batching 解决调度问题（请求级别的 iteration-level scheduling，不必等 batch 中最长序列完成）；PagedAttention 解决显存管理问题（KV Cache 的高效分配）。两者互补。*

</details>

---

<details>
<summary>Q13. MoE (Mixture of Experts) 的基本原理是什么？</summary>

**A**: 将 FFN 替换为 $E$ 个 expert FFN。每个 token 经 router（线性层 + softmax + top-$k$ 选择）选 $k$ 个 expert（通常 $k=2$），只激活被选中的 expert 计算。总参数量 = $E \times$ expert 参数（远大于 dense），但每 token 的 FLOPs ≈ $(k/E) \times$ dense FLOPs。加负载均衡 loss 防止 expert collapse（所有 token 涌向少数 expert）。

> **Follow-up**: MoE 模型在推理和训练时的显存和计算特点有什么不同？MoE 对 tensor parallelism 有什么特殊要求？
>
> *提示：推理时每个 token 只激活 $k$ 个 expert，但所有 expert 参数都需加载到显存（显存需求大但计算少）。训练时需要 expert-level parallelism 或 capacity factor 控制负载。*

</details>

---

<details>
<summary>Q14. Chinchilla Scaling Law 的核心结论是什么？</summary>

**A**: 给定训练 FLOPs 预算 $C$，最优的模型参数量 $N$ 和训练 token 数 $D$ 应大致满足 $N \propto D$（约 1:1 比例）。即之前的很多大模型（如 175B 参数只训练 300B tokens）属于"过度参数化、训练不足"。用更小模型训练更多数据可以获得更好的 loss per FLOP。

> **Follow-up**: Scaling Law 在 post-training（SFT/RLHF）阶段还适用吗？
>
> *提示：目前关于 post-training 的 scaling law 研究仍在进行中。有工作探索了 SFT 数据量和模型大小的关系，但 RLHF 的 scaling 行为更复杂，受 reward model 质量、KL 约束等因素影响。*

</details>

---

<details>
<summary>Q15. 估算一个 Transformer 模型的 KV Cache 显存占用。</summary>

**A**: 公式：$\text{KV Cache} = 2 \times L \times n_{\text{kv}} \times d_k \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes}$

其中 factor 2 对应 K 和 V，$n_{\text{kv}}$ 是 K/V head 数（GQA 时 $< h$），$d_k$ 是 head dimension。以 32 层、$n_{\text{kv}} = 8$、$d_k = 128$、BF16 为例：每 token 约 128 KB，4096 token 序列约 0.5 GB（单序列），batch 32 则约 16 GB。

> **Follow-up**: KV Cache 量化（如 FP8/INT8）对生成质量的影响如何？有哪些实现方案？
>
> *提示：INT8 量化通常对质量影响很小，FP8 进一步减少。vLLM、TensorRT-LLM 等推理框架已内置 KV Cache 量化支持。*

</details>

---

<details>
<summary>Q16. 什么是 Sliding Window Attention？有什么优缺点？</summary>

**A**: 每个 token 只 attend 到距离 $\leq W$ 的 token（局部窗口），复杂度 $O(NW)$。多层叠加后感受野以层数线性增长（$L$ 层后理论感受野 $\approx L \times W$），类比 CNN。Mistral-7B 使用 $W = 4096$。

优点：复杂度低，推理可用 **rolling buffer KV Cache**（固定大小环形缓冲区）。缺点：单层内无法建立跨窗口的精确 attention，对 needle-in-haystack 类检索任务表现较弱。

> **Follow-up**: Mistral 的 rolling buffer KV cache 是如何工作的？
>
> *提示：KV Cache 用固定大小 $W$ 的循环缓冲区存储，新 token 覆盖最旧的 token（位置 $i \bmod W$），无需显存随序列增长。*

</details>

---

<details>
<summary>Q17. Neural Scaling Laws 的基本形式是什么？</summary>

**A**:

$$
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty
$$


$N$ = 参数量，$D$ = 训练 token 数，$L_\infty$ = 不可约 loss（数据本身的信息熵）。$\alpha$ 和 $\beta$ 分别衡量参数量和数据量的边际收益递减速率。在很宽的范围内 loss 与 $N$, $D$ 保持幂律关系。

> **Follow-up**: 为什么 emergent abilities（如 chain-of-thought 推理）无法从 scaling law 直接预测？
>
> *提示：Scaling law 描述的是 loss（困惑度）的平滑下降；emergent abilities 是特定任务指标（如 accuracy）在某个 scale 突然从接近随机跳到远超随机，可能是指标选择的假象（Wei et al., 2022; Schaeffer et al., 2023）。*

</details>

---

<details>
<summary>Q18. Tokenizer 的 vocabulary size 对模型能力有什么影响？</summary>

**A**: Embedding 层参数量 = $|V| \times d_{\text{model}}$（通常与 LM head 共享权重）。大词表：embedding 参数占比增加、softmax 计算量增加、但每 token 信息密度更高、对多语言和代码更友好。小词表：embedding 参数少、但对中文等语言需要更多 token 表示同一文本（序列更长、推理更慢）。

> **Follow-up**: 有没有研究词表大小与模型能力的 scaling 关系？如何选择最优词表大小？
>
> *提示：词表大小的选择需要平衡多语言覆盖、推理效率和 embedding 参数开销。LLaMA-3 扩展到 128K 的主要动机是提升多语言和代码的 token 效率。*

</details>

---

### L3 · 深度 (Advanced)

---

<details>
<summary>Q19. FlashAttention 解决了什么问题？它是精确算法还是近似算法？</summary>

**A**: FlashAttention 解决标准 attention 的 **memory-bound** 问题——需要将 $N \times N$ 的 attention 矩阵写回 HBM。核心是 **tiling**（分块）+ **online softmax**：将 Q/K/V 分块加载到 SRAM，在 SRAM 内完成分块 attention 计算，通过 online softmax 算法在不存储完整 $N \times N$ 矩阵的情况下得到**数学等价**的结果。FlashAttention 是**精确算法**，不是近似。IO 复杂度从 $O(N^2)$ 降至 $O(N^2 d / M)$（$M$ 为 SRAM 大小），显存从 $O(N^2)$ 降至 $O(N)$。

> **Follow-up**: Online softmax 如何在不存储完整 score 矩阵的情况下保证数学等价？
>
> *提示：Online softmax 维护 running max 和 running sum 的统计量，每处理新 block 时通过 rescaling 前面 block 的贡献来修正，最终得到与全局 softmax 完全相同的结果（需要额外的 rescaling pass）。*

</details>

---

<details>
<summary>Q20. RoPE 长度外推的主要方法有哪些？各自的思路？</summary>

**A**:
1. **Position Interpolation (PI)**：将位置索引线性缩放 $m \leftarrow m \cdot L_{\text{train}} / L_{\text{test}}$，简单但高频信息损失
2. **NTK-aware Scaling**：修改 base ($10000 \to \alpha \cdot 10000$)，高频维度少缩放、低频维度多缩放
3. **YaRN**：NTK + 非均匀插值 + attention temperature 调整，效果更好
4. **Dynamic NTK**：推理时根据实际序列长度动态调整 base，无需重新训练

核心挑战：训练时未见过的高频旋转角会导致 attention score 分布偏移，外推方法本质上是在保持已学模式和适应新长度之间找平衡。

> **Follow-up**: YaRN 相比 naive PI 做了哪些具体改进？为什么需要 attention temperature 调整？
>
> *提示：YaRN 区分了高频和低频维度的插值策略，并加入了 attention temperature 因子 $1/\sqrt{t}$ 来补偿插值导致的 attention score 幅度变化。*

</details>

---

<details>
<summary>Q21. Speculative Decoding 的原理是什么？为什么输出分布与直接用大模型等价？</summary>

**A**: 用小 draft model 并行生成 $k$ 个候选 token，再用大 verifier 一次性前向验证。验证时用 rejection sampling：接受概率 $\min(1, p_{\text{verifier}}(x) / p_{\text{draft}}(x))$，拒绝时从修正分布 $\max(0, p_v - p_d)$ 归一化后重新采样。数学上可证明，最终每个位置的 token 分布恰好等于 $p_{\text{verifier}}$。

吞吐提升来自：draft model 极快（小模型），且验证 $k$ 个 token 只需一次前向传播（并行），而正常生成需要 $k$ 次。

> **Follow-up**: Draft model 和 verifier 必须同架构吗？Self-speculative decoding 是怎么做的？
>
> *提示：不需要同架构，但词表必须相同。Self-speculative decoding 用同一模型的部分层（early exit）或跳过部分层（layer skipping）作为 draft，省去额外模型。*

</details>

---

<details>
<summary>Q22. 扩展 LLM 上下文长度的主要技术路线有哪些？</summary>

**A**:
1. **位置编码外推**：PI/NTK/YaRN 修改 RoPE，需少量 continue training
2. **Sliding Window Attention**：局部窗口 $O(NW)$，多层堆叠扩大感受野
3. **Sparse Attention**：Longformer/BigBird 的全局+局部+随机混合 attention
4. **Ring Attention**：将长序列切分到多设备，通过 all-to-all 通信传递 KV，理论上支持无限长度
5. **Attention sink**：保留初始几个 token 的 KV（StreamingLLM），解决 sliding window 的 "attention sink" 现象

> **Follow-up**: Ring Attention 对通信带宽有什么要求？它和 Sequence Parallelism 有什么区别？
>
> *提示：Ring Attention 需要设备间 high-bandwidth interconnect（如 NVLink），通信量与窗口大小成正比。与传统 Sequence Parallelism 的区别在于 Ring Attention 的分块是序列维度的 pipeline 式传递，而非简单的数据并行。*

</details>

---

<details>
<summary>Q23. Expert Capacity 和 Token Dropping 是什么？对训练有什么影响？</summary>

**A**: 每个 expert 在一个 batch 内有容量上限（capacity factor × tokens/expert）。超出容量的 token 被 **drop**——跳过该 expert，直接使用 residual 或 zero 输出。训练时 capacity factor 通常设 1.0–1.25；设太大会浪费计算，设太小会导致 token 丢失。

Token dropping 会引入训练噪声，影响梯度质量。缓解方法包括：增加 capacity factor（代价是计算浪费）、改进 router 设计（如 load-balancing loss）、使用 expert choice routing（expert 选 token 而非 token 选 expert）。

> **Follow-up**: Expert choice routing 和 token choice routing 有什么区别？各自的优缺点？
>
> *提示：Token choice（标准 top-k）：每个 token 选 $k$ 个 expert，可能导致负载不均。Expert choice：每个 expert 选 top-$k$ 个 token，天然负载均衡，但部分 token 可能不被任何 expert 选中。*

</details>

---

<details>
<summary>Q24. Causal Mask + KV Cache 下的 attention 计算与训练时有何不同？</summary>

**A**: **训练时**：对整个序列并行计算，构造 $N \times N$ 的 causal mask 矩阵，所有位置的 Q/K/V 一次性计算完成。**推理时**（有 KV Cache）：新 token 的 Q 是 $(1, d)$ 向量，K/V Cache 是 $(n, d)$ 矩阵（$n$ 为已有序列长度），attention 计算为 $(1, d) \times (d, n) = (1, n)$，不需要显式的 causal mask（因为 K/V 中不包含未来 token）。Prefill 阶段（处理 prompt）与训练类似，使用 causal mask 并行计算。

> **Follow-up**: 在 prefill 阶段，如果 prompt 非常长（如 100K tokens），计算瓶颈在哪里？有什么优化方法？
>
> *提示：Prefill 的瓶颈是 $O(N^2)$ 的 attention 计算和 $O(N)$ 的 KV Cache 写入。优化方法包括 chunked prefill（分块处理，与 decode 请求交织）、FlashAttention 减少显存和计算、以及 prefix caching（对相同前缀的请求复用 KV Cache）。*

</details>

---

<details>
<summary>Q25. 比较 dense Transformer 和 MoE Transformer 在 scaling 行为上的区别。</summary>

**A**: Dense 模型：增加参数量 = 增加计算量（FLOPs/token $\propto N$）。MoE 模型：总参数量 $N_{\text{total}} \gg N_{\text{active}}$（激活参数），FLOPs/token $\propto N_{\text{active}}$。这意味着 MoE 可以在相同计算预算下拥有更大的"知识容量"。

Scaling law 角度：MoE 的 loss 主要由激活参数 $N_{\text{active}}$ 和数据量 $D$ 决定（与 dense 类似），但 expert 数量增加在数据充足时仍有收益（不同 expert 可以 specialize 不同领域的知识）。不过 MoE 的 scaling 效率受 router 质量、负载均衡、expert 利用率等因素制约。

> **Follow-up**: DeepSeek 的 MoE 设计（如 DeepSeek-V2/V3）有什么创新？共享 expert 和 routed expert 的分离有什么好处？
>
> *提示：DeepSeek-V2/V3 引入了 shared expert（所有 token 都经过）+ routed expert（按 router 选择）的设计，shared expert 负责通用能力，routed expert 负责领域知识。还有细粒度 expert 分割（更多但更小的 expert）等设计。*

</details>

---

> **License**: 本手册仅供学习参考，请勿用于商业用途。内容基于公开发表的研究论文和技术博客整理。
>
> **Last Updated**: 2025


## 更多 L3 深挖 / Extended L3

<details>
<summary>Q26. FlashAttention 中的 Online Softmax 算法是如何实现的？为什么能在不存储完整 attention 矩阵的前提下得到精确结果？</summary>

Online Softmax 的核心思想是：softmax 可以通过维护一个**运行中的最大值**（running max）和**修正因子**来增量计算。将 Q、K 分成 block（tile）后，逐块计算局部 $QK^\top$ 分数，每块得到局部 max $m_{\text{local}}$；与当前全局 max $m_{\text{global}}$ 比较后，对之前已累加的 $\sum \exp$ 乘以修正因子 $e^{m_{\text{global}} - m_{\text{new}}}$，然后更新 $m_{\text{global}}$。这样每块的 partial sum 都可被正确缩放，最终结果与一次性计算完整 softmax **逐位一致**。输出 $O$ 也做类似的在线加权累积。整个过程中 $N \times N$ 矩阵从未在 HBM 中完整存在，仅在 SRAM 中处理局部 tile。

> **追问**：如果 Online Softmax 需要在处理每个 tile 时回溯修正之前累积的 $O$，那么最终输出 $O$ 是一次性写回还是经过多次修正？FlashAttention 如何避免修正带来的额外显存写入？

</details>

<details>
<summary>Q27. 为什么 Pre-LN 比 Post-LN 在深层 Transformer 中训练更稳定？从梯度传播的角度分析。</summary>

Post-LN 中 LayerNorm 在残差之后：$\text{output} = \text{LN}(x + \text{SubLayer}(x))$。在深层网络中，残差路径上的梯度需要穿过每一层的 LN，LN 的 Jacobian 依赖于该层激活的统计量，导致**层间梯度尺度耦合**。随着深度增加，梯度在早期层容易出现指数级增长或衰减（梯度爆炸/消失的变体），需要精细的 warmup 策略。Pre-LN 中 $\text{output} = x + \text{SubLayer}(\text{LN}(x))$，LN 在子层输入端，残差路径上**无 LN 阻隔**，梯度可沿残差直通（类似 ResNet 的 identity shortcut），深层模型无需 warmup 即可稳定训练。代价是 Pre-LN 的最终性能可能略低于精心调参的 Post-LN。

> **追问**：DeepNorm（微软提出的变体）在 Post-LN 基础上通过调整残差缩放系数 $\alpha$ 和初始化来稳定训练，其核心数学原理是什么？

</details>

<details>
<summary>Q28. MoE 中 router collapse 有哪些不同的表现形式？除辅助 loss 外还有哪些缓解方法？</summary>

Router collapse 有多种模式：(1) **完全坍缩**——几乎所有 token 只选中一两个 expert，其余 expert 无梯度更新；(2) **部分不均衡**——大多数 expert 被利用但少数"明星 expert"被过度使用；(3) **循环振荡**——expert 利用率随训练轮次交替变化但不收敛。辅助 loss（如 $\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum f_i p_i$）仅是最基础的手段。其他策略包括：**Expert-Choice Routing**——让每个 expert 主动选 top-$k$ token（而非 token 选 expert），天然保证负载均衡；**Random routing with noise**——在 router logits 上加可调噪声防止确定性坍缩；**Capacity factor 动态调整**——根据训练进度自适应容量；以及对 router 参数使用**更小的学习率**或独立 optimizer 避免路由决策过度波动。

> **追问**：Expert-Choice Routing 中，某些 token 可能被零个或多个 expert 同时选中，这对模型训练和推理分别有什么影响？如何处理？

</details>

<details>
<summary>Q29. PagedAttention 中的 Copy-on-Write (CoW) 和 block 共享机制如何工作？在哪些场景下利用 block 共享？</summary>

PagedAttention 给每个请求维护一个 **block table**（逻辑→物理页映射）。当多个请求需要共享相同的 KV Cache 前缀（如 system prompt 或 beam search 中同一 parent 序列的多个分支）时，它们可以**共享相同的物理 block**，避免重复存储。此时 block 的 **reference count** > 1。当某个请求需要修改（追加新 token）共享 block 中的内容时，触发 **Copy-on-Write**：分配新的物理 block，复制原 block 内容，减少原 block 引用计数，再写入新数据。这种机制在以下场景收益显著：(1) **beam search** 中多个 beam 共享前缀；(2) **parallel sampling**（同一 prompt 生成多个回答）；(3) **prefix caching**（多轮对话复用 system prompt 的 KV）。

> **追问**：PagedAttention 的 block size 如何选择？太小和太大分别带来什么问题？实际系统中典型的 block size 是多少？

</details>

<details>
<summary>Q30. 除 GQA/MQA 外，还有哪些 KV Cache 压缩与驱逐策略？各自的权衡是什么？</summary>

主要路线包括：(1) **KV Cache 量化**——将 K/V 从 FP16 量化到 INT8/INT4 甚至更低精度，减少显存但需注意 attention 精度损失，尤其是 V 的量化对输出质量影响更大；(2) **Token 驱逐/淘汰**——基于策略丢弃不重要的历史 token 的 KV，如 H2O（Heavy Hitter Oracle）保留注意力分数高的 token，StreamingLLM 保留"attention sink"（最初几个 token）+ 滑动窗口；(3) **KV 合并**——将多个相邻 token 的 K/V 向量合并为一个（如通过加权平均或 PCA），以精度换空间；(4) **层间 KV 共享**——相邻层共享同一组 K/V，减少总 Cache 量。每种方法都是**精度-显存-计算开销**的三方权衡：量化计算开销最小但精度上限受限；驱逐策略简单但可能丢失关键信息；合并策略灵活性高但引入额外计算。

> **追问**：StreamingLLM 观察到 attention sink 现象——即模型前几个 token 无论语义相关性如何都会获得较高的注意力分数。你认为这背后可能的原因是什么？

</details>

<details>
<summary>Q31. SwiGLU 相比标准 ReLU FFN 的优势来自哪里？gated activation 的设计动机是什么？</summary>

标准 ReLU FFN $\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x)$ 中，ReLU 是逐元素的硬门控（输出为 0 或线性），信息通路较粗暴。SwiGLU 引入**乘性门控**：$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2$，其中 $xW_3$ 作为 gate，$\odot$ 为逐元素乘。这个乘性交互使 FFN 具有**更丰富的特征组合能力**——两个线性变换的逐元素乘积等价于一种双线性操作，能建模特征间更复杂的依赖。Swish（$x \cdot \sigma(x)$）本身是光滑的非单调激活，相比 ReLU 的稀疏性提供了更平滑的梯度流。实验上，控制总参数量相同时（$d_{\text{ffn}}$ 从 $4d$ 缩减到约 $\frac{8}{3}d$，因为多了一个 gate 矩阵 $W_3$），SwiGLU 在多项 benchmark 上一致性优于 ReLU/GELU FFN。

> **追问**：GLU 家族（包括 ReGLU、GeGLU、SwiGLU）共享同一个门控框架，为什么 SwiGLU 被选择成为主流（如 LLaMA、Mistral 系列），而非其他变体？

</details>

<details>
<summary>Q32. Speculative Decoding 的加速效果在哪些场景下会显著下降？Self-Speculative 方法的基本思路是什么？</summary>

Speculative Decoding 的加速取决于 **acceptance rate**（大模型接受小模型预测的比例）。加速效果下降的场景包括：(1) **分布高度发散**——当 draft model 与 verifier 的分布差异大时（如不同架构家族），大量 token 被拒绝，speculation 几乎无效；(2) **高度随机的生成**——temperature 较高时，正确 token 的概率分布平坦，draft model 难以猜中 verifier 的采样结果；(3) **结构化/受限输出**——如 JSON 格式输出中某些位置只有少数合法 token，draft model 未必能覆盖；(4) **draft model 过小**——质量太低导致 $\mathbb{E}[\text{accepted tokens}]$ 趋近 1。**Self-Speculative Decoding**（如 Medusa、EAGLE）避免使用独立 draft model，而是在大模型自身上做轻量级推测：例如在最后几层后接额外的 **prediction head**（Medusa），或用一个小型 **draft decoder** 复用大模型的 hidden states（EAGLE），省去独立小模型的显存和加载开销。

> **追问**：Medusa 方法在大模型顶层并行添加多个 prediction head，每个 head 负责预测未来第 $k$ 个 token。这些 head 如何训练？为什么不能简单共享一个 head 同时预测多个位置？

</details>

<details>
<summary>Q33. Post-Chinchilla 时代，为何模型"训练过度"（远超 Chinchilla 最优 token 数）反而成为主流？"Inference-Optimal" 训练范式的逻辑是什么？</summary>

Chinchilla Scaling Law 最优化的是**给定 FLOPs 预算下的训练 loss**。但实际部署中，训练是一次性成本，而**推理是持续成本**——一个模型可能服务数十亿次查询。较小模型推理时 FLOPs/decode token 更低、吞吐更高、延迟更小、部署成本更低。因此，如果用远超 Chinchilla 最优的 token 数训练一个较小模型，虽然训练 loss 不如同等 FLOPs 训练的更大模型，但该较小模型在推理侧的**性价比**可能显著更优——即在相同推理预算下能服务更多请求，而性能损失有限。LLaMA 系列率先验证了这一思路：用较少参数、更多训练 token 得到的模型，在推理效率上优于 Chinchilla 最优配比得到的更大模型。核心逻辑是将优化目标从"**最小化训练 loss**"转向"**最小化推理成本下达到目标性能**"。

> **追问**：如果推理侧也在优化目标中，那么从训练 token 数、模型参数量、推理硬件约束三者的视角，如何定义一个"inference-optimal"的模型大小选择？

</details>
