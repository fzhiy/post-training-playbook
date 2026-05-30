# ML / LLM System Design — Cheat Sheet

> For LLM research intern preparation
> Public edition · No proprietary results included

---

## Part 1 — Concepts & Key Formulas

### 1.1 Causal Language Modeling (CLM)

**Core Idea:** Predict the next token autoregressively; during training, a causal mask prevents future information leakage.

**Loss Function:**

$$
\mathcal{L}_{\text{CLM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

**Derivation:**
- By the chain rule: $P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$
- Taking the logarithm and negating gives the cross-entropy loss
- In practice, logits have shape `(batch, seq_len, vocab_size)`; targets are token ids shifted left by one position

---

### 1.2 Softmax & Attention

**Scaled Dot-Product Attention:**

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**Why scale by $\sqrt{d_k}$?**
- Assume elements of $Q$ and $K$ are i.i.d. with mean 0 and variance 1
- Then each element of $QK^\top$ has variance $d_k$
- When $d_k$ is large, softmax inputs are large → vanishing gradients (softmax saturation)
- Dividing by $\sqrt{d_k}$ normalizes the variance to 1, keeping gradients stable

**Multi-Head Attention (MHA):**

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attn}(XW_i^Q,\; XW_i^K,\; XW_i^V)
$$

where $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $d_k = d_v = d_{\text{model}} / h$.

**GQA / MQA Variants:**
- **Multi-Query Attention (MQA):** All heads share the same $K, V$; only $Q$ differs → KV cache significantly reduced
- **Grouped-Query Attention (GQA):** The $h$ query heads are divided into $g$ groups, each sharing $K, V$; a middle ground between MHA and MQA

---

### 1.3 Position Encoding

**Rotary Position Embedding (RoPE):**

$$
\tilde{q}_m = q_m e^{im\theta}, \quad \tilde{k}_n = k_n e^{in\theta}
$$

where $\theta_j = 10000^{-2j/d}$.

$$
\langle \tilde{q}_m, \tilde{k}_n \rangle = \text{Re}[q_m^* k_n \, e^{i(m-n)\theta}]
$$

**Properties:**
- The inner product depends only on the relative position $(m-n)$ → naturally encodes relative positions
- No learnable parameters (deterministic)
- Better extrapolation than learned positional embeddings (length can be extended with NTK-aware scaling)

**Practical Implementation of RoPE:**

$$
\text{RoPE}(x) = x \odot \cos(m\theta) + \text{rotate\_half}(x) \odot \sin(m\theta)
$$

For $x \in \mathbb{R}^{d_k}$, each pair of adjacent dimensions $(x_{2i}, x_{2i+1})$ undergoes a 2D rotation.

---

### 1.4 LoRA — Low-Rank Adaptation

**Motivation:** Full fine-tuning of large models has heavy GPU memory overhead (requires storing parameters, gradients, and optimizer states). LoRA freezes the pretrained weights and trains only a low-rank delta.

**Key Formula:**

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

where $W_0 \in \mathbb{R}^{d \times k}$ is frozen, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.

**Scaling:**

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

$\alpha$ is a scaling hyperparameter, typically set to $\alpha = 2r$ or $\alpha = r$.

**Parameter Count Analysis:**
- Original parameters: $d \times k$
- LoRA parameters: $d \times r + r \times k = r(d + k)$
- Example: $d = 4096, k = 4096, r = 16$ → LoRA params $= 16 \times 8192 = 131072$, which is $131072 / (4096^2) \approx 0.78\%$ of the original

**Initialization:**
- $A$: Kaiming uniform initialization (or Gaussian)
- $B$: zero initialization → at training start $\Delta W = BA = 0$, preserving pretrained output

**Merge for Inference:**

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA
$$

After merging, inference has no extra overhead.

---

### 1.5 Reinforcement Learning from Human Feedback

**Reward Model Training (Bradley-Terry):**

$$
\mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
$$

where $y_w \succ y_l$ is a human-annotated preference pair (preferred vs. rejected).

**PPO Objective:**

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)} \!\Big[ r_\phi(x, y) - \beta \, \text{KL}\big(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\big) \Big]
$$

**Role of KL Divergence:**
- $\beta$ too small → reward hacking (the policy exploits gaps in the reward model)
- $\beta$ too large → policy barely moves (degenerates to the SFT model)

**DPO (Direct Preference Optimization):**

Bypasses an explicit reward model; derived from the Bradley-Terry model:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma \!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

**DPO Advantages:**
- No RL sampling loop (no need to generate responses during training)
- No explicit reward model required
- More stable training, fewer hyperparameters

**DPO Limitations:**
- The implicit reward may generalize less well than an explicit RM
- More sensitive to preference data quality (no RM "buffer")
- Not easily extended to online RL (requires on-policy sampling for improvement)


---

### 1.5b Distributed RLHF Architecture

#### GPU Utilization Problem in Naive Co-located PPO

The simplest implementation runs the actor, reference model, critic, and reward model all on the same set of GPUs (co-located). The bottleneck is the **rollout phase**:

```
┌─────────────────────────────────────────────────────┐
│  Co-located PPO (simplified timeline)               │
│                                                     │
│  ──[rollout: actor autoregressive gen]──►  ──[train: PPO update]──► │
│         GPU busy with inference           trainer busy, actor idle   │
└─────────────────────────────────────────────────────┘
```

- **During rollout**: the actor generates tokens one at a time (memory-bound; throughput limited by HBM bandwidth); GPU MFU (Model FLOP Utilization) is often low; the trainer (ZeRO/FSDP) sits idle.
- **During training**: forward + backward is compute-intensive; the rollout worker sits idle.
- Result: the two phases alternate, and overall GPU utilization is a weighted average of each phase's utilization — well below peak utilization for pure training or pure inference.

⚠️ These are not precise measurements; actual MFU depends on model size, batch size, and hardware. The description above captures the **qualitative problem**; for actual numbers consult the technical reports of the relevant frameworks (OpenRLHF, veRL, etc.).

---

#### Disaggregated Rollout + Train Topology

To address the above, **disaggregate** rollout workers from train workers:

```
┌──────────────────────────────────────────────────────────────────┐
│  Disaggregated PPO topology                                      │
│                                                                  │
│  ┌─────────────────────────┐      ┌──────────────────────────┐  │
│  │   Rollout Workers        │      │   Train Workers           │  │
│  │   (vLLM / SGLang engine) │      │   (ZeRO-3 / FSDP)        │  │
│  │                         │      │                          │  │
│  │  actor (inference mode) │─────►│  actor (grad update)     │  │
│  │  ref model (frozen)     │      │  critic (grad update)    │  │
│  │  reward model (frozen)  │      │                          │  │
│  └─────────────────────────┘      └──────────────────────────┘  │
│           │  generate responses + rewards          ▲             │
│           │  (rollout buffer)                      │ weight sync  │
│           └────────────────────────────────────────┘             │
│              sync actor weights every N steps (or each rollout)  │
└──────────────────────────────────────────────────────────────────┘
```

**Key design points:**

- **Rollout workers** load actor inference weights (FP16/BF16) and use vLLM or SGLang for continuous-batching autoregressive generation.
- **Train workers** hold the full trainable parameters (including optimizer state) via ZeRO-3 or FSDP, and execute PPO/GRPO gradient updates.
- **Weight sync**: after train workers complete a batch, they broadcast the latest actor weights to rollout workers. Sync frequency is typically once per PPO iteration (one full rollout + train cycle); some implementations support finer-grained step-by-step sync.
- **Ref model / RM**: generally reside on the rollout side in inference mode (frozen weights, no gradients), saving memory on the train side.

---

#### 4-Model Memory Breakdown + How LoRA-in-RL Saves Memory

Standard RLHF involves four models:

| Model | Parameters | Gradients | Optimizer state (AdamW) | Typical location |
|-------|-----------|-----------|------------------------|-----------------|
| **Actor** | ✅ (trainable) | ✅ | ✅ ($m, v$, FP32 ≈ 8 bytes/param) | Train workers |
| **Ref model** | ✅ (frozen) | ✗ | ✗ | Rollout workers or separate node |
| **Critic** | ✅ (trainable) | ✅ | ✅ | Train workers (can share GPU with actor) |
| **Reward Model** | ✅ (frozen) | ✗ | ✗ | Rollout workers |

**Single-model memory estimate (using a 7B model as example; order of magnitude, not exact):**

$$
M_{\text{param}} \approx 7 \times 10^9 \times 2\,\text{bytes/param (BF16)} \approx 14\,\text{GB}
$$

$$
M_{\text{opt}} \approx 7 \times 10^9 \times 8\,\text{bytes} \approx 56\,\text{GB}
$$

where $M_{\text{param}}$ is parameter memory (BF16) and $M_{\text{opt}}$ is AdamW optimizer state memory (FP32 $m$ + $v$, 8 bytes/param each). Naively co-locating all 4 models puts memory requirements in the hundreds-of-GB range — a 7B model can still fit on a single 8×80 GB machine, but naive co-location yields low GPU utilization (see above); larger models (e.g., 70B) far exceed a single node.

**Memory savings with LoRA-in-RL:**

- Only the LoRA adapters of the actor and critic are trained ($r \ll d$); pretrained weights are frozen.
- Gradients and optimizer states scale only with the LoRA parameter count. At ~99% reduction in trainable parameters (e.g., rank=16), optimizer state drops from ~56 GB to ~1 GB (order-of-magnitude estimate).
- Trade-off: LoRA's expressiveness is limited by its rank; policy update magnitude during RL may be constrained. In practice, PPO + LoRA has been validated in several public works (exact results depend on task and rank; consult original papers).

---

#### Async vs Sync Rollout — Staleness

| Mode | Description | Advantages | Disadvantages |
|------|-------------|------------|---------------|
| **Sync rollout** | Training begins only after rollout completes; the next rollout begins only after training completes | No staleness, on-policy | Low GPU utilization (two phases alternate idle) |
| **Async rollout** | Rollout workers continuously generate; train workers continuously update; weight sync is delayed | High GPU utilization, high throughput | **Staleness**: rollout uses weights from $k$ steps ago; data is off-policy |

**Impact of staleness:**

- The divergence grows between the policy $\pi_{\theta_{\text{old}}}$ used during generation and the target $\pi_\theta$ being updated.
- PPO's clipped objective tolerates mild off-policy data (via the importance ratio $r_t(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$), but when staleness is large, the variance of the importance ratio grows sharply.
- In practice, many frameworks opt for **near-synchronous** operation (syncing weights every $k$ steps), balancing throughput against staleness.

---

#### Reference Implementations: OpenRLHF vs veRL

| Dimension | **OpenRLHF** | **veRL** |
|-----------|-------------|---------|
| Focus | Research-friendly, clean, quick to get started | Production-scale, more aggressive performance optimization |
| Rollout engine | vLLM (deeply integrated) | Supports both vLLM and SGLang |
| Training parallelism | DeepSpeed ZeRO-3 | Supports both FSDP and Megatron-LM TP/PP |
| 4-model scheduling | Supports co-located and disaggregated modes | Hybrid Engine (rollout/train share GPUs with dynamic switching) |
| LoRA-in-RL | ✅ | ✅ |
| Code size | Smaller; clean architecture; good for custom extensions | Larger, but production-complete (checkpoint, fault tolerance) |
| Typical use case | Academic experiments, quick algorithm validation | Large-scale post-training pipelines |

✅ **Both are public implementations and can serve as reference skeletons for system design questions.** Consult official technical reports and GitHub for specific performance numbers — they vary significantly across versions and hardware. In interviews, cite "order-of-magnitude" rather than exact values.

---

#### Throughput Estimation: Rollout vs Train GPU-hours Ratio

⚠️ The following is a **qualitative order-of-magnitude analysis**. Actual numbers are highly sensitive to model scale, response length, and hardware configuration. In interviews, explicitly say "rough estimate" rather than citing precise benchmarks.

**Reasoning framework (using a 7B actor as example):**

- **Rollout cost**: Autoregressive generation is memory-bound; each generated token still requires a full forward pass through all layers (KV cache means only 1 new token is processed per step, but the number of layers is unchanged); throughput is limited by HBM bandwidth. With average response length $L_r$, rollout compute is roughly proportional to $B \times L_r \times \text{param\_size}$ (memory access volume).
- **Train cost**: Forward + backward ≈ $6 \times B \times s \times P$ FLOPs ($s$ = sequence length, $P$ = parameter count; forward ≈ $2P$, backward ≈ $4P$, total $6P$ per token).
- **Typical conclusion** (order of magnitude): when responses are long (hundreds of tokens), rollout GPU-hours are often **comparable to or even greater than** train GPU-hours — this is one of the core motivations for disaggregated architectures. If rollout is much faster than training, disaggregation adds limited benefit; if rollout is the bottleneck, allocating more rollout workers is the natural scaling approach.

---

### 1.6 Distributed Training Parallelism

#### Data Parallelism (DP)

Each GPU holds a complete model replica; gradients are synchronized via All-Reduce.

**Communication:** All-Reduce of parameter gradients per step = $2 \times |\theta|$ (ring all-reduce).

#### ZeRO (Zero Redundancy Optimizer)

| Stage | Sharded content | Memory per GPU |
|-------|----------------|---------------|
| ZeRO-1 | Optimizer states (Adam: $m, v$) | ~4× parameter count (same parameter memory as DP) |
| ZeRO-2 | + Gradients | ~2× parameter count |
| ZeRO-3 | + Parameters | ~$1/P$ of parameter count ($P$ = number of GPUs) |

**Overhead:** ZeRO-3 requires All-Gather of parameters during the forward pass, increasing communication volume.

**The $16\Phi$ memory breakdown (mixed-precision Adam, $\Phi$ = #params):**

| Component | Precision | Bytes/param | Memory |
|---|---|---|---|
| Model params (fp16) | fp16 | 2 | $2\Phi$ |
| Gradients (fp16) | fp16 | 2 | $2\Phi$ |
| Adam optimizer states | fp32 | 12 | $12\Phi$ |

The $12\Phi$ optimizer states = fp32 master-weight copy ($4\Phi$) + first moment $m$ ($4\Phi$) + second moment $v$ ($4\Phi$), totaling **$16\Phi$** (a 7.5B model → 120 GB, too big for one GPU). Per-GPU memory across ZeRO stages on $P$ GPUs:

| Stage | Sharded | Per-GPU memory | $P\to\infty$ |
|---|---|---|---|
| baseline (DP) | none | $16\Phi$ | $16\Phi$ |
| ZeRO-1 | optimizer states | $2\Phi + 2\Phi + \tfrac{12\Phi}{P}$ | $4\Phi$ |
| ZeRO-2 | + gradients | $2\Phi + \tfrac{14\Phi}{P}$ | $2\Phi$ |
| ZeRO-3 | + parameters | $\tfrac{16\Phi}{P}$ | $\to 0$ |

> ZeRO-3 shards all three; communication is ~1.5× of plain DP (forward all-gather params, backward all-gather params + reduce-scatter grads) — trading communication for memory. Source: Rajbhandari et al. 2020, arXiv:1910.02054.

#### Tensor Parallelism (TP)

Each layer's weight matrix is split column-wise or row-wise across multiple GPUs.

- **Column-parallel:** $Y = XA$; $A$ is split column-wise as $[A_1, A_2]$; each GPU computes $XA_i$ without communication. If followed by a row-parallel layer, one AllReduce can be fused.
- **Row-parallel:** $Y = A_1 X_1 + A_2 X_2$; each GPU computes independently, then one AllReduce.

**Megatron-LM design:** Column-parallel Linear → GeLU (local) → Row-parallel Linear → AllReduce. The entire MLP block requires only **one** AllReduce (plus one in the backward pass).

#### Pipeline Parallelism (PP)

The model is split into layer segments assigned to different machines.

- **GPipe strategy:** Split the mini-batch into $M$ micro-batches; process forward passes sequentially, then backward passes in reverse order.
- **1F1B schedule:** Alternately execute 1 forward and 1 backward, reducing pipeline bubble and peak memory.
- **Bubble rate:** $\text{Bubble} \approx (P-1) / (M + P - 1)$, $P$ = pipeline stages, $M$ = micro-batches.

#### Sequence Parallelism (SP)

Operations like LayerNorm and Dropout that carry no parameters but occupy activation memory are split along the sequence dimension.

- Ring Attention: the long sequence is split into $P$ segments across $P$ GPUs; KV is passed via ring communication, reducing activation memory from $O(N)$ to $O(N/P)$.

**Practical Guidance:**
- Single-node 8 GPUs: DP/ZeRO-2 + TP (NVLink is fast)
- Multi-node: DP/ZeRO-3 + PP (low cross-node bandwidth) + TP (within node)
- Very long contexts: add SP (Ring Attention)

---

### 1.7 KV Cache Memory Analysis

Each layer needs to cache $K$ and $V$ for every token:

$$
\text{KV cache (bytes)} = 2 \times L \times n_{\text{heads}} \times d_{\text{head}} \times s \times \text{bytes\_per\_param}
$$

- $L$ = number of layers, $n_{\text{heads}}$ = number of KV heads (fewer than Q heads with GQA), $d_{\text{head}}$ = dimension per head, $s$ = sequence length
- With FP16, bytes_per_param = 2

**PagedAttention (vLLM):** KV cache is divided into fixed-size pages (e.g., 16 tokens/page), allocated on demand, eliminating memory fragmentation and supporting more concurrent requests.

---

### 1.8 Quantization Fundamentals

**Symmetric Quantization:**

$$
x_q = \text{round}\!\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}
$$

**Asymmetric Quantization:**

$$
x_q = \text{round}\!\left(\frac{x - z}{s}\right), \quad s = \frac{x_{\max} - x_{\min}}{2^b - 1}, \quad z = x_{\min}
$$

**GPTQ — layer-wise post-training quantization via OBS (Frantar et al., ICLR 2023, arXiv:2210.17323):**
- Minimizes per-layer reconstruction error $\|WX - \hat{W}X\|_2^2$; follows OBS/OBQ using the inverse Hessian $H = 2XX^\top$ to compensate.
- After quantizing weight $q$, the error is redistributed to the **not-yet-quantized** weights via $\delta = -\dfrac{w_q - \mathrm{quant}(w_q)}{[H^{-1}]_{qq}}\,(H^{-1})_{:,q}$, canceling the output shift from quantization.
- Engineering: fixed column order (drops OBQ's greedy per-weight selection) + Cholesky factorization for stability + block updates — quantizes 175B to 3–4 bit in hours.

**AWQ — activation-aware weight quantization (Lin et al. 2023, arXiv:2306.00978):**
- Observation: weights are not equally important; the ~0.1–1% "salient weights" are identified by **activation** magnitude (not weight magnitude).
- Method: per-channel scaling of salient channels — multiply weights by $s>1$ and divide the corresponding activations by $s$ ($\hat{W}=W\,\mathrm{diag}(s),\ \hat{X}=X\,\mathrm{diag}(s)^{-1}$, with $\hat{X}\hat{W}^\top=XW^\top$ unchanged), shrinking the relative quant error on salient weights; grid-search $s$ per layer. Forward-only, no backprop.

**SmoothQuant — migrate quantization difficulty from activations to weights (Xiao et al., ICML 2023, arXiv:2211.10438):**
- Problem: activations have per-channel outliers that are very hard to quantize, while weights are smooth and easy.
- Method: per-channel smoothing $\hat{X}=X\,\mathrm{diag}(s)^{-1},\ \hat{W}=\mathrm{diag}(s)\,W$, with scale $s_j=\dfrac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$ ($\alpha\approx0.5$), moving part of the activation dynamic range into the weights to enable W8A8.

**FP8 (Hopper/H100):** E4M3 (4 exponent, 3 mantissa, range ±448) for forward weights/activations; E5M2 (5 exponent, 2 mantissa, larger range ±57344) for gradients. Vs INT8: no scale calibration, more robust to outliers.

**KV-cache quantization:** at long context the KV cache dominates memory. K has per-channel outliers → quantize per-channel; V is smoother → per-token (e.g., KIVI, arXiv:2402.02750). int8/int4/fp8 cut KV memory 2–4×; int8/fp8 with negligible loss on most tasks, while int4 is task-sensitive (long-context retrieval especially).


---

### 1.9 Speculative Decoding

**Core Idea:** A small draft model predicts $k$ tokens in parallel; the target model then verifies them all in a single forward pass.

**Accept-Reject Sampling:**
- For position $t$: if target model probability $p(x_t) \geq$ draft model probability $q(x_t)$ → accept
- If $p(x_t) < q(x_t)$, accept with probability $p(x_t)/q(x_t)$; otherwise resample from $\max(0, p(x_t) - q(x_t))$
- The output distribution is **exactly identical** to direct sampling from the target model (lossless)

**Speedup:** Depends on the token acceptance rate between the draft model and the target model. In typical scenarios a $1.5\times$–$2.5\times$ speedup is achievable.


---

### 1.10 7-Step ML System Design Framework

| Step | Name | Key points |
|------|------|-----------|
| 1 | Clarify | Data volume, model scale, QPS, latency SLA, memory budget, success metrics |
| 2 | Data | Sources, cleaning strategy, labeling approach (human / weak supervision / model-generated), data flywheel |
| 3 | Model | Architecture choice, parameter count, Pre-train vs Fine-tune vs RAG, PEFT vs full fine-tuning |
| 4 | Training Infra | Parallelism strategy (DP/TP/PP/SP), memory optimization, batch size, LR schedule |
| 5 | Evaluation | Offline benchmark + human evaluation + safety eval |
| 6 | Serving | Quantization, dynamic batching, KV cache management, latency vs throughput |
| 7 | Monitoring | Quality drift (PPL, accuracy), data distribution shift, safety incidents |

---

## Part 2 — From-Scratch Snippets

> The following are minimal educational implementations highlighting core logic, omitting production-level error handling and optimization.

### 2.1 Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    q: torch.Tensor,   # (batch, n_heads, seq_q, d_k)
    k: torch.Tensor,   # (batch, n_heads, seq_k, d_k)
    v: torch.Tensor,   # (batch, n_heads, seq_k, d_v)
    mask: torch.Tensor | None = None,  # (batch, 1, seq_q, seq_k) or broadcastable
) -> torch.Tensor:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v), attn_weights
```

### 2.2 Causal Self-Attention Layer

```python
import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)
```

### 2.3 LoRA Layer

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank delta."""

    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))  # B init to 0
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Return a new Linear with merged weights (for deployment)."""
        merged_weight = self.base.weight.data + (self.lora_b @ self.lora_a) * self.scaling
        new_linear = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        new_linear.weight.data.copy_(merged_weight)
        if self.base.bias is not None:
            new_linear.bias.data.copy_(self.base.bias.data)
        return new_linear
```

### 2.4 Grouped-Query Attention (GQA)

```python
import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads  # repeat factor
        self.d_k = d_model // n_q_heads

        self.wq = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match Q heads: (B, n_kv, T, d_k) -> (B, n_q, T, d_k)."""
        if n_rep == 1:
            return x
        B, N, T, D = x.shape
        return x[:, :, None, :, :].expand(B, N, n_rep, T, D).reshape(B, N * n_rep, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_q_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)
```

### 2.5 RoPE (Rotary Position Embedding)

```python
import torch

def precompute_rope_freqs(dim: int, max_len: int = 4096, base: float = 10000.0):
    """Precompute sin/cos tables for RoPE."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
    t = torch.arange(max_len).float()           # (max_len,)
    freqs = torch.outer(t, freqs)                # (max_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)    # each (max_len, dim/2)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor.
    x: (batch, n_heads, seq_len, d_k)
    cos, sin: (seq_len, d_k/2)
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    cos = cos.unsqueeze(0).unsqueeze(0)  # broadcast
    sin = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)
```

### 2.6 DPO Loss

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_logps_w: torch.Tensor,   # log pi_theta(y_w | x)
    policy_logps_l: torch.Tensor,   # log pi_theta(y_l | x)
    ref_logps_w: torch.Tensor,      # log pi_ref(y_w | x)
    ref_logps_l: torch.Tensor,      # log pi_ref(y_l | x)
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization loss."""
    log_ratio_w = policy_logps_w - ref_logps_w
    log_ratio_l = policy_logps_l - ref_logps_l
    logits = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(logits).mean()
```

### 2.7 KV Cache Wrapper (Minimal)

```python
import torch

class KVCache:
    """Minimal KV cache for autoregressive generation."""

    def __init__(self, max_len: int, n_heads: int, d_k: int, device: torch.device):
        self.max_len = max_len
        self.k = torch.zeros(1, n_heads, max_len, d_k, device=device)
        self.v = torch.zeros(1, n_heads, max_len, d_k, device=device)
        self.cur_len = 0

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor):
        """Append new KV from one decoding step."""
        seq_len = new_k.shape[2]
        self.k[:, :, self.cur_len:self.cur_len + seq_len] = new_k
        self.v[:, :, self.cur_len:self.cur_len + seq_len] = new_v
        self.cur_len += seq_len

    def get(self):
        """Return the current cached KV (trimmed to cur_len)."""
        return self.k[:, :, :self.cur_len], self.v[:, :, :self.cur_len]
```

### 2.8 Symmetric INT8 Quantize / Dequantize

```python
import torch

def symmetric_quantize_int8(weight: torch.Tensor):
    """Per-tensor symmetric INT8 quantization."""
    scale = weight.abs().max() / 127.0
    w_q = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return w_q, scale

def symmetric_dequantize_int8(w_q: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize INT8 back to float."""
    return w_q.float() * scale
```

---

### 2.9 Tensor-Parallel Linear (Column / Row)

```python
import torch
import torch.nn as nn

# Megatron tensor-parallel Linear hinges on a pair of conjugate comm operators f / g:
#   f: forward identity, backward all-reduce;  g: forward all-reduce, backward identity.
# Below we simulate 2-way sharding in a single process (all-reduce -> sum over shards,
# all-gather -> cat) and verify TP equals an unsharded Linear exactly.

def column_parallel(X, W, b, n_shards=2):
    """Column-parallel: split W along output dim; each rank computes X·W_iᵀ locally;
    output is feature-sharded (no comm needed to obtain the partial output)."""
    Ws, bs = torch.chunk(W, n_shards, dim=0), torch.chunk(b, n_shards, dim=0)
    outs = [X @ Wi.T + bi for Wi, bi in zip(Ws, bs)]   # local matmul per rank
    return torch.cat(outs, dim=-1)                      # g: gather (cat here)

def row_parallel(X, W, b, n_shards=2):
    """Row-parallel: input X is feature-sharded; split W along input dim;
    each rank computes a partial product, summed via all-reduce."""
    Xs, Ws = torch.chunk(X, n_shards, dim=-1), torch.chunk(W, n_shards, dim=1)
    partial = [Xi @ Wi.T for Xi, Wi in zip(Xs, Ws)]
    return sum(partial) + b                             # conjugate of f: all-reduce (sum); bias added once

# --- Verify: TP equals a plain Linear ---
torch.manual_seed(0)
B, d_in, d_out = 4, 8, 6
X = torch.randn(B, d_in)
ref = nn.Linear(d_in, d_out)
W, b = ref.weight.data, ref.bias.data                  # W: (d_out, d_in), b: (d_out,)
Y_ref = ref(X)
print("column-parallel max err:", (column_parallel(X, W, b) - Y_ref).abs().max().item())  # ~0
print("row-parallel    max err:", (row_parallel(X, W, b) - Y_ref).abs().max().item())     # ~0
```

> A Megatron MLP chains **column-parallel → GeLU (local) → row-parallel**, so the whole block needs only **one** all-reduce in the forward pass (one in backward) — minimizing communication.

---

## Part 3 — Interview Questions

### L1 — Basic

<details>
<summary>Q1: What is the time complexity of self-attention in a Transformer? How can it be reduced?</summary>

**Answer:** Standard self-attention has time complexity $O(n^2 d)$ ($n$ = sequence length, $d$ = dimension) because it must compute an $n \times n$ attention matrix. Methods to reduce it include:
- **FlashAttention:** Does not change the mathematical result; reduces wall-clock time by tiling and recomputation to minimize HBM accesses
- **Sparse Attention:** Longformer/BigBird uses local windows + global tokens, reducing complexity to $O(n \cdot w)$
- **Linear Attention:** Approximates softmax with a kernel function, reducing complexity to $O(n d^2)$, but usually with some accuracy loss

**Follow-up:** Why is FlashAttention not considered an "approximate" attention? What low-level optimizations does it perform?

> FlashAttention loads Q, K, V in tiles into SRAM, computes the softmax online normalization in SRAM (by maintaining a running max and running sum), then writes the result back to HBM. It is mathematically equivalent to standard attention — it merely reduces the number of HBM read/write operations.

</details>

---

<details>
<summary>Q2: What is Layer Normalization? How does it differ from Batch Normalization?</summary>

**Answer:**
- **BatchNorm:** Computes mean and variance across the batch dimension for each feature. Requires maintaining running mean/var during training and uses fixed statistics at inference. Sensitive to batch size; not suitable for variable-length sequences.
- **LayerNorm:** Computes mean and variance across the feature dimension for each sample (each token is normalized independently); does not depend on batch statistics. The standard choice in Transformers.

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

**Follow-up:** What advantage does RMSNorm have over LayerNorm?

> RMSNorm removes the mean-centering step and only performs variance normalization: $\text{RMSNorm}(x) = \gamma \odot x / \sqrt{\text{mean}(x^2) + \epsilon}$. Slightly less computation, similar practical performance; adopted by the LLaMA series.

</details>

---

<details>
<summary>Q3: What is gradient clipping? Why is it nearly universal in LLM training?</summary>

**Answer:** Gradient clipping constrains the norm of the gradient to a threshold:
$$
\text{if } \|g\| > c: \quad g \leftarrow g \cdot \frac{c}{\|g\|}
$$
In LLM training, a small number of anomalous samples can produce extremely large gradients (gradient spikes), causing abrupt parameter changes or NaN values. Gradient clipping (typically $c = 1.0$) is the standard safeguard against training collapse.

**Follow-up:** How can you tell whether the gradient clipping threshold is set appropriately?

> Monitor the frequency of clipping events in the training log. Occasional triggering (< 5% of steps) is normal; frequent triggering suggests the learning rate may be too large; never triggering during otherwise stable training suggests the threshold may be too loose.

</details>

---

<details>
<summary>Q4: What are warmup and cosine decay? Why are they commonly used in LLM pre-training?</summary>

**Answer:**
- **Warmup:** Linearly increase the learning rate at the beginning of training (typically for the first 1%–3% of steps), because with random initialization the gradient direction is unstable and a large LR may cause divergence.
- **Cosine decay:** After warmup, the LR follows a cosine curve from peak to near zero: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$

**Follow-up:** How does a WSD (Warmup-Stable-Decay) schedule differ from a cosine schedule?

> WSD maintains a constant LR after warmup (the stable phase), then decays rapidly at the end. Its advantage is that mid-training checkpoints have good quality, making it suitable for scenarios that need to evaluate downstream tasks from intermediate checkpoints.

</details>

---

<details>
<summary>Q5: Explain the basic principle of FlashAttention and why it is faster.</summary>

**Answer:** The core of FlashAttention is an **IO-aware** algorithm design:
1. Split Q, K, V into small blocks, each small enough to fit in GPU SRAM (on-chip memory)
2. Compute softmax and matrix multiplication within SRAM
3. Use **online softmax** (maintaining row-wise running max and sum statistics) to avoid needing global information for the softmax
4. Avoid writing the $n \times n$ attention matrix to HBM (GPU memory), thereby reducing HBM reads and writes

Source of speedup: standard attention must write/read the attention matrix from HBM, making HBM bandwidth the bottleneck. FlashAttention concentrates computation in SRAM, reducing HBM access from $O(n^2)$ to $O(n^2 d^2 / M)$ ($M$ = SRAM size).

**Follow-up:** How large is the benefit of FlashAttention for training vs inference respectively?

> In training, the main saving is in HBM accesses during backpropagation (the attention matrix is not stored in the forward pass; it is recomputed as needed during backward). In inference, the benefit is primarily in the prefill stage (long prompts); the benefit in the decode stage (single token per step) is smaller.

</details>

---

<details>
<summary>Q6: What is PEFT (Parameter-Efficient Fine-Tuning)? Name at least three methods and briefly describe each.</summary>

**Answer:**
- **LoRA / QLoRA:** Insert a low-rank bypass ($BA$) alongside a weight matrix; only the bypass parameters are trained. QLoRA further quantizes the base weights to 4-bit.
- **Prefix Tuning:** Prepend trainable "virtual token" vectors to the keys and values of each attention layer.
- **Adapter:** Insert a small MLP bottleneck (down-projection → nonlinearity → up-projection) between Transformer sublayers; only adapter parameters are trained.
- **Prompt Tuning:** Prepend a small number of trainable soft prompt vectors to the input embeddings (only at the input layer).

**Follow-up:** What is the trade-off between parameter efficiency and expressiveness for these methods?

> Fewer parameters save more memory but cap the expressiveness. LoRA, acting directly on weight matrices, typically outperforms adapters and prefix tuning at similar parameter counts. In extreme scenarios (e.g., only tens of examples), fewer parameters can actually prevent overfitting.

</details>

---

<details>
<summary>Q7: What is the difference between continuous batching and static batching?</summary>

**Answer:**
- **Static batching:** Collects a batch of requests and waits until all requests finish generation before releasing the batch. If one request is much shorter than others, GPU resources are wasted once it completes (padding and idle waiting).
- **Continuous batching (iteration-level scheduling):** After each generation step (one token), checks whether any request has finished; completed requests are immediately replaced by new ones. GPU utilization is significantly improved.

**Follow-up:** Are PagedAttention and continuous batching used together?

> Yes. Continuous batching answers "when to schedule requests"; PagedAttention answers "how to allocate KV cache memory" — it splits the KV cache into fixed-size pages, allocates on demand, and avoids memory fragmentation caused by variable request lengths.

</details>

---

### L2 — Intermediate

<details>
<summary>Q8: Explain what each of the three ZeRO stages does, and their respective communication overhead.</summary>

**Answer:**
- **ZeRO-1:** Shards optimizer states (Adam's $m$ and $v$, 8 bytes/param in FP32) across GPUs. Each GPU holds only $1/P$ of the optimizer state; after the update, parameters are gathered via AllGather.
- **ZeRO-2:** On top of ZeRO-1, also shards gradients. Each GPU keeps only $1/P$ of the gradients (the rest are discarded after Reduce-Scatter).
- **ZeRO-3:** Also shards parameters. During forward and backward passes, parameters are All-Gathered on demand and released after use.

Communication: ZeRO-1 and ZeRO-2 have the same communication as standard DP ($2|\theta|$ per step). ZeRO-3 adds an extra AllGather during the forward pass (~$1.5\times$ total communication volume).

**Follow-up:** When is ZeRO-2 better than ZeRO-3?

> When model parameters fit on a single GPU but optimizer states do not, ZeRO-2 has lower communication overhead. A typical scenario is fine-tuning a medium-scale model (e.g., 7B–13B) with gradient checkpointing + ZeRO-2.

</details>

---

<details>
<summary>Q9: What is the concrete RLHF-PPO training loop? Why is KL penalization needed?</summary>

**Answer:** Each RLHF-PPO step:
1. Sample a batch of prompts; generate responses with the current policy $\pi_\theta$
2. Score each (prompt, response) pair with the reward model
3. Compute the KL penalty using the reference policy $\pi_{\text{ref}}$
4. Compute advantages (typically with GAE)
5. Update the policy with the PPO clip objective (multiple mini-batch update rounds)

**Why KL penalization is needed:** Without a KL constraint, the policy quickly drifts into out-of-distribution (OOD) blind spots of the reward model — generating responses that score high under the RM but that humans actually dislike (reward hacking). The KL penalty keeps the policy close to $\pi_{\text{ref}}$ (i.e., the SFT model).

**Follow-up:** Can you give a concrete example of reward hacking?

> For example, if the reward model favors long answers (because good answers in the training data tended to be longer), the policy may learn to produce very long, repetitive responses to any question to obtain a high score — even though a human evaluator would find them verbose and unhelpful.

</details>

---

<details>
<summary>Q10: How do you prevent catastrophic forgetting during instruction tuning?</summary>

**Answer:** Common approaches:
- **Replay / mixed training:** Mix some general instruction data or pretraining data into the SFT data
- **LoRA / PEFT:** Only a small number of parameters are updated; pretrained knowledge is preserved in the frozen base weights
- **Regularization:** Methods like EWC (Elastic Weight Consolidation) penalize large deviations of important parameters
- **Low learning rate:** When doing full fine-tuning, use an LR 1–2 orders of magnitude lower than pre-training

**Follow-up:** How do you quantify the degree of catastrophic forgetting?

> Evaluate on both a general benchmark (e.g., MMLU, HellaSwag) and the target task benchmark before and after fine-tuning. If performance on the general benchmark drops by more than a few percentage points, significant forgetting has occurred.

</details>

---

<details>
<summary>Q11: What is the core difference between GPTQ and AWQ?</summary>

**Answer:**
- **GPTQ (Optimal Brain Quantization series):** Quantizes layer by layer, using second-order information (inverse Hessian) to minimize the reconstruction error of the layer output. Quantizes column by column; after quantizing each column, compensates the remaining columns.
- **AWQ (Activation-Aware Weight Quantization):** The core observation is that a small number of "salient channels" (channels with large activations) are critical to output quality. AWQ protects the weights of those channels (e.g., using per-channel scaling to effectively increase precision), rather than quantizing all weights uniformly.


**Follow-up:** When quantizing to INT4, why is SmoothQuant on activations so important?

> Activations often contain outliers (abnormally large values), which stretch the quantization range and reduce effective precision. SmoothQuant migrates activation outliers to weights via a mathematically equivalent per-channel scaling, making the activation distribution more uniform so that both weights and activations can be quantized to lower bit-widths.

</details>

---

<details>
<summary>Q12: How do Sequence Parallelism and Tensor Parallelism work together?</summary>

**Answer:** In Megatron-LM's design:
- TP splits linear layers (weight matrices of attention and MLP)
- SP (Sequence Parallelism) splits activations of **non-linear operations** (LayerNorm, Dropout) — along the sequence dimension
- Junction: a TP layer ends with AllReduce (or ReduceScatter); an SP layer also requires communication. Megatron-LM fuses these two communication operations, so no additional communication is introduced.

Benefit: after the TP AllReduce, each GPU holds the full sequence in its activations (redundant). SP eliminates this redundancy; each GPU holds only $1/P$ of the sequence activations, significantly reducing activation memory.

**Follow-up:** Does SP help with gradient checkpointing?

> Yes. SP reduces the activation volume stored on each GPU. Without gradient checkpointing, activation memory drops from $O(L \cdot n \cdot d)$ to $O(L \cdot n/P \cdot d)$. Even with gradient checkpointing, the temporary memory used during recomputation is reduced proportionally.

</details>

---

<details>
<summary>Q13: Explain how a reward model is trained in RLHF and how reward model quality is evaluated.</summary>

**Answer:**
- **Training:** Uses the Bradley-Terry preference model. Given prompt $x$ and a response pair $(y_w, y_l)$ ($y_w$ labeled as better), the reward model loss is $-\log \sigma(r(x, y_w) - r(x, y_l))$. The model is typically initialized from the SFT model, with the language model head replaced by a scalar output head.
- **Evaluation metrics:**
  - **Preference prediction accuracy:** Accuracy at predicting which of a held-out preference pair is better
  - **Reward distribution separability:** Whether the reward distributions of chosen and rejected responses are sufficiently separated
  - **Reward hack robustness:** Whether the reward still ranks OOD responses (generated by the policy) reasonably

**Follow-up:** Why does the reward model need to be updated periodically?

> Because the policy changes continuously during RL training, the distribution of generated responses gradually shifts away from the RM's training distribution (the SFT model's output distribution). On out-of-distribution data, the RM may produce inaccurate scores, leading to reward hacking.

</details>

---

<details>
<summary>Q14: What problem does vLLM's PagedAttention solve? What is the mechanism?</summary>

**Answer:** 
- **Problem:** Traditional KV cache pre-allocates a contiguous memory block for each request (at maximum sequence length). But actual generation lengths vary, causing significant memory waste (internal fragmentation) and preventing sharing across requests (external fragmentation).
- **PagedAttention mechanism:** Borrows the virtual memory paging idea from operating systems:
  1. KV cache is divided into fixed-size **blocks** (e.g., each block stores KV for 16 tokens)
  2. A **block table** records the mapping from logical blocks to physical blocks for each request
  3. New blocks are dynamically allocated when new tokens are generated; blocks are freed when a request finishes
  4. Supports **copy-on-write**: for beam search candidates sharing the same prefix, KV blocks can be shared

**Follow-up:** Does PagedAttention negatively affect latency?

> The indirect addressing via the block table introduces a small overhead (compared to direct access with contiguous memory), but in practice this overhead is very small (typically < 5%), since attention computation itself is compute-bound or memory-bound and the addressing overhead is not the bottleneck.

</details>

---

<details>
<summary>Q15: How would you design an offline evaluation harness for an LLM? What aspects need to be considered?</summary>

**Answer:**
- **Task abstraction:** Each task defines a dataset, prompt template (few-shot format), metric, and output type (generation / loglikelihood)
- **Evaluation modes:**
  - Likelihood-based (e.g., MMLU): compute log-prob for each option, select the maximum
  - Generation-based (e.g., GSM8K): generate output then evaluate with rules / code execution
  - LLM-as-judge (e.g., MT-Bench): score with a stronger model
- **Reproducibility:** Fix seeds; record prompt templates and few-shot examples; temperature=0 (or fixed)
- **Efficiency:** Likelihood tasks suit large batches; generation tasks should be sorted by length to reduce padding
- **Contamination detection:** Check for n-gram overlap between training data and test sets

**Follow-up:** Why distinguish between "knowledge" and "reasoning" evaluation?

> Because a model may perform well on knowledge-heavy tasks (e.g., factual questions in MMLU) but poorly on reasoning-heavy tasks (e.g., math, code), or vice versa. Evaluating them separately helps pinpoint where the model's capabilities are weak.

</details>

---

<details>
<summary>Q16: How do you choose an appropriate LoRA rank for LLM fine-tuning?</summary>

**Answer:** Factors to consider:
- **Task complexity:** Simple classification/extraction tasks usually need only r=4–16; complex reasoning/generation tasks may require r=32–64
- **Data size:** Use a small rank with little data to prevent overfitting; increase rank when data is plentiful to raise capacity
- **Target modules:** Applying LoRA only to q_proj and v_proj (fewest parameters) vs. all linear layers (q/k/v/o + MLP gate/up/down) trades parameter count against effectiveness — all linear layers typically yield better results
- **Common practice:** Start from r=16, α=2r; compare r=8/16/32/64 on the validation set

**Follow-up:** Can LoRA be combined with QLoRA? Is the accuracy loss large when using 4-bit quantized base weights + LoRA?

> Yes — QLoRA is exactly this approach. In practice, 4-bit NF4 quantized base weights + LoRA fine-tuning matches FP16 full fine-tuning within an acceptable margin on most tasks (typically within 1–3 percentage points) while saving enormous memory.

</details>

---

<details>
<summary>Q-RLHF-A (L2): Why is GPU utilization low in naive co-located PPO? How does a disaggregated architecture solve it?</summary>

**Answer:**

Naive co-located PPO runs rollout and training serially on the same set of GPUs:

- **Rollout phase**: the actor performs autoregressive inference (memory-bound; throughput limited by HBM bandwidth); the trainer waits idle.
- **Train phase**: PPO backpropagation is compute-intensive; the rollout worker waits idle.

The two phases alternate, and overall GPU utilization is the weighted average of each phase's utilization — well below peak training utilization.

**How the disaggregated architecture solves it:**
1. Independent **rollout workers** (vLLM/SGLang engines) continuously generate responses, producing a rollout buffer.
2. Independent **train workers** (ZeRO-3/FSDP) pull data from the buffer and continuously execute PPO/GRPO updates.
3. The two sets of workers run **concurrently**; weights are synchronized at a fixed frequency (typically each iteration).

This way, rollout and training are each optimized for their own workload (inference engine vs. training framework) without blocking each other.

**Follow-up 1:** How much weight-sync bandwidth is needed between rollout workers and train workers in a disaggregated architecture?

> For a 7B BF16 model, one complete weight sync transfers ~14 GB. If syncing once per minute, that is ~14 GB ÷ 60 s ≈ 0.23 GB/s, well below the bandwidth ceiling of NVLink/RDMA (sync overhead is negligible). With LoRA-in-RL, only LoRA parameters need syncing (~100 MB scale), greatly reducing sync overhead.

**Follow-up 2:** What effect does staleness from async rollout have on PPO? How can it be mitigated?

> Staleness causes rollout to generate data with old parameters $\pi_{\theta_{\text{old}}}$, introducing an off-policy bias. PPO's importance ratio clip ($\epsilon \approx 0.1\text{–}0.2$) tolerates mild staleness, but when staleness is large, gradient estimate variance grows and training becomes unstable. Mitigation: control the weight sync frequency (no more than a few mini-batch updates), or use more aggressive importance sampling correction.

</details>

---

### L3 — Deep

<details>
<summary>Q17: How does Megatron-LM's Column-Parallel and Row-Parallel Linear reduce the number of AllReduce operations?</summary>

**Answer:**

Consider two consecutive linear transforms $Y = GELU(XA)B$ (an MLP block), $A \in \mathbb{R}^{h \times 4h}$, $B \in \mathbb{R}^{4h \times h}$:

1. **Column-Parallel $A$:** Split $A$ column-wise as $[A_1, A_2]$; each GPU computes $GELU(X A_i)$ independently, **no communication needed**. GeLU is element-wise and naturally separable.
2. **Row-Parallel $B$:** Split $B$ row-wise as $\begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$; each GPU computes $Y_i = GELU(XA_i) B_i$.
3. **Final AllReduce:** $Y = Y_1 + Y_2$ (one AllReduce).

Key insight: Column-Parallel output is exactly the input of Row-Parallel; the intermediate nonlinearity (GeLU) is element-wise and requires no communication. Therefore **the entire MLP block needs only one AllReduce** (forward), and one in the backward pass as well.

Without this design, each layer would require a separate AllReduce, doubling the communication volume.

**Follow-up:** Can the same trick be applied to the QKV projection and output projection in the attention block?

> Yes. QKV projection uses Column-Parallel (outputs are distributed to each head, which naturally splits column-wise); the output projection uses Row-Parallel, followed by AllReduce. The entire attention block also needs only one AllReduce.

</details>

---

<details>
<summary>Q18: Why is speculative decoding lossless? Derive the acceptance probability.</summary>

**Answer:**

Let target model distribution be $p(x)$ and draft model distribution be $q(x)$.

**Accept-reject sampling:**
1. Sample token $x$ from $q(x)$
2. If $p(x) \geq q(x)$: accept (probability 1)
3. If $p(x) < q(x)$: accept with probability $p(x)/q(x)$

**Total probability of accepting token $x$:**
- Sampled from $q$ and accepted: $q(x) \cdot \min(1, p(x)/q(x)) = \min(p(x), q(x))$
- Sampled from $q$, rejected, then resampled to $x$: more complex but derivable

**Final effective probability:**

$$
P(\text{output}=x) = \min(p(x), q(x)) + \frac{\max(0, p(x) - q(x))}{1 - \sum_i \min(p(i), q(i))} \cdot \delta
$$

In fact, it can be proven that through this rejection sampling + correction sampling, the final output distribution **exactly equals** $p(x)$.

Core intuition: when $p(x) > q(x)$, the draft model "under-sampled" $x$ and the deficit is compensated from the residual probability mass after rejection; when $p(x) < q(x)$, rejection removes the excess probability.

**Follow-up:** Where is the efficiency bottleneck of speculative decoding?

> The bottleneck is the draft model's acceptance rate. If the distributions of the draft model and target model diverge significantly, the acceptance rate is low, most draft tokens are rejected, and the speedup is poor. Improvements include: Medusa-style multi-head prediction, or selecting a draft model whose distribution is closer to the target model.

</details>

---

<details>
<summary>Q19: How is DPO derived from the Bradley-Terry preference model?</summary>

**Answer:**

**Step 1:** The Bradley-Terry model assumes the optimal policy $\pi^*$ satisfies:

$$
p(y_w \succ y_l | x) = \sigma(r^*(x, y_w) - r^*(x, y_l))
$$

**Step 2:** Under a KL constraint, the closed-form solution for the optimal policy is:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

where $Z(x)$ is the partition function.

**Step 3:** Solve for the reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

**Step 4:** Substitute $r$ into the Bradley-Terry model; $Z(x)$ cancels in the difference:

$$
p(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

**Step 5:** Replace $\pi^*$ with the trainable $\pi_\theta$ and take the negative log-likelihood to obtain the DPO loss.

**Follow-up:** DPO's derivation assumes preference data comes from the optimal policy; what practical problems does this assumption cause?

> In practice, preference data usually comes from the SFT model (not the optimal policy), which means the reward implicitly learned by DPO may be inaccurate. This is also why online DPO (iterative DPO, where each round generates data with the latest policy) typically outperforms offline DPO.

</details>


---

<details>
<summary>Q20: What is benchmark saturation in LLM evaluation, and how do you address it?</summary>

**Answer:**
- **Problem:** When mainstream models score near the ceiling on a benchmark (e.g., >90% on MMLU), discriminability drops. Possible causes include:
  - Training data contamination (test set data leaked into the training set)
  - Insufficient task difficulty (primarily knowledge retrieval, not deep reasoning)
  - Format optimization (models tuned to the benchmark's prompt format)
- **Approaches:**
  - Use harder benchmarks (e.g., MMLU-Pro, GPQA, MATH)
  - Use dynamically generated evaluation questions
  - Rely on human evaluation (e.g., Chatbot Arena Elo rankings)
  - Detect and report data contamination

**Follow-up:** What are the design philosophies of HELM and lm-evaluation-harness?

> HELM (Stanford) emphasizes "comprehensiveness" — covering multiple dimensions (accuracy, calibration, robustness, fairness, efficiency) with detailed documentation and standardized evaluation procedures for each scenario, but extending to new tasks is relatively heavy. lm-evaluation-harness (EleutherAI) emphasizes "flexibility and community contribution" — tasks are defined concisely (config-driven); the community can quickly add new tasks; 400+ tasks provide broad coverage, though standardization is relatively lower.

</details>

---

<details>
<summary>Q21: Explain the motivation and design of disaggregated serving (prefill/decode separation).</summary>

**Answer:**

**Motivation:** Prefill (processing the prompt) and decode (generating tokens one at a time) have completely different computational characteristics:

| Characteristic | Prefill | Decode |
|---------------|---------|--------|
| Computation type | Compute-bound (large matrix multiplications) | Memory-bound (small batch, heavy KV cache access) |
| GPU utilization | High (compute-intensive) | Low (memory bandwidth bottleneck) |
| Optimal configuration | High-compute GPU | High-bandwidth memory GPU |

**Disaggregated Serving Design:**
- Prefill nodes: high-compute configuration; process prompts in large batches → generate KV cache
- Decode nodes: high-bandwidth configuration; receive KV cache → generate tokens one at a time
- KV cache is transferred between nodes over a high-speed network (RDMA/NCCL)

**Benefits:** The two stages can be scaled independently, preventing the memory-bound nature of the decode stage from dragging down the compute utilization of the prefill stage.

**Follow-up:** How large is the bandwidth requirement for KV cache transfer?

> For a 70B model with sequence length 4K and FP16 KV cache, the KV cache per request is on the order of a few hundred MB. If decode nodes need to ingest KV caches from tens of requests per second, tens of GB/s of network bandwidth is required — feasible on modern data-center RDMA networks.

</details>


---

<details>
<summary>Q22: How do you manage the memory–compute trade-off of gradient checkpointing in distributed training?</summary>

**Answer:**
- **Principle:** During the forward pass, intermediate activations are not saved; only some "checkpoints" are kept (typically one per layer boundary). During backpropagation, activations are recomputed from the nearest checkpoint.
- **Memory:** Reduced from $O(L \cdot a)$ ($a$ = activation size per layer) to $O(\sqrt{L} \cdot a)$ or $O(L')$ ($L'$ = number of checkpoints)
- **Compute:** Approximately 33% extra forward computation (each checkpoint segment is recomputed forward once)

**Practical choice:**
- Do not use if memory is sufficient (saves time)
- Enable when memory is insufficient but a 33% training slowdown is acceptable
- Can be selectively enabled (e.g., only for certain large layers)

**Follow-up:** For selective gradient checkpointing, how do you choose which layers to checkpoint?

> Typically choose layers with the largest activations (e.g., the attention matrix is an $O(n^2)$ memory consumer). Layers with small activations (e.g., LayerNorm, embedding) are not checkpointed, achieving a better balance between memory savings and computation overhead.

</details>

---

<details>
<summary>Q23: Explain PPO's clipping mechanism and why it may need adjustment in RLHF.</summary>

**Answer:**

PPO's clipped surrogate objective:

$$
L^{CLIP} = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$; $\epsilon$ is typically 0.1–0.2.

**Purpose:** When $r_t$ deviates too far from 1, the clip limits the change in the objective function, preventing excessively large single-step updates.

**Special considerations in RLHF:**
- In standard RL (games, etc.), the state-action space is large and $r_t$ does not deviate much
- In RLHF, the language model's generation space is exponential and the policy may change rapidly
- Therefore $\epsilon$ may need to be reduced, or the number of PPO update epochs increased to better exploit each batch of samples

**Follow-up:** How is the value function loss balanced against the policy loss in PPO?

> Typically a weighted sum: $L = L^{CLIP} + c_1 L^{VF} - c_2 H(\pi)$, where $L^{VF}$ is the MSE loss of the value function and $H(\pi)$ is an entropy bonus to prevent premature collapse. In RLHF, tuning $c_1$ and $c_2$ is critical for training stability.

</details>

---

<details>
<summary>Q24: How would you design a system to detect benchmark data contamination?</summary>

**Answer:**
- **N-gram overlap detection:** Compute the intersection of n-grams (e.g., 8-gram, 13-gram) from the test set against the training data. If the overlap rate exceeds a threshold, flag as potentially contaminated.
- **Membership inference:** Check whether the model's perplexity on test set samples is anomalously low compared to held-out data; low perplexity may indicate the sample appeared in training.
- **Canonical order test:** Shuffle the answer choices; if accuracy drops significantly, the model may have memorized the answer at a specific position (suggesting contamination rather than genuine understanding).
- **Canary test:** Insert unique "canary" sentences into the test set; after training, check whether the model can reproduce them perfectly.

**Follow-up:** Why might n-gram overlap detection produce false positives?

> Because some common knowledge (e.g., "the sun rises in the east") will appear in both training and test sets; n-gram overlap does not mean genuine "memorization." One needs to distinguish "factual public knowledge" from "verbatim copying of specific test samples."

</details>

---

<details>
<summary>Q-RLHF-B (L3): Design an RLHF training system supporting a 70B actor. Describe the 4-model memory decomposition, rollout/train topology, and how you would choose between LoRA-in-RL vs full parameter updates.</summary>

**Answer:**

**Step 1: Clarify**
- 70B actor (~140 GB BF16 parameters) + critic (similar or slightly smaller) + ref model + RM
- Naive co-located memory for all 4 models: parameters + optimizer states on the order of 1 TB (not feasible; separation required)
- Goal: run on 8–64 × 80 GB A100/H100 GPUs with throughput that meets a reasonable training schedule

**Step 2: 4-model memory decomposition (order-of-magnitude estimates)**

| Model | Parameters (BF16) | Gradients | Optimizer (FP32 AdamW) | Deployment strategy |
|-------|------------------|-----------|----------------------|---------------------|
| Actor (trainable) | ~140 GB | ~140 GB | ~560 GB | Train workers, ZeRO-3 sharding |
| Critic (trainable) | ~140 GB (smaller model possible) | ~140 GB | ~560 GB | Same, or separate ZeRO group |
| Ref model (frozen) | ~140 GB | None | None | Rollout workers, inference mode |
| Reward model (frozen) | few GB–~140 GB | None | None | Rollout workers |

- With full parameter training, the complete training state for actor + critic (parameters + gradients + optimizer) is ~1.5–2 TB scale; ZeRO-3 sharding across train workers requires **tens of** 80 GB GPUs (exact count depends on whether FP32 master copy, activation, and framework overhead are included).
- With **LoRA-in-RL** (rank=16–32), trainable parameters drop to $\lesssim 1\%$ of total, optimizer states fall from ~560 GB to a few GB — greatly reducing train worker memory requirements.

**Step 3: Topology design**

```
Rollout cluster (inference-optimized)    Train cluster (training-optimized)
┌──────────────────────────┐         ┌─────────────────────────┐
│ vLLM / SGLang            │         │ ZeRO-3 / FSDP           │
│  - actor (FP16 weights)  │◄─weights│  - actor (trainable)    │
│  - ref model (frozen)    │  sync   │  - critic (trainable)   │
│  - RM (frozen)           │         │                         │
│                          │──data──►│  rollout buffer         │
│  continuous rollout,     │         │  PPO / GRPO updates     │
│  output                  │         │                         │
│  (prompt, resp, reward,  │         │                         │
│   log_prob, value)       │         │                         │
└──────────────────────────┘         └─────────────────────────┘
```

- Rollout and train run **concurrently** (async) or **alternately** (sync); weights synced once per iteration.
- Ref model and RM require only inference; placing them on the rollout side saves train-side memory.

**Step 4: LoRA-in-RL vs full parameter updates**

| Consideration | Favors LoRA-in-RL | Favors full updates |
|---------------|-------------------|---------------------|
| Memory budget | Tight (fewer GPUs) | Abundant (many GPUs) |
| Required policy change magnitude | Small (conversational style alignment) | Large (complex reasoning improvement) |
| Training stability | More stable (low-rank constraint) | Needs more careful tuning of $\beta$, clip |
| Reference | OpenRLHF LoRA mode | veRL / Megatron-LM full parameters |

⚠️ The memory figures above are **order-of-magnitude estimates** (derived from parameter count × bytes/param formulas). Actual values differ substantially due to activations, KV cache, and framework overhead. In interviews, explicitly state "estimate."

**Follow-up:** How do you decide the resource ratio of rollout to train workers in a disaggregated architecture?

> It depends on the ratio of rollout throughput to train throughput. If rollout is the bottleneck (long responses, large batches), add more rollout workers. If train is the bottleneck (large critic, many PPO mini-batches), add more train workers. In practice, first profile the GPU-hours per iteration for each side, allocate proportionally, then adjust based on observed queue utilization.

</details>

---

<details>
<summary>Q25: Comprehensive design question: Design a complete LLM system for an AI customer service application with 10 million daily active users, from data to deployment.</summary>

**Answer (high-level overview):**

**1. Clarify:**
- 10M DAU → estimated QPS of ~100–1000 (assuming 1–3 conversation turns per user per day)
- Latency SLA: P95 < 2s (time to first token), P99 < 5s
- Domain adaptation needed (customer service phrasing, product knowledge)

**2. Data:**
- Historical customer service conversation logs → clean and anonymize → build SFT data
- Periodically sample online bad cases (low ratings, escalated to human agents) → human annotation → feed back into training
- RAG: build a vector knowledge base from product documentation and FAQs

**3. Model:**
- Base model: 7B–13B scale (balance quality and inference cost)
- SFT (LoRA) fine-tuned on customer service data
- RAG retrieval augmentation: user query → retrieve relevant documents → append to prompt context

**4. Serving:**
- Quantization: INT8 or INT4 (GPTQ/AWQ) → reduce per-GPU inference cost
- vLLM / TensorRT-LLM deployment, continuous batching + PagedAttention
- Multiple replicas + load balancing, auto-scaling with traffic

**5. Monitoring:**
- Online metrics: escalation rate, user satisfaction score, average conversation turns
- Quality drift: regularly run eval on a standard test set and monitor score changes
- Safety: apply sensitive word and harmful content filtering to outputs

**Follow-up:** In this system, what problems do RAG and fine-tuning each solve? Can they replace each other?

> Fine-tuning handles "style and format" — making the model respond in a customer-service tone and follow the correct workflow. RAG handles "knowledge and facts" — providing up-to-date product information and company policy. They are complementary, not interchangeable: fine-tuning alone causes hallucinations about product details; RAG alone makes the model sound like a generic assistant rather than a professional customer service agent. The ideal solution combines both.

</details>

---

## Appendix: Key Term Glossary

| Chinese | English | Abbreviation |
|---------|---------|-------------|
| 因果语言模型 | Causal Language Model | CLM |
| 低秩适配 | Low-Rank Adaptation | LoRA |
| 参数高效微调 | Parameter-Efficient Fine-Tuning | PEFT |
| 人类反馈强化学习 | Reinforcement Learning from Human Feedback | RLHF |
| 直接偏好优化 | Direct Preference Optimization | DPO |
| 奖励模型 | Reward Model | RM |
| 数据并行 | Data Parallelism | DP |
| 张量并行 | Tensor Parallelism | TP |
| 流水线并行 | Pipeline Parallelism | PP |
| 序列并行 | Sequence Parallelism | SP |
| 零冗余优化器 | Zero Redundancy Optimizer | ZeRO |
| 完全分片数据并行 | Fully Sharded Data Parallel | FSDP |
| 键值缓存 | Key-Value Cache | KV Cache |
| 训练后量化 | Post-Training Quantization | PTQ |
| 基于激活感知的权重量化 | Activation-Aware Weight Quantization | AWQ |
| 投机解码 | Speculative Decoding | — |
| 分页注意力 | PagedAttention | — |
| 检索增强生成 | Retrieval-Augmented Generation | RAG |
| 指令微调 | Instruction Tuning / SFT | SFT |
| 灾难性遗忘 | Catastrophic Forgetting | — |
| 知识蒸馏 | Knowledge Distillation | KD |
| 领域自适应预训练 | Domain-Adaptive Pretraining | DAP |

---



## Extended L3

<details>
<summary>Q26: Explain the IO-aware tiling strategy in FlashAttention. Why does standard attention have a memory access bottleneck, and how does the online softmax trick enable block-wise computation without materializing the full N×N attention matrix?</summary>

Standard attention must write the complete $N \times N$ attention matrix to HBM (High Bandwidth Memory), making IO the bottleneck. FlashAttention uses GPU SRAM (fast but small) via tiling:

1. Split $Q, K, V$ into blocks of size $B_r \times d$ and $B_c \times d$; load only one block into SRAM at a time
2. For each Q block, iterate over all K/V blocks and compute local attention within SRAM
3. Use **online softmax** by maintaining a running max $m$ and running sum $\ell$: after processing the $j$-th KV block, update the previously accumulated output $O_j$ with a correction factor $e^{m_{j-1} - m_j}$, avoiding the need for global normalization

$$O_j = \text{diag}(\ell_j)^{-1}\Big(e^{m_{j-1}-m_j}\,\ell_{j-1}\,O_{j-1} + \tilde{P}_j V_j\Big)$$

IO complexity drops from $O(N^2 d)$ HBM accesses to $O(N^2 d^2 / M)$ ($M$ = SRAM size); memory drops from $O(N^2)$ to $O(N)$ (the full attention matrix is never materialized).

> **Follow-up:** FlashAttention's backward pass requires recomputing the attention matrix (recomputation) — what are the similarities and differences with gradient checkpointing? In very-long-sequence settings, what additional parallelization optimizations does FlashAttention v2 introduce?

</details>

---

<details>
<summary>Q27: How does RoPE's NTK-aware interpolation address the long-sequence extrapolation problem? Why does simple position interpolation lose high-frequency information?</summary>

Simple position interpolation (PI) uniformly scales position $m$ to $m \cdot L_{\text{train}} / L_{\text{target}}$. The problem is that RoPE frequencies $\theta_j = 10000^{-2j/d}$ span multiple orders of magnitude:
- **Low dimensions** (small $j$) → high frequency, encoding fine-grained positional relationships at short distances
- **High dimensions** (large $j$) → low frequency, encoding coarse positional relationships at long distances

After uniform scaling, the rotation angle in high-frequency dimensions changes too densely; the model cannot distinguish adjacent tokens (high-frequency information is "compressed together"), analogous to applying a low-pass filter to an image and losing edge details.

**NTK-aware interpolation** rescales the base frequency from $b$ to $b' = b \cdot \alpha^{d/(d-2)}$ ($\alpha = L_{\text{target}}/L_{\text{train}}$):
- Low-dimensional high-frequency components are nearly unchanged → preserving local resolution
- High-dimensional low-frequency components are stretched → encoding longer distances

This is analogous to the NTK theory's observation about the difference in learning difficulty between high-frequency and low-frequency features: high-frequency features require higher resolution; low-frequency features can be safely extrapolated.

> **Follow-up:** YaRN further applies temperature scaling to the attention score on top of NTK-aware. What is the motivation? Why is modifying position encoding alone insufficient to fully recover long-context task performance?

</details>

---

<details>
<summary>Q28: In a Mixture of Experts (MoE) architecture, how do you design an auxiliary load balancing loss to prevent expert collapse? What is the role of the capacity factor?</summary>

**Expert collapse** in MoE: a small number of experts are selected frequently while the rest are nearly idle, wasting the model's effective capacity.

**Auxiliary load balancing loss:**

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

- $N$ = number of experts, $f_i$ = fraction of tokens routed to expert $i$ (discrete statistics), $P_i$ = average probability the router assigns to expert $i$ (continuous, differentiable)
- The $f_i \cdot P_i$ term encourages both to be uniformly distributed: when an expert is both frequently selected and has high router confidence, the penalty is largest
- $\alpha$ is set to a small value to prevent it from dominating the main training loss

**Capacity factor (CF):** Limits the maximum number of tokens each expert can process in one batch = $\text{CF} \times T/N$. CF too small → tokens are dropped (overflow) → information loss; CF too large → computational waste (padding). CF needs to be dynamically adjusted based on the degree of load imbalance.

> **Follow-up:** DeepSeek-MoE proposes fine-grained expert segmentation (splitting large experts into multiple smaller ones) and a shared expert mechanism. How does this design fundamentally mitigate the tension between load balancing (requiring uniformity) and model capability (requiring specialization)?

</details>

---

<details>
<summary>Q29: How does ZeRO-3's All-Gather communication overlap with forward/backward computation? Why does a naive implementation lead to a significant communication bottleneck?</summary>

ZeRO-3 requires All-Gather of the complete parameters for each layer before the forward pass can proceed. **Naive implementation**: All-Gather → wait → compute → free; communication and computation are serial, and the GPU spends a long time waiting.

**Overlap strategy (dependency graph analysis using backward as an example):**

```
Forward: compute(L) ← All-Gather(L)          compute(L+1) ← All-Gather(L+1)
          ↓ can overlap: while compute(L) runs, asynchronously prefetch All-Gather(L+1)
```

- **Forward:** While computing layer $l$, asynchronously launch the All-Gather for layer $l+1$ parameters (prefetch). Requirement: compute time for layer $l$ ≥ communication time for layer $l+1$.
- **Backward:** Similarly, while computing layer $l$ gradients, prefetch layer $l-1$ parameters; Reduce-Scatter of layer $l$ gradients can also be overlapped with the next layer's computation.

**Cost:** Simultaneously holding more parameter copies increases (current layer + prefetch layer), adding memory pressure. Total communication is ~$3 \times |\theta|$ per step (higher than DP's $2 \times |\theta|$); when cross-node bandwidth is limited this can become a bottleneck.

> **Follow-up:** At what model scale and hardware conditions does ZeRO-3's communication overhead become unacceptable, making TP (intra-node NVLink) + ZeRO-2 the better choice? Analyze from the perspective of the ratio of communication volume to computation volume.

</details>

---

<details>
<summary>Q30: DPO's training data is off-policy (generated by $\pi_{\text{ref}}$) — what theoretical bias does this introduce? How does iterative DPO mitigate this?</summary>

The $\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ term in the DPO loss is essentially an importance-weighted reward estimate.

**Source of off-policy bias:**
- As the divergence between $\pi_\theta$ and $\pi_{\text{ref}}$ grows, the variance of importance weights increases and gradient estimates become unstable
- The training data covers a fixed $y$-space anchored to $\pi_{\text{ref}}$'s support. $\pi_\theta$ may have learned to generate responses not seen in training data, but those responses cannot be evaluated by the DPO loss → optimization signal has blind spots
- Analogous to distribution shift in off-policy RL: the further the policy departs from the data-collection policy, the less reliable the estimates

**How iterative DPO mitigates this:**
1. Sample new responses with the current $\pi_\theta$
2. Annotate preferences with a reward model or human annotators
3. Use the new $\pi_\theta$ as the new $\pi_{\text{ref}}$; retrain DPO
4. Repeat → training data gradually becomes on-policy

**Online DPO** goes further: within the training loop, it samples $\pi_\theta$'s outputs in real time, scores them with the RM, and immediately updates.

> **Follow-up:** In online DPO, if the reward model itself has a systematic bias (e.g., preferring verbose answers), how would online iteration amplify that problem? What are the mechanistic similarities and differences with reward hacking in PPO?

</details>

---

<details>
<summary>Q31: How can reward model over-optimization (overoptimization) in RLHF be explained theoretically? How does the divergence between proxy reward and true quality change as KL increases?</summary>

This is a manifestation of **Goodhart's Law**: when a proxy metric is optimized to the extreme, it decouples from the true objective.

**Theoretical intuition:**
- Let the true reward be $r^*(x,y)$, proxy RM $r_\phi(x,y)$, and their difference $\delta(x,y) = r_\phi - r^*$
- When $\pi_\theta$ optimizes in the direction of $\nabla_\theta \mathbb{E}[r_\phi]$, it not only improves $r^*$ but also exploits $\delta$ — entering regions where $r_\phi$ is overestimated
- As $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ increases, the policy departs further from the training distribution, and the generalization error of $r_\phi$ (i.e., $|\delta|$) grows monotonically
- Qualitative observation: the proxy reward keeps rising; true quality first rises then falls; the crossing point of the two curves is the "over-optimization inflection point"

**Factors influencing the divergence rate:**
- Larger RM capacity and more diverse preference data → the inflection point appears later
- Larger policy exploration space (longer, more diverse generation) → easier to find reward-hacking paths

**Mitigation strategies:** KL penalty, RM ensemble (taking the min or variance penalty across multiple RMs), periodic RM updates.

> **Follow-up:** In practice, how does a reward model ensemble exploit agreement and disagreement among multiple RMs? What are the pros and cons of taking the min, the mean, or using disagreement as an uncertainty signal? How does computational cost affect feasibility?

</details>

---

<details>
<summary>Q32: How does Multi-head Latent Attention (MLA) reduce KV cache memory through low-rank compression? What is the fundamental difference from GQA in terms of compression mechanism?</summary>

MLA no longer stores the complete $K, V$; instead it stores a low-dimensional **latent vector** $c_t^{KV}$, which is decompressed at inference time:

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll n_h \cdot d_h$$

The KV cache stores only $c_t^{KV}$ (dimension $d_c$); at attention computation time it projects back:

$$k_t = W^{UK} c_t^{KV}, \quad v_t = W^{UV} c_t^{KV}$$

KV cache size drops from $2 \times L \times n_h \times d_h \times s$ to $L \times d_c \times s$ ($d_c$ can be much smaller than $2 n_h d_h$).

**Fundamental difference from GQA:**

| Dimension | GQA | MLA |
|-----------|-----|-----|
| Compression target | Head dimension (reduce number of KV heads) | Feature dimension (low-rank projection) |
| Compression nature | Discrete, structured (head grouping) | Continuous, flexible (learnable subspace) |
| Cache contents | Actual K, V values (just fewer heads) | Compressed latent vector (requires decompression) |
| Diversity preservation | Directly preserves independent heads | Relies on expressiveness of low-rank subspace |

MLA's advantage: the number of Q heads is no longer directly tied to cache size, enabling large cache compression while retaining many Q heads. Trade-off: inference requires extra projection computation, and the low-rank constraint may limit pattern diversity across heads.

> **Follow-up:** Does MLA's low-rank compression cause different attention heads' patterns to converge (loss of head diversity)? Can the high rank of the projection matrix $W^{UK}$ fully mitigate this risk? In practice, what signals can detect degradation of head diversity?

</details>

## §A Key Papers Timeline

- **2018-11 · GPipe** — Huang et al., NeurIPS 2019. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965) — Foundational pipeline parallelism: partitions layers into stages across devices and splits each mini-batch into micro-batches fed through the pipeline to amortize the bubble, trading recomputation for activation memory so giant models fit across devices.

- **2019-09 · Megatron-LM** — Shoeybi et al., arXiv preprint. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) — Intra-layer tensor parallelism: shards attention and MLP weight matrices column/row-wise across GPUs with one all-reduce each in the forward ($f$) and backward ($g$) pass, scaling to billions of parameters with no change to model structure.

- **2019-10 · ZeRO** — Rajbhandari et al., SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) — Shards the redundant optimizer states / gradients / parameters of data parallelism across ranks (Stages 1/2/3), cutting per-GPU memory from $16\Phi$ to roughly $16\Phi/N$ without incurring tensor-parallel communication cost.

- **2022-05 · Reducing Activation Recomputation** — Korthikanti et al., MLSys 2023. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) — Sequence parallelism + selective recomputation: shards activations of element-wise ops (LayerNorm/Dropout) along the sequence dimension and recomputes only the cheapest-to-redo ops, cutting activation memory ~5×, orthogonal to tensor parallelism.

- **2022-05 · FlashAttention** — Dao et al., NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) — IO-aware exact attention: uses tiling + online softmax to keep the $QK^\top$ intermediate in SRAM instead of HBM, freeing attention from the memory-bandwidth bottleneck and making memory linear (not quadratic) in sequence length.

- **2022-09 · FP8 Formats for Deep Learning** — Micikevicius et al., arXiv preprint. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433) — Defines two 8-bit floating-point encodings for deep learning: E4M3 (range ±448, precision-first, forward pass) and E5M2 (range ±57344, dynamic-range-first, gradients), setting the standard for H100-era FP8 training/inference.

- **2022-10 · GPTQ** — Frantar et al., ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) — One-shot post-training weight quantization via the OBS (Optimal Brain Surgeon) second-order approximation: quantizes column-by-column and compensates the remaining weights using the inverse Hessian, compressing 175B models to 3–4 bit with little accuracy loss.

- **2022-11 · SmoothQuant** — Xiao et al., ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) — W8A8 quantization: activations have hard-to-quantize outlier channels, so it per-channel "migrates" the difficulty from activations to weights ($X\to X/s$, $W\to sW$), letting both use INT8 without mixed precision.

- **2022-11 · Speculative Decoding** — Leviathan et al., ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) — A small draft model proposes several tokens that the large target model verifies in parallel, with a carefully designed accept-reject sampling rule guaranteeing the output distribution exactly matches target-only decoding (lossless speedup).

- **2023-06 · AWQ** — Lin et al., MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) — Activation-aware weight quantization: observes that a tiny fraction of "salient" weight channels dominate error and uses activation magnitude to guide per-channel scaling that protects them, preserving accuracy at 4-bit in a hardware-friendly way.

- **2023-09 · PagedAttention / vLLM** — Kwon et al., SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) — Manages the KV cache like OS virtual-memory paging: stores KV in non-contiguous blocks allocated on demand, eliminating fragmentation and reservation waste and enabling prefix sharing, greatly raising serving throughput.

- **2024-02 · KIVI** — Liu et al., ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) — Asymmetric 2-bit quantization for the KV cache: quantizes keys per-channel and values per-token (matching their distinct outlier distributions), shrinking KV memory ~8× in long-context inference with near-lossless accuracy.
