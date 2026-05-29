# PEFT — Parameter-Efficient Fine-Tuning Cheat Sheet

> For researchers and engineers working on LLM post-training, covering the full knowledge stack of mainstream PEFT methods.

---

## Part 1: Concepts & Formula Derivations

### 1.1 Why PEFT?

Full Fine-Tuning (FFT) applies gradient updates to **all parameters** of a pretrained model, which poses the following problems:

| Problem | Description |
|---------|-------------|
| **High memory cost** | Parameters (BF16) + gradients (BF16) + Adam optimizer states m/v (FP32): a 7B model requires roughly 84 GB+ (excluding activations and FP32 master copy; ~112 GB with master copy) |
| **Catastrophic Forgetting** | Full-parameter updates may overwrite pretrained knowledge |
| **High multi-task deployment cost** | Storing a complete model per task incurs high storage and switching overhead |
| **Training instability** | Sensitive to learning rate; requires careful tuning |

**Core idea of PEFT**: freeze the base model parameters $W_0$, train only a small number of additional parameters $\Delta\Theta$, and achieve performance close to FFT.

$$W_{\text{final}} = W_0 + \Delta W(\Theta), \quad |\Theta| \ll |W_0|$$

---

### 1.2 LoRA — Low-Rank Adaptation

**Core assumption**: the update $\Delta W$ to pretrained weights is low-rank.

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B A$$

Where:

- $W_0 \in \mathbb{R}^{d \times k}$: frozen pretrained weight
- $A \in \mathbb{R}^{r \times k}$: down-projection matrix, initialized with Kaiming uniform
- $B \in \mathbb{R}^{d \times r}$: up-projection matrix, initialized to **zero**
- $r \ll \min(d, k)$: rank, controls expressiveness
- $\alpha$: scaling hyperparameter, controls the overall magnitude of the LoRA update

**Trainable parameter count** (per layer):

$$|\Theta_{\text{LoRA}}| = r \times (d + k)$$

**Initialization design rationale**:

- $B = 0$: at the start of training $\Delta W = BA = 0$, leaving the base model output undisturbed; gradient signal is stable
- $A \sim \text{Kaiming Uniform}$: non-zero initialization ensures gradients can flow (if $A$ were also zero, gradients would always be zero)

**Merging**: at inference time the LoRA weights can be merged back into the base model for **zero additional inference latency**.

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} \cdot B A$$

---

### 1.3 rsLoRA — Rank-Stabilized LoRA

Standard LoRA uses the scaling factor $\alpha / r$. As rank $r$ grows, the Frobenius norm of $BA$ increases as $\sqrt{r}$, leading to excessively large gradients and training instability.

**rsLoRA fix**: change the scaling to $\alpha / \sqrt{r}$:

$$h = W_0 x + \frac{\alpha}{\sqrt{r}} \cdot B A x$$

This ensures that the magnitude of updates is consistent across different ranks $r$, making training more stable at large rank.

---

### 1.4 DoRA — Weight-Decomposed Low-Rank Adaptation

DoRA decomposes the weight update into **magnitude** and **direction** components:

$$W = m \cdot \frac{W_0 + \frac{\alpha}{r} B A}{\left\| W_0 + \frac{\alpha}{r} B A \right\|_c}$$

- $m \in \mathbb{R}^{k}$: learnable magnitude vector (per-column magnitude), one scaling factor per column
- $\|\cdot\|_c$: column-wise norm, making each column a unit direction vector

**Motivation**: analysis shows that FFT modifies both the magnitude and direction of weights simultaneously, whereas standard LoRA mainly modifies direction and has limited capacity to adjust magnitude. Adding a learnable magnitude vector makes the gradient behavior closer to full fine-tuning.

**Additional parameter count**: $k$ (one magnitude vector per layer), negligible relative to LoRA's $r \times (d+k)$.

---

### 1.5 VeRA — Vector-based Random Matrix Adaptation

VeRA achieves extreme parameter efficiency: all layers **share the same set of randomly fixed matrices** $A$ and $B$, training only **diagonal scaling vectors**:

$$\Delta W = \text{diag}(b) \cdot B \cdot \text{diag}(d) \cdot A$$

- $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$: randomly initialized then **frozen**, shared across all layers
- $b \in \mathbb{R}^{d}$, $d \in \mathbb{R}^{r}$: trainable diagonal scaling vectors

**Parameter count**: only $r + d$ trainable parameters per layer (two vectors), drastically less than LoRA's $r \times (d + k)$.

**Trade-off**: shared matrices and purely diagonal scaling limit expressiveness, resulting in a gap versus LoRA on complex tasks.

---

### 1.6 AdaLoRA — Adaptive LoRA

The observation that different layers and matrices vary in importance for a given task makes a fixed rank suboptimal.

**Parameterization** (SVD-like form):

$$\Delta W = P \cdot \Lambda \cdot Q$$

where $P$ and $Q$ are orthogonal matrices and $\Lambda$ is a diagonal matrix (singular values).

**Importance Scoring**: for each singular value triplet $(p_i, \lambda_i, q_i)$, compute:

$$\text{Importance}_i = |\lambda_i| \times \left| \frac{\partial \mathcal{L}}{\partial \lambda_i} \right|$$

During training, singular values with low importance are masked to zero (pruned), redistributing the parameter budget to more important matrices/layers.

**Effect**: key layers (e.g., higher-level attention) receive higher rank under the same parameter budget.

---

### 1.7 (IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

(IA)³ multiplies the attention key/value and FFN output by **learnable scaling vectors** (not matrices):

$$K' = l_k \odot K, \quad V' = l_v \odot V, \quad h_{\text{FFN}}' = l_{\text{ff}} \odot h_{\text{FFN}}$$

- $l_k \in \mathbb{R}^{d_k}$, $l_v \in \mathbb{R}^{d_v}$, $l_{\text{ff}} \in \mathbb{R}^{d_{\text{ff}}}$: trainable vectors

**Very small parameter count**: only about $d_k + d_v + d_{\text{ff}}$ parameters per layer. Suitable for few-shot scenarios, but insufficient expressiveness on complex tasks (only activation rescaling; cannot learn directional changes).

At inference time the scaling vectors can be fused into $W_K$, $W_V$, and $W_{\text{FFN}}$ for zero additional latency.

---

### 1.8 Soft Prompt Family

| Method | Location of learnable parameters | Every layer or input only | Notes |
|--------|-----------------------------------|---------------------------|-------|
| **Prompt Tuning** | $k$ soft tokens in the input embedding layer | Input layer only | Very few parameters; works well for large models, poorly for small ones |
| **Prefix Tuning** | Learnable prefix prepended to K/V at every attention layer | Every layer | More expressive than Prompt Tuning; requires reparameterization (MLP) for stable training |
| **P-Tuning v2** | Prefix at every layer (similar to Prefix Tuning) | Every layer | Unified framework; approaches FFT on NLU tasks |

**Shared limitation**: cannot be merged into the base model at inference time (the prefix occupies space in the KV cache), incurring additional memory overhead.

---

### 1.9 Adapter — Bottleneck Module

Inserts a small **bottleneck MLP** into each Transformer block:

$$h \rightarrow \text{LayerNorm} \rightarrow W_{\text{down}} (d \to r) \rightarrow \sigma(\cdot) \rightarrow W_{\text{up}} (r \to d) \rightarrow \text{Residual} \rightarrow h'$$

**Parameter count**: $2 \times r \times d$ per layer, comparable to LoRA.

**Inference-time distinction from LoRA**:
- **Adapter**: introduces additional serial forward-pass layers (bottleneck MLP) at inference time, increasing latency
- **LoRA**: zero additional layers after merging, zero latency

---

### 1.10 Hadamard Product Family

**Core motivation**: standard LoRA's $\Delta W = BA$ satisfies $\text{rank}(\Delta W) \leq r$. Can we achieve a higher effective rank with the same parameter count?

**Mathematical foundation**: the Hadamard product (element-wise product) of two $\text{rank}{-}r$ matrices can have rank as high as $r^2$ (Khatri-Rao product property). That is, the parameter count stays the same, but the representable rank space is much larger.

**HiRA** (High-rank Adaptation):

$$\Delta W = W_0 \odot (B A)$$

Takes the Hadamard product of the frozen $W_0$ (full rank) with LoRA's $BA$; the effective rank of the product exceeds $r$. Parameter count is the same as LoRA.

**BoHA** (Block-wise Hadamard Product Adaptation):

Partitions $W_0$ into $b \times b$ blocks along the row/column directions; each block independently learns LoRA factors $(A_i, B_i)$, and the Hadamard product update is applied at the block level:

$$(\Delta W)_{\text{block}_i} = (W_0)_{\text{block}_i} \odot (B_i A_i)$$

- Total parameter count is equivalent to LoRA (sum of block LoRA factors ≈ a single LoRA at the same rank)
- The block-wise strategy allows each block to adapt independently, providing finer granularity and better preservation of local structure
- **Zero additional inference overhead** after merging

**ABBA**: another alternating block Hadamard structure, based on similar principles.


---

### 1.11 LoRA+ — Decoupled Learning Rate

Based on Maximal Update Parameterization (muP) analysis, using the same learning rate for $A$ and $B$ in standard LoRA is suboptimal.

**Core finding**: $B$ should use a higher learning rate than $A$:

$$\eta_B / \eta_A = \lambda, \quad \text{recommended } \lambda \in [2, 16]$$

**Reason**: in the infinite-width limit, the gradient signal magnitudes of $A$ and $B$ differ; a uniform learning rate causes one to underfit and the other to overfit.

**In practice**: set a higher learning rate for the $B$ parameter group in the optimizer (e.g., $\eta_A = 10^{-4}$, $\eta_B = 2 \times 10^{-3}$).

---

### 1.12 GaLore — Gradient Low-Rank Projection

GaLore does not apply low-rank parameterization to weights; instead it applies low-rank projection to **gradients**:

$$G \approx U_r \Sigma_r V_r^T \quad \text{(top-}r\text{ SVD of the gradient)}$$

Optimizer state is maintained only in the $r$-dimensional projection space, with the projection direction recomputed every $T$ steps (subspace refresh).

**Fundamental difference from LoRA**:
- **LoRA**: freezes the base model, trains only A/B; supports merging and adapter reuse
- **GaLore**: updates all parameters; the optimizer maintains low-dimensional state; the result is a standard model at inference time and does not support adapter reuse

Suitable for pre-training / continual pre-training scenarios; LoRA is more practical for SFT.

---

### 1.13 PiSSA — Principal SVD Initialization

Standard LoRA starts learning from $\Delta W = 0$; PiSSA starts from the principal components of $W_0$.

Decompose $W_0$ by SVD:

$$W_0 = U \Sigma V^T$$

Initialize using the top $r$ singular values/vectors:

$$A = \sqrt{\Sigma_r} \cdot V_r^T, \quad B = U_r \cdot \sqrt{\Sigma_r}$$

That is, $BA = U_r \Sigma_r V_r^T \approx$ the principal components of $W_0$. The frozen part becomes the residual: $W_0 - BA$.

**Advantage**: the initial point is closer to the structure of $W_0$, resulting in faster convergence. **Cost**: requires a one-time SVD initialization on $W_0$.

---

## Part 2: PyTorch Snippets

> The code below consists of educational snippets implemented from scratch, depending only on `torch` and `torch.nn`. For production projects, use the [PEFT](https://github.com/huggingface/peft) or [LoRA](https://github.com/microsoft/LoRA) library.

### Snippet 1 — LoRA Linear Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """LoRA: W = W0 + (alpha/r) * B @ A"""

    def __init__(self, in_dim: int, out_dim: int,
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        # Frozen base weight
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.base.weight.requires_grad_(False)

        # LoRA factors
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight)          # x @ W0^T
        lora_out = F.linear(F.linear(x, self.A), self.B)  # x @ A^T @ B^T
        return base_out + lora_out * self.scaling

    def merge_weights(self) -> torch.Tensor:
        """Return merged weight W0 + (alpha/r) * B @ A"""
        return self.base.weight + (self.B @ self.A) * self.scaling
```

### Snippet 2 — DoRA Layer

```python
class DoRALinear(nn.Module):
    """DoRA: W = m * (W0 + BA) / ||W0 + BA||_c"""

    def __init__(self, in_dim: int, out_dim: int,
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.base.weight.requires_grad_(False)

        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # Per-column magnitude vector
        self.m = nn.Parameter(torch.ones(out_dim))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Combined weight with LoRA delta
        W_combined = self.base.weight + (self.B @ self.A) * self.scaling
        # Column-wise normalization (each row of weight matrix)
        col_norms = W_combined.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_dora = self.m.unsqueeze(1) * W_combined / col_norms
        return F.linear(x, W_dora)
```

### Snippet 3 — VeRA Layer

```python
class VeRALinear(nn.Module):
    """VeRA: delta_W = diag(b) @ B_shared @ diag(d) @ A_shared"""

    # Shared frozen random matrices (class-level, shared across layers)
    _shared_A: dict = {}
    _shared_B: dict = {}

    def __init__(self, in_dim: int, out_dim: int, rank: int = 8,
                 layer_id: str = "default"):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.base.weight.requires_grad_(False)

        # Create or reuse shared random matrices
        key = (rank, in_dim, out_dim)
        if key not in VeRALinear._shared_A:
            VeRALinear._shared_A[key] = torch.randn(rank, in_dim)
            VeRALinear._shared_B[key] = torch.randn(out_dim, rank)

        self.register_buffer("A", VeRALinear._shared_A[key])
        self.register_buffer("B", VeRALinear._shared_B[key])

        # Trainable diagonal scaling vectors only
        self.d = nn.Parameter(torch.ones(rank))   # scales rows of A
        self.b = nn.Parameter(torch.ones(out_dim)) # scales rows of B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight)
        # diag(d) @ A  -> scale each row of A by d
        scaled_A = self.d.unsqueeze(1) * self.A
        # diag(b) @ B  -> scale each row of B by b
        scaled_B = self.b.unsqueeze(1) * self.B
        lora_out = F.linear(F.linear(x, scaled_A), scaled_B)
        return base_out + lora_out
```

### Snippet 4 — Adapter (Bottleneck MLP)

```python
class AdapterLayer(nn.Module):
    """Bottleneck adapter: h -> LN -> Down -> Act -> Up -> Residual"""

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck)
        self.up_proj = nn.Linear(bottleneck, dim)
        self.act = nn.ReLU()
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.act(self.down_proj(x))
        return residual + self.up_proj(h)
```

### Snippet 5 — (IA)³ Scaling Vectors

```python
class IA3Rescaling(nn.Module):
    """(IA)³: rescale key, value, and FFN output with learned vectors."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.l_k = nn.Parameter(torch.ones(d_model))
        self.l_v = nn.Parameter(torch.ones(d_model))
        self.l_ff = nn.Parameter(torch.ones(d_ff))

    def rescale_kv(self, K: torch.Tensor, V: torch.Tensor):
        return self.l_k * K, self.l_v * V

    def rescale_ffn(self, ffn_out: torch.Tensor):
        return self.l_ff * ffn_out
```

### Snippet 6 — Prefix Tuning (Learnable K/V Prefix)

```python
class PrefixKV(nn.Module):
    """Prefix Tuning: prepend learnable vectors to K and V in attention."""

    def __init__(self, num_layers: int, num_heads: int,
                 head_dim: int, prefix_len: int = 10):
        super().__init__()
        self.prefix_len = prefix_len
        # Reparameterize via MLP for stable training
        self.mlp = nn.Sequential(
            nn.Linear(head_dim, head_dim * 4),
            nn.Tanh(),
            nn.Linear(head_dim * 4, num_layers * 2 * num_heads * head_dim),
        )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Learnable input tokens
        self.prefix_tokens = nn.Parameter(
            torch.randn(prefix_len, head_dim) * 0.01
        )

    def get_prefix(self, layer_idx: int):
        """Return (prefix_K, prefix_V) for a given layer."""
        raw = self.mlp(self.prefix_tokens)  # (prefix_len, L*2*H*D)
        raw = raw.view(self.prefix_len, self.num_layers, 2,
                       self.num_heads, self.head_dim)
        prefix_k = raw[:, layer_idx, 0]  # (prefix_len, num_heads, head_dim)
        prefix_v = raw[:, layer_idx, 1]
        return prefix_k, prefix_v
```

### Snippet 7 — HiRA / Hadamard-style LoRA

```python
class HiRALinear(nn.Module):
    """HiRA: delta_W = W0 ⊙ (B @ A), higher effective rank than LoRA."""

    def __init__(self, in_dim: int, out_dim: int,
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.base.weight.requires_grad_(False)

        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-rank delta (same params as LoRA)
        delta = (self.B @ self.A) * self.scaling          # (out, in)
        # Element-wise product with frozen base → higher effective rank
        hadamard_delta = self.base.weight * delta          # (out, in)
        return F.linear(x, self.base.weight + hadamard_delta)
```

### Snippet 8 — QLoRA Concept Demo (4-bit NF4 Quantization Wrapper)

```python
class QLoRALinear4Bit(nn.Module):
    """QLoRA concept: 4-bit quantized base + BF16 LoRA adapters.

    Note: Real NF4 quantization (double quant, paged optimizer) requires
    bitsandbytes or similar libraries. This is a simplified demonstration.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        # In practice: use bitsandbytes.nn.Linear4bit with nf4 quant_type
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.base.weight.requires_grad_(False)
        # Simulate frozen quantized weights (real impl would store as int4)

        # LoRA adapters in BF16 (trainable)
        self.A = nn.Parameter(torch.empty(rank, in_dim, dtype=torch.bfloat16))
        self.B = nn.Parameter(torch.zeros(out_dim, rank, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize base weight on-the-fly (simplified)
        base_out = F.linear(x.to(self.base.weight.dtype), self.base.weight)
        lora_out = F.linear(
            F.linear(x.to(self.A.dtype), self.A), self.B
        ) * self.scaling
        return base_out.to(lora_out.dtype) + lora_out
```

---

## Part 3: Interview Questions

### ━━━ L1 Basic ━━━

---

<details>
<summary>Q1: Why is PEFT needed? What are the problems with Full Fine-Tuning?</summary>

**Answer:** FFT applies gradient updates to all parameters and suffers from four major problems: (1) **High memory cost** (7B model: BF16 parameters 14 GB + BF16 gradients 14 GB + FP32 Adam m/v 56 GB ≈ **84 GB+**, excluding activations; ~112 GB with FP32 master copy); (2) **Catastrophic forgetting**, potentially overwriting the model's general pretrained capabilities; (3) **High multi-task deployment cost**, requiring a full model stored per task; (4) **Training instability**, with high sensitivity to learning rate.

PEFT freezes the base model and trains only a small number of additional parameters (e.g., 0.1%–1%), significantly reducing memory, storage, and training costs while achieving performance close to FFT.

**Follow-up:** Does PEFT incur additional latency at inference time? Which methods can be merged back into the base model?

> **Answer:** LoRA, DoRA, VeRA, (IA)³, HiRA, and BoHA can all be merged back into the base model with **zero additional inference latency**. Adapter and Prefix Tuning cannot be merged — Adapter requires an additional bottleneck forward pass at inference time, and Prefix Tuning occupies space in the KV cache; both incur inference overhead.

</details>

---

<details>
<summary>Q2: What is the mathematical principle behind LoRA? What are rank $r$, scaling $\alpha$, and target modules?</summary>

**Answer:** LoRA assumes that the pretrained weight update $\Delta W$ is low-rank: $W = W_0 + \frac{\alpha}{r} BA$. Here $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$. $A$ is initialized with Kaiming uniform, $B$ is initialized to zero, ensuring $\Delta W = 0$ at the start of training.

- **$r$**: controls expressiveness; larger $r$ allows higher rank in $\Delta W$, with parameter count increasing linearly
- **$\alpha$**: scaling factor; the actual scale applied is $\alpha/r$; typically $\alpha = r$ or $2r$
- **Target modules**: typically the Q/K/V/O matrices of attention; for complex tasks, FFN's up/down/gate matrices may also be included

**Follow-up:** Why is the low-rank update assumption in LoRA reasonable? Is there theoretical support?

> **Answer:** The main support comes from: (1) the low-intrinsic-dimensionality theory: random subspace probing finds that many tasks can reach performance close to FFT within a low-dimensional subspace; (2) in practice, after FFT the singular-value distribution of $\Delta W$ decays rapidly, with the top-$r$ components covering most of the energy; (3) the BA decomposition implicitly imposes nuclear-norm regularization on $\Delta W$.

</details>

---

<details>
<summary>Q3: How do you tune LoRA's rank $r$ and scaling $\alpha$? What problem does rsLoRA solve?</summary>

**Answer:**
- **$r$**: harder tasks (e.g., mathematical reasoning) typically require larger $r$ (commonly 4 / 8 / 16 / 64)
- **$\alpha$**: implicitly modulates the LoRA learning rate; increasing $\alpha$ → larger update magnitude

**rsLoRA** addresses: with standard LoRA at large $r$, the Frobenius norm of $BA$ grows as $\sqrt{r}$, leading to excessively large gradients and training instability. rsLoRA changes the scaling to $\alpha / \sqrt{r}$, ensuring consistent update magnitude across different $r$.

**Follow-up:** How do you determine the optimal $r$ in practice? Are there adaptive methods?

> **Answer:** In practice, start from $r=8$ and gradually increase while observing loss/eval plateau; if increasing to $r=64$ still yields no improvement, the bottleneck is likely not expressiveness. Adaptive methods such as **AdaLoRA** dynamically allocate rank per layer via SVD + importance scoring.

</details>

---

<details>
<summary>Q4: What is <code>(IA)³</code>? Why does it have even fewer parameters than LoRA?</summary>

**Answer:** (IA)³ multiplies attention key/value and FFN output by **learnable scaling vectors** ($l_k, l_v, l_{\text{ff}}$), with dimensions $d_k, d_v, d_{\text{ff}}$ respectively. Parameter count is about $d_k + d_v + d_{\text{ff}}$ per layer (a few thousand), far fewer than LoRA's $r \times (d+k)$.

Suitable for few-shot scenarios; the trade-off is weak expressiveness (only rescaling; cannot learn directional changes), resulting in underfitting on complex tasks.

**Follow-up:** Can (IA)³ be merged into the base model at inference time?

> **Answer:** Yes. $l_k \odot K = (l_k \odot W_K) x$, so $l_k$ can be multiplied directly into the $W_K$ weight matrix, yielding zero additional latency after merging.

</details>

---

<details>
<summary>Q5: What are the differences between Prefix Tuning, Prompt Tuning, and P-Tuning v2?</summary>

**Answer:**

| Method | Parameter location | Per-layer / Input only | Parameter count |
|--------|--------------------|------------------------|-----------------|
| **Prompt Tuning** | Soft tokens in input embedding | Input layer only | Very small |
| **Prefix Tuning** | K/V prefix at every attention layer | Per layer | Small (requires MLP reparameterization) |
| **P-Tuning v2** | Prefix at every layer | Per layer | Medium, expressiveness close to FFT |

Prompt Tuning has the fewest parameters but performs poorly on small models; Prefix Tuning is more expressive but requires reparameterization for stable training; P-Tuning v2 has a unified framework that approaches FFT on NLU tasks. Shared limitation: requires KV cache space at inference time; cannot be merged.

**Follow-up:** Why does Prefix Tuning require reparameterization (MLP) for stable training?

> **Answer:** Directly optimizing prefix vectors is prone to overfitting and instability (parameters lack structural constraints in the embedding space). Mapping a low-dimensional vector to the high-dimensional prefix space via an MLP introduces implicit regularization, making the optimization smoother and generalization better.

</details>

---

<details>
<summary>Q6: What is the structure of the Adapter method? How does it differ from LoRA at inference time?</summary>

**Answer:** Adapter inserts a bottleneck MLP after the attention/FFN in each Transformer block: $h \to W_{\text{down}}(d \to r) \to \text{ReLU} \to W_{\text{up}}(r \to d) \to \text{Residual}$. Parameter count is about $2rd$, similar to LoRA.

**Key distinction**: Adapter incurs an additional serial forward pass (bottleneck MLP) at inference time, increasing latency; LoRA has zero additional latency after merging. LoRA is preferable in latency-sensitive production scenarios.

**Follow-up:** How can multiple LoRA adapters be served simultaneously? Is there system-level work on this?

> **Answer:** Representative works include **S-LoRA** (unified KV cache management + custom CUDA kernels for batched LoRA) and **Punica** (open-source implementation of S-LoRA), which support serving thousands of LoRA adapters simultaneously on a single base model by parallelizing the BA computation for different adapters within a batch to achieve high throughput.

</details>

---

<details>
<summary>Q7: What is VeRA? What level of parameter efficiency does it achieve?</summary>

**Answer:** VeRA lets all layers share the same randomly fixed matrices $A$ and $B$, training only diagonal scaling vectors $b$ and $d$: $\Delta W = \text{diag}(b) \cdot B \cdot \text{diag}(d) \cdot A$. Only $r + d$ trainable parameters are needed per layer, approximately $1/k$ of LoRA ($k$ = hidden dim).

The trade-off is limited expressiveness — shared matrices + purely diagonal scaling fall short of LoRA on complex tasks; suitable for edge deployment scenarios with extreme parameter constraints.

**Follow-up:** Comparing parameter counts between VeRA and (IA)³?

> **Answer:** Both have very few parameters: VeRA has $r + d$ per layer, (IA)³ has $d_k + d_v + d_{\text{ff}}$ per layer, similar orders of magnitude. However, (IA)³ only performs activation rescaling, while VeRA performs more structured weight updates through shared matrices, making VeRA slightly more expressive.

</details>

---

<details>
<summary>Q8: Why is $A$ initialized with Kaiming uniform and $B$ initialized to zero in LoRA?</summary>

**Answer:**
- **$B = 0$**: ensures $\Delta W = BA = 0$ at the start of training, so LoRA does not disturb the base model output; gradient signal is stable
- **A ~ Kaiming uniform**: non-zero ensures gradients can flow (if $A$ were also zero, $\Delta W \equiv 0$ and gradients would always be zero); Kaiming initialization ensures activation variance does not explode or vanish during the forward pass

If both $A$ and $B$ were randomly initialized ($B \neq 0$), $\Delta W \neq 0$ at the start of training would disturb the initial behavior of the base model, leading to instability in the early training phase.

**Follow-up:** Are there better LoRA initialization strategies?

> **Answer:** **PiSSA** initializes $A$ and $B$ from the top $r$ SVD components of $W_0$, starting optimization from the principal components. **LoRA-GA** initializes with gradient-aligned directions so that the initial LoRA update direction matches the full FT gradient. These methods show improvements in convergence speed.

</details>

---

### ━━━ L2 Intermediate ━━━

---

<details>
<summary>Q9: What is QLoRA? How does it drastically reduce memory usage while maintaining performance?</summary>

**Answer:** QLoRA quantizes the base model to **4-bit NF4** format while keeping the LoRA adapter in BF16 precision for training. Three key techniques:

1. **NF4 quantization** (Normal Float 4): a 4-bit format designed to minimize information loss on Gaussian distributions
2. **Double Quantization**: the quantization constants (scales) are themselves quantized for further compression
3. **Paged Optimizer**: uses unified memory to page optimizer states between GPU/CPU, preventing OOM

Effect: dramatically reduces memory requirements (e.g., a 65B model can be trained on a single GPU), with the quantization error of the 4-bit base model compensated by the LoRA adapter.

**Follow-up:** What is the difference between NF4 and INT4? How does quantization error affect LoRA gradients?

> **Answer:** NF4 is designed based on information-theoretic optimality; quantization points are distributed according to the quantiles of a Gaussian distribution. INT4 uses uniform spacing. For pretrained weights that are near-Gaussian, NF4 has smaller quantization error. LoRA gradients propagate through the quantized weights via straight-through estimation or a full-precision frozen base; in practice, quantization error mainly affects forward-pass accuracy, and its impact on gradient direction is mitigated by LoRA's low-rank structure.

</details>

---

<details>
<summary>Q10: What is DoRA? How does it improve LoRA's expressiveness?</summary>

**Answer:** DoRA decomposes the weight update into **direction** and **magnitude**: $W = m \cdot (W_0 + BA) / \|W_0 + BA\|_c$, where $m$ is a learnable magnitude vector. Analysis shows that FFT modifies both magnitude and direction simultaneously, while LoRA mainly modifies direction; adding a learnable magnitude vector makes DoRA's gradient behavior closer to full FT, with the original paper reporting certain improvements over LoRA at the same parameter count on several tested tasks (competitive with or better than LoRA).

**Additional parameters**: only $k$ parameters (the magnitude vector), negligible relative to LoRA.

**Follow-up:** How many additional parameters does DoRA have? How is the dimension of $m$ determined?

> **Answer:** $m \in \mathbb{R}^{k}$, where $k$ is the input dimension (number of columns) of the weight matrix — one magnitude factor per input feature direction. Only $k$ extra parameters per layer (typically a few thousand), while LoRA has $r \times (d+k)$ per layer (typically hundreds of thousands), so DoRA's overhead is nearly zero.

</details>

---

<details>
<summary>Q11: How does AdaLoRA adaptively allocate rank?</summary>

**Answer:** AdaLoRA parameterizes $\Delta W$ in an SVD-like form $P \Lambda Q$, computing an importance score for each singular value triplet: $\text{Importance}_i = |\lambda_i| \times |\partial \mathcal{L}/\partial \lambda_i|$. During training, singular values with low importance are masked to zero (pruned), redirecting the budget to important layers/matrices.

Effect: key layers (e.g., higher-level attention) receive more rank, which is better than uniform allocation.

**Follow-up:** How exactly is the importance score in AdaLoRA computed? Is the mask operation hard or soft?

> **Answer:** Importance combines singular value magnitude ($|\lambda_i|$) and sensitivity (gradient norm of the loss with respect to $\lambda_i$). The mask is **hard** during training (directly set to zero), implemented via a regularization budget constraint on $P$, $\Lambda$, and $Q$. At the end of training, unmasked singular values are retained, corresponding to the final rank allocation.

</details>

---

<details>
<summary>Q12: Can LoRA be used for RLHF? What are the key considerations?</summary>

**Answer:** Yes, and it is widely used. Key considerations:

1. **Actor/Ref sharing the base**: the reference policy can directly disable the LoRA adapter without needing a separate base model copy (supported by TRL, OpenRLHF)
2. **LoRA for the Critic**: the critic head is typically randomly initialized and needs a higher rank to converge
3. **Gradient path**: LoRA's A/B gradients propagate through the frozen base forward pass; the PPO gradient path is unchanged, but activations still need to be stored in memory
4. **Merge timing**: merge after PPO finishes before evaluation

**Follow-up:** Is there any risk in using disable-adapter for the reference policy instead of storing a separate model?

> **Answer:** There is some risk. After disabling the adapter, the reference policy equals the base model, while the actor's LoRA updates create a divergence from the base model during PPO training. If the LoRA update magnitude is large (large $\alpha$ or many iterations), the KL constraint may not be sufficient to prevent policy drift. In practice, monitor the KL divergence closely and reduce $\alpha$ or the clip ratio if needed.

</details>

---

<details>
<summary>Q13: On which tasks is the gap between LoRA rank and full fine-tuning largest?</summary>

**Answer:** Scenarios with the largest gap: (1) **Large-scale knowledge injection** (continual pre-training / domain adaptation), where low-rank $\Delta W$ lacks sufficient expressiveness; (2) **Complex reasoning tasks** (challenging math, code), where the required parameter update directions are complex and low rank underfits; (3) **Safety alignment**, where low-rank updates sometimes only modify surface behavior.

Scenarios with small gaps: instruction tuning (simple task structure), single-task fine-tuning (classification, NER, etc. with low intrinsic dimensionality).

**Follow-up:** How can you empirically determine whether LoRA has sufficient expressiveness for a given task?

> **Answer:** Progressively increase $r$ (e.g., 4→8→16→32→64) and plot the eval metric vs. $r$ curve. If a plateau appears at a larger $r$ (no further improvement), LoRA has sufficient expressiveness. If performance keeps improving and still has not reached the FFT level, the low-rank assumption is insufficient, and high-rank methods or FFT should be considered.

</details>

---

<details>
<summary>Q14: How can multiple LoRA adapters be merged? What are the available methods?</summary>

**Answer:**
1. **Linear merge**: $W = W_0 + \sum_i \alpha_i B_i A_i$, direct linear superposition
2. **TIES**: sign voting over multiple deltas (trim + elect sign + disjoint merge), reducing conflicts
3. **DARE**: randomly prunes a portion of delta parameters (sets them to zero with probability $p$ then rescales), reducing interference
4. **LoRAHub**: uses a small number of examples to learn per-adapter weights, automatically discovering the optimal composition
5. **SLERP**: spherical linear interpolation, preserving the characteristics of both checkpoints

**Follow-up:** Which is better — a merged model or multi-task joint training?

> **Answer:** It depends on the relationships between tasks. When task similarity is high, merging approaches joint training in quality; when tasks conflict significantly, merging can degrade (interference) and joint training is better through gradient coordination. However, the advantage of merging is that it requires no joint training data and supports incremental composition. In practice, merge + a small amount of mixed-data fine-tuning is a common compromise.

</details>

---

<details>
<summary>Q15: How do you select LoRA target modules for a specific task?</summary>

**Answer:** Empirical guidelines:
- **Attention-only** (Q/K/V/O): the default choice; fewer parameters, good for language understanding
- **All linear** (Q/K/V/O + FFN up/down/gate): 2–3× more parameters; better for complex reasoning tasks
- **V/O only**: some studies find the V matrix is most important for task-specific adaptation

Data-driven approach: train on all matrices first, then identify matrices with the largest update magnitude using importance scores.

**Follow-up:** Should the embedding layer and LM head be included in LoRA?

> **Answer:** Generally not recommended — embeddings and the LM head are sensitive to tokenization and share weights (weight tying); adding LoRA may disrupt the token semantic space that has already been learned. However, when new special tokens or a domain-specific vocabulary need to be added, the embedding layer can be separately full-fine-tuned rather than LoRA-adapted.

</details>

---

<details>
<summary>Q16: What is rank collapse in LoRA? How is it detected and mitigated?</summary>

**Answer:** Rank collapse refers to the effective rank of $BA$ (measured by its singular value distribution) during training converging to a value much smaller than $r$; the nominal rank is $r$ but expressiveness approaches rank 1–2.

**Cause**: optimization does not explicitly constrain rank; gradients tend toward the "simplest" low-dimensional solution.

**Detection**: periodically compute the effective rank of $BA$ = $\exp(H(\sigma))$, where $H$ is the entropy of the normalized singular values.

**Mitigation**: rsLoRA (adjusting scaling to prevent overly large updates early on), AdaLoRA (optimization in SVD space), periodic orthogonalization of A/B.

**Follow-up:** Is rank collapse related to gradient vanishing?

> **Answer:** They are indirectly related. When rank collapse occurs, the directions of $BA$ converge, weakening gradient signals in different singular directions (similar to vanishing), forming a positive feedback loop: lower effective rank → more concentrated gradients → even lower effective rank. Breaking this cycle requires external intervention (e.g., scaling corrections, SVD projection, etc.).

</details>

---

<details>
<summary>Q17: What is the core idea of LoRA+? How does it differ from standard LoRA?</summary>

**Answer:** Based on muP (Maximal Update Parameterization) analysis, using the same learning rate for $A$ and $B$ in standard LoRA is suboptimal. Theoretically, $B$ should use a higher learning rate than $A$ ($\eta_B / \eta_A = \lambda$, recommended $\lambda \in [2, 16]$).

Reason: in the infinite-width limit, the gradient signal magnitudes of $A$ and $B$ differ; a uniform learning rate causes one to underfit and the other to overfit. In practice, only a higher learning rate for $B$ in the optimizer needs to be set — a minimal change.

**Follow-up:** What is the core idea of muP? How can muP be used for LoRA hyperparameter search?

> **Answer:** The core of muP is deriving, in the infinite-width limit, a "maximally updated" parameterization scheme such that hyperparameters tuned on a small model can be transferred directly to a large model (width transfer). For LoRA+, this means the optimal $\lambda$, learning rate, etc. found on a small rank, small model can be applied directly to a large model, greatly reducing the search cost on large models.

</details>

---

### ━━━ L3 Deep ━━━

---

<details>
<summary>Q18: Is LoRA's low-rank assumption theoretically supported? What are its limitations?</summary>

**Answer:** Partial support: (1) **intrinsic dimensionality** (Aghajanyan et al., 2020): random subspace probing shows that task-relevant parameter changes have low-dimensional structure; (2) **spectral analysis**: after FFT, the singular values of $\Delta W$ decay rapidly; (3) **implicit regularization**: the BA structure implicitly applies nuclear norm regularization.

**Limitations**: not all tasks suit low-rank updates — tasks involving large-scale knowledge injection, complex reasoning, etc. find low-rank $\Delta W$ insufficient in expressiveness, which is precisely the motivation behind high-rank methods like HiRA and BoHA.

**Follow-up:** How is intrinsic dimensionality measured? Does it still hold for LoRA at the RLHF stage?

> **Answer:** The measurement method is to train in a random subspace: freeze $W_0$, optimize within a low-dimensional random projection $W_0 + P_d \theta$ ($P_d \in \mathbb{R}^{|W| \times d}$ random matrix, $\theta \in \mathbb{R}^d$), and gradually increase $d$ to observe performance. The low-rank assumption may be weaker at the RLHF stage — behavior modifications guided by reward signals are more complex, and in practice RLHF with LoRA typically requires larger rank or more careful hyperparameter tuning.

</details>

---

<details>
<summary>Q19: What are HiRA and ABBA? How does the Hadamard product increase effective rank?</summary>

**Answer:** Mathematical foundation: the Hadamard product of two rank-$r$ matrices can have rank as high as $r^2$ (Khatri-Rao product property).

**HiRA**: $\Delta W = W_0 \odot (BA)$, taking the Hadamard product of the full-rank frozen $W_0$ with the low-rank $BA$. $W_0$ is frozen; only $BA$ is trainable. Parameter count is the same as LoRA but the effective rank far exceeds $r$.

**ABBA**: an alternating block Hadamard structure based on similar principles, with a slightly different formulation.

According to their respective original papers, both outperform or are competitive with standard LoRA at the same parameter count on tasks requiring high expressiveness; gains may vary across different tasks and scales.

**Follow-up:** Mathematical principle behind the high-rank property of the Khatri-Rao product? Can HiRA and DoRA be combined?

> **Answer:** The columns of the Khatri-Rao product $A \odot B$ are the Kronecker products of the corresponding columns; the Kronecker product of two rank-$r$ matrices has rank $r^2$, and the Khatri-Rao product, as a column-aligned Kronecker product, retains part of this high-rank property. HiRA + DoRA can be combined: applying magnitude-direction decomposition on top of HiRA's Hadamard update would simultaneously benefit from high-rank expressiveness and magnitude learning, but experimental validation is needed to confirm whether there is additional gain.

</details>

---

<details>
<summary>Q20: What is the difference between BoHA's block-wise Hadamard strategy and HiRA?</summary>

**Answer:** BoHA partitions $W_0$ into $b \times b$ blocks, with each block independently learning LoRA factors $(A_i, B_i)$ and applying a Hadamard product:

$$(\Delta W)_{\text{block}_i} = (W_0)_{\text{block}_i} \odot (B_i A_i)$$

Key differences from HiRA:
- HiRA applies the Hadamard product using the entire $W_0$; all blocks share the same LoRA factors
- BoHA partitions independently — finer granularity, capable of capturing differences across attention heads / feature groups
- Total parameter count is equivalent to LoRA; zero additional inference overhead after merging


**Follow-up:** How does the block strategy (block size $b$) affect performance? How does expressiveness change as $b$ increases?

> **Answer:** Larger $b$ means larger (more parameter-rich) LoRA factors per block, increasing each block's expressiveness; but the number of blocks decreases, weakening global cross-block coordination. Too small a $b$ leaves too few parameters per block, resulting in insufficient expressiveness. In theory, there exists an optimal $b$ that aligns with the intrinsic structure of $W_0$ (e.g., the attention head dimension). In practice, $b$ is typically set to a multiple of the head dimension.

</details>

---

<details>
<summary>Q21: What is PiSSA? How does it differ from LoRA initialization?</summary>

**Answer:** PiSSA applies SVD to $W_0$ and initializes $A$ and $B$ from the top $r$ components ($BA \approx U_r \Sigma_r V_r^T \approx$ the principal components of $W_0$); the frozen part becomes the residual $W_0 - BA$.

Difference from LoRA: LoRA learns $\Delta W$ starting from zero; PiSSA starts from the principal components, with an initial point closer to the structure of $W_0$, resulting in faster convergence.

**Follow-up:** Is the merge operation for PiSSA consistent with that of LoRA? How is the residual handled differently?

> **Answer:** The merge operation is similar: $W_{\text{merged}} = (W_0 - U_r\Sigma_r V_r^T) + B_{\text{trained}} A_{\text{trained}}$. The difference is that the frozen part is the residual rather than the original $W_0$, meaning the merged model's mathematical structure differs from LoRA's $W_0 + BA$ merge — but the inference behavior is equivalent (both result in a $d \times k$ weight matrix after merging).

</details>

---

<details>
<summary>Q22: What is GaLore? How is its approach fundamentally different from LoRA?</summary>

**Answer:** GaLore applies low-rank projection to **gradients** (not weights): gradient $G \approx U_r \Sigma_r V_r^T$; optimizer state is maintained only in the $r$-dimensional projection; the projection direction is refreshed every $T$ steps. All parameters are updated; the optimizer maintains state in low dimensions.

**Fundamental difference**:
- **LoRA**: fixed parameter structure, freezes the base model; supports merging and adapter reuse
- **GaLore**: updates all parameters; the result is a standard model at inference time; does not support adapter reuse

GaLore is suitable for pre-training / continual pre-training (requiring full updates but with limited memory); LoRA is more practical for SFT.

**Follow-up:** How should the subspace refresh frequency $T$ in GaLore be chosen? How is optimizer state handled at refresh?

> **Answer:** The refresh frequency $T$ is typically set to a few hundred to a few thousand steps (the paper recommends 200). At refresh, the old optimizer state (momentum, variance) is discarded and re-initialized in the new projection subspace — this introduces periodic optimization discontinuities, but in practice the impact on convergence is limited. Too small a $T$ leads to frequent state discards and reduced efficiency; too large a $T$ causes the gradient projection to deviate from the optimal subspace.

</details>

---

<details>
<summary>Q23: What advantages do PEFT methods have in continual learning (CL) scenarios?</summary>

**Answer:** PEFT's CL advantages: (1) the base model is frozen and each task trains an independent adapter → natural isolation in parameter space; (2) multi-task storage requires only multiple lightweight adapters; (3) supports dynamic loading/switching.

$W_0$-coupled Hadamard structures (such as HiRA and BoHA) have an additional advantage in CL — $W_0$ acts as an implicit "anchor", constraining updates from drifting too far from the original parameter space, theoretically preserving more knowledge from previous tasks.


**Follow-up:** Is BoHA's advantage in CL inherent to the architecture ($W_0$-coupling) or due to having more parameters? How would you verify this?

> **Answer:** This should be verified through ablation studies: (1) compare BoHA and LoRA with the same parameter count on CL retention, ruling out the parameter count factor; (2) compare BoHA with non-coupled block LoRA (block-independent LoRA without the Hadamard coupling to $W_0$), verifying the contribution of $W_0$-coupling; (3) progressively increase the number of tasks and observe the catastrophic forgetting curve.

</details>

---

<details>
<summary>Q24: How do you evaluate the quality of PEFT methods? What are the key considerations for fair comparison?</summary>

**Answer:** Evaluation dimensions:
1. **Parameter count alignment**: must compare under the same trainable parameter count
2. **Task diversity**: cover NLU (GLUE), reasoning (GSM8K/MATH), and generation (MT-Bench)
3. **Model scale**: conclusions may reverse at different scales
4. **Training hyperparameters**: per-method tuning rather than uniform hyperparameters
5. **Inference cost**: can it be merged? Is the latency consistent after merging?

Pitfalls in fair comparison: LoRA vs Adapter must account for inference latency differences; comparisons among Hadamard methods must strictly align total rank and parameter count.

**Follow-up:** Should the comparison baseline for PEFT be FFT or a LoRA baseline? Why are both needed?

> **Answer:** Both are needed: FFT is the **upper bound** (how much can the task benefit from more parameters), and LoRA is the **practical baseline** (whether the new method is better at the same cost). Comparing only against FFT fails to demonstrate parameter efficiency advantages; comparing only against LoRA cannot determine the overall ceiling. A complete evaluation should present three bars: FFT, LoRA, and the new method.

</details>

---

<details>
<summary>Q25: What are the applications of PEFT methods in multimodal (Vision-Language) and Diffusion models?</summary>

**Answer:**

**Vision-Language (VLM)**:
- LoRA is widely used for SFT in VLMs such as CLIP, LLaVA, and InternVL; LoRA can be applied to both the visual encoder and LLM decoder simultaneously, or only to the LLM part (with the visual encoder frozen)
- Prefix tuning / adapter were used in early VLMs (e.g., Flamingo) for cross-attention adapter injection of visual information

**Diffusion Models**:
- **LoRA for Diffusion** (e.g., SDXL LoRA): adds LoRA to attention/conv layers in the UNet for style/character customization, with only a few MB of parameters
- **IP-Adapter**: frozen CLIP image encoder + cross-attention adapter
- **ControlNet**: copies the entire encoder branch (not strict PEFT); belongs to the "lock original + train copy" paradigm

**Follow-up:** What toolchains support merging multiple style LoRAs in diffusion? What are the differences from LLM LoRA merging?

> **Answer:** The diffusion community has active merging tools (e.g., built-in merge in `sd-webui`, `mergekit` extensions, `ComfyUI` LoRA stacking). Key differences from LLMs: (1) diffusion LoRAs typically target multi-layer attention in the UNet, and per-layer weights can be set differently when merging; (2) multi-LoRA combinations are often used for blending styles (e.g., "anime + realistic"), and the community has developed empirical weight recipes; (3) diffusion merging focuses more on visual quality (FID / human evaluation) rather than accuracy metrics.

</details>

---

## Quick Reference

| Method | Parameter scale | Merge? | Extra inference latency | Use case |
|--------|----------------|--------|-------------------------|----------|
| **LoRA** | 0.1–1% | ✅ | None | General-purpose SFT first choice |
| **QLoRA** | Same as LoRA (4-bit base) | ✅ | None (dequant overhead minimal) | Memory-constrained |
| **DoRA** | LoRA + minimal | ✅ | None | Near-FFT performance needed |
| **VeRA** | ≈ LoRA / $k$ | ✅ | None | Extreme parameter constraints |
| **AdaLoRA** | Adaptive | ✅ | None | Automatic rank allocation needed |
| **(IA)³** | Very small (vectors) | ✅ | None | Few-shot |
| **HiRA / BoHA** | Same as LoRA | ✅ | None | High-rank expressiveness needed |
| **Adapter** | ~LoRA | ❌ | Yes | Dynamic swap |
| **Prefix Tuning** | Small | ❌ | Yes (KV cache occupancy) | Generative tasks |
| **Prompt Tuning** | Very small | ❌ | Yes (same as above) | Few-shot with large models |
| **GaLore** | Full parameters | ❌ (not an adapter) | None | Pre-training |

---

*Built for research learning. No benchmark numbers are fabricated; specific experimental claims from individual papers are excluded or marked for verification.*


## Extended L3

<details>
<summary>Q: Why does AdaLoRA adopt an SVD form ($P\Lambda Q$) rather than standard BA parameterization for adaptive rank allocation? What mathematical advantage does the orthogonality constraint on $P\Lambda Q$ provide in this setting?</summary>

   A: In standard LoRA's $BA$ parameterization, there is no orthogonality constraint between different singular value directions; during training the basis vectors can gradually "collapse" into the same subspace (mode collapse), causing the effective rank to fall far below the nominal rank and making importance scores less discriminative. The SVD form enforces $P$ and $Q$ to be orthogonal matrices, ensuring that singular value directions are mutually independent, so that the ranking of $\text{Importance}_i = |\lambda_i| \cdot |\partial\mathcal{L}/\partial\lambda_i|$ is more meaningful — after removing low-importance singular values, the remaining directions genuinely span different signal subspaces. Moreover, the diagonal $\Lambda$ structure naturally supports per-singular-value mask/prune operations, whereas in the $BA$ form "removing the rank-$i$ component" requires simultaneously modifying the $i$-th column of both matrices, leading to a discontinuous optimization path. The cost is that orthogonality constraints on $P$ and $Q$ require additional projection operations (e.g., Cayley parameterization or periodic QR decomposition), increasing implementation complexity.

   **Follow-up**: If the orthogonality constraint is enforced via an approximate method (e.g., a regularization term $\|P^T P - I\|_F^2$) rather than exact projection, how does the reliability of importance scores degrade when orthogonality deviates significantly in the late training phase? What is the theoretical bound on the impact on final task performance?

</details>



<details>
<summary>Q: The learnable vectors in <code>(IA)³</code> are mathematically equivalent to learning a diagonal matrix to rescale activations. From the perspective of the rank of linear operators, why does this "purely diagonal" adaptation have a theoretical expressiveness ceiling? Is there a strict inclusion relationship between it and LoRA (low-rank matrix adaptation)?</summary>

   A: The operation that (IA)³ applies to key/value/FFN outputs can be written as $h' = D \cdot h$, where $D = \text{diag}(l)$ is a diagonal matrix. A diagonal matrix is a linear operator that is full-rank (rank $d$) but has only $d$ degrees of freedom; it can only perform anisotropic scaling along existing coordinate axes and cannot rotate feature directions or establish new cross-dimensional associations. By contrast, LoRA's $\Delta W = BA$ is a rank-$r$ matrix with $r(d+k)$ degrees of freedom, capable of producing linear combinations across dimensions. The two do not have a strict inclusion relationship: certain diagonal scalings achievable by (IA)³ (e.g., "amplify dimension $i$ by 100×") require LoRA to use a higher rank to approximate (because the diagonal elements of $BA$ are coupled), while LoRA's off-diagonal mappings are strictly inexpressible by (IA)³. In an information-theoretic sense, the per-layer transformation after (IA)³ adaptation is still a coordinate-scaled version of the original weight $W_0$, while LoRA can map the null space of $W_0$ to nonzero outputs — this is a fundamental capability gap.

   **Follow-up**: If the diagonal vectors in (IA)³ are extended to block-diagonal matrices (each block $b \times b$), with parameter count growing linearly to $b \cdot (d/b)$, how does expressiveness change with block size $b$? At what value of $b$ can the effective rank approach that of a rank-$r$ LoRA?

</details>



<details>
<summary>Q: DoRA decomposes the weight update into magnitude and direction components and introduces column-norm normalization. From the perspective of Riemannian optimization or constrained optimization, how does this decomposition change the optimization landscape of LoRA's parameter space? Why can it more closely approximate FFT's gradient behavior?</summary>

   A: Standard LoRA's parameter space is the unconstrained $(A, B) \in \mathbb{R}^{r \times k} \times \mathbb{R}^{d \times r}$; $\Delta W = BA$ couples magnitude and direction: increasing $A$ and $B$ simultaneously changes both norm and direction. DoRA's column normalization $\hat{W}_c = (W_0 + BA)_c / \|(W_0 + BA)_c\|$ constrains direction to the column unit sphere $\mathbb{S}^{d-1}$ (a Riemannian manifold), while magnitude is independently controlled by the vector $m$ in Euclidean space. This forms a product manifold of "sphere × Euclidean space," naturally decoupling the gradients of magnitude and direction. In FFT, gradients simultaneously update both degrees of freedom (magnitude and direction) of $W$; in LoRA, when only direction is updated, magnitude is "anchored" by the norm of $W_0$, constraining the search. DoRA restores an independent control channel for magnitude, making the geometry of parameter updates closer to the full parameter space. A key additional constraint is introduced by the column normalization: it implicitly imposes a form of regularization that prevents any single column's weight from growing excessively, consistent with the implicit magnitude-adaptive behavior observed in FFT.

   **Follow-up**: DoRA's column normalization introduces a division operation into the computation graph. In mixed-precision training (BF16 forward, FP32 gradient accumulation), when certain column norms approach zero, does numerical stability become a concern? Are there alternative normalization schemes that could preserve the same optimization benefits while improving numerical behavior?

</details>



<details>
<summary>Q: The core challenge in multi-task LoRA merging (e.g., TIES-Merging, DARE) is parameter interference between adapters. From a subspace geometry perspective, how does the angle between two tasks' LoRA subspaces $\mathcal{S}_1 = \text{col}(B_1)$ and $\mathcal{S}_2 = \text{col}(B_2)$ affect performance after linear merging? What happens when the angle is very small (near overlap) versus near orthogonal?</summary>

   A: The angle between two tasks' LoRA column spaces can be measured by principal angles. When $\mathcal{S}_1$ and $\mathcal{S}_2$ nearly overlap (principal angle $\approx 0$), the two tasks impose updates of different signs in the same direction, and linear merging causes them to cancel each other — this is precisely the situation that TIES addresses with its sign disagreement detection and trim operation. When $\mathcal{S}_1 \perp \mathcal{S}_2$ (principal angle $\approx \pi/2$), merging introduces almost no interference because the signals lie in orthogonal subspaces. In practice, most cases are intermediate: some directions overlap, others are independent. DARE reduces interference in overlapping directions through random drop + rescale of low-magnitude parameters; its implicit assumption is that frequently dropped parameters are more likely to be task-shared noise directions, while those that survive are task-specific high-signal directions. The deeper theoretical question is whether PEFT parameter space exhibits "task additivity" — i.e., whether the adaptations for different tasks approximately lie in disjoint low-dimensional subspaces. If this assumption holds (intrinsic task dimensionalities are mutually orthogonal), simple averaging suffices; if subspaces overlap significantly, more aggressive interference-removal strategies are needed.

   **Follow-up**: Can performance degradation after merging be predicted in advance by computing $\text{tr}(B_1^T B_2 B_2^T B_1)$ or a similar metric? If literature or experiments show that this metric is strongly correlated with positive/negative transfer between tasks, what guidance does it offer for the strategy of "choosing which LoRAs to merge together"?

</details>



<details>
<summary>Q: The original LoRA paper suggests that applying adaptation to the Q and V matrices of attention layers alone captures most of the gains. From the computation graph of the Transformer attention mechanism, why is low-rank perturbation of Q and V more "efficient" than perturbation of K and the projection matrix ($W_O$)? Is this related to the softmax nonlinearity in attention scores?</summary>

   A: In the attention computation $\text{softmax}(QK^T/\sqrt{d_k})V$, Q and K jointly determine the attention pattern (weight distribution), while V provides the content being aggregated. A low-rank perturbation $\Delta Q$ to Q changes the projection angle of queries in key space, directly affecting "which tokens to attend to" — a decision highly sensitive to downstream tasks. A perturbation $\Delta V$ to V directly changes the value vectors being aggregated, directly affecting the output representation. The key is the nonlinear effect of softmax: softmax has a saturation region for changes to $QK^T$ — when some attention logits are much larger than others, the softmax output approaches one-hot, at which point perturbations to K are "compressed" (gradients shrink after passing through softmax). In other words, K's perturbation has its influence attenuated by the softmax nonlinearity and requires larger magnitude to produce equivalent output changes, which is exactly what the low-rank constraint limits in magnitude. Therefore, under the same rank budget, perturbing Q (directly controlling attention direction before softmax) is more efficient than perturbing K (attenuated after softmax). The $W_O$ projection matrix maps multi-head outputs back to the hidden dimension and functions more as a "general linear transform"; the base model has already learned a good initialization for it, leaving less "room" for task-specific adjustment in the low-rank space.

   **Follow-up**: For models using GQA (Grouped Query Attention), where the number of K/V heads is fewer than Q heads and multiple Q heads share a single K/V group — would applying LoRA to the shared K/V have amplified effects (since a single KV head's perturbation affects multiple Q heads), potentially changing the Q/V-first strategy?

</details>



<details>
<summary>Q: Soft Prompt methods (Prompt Tuning / Prefix Tuning) influence model behavior by prepending learnable tokens to the input sequence, while LoRA modifies weight matrices. From a function class perspective, what subspace does each class of methods restrict the model's search for an optimal solution to? Are there task families where Soft Prompt can theoretically express things that LoRA cannot, or vice versa?</summary>

   A: After LoRA adaptation, the model's transformation is $h = (W_0 + BA)x$, which changes the model's mapping function for **all inputs** — this is a global affine perturbation. Soft Prompt's influence works by prepending prefix tokens $p_1, \ldots, p_k$ to the input sequence, using the attention mechanism to let these tokens serve as additional context that modulates the representations of subsequent tokens — it fundamentally does not change the model's **parameterized function** but changes the **input distribution** of that function. This means: (1) Soft Prompt's effect on all tokens exhibits **positional decay** — tokens further from the prefix are less affected (as attention weights decay with distance), while LoRA affects every token in the sequence equally. (2) From a function class perspective, LoRA can implement linear rotations in input space in any direction (limited by rank), whereas Soft Prompt can only influence representations **indirectly** through the QKV mechanism of attention — limited by the "transmission bandwidth" of the model's own attention pattern. Theoretically, for tasks requiring **globally consistent modification of a reasoning rule** (e.g., changing a language's syntactic preference), LoRA is more appropriate; for tasks requiring **dynamic behavior adjustment based on specific context** (e.g., few-shot example guidance), Soft Prompt is more natural since it directly provides context. In the limit, a sufficient number of prefix tokens (occupying the entire context window) can simulate arbitrary behavior changes, at which point Soft Prompt's expressiveness approaches that of LoRA.

   **Follow-up**: Prefix Tuning prepends a prefix to K/V in every attention layer, while Prompt Tuning only adds tokens at the input layer. From an information flow perspective, the per-layer prefix is equivalent to injecting additional "virtual context" in each attention computation. Does this layer-by-layer injection compared to input-only injection equate to increasing the model's **effective depth**? Can this gain be quantified using a recursive information propagation framework?

</details>



<details>
<summary>Q: LoRA's rank selection is typically a global hyperparameter, but different layers' weight matrices in a Transformer learn very different structures during pretraining. From the perspective of the singular value spectrum of weight matrices, can one infer each layer's "intrinsic adaptation dimensionality" from the spectral decay characteristics of $W_0$, thereby guiding per-layer rank allocation? What are the advantages and disadvantages compared to AdaLoRA's online allocation?</summary>

   A: Performing SVD analysis on pretrained weights $W_0$, the decay rate of the singular value spectrum $\sigma_1 \geq \sigma_2 \geq \cdots$ reflects the matrix's "effective rank" — rapidly decaying layers have weights concentrated in a low-dimensional subspace; theoretically their updates $\Delta W$ are also more likely to be low-rank (since the adjustable directions are limited). Slowly decaying layers (more uniform singular values) mean the weights are more "spread out" and may require higher rank to cover the adaptation directions. A reasonable heuristic: the "elbow point" of the singular value spectrum, or the minimum $r$ satisfying $\sum_{i=1}^{r}\sigma_i / \sum_i \sigma_i > \theta$ (e.g., $\theta = 0.9$), can serve as a reference for each layer's intrinsic adaptation dimensionality. However, this static analysis has a limitation: the spectrum of $W_0$ reflects the structure of the **pretraining task**, not the adaptation needs of the **downstream task** — some spectral tail directions unimportant for pretraining may be critical for downstream tasks. AdaLoRA's online allocation advantage is that it dynamically adjusts based on the gradient signal from the downstream task ($\partial\mathcal{L}/\partial\lambda_i$), capturing "which directions the downstream task needs" rather than "which directions were important in pretraining." The two approaches can be complementary: use spectral analysis as a prior (warm start) and online allocation as a correction.

   **Follow-up**: If AdaLoRA's final rank distributions are computed separately for multiple different downstream tasks, is there a "cross-task consistency" in these distributions (i.e., different tasks agree that certain layers/matrices need high rank)? If such consistency exists, would it imply that a "universal rank allocation table" could be precomputed for direct use on new tasks without the overhead of online adjustment?

</details>



<details>
<summary>Q: GaLore compresses optimizer state by periodically applying low-rank SVD projection to gradients, with subspace refresh interval $T$ as the core design choice. From the perspective of online subspace tracking, what failure modes does too large or too small a value of $T$ lead to? Can the optimal $T$ be formalized using the "drift rate" of the gradient subspace?</summary>

   A: Problems when $T$ is too small (frequent refresh): the computational cost of each SVD is non-negligible (performing SVD on the full gradient is $O(\min(mn^2, m^2n))$), and frequent execution significantly slows training. More importantly, too short a subspace lifetime means optimizer state (e.g., Adam's m, v) cannot accumulate sufficient statistical information in the projection space, making momentum and adaptive learning rate estimates inaccurate. Problems when $T$ is too large: as training progresses, the curvature structure of the loss function changes, and the principal subspace of the gradient drifts accordingly. If $T$ is too large, the current projection basis $U_r$ is no longer the optimal low-rank approximation — the true principal directions of gradients may have "drifted out" of the current subspace, causing the projected gradient signal to lose key components, and training stagnates at a suboptimal point. Formally, one can define the "drift" of the gradient subspace between two steps as $\delta(t) = \|P_{\mathcal{S}_{t+T}} - P_{\mathcal{S}_t}\|$ (difference norm of projection operators); the optimal $T$ should keep $\mathbb{E}[\delta(T)]$ below some threshold — i.e., the cumulative drift within $T$ steps should not exceed the error tolerance allowed by the projection dimensionality. In practice, curvature changes quickly in the early training phase (large $\delta$), requiring smaller $T$; as training approaches convergence (small $\delta$), larger $T$ can be used — so an adaptive refresh interval (based on monitoring gradient projection residuals) is better than a fixed $T$.

   **Follow-up**: GaLore uses top-$r$ SVD for projection, and the top singular vectors of the gradient matrix may mainly encode high-frequency batch noise rather than stable optimization directions. Is it possible to replace exact SVD with randomized SVD or incremental PCA to reduce noise while lowering computational cost? How would this approximation propagate statistical estimation errors in optimizer state to final model quality?

</details>

## §A Key Papers Timeline

- **2019-02 · Adapter (Parameter-Efficient Transfer Learning)** — Houlsby et al., ICML 2019. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751) — Inserts small bottleneck MLP modules into each Transformer block while freezing the pretrained backbone, establishing the foundational PEFT paradigm that "lightweight inserted layers can match full fine-tuning."

- **2021-01 · Prefix Tuning** — Li & Liang, ACL 2021. [arXiv:2101.00190](https://arxiv.org/abs/2101.00190) — Prepends learnable prefix vectors to the K and V of every attention layer (reparameterized via MLP for stability); achieves near full fine-tuning performance on generation tasks with only 0.1% of parameters.

- **2021-04 · Prompt Tuning** — Lester et al., EMNLP 2021. [arXiv:2104.08691](https://arxiv.org/abs/2104.08691) — Appends learnable soft tokens only at the input embedding layer while freezing the entire model; shows that at sufficient scale, soft-prompt tuning matches full model tuning with minimal parameters.

- **2021-06 · LoRA (Low-Rank Adaptation)** — Hu et al., ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — Hypothesizes that weight updates ΔW have low intrinsic rank and parameterizes them as ΔW = BA; adapters can be merged back into the base model for zero inference overhead, forming the basis for all subsequent LoRA variants.

- **2022-05 · (IA)³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)** — Liu et al., NeurIPS 2022. [arXiv:2205.05638](https://arxiv.org/abs/2205.05638) — Multiplies keys, values, and FFN outputs by learned per-layer scaling vectors; achieves extreme parameter efficiency (few thousand parameters per layer) suited for few-shot fine-tuning with mergeable zero-overhead inference.

- **2023-03 · AdaLoRA (Adaptive Budget Allocation)** — Zhang et al., ICLR 2023. [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) — Parameterizes ΔW in SVD form PΛQ and prunes singular value triplets by importance score (magnitude × gradient sensitivity), adaptively allocating higher rank to more critical layers under a fixed parameter budget.

- **2023-05 · QLoRA (Quantized LoRA)** — Dettmers et al., NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — Quantizes the base model to 4-bit NF4 while keeping LoRA adapters in BF16, with Double Quantization and Paged Optimizer; first demonstrated fine-tuning a 65B model on a single GPU.

- **2023-10 · VeRA (Vector-based Random Matrix Adaptation)** — Kopiczko et al., ICLR 2024. [arXiv:2310.11454](https://arxiv.org/abs/2310.11454) — Shares a single pair of frozen random matrices across all layers and trains only per-layer diagonal scaling vectors, compressing trainable parameters to ~1/k of LoRA for extreme parameter-efficiency scenarios.

- **2024-02 · DoRA (Weight-Decomposed Low-Rank Adaptation)** — Liu et al., ICML 2024. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) — Decomposes weight updates into magnitude (learnable vector m) and direction (LoRA low-rank update), making gradient behavior closer to full fine-tuning with negligible extra parameters.

- **2024-02 · LoRA+ (Decoupled Learning Rates for LoRA)** — Hayou et al., ICML 2024. [arXiv:2402.12354](https://arxiv.org/abs/2402.12354) — Uses muP scaling analysis to show B should receive a higher learning rate than A (recommended ratio λ∈[2,16]), correcting suboptimal optimization of standard LoRA in large-width networks.

- **2024-03 · GaLore (Gradient Low-Rank Projection)** — Zhao et al., ICML 2024. [arXiv:2403.03507](https://arxiv.org/abs/2403.03507) — Projects gradients onto a low-rank subspace via SVD and maintains optimizer states only in that r-dimensional space (refreshed every T steps), enabling memory-efficient full-parameter pre-training without adapter constraints.

- **2024-04 · PiSSA (Principal Singular Values and Singular Vectors Adaptation)** — Meng et al., NeurIPS 2024. [arXiv:2404.02948](https://arxiv.org/abs/2404.02948) — Initializes LoRA's A and B matrices with the top-r singular components of W₀, leaving the residual as the frozen part; starts optimization from the principal subspace of the pretrained weights for faster convergence than standard LoRA.
