# PEFT 参数高效微调 速查手册
# PEFT — Parameter-Efficient Fine-Tuning Cheat Sheet

> 面向 LLM post-training 方向的研究与工程人员，覆盖主流 PEFT 方法全栈知识。

---

## 第一部分：概念与公式推导 / Part 1: Concepts & Formula Derivations

### 1.1 为什么需要 PEFT？ / Why PEFT?

Full Fine-Tuning（FFT，全参数微调）对预训练模型 **全部参数** 做梯度更新，面临以下问题：

| 问题 | 说明 |
|------|------|
| **显存成本高** | 参数(BF16) + 梯度(BF16) + Adam 优化器状态 m/v(FP32)，7B 模型约 84 GB+（不含激活值、不含 FP32 master copy；含 master copy 约 112 GB） |
| **灾难性遗忘** (Catastrophic Forgetting) | 全参数更新可能覆盖预训练知识 |
| **多任务部署代价大** | 每个任务保存一份完整模型，存储和切换成本高 |
| **训练不稳定** | 学习率敏感，需精细调优 |

**PEFT 核心思路**：冻结 base model 参数 $W_0$，仅训练少量额外参数 $\Delta\Theta$，达到接近 FFT 的性能。

$$W_{\text{final}} = W_0 + \Delta W(\Theta), \quad |\Theta| \ll |W_0|$$

---

### 1.2 LoRA — Low-Rank Adaptation / 低秩适配

**核心假设**：预训练权重的更新 $\Delta W$ 是低秩的。

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B A$$

其中：

- $W_0 \in \mathbb{R}^{d \times k}$：冻结的预训练权重（frozen pretrained weight）
- $A \in \mathbb{R}^{r \times k}$：下投影矩阵（down-projection），初始化为 Kaiming uniform
- $B \in \mathbb{R}^{d \times r}$：上投影矩阵（up-projection），初始化为 **零矩阵**
- $r \ll \min(d, k)$：秩（rank），控制表达能力
- $\alpha$：缩放超参（scaling hyperparameter），控制 LoRA 更新的整体幅度

**可训练参数量**（每层）：

$$|\Theta_{\text{LoRA}}| = r \times (d + k)$$

**初始化设计意图**：

- $B = 0$：训练起始 $\Delta W = BA = 0$，不扰动 base model 输出，梯度信号稳定
- $A \sim \text{Kaiming Uniform}$：非零初始化保证梯度能流过（若 $A$ 也为零，则梯度恒为零）

**Merge（合并）**：推理时可将 LoRA 权重合并回 base model，**零额外推理延迟**。

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} \cdot B A$$

---

### 1.3 rsLoRA — Rank-Stabilized LoRA / 秩稳定缩放

标准 LoRA 使用缩放因子 $\alpha / r$。当 rank $r$ 增大时，$BA$ 的 Frobenius 范数随 $\sqrt{r}$ 增长，导致梯度过大、训练不稳定。

**rsLoRA 修正**：将缩放改为 $\alpha / \sqrt{r}$：

$$h = W_0 x + \frac{\alpha}{\sqrt{r}} \cdot B A x$$

这保证不同 rank $r$ 下更新幅度的量级一致，使大 rank 下训练更稳定。

---

### 1.4 DoRA — Weight-Decomposed Low-Rank Adaptation / 幅度-方向分解

DoRA 将权重更新分解为 **幅度（magnitude）** 和 **方向（direction）** 两部分：

$$W = m \cdot \frac{W_0 + \frac{\alpha}{r} B A}{\left\| W_0 + \frac{\alpha}{r} B A \right\|_c}$$

- $m \in \mathbb{R}^{k}$：可学习幅度向量（per-column magnitude），每列一个缩放因子
- $\|\cdot\|_c$：按列取范数（column-wise norm），使每列成为单位方向向量

**动机**：分析发现 FFT 同时改变权重的幅度和方向，而标准 LoRA 主要改变方向、幅度调整能力弱。加入可学习幅度后，梯度行为更接近 full fine-tuning。

**额外参数量**：$k$（每层一个幅度向量），相对于 LoRA 的 $r \times (d+k)$ 可忽略。

---

### 1.5 VeRA — Vector-based Random Matrix Adaptation / 向量随机矩阵适配

VeRA 的极致参数效率：所有层 **共享同一套随机固定矩阵** $A$ 和 $B$，仅训练 **对角缩放向量**：

$$\Delta W = \text{diag}(b) \cdot B \cdot \text{diag}(d) \cdot A$$

- $A \in \mathbb{R}^{r \times k}$，$B \in \mathbb{R}^{d \times r}$：随机初始化后 **冻结**，所有层共享
- $b \in \mathbb{R}^{d}$，$d \in \mathbb{R}^{r}$：可训练的对角缩放向量

**参数量**：每层仅需 $r + d$ 个参数（两个向量），相比 LoRA 的 $r \times (d + k)$ 极大压缩。

**代价**：共享矩阵 + 仅对角缩放限制了表达力，在复杂任务上与 LoRA 有差距。

---

### 1.6 AdaLoRA — Adaptive LoRA / 自适应秩分配

观察到不同层、不同矩阵对任务的重要性不同，固定 rank 是次优的。

**参数化**（类 SVD 形式）：

$$\Delta W = P \cdot \Lambda \cdot Q$$

其中 $P$、$Q$ 为正交矩阵，$\Lambda$ 为对角矩阵（奇异值）。

**Importance Scoring（重要性评分）**：对每个奇异值三元组 $(p_i, \lambda_i, q_i)$ 计算：

$$\text{Importance}_i = |\lambda_i| \times \left| \frac{\partial \mathcal{L}}{\partial \lambda_i} \right|$$

训练过程中将 importance 低的奇异值 mask 为零（prune），将参数预算留给重要性高的矩阵/层。

**效果**：在相同参数预算下，关键层（如较高层 attention）获得更多 rank。

---

### 1.7 (IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

(IA)³ 对 attention 的 key/value 和 FFN 的输出分别乘以 **可学习缩放向量**（非矩阵）：

$$K' = l_k \odot K, \quad V' = l_v \odot V, \quad h_{\text{FFN}}' = l_{\text{ff}} \odot h_{\text{FFN}}$$

- $l_k \in \mathbb{R}^{d_k}$，$l_v \in \mathbb{R}^{d_v}$，$l_{\text{ff}} \in \mathbb{R}^{d_{\text{ff}}}$：可训练向量

**参数量极小**：每层仅约 $d_k + d_v + d_{\text{ff}}$ 个参数。适合 few-shot 场景，但在复杂任务上表达力不足（仅做 activation rescaling，无法学方向性变化）。

推理时可将缩放向量融合进 $W_K$、$W_V$ 和 $W_{\text{FFN}}$，实现零额外延迟。

---

### 1.8 Soft Prompt 家族 / Soft Prompt Family

| 方法 | 可学习参数位置 | 每层 or 仅输入层 | 特点 |
|------|-------------|-----------------|------|
| **Prompt Tuning** | input embedding 层的 $k$ 个 soft token | 仅输入层 | 参数极少；大模型效果好，小模型差 |
| **Prefix Tuning** | 每层 attention 的 K/V 前拼 learnable prefix | 每层 | 表达力比 Prompt Tuning 强；需 reparameterize (MLP) 才训练稳定 |
| **P-Tuning v2** | 每层都加 prefix（类 Prefix Tuning） | 每层 | 统一框架，NLU 上接近 FFT |

**共同局限**：推理时无法 merge 到 base model（prefix 在 KV cache 中占位），对 KV cache 有额外显存消耗。

---

### 1.9 Adapter / 瓶颈适配模块

在每个 Transformer block 中插入小型 **bottleneck MLP**：

$$h \rightarrow \text{LayerNorm} \rightarrow W_{\text{down}} (d \to r) \rightarrow \sigma(\cdot) \rightarrow W_{\text{up}} (r \to d) \rightarrow \text{Residual} \rightarrow h'$$

**参数量**：$2 \times r \times d$（每层），与 LoRA 相当。

**与 LoRA 的推理时区别**：
- **Adapter**：推理时有额外的串行前向传播层（bottleneck MLP），增加 latency
- **LoRA**：merge 后零额外层，零延迟

---

### 1.10 Hadamard 乘积系列 / Hadamard Product Family

**核心动机**：标准 LoRA 的 $\Delta W = BA$ 满足 $\text{rank}(\Delta W) \leq r$。能否在同样参数量下获得更高有效秩（effective rank）？

**数学基础**：两个 $\text{rank}{-}r$ 矩阵的 Hadamard 积（element-wise product）的秩可高达 $r^2$（Khatri-Rao 积性质）。即参数量不变，但表达的秩空间更大。

**HiRA**（High-rank Adaptation）：

$$\Delta W = W_0 \odot (B A)$$

用 frozen 的 $W_0$（满秩）与 LoRA 的 $BA$ 做 Hadamard 积，乘积的有效秩高于 $r$。参数量与 LoRA 相同。

**BoHA**（Block-wise Hadamard Product Adaptation）：

将 $W_0$ 沿行/列方向划分为 $b \times b$ 的 block，每个 block 独立学习 LoRA 因子 $(A_i, B_i)$，在 block 级别做 Hadamard 乘积更新：

$$(\Delta W)_{\text{block}_i} = (W_0)_{\text{block}_i} \odot (B_i A_i)$$

- 总参数量与 LoRA 等价（block LoRA factors 总和 ≈ 同 rank 的单一 LoRA）
- 分块策略使每块独立适应，粒度更细，局部结构保留更好
- Merge 后 **零额外推理开销**

**ABBA**：另一种 alternating block Hadamard 结构，原理类似。


---

### 1.11 LoRA+ — 学习率解耦 / Decoupled Learning Rate

基于 Maximal Update Parameterization (muP) 分析，标准 LoRA 中 $A$ 和 $B$ 用相同学习率是次优的。

**核心结论**：$B$ 矩阵应使用比 $A$ 更高的学习率：

$$\eta_B / \eta_A = \lambda, \quad \text{推荐 } \lambda \in [2, 16]$$

**原因**：在宽度缩放极限下，$A$ 和 $B$ 的梯度信号量级不同；统一 lr 会导致一个欠拟合、另一个过拟合。

**实践**：在 optimizer 中对 $B$ 的参数组设置更高 lr（如 $\eta_A = 10^{-4}$，$\eta_B = 2 \times 10^{-3}$）。

---

### 1.12 GaLore — Gradient Low-Rank Projection / 梯度低秩投影

GaLore 不对权重做低秩参数化，而是对 **梯度** 做低秩投影：

$$G \approx U_r \Sigma_r V_r^T \quad \text{(梯度的 top-}r\text{ SVD)}$$

optimizer state 只维护 $r$ 维投影空间，每 $T$ 步重新计算投影方向（subspace refresh）。

**与 LoRA 的根本区别**：
- **LoRA**：冻结 base model，仅训练 A/B；支持 merge 和 adapter 复用
- **GaLore**：全量参数都更新，optimizer 低维维护；推理时是标准模型，不支持 adapter 复用

适合 pre-training / continual pre-training 场景；SFT 场景 LoRA 更实用。

---

### 1.13 PiSSA — Principal SVD Initialization / 主成分初始化

标准 LoRA 从 $\Delta W = 0$ 开始学习；PiSSA 从 $W_0$ 的主成分出发。

对 $W_0$ 做 SVD：

$$W_0 = U \Sigma V^T$$

取前 $r$ 个奇异值/向量初始化：

$$A = \sqrt{\Sigma_r} \cdot V_r^T, \quad B = U_r \cdot \sqrt{\Sigma_r}$$

即 $BA = U_r \Sigma_r V_r^T \approx W_0$ 的主成分。Frozen 部分变为 residual：$W_0 - BA$。

**优点**：初始点更接近 $W_0$ 结构，收敛更快。**代价**：需对 $W_0$ 做一次性 SVD 初始化。

---

## 第二部分：PyTorch 代码片段 / Part 2: PyTorch Snippets

> 以下代码为从零实现的教学级片段，仅依赖 `torch` 和 `torch.nn`。实际项目建议使用 [PEFT](https://github.com/huggingface/peft) 或 [Lora](https://github.com/microsoft/LoRA) 库。

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

### Snippet 8 — QLoRA 概念演示（4-bit NF4 Quantization Wrapper）

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

## 第三部分：面试题 / Part 3: Interview Questions

### ━━━ L1 基础 / Basic ━━━

---

<details>
<summary>Q1：为什么需要 PEFT？Full Fine-Tuning 有哪些问题？</summary>

**答：** FFT 对全部参数做梯度更新，面临四大问题：(1) **显存成本高**（7B 模型 BF16 参数 14 GB + BF16 梯度 14 GB + FP32 Adam m/v 56 GB ≈ **84 GB+**，不含激活值；含 FP32 master copy 约 112 GB）；(2) **灾难性遗忘**，可能覆盖预训练通用能力；(3) **多任务部署代价大**，每个任务需保存一份完整模型；(4) **训练不稳定**，学习率敏感。

PEFT 冻结 base model，仅训练少量额外参数（如 0.1%–1%），在显存、存储、训练成本上大幅降低，同时接近 FFT 性能。

**追问 Follow-up：** PEFT 在推理时是否有额外延迟？哪些方法可以 merge 回 base model？

> **答：** LoRA、DoRA、VeRA、(IA)³、HiRA、BoHA 均可 merge 回 base model，推理时 **零额外延迟**。Adapter 和 Prefix Tuning 无法 merge——Adapter 推理时需额外 bottleneck 前向传播，Prefix Tuning 在 KV cache 中占位，二者均有推理开销。

</details>

---

<details>
<summary>Q2：LoRA 的数学原理是什么？rank $r$、scaling $\alpha$、target modules 分别是什么？</summary>

**答：** LoRA 假设预训练权重更新 $\Delta W$ 是低秩的：$W = W_0 + \frac{\alpha}{r} BA$。其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。$A$ 初始化为 Kaiming uniform，$B$ 初始化为零，确保训练起始 $\Delta W = 0$。

- **$r$**：控制表达力，越大 $\Delta W$ 的秩越高，参数量线性增加
- **$\alpha$**：缩放因子，实际缩放为 $\alpha/r$；通常 $\alpha = r$ 或 $2r$
- **Target modules**：通常选 attention 的 Q/K/V/O 矩阵，复杂任务可加 FFN 的 up/down/gate

**追问：** 为什么 LoRA 假设低秩更新是合理的？有没有理论支持？

> **答：** 主要来自 (1) 低内在维度（intrinsic dimensionality）理论：随机子空间探测发现很多任务在低维子空间内即可达到接近 FFT 的性能；(2) 实际 FFT 后 $\Delta W$ 的奇异值分布快速衰减，top-$r$ 分量覆盖大部分能量；(3) BA 分解隐含对 $\Delta W$ 的 nuclear norm 正则化。

</details>

---

<details>
<summary>Q3：LoRA 的 rank $r$ 和 scaling $\alpha$ 如何调？rsLoRA 解决了什么问题？</summary>

**答：**
- **$r$**：任务越难（如数学推理）通常需要更大 $r$（常用 4 / 8 / 16 / 64）
- **$\alpha$**：隐式调节 LoRA 学习率；$\alpha$ 调大 → 更新幅度更大

**rsLoRA** 解决的问题：标准 LoRA 在大 $r$ 时，$BA$ 的 Frobenius 范数随 $\sqrt{r}$ 增长，导致梯度过大、训练不稳。rsLoRA 将缩放改为 $\alpha / \sqrt{r}$，保证不同 $r$ 下更新幅度一致。

**追问：** 实践中如何确定最优 $r$？有没有自适应方法？

> **答：** 实践中从 $r=8$ 开始，逐步增大观察 loss/eval plateau；若 $r=64$ 仍无提升则问题可能不在表达力。自适应方法如 **AdaLoRA** 通过 SVD + importance scoring 动态分配各层 rank。

</details>

---

<details>
<summary>Q4：<code>(IA)³</code> 是什么？为什么参数量比 LoRA 还小？</summary>

**答：** (IA)³ 对 attention 的 key/value 和 FFN 输出分别乘以 **可学习缩放向量**（$l_k, l_v, l_{\text{ff}}$），维度分别为 $d_k, d_v, d_{\text{ff}}$。参数量约每层 $d_k + d_v + d_{\text{ff}}$（几千个），远少于 LoRA 的 $r \times (d+k)$。

适合 few-shot 场景；代价是表达力弱（仅做 rescaling，无法学方向性变化），复杂任务上欠拟合。

**追问：** (IA)³ 能否在推理时 merge 到 base model？

> **答：** 可以。$l_k \odot K = (l_k \odot W_K) x$，可直接将 $l_k$ 乘入 $W_K$ 权重矩阵，merge 后零额外延迟。

</details>

---

<details>
<summary>Q5：Prefix Tuning、Prompt Tuning、P-Tuning v2 有什么区别？</summary>

**答：**

| 方法 | 参数位置 | 每层/仅输入 | 参数量 |
|------|---------|-----------|--------|
| **Prompt Tuning** | input embedding 的 soft token | 仅输入层 | 极少 |
| **Prefix Tuning** | 每层 attention 的 K/V prefix | 每层 | 较少（需 MLP reparameterize） |
| **P-Tuning v2** | 每层 prefix | 每层 | 中等，接近 FFT 表达力 |

Prompt Tuning 参数最少但在小模型上效果差；Prefix Tuning 表达力更强但需 reparameterize 训练才稳定；P-Tuning v2 统一框架在 NLU 上接近 FFT。共同局限：推理时需在 KV cache 中占位，无法 merge。

**追问：** Prefix Tuning 为什么需要 reparameterization（MLP）才能训练？

> **答：** 直接优化 prefix 向量容易过拟合且不稳定（参数在 embedding 空间中缺乏结构约束）。通过 MLP 将低维向量映射到高维 prefix 空间，引入隐式正则化，使优化更平滑、泛化更好。

</details>

---

<details>
<summary>Q6：Adapter 方法的结构是什么？与 LoRA 相比推理时有何区别？</summary>

**答：** Adapter 在 Transformer block 的 attention/FFN 之后插入 bottleneck MLP：$h \to W_{\text{down}}(d \to r) \to \text{ReLU} \to W_{\text{up}}(r \to d) \to \text{Residual}$。参数量约 $2rd$，类似 LoRA。

**关键区别**：Adapter 推理时有额外串行前向传播（bottleneck MLP），增加 latency；LoRA merge 后零额外延迟。在推理速度敏感的生产场景中 LoRA 更优。

**追问：** 如何同时服务多个 LoRA adapter？有没有系统层面的工作？

> **答：** 代表工作有 **S-LoRA**（统一 KV cache 管理 + custom CUDA kernel 做 batched LoRA）和 **Punica**（S-LoRA 的开源实现），支持在单个 base model 上同时 serving 数千个 LoRA adapter，通过 batch 内并行不同 adapter 的 BA 计算实现高吞吐。

</details>

---

<details>
<summary>Q7：VeRA 是什么？参数效率达到了什么程度？</summary>

**答：** VeRA 让所有层共享同一套随机固定矩阵 $A$、$B$，仅训练对角缩放向量 $b$ 和 $d$：$\Delta W = \text{diag}(b) \cdot B \cdot \text{diag}(d) \cdot A$。每层仅需 $r + d$ 个可训练参数，约为 LoRA 的 $1/k$（$k$ = hidden dim）。

代价是表达力受限——共享矩阵 + 仅对角缩放在复杂任务上与 LoRA 有差距，适合参数极度受限的边缘部署场景。

**追问：** VeRA 和 (IA)³ 的参数量比较？

> **答：** 两者都是极少参数：VeRA 每层 $r + d$ 个参数，(IA)³ 每层 $d_k + d_v + d_{\text{ff}}$ 个参数，数量级相似。但 (IA)³ 仅做 activation rescaling，VeRA 通过共享矩阵做更结构化的权重更新，表达力略强。

</details>

---

<details>
<summary>Q8：LoRA 中 $A$ 初始化为 Kaiming uniform、$B$ 初始化为零的原因是什么？</summary>

**答：**
- **$B = 0$**：保证训练起始 $\Delta W = BA = 0$，LoRA 不改变 base model 输出，梯度信号稳定
- **A ~ Kaiming uniform**：非零保证梯度能流过（若 $A$ 也为零，则 $\Delta W \equiv 0$，梯度恒为零）；Kaiming 初始化保证前向传播时 activation 方差不爆炸/消失

若 $A$、$B$ 都随机初始化（$B \neq 0$），训练开始时 $\Delta W \neq 0$，会扰动 base model 初始行为，导致前期不稳定。

**追问：** 有没有更好的 LoRA 初始化方式？

> **答：** **PiSSA** 用 $W_0$ 的 SVD 前 $r$ 个分量初始化 $A$、$B$，从主成分出发优化；**LoRA-GA** 用梯度对齐方向初始化，使初始 LoRA 更新方向与 full FT 梯度一致。这些方法在收敛速度上有提升。

</details>

---

### ━━━ L2 中级 / Intermediate ━━━

---

<details>
<summary>Q9：QLoRA 是什么？如何在保持效果的同时大幅减少显存？</summary>

**答：** QLoRA 把 base model 量化为 **4-bit NF4** 格式，LoRA adapter 保持 BF16 精度训练。三项关键技术：

1. **NF4 量化**（Normal Float 4）：设计为在高斯分布上信息损失最小的 4-bit 格式
2. **Double Quantization**：量化常数（scale）本身也量化，进一步压缩
3. **Paged Optimizer**：利用统一内存，optimizer states 在 GPU/CPU 间分页，防 OOM

效果：极大降低显存需求（如 65B 模型可单卡训练），4-bit base model 的精度损失由 LoRA adapter 补偿。

**追问：** NF4 和 INT4 的区别？量化误差如何影响 LoRA 梯度？

> **答：** NF4 基于信息论最优设计，量化点按高斯分布的分位数分布；INT4 为均匀间隔。对近高斯分布的预训练权重，NF4 量化误差更小。LoRA 梯度通过量化权重的直通估计（straight-through estimator）或 full precision 的 frozen base 传播，实际训练中量化误差主要影响前向传播精度，对梯度方向的影响被 LoRA 的低秩结构缓解。

</details>

---

<details>
<summary>Q10：DoRA 是什么？它如何改进 LoRA 的表达力？</summary>

**答：** DoRA 将权重更新分解为**方向**和**幅度**：$W = m \cdot (W_0 + BA) / \|W_0 + BA\|_c$，其中 $m$ 是可学习幅度向量。分析发现 FFT 同时改变幅度和方向，而 LoRA 主要改变方向；加入可学习幅度后，DoRA 的梯度行为更接近 full FT，据原论文在多个测试任务上相同参数量下与 LoRA 相比有一定提升（competitive with or better than LoRA）。

**额外参数**：仅 $k$ 个参数（幅度向量），相对 LoRA 可忽略。

**追问：** DoRA 的额外参数量是多少？$m$ 的维度怎么决定的？

> **答：** $m \in \mathbb{R}^{k}$，其中 $k$ 是权重矩阵的输入维度（列数），即每个输入特征方向一个幅度因子。每层额外仅 $k$ 个参数（通常数千），而 LoRA 每层 $r \times (d+k)$（通常数十万），因此 DoRA 的额外开销几乎为零。

</details>

---

<details>
<summary>Q11：AdaLoRA 如何自适应分配 rank？</summary>

**答：** AdaLoRA 将 $\Delta W$ 参数化为类 SVD 形式 $P \Lambda Q$，对每个奇异值三元组计算 importance score：$\text{Importance}_i = |\lambda_i| \times |\partial \mathcal{L}/\partial \lambda_i|$。训练过程中将 importance 低的奇异值 mask 为零（prune），将预算留给重要层/矩阵。

效果：关键层（如较高层 attention）获得更多 rank，比均匀分配更优。

**追问：** AdaLoRA 的 importance score 具体如何计算？mask 操作是 hard 还是 soft？

> **答：** Importance 综合考虑奇异值大小（$|\lambda_i|$）和灵敏度（loss 对 $\lambda_i$ 的梯度范数）。Mask 在训练中是 **hard** 的（直接置零），通过在 $P$、$\Lambda$、$Q$ 上加正则化预算约束实现。训练结束后未被 mask 的奇异值保留，对应最终的 rank 分配。

</details>

---

<details>
<summary>Q12：LoRA 可以用于 RLHF 吗？有哪些注意事项？</summary>

**答：** 可以，且广泛使用。主要注意：

1. **Actor/Ref 共享 base**：reference policy 可直接 disable LoRA adapter，不需要单独保存 base model（TRL、OpenRLHF 支持）
2. **Critic 的 LoRA**：critic head 通常随机初始化，需要较高 rank 才能收敛
3. **梯度路径**：LoRA 的 A/B 梯度通过 frozen base 前向传播，PPO 梯度路径不变，但显存仍需保存 activation
4. **Merge 时机**：PPO 结束后 merge 再评估

**追问：** Reference policy 用 disable-adapter 而非存独立模型，有没有风险？

> **答：** 有一定风险。Disable adapter 后 reference policy = base model，而 PPO 训练中 actor 的 LoRA 更新后与 base model 有差异。如果 LoRA 更新幅度较大（大 $\alpha$ 或多轮迭代），KL 约束可能不足以防止 policy 漂移。实践中需监控 KL divergence，必要时降低 $\alpha$ 或 clip ratio。

</details>

---

<details>
<summary>Q13：LoRA rank 与 full fine-tuning 的差距在哪些任务上最大？</summary>

**答：** 差距最大的场景：(1) **大规模知识注入**（continual pre-training / domain adaptation），低秩 $\Delta W$ 表达力不足；(2) **复杂推理任务**（高难度数学、代码），参数更新方向复杂，低 rank 欠拟合；(3) **safety alignment**，低秩更新有时只改表面行为。

差距小的场景：instruction tuning（任务结构简单）、single-task fine-tuning（分类、NER 等内在维度低）。

**追问：** 如何实验判断一个任务是否 LoRA 表达力足够？

> **答：** 逐步增大 $r$（如 4→8→16→32→64），绘制 eval metric vs $r$ 曲线。若在较大 $r$ 处出现 plateau（性能不再提升），说明 LoRA 表达力已足够；若持续提升且仍未达到 FFT 水平，说明低秩假设不足，需考虑 high-rank 方法或 FFT。

</details>

---

<details>
<summary>Q14：多个 LoRA adapter 如何合并（merge）？有哪些方法？</summary>

**答：**
1. **Linear merge**：$W = W_0 + \sum_i \alpha_i B_i A_i$，直接线性叠加
2. **TIES**：对多个 delta 的符号投票（trim + elect sign + disjoint merge），减少冲突
3. **DARE**：随机 prune delta 的部分参数（概率 $p$ 置零后 rescale），降低干扰
4. **LoRAHub**：用少量 examples 学习 per-adapter 权重，自动发现最优 composition
5. **SLERP**：球面线性插值，保留两个 checkpoint 各自特性

**追问：** Merge 后的模型和 multi-task joint 训练相比哪个更好？

> **答：** 取决于任务间的关系。任务相关性高时 merge 效果接近 joint 训练；任务冲突大时 merge 可能退化（干扰），joint 训练通过梯度协调更优。但 merge 的优势是无需联合训练数据、支持增量组合，实践中 merge + 少量混合数据 finetuning 是常见折中。

</details>

---

<details>
<summary>Q15：如何为特定任务选择 LoRA 的 target modules？</summary>

**答：** 经验原则：
- **Attention-only**（Q/K/V/O）：默认选择，参数量少，language understanding 效果好
- **All linear**（Q/K/V/O + FFN up/down/gate）：参数量增加 2–3×，对复杂推理任务更好
- **只加 V/O**：部分工作发现 V 矩阵对 task-specific 适应最重要

数据驱动：先在所有矩阵上训练，用 importance score 找出更新幅度最大的矩阵。

**追问：** Embedding layer 和 LM head 是否应该纳入 LoRA？

> **答：** 一般不推荐——embedding 和 LM head 对 tokenization 敏感，且共享权重矩阵（weight tying），加入 LoRA 可能破坏已学好的 token 语义空间。但在需要新增 special tokens 或 domain-specific vocabulary 时，可单独对 embedding 做 FFT 而非 LoRA。

</details>

---

<details>
<summary>Q16：LoRA 中 rank collapse 是什么？如何检测和缓解？</summary>

**答：** Rank collapse 指训练中 $BA$ 的有效秩（numerical rank，用奇异值分布衡量）收敛到远小于 $r$ 的值，理论 rank $= r$，实际表达力接近 rank 1–2。

**成因**：优化不显式约束 rank，梯度倾向找"最简单"的低维解。

**检测**：定期计算 $BA$ 的有效秩 = $\exp(H(\sigma))$，其中 $H$ 是归一化奇异值的熵。

**缓解**：rsLoRA（调整缩放防止初期过大更新）、AdaLoRA（SVD 空间优化）、定期 orthogonalize A/B。

**追问：** Rank collapse 和 gradient vanishing 有关系吗？

> **答：** 有间接关系。Rank collapse 时 $BA$ 方向趋同，不同奇异方向的梯度信号减弱（类似 vanishing），形成正反馈：有效秩越低→梯度越集中→有效秩更低。打破这个循环需要外部干预（如 scaling 修正、SVD 投影等）。

</details>

---

<details>
<summary>Q17：LoRA+ 的核心思路是什么？和标准 LoRA 有什么区别？</summary>

**答：** 基于 muP (Maximal Update Parameterization) 分析，标准 LoRA 中 $A$ 和 $B$ 用相同学习率是次优的。理论上 $B$ 应使用比 $A$ 更高的学习率（$\eta_B / \eta_A = \lambda$，推荐 $\lambda \in [2, 16]$）。

原因：宽度缩放极限下 $A$、$B$ 梯度信号量级不同，统一 lr 导致一个欠拟合、另一个过拟合。实践中仅需在 optimizer 中对 $B$ 设更高 lr，改动极小。

**追问：** muP 的核心思路是什么？如何用 muP 做 LoRA 超参搜索？

> **答：** muP 的核心是在 infinite-width 极限下推导出一组"最大更新"的参数化方案，使得小模型上调好的超参可直接迁移到大模型（width transfer）。对 LoRA+，这意味着在小 rank 小模型上搜索的最优 $\lambda$、lr 等可直接用于大模型，大幅减少大模型上的搜索成本。

</details>

---

### ━━━ L3 深度 / Deep ━━━

---

<details>
<summary>Q18：LoRA 的低秩假设有理论支撑吗？有哪些局限？</summary>

**答：** 部分支撑：(1) **低内在维度**（Aghajanyan et al., 2020）：随机子空间探测发现任务相关参数变化有低维结构；(2) **Spectral analysis**：FFT 后 $\Delta W$ 奇异值快速衰减；(3) **隐式正则化**：BA 结构隐含 nuclear norm 正则化。

**局限**：并非所有任务适合低秩更新——大规模知识注入、复杂推理等任务低秩 $\Delta W$ 表达力不足，这也是 HiRA、BoHA 等 high-rank 方法的动机。

**追问：** Intrinsic dimensionality 怎么测量的？对 RLHF 阶段的 LoRA 还成立吗？

> **答：** 测量方法是在随机子空间中训练：固定 $W_0$ 冻结，在低维随机投影 $W_0 + P_d \theta$（$P_d \in \mathbb{R}^{|W| \times d}$ 随机矩阵，$\theta \in \mathbb{R}^d$）中优化，逐步增大 $d$ 观察性能。RLHF 阶段的低秩假设可能更弱——reward signal 引导的行为修改更复杂，实践中 RLHF 的 LoRA 通常需要更大 rank 或更仔细的超参调优。

</details>

---

<details>
<summary>Q19：HiRA 和 ABBA 是什么？Hadamard 乘积如何提升有效秩？</summary>

**答：** 数学基础：两个 rank-$r$ 矩阵的 Hadamard 积的秩可高达 $r^2$（Khatri-Rao 积性质）。

**HiRA**：$\Delta W = W_0 \odot (BA)$，用满秩 $W_0$ 与低秩 $BA$ 做 Hadamard 积。$W_0$ 冻结，只有 $BA$ 可训练，参数量与 LoRA 相同，但有效秩远高于 $r$。

**ABBA**：alternating block Hadamard 结构，原理类似，形式略有不同。

据各自原论文，两者在同等参数量下、需要高表达力的测试任务上优于或 competitive with 标准 LoRA；在不同任务/规模上收益可能不同。

**追问：** Khatri-Rao 积的高秩性质的数学原理？HiRA 和 DoRA 能否结合？

> **答：** Khatri-Rao 积 $A \odot B$ 的列是对应列的 Kronecker 积；两个 rank-$r$ 矩阵的 Kronecker 积的秩为 $r^2$，Khatri-Rao 作为列对齐的 Kronecker 积保留部分高秩性质。HiRA + DoRA 可以结合：在 HiRA 的 Hadamard 更新上加幅度-方向分解，同时获得 high-rank 和 magnitude learning 的优势，但需实验验证是否有额外收益。

</details>

---

<details>
<summary>Q20：BoHA 的 block-wise Hadamard 策略和 HiRA 有什么区别？</summary>

**答：** BoHA 将 $W_0$ 划分为 $b \times b$ 的 block，每个 block 独立学习 LoRA 因子 $(A_i, B_i)$ 并做 Hadamard 乘积：

$$(\Delta W)_{\text{block}_i} = (W_0)_{\text{block}_i} \odot (B_i A_i)$$

与 HiRA 的关键区别：
- HiRA 用整个 $W_0$ 做 Hadamard，所有 block 共享同一 LoRA 因子
- BoHA 分块独立——更细粒度的适应，可捕捉不同 attention head / feature group 的差异
- 总参数量与 LoRA 等价，merge 后零额外推理开销


**追问：** 分块策略（block size $b$）如何影响性能？$b$ 增大时表达力如何变化？

> **答：** $b$ 越大，每个 block 内的 LoRA 因子越大（参数越多），单个 block 的表达力越强；但 block 数量减少，全局跨 block 的协调能力变弱。$b$ 过小则每个 block 参数极少，表达力不足。理论上存在最优 $b$ 与 $W_0$ 的内在结构（如 attention head 维度）对齐。实践中 $b$ 通常设为 head dimension 的倍数。

</details>

---

<details>
<summary>Q21：PiSSA 是什么？和 LoRA 初始化有什么不同？</summary>

**答：** PiSSA 对 $W_0$ 做 SVD，取前 $r$ 个分量初始化 $A$、$B$（$BA \approx U_r \Sigma_r V_r^T \approx W_0$ 的主成分），frozen 部分变为 residual $W_0 - BA$。

与 LoRA 的区别：LoRA 从零开始学习 $\Delta W$；PiSSA 从主成分出发，初始点更接近 $W_0$ 结构，收敛更快。

**追问：** PiSSA 和 LoRA 的 merge 操作是否一致？residual 的处理有何不同？

> **答：** Merge 操作类似：$W_{\text{merged}} = (W_0 - U_r\Sigma_r V_r^T) + B_{\text{trained}} A_{\text{trained}}$。不同在于 frozen 部分是 residual 而非原始 $W_0$，这意味着 merge 后的模型结构与 LoRA merge 的 $W_0 + BA$ 在数学上不同，但推理时行为等价（合并后都是 $d \times k$ 权重矩阵）。

</details>

---

<details>
<summary>Q22：GaLore 是什么？它和 LoRA 的思路有什么根本不同？</summary>

**答：** GaLore 对 **梯度** 做低秩投影（非权重）：梯度 $G \approx U_r \Sigma_r V_r^T$，optimizer state 只维护 $r$ 维投影，每 $T$ 步刷新投影方向。全量参数都更新，只是 optimizer 在低维空间中维护。

**根本区别**：
- **LoRA**：固定参数结构，冻结 base model；支持 merge 和 adapter 复用
- **GaLore**：全量参数更新，推理时是标准模型；不支持 adapter 复用

GaLore 适合 pre-training / continual pre-training（需 full update 但显存受限）；SFT 场景 LoRA 更实用。

**追问：** GaLore 的 subspace refresh 频率如何选？refresh 时 optimizer state 如何处理？

> **答：** Refresh 频率 $T$ 通常设为几百到几千步（论文推荐 200）。Refresh 时丢弃旧的 optimizer state（momentum、variance），在新投影子空间中重新初始化——这引入了周期性的优化不连续性，但实践中对收敛影响有限。$T$ 过小会频繁丢弃状态、降低效率；$T$ 过大会使梯度投影偏离最优子空间。

</details>

---

<details>
<summary>Q23：PEFT 方法在 continual learning（CL）场景下有什么优势？</summary>

**答：** PEFT 的 CL 优势：(1) base model frozen，每个任务训练独立 adapter → 参数空间天然隔离；(2) 存储多任务仅需保存多套轻量 adapter；(3) 支持动态加载/切换。

W₀-coupled 的 Hadamard 结构（如 HiRA、BoHA）在 CL 下有额外优势——$W_0$ 作为隐式"锚点"，约束更新不偏离原始参数空间过远，理论上保留更多之前任务的知识。


**追问：** BoHA 在 CL 的优势是架构固有的（W₀-coupled）还是参数量更多导致的？如何验证？

> **答：** 应通过消融实验验证：(1) 对比 BoHA 与相同参数量的 LoRA 在 CL 上的 retention，排除参数量因素；(2) 对比 BoHA 与 non-coupled block LoRA（block 独立 LoRA 但无 Hadamard 与 $W_0$ 耦合），验证 $W_0$-coupling 的贡献；(3) 逐步增大任务数观察灾难性遗忘曲线。

</details>

---

<details>
<summary>Q24：如何评估 PEFT 方法的优劣？有哪些公平对比的注意事项？</summary>

**答：** 评估维度：
1. **参数量对齐**：必须在相同可训练参数量下对比
2. **任务多样性**：覆盖 NLU（GLUE）、推理（GSM8K/MATH）、生成（MT-Bench）
3. **模型规模**：不同 scale 下结论可能反转
4. **训练超参**：per-method tuning 而非统一超参
5. **推理成本**：能否 merge？merge 后延迟是否一致？

公平对比的坑：LoRA vs Adapter 需考虑推理延迟差异；Hadamard 方法间对比需严格对齐 total rank 和参数量。

**追问：** PEFT 的比较基准应该是 full FT 还是 LoRA baseline？为什么两者都要有？

> **答：** 两者都需要：full FT 是**天花板**（该任务是否需要更多参数），LoRA 是**实用 baseline**（新增方法是否在相同成本下更好）。仅比 FFT 无法体现参数效率优势；仅比 LoRA 无法判断整体上限。完整的评估应呈现三个柱状图：FFT、LoRA、新方法。

</details>

---

<details>
<summary>Q25：PEFT 方法在多模态（Vision-Language）和 Diffusion model 中有哪些应用？</summary>

**答：**

**Vision-Language（VLM）**：
- LoRA 广泛用于 CLIP、LLaVA、InternVL 等 VLM 的 SFT；可同时对 visual encoder 和 LLM decoder 加 LoRA，或仅对 LLM 部分加（visual encoder 冻结）
- Prefix tuning / adapter 在早期 VLM（如 Flamingo）中用于 cross-attention adapter 注入视觉信息

**Diffusion Model**：
- **LoRA for Diffusion**（如 SDXL LoRA）：对 UNet 的 attention / conv 层加 LoRA，实现风格/人物定制，参数量仅数 MB
- **IP-Adapter**：冻结 CLIP image encoder + 加 cross-attention adapter
- **ControlNet**：拷贝整个 encoder 分支（非严格 PEFT），属于"锁定原始 + 训练 copy"思路

**追问：** 多个风格 LoRA 的 merge 在 diffusion 中有哪些工具链支持？和 LLM LoRA merge 的差异？

> **答：** Diffusion 社区有活跃的 merge 工具（如 `sd-webui` 内置 merge、`mergekit` 扩展、`ComfyUI` LoRA stacking）。与 LLM 的主要差异：(1) Diffusion LoRA 通常作用于 UNet 的多层 attention，merge 时可按层设置不同权重；(2) 多 LoRA 组合常用于混合风格（如 "anime + realistic"），社区发展出经验性的权重配方；(3) Diffusion 的 merge 更关注视觉质量（FID / 人工评估），而非 accuracy 指标。

</details>

---

## 快速参考 / Quick Reference

| 方法 | 参数量级 | Merge? | 额外推理延迟 | 适用场景 |
|------|---------|--------|-------------|---------|
| **LoRA** | 0.1–1% | ✅ | 无 | SFT 通用首选 |
| **QLoRA** | 同 LoRA（base 4-bit） | ✅ | 无（dequant 开销小） | 显存受限 |
| **DoRA** | LoRA + 极少 | ✅ | 无 | 需接近 FFT 性能 |
| **VeRA** | ≈ LoRA / $k$ | ✅ | 无 | 极端参数受限 |
| **AdaLoRA** | 自适应 | ✅ | 无 | 需自动 rank 分配 |
| **(IA)³** | 极少（向量） | ✅ | 无 | Few-shot |
| **HiRA / BoHA** | 同 LoRA | ✅ | 无 | 需 high-rank 表达力 |
| **Adapter** | ~LoRA | ❌ | 有 | 动态 swap |
| **Prefix Tuning** | 少量 | ❌ | 有（KV cache 占位） | 生成式任务 |
| **Prompt Tuning** | 极少 | ❌ | 有（同上） | 大模型 few-shot |
| **GaLore** | 全量参数 | ❌（非 adapter） | 无 | Pre-training |

---

*Built for research learning. No benchmark numbers are fabricated; specific experimental claims from individual papers are excluded or marked for verification.*


## 更多 L3 深挖 / Extended L3

<details>
<summary>Q：为什么 AdaLoRA 采用 SVD 形式（$P\Lambda Q$）而非标准 BA 参数化来实现自适应秩分配？$P\Lambda Q$ 的正交性约束在此场景下提供了什么数学优势？</summary>

   A：标准 LoRA 的 $BA$ 参数化中，不同奇异值方向之间没有正交约束，训练过程中基向量可能逐渐"塌缩"到同一子空间（mode collapse），导致有效秩远低于名义 rank，importance score 失去区分度。SVD 形式强制 $P$、$Q$ 为正交矩阵，保证奇异值方向彼此独立，使得 $\text{Importance}_i = |\lambda_i| \cdot |\partial\mathcal{L}/\partial\lambda_i|$ 的排序更有意义——删除低重要性奇异值后，剩余方向确实覆盖了不同信号子空间。此外，对角 $\Lambda$ 结构天然支持逐奇异值的 mask/prune 操作，而 $BA$ 形式下"删除 rank-$i$ 分量"需要同时修改两个矩阵的第 $i$ 列，优化路径不连续。代价是 $P$、$Q$ 的正交约束需要额外的投影操作（如 Cayley 参数化或定期 QR 分解），增加了实现复杂度。

   **追问**：如果正交约束通过近似方法（如 regularization term $\|P^T P - I\|_F^2$）而非精确投影实现，在训练后期当正交性偏离较大时，importance score 的可靠性会如何退化？对最终任务性能的理论影响边界是什么？

</details>



<details>
<summary>Q：<code>(IA)³</code> 的可学习向量在数学上等价于学习一个对角矩阵来 rescale 激活。从线性算子的秩角度来看，为什么这种"纯对角"适配的表达力存在理论天花板？它与 LoRA（低秩矩阵适配）之间是否存在严格的包含关系？</summary>

   A：(IA)³ 对 key/value/FFN 输出施加的操作可写为 $h' = D \cdot h$，其中 $D = \text{diag}(l)$ 是对角矩阵。对角矩阵是一个秩为 $d$（满秩）但自由度仅为 $d$ 的线性算子，它只能沿已有坐标轴做各向异性缩放，无法旋转特征方向或建立维度间的新关联。相比之下，LoRA 的 $\Delta W = BA$ 是一个自由度为 $r(d+k)$ 的秩-$r$ 矩阵，可以产生维度间的线性组合。两者不存在严格包含关系：(IA)³ 可以实现的某些对角缩放（如"仅放大维度 $i$ 100 倍"）需要 LoRA 用较高 rank 才能近似（因为 $BA$ 的对角线元素之间存在耦合），反之 LoRA 的非对角线映射是 (IA)³ 严格不可表达的。在信息论意义上，(IA)³ 适配后的每层变换仍是原权重 $W_0$ 的坐标缩放版本，而 LoRA 可以将 $W_0$ 的零空间映射到非零输出——这是本质的能力差距。

   **追问**：如果将 (IA)³ 的对角向量扩展为 block-diagonal 矩阵（每块 $b \times b$），参数量线性增长为 $b \cdot (d/b)$，表达力如何随 block size $b$ 变化？在什么 $b$ 值下可以达到与 rank-$r$ LoRA 相当的有效秩？

</details>



<details>
<summary>Q：DoRA 将权重更新分解为幅度和方向两部分，并引入列范数归一化。从 Riemannian 优化或约束优化的视角来看，这种分解如何改变了 LoRA 参数空间的优化 landscape？为什么它能更接近 FFT 的梯度行为？</summary>

   A：标准 LoRA 的参数空间是无约束的 $(A, B) \in \mathbb{R}^{r \times k} \times \mathbb{R}^{d \times r}$，$\Delta W = BA$ 对幅度和方向耦合：增大 $A$、$B$ 同时改变范数和方向。DoRA 的列归一化 $\hat{W}_c = (W_0 + BA)_c / \|(W_0 + BA)_c\|$ 将方向约束到列单位球面 $\mathbb{S}^{d-1}$ 上（一个 Riemannian 流形），幅度则由独立向量 $m$ 在欧氏空间中控制。这形成了一个"球面 × 欧氏空间"的乘积流形，幅度和方向的梯度自然解耦。FFT 中梯度同时更新 $W$ 的幅度和方向两个自由度；LoRA 只更新方向时，幅度被 $W_0$ 的范数"锚定"，搜索受限。DoRA 恢复了幅度的独立控制通道，使参数更新的几何结构更接近全参数空间。关键的额外约束是列归一化带来的：它隐式地施加了一种正则化，防止任何单列权重过大，这与 FFT 中隐式出现的幅度自适应行为一致。

   **追问**：DoRA 的列归一化在计算图中引入了除法操作，在混合精度训练（BF16 forward, FP32 grad accumulation）中，当某些列范数接近零时，数值稳定性是否构成隐患？是否有替代归一化方案可以在保持同等优化效果的同时改善数值行为？

</details>



<details>
<summary>Q：多任务 LoRA 合并（如 TIES-Merging、DARE）的核心挑战是适配器之间的参数干扰。从子空间几何角度，两个任务的 LoRA 子空间 $\mathcal{S}_1 = \text{col}(B_1)$ 和 $\mathcal{S}_2 = \text{col}(B_2)$ 的夹角如何影响线性合并后的性能？当夹角极小（近乎重叠）或接近正交时，分别出现什么现象？</summary>

   A：两个任务 LoRA 的列空间夹角可通过主夹角（principal angles）衡量。当 $\mathcal{S}_1$ 与 $\mathcal{S}_2$ 近乎重叠（主夹角 $\approx 0$）时，两个任务在相同方向上施加不同符号的更新，线性合并导致互相抵消——这正是 TIES 中 sign disagreement 检测和 trim 操作要解决的情况。当 $\mathcal{S}_1 \perp \mathcal{S}_2$（主夹角 $\approx \pi/2$）时，合并几乎无干扰，因为信号落在正交子空间中。实际中大多处于中间状态：部分方向重叠、部分方向独立。DARE 通过随机 drop + rescale 低幅度参数来减小重叠方向的干扰，其隐式假设是：被频繁 drop 的参数更可能是任务共享的噪声方向，而存活下来的是任务特异的高信号方向。更深层的理论问题是：PEFT 参数空间是否具有"任务可加性"——即不同任务的适配是否近似落在不相交的低维子空间中？如果这个假设成立（intrinsic task dimensionality 相互正交），则简单平均就足够好；如果子空间大面积重叠，则需要更激进的去干扰策略。

   **追问**：能否在合并前通过计算 $\text{tr}(B_1^T B_2 B_2^T B_1)$ 或类似度量来预测合并后的性能下降幅度？如果有文献或实验表明这个度量与任务间正迁移/负迁移强相关，这对"选择哪些 LoRA 一起合并"的策略有何指导意义？

</details>



<details>
<summary>Q：LoRA 原始论文建议对 attention 层的 Q、V 矩阵施加适配即可获得大部分收益。从 Transformer attention 机制的计算图出发，为什么 Q 和 V 的低秩扰动比 K 和投影矩阵（$W_O$）的扰动更"高效"？这是否与 attention score 的 softmax 非线性有关？</summary>

   A：在 attention 计算 $\text{softmax}(QK^T/\sqrt{d_k})V$ 中，Q 和 K 共同决定 attention pattern（权重分配），而 V 提供被加权的内容。对 Q 施加低秩扰动 $\Delta Q$ 改变的是 query 在 key 空间中的投影角度，这直接影响"关注哪些 token"——一个对下游任务高度敏感的决策。对 V 施加 $\Delta V$ 则改变被聚合的 value 向量本身，直接影响输出表示。关键在于 softmax 的非线性效应：softmax 对输入 $QK^T$ 的变化具有饱和区——当某些 attention logit 远大于其他时，softmax 输出接近 one-hot，此时对 K 的扰动被"压缩"（梯度通过 softmax 后变小）。换言之，K 的扰动经过 softmax 非线性后影响力被衰减，需要更大的幅度才能产生等效输出变化，而低秩约束恰好限制了幅度。因此，在相同 rank 预算下，扰动 Q（在 softmax 之前直接控制注意力方向）比扰动 K（经过 softmax 衰减）效率更高。$W_O$ 投影矩阵将多头输出映射回隐藏维度，其功能更接近"通用线性变换"，base model 已学得较好的初始化，低秩空间中留给任务特异调整的"余量"相对较小。

   **追问**：对于采用 GQA（Grouped Query Attention）的模型，K/V head 数少于 Q head 数且多个 Q head 共享一组 K/V。此时对共享的 K/V 施加 LoRA 的影响是否会被放大（因为一个 KV head 的扰动影响多个 Q head），从而改变 Q/V 优先的策略？

</details>



<details>
<summary>Q：Soft Prompt（Prompt Tuning / Prefix Tuning）通过在输入序列前拼接可学习 token 来影响模型行为，而 LoRA 通过修改权重矩阵。从函数类（function class）的角度分析，这两类方法分别限制模型在什么子空间中搜索最优解？是否存在某些任务族，理论上 Soft Prompt 可以表达但 LoRA 不能，或反之？</summary>

   A：LoRA 适配后的模型变换为 $h = (W_0 + BA)x$，它改变了模型对 **所有输入** 的映射函数——这是一个全局性的仿射扰动。Soft Prompt 的影响是通过拼接 prefix token $p_1, \ldots, p_k$ 到输入序列，利用 attention 机制让这些 token 作为额外的 context 来调制后续 token 的表示——它本质上不改变模型的 **参数化函数**，而是改变了函数的 **输入分布**。这意味着：(1) Soft Prompt 对所有 token 的影响是 **位置衰减** 的——离 prefix 越远的 token 受到的影响越弱（因为 attention 权重随距离衰减），而 LoRA 对序列中每个 token 的影响是等权的。(2) 从 function class 角度，LoRA 可以实现对输入空间任意方向的线性旋转（受限于 rank），而 Soft Prompt 只能通过 attention 的 QKV 机制 **间接** 影响表示——它受限于模型自身 attention pattern 的"传输带宽"。理论上，对于需要 **全局一致地修改某种推理规则** 的任务（如改变语言的语法偏好），LoRA 更合适；对于需要 **根据特定 context 动态调节行为** 的任务（如 few-shot 示例引导），Soft Prompt 更自然，因为它直接提供了 context。不过在极限情况下，足够多的 prefix token（占据整个上下文窗口）可以模拟任意行为改变，此时 Soft Prompt 的表达力可逼近 LoRA。

   **追问**：Prefix Tuning 在每层 attention 的 K/V 前拼接 prefix，而 Prompt Tuning 仅在输入层添加。从信息流的角度，每层 prefix 相当于在每个 attention 计算中注入了额外的"虚拟 context"。这种逐层注入相比仅输入层注入，是否等价于增加了模型的 **有效深度**（effective depth）？能否用递归信息传递的框架来量化这种增益？

</details>



<details>
<summary>Q：LoRA 的 rank 选择通常是一个全局超参数，但 Transformer 不同层的权重矩阵在预训练中学习到的结构差异很大。从权重矩阵的奇异值谱（singular value spectrum）角度，能否基于 $W_0$ 的谱衰减特性来推断每层的"内在适配维度"（intrinsic adaptation dimensionality），从而指导逐层 rank 分配？这与 AdaLoRA 的在线分配相比有何优劣？</summary>

   A：对预训练权重 $W_0$ 做 SVD 分析，其奇异值谱 $σ_1 \geq \sigma_2 \geq \cdots$ 的衰减速度反映了矩阵的"有效秩"——快速衰减的层其权重集中在低维子空间中，理论上其更新 $\Delta W$ 也更可能低秩（因为可调整的方向有限）；衰减缓慢的层（奇异值较均匀）意味着权重分布更"弥散"，可能需要更高 rank 来覆盖适配方向。一个合理的启发式是：奇异值谱的"拐点"（elbow point）或使 $\sum_{i=1}^{r}\sigma_i / \sum_i \sigma_i > \theta$（如 $\theta = 0.9$）的最小 $r$ 可作为该层的内在适配维度参考。然而这个静态分析的局限在于：$W_0$ 的谱反映的是 **预训练任务** 的结构，而非 **下游任务** 的适配需求——某些谱尾方向对预训练不重要但对下游任务关键。AdaLoRA 的在线分配优势在于：它基于下游任务的梯度信号（$\partial\mathcal{L}/\partial\lambda_i$）动态调整，捕获的正是"下游任务需要哪些方向"，而非"预训练中哪些方向重要"。两种方法可以互补：用谱分析做先验（warm start），用在线分配做修正。

   **追问**：如果对多个不同下游任务分别计算 AdaLoRA 最终分配的 rank 分布，这些分布之间是否存在某种"跨任务一致性"（即不同任务都认为某层某矩阵需要高 rank）？如果存在这种一致性，是否意味着可以预计算一个"通用 rank 分配表"，在新任务上直接使用而无需在线调整的开销？

</details>



<details>
<summary>Q：GaLore 通过对梯度做周期性低秩 SVD 投影来压缩优化器状态，其核心设计选择是 subspace refresh 间隔 $T$。从在线子空间追踪（online subspace tracking）的角度，$T$ 过大或过小分别会导致什么失效模式？能否用梯度子空间的"漂移速率"（drift rate）来形式化最优 $T$ 的选取？</summary>

   A：$T$ 过小（频繁 refresh）的问题：每次 SVD 的计算开销不可忽略（对全量梯度做 SVD 是 $O(\min(mn^2, m^2n))$），频繁执行会显著拖慢训练；更重要的是，过短的子空间生命期意味着 optimizer state（如 Adam 的 m, v）在投影空间中来不及积累足够的统计信息，动量和自适应学习率的估计不准确。$T$ 过大的问题：随着训练进行，损失函数的曲率结构（curvature landscape）在变化，梯度的主子空间也随之漂移。如果 $T$ 太大，当前使用的投影基 $U_r$ 已不再是最优低秩近似——梯度的真实主方向可能已"转出"当前子空间，导致投影后的梯度信号丢失关键分量，训练陷入次优点。形式化地，可定义梯度子空间在两步之间的"漂移"为 $\delta(t) = \|P_{\mathcal{S}_{t+T}} - P_{\mathcal{S}_t}\|$（投影算子的差范数），最优 $T$ 应使 $\mathbb{E}[\delta(T)]$ 不超过某个阈值，即子空间在 $T$ 步内的累计漂移不超过投影维度所允许的误差容忍度。实践中，训练早期曲率变化快（$\delta$ 大），需要较小 $T$；后期趋近收敛（$\delta$ 小），可用较大 $T$——因此自适应 refresh 间隔（基于梯度投影残差监控）比固定 $T$ 更优。

   **追问**：GaLore 采用 top-$r$ SVD 做投影，而梯度矩阵的 top 奇异向量可能主要编码的是高频变化的 batch noise 而非稳定的优化方向。是否有可能用随机化 SVD 或增量 PCA 替代精确 SVD，在降噪的同时降低计算开销？这种近似对 optimizer state 的统计估计误差会如何传播到最终模型质量？

</details>