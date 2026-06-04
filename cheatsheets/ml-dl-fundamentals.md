# ML / DL Fundamentals 公开速查手册
# ML / DL Fundamentals — Public Cheat Sheet

> 覆盖优化器、学习率调度、正则化、归一化、初始化、反向传播、梯度问题、损失函数、混合精度等核心基础。
> Covers optimizers, LR scheduling, regularization, normalization, initialization, backpropagation, gradient issues, loss functions, and mixed precision.

---

## 第一部分：概念与公式推导
## Part 1: Concepts & Formula Derivations

---

### 1.1 优化器 Optimizers

#### SGD（随机梯度下降 Stochastic Gradient Descent）

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)$$

- 无自适应学习率，收敛慢但泛化有时更好。
- No per-parameter adaptive LR; converges slowly but can generalize better.

#### SGD + Momentum

$$v_t = \mu \, v_{t-1} + \nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \, v_t$$

- 动量 $\mu$（通常 0.9）累积历史梯度方向，加速穿越平坦区域、抑制震荡。
- Momentum $\mu$ (typically 0.9) accumulates past gradients to accelerate through flat regions and dampen oscillations.

#### Adam（Adaptive Moment Estimation）

维护一阶矩（梯度均值）和二阶矩（梯度平方的指数移动平均）：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

偏差修正（Bias Correction），解决初始化为零导致的偏置问题：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

参数更新：

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- 自适应 per-parameter 学习率，对稀疏梯度友好。
- Adaptive per-parameter learning rate; friendly to sparse gradients.

#### AdamW（Decoupled Weight Decay）

标准 Adam 中，若将 L2 正则项 $\frac{\lambda}{2}\|\theta\|^2$ 加入 loss，正则化梯度 $\lambda\theta$ 被 $\sqrt{\hat{v}_t}$ 缩放，实际衰减幅度随梯度大小变化——不是真正的均匀 weight decay。

AdamW 将 weight decay 从自适应更新中**解耦**：

$$\theta_{t+1} = (1 - \eta\lambda)\,\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- 衰减力度均匀，与 SGD 下的 weight decay 行为一致。
- Uniform decay strength, consistent with weight decay under SGD.

#### 关键公式汇总 Key Formula Summary

| 优化器 Optimizer | 更新规则 Update Rule |
|---|---|
| SGD | $\theta \leftarrow \theta - \eta \nabla L$ |
| SGD + Momentum | $\theta \leftarrow \theta - \eta\,v_t,\; v_t = \mu v_{t-1} + \nabla L$ |
| Adam | $\theta \leftarrow \theta - \eta \,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ |
| AdamW | $\theta \leftarrow (1-\eta\lambda)\theta - \eta\,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ |

---

### 1.2 学习率调度 Learning Rate Scheduling

#### Warmup

训练初期 $m_t, v_t$ 估计不准（严重偏小），直接用大学习率可能导致更新步长异常。Warmup 在前 $T_w$ 步线性增大 LR，让矩估计充分积累。

#### Cosine Decay（余弦退火）

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

- 尾端 LR 平滑趋近 $\eta_{\min}$，收敛更稳定。
- LR smoothly decays to $\eta_{\min}$ at the tail for stable convergence.

#### 线性衰减 Linear Decay

$$\eta_t = \eta_{\max} - (\eta_{\max} - \eta_{\min}) \cdot \frac{t}{T}$$

- 匀速下降，尾端 LR 仍较大，可能在收敛区震荡。
- Uniform decrease; LR remains relatively large near the end, potentially causing oscillation.

---

### 1.3 正则化 Regularization

#### Dropout

训练时以概率 $p$ 将神经元输出置零。**Inverted Dropout**（PyTorch 默认）在训练时将非零输出缩放 $1/(1-p)$，推理时无需额外操作。

#### Weight Decay vs L2 正则化

- **SGD 下等价**：L2 正则 $\frac{\lambda}{2}\|\theta\|^2$ 的梯度 $\lambda\theta$ 加入总梯度，效果为 $\theta \leftarrow \theta(1 - \eta\lambda) - \eta\nabla L$，即 weight decay。
- **Adam 下不等价**：L2 正则的梯度 $\lambda\theta$ 被 $\sqrt{\hat{v}_t}$ 除，decay 幅度与梯度量级耦合。AdamW 直接做 $\theta \leftarrow (1-\eta\lambda)\theta$，实现真正的解耦 weight decay。

#### Label Smoothing

将 one-hot 标签 $y$ 替换为：

$$y_{\text{smooth}} = (1 - \epsilon)\,y + \frac{\epsilon}{K}$$

其中 $K$ 为类别数。防止模型对 logits 过度自信，提升校准性（calibration）。

---

### 1.4 归一化 Normalization

假设输入张量形状为 $(B, C)$（简化为二维，NLP 场景）或 $(B, C, H, W)$（CV 场景）。

| 方法 Method | 归一化维度 Norm Dimension | 公式 Formula |
|---|---|---|
| **BatchNorm** | 跨 batch 维度：对每个 channel 在 $(B, H, W)$ 上计算均值/方差 | $\hat{x}_c = \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$，学习 $\gamma, \beta$ |
| **LayerNorm** | 跨 feature 维度：对每个样本在 $C$（或 $C, H, W$）上计算均值/方差 | $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$，学习 $\gamma, \beta$ |
| **RMSNorm** | 仅做 RMS 缩放，不做均值中心化 | $\hat{x} = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$ |

- **RMSNorm** 省去均值计算，速度更快，LLM 主流。
- **Pre-LN**（LN 在 sublayer 输入前）：residual 路径梯度直通，训练更稳定。
- **Post-LN**（LN 在 residual 之后）：最终性能可能略高，但训练不稳定，需精细调参。

---

### 1.5 参数初始化 Weight Initialization

#### Xavier（Glorot）初始化

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

或高斯版：$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$

- 设计目标：保持前向传播中每层输出方差恒定（假设线性或 tanh 激活）。
- Goal: maintain constant variance of activations across layers (for linear/tanh).

#### Kaiming（He）初始化

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

- 针对 ReLU：负半轴置零使方差减半，因此需要额外因子 2 修正。
- For ReLU: negative-half zeroing halves variance; the factor of 2 compensates.

---

### 1.6 反向传播 Backpropagation

#### 链式法则 Chain Rule

对于计算图 $L = f(g(x))$：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial x}$$

对多层网络中的第 $l$ 层：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial a_l} \cdot \frac{\partial a_l}{\partial W_l}$$

其中 $a_l$ 为该层输出。前向传播时需**缓存中间激活值**以供反向传播使用。

#### 梯度消失与爆炸 Vanishing / Exploding Gradients

梯度在深层连乘：

$$\frac{\partial L}{\partial W_1} = \prod_{l=2}^{L} \frac{\partial a_l}{\partial a_{l-1}} \cdot \frac{\partial a_L}{\partial W_1}$$

- 若每项 $< 1$：梯度消失 → 浅层不更新。
- 若每项 $> 1$：梯度爆炸 → 参数剧烈震荡。

#### 梯度裁剪 Gradient Clipping

- **按值裁剪 Clip by value**：$g_i \leftarrow \text{clip}(g_i, -c, c)$，可能改变梯度方向。
- **按范数裁剪 Clip by norm**（主流）：$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1,\, \frac{c}{\|\mathbf{g}\|_2}\right)$，保持方向不变。

---

### 1.7 损失函数 Loss Functions

#### Cross-Entropy Loss（交叉熵损失）

对于 one-hot 标签 $y$ 和 softmax 输出 $q$：

$$\text{CE} = -\sum_{i=1}^{K} y_i \log q_i = -\log q_y$$

PyTorch 的 `CrossEntropyLoss` 内部先做 LogSoftmax，再做 NLLLoss：

$$\text{CE}(z, y) = -z_y + \log\!\left(\sum_{j=1}^{K} e^{z_j}\right)$$

#### Focal Loss

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

降低易分样本权重，聚焦难例，适用于类别不平衡场景。

#### Perplexity（困惑度）

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)$$

- 即 per-token cross-entropy 的指数化，越低越好。
- Exponentiated per-token cross-entropy; lower is better.

#### 知识蒸馏 Knowledge Distillation

$$L = \alpha \cdot \text{CE}(y, \hat{y}_s) + (1 - \alpha) \cdot T^2 \cdot \text{KL}\!\left(\hat{y}_t^{(T)} \,\|\, \hat{y}_s^{(T)}\right)$$

- Teacher 的 soft label（温度 $T$ 平滑后的分布）提供 "dark knowledge"。
- Teacher's soft labels (smoothed by temperature $T$) provide "dark knowledge."

---

### 1.8 混合精度训练 Mixed Precision Training

| 数据类型 Dtype | 指数位 Exponent | 尾数位 Mantissa | 动态范围 Dynamic Range |
|---|---|---|---|
| FP32 | 8 | 23 | $\sim 10^{\pm 38}$ |
| FP16 | 5 | 10 | $\sim 10^{\pm 4.8}$ |
| BF16 | 8 | 7 | $\sim 10^{\pm 38}$ |

**核心流程 Core Pipeline：**

1. **FP32 Master Weights**：维护一份 FP32 精度的参数副本用于梯度更新。
2. **前向 + 反向**：用 FP16 或 BF16 执行，减少显存、加速计算（Tensor Core）。
3. **Loss Scaling**（仅 FP16）：放大 loss 防止梯度下溢，更新时再缩小。
4. **BF16 优势**：动态范围与 FP32 相同，无需 loss scaling。

---

### 1.9 偏差-方差分解 Bias-Variance Decomposition

$$\mathbb{E}[\text{MSE}] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **高偏差 High Bias**（欠拟合 Underfitting）：train / val loss 都高。
- **高方差 High Variance**（过拟合 Overfitting）：train loss 低，val loss 明显高。

---

### 1.10 Batch Size 与梯度累积 Gradient Accumulation

**Linear Scaling Rule**：batch size 乘以 $k$ 时，LR 也乘以 $k$，保持每次参数更新的期望步长不变。

**梯度累积**：将 mini-batch 拆为 $k$ 个 micro-batch，前 $k-1$ 步累加梯度（不更新），第 $k$ 步统一更新，等效于 $k$ 倍 batch size 但显存不变。

---

## 第二部分：PyTorch 代码片段
## Part 2: PyTorch Snippets (From Scratch)

> 以下片段用于教学理解，不追求完整封装，重点展示核心计算逻辑。
> These snippets prioritize pedagogical clarity over production-ready packaging.

---

### 2.1 SGD + Momentum

```python
import torch

class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocities):
            if p.grad is None:
                continue
            v.mul_(self.momentum).add_(p.grad)
            p.data.sub_(self.lr * v)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

---

### 2.2 Adam

```python
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m.mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

---

### 2.3 AdamW（Decoupled Weight Decay）

```python
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            # Decoupled weight decay (applied directly to params)
            p.data.mul_(1 - self.lr * self.wd)
            # Adaptive update
            m.mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

---

### 2.4 Cosine Annealing LR Schedule

```python
import math

def cosine_lr(step, total_steps, lr_max, lr_min=0.0):
    """Cosine decay from lr_max to lr_min over total_steps."""
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * step / total_steps)
    )

# Usage with PyTorch scheduler (built-in):
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=total_steps, eta_min=lr_min
# )
```

---

### 2.5 Layer Normalization from Scratch

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
```

---

### 2.6 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
```

---

### 2.7 Gradient Clipping by Norm

```python
def clip_grad_norm(parameters, max_norm=1.0):
    """Clip gradient norm in-place; returns total norm before clipping."""
    params = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(
        sum(p.grad.data.norm(2).item() ** 2 for p in params)
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.data.mul_(clip_coef)
    return total_norm

# PyTorch built-in (recommended):
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 2.8 Mixed Precision Training (AMP)

```python
import torch

# Setup
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")  # for FP16; not needed for BF16

for inputs, targets in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16/BF16
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Backward with loss scaling (FP16 only)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)           # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

# For BF16: simply replace dtype=torch.bfloat16 and remove GradScaler
```

---

### 2.9 Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        # Use checkpoint to save memory at the cost of recomputation
        x = x + checkpoint(self.attn, x, use_reentrant=False)
        x = x + checkpoint(self.ffn, x, use_reentrant=False)
        return x
```

---

### 2.10 Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)           # (B,)
        p_t = torch.exp(-ce_loss)                     # p of correct class
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()
```

---

### 2.11 GELU 与 SwiGLU

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# GELU (exact and approximate)
class GELUApprox(nn.Module):
    """Common tanh approximation of GELU."""
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))
            )
        )

# SwiGLU FFN (as used in LLaMA)
class SwiGLU_FFN(nn.Module):
    """SwiGLU FFN: gate and up proj, then elementwise multiply, then down proj.
    d_ffn is typically (8/3) * d_model to match parameter count of standard FFN.
    """
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up   = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

---

### 2.12 Knowledge Distillation Loss

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, targets,
                      temperature=4.0, alpha=0.7):
    """
    student_logits, teacher_logits: (B, K) raw logits
    targets: (B,) integer labels
    """
    # Hard loss
    hard_loss = F.cross_entropy(student_logits, targets)

    # Soft loss (KL divergence)
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
    soft_loss = soft_loss * (temperature ** 2)  # scale back gradients

    return alpha * hard_loss + (1 - alpha) * soft_loss
```

---

### 2.13 LR Finder（学习率搜索器）

```python
import torch
import math

def lr_finder(model, dataloader, criterion, optimizer,
              init_lr=1e-7, final_lr=10.0, num_steps=100):
    """Exponentially increase LR from init_lr to final_lr over num_steps."""
    lr_mult = (final_lr / init_lr) ** (1 / num_steps)
    lr = init_lr
    results = []

    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_steps:
            break
        # Set LR
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        results.append((lr, loss.item()))
        lr *= lr_mult

    return results  # Plot loss vs. lr; pick LR where loss drops fastest
```

---

### 2.14 Gradient Accumulation

```python
accumulation_steps = 4  # simulate 4x batch size

for i, (inputs, targets) in enumerate(dataloader):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = criterion(model(inputs), targets) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 第三部分：面试题集（25 题）
## Part 3: Interview Questions (25 Questions)

---

### L1 — 基础题 Basic

---

<details>
<summary>Q1. SGD、Adam、AdamW 的核心区别是什么？ / What are the key differences among SGD, Adam, and AdamW?</summary>

**答：**
- **SGD**：固定学习率，无自适应机制，收敛慢但泛化有时更好。
- **Adam**：维护一阶矩 $m_t$（梯度均值）和二阶矩 $v_t$（梯度均值平方），经偏差修正后自适应调整每个参数的学习率。但其 weight decay 实现将正则化梯度混入二阶矩分母，衰减力度随梯度大小变化。
- **AdamW**：将 weight decay 从自适应更新中解耦，直接作用于参数：$\theta \leftarrow (1-\eta\lambda)\theta - \eta\,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$，衰减行为与 SGD 下一致。

**追问 Follow-up：** 为什么 Adam 的 L2 正则化实现"不正确"？AdamW 的解耦带来了什么本质变化？
**Why is Adam's L2 regularization implementation considered "incorrect"? What does AdamW's decoupling fundamentally change?**
> L2 正则梯度 $\lambda\theta$ 被 $\sqrt{\hat{v}_t}$ 除，导致大梯度参数衰减弱、小梯度参数衰减强，与设计意图矛盾。AdamW 使所有参数均匀衰减。

</details>

---

<details>
<summary>Q2. Adam 的偏差修正（Bias Correction）为什么是必要的？ / Why is bias correction necessary in Adam?</summary>

**答：**
$m_0 = v_0 = 0$，在训练初期几步中，指数移动平均严重偏向零。以 $m_t$ 为例：$m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$，其期望 $\mathbb{E}[m_t] = \mathbb{E}[g_t]\cdot(1-\beta_1^t)$，不除以 $(1-\beta_1^t)$ 会导致初始步长系统性偏小。修正 $\hat{m}_t = m_t/(1-\beta_1^t)$ 使早期更新幅度合理。

**追问 Follow-up：** $\beta_1 = 0.9$ 和 $\beta_2 = 0.999$ 分别对应什么直觉？
**What intuition do the default values $\beta_1 = 0.9$ and $\beta_2 = 0.999$ correspond to?**
> $\beta_1=0.9$ 对应约 10 步的历史窗口（梯度方向估计）；$\beta_2=0.999$ 对应约 1000 步（梯度幅度的慢速跟踪，提供稳定的自适应缩放）。

</details>

---

<details>
<summary>Q3. 什么时候用 SGD+Momentum，什么时候用 AdamW？ / When should you use SGD+Momentum vs AdamW?</summary>

**答：**
- **SGD+Momentum**：CV 任务（如 ResNet、ViT 等图像模型），经过精细调参后泛化往往更好，但 LR schedule 和 momentum 需仔细调整。
- **AdamW**：NLP / LLM 预训练与微调的标准选择，自适应 LR 对稀疏 token 梯度友好，调参成本低。

**追问 Follow-up：** LoRA 微调时为什么几乎都选 AdamW？
**Why is AdamW almost universally used for LoRA fine-tuning?**
> LoRA 参数量小，需快速收敛到一个好的低秩子空间；自适应 LR 降低了调参负担；LoRA 常见于 LLM 微调场景，继承了预训练时 AdamW 的优化器选择。

</details>

---

<details>
<summary>Q4. 为什么 LLM 训练需要学习率 Warmup？ / Why do LLMs need LR warmup?</summary>

**答：**
初始参数随机，梯度方差大。Adam 的二阶矩 $v_t$ 初始估计不充分，若直接用大学习率，实际步长 $\eta/\sqrt{\hat{v}_t}$ 会异常大，导致 loss spike 甚至发散。Warmup 让 $v_t$ 在低 LR 下充分积累，再逐步放大 LR，稳定早期训练。

**追问 Follow-up：** 从预训练 checkpoint 继续微调时，warmup 还必要吗？
**Is warmup still necessary when fine-tuning from a pre-trained checkpoint?**
> 通常必要但幅度可大幅减小（如仅几百步）。因为预训练权重已稳定，但新加入的 LoRA adapter 或新数据分布仍需短暂适应。

</details>

---

<details>
<summary>Q5. Cross-Entropy Loss 和 NLL Loss 的关系？为什么分类不用 MSE？ / What is the relationship between Cross-Entropy and NLL Loss? Why not use MSE for classification?</summary>

**答：**
- 交叉熵 $H(p,q) = -\sum_i p_i \log q_i$。当 $p$ 为 one-hot 时化简为 $-\log q_y$，即 NLL。PyTorch 的 <code>CrossEntropyLoss = LogSoftmax + NLLLoss</code>。
- **不用 MSE**：MSE 对 softmax 输出的梯度在饱和区极小（梯度消失），收敛慢；CE 的梯度 $\hat{q}_y - 1$ 永不消失，数值更友好。

**追问 Follow-up：** Label smoothing 如何修改 CE loss？为什么能提升校准性？
**How does label smoothing modify CE loss? Why does it improve calibration?**
> 将 one-hot 替换为 $(1-\epsilon)y + \epsilon/K$，使模型不追求极端置信度，减少过自信预测，改善概率校准（calibration）。

</details>

---

<details>
<summary>Q6. Dropout 的工作原理？推理时如何处理？ / How does Dropout work? How is it handled at inference?</summary>

**答：**
训练时以概率 $p$ 将每个神经元输出随机置零，相当于训练指数级子网络的集成。推理时不 dropout。PyTorch 默认使用 **inverted dropout**：训练时将非零输出除以 $(1-p)$ 缩放，推理时无需额外操作，期望输出一致。

**追问 Follow-up：** Transformer 中 Dropout 通常加在哪些位置？大规模预训练时还用吗？
**Where is Dropout typically applied in a Transformer? Is it still used in large-scale pre-training?**
> 通常在 attention 权重后、残差连接前、FFN 隐藏层后。大规模预训练（如 LLaMA、GPT）中通常**不用 Dropout**，因为数据量巨大不易过拟合，且 Dropout 会与分布式训练中的通信产生干扰。

</details>

---

<details>
<summary>Q7. 什么是 Perplexity？它和 Cross-Entropy 的关系？ / What is Perplexity and how does it relate to Cross-Entropy?</summary>

**答：**
$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)$$
即 per-token cross-entropy 的指数化。直觉上表示模型在每个位置平均有"多少个等概率选择"，越低说明模型越确信。

**追问 Follow-up：** 不同 tokenizer 下 PPL 可比吗？
**Is PPL comparable across different tokenizers?**
> 不直接可比。不同 tokenizer 的 token 粒度不同（如 BPE vs SentencePiece），每个 token 的信息量不同。需用 bits-per-byte（BPB）等归一化指标才能公平比较。

</details>

---

<details>
<summary>Q8. 梯度消失和梯度爆炸分别是什么？如何解决？ / What are vanishing and exploding gradients? How to address each?</summary>

**答：**
- **梯度消失**：深层网络中梯度连乘多个小于 1 的因子后趋近 0，浅层参数几乎不更新。解决：ReLU/GELU 激活函数、残差连接（shortcut）、归一化（LayerNorm/BatchNorm）、合理初始化。
- **梯度爆炸**：梯度连乘大于 1 的因子后指数增长。解决：梯度裁剪（最直接有效）、权重正则化、减小学习率。

**追问 Follow-up：** Transformer 的残差连接如何帮助梯度流？
**How do residual connections in Transformers help gradient flow?**
> 残差路径 $x + F(x)$ 的梯度为 $\partial L/\partial x = \partial L/\partial(\text{output}) \cdot (1 + \partial F/\partial x)$，其中恒等项 1 保证梯度可直达浅层，不被中间层衰减。

</details>

---

<details>
<summary>Q9. BatchNorm vs LayerNorm vs RMSNorm 的区别及适用场景？ / What are the differences and use cases of BatchNorm, LayerNorm, and RMSNorm?</summary>

**答：**

| 方法 | 归一化维度 | 适用场景 |
|---|---|---|
| BatchNorm | 跨 batch 维度（每个 channel 在 batch 上统计） | CV 任务，batch size 较大 |
| LayerNorm | 跨 feature 维度（每个样本自身统计） | NLP/Transformer，batch 小或变长序列 |
| RMSNorm | 仅做 RMS 缩放，省去均值中心化 | LLM（如 LLaMA），更快，效果相当 |

BatchNorm 在 NLP 中效果差：序列长度可变，batch 内统计不稳定；推理时需维护 running stats，不适合自回归生成。

**追问 Follow-up：** Pre-LN 和 Post-LN 有什么区别？哪个训练更稳定？
**What is the difference between Pre-LN and Post-LN? Which trains more stably?**
> Pre-LN 将 LN 放在 sublayer 输入前，residual 路径梯度直通，训练更稳定；Post-LN 在 residual 后，深层梯度可能缩小，需精细调参+warmup。现代 LLM 基本用 Pre-LN（或 Pre-RMSNorm）。

</details>

---

<details>
<summary>Q10. Xavier 和 Kaiming 初始化的区别？ / What is the difference between Xavier and Kaiming initialization?</summary>

**答：**
- **Xavier**：$W \sim \mathcal{N}(0, 2/(n_{\text{in}}+n_{\text{out}}))$，假设线性或 tanh 激活，目标是保持每层输出方差恒定。
- **Kaiming**：$W \sim \mathcal{N}(0, 2/n_{\text{in}})$，针对 ReLU——负半轴置零使方差减半，因子 2 修正了这一损失。

**追问 Follow-up：** Transformer 中 Attention 层的 projection 矩阵通常如何初始化？
**How are attention projection matrices typically initialized in Transformers?**
> 通常用 Xavier（因为输入经过 softmax 或 LayerNorm，近似线性）。不同框架的默认可能略有差异，但主流做法是 Xavier uniform 或 normal。

</details>

---

### L2 — 中等题 Intermediate

---

<details>
<summary>Q11. 反向传播算法的核心步骤是什么？ / What are the core steps of the backpropagation algorithm?</summary>

**答：**
1. **前向传播**：逐层计算输出，缓存中间激活值 $a_l$。
2. **计算 loss**：标量损失 $L$。
3. **反向传播**：从 $\partial L / \partial L = 1$ 出发，按链式法则逐层计算 $\partial L / \partial W_l$，利用缓存的激活。
4. **参数更新**：优化器根据梯度更新参数。

关键机制：前向缓存→反向复用，链式法则保证每层梯度正确计算。

**追问 Follow-up：** Gradient checkpointing 如何在内存和计算之间做权衡？
**How does gradient checkpointing trade off memory vs computation?**
> 只保存每 $k$ 层的激活，反向传播时重新前向计算中间层。显存从 $O(L)$ 降至约 $O(\sqrt{L})$，但增加约 33% 的计算量。适用于显存受限的场景。

</details>

---

<details>
<summary>Q12. 梯度裁剪的两种方式？什么时候用 Clip by Norm？ / What are two types of gradient clipping? When to use clip-by-norm?</summary>

**答：**
1. **Clip by value**：将每个梯度分量裁剪到 $[-c, c]$，简单但可能改变梯度方向。
2. **Clip by norm**：$\mathbf{g} \leftarrow \mathbf{g} \cdot \min(1, c/\|\mathbf{g}\|_2)$，保持梯度方向不变，只缩放幅度。

LLM 训练几乎都用 clip by norm（典型阈值 1.0），因为它不扭曲优化方向。

**追问 Follow-up：** Loss spike（梯度范数突然飙高）时该如何排查？
**How do you diagnose a loss spike (sudden gradient norm surge)?**
> 常见原因：数据异常（如长序列、异常 token）、学习率过大、数值溢出（FP16）、或特定 batch 的梯度与历史方向冲突。可检查数据质量、降低 LR、切换 BF16 或增加 clip 阈值监控。

</details>

---

<details>
<summary>Q13. Weight Decay 和 L2 正则化有什么关系？在 Adam 中为什么不同？ / What is the relationship between Weight Decay and L2 regularization? Why do they diverge in Adam?</summary>

**答：**
- **SGD 下等价**：在 loss 中加 $\frac{\lambda}{2}\|\theta\|^2$，其梯度 $\lambda\theta$ 加入总梯度，等效每步 $\theta \leftarrow \theta(1-\eta\lambda) - \eta\nabla L$，即 weight decay。
- **Adam 下不等价**：$\lambda\theta$ 被 $\sqrt{\hat{v}_t}$ 除，大梯度参数衰减弱，小梯度参数衰减强。AdamW 将 decay 解耦为直接 $\theta \leftarrow (1-\eta\lambda)\theta$，实现均匀衰减。

**追问 Follow-up：** 实践中 AdamW 的 weight decay 通常设多大？哪些参数通常不加 decay？
**What is a typical weight decay value for AdamW? Which parameters are typically excluded?**
> 常见值 0.01 ~ 0.1。LayerNorm 的 $\gamma, \beta$ 和 bias 通常不加 weight decay（它们已有其他正则化机制或维度很小），只对权重矩阵施加。

</details>

---

<details>
<summary>Q14. 什么是 Early Stopping？有什么缺点？ / What is Early Stopping and what are its drawbacks?</summary>

**答：**
验证集 loss 不再下降（经过 $p$ 个 epoch 无改善）时停止训练，防止过拟合。

**缺点：**
1. 需要独立验证集，减少可用训练数据。
2. 需要持续监控训练过程（工程开销）。
3. Plateau 可能误触发（需设 patience）。
4. 最终模型是某中间 checkpoint，非最优收敛点。

**追问 Follow-up：** Early Stopping 和 L2 正则化有什么理论联系？
**What is the theoretical connection between Early Stopping and L2 regularization?**
> 两者都有约束参数空间的效果。Early Stopping 限制了参数从初始化点移动的距离（隐式的范数约束），在梯度下降的线性近似下，等价于 L2 正则化中特定正则系数的解。

</details>

---

<details>
<summary>Q15. Bias-Variance Tradeoff 的含义？如何诊断？ / What is the bias-variance tradeoff and how do you diagnose it?</summary>

**答：**
$$\mathbb{E}[\text{MSE}] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$
- **高偏差（欠拟合）**：train 和 val loss 都高，模型容量不足。
- **高方差（过拟合）**：train loss 低、val loss 明显高，模型记忆训练集。

诊断工具：train/val loss 曲线、learning curve（不同训练集规模下的误差变化）。

**追问 Follow-up：** 在 LLM 时代，"过拟合"的诊断有什么新挑战？
**What new challenges arise in diagnosing "overfitting" in the LLM era?**
> 主要挑战：① 训练数据与评测数据可能重叠（benchmark 污染），导致表面 loss 低但实际泛化未改善；② LLM 的 in-context learning 能力不在 train loss 中体现；③ 评估需靠下游任务表现，而非单纯的 val loss。

</details>

---

<details>
<summary>Q16. 知识蒸馏（Knowledge Distillation）的核心思想？ / What is the core idea of Knowledge Distillation?</summary>

**答：**
用大模型（teacher）的 soft label（温度 $T$ 平滑后的概率分布）训练小模型（student），而非仅用 hard label。Loss 为：
$$L = \alpha \cdot \text{CE}(y, \hat{y}_s) + (1-\alpha) \cdot T^2 \cdot \text{KL}(\hat{y}_t^{(T)} \| \hat{y}_s^{(T)})$$
Soft label 传递了类别间的相似度关系（"dark knowledge"），帮助小模型学到更丰富的表征。

**追问 Follow-up：** Token-level KD 和 Sequence-level KD 有什么区别？
**What is the difference between token-level and sequence-level KD?**
> Token-level KD 对每个位置的 logits 做 KL 散度（逐 token 对齐）；Sequence-level KD 对完整序列的概率分布或输出序列做匹配（如用 teacher 的 beam search 输出作为伪标签）。后者更灵活但训练信号更稀疏。

</details>

---

<details>
<summary>Q17. Batch Size 对训练有什么影响？Linear Scaling Rule 是什么？ / How does batch size affect training? What is the Linear Scaling Rule?</summary>

**答：**
- 大 batch 梯度估计更准确，但减小了梯度噪声（噪声有正则化效果），可能陷入 sharp minima，泛化变差。
- **Linear Scaling Rule**：batch size 乘以 $k$ 时，LR 也乘以 $k$，保持每次参数更新的期望步长不变。超过一定规模后需更保守的缩放（如 sqrt rule）。

**追问 Follow-up：** Gradient Accumulation 如何模拟大 batch？有什么限制？
**How does gradient accumulation simulate large batches? Any limitations?**
> 将 mini-batch 拆为 $k$ 个 micro-batch，累加梯度后统一更新，等效 $k$ 倍 batch size。限制：① BatchNorm 的统计量仍是小 batch 计算的（如果用 BN 的话）；② 训练步数减少 $k$ 倍，总训练时间不变但通信频率降低。

</details>

---

<details>
<summary>Q18. GELU 激活函数是什么？为什么 Transformer 偏好它？ / What is GELU and why do Transformers prefer it?</summary>

**答：**
$$\text{GELU}(x) = x \cdot \Phi(x)$$
其中 $\Phi(x)$ 是标准正态 CDF。GELU 是平滑的近似 ReLU：在负区间有小梯度（非硬截断），对输入有概率性的门控（$x$ 越大被保留的概率越高）。对 NLP 任务比 ReLU 更友好，梯度在零点附近更平滑。

**追问 Follow-up：** SwiGLU 是什么？相比 GELU FFN 有什么结构变化？
**What is SwiGLU? How does it structurally differ from a GELU FFN?**
> $\text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV)$，使用两个线性投影做门控。标准 FFN 是 $W_{\text{up}} \to \text{GELU} \to W_{\text{down}}$（2 个矩阵），SwiGLU FFN 需 3 个矩阵（gate、up、down），通常将隐藏维度设为 $(8/3) \cdot d_{\text{model}}$ 以保持总参数量相近。

</details>

---

<details>
<summary>Q19. 混合精度训练（AMP）的原理？为什么需要 FP32 Master Weights？ / What is the principle of AMP training? Why do we need FP32 master weights?</summary>

**答：**
- **动机**：FP16/BF16 减少显存、加速计算（Tensor Core），但 FP16 动态范围小（max ~65504），梯度易下溢/上溢。
- **做法**：前向+反向用低精度；梯度更新时转回 FP32 master weights 保留精度；FP16 需 loss scaling 防止梯度下溢。
- **BF16**：动态范围与 FP32 相同（8 位指数），无需 loss scaling，是当前 LLM 训练的主流选择。

**追问 Follow-up：** FP8 训练有什么额外挑战？
**What additional challenges does FP8 training introduce?**
> FP8 的精度更低（E4M3/E5M2 两种格式），需要更精细的 scaling 策略（如 per-tensor / per-channel scaling）；梯度累积精度要求更高；模型的某些层（如 attention softmax 前的 logits）对精度更敏感，可能需要混合使用 FP8 和更高精度。

</details>

---

<details>
<summary>Q20. 什么是 Gradient Checkpointing？什么时候使用？ / What is gradient checkpointing and when should you use it?</summary>

**答：**
训练时只保存部分关键层的中间激活（如每隔 $k$ 层），反向传播时重新前向计算丢弃的激活。以约 30% 额外计算换取显存从 $O(L)$ 降至 $O(\sqrt{L})$（$L$ 为层数）。

**适用场景**：大 batch、长序列、显存受限。如果使用 LoRA 且冻结大部分参数，激活保存开销相对可控，可能不需要开启。

**追问 Follow-up：** Gradient Checkpointing 和 ZeRO-Offload 解决的是同一个问题吗？
**Do gradient checkpointing and ZeRO-Offload solve the same problem?**
> 不完全是。Checkpointing 通过重计算减少**激活显存**；ZeRO-Offload 将**优化器状态和梯度**从 GPU 卸载到 CPU/SSD。两者解决不同显存瓶颈，可以组合使用。

</details>

---

### L3 — 深度题 Deep

---

<details>
<summary>Q21. 如果训练中 loss 出现 spike（突然跳升），系统性排查步骤是什么？ / What is a systematic debugging procedure for loss spikes during training?</summary>

**答：**
1. **数据检查**：是否有异常输入（过长序列、NaN、特殊 token）。
2. **梯度监控**：记录各层梯度 norm，确认是否某层爆炸。
3. **精度检查**：FP16 是否溢出？检查 loss scaling factor 是否合理。
4. **LR / optimizer state**：是否学习率过大？Adam 的 $v_t$ 估计是否因 spike 被污染。
5. **恢复策略**：是否可用 spike 前的 checkpoint 重启（跳过问题 batch）。
6. **长期缓解**：增加梯度裁剪频率（per-step vs per-accumulation）、使用 BF16 替代 FP16。

**追问 Follow-up：** Adam 的二阶矩 $v_t$ 被 spike 污染后，为什么恢复很慢？
**Why does it take a long time for Adam to recover after $v_t$ is contaminated by a spike?**
> $v_t$ 是指数移动平均（$\beta_2=0.999$），单次 spike 贡献的 $g_t^2$ 会以 $\beta_2^k$ 缓慢衰减，大约需要 1000 步才能基本消除。在衰减期间，$\sqrt{\hat{v}_t}$ 偏大，导致所有参数的学习率 $\eta/\sqrt{\hat{v}_t}$ 系统性偏小。

</details>

---

<details>
<summary>Q22. Pre-LN vs Post-LN：从梯度流角度分析为什么 Pre-LN 更稳定？ / Pre-LN vs Post-LN: why is Pre-LN more stable from a gradient flow perspective?</summary>

**答：**
设 Transformer 层输出为 $x_{l+1} = x_l + F_l(\text{LN}(x_l))$（Pre-LN）或 $x_{l+1} = \text{LN}(x_l + F_l(x_l))$（Post-LN）。

**Pre-LN**：$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \left(I + \frac{\partial F_l}{\partial x_l}\right)$，恒等项 $I$ 保证梯度直通，不经过 LN 的 Jacobian 缩放。

**Post-LN**：梯度需穿过每层 LN 的 Jacobian，LN 的 $\gamma/\sqrt{\sigma^2+\epsilon}$ 缩放因子在深层可能累积缩小梯度，导致训练不稳定。

代价：理论上 Post-LN 的表征能力可能更强（LN 作用在 residual 之后能更好地控制激活尺度），但训练难度大。

**追问 Follow-up：** DeepNorm 等方法如何试图结合两者优点？
**How do methods like DeepNorm attempt to combine the benefits of both?**
> DeepNorm 在 Post-LN 框架中对 residual 分支做缩放：$x_{l+1} = \text{LN}(\alpha x_l + F_l(x_l))$，其中 $\alpha > 1$ 放大 residual 路径的权重，同时缩小 $F_l$ 的初始化方差，从而在保持 Post-LN 的表征能力的同时稳定深层梯度。

</details>

---

<details>
<summary>Q23. 为什么大模型训练中经常观察到 "loss plateau"？可能的原因和对策？ / Why do we often observe "loss plateaus" during large model training? Possible causes and remedies?</summary>

**答：**
可能原因：
1. **学习率过大或过小**：在平坦区域 LR 太大导致震荡、太小则进展缓慢。
2. **数据质量瓶颈**：训练数据中已有的信息被充分学习，剩余 loss 来自噪声或不可压缩的部分。
3. **优化器状态问题**：Adam 的 $v_t$ 在 plateau 期间可能过于自信，导致步长偏小。
4. **Loss landscape 中的 saddle point**：梯度接近零但非最优解。

对策：调整 LR schedule（如 WSD：warmup-stable-decay）、数据混合策略（curriculum）、重新初始化优化器状态、或短暂增大 LR 跳出 plateau。

**追问 Follow-up：** WSD（Warmup-Stable-Decay）schedule 是什么？相比 Cosine 有什么优势？
**What is the WSD schedule? What advantages does it have over cosine decay?**
> WSD 分三阶段：linear warmup → 恒定 LR 维持 → 最后阶段快速 decay。优势：① stable 阶段可灵活延长（训练可随时续训或停训）；② 对于需要 checkpoint selection 的场景更方便；③ Cosine decay 需预先确定总步数，灵活性较低。

</details>

---

<details>
<summary>Q24. 在分布式训练中，为什么 Adam 优化器的状态（m, v）是显存的主要瓶颈？如何缓解？ / Why are Adam's optimizer states (m, v) the main memory bottleneck in distributed training? How to mitigate?</summary>

**答：**
每个参数需维护 FP32 master weight（4B）、$m_t$（4B）、$v_t$（4B），共 12B/参数。对于 7B 参数模型，仅优化器状态就需 ~84GB（单卡放不下）。

**缓解方案：**
1. **ZeRO Stage 1**：将优化器状态分片到各 GPU，每卡只存 $1/N$ 的 $(m, v)$。
2. **ZeRO-Offload / ZeRO-Infinity**：将优化器状态卸载到 CPU 内存或 NVMe SSD。
3. **8-bit Adam**：将 $m, v$ 量化为 INT8 存储，节省约 75% 显存。
4. **Adafactor**：用因式分解的二阶矩估计，避免存储完整的 $v_t$。

**追问 Follow-up：** ZeRO 的三个阶段分别优化了什么？通信开销如何变化？
**What does each of the three ZeRO stages optimize? How does communication overhead change?**
> Stage 1 分片优化器状态（几乎无额外通信），Stage 2 额外分片梯度（需一次 AllReduce 替换为 Reduce-Scatter + AllGather），Stage 3 额外分片模型参数（前向和反向都需要 AllGather 通信量增大）。Stage 3 通信量约为 Stage 1 的 1.5 倍，但显存节省最大。

</details>

---

<details>
<summary>Q25. 从理论到实践：设计一个 LLM 的完整训练计划（选择优化器、LR schedule、归一化、精度、正则化等），并解释每个选择。 / Design a complete training plan for an LLM, explaining each choice.</summary>

**答：**

| 设计决策 | 选择 | 理由 |
|---|---|---|
| 优化器 Optimizer | AdamW | 自适应 LR 对稀疏梯度友好，解耦 weight decay 泛化更好 |
| 学习率 LR | Peak LR $\sim 3\text{e-4}$（7B 级），按 sqrt rule 随模型规模缩小 | 过大不收敛，过小效率低 |
| LR Schedule | Linear warmup (1-2% steps) + Cosine decay 到 peak 的 1/10 | Warmup 稳定 Adam 矩估计；cosine 尾端平滑收敛 |
| Weight Decay | 0.1，排除 LN / bias 参数 | 防止权重过大，但不影响归一化层 |
| Gradient Clipping | Clip by norm, max_norm=1.0 | 防止偶发梯度爆炸 |
| 归一化 Normalization | Pre-RMSNorm | 比 LayerNorm 更快（省 mean 计算），Pre-LN 结构训练稳定 |
| 激活函数 Activation | SwiGLU | 比 GELU 效率更高，是当前 LLM 标配 |
| 精度 Precision | BF16（A100/H100） | 动态范围同 FP32，无需 loss scaling |
| Dropout | 不使用 | 大规模数据下不易过拟合，且干扰分布式通信 |
| 梯度累积 | 按需 | 目标 batch size 超出单卡容量时使用 |
| Gradient Checkpointing | 按需 | 显存受限时对部分层开启，牺牲约 30% 计算 |

**追问 Follow-up：** 如果要从这个 base plan 出发做 SFT（Supervised Fine-Tuning），哪些超参数需要调整？为什么？
**If you adapt this plan for SFT, which hyperparameters should change and why?**
> ① LR 降低 1-2 个数量级（如 2e-5），避免破坏预训练知识；② Warmup 步数可减少（预训练权重已稳定）；③ Weight decay 可适当降低；④ 若使用 LoRA，优化器状态显存大幅减小，可关闭 gradient checkpointing；⑤ 总训练步数大幅减少（通常几千到几万步），cosine decay 的 $T$ 需相应调整。

</details>

---


---

*本速查手册为公开教育用途，内容基于已有文献和通用工程实践，不包含任何特定研究的未发表结果。*
*This cheat sheet is for public educational use, based on established literature and common engineering practice. It does not contain any unpublished results from specific research projects.*


## 更多 L3 深挖 / Extended L3

<details>
<summary>Q: 为什么 Adam 有时泛化性能不如 SGD？从优化轨迹的平坦性（flat minima vs. sharp minima）角度分析。</summary>

A: Adam 的自适应学习率可能导致参数收敛到**锐利极小值**（sharp minima），该处曲率大、泛化性差。SGD 的随机梯度噪声天然具有探索性，更倾向于找到**平坦极小值**（flat minima），其对输入扰动更鲁棒。  
   追问：在实际训练中，有哪些方法可以结合 Adam 的收敛速度和 SGD 的泛化能力？

</details>

<details>
<summary>Q: Warmup 除了缓解 Adam 的矩估计偏差，对于 SGD + Momentum 有哪些更深层的必要性？</summary>

A: Momentum 的速度项 \(v_t\) 在训练初期也是冷启动的，直接用较大学习率可能导致初期更新方向被早期少数样本主导，形成**不良的累积动量方向**。Warmup 提供一个低速探索期，让动量方向在大量数据上充分校准后再加速。  
   追问：如果 Warmup 步数设置过短，在什么情况下仍可能造成训练不稳定？

</details>

<details>
<summary>Q: 为什么 RMSNorm 在 LLM 中替代 LayerNorm 的效果通常更好，除了计算效率之外，从参数学习角度分析。</summary>

A: LayerNorm 学习的均值 \(\beta\) 和缩放 \(\gamma\) 存在**耦合**：\(\beta\) 用于中心化，但后续 \(\gamma\) 的缩放会影响中心化的效果。RMSNorm 去掉了均值中心化，直接进行缩放，减少了**自由度之间的相互干扰**，使优化更简单直接，尤其对深层网络稳定。  
   追问：在什么类型的网络结构或任务中，这种中心化可能是必要或有益的？

</details>

<details>
<summary>Q: Kaiming 初始化假设了 ReLU 的方差特性，对于 Swish 或 GELU 等更平滑的激活函数，初始化策略应如何调整？</summary>

A: Swish/GELU 在负半轴并非完全置零，而是保留微小值。因此其方差衰减**小于 ReLU 的 50%**。需要将初始化方差的修正因子（如 Kaiming 中的 2）**适当调小**，以匹配这类激活函数的实际方差保持比例。  
   追问：如何在实际工程中为非标准激活函数确定一个合理的初始化方差修正因子？

</details>

<details>
<summary>Q: 梯度裁剪（Clip by norm）在优化器（如 Adam）内部和外部执行，对训练动力学的影响有何不同？</summary>

A: 在优化器外部裁剪（如在计算梯度后、更新前），直接约束了**参数更新向量的范数**，与优化器状态解耦。在 Adam 内部裁剪（如在计算 \(m_t, v_t\) 后裁剪梯度），会影响**二阶矩估计 \(v_t\)**，进而改变后续自适应学习率的计算，产生更复杂的耦合效应。  
   追问：在使用 AdamW 时，梯度裁剪的最佳实践是放在哪一步？为什么？

</details>

<details>
<summary>Q: 梯度累积在数学上严格等价于大批量训练吗？在哪些情况下这种等价性会被破坏？</summary>

A: 理论上**严格等价**。但当使用 **BatchNorm** 等依赖 batch 统计量的归一化层时，每个 micro-batch 的统计量是独立计算的，导致累积后的归一化统计量**不等于**大批量下的统计量。对于 LayerNorm/RMSNorm 则无此问题。  
   追问：除了归一化层，还有哪些训练组件（如某些正则化技术）可能破坏这种等价性？

</details>

<details>
<summary>Q: Label Smoothing 可以提升模型校准性，但为什么在知识蒸馏中，教师模型通常不使用 Label Smoothing？</summary>

A: 蒸馏的目的是让学生学习教师的**暗知识**，即类别间的相似性关系。Label Smoothing 会将概率分布**过度平滑**，模糊掉这些有价值的关系细节。教师模型应保留其原始、更”尖锐”的分布，以便传递更丰富的结构化信息。  
   追问：如果教师模型本身存在过度自信（校准性差）的问题，在蒸馏前对其输出进行软化，应该采用什么策略？

</details>

<details>
<summary>Q: Focal Loss 的超参数 \(\gamma\) 如何影响梯度流？从梯度视角分析其与重采样/重加权策略的本质区别。</summary>

A: Focal Loss 通过因子 \((1-p_t)^\gamma\) **动态地、连续地**调整每个样本的梯度权重，是一种**隐式的、基于置信度的软重加权**。而传统的重采样/重加权是**显式的、离散的**，且通常基于类别频率而非样本置信度。Focal Loss 的梯度流更平滑，且能自动聚焦于分类不确定的样本。  
   追问：在什么情况下，Focal Loss 可能不如简单的类别重加权有效？

</details>