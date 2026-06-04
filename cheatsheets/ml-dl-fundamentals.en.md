# ML / DL Fundamentals — Public Cheat Sheet

> Covers optimizers, LR scheduling, regularization, normalization, initialization, backpropagation, gradient issues, loss functions, and mixed precision.

---

## Part 1: Concepts & Formula Derivations

---

### 1.1 Optimizers

#### SGD (Stochastic Gradient Descent)

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta L(\theta_t)$$

- No per-parameter adaptive LR; converges slowly but can generalize better.

#### SGD + Momentum

$$v_t = \mu \, v_{t-1} + \nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \, v_t$$

- Momentum $\mu$ (typically 0.9) accumulates past gradients to accelerate through flat regions and dampen oscillations.

#### Adam (Adaptive Moment Estimation)

Maintains the first moment (mean of gradients) and second moment (exponential moving average of squared gradients):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias correction — addresses the bias introduced by zero initialization:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- Adaptive per-parameter learning rate; friendly to sparse gradients.

#### AdamW (Decoupled Weight Decay)

In standard Adam, if the L2 regularization term $\frac{\lambda}{2}\|\theta\|^2$ is added to the loss, the regularization gradient $\lambda\theta$ is scaled by $\sqrt{\hat{v}_t}$, so the effective decay magnitude varies with gradient magnitude — this is not true uniform weight decay.

AdamW **decouples** weight decay from the adaptive update:

$$\theta_{t+1} = (1 - \eta\lambda)\,\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- Uniform decay strength, consistent with weight decay under SGD.

#### Key Formula Summary

| Optimizer | Update Rule |
|---|---|
| SGD | $\theta \leftarrow \theta - \eta \nabla L$ |
| SGD + Momentum | $\theta \leftarrow \theta - \eta\,v_t,\; v_t = \mu v_{t-1} + \nabla L$ |
| Adam | $\theta \leftarrow \theta - \eta \,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ |
| AdamW | $\theta \leftarrow (1-\eta\lambda)\theta - \eta\,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ |

---

### 1.2 Learning Rate Scheduling

#### Warmup

In early training, the estimates $m_t, v_t$ are inaccurate (severely underestimated), so using a large LR directly can cause abnormally large update steps. Warmup linearly increases the LR over the first $T_w$ steps, allowing the moment estimates to accumulate sufficiently.

#### Cosine Decay

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

- LR smoothly decays to $\eta_{\min}$ at the tail for stable convergence.

#### Linear Decay

$$\eta_t = \eta_{\max} - (\eta_{\max} - \eta_{\min}) \cdot \frac{t}{T}$$

- Uniform decrease; LR remains relatively large near the end, potentially causing oscillation.

---

### 1.3 Regularization

#### Dropout

During training, each neuron output is set to zero with probability $p$. **Inverted Dropout** (the PyTorch default) scales non-zero outputs by $1/(1-p)$ during training, requiring no extra operation at inference time.

#### Weight Decay vs L2 Regularization

- **Equivalent under SGD**: The L2 regularization term $\frac{\lambda}{2}\|\theta\|^2$ contributes gradient $\lambda\theta$ to the total gradient, resulting in $\theta \leftarrow \theta(1 - \eta\lambda) - \eta\nabla L$ — i.e., weight decay.
- **Not equivalent under Adam**: The gradient $\lambda\theta$ is divided by $\sqrt{\hat{v}_t}$, so the decay magnitude is coupled to gradient scale. AdamW directly applies $\theta \leftarrow (1-\eta\lambda)\theta$, achieving true decoupled weight decay.

#### Label Smoothing

Replaces the one-hot label $y$ with:

$$y_{\text{smooth}} = (1 - \epsilon)\,y + \frac{\epsilon}{K}$$

where $K$ is the number of classes. This prevents the model from being overconfident in logits and improves probability calibration.

---

### 1.4 Normalization

Assume input tensor shape $(B, C)$ (simplified to 2D, NLP setting) or $(B, C, H, W)$ (CV setting).

| Method | Normalization Dimension | Formula |
|---|---|---|
| **BatchNorm** | Across the batch dimension: computes mean/variance per channel over $(B, H, W)$ | $\hat{x}_c = \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$, learns $\gamma, \beta$ |
| **LayerNorm** | Across the feature dimension: computes mean/variance per sample over $C$ (or $C, H, W$) | $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$, learns $\gamma, \beta$ |
| **RMSNorm** | RMS scaling only, no mean centering | $\hat{x} = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$ |

- **RMSNorm** skips mean computation, is faster, and is the mainstream choice for LLMs.
- **Pre-LN** (LN before sublayer input): gradients flow through the residual path unimpeded, giving more stable training.
- **Post-LN** (LN after the residual): may achieve slightly higher final performance, but training is less stable and requires careful tuning.

---

### 1.5 Weight Initialization

#### Xavier (Glorot) Initialization

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

Or the Gaussian variant: $W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$

- Goal: maintain constant variance of activations across layers (for linear/tanh).

#### Kaiming (He) Initialization

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

- For ReLU: negative-half zeroing halves variance; the factor of 2 compensates.

---

### 1.6 Backpropagation

#### Chain Rule

For a computation graph $L = f(g(x))$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial x}$$

For layer $l$ in a multi-layer network:

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial a_l} \cdot \frac{\partial a_l}{\partial W_l}$$

where $a_l$ is the output of that layer. During the forward pass, **intermediate activations must be cached** for use in the backward pass.

#### Vanishing / Exploding Gradients

Gradients are multiplied across deep layers:

$$\frac{\partial L}{\partial W_1} = \prod_{l=2}^{L} \frac{\partial a_l}{\partial a_{l-1}} \cdot \frac{\partial a_L}{\partial W_1}$$

- If each factor $< 1$: vanishing gradients → shallow layers barely update.
- If each factor $> 1$: exploding gradients → parameters oscillate violently.

#### Gradient Clipping

- **Clip by value**: $g_i \leftarrow \text{clip}(g_i, -c, c)$ — may change the gradient direction.
- **Clip by norm** (mainstream): $\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1,\, \frac{c}{\|\mathbf{g}\|_2}\right)$ — preserves direction, scales only magnitude.

---

### 1.7 Loss Functions

#### Cross-Entropy Loss

For one-hot label $y$ and softmax output $q$:

$$\text{CE} = -\sum_{i=1}^{K} y_i \log q_i = -\log q_y$$

PyTorch's `CrossEntropyLoss` internally applies LogSoftmax followed by NLLLoss:

$$\text{CE}(z, y) = -z_y + \log\!\left(\sum_{j=1}^{K} e^{z_j}\right)$$

#### Focal Loss

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

Down-weights easy examples and focuses on hard ones; suited for class-imbalanced settings.

#### Perplexity

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)$$

- Exponentiated per-token cross-entropy; lower is better.

#### Knowledge Distillation

$$L = \alpha \cdot \text{CE}(y, \hat{y}_s) + (1 - \alpha) \cdot T^2 \cdot \text{KL}\!\left(\hat{y}_t^{(T)} \,\|\, \hat{y}_s^{(T)}\right)$$

- Teacher's soft labels (smoothed by temperature $T$) provide "dark knowledge."

---

### 1.8 Mixed Precision Training

| Dtype | Exponent Bits | Mantissa Bits | Dynamic Range |
|---|---|---|---|
| FP32 | 8 | 23 | $\sim 10^{\pm 38}$ |
| FP16 | 5 | 10 | $\sim 10^{\pm 4.8}$ |
| BF16 | 8 | 7 | $\sim 10^{\pm 38}$ |

**Core Pipeline:**

1. **FP32 Master Weights**: maintain a copy of parameters in FP32 for gradient updates.
2. **Forward + Backward**: executed in FP16 or BF16 to reduce memory and accelerate computation (Tensor Cores).
3. **Loss Scaling** (FP16 only): scale up the loss to prevent gradient underflow; scale back down before the update.
4. **BF16 advantage**: same dynamic range as FP32; no loss scaling needed.

---

### 1.9 Bias-Variance Decomposition

$$\mathbb{E}[\text{MSE}] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **High Bias** (Underfitting): both train and val loss are high.
- **High Variance** (Overfitting): train loss is low, val loss is notably higher.

---

### 1.10 Batch Size and Gradient Accumulation

**Linear Scaling Rule**: when batch size is multiplied by $k$, scale the LR by $k$ as well, to keep the expected per-update step size unchanged.

**Gradient Accumulation**: split a mini-batch into $k$ micro-batches, accumulate gradients for the first $k-1$ steps (no update), then perform a single update at step $k$ — equivalent to a $k\times$ larger batch size with no additional memory cost.

---

## Part 2: PyTorch Snippets (From Scratch)

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

### 2.3 AdamW (Decoupled Weight Decay)

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

### 2.11 GELU and SwiGLU

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

### 2.13 LR Finder

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

## Part 3: Interview Questions (25 Questions)

---

### L1 — Basic

---

<details>
<summary>Q1. What are the key differences among SGD, Adam, and AdamW?</summary>

**Answer:**
- **SGD**: Fixed learning rate, no adaptive mechanism; converges slowly but can generalize better.
- **Adam**: Maintains first moment $m_t$ (gradient mean) and second moment $v_t$ (mean squared gradient); after bias correction, adaptively adjusts the learning rate per parameter. However, its weight decay implementation mixes the regularization gradient into the second-moment denominator, so the effective decay magnitude varies with gradient scale.
- **AdamW**: Decouples weight decay from the adaptive update, applying it directly to parameters: $\theta \leftarrow (1-\eta\lambda)\theta - \eta\,\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$, giving decay behavior consistent with SGD.

**Follow-up:** Why is Adam's L2 regularization implementation considered "incorrect"? What does AdamW's decoupling fundamentally change?
> The L2 regularization gradient $\lambda\theta$ is divided by $\sqrt{\hat{v}_t}$, so parameters with large gradients receive weak decay and those with small gradients receive strong decay — contrary to the design intent. AdamW applies uniform decay to all parameters.

</details>

---

<details>
<summary>Q2. Why is bias correction necessary in Adam?</summary>

**Answer:**
$m_0 = v_0 = 0$, so during the first few steps the exponential moving averages are heavily biased toward zero. Taking $m_t$ as an example: $m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$, whose expectation is $\mathbb{E}[m_t] = \mathbb{E}[g_t]\cdot(1-\beta_1^t)$. Without dividing by $(1-\beta_1^t)$, the initial step sizes are systematically too small. The correction $\hat{m}_t = m_t/(1-\beta_1^t)$ ensures reasonable update magnitudes early in training.

**Follow-up:** What intuition do the default values $\beta_1 = 0.9$ and $\beta_2 = 0.999$ correspond to?
> $\beta_1=0.9$ corresponds to a history window of roughly 10 steps (gradient direction estimation); $\beta_2=0.999$ corresponds to roughly 1000 steps (slow tracking of gradient magnitude for stable adaptive scaling).

</details>

---

<details>
<summary>Q3. When should you use SGD+Momentum vs AdamW?</summary>

**Answer:**
- **SGD+Momentum**: CV tasks (e.g., ResNet, ViT), where careful tuning often yields better generalization, but LR schedule and momentum must be dialed in carefully.
- **AdamW**: The standard choice for NLP / LLM pre-training and fine-tuning; adaptive LR is friendly to sparse token gradients and reduces tuning overhead.

**Follow-up:** Why is AdamW almost universally used for LoRA fine-tuning?
> LoRA has few parameters and needs to converge quickly to a good low-rank subspace; adaptive LR reduces tuning burden. LoRA is commonly used in LLM fine-tuning settings, where AdamW is inherited from the pre-training choice.

</details>

---

<details>
<summary>Q4. Why do LLMs need LR warmup?</summary>

**Answer:**
Initial parameters are random and gradient variance is high. Adam's second moment $v_t$ is poorly estimated at the start; using a large LR immediately causes the effective step size $\eta/\sqrt{\hat{v}_t}$ to be abnormally large, leading to loss spikes or divergence. Warmup lets $v_t$ accumulate sufficiently under a low LR before the LR is gradually increased, stabilizing early training.

**Follow-up:** Is warmup still necessary when fine-tuning from a pre-trained checkpoint?
> Usually yes, but the warmup duration can be greatly reduced (e.g., only a few hundred steps). The pre-trained weights are already stable, but newly added LoRA adapters or shifts in the data distribution still require a brief adaptation period.

</details>

---

<details>
<summary>Q5. What is the relationship between Cross-Entropy and NLL Loss? Why not use MSE for classification?</summary>

**Answer:**
- Cross-entropy $H(p,q) = -\sum_i p_i \log q_i$. When $p$ is one-hot, this simplifies to $-\log q_y$, i.e., NLL. PyTorch's <code>CrossEntropyLoss = LogSoftmax + NLLLoss</code>.
- **Why not MSE**: MSE gradients on softmax outputs are extremely small in the saturation region (vanishing gradients), leading to slow convergence. CE's gradient $\hat{q}_y - 1$ never vanishes and is numerically well-behaved.

**Follow-up:** How does label smoothing modify CE loss? Why does it improve calibration?
> It replaces the one-hot target with $(1-\epsilon)y + \epsilon/K$, preventing the model from chasing extreme confidence scores, reducing overconfident predictions, and improving probability calibration.

</details>

---

<details>
<summary>Q6. How does Dropout work? How is it handled at inference?</summary>

**Answer:**
During training, each neuron output is randomly set to zero with probability $p$, equivalent to training an ensemble of exponentially many sub-networks. At inference, no dropout is applied. PyTorch uses **inverted dropout** by default: during training, non-zero outputs are scaled by $1/(1-p)$, so inference requires no extra operation and expected outputs remain consistent.

**Follow-up:** Where is Dropout typically applied in a Transformer? Is it still used in large-scale pre-training?
> Typically after attention weights, before residual addition, and after the FFN hidden layer. In large-scale pre-training (e.g., LLaMA, GPT), **Dropout is generally not used**, because massive data makes overfitting unlikely and Dropout can interfere with distributed training communication.

</details>

---

<details>
<summary>Q7. What is Perplexity and how does it relate to Cross-Entropy?</summary>

**Answer:**
$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)$$
This is the exponentiated per-token cross-entropy. Intuitively, it represents the average number of equally probable choices the model faces at each position; lower PPL means the model is more confident.

**Follow-up:** Is PPL comparable across different tokenizers?
> Not directly. Different tokenizers produce tokens of different granularity (e.g., BPE vs SentencePiece), so each token carries different amounts of information. A normalized metric such as bits-per-byte (BPB) is needed for fair comparison.

</details>

---

<details>
<summary>Q8. What are vanishing and exploding gradients? How to address each?</summary>

**Answer:**
- **Vanishing gradients**: In deep networks, repeated multiplication by factors less than 1 drives gradients toward zero, leaving shallow-layer parameters nearly unchanged. Remedies: ReLU/GELU activations, residual connections, normalization (LayerNorm/BatchNorm), proper initialization.
- **Exploding gradients**: Repeated multiplication by factors greater than 1 causes exponential growth. Remedies: gradient clipping (most direct), weight regularization, reducing learning rate.

**Follow-up:** How do residual connections in Transformers help gradient flow?
> The residual path $x + F(x)$ gives gradient $\partial L/\partial x = \partial L/\partial(\text{output}) \cdot (1 + \partial F/\partial x)$, where the constant term 1 guarantees gradients can reach shallow layers directly without being attenuated by intermediate layers.

</details>

---

<details>
<summary>Q9. What are the differences and use cases of BatchNorm, LayerNorm, and RMSNorm?</summary>

**Answer:**

| Method | Normalization Dimension | Use Case |
|---|---|---|
| BatchNorm | Across batch dimension (per-channel statistics over the batch) | CV tasks, larger batch sizes |
| LayerNorm | Across feature dimension (per-sample statistics) | NLP/Transformer, small batches or variable-length sequences |
| RMSNorm | RMS scaling only, no mean centering | LLMs (e.g., LLaMA), faster, comparable performance |

BatchNorm performs poorly in NLP: variable sequence lengths make batch statistics unstable; inference requires maintaining running statistics, which is ill-suited to autoregressive generation.

**Follow-up:** What is the difference between Pre-LN and Post-LN? Which trains more stably?
> Pre-LN places LN before the sublayer input; gradients flow through the residual path unimpeded, making training more stable. Post-LN places LN after the residual; gradients in deep layers may shrink, requiring careful tuning and warmup. Modern LLMs predominantly use Pre-LN (or Pre-RMSNorm).

</details>

---

<details>
<summary>Q10. What is the difference between Xavier and Kaiming initialization?</summary>

**Answer:**
- **Xavier**: $W \sim \mathcal{N}(0, 2/(n_{\text{in}}+n_{\text{out}}))$; assumes linear or tanh activations; goal is to maintain constant output variance per layer.
- **Kaiming**: $W \sim \mathcal{N}(0, 2/n_{\text{in}})$; designed for ReLU — zeroing the negative half halves the variance, and the factor of 2 compensates.

**Follow-up:** How are attention projection matrices typically initialized in Transformers?
> Typically with Xavier, since inputs pass through softmax or LayerNorm and are approximately linear. Defaults vary slightly across frameworks, but Xavier uniform or normal is the mainstream practice.

</details>

---

### L2 — Intermediate

---

<details>
<summary>Q11. What are the core steps of the backpropagation algorithm?</summary>

**Answer:**
1. **Forward pass**: compute outputs layer by layer, caching intermediate activations $a_l$.
2. **Compute loss**: obtain the scalar loss $L$.
3. **Backward pass**: starting from $\partial L / \partial L = 1$, apply the chain rule layer by layer to compute $\partial L / \partial W_l$, using the cached activations.
4. **Parameter update**: the optimizer updates parameters using the gradients.

Key mechanism: cache during forward pass → reuse during backward pass; the chain rule ensures correct gradient computation at each layer.

**Follow-up:** How does gradient checkpointing trade off memory vs computation?
> Only activations at every $k$-th layer are saved; intermediate activations are recomputed during the backward pass. Memory drops from $O(L)$ to roughly $O(\sqrt{L})$, at the cost of approximately 33% extra computation. Useful when memory is the bottleneck.

</details>

---

<details>
<summary>Q12. What are two types of gradient clipping? When to use clip-by-norm?</summary>

**Answer:**
1. **Clip by value**: clamp each gradient component to $[-c, c]$; simple but may alter the gradient direction.
2. **Clip by norm**: $\mathbf{g} \leftarrow \mathbf{g} \cdot \min(1, c/\|\mathbf{g}\|_2)$; preserves gradient direction, only scales magnitude.

LLM training almost always uses clip by norm (typical threshold 1.0) because it does not distort the optimization direction.

**Follow-up:** How do you diagnose a loss spike (sudden gradient norm surge)?
> Common causes: data anomalies (very long sequences, abnormal tokens), LR too large, numerical overflow (FP16), or a batch whose gradients conflict sharply with historical directions. Check data quality, lower the LR, switch to BF16, or increase the clipping threshold while monitoring.

</details>

---

<details>
<summary>Q13. What is the relationship between Weight Decay and L2 regularization? Why do they diverge in Adam?</summary>

**Answer:**
- **Equivalent under SGD**: adding $\frac{\lambda}{2}\|\theta\|^2$ to the loss contributes gradient $\lambda\theta$ to the total gradient, resulting in $\theta \leftarrow \theta(1-\eta\lambda) - \eta\nabla L$ — i.e., weight decay.
- **Not equivalent under Adam**: $\lambda\theta$ is divided by $\sqrt{\hat{v}_t}$; parameters with large gradients receive weak decay and those with small gradients receive strong decay. AdamW decouples decay as $\theta \leftarrow (1-\eta\lambda)\theta$, achieving uniform decay.

**Follow-up:** What is a typical weight decay value for AdamW? Which parameters are typically excluded?
> Common values: 0.01–0.1. LayerNorm's $\gamma, \beta$ and biases are typically excluded from weight decay (they have other regularization mechanisms or are low-dimensional); decay is applied only to weight matrices.

</details>

---

<details>
<summary>Q14. What is Early Stopping and what are its drawbacks?</summary>

**Answer:**
Stop training when validation loss stops improving (no improvement for $p$ epochs), to prevent overfitting.

**Drawbacks:**
1. Requires a separate validation set, reducing available training data.
2. Requires continuous training monitoring (engineering overhead).
3. Plateaus may trigger early stopping spuriously (requires tuning patience).
4. The final model is an intermediate checkpoint, not a fully converged solution.

**Follow-up:** What is the theoretical connection between Early Stopping and L2 regularization?
> Both have the effect of constraining the parameter space. Early Stopping limits how far parameters can move from initialization (an implicit norm constraint); in the linear approximation of gradient descent, this is equivalent to an L2 regularized solution with a specific regularization coefficient.

</details>

---

<details>
<summary>Q15. What is the bias-variance tradeoff and how do you diagnose it?</summary>

**Answer:**
$$\mathbb{E}[\text{MSE}] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$
- **High bias (underfitting)**: both train and val loss are high; the model lacks capacity.
- **High variance (overfitting)**: train loss is low, val loss is notably higher; the model has memorized the training set.

Diagnostic tools: train/val loss curves, learning curves (error vs. training set size).

**Follow-up:** What new challenges arise in diagnosing "overfitting" in the LLM era?
> Key challenges: (1) training data and evaluation data may overlap (benchmark contamination), making low loss appear without genuine generalization improvement; (2) LLMs' in-context learning ability is not reflected in train loss; (3) evaluation must rely on downstream task performance, not just val loss.

</details>

---

<details>
<summary>Q16. What is the core idea of Knowledge Distillation?</summary>

**Answer:**
Train a small model (student) using a large model's (teacher's) soft labels (the probability distribution smoothed by temperature $T$), rather than hard labels alone. The loss is:
$$L = \alpha \cdot \text{CE}(y, \hat{y}_s) + (1-\alpha) \cdot T^2 \cdot \text{KL}(\hat{y}_t^{(T)} \| \hat{y}_s^{(T)})$$
Soft labels convey inter-class similarity relationships ("dark knowledge"), helping the student learn richer representations.

**Follow-up:** What is the difference between token-level and sequence-level KD?
> Token-level KD computes KL divergence on logits at each position (per-token alignment); sequence-level KD matches full-sequence probability distributions or output sequences (e.g., using teacher's beam search output as pseudo-labels). The latter is more flexible but provides sparser training signal.

</details>

---

<details>
<summary>Q17. How does batch size affect training? What is the Linear Scaling Rule?</summary>

**Answer:**
- Large batches provide more accurate gradient estimates, but reduce gradient noise (which has a regularizing effect) and may converge to sharp minima with worse generalization.
- **Linear Scaling Rule**: when batch size is multiplied by $k$, scale the LR by $k$ to keep the expected per-update step size unchanged. Beyond a certain scale, more conservative scaling (e.g., sqrt rule) is needed.

**Follow-up:** How does gradient accumulation simulate large batches? Any limitations?
> Split a mini-batch into $k$ micro-batches, accumulate gradients, then update once — equivalent to $k\times$ batch size. Limitations: (1) if BatchNorm is used, statistics are still computed per micro-batch, not the effective large batch; (2) the number of optimizer steps decreases by $k\times$, reducing communication frequency.

</details>

---

<details>
<summary>Q18. What is GELU and why do Transformers prefer it?</summary>

**Answer:**
$$\text{GELU}(x) = x \cdot \Phi(x)$$
where $\Phi(x)$ is the standard normal CDF. GELU is a smooth approximation of ReLU: it retains small values in the negative region (no hard cutoff) and acts as a probabilistic gate (larger $x$ has a higher probability of being retained). It is more suitable than ReLU for NLP tasks and has smoother gradients near zero.

**Follow-up:** What is SwiGLU? How does it structurally differ from a GELU FFN?
> $\text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV)$, using two linear projections for gating. A standard FFN uses $W_{\text{up}} \to \text{GELU} \to W_{\text{down}}$ (2 matrices); SwiGLU FFN requires 3 matrices (gate, up, down), and the hidden dimension is typically set to $(8/3) \cdot d_{\text{model}}$ to keep the total parameter count comparable.

</details>

---

<details>
<summary>Q19. What is the principle of AMP training? Why do we need FP32 master weights?</summary>

**Answer:**
- **Motivation**: FP16/BF16 reduces memory and speeds up computation (Tensor Cores), but FP16's small dynamic range (max ~65504) makes gradients prone to underflow/overflow.
- **Approach**: forward + backward in low precision; convert back to FP32 master weights for gradient updates to preserve precision; FP16 requires loss scaling to prevent gradient underflow.
- **BF16**: same dynamic range as FP32 (8 exponent bits); no loss scaling needed; the mainstream choice for LLM training today.

**Follow-up:** What additional challenges does FP8 training introduce?
> FP8 has lower precision (E4M3/E5M2 formats), requiring finer-grained scaling strategies (e.g., per-tensor or per-channel scaling); gradient accumulation requires higher precision; some layers (e.g., logits before attention softmax) are more precision-sensitive and may need to be kept at higher precision.

</details>

---

<details>
<summary>Q20. What is gradient checkpointing and when should you use it?</summary>

**Answer:**
During training, only a subset of key layer activations are saved (e.g., every $k$ layers); activations that were discarded are recomputed via a forward pass during backpropagation. This trades roughly 30% extra computation to reduce memory from $O(L)$ to $O(\sqrt{L})$ ($L$ = number of layers).

**When to use**: large batches, long sequences, or memory-constrained settings. If using LoRA with most parameters frozen, activation memory overhead is relatively manageable and checkpointing may not be needed.

**Follow-up:** Do gradient checkpointing and ZeRO-Offload solve the same problem?
> Not exactly. Checkpointing reduces **activation memory** through recomputation; ZeRO-Offload offloads **optimizer states and gradients** from GPU to CPU/SSD. They address different memory bottlenecks and can be used together.

</details>

---

### L3 — Deep

---

<details>
<summary>Q21. What is a systematic debugging procedure for loss spikes during training?</summary>

**Answer:**
1. **Data check**: look for anomalous inputs (very long sequences, NaN values, special tokens).
2. **Gradient monitoring**: log per-layer gradient norms to confirm whether a specific layer is exploding.
3. **Precision check**: is FP16 overflowing? Verify that the loss scaling factor is reasonable.
4. **LR / optimizer state**: is the LR too large? Has Adam's $v_t$ estimate been contaminated by the spike?
5. **Recovery strategy**: can you restart from the checkpoint before the spike (skipping the problematic batch)?
6. **Long-term mitigation**: increase clipping frequency (per-step vs per-accumulation), switch from FP16 to BF16.

**Follow-up:** Why does it take a long time for Adam to recover after $v_t$ is contaminated by a spike?
> $v_t$ is an exponential moving average ($\beta_2=0.999$); a single spike's contribution $g_t^2$ decays as $\beta_2^k$, taking roughly 1000 steps to dissipate. During this decay, $\sqrt{\hat{v}_t}$ is inflated, causing the effective learning rate $\eta/\sqrt{\hat{v}_t}$ to be systematically reduced for all parameters.

</details>

---

<details>
<summary>Q22. Pre-LN vs Post-LN: why is Pre-LN more stable from a gradient flow perspective?</summary>

**Answer:**
Let the Transformer layer output be $x_{l+1} = x_l + F_l(\text{LN}(x_l))$ (Pre-LN) or $x_{l+1} = \text{LN}(x_l + F_l(x_l))$ (Post-LN).

**Pre-LN**: $\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \left(I + \frac{\partial F_l}{\partial x_l}\right)$ — the identity term $I$ guarantees direct gradient flow, unaffected by the LN Jacobian.

**Post-LN**: gradients must pass through the LN Jacobian at every layer; the $\gamma/\sqrt{\sigma^2+\epsilon}$ scaling factor can accumulate and shrink gradients in deep layers, causing training instability.

Trade-off: Post-LN may theoretically have stronger representational capacity (LN after the residual better controls activation scale), but training difficulty is higher.

**Follow-up:** How do methods like DeepNorm attempt to combine the benefits of both?
> DeepNorm scales the residual branch in a Post-LN setting: $x_{l+1} = \text{LN}(\alpha x_l + F_l(x_l))$, where $\alpha > 1$ amplifies the residual path while the initialization variance of $F_l$ is reduced, stabilizing deep-layer gradients while preserving Post-LN's representational capacity.

</details>

---

<details>
<summary>Q23. Why do we often observe "loss plateaus" during large model training? Possible causes and remedies?</summary>

**Answer:**
Possible causes:
1. **LR too large or too small**: in a flat region, too-large LR causes oscillation; too-small LR leads to slow progress.
2. **Data quality bottleneck**: information already present in the training data has been learned; remaining loss comes from noise or incompressible content.
3. **Optimizer state issues**: Adam's $v_t$ may become overconfident during plateaus, causing step sizes to shrink.
4. **Saddle point in the loss landscape**: gradients near zero but not at the optimum.

Remedies: adjust the LR schedule (e.g., WSD: warmup-stable-decay), revise data mixing strategy (curriculum), reinitialize optimizer states, or temporarily increase LR to escape the plateau.

**Follow-up:** What is the WSD schedule? What advantages does it have over cosine decay?
> WSD has three phases: linear warmup → constant LR (stable phase) → rapid decay in the final stage. Advantages: (1) the stable phase can be extended flexibly (training can be continued or stopped at any time); (2) more convenient for checkpoint selection; (3) cosine decay requires the total number of steps to be fixed in advance, offering less flexibility.

</details>

---

<details>
<summary>Q24. Why are Adam's optimizer states (m, v) the main memory bottleneck in distributed training? How to mitigate?</summary>

**Answer:**
Each parameter requires FP32 master weight (4 bytes), $m_t$ (4 bytes), and $v_t$ (4 bytes) — 12 bytes per parameter. For a 7B-parameter model, optimizer states alone require ~84 GB (too large for a single GPU).

**Mitigations:**
1. **ZeRO Stage 1**: shard optimizer states across GPUs; each GPU stores only $1/N$ of $(m, v)$.
2. **ZeRO-Offload / ZeRO-Infinity**: offload optimizer states to CPU memory or NVMe SSD.
3. **8-bit Adam**: quantize $m, v$ to INT8, saving ~75% memory.
4. **Adafactor**: use a factorized second-moment estimate, avoiding storage of the full $v_t$.

**Follow-up:** What does each of the three ZeRO stages optimize? How does communication overhead change?
> Stage 1 shards optimizer states (almost no extra communication); Stage 2 additionally shards gradients (replaces AllReduce with Reduce-Scatter + AllGather); Stage 3 additionally shards model parameters (AllGather needed during both forward and backward, increasing communication). Stage 3 communication is roughly 1.5× that of Stage 1, but saves the most memory.

</details>

---

<details>
<summary>Q25. Design a complete training plan for an LLM, explaining each choice.</summary>

**Answer:**

| Design Decision | Choice | Rationale |
|---|---|---|
| Optimizer | AdamW | Adaptive LR is friendly to sparse gradients; decoupled weight decay gives better generalization |
| Learning Rate | Peak LR $\sim 3\text{e-4}$ (7B scale), scaled down with model size via sqrt rule | Too large: no convergence; too small: inefficient |
| LR Schedule | Linear warmup (1–2% of steps) + cosine decay to 1/10 of peak | Warmup stabilizes Adam moment estimates; cosine gives smooth tail convergence |
| Weight Decay | 0.1, excluding LN / bias parameters | Prevents weights from growing too large without affecting normalization layers |
| Gradient Clipping | Clip by norm, max_norm=1.0 | Guards against occasional gradient explosions |
| Normalization | Pre-RMSNorm | Faster than LayerNorm (skips mean computation); Pre-LN structure is more stable |
| Activation | SwiGLU | More efficient than GELU; the current standard for LLMs |
| Precision | BF16 (A100/H100) | Same dynamic range as FP32; no loss scaling needed |
| Dropout | Not used | Large-scale data makes overfitting unlikely; Dropout interferes with distributed communication |
| Gradient Accumulation | As needed | Used when target batch size exceeds single-GPU capacity |
| Gradient Checkpointing | As needed | Enable for select layers when memory-constrained, at ~30% compute cost |

**Follow-up:** If you adapt this plan for SFT, which hyperparameters should change and why?
> (1) LR reduced by 1–2 orders of magnitude (e.g., 2e-5) to avoid overwriting pre-trained knowledge; (2) warmup steps can be reduced (pre-trained weights are already stable); (3) weight decay can be lowered slightly; (4) if using LoRA, optimizer state memory is much smaller and gradient checkpointing may be turned off; (5) total training steps are greatly reduced (typically a few thousand to tens of thousands), so the cosine decay $T$ must be adjusted accordingly.

</details>

---


---

*This cheat sheet is for public educational use, based on established literature and common engineering practice. It does not contain any unpublished results from specific research projects.*


## Extended L3

<details>
<summary>Q: Why does Adam sometimes generalize worse than SGD? Analyze from the perspective of flat vs. sharp minima in the optimization trajectory.</summary>

A: Adam's adaptive learning rate can cause parameters to converge to **sharp minima**, which have high curvature and poor generalization. SGD's stochastic gradient noise has a natural exploratory quality that tends to find **flat minima**, which are more robust to input perturbations.  
   Follow-up: In practice, what methods can combine Adam's convergence speed with SGD's generalization ability?

</details>

<details>
<summary>Q: Beyond alleviating Adam's moment bias, what are the deeper reasons warmup is necessary for SGD + Momentum?</summary>

A: The velocity term $v_t$ in momentum also starts cold. Using a large LR immediately can cause early updates to be dominated by the first few batches, leading to a **poorly calibrated accumulated momentum direction**. Warmup provides a low-speed exploration phase, allowing the momentum direction to be properly calibrated over a large number of samples before accelerating.  
   Follow-up: If the warmup duration is set too short, under what circumstances could training instability still occur?

</details>

<details>
<summary>Q: Why does RMSNorm generally work as well as or better than LayerNorm in LLMs, beyond computational efficiency? Analyze from the perspective of parameter learning.</summary>

A: In LayerNorm, the learned mean shift $\beta$ and scale $\gamma$ are **coupled**: $\beta$ handles centering, but the subsequent $\gamma$ scaling interacts with the centering effect. RMSNorm removes mean centering and applies scaling directly, reducing **interference between degrees of freedom** and making optimization simpler and more direct, especially for stabilizing deep networks.  
   Follow-up: In what types of network architectures or tasks might mean centering still be necessary or beneficial?

</details>

<details>
<summary>Q: Kaiming initialization assumes the variance characteristics of ReLU. How should the initialization strategy be adjusted for smoother activations like Swish or GELU?</summary>

A: Swish/GELU do not completely zero out the negative half; they retain small negative values. Therefore, their variance attenuation is **less than ReLU's 50%**. The correction factor in the initialization variance (such as the factor of 2 in Kaiming) should be **reduced appropriately** to match the actual variance preservation ratio of these activations.  
   Follow-up: How would you determine a reasonable initialization variance correction factor for a non-standard activation function in practice?

</details>

<details>
<summary>Q: What are the different effects on training dynamics when gradient clipping (clip by norm) is applied outside vs. inside the optimizer (e.g., Adam)?</summary>

A: Clipping outside the optimizer (after gradient computation, before the update) directly constrains the **norm of the parameter update vector**, decoupled from optimizer state. Clipping inside Adam (after computing $m_t, v_t$) affects the **second-moment estimate $v_t$**, which then alters the adaptive learning rate in subsequent steps, introducing more complex coupling effects.  
   Follow-up: When using AdamW, where in the training step is the best practice to apply gradient clipping? Why?

</details>

<details>
<summary>Q: Is gradient accumulation mathematically strictly equivalent to large-batch training? Under what circumstances does this equivalence break down?</summary>

A: Theoretically **strictly equivalent**. However, when using **BatchNorm** or other normalization layers that rely on batch statistics, each micro-batch computes its own statistics independently, so the accumulated normalization statistics are **not equal** to those from a single large batch. For LayerNorm/RMSNorm, this problem does not arise.  
   Follow-up: Besides normalization layers, what other training components (e.g., certain regularization techniques) might break this equivalence?

</details>

<details>
<summary>Q: Label Smoothing can improve model calibration, but why is the teacher model typically not trained with Label Smoothing in knowledge distillation?</summary>

A: The purpose of distillation is for the student to learn the teacher's **dark knowledge** — namely, inter-class similarity relationships. Label Smoothing **over-smooths** the probability distribution, blurring these valuable relational details. The teacher model should retain its original, "sharper" distribution in order to convey richer structured information.  
   Follow-up: If the teacher model itself is overconfident (poor calibration), what strategy should be used to soften its outputs before distillation?

</details>

<details>
<summary>Q: How does the hyperparameter $\gamma$ in Focal Loss affect gradient flow? Analyze its fundamental difference from resampling/reweighting strategies from a gradient perspective.</summary>

A: Focal Loss uses the factor $(1-p_t)^\gamma$ to **dynamically and continuously** adjust the gradient weight of each sample — it is an **implicit, confidence-based soft reweighting**. Traditional resampling/reweighting is **explicit and discrete**, typically based on class frequency rather than per-sample confidence. Focal Loss has smoother gradient flow and automatically focuses on samples where classification is uncertain.  
   Follow-up: Under what circumstances might Focal Loss be less effective than simple class-frequency reweighting?

</details>
