# 机器学习数学与统计速查手册 / Math & Statistics for ML Cheat Sheet

> 覆盖概率、线性代数、微积分、信息论、MLE / MAP、常见分布。双语标注：中文为主要语言，英文术语用括号标注。
>
> **适用场景**：ML/DL 面试复习、课程笔记速查、技术博客参考。

---

## Part 1: 概念与公式推导 / Concepts & Formula Derivations

### 1.1 概率基础 / Probability Fundamentals

**条件概率 (Conditional Probability)、联合概率 (Joint)、边缘概率 (Marginal)** 的基本关系：

$$P(A, B) = P(A \mid B)\, P(B) = P(B \mid A)\, P(A)$$

$$P(A) = \sum_B P(A, B) \quad \text{（边缘化 / Marginalization）}$$

**贝叶斯定理 (Bayes' Theorem)**：

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

> 后验 (posterior) ∝ 似然 (likelihood) × 先验 (prior)。

**链式法则 (Chain Rule of Probability)**：

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid X_1, \ldots, X_{i-1})$$

**条件独立 (Conditional Independence)**：$A \perp B \mid C$ 表示在给定 $C$ 时，$A$ 与 $B$ 独立，即 $P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$。

---

### 1.2 期望、方差与协方差 / Expectation, Variance, Covariance

| 性质 | 公式 |
|------|------|
| 线性 | $E[aX + b] = a\,E[X] + b$ |
| 方差定义 | $\text{Var}(X) = E[X^2] - (E[X])^2$ |
| 缩放 | $\text{Var}(aX + b) = a^2\,\text{Var}(X)$ |
| 和的方差 | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X,Y)$ |
| 协方差 | $\text{Cov}(X,Y) = E[XY] - E[X]\,E[Y]$ |

**协方差矩阵 (Covariance Matrix)** $\Sigma$：对角元素 $\Sigma_{ii} = \text{Var}(X_i)$；矩阵总是**半正定 (PSD)** 的，因为 $\forall w: w^\top \Sigma w = \text{Var}(w^\top X) \geq 0$。

---

### 1.3 常见概率分布 / Common Distributions

| 分布 | 参数 | 均值 | 方差 | 典型应用 |
|------|------|------|------|----------|
| 伯努利 Bernoulli | $p$ | $p$ | $p(1-p)$ | 二分类标签 |
| 二项 Binomial | $n, p$ | $np$ | $np(1-p)$ | $n$ 次独立实验成功次数 |
| 泊松 Poisson | $\lambda$ | $\lambda$ | $\lambda$ | 稀有事件计数 |
| 均匀 Uniform | $[a, b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | 参数初始化 |
| 高斯 Gaussian | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | 噪声模型、权重先验 |
| 指数 Exponential | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | 等待时间建模 |

**多元高斯 PDF (Multivariate Gaussian)**：

$$\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\!\Big(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\Big)$$

**经验法则 (68-95-99.7)**：对于 $\mathcal{N}(\mu, \sigma^2)$，约 68% 数据落在 $\mu \pm 1\sigma$，95% 在 $\mu \pm 2\sigma$，99.7% 在 $\mu \pm 3\sigma$。

---

### 1.4 中心极限定理 / Central Limit Theorem (CLT)

设 $X_1, \ldots, X_n$ i.i.d.，均值 $\mu$，方差 $\sigma^2$，当 $n \to \infty$：

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

**ML 联系**：Mini-batch SGD 中，batch 梯度是总体梯度的无偏估计，其方差 $\propto 1/B$（$B$ 为 batch size）。SGD 噪声在非凸优化中有隐式正则化 (implicit regularization) 效果，帮助逃离尖锐极小值 (sharp minima)。

---

### 1.5 MLE 与 MAP / Maximum Likelihood & Maximum A Posteriori

**MLE（最大似然估计）**：

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log P(x_i \mid \theta)$$

纯数据驱动，无先验信息。

**MAP（最大后验估计）**：

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \Big[\sum_i \log P(x_i \mid \theta) + \log P(\theta)\Big]$$

**正则化等价关系**：

| 先验分布 | 对应正则化 |
|----------|-----------|
| 高斯先验 $P(\theta) \propto e^{-\lambda \|\theta\|^2}$ | L2 正则化 (Weight Decay) |
| 拉普拉斯先验 $P(\theta) \propto e^{-\lambda \|\theta\|_1}$ | L1 正则化 (Lasso) |

---

### 1.6 偏差-方差分解 / Bias-Variance Decomposition

$$E\!\Big[\big(\hat{f}(x) - f^*(x)\big)^2\Big] = \underbrace{\big(E[\hat{f}(x)] - f^*(x)\big)^2}_{\text{Bias}^2} + \underbrace{E\!\Big[\big(\hat{f}(x) - E[\hat{f}(x)]\big)^2\Big]}_{\text{Variance}} + \underbrace{\sigma_\epsilon^2}_{\text{Noise}}$$

- **Bias**：模型系统性偏离 → 欠拟合 (underfitting)
- **Variance**：模型对训练数据扰动的敏感性 → 过拟合 (overfitting)
- **Noise**：数据内在的不可约误差

集成方法 (ensemble) 如 Bagging、Dropout 主要降低 Variance。

---

### 1.7 线性代数核心 / Core Linear Algebra

#### 1.7.1 矩阵的秩 (Rank) 与低秩分解

矩阵 $W \in \mathbb{R}^{m \times n}$ 的秩 $r = \text{rank}(W)$ 是其线性无关行（或列）的最大数目，即列空间 (column space) 的维数。

**低秩矩阵的参数效率**：若 $\text{rank}(W) = r \ll \min(m, n)$，则可分解为 $W = AB$，其中 $A \in \mathbb{R}^{m \times r}$，$B \in \mathbb{R}^{r \times n}$。

- 原始参数量：$mn$
- 低秩分解参数量：$r(m + n)$
- 当 $r \ll \min(m, n)$ 时，参数量大幅降低

#### 1.7.2 SVD 分解 (Singular Value Decomposition)

任意矩阵 $W \in \mathbb{R}^{m \times n}$ 可分解为：

$$W = U \Sigma V^\top$$

其中 $U \in \mathbb{R}^{m \times m}$ 正交（左奇异向量），$\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots)$（奇异值，$\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$），$V \in \mathbb{R}^{n \times n}$ 正交（右奇异向量）。

**最优低秩近似 (Eckart–Young–Mirsky Theorem)**：

$$\hat{W}_r = U_r \Sigma_r V_r^\top = \arg\min_{\text{rank}(M) = r} \|W - M\|_F$$

其中 $U_r \in \mathbb{R}^{m \times r}$，$\Sigma_r \in \mathbb{R}^{r \times r}$，$V_r \in \mathbb{R}^{n \times r}$ 由前 $r$ 个奇异值/向量组成。

**两种范数与奇异值的关系**：

- **Frobenius 范数**：$\|W\|_F = \sqrt{\sum_i \sigma_i^2}$
- **谱范数 (Spectral Norm)**：$\|W\|_2 = \sigma_1$（最大奇异值）

**ML 应用**：
- PCA：协方差矩阵的右奇异向量 = 主成分方向
- 参数压缩：SVD 截断用于模型压缩
- LoRA 初始化策略：对预训练权重做 SVD 截断来初始化低秩因子（参见 LoftQ 等方法）

#### 1.7.3 特征值分解 (Eigendecomposition) vs SVD

| | 特征值分解 | SVD |
|--|-----------|-----|
| 适用范围 | 方阵（需可对角化） | 任意 $m \times n$ 矩阵 |
| 分解形式 | $A = Q \Lambda Q^{-1}$ | $W = U \Sigma V^\top$ |
| 对称矩阵 | $A = Q \Lambda Q^\top$，$Q$ 正交 | 同形式，奇异值 = \|特征值\|（PSD 时特征值 = 奇异值） |
| 值域 | 特征值可为负 | 奇异值 $\geq 0$ |

当 $A$ 对称正定 (SPD) 时，特征值分解与 SVD 等价。

#### 1.7.4 正定矩阵 (Positive Definite Matrix)

对称矩阵 $A$ 是正定的，充要条件：

1. 所有特征值 $> 0$
2. 所有顺序主子式 $> 0$（Sylvester 准则）
3. 存在 Cholesky 分解 $A = LL^\top$
4. $\forall x \neq 0: x^\top A x > 0$

#### 1.7.5 伪逆 (Moore–Penrose Pseudoinverse)

由 SVD 给出：$A^+ = V \Sigma^+ U^\top$，其中 $\Sigma^+$ 将非零奇异值取倒数。用于求解 $Ax = b$ 的最小范数最小二乘解。

线性回归的解析解：$\hat{\theta} = (X^\top X)^{-1} X^\top y = X^+ y$。当 $X^\top X$ 奇异时，加 L2 正则化（Ridge 回归）：$\hat{\theta} = (X^\top X + \lambda I)^{-1} X^\top y$，同时改善条件数 (condition number)。

#### 1.7.6 矩阵乘法计算复杂度

$A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$：$AB$ 需 $O(mkn)$ FLOPs。

**LLM FLOPs 估算**：对于参数量为 $N$ 的 Transformer 模型，训练 $D$ 个 token 的总计算量约为：

$$\text{FLOPs} \approx 6ND$$

（前向约 2ND，反向约 4ND；推理/仅前向时为 2ND。）此公式即 scaling law 文献中的经典估算来源。

---

### 1.8 微积分与梯度 / Calculus & Gradients

#### 1.8.1 Jacobian 与 Hessian

**Jacobian 矩阵**：向量函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 的一阶导数：

$$J_{ij} = \frac{\partial f_i}{\partial x_j}, \quad J \in \mathbb{R}^{m \times n}$$

**Hessian 矩阵**：标量函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的二阶导数：

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}, \quad H \in \mathbb{R}^{n \times n}$$

$H$ 对称。正定 → 局部极小；负定 → 局部极大；不定 → 鞍点 (saddle point)。

#### 1.8.2 链式法则与反向传播 (Backpropagation)

对于复合函数 $L = f(g(h(x)))$：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

矩阵形式（向量 → 向量），使用 **向量-Jacobian 乘积 (VJP)**：

$$\frac{\partial L}{\partial x} = J_h^\top J_g^\top \frac{\partial L}{\partial f}$$

反向传播的本质是从输出到输入高效计算 VJP 链式乘积，利用前向缓存的激活值。

---

### 1.9 信息论 / Information Theory

#### 1.9.1 熵 (Entropy)

$$H(X) = -\sum_x P(x) \log P(x)$$

度量随机变量的不确定性。均匀分布时 $H$ 最大（$= \log |X|$）；确定性分布时 $H = 0$。单位：$\log_2$ → bits，$\ln$ → nats（深度学习框架通常用 nats）。

#### 1.9.2 KL 散度 (Kullback–Leibler Divergence)

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = E_P\!\left[\log \frac{P}{Q}\right]$$

**核心性质**：
- $D_{\text{KL}} \geq 0$（由 Jensen 不等式证明），等号当且仅当 $P = Q$
- **非对称**：$D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$，不是距离度量
- **Forward KL** ($P \| Q$)：均值追逐 (mean-seeking)，$Q$ 倾向覆盖 $P$ 的全部支撑
- **Reverse KL** ($Q \| P$)：模式追逐 (mode-seeking)，$Q$ 倾向集中在 $P$ 的主模式

**ML 应用**：RLHF 中的 KL 惩罚 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 约束策略不偏离参考模型；DPO 目标中隐含 KL 约束。

#### 1.9.3 交叉熵 (Cross-Entropy)

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

与熵和 KL 散度的关系：

$$\boxed{H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)}$$

当 $P$ 为确定性分布（one-hot label）时，$H(P) = 0$，因此最小化交叉熵 $\Leftrightarrow$ 最小化 $D_{\text{KL}}(P \| Q)$ $\Leftrightarrow$ MLE。

**困惑度 (Perplexity, PPL)** 与交叉熵的关系：

$$\text{PPL} = \exp\!\big(H(P, Q)\big) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^N \log Q(x_i)\right)$$

#### 1.9.4 互信息 (Mutual Information)

$$I(X; Y) = D_{\text{KL}}\!\big(P(X,Y) \| P(X) P(Y)\big) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$$

度量 $X, Y$ 共享的信息量。$I(X;Y) = 0$ 当且仅当 $X \perp Y$。

**ML 应用**：表示学习中最大化 MI（如 InfoNCE loss 的对比学习）；特征选择中用 MI 筛选与标签最相关的特征。

#### 1.9.5 Jensen 不等式 (Jensen's Inequality)

对凸函数 $f$：$f(E[X]) \leq E[f(X)]$。

**核心应用**：
1. 证明 KL 散度非负：$D_{\text{KL}}(P\|Q) = E_P[-\log(Q/P)] \geq -\log E_P[Q/P] = 0$
2. 推导 ELBO：$\log P(x) = \log E_{z \sim Q}\!\left[\frac{P(x,z)}{Q(z)}\right] \geq E_Q\!\left[\log \frac{P(x,z)}{Q(z)}\right] = \text{ELBO}$

---

### 1.10 重要性采样 / Importance Sampling

当从目标分布 $P$ 直接采样困难时，从建议分布 (proposal) $Q$ 采样并加权修正：

$$E_P[f(x)] = E_Q\!\left[f(x) \frac{P(x)}{Q(x)}\right] \approx \frac{1}{N}\sum_{i=1}^N f(x_i) \frac{P(x_i)}{Q(x_i)}, \quad x_i \sim Q$$

权重 $w_i = P(x_i)/Q(x_i)$ 即 importance weight。当 $Q$ 与 $P$ 差异过大时，方差可能爆炸。

---

### 1.11 正态分布 KL 散度闭合解 / Closed-Form KL for Gaussians

对于 $P = \mathcal{N}(\mu_1, \Sigma_1)$，$Q = \mathcal{N}(\mu_2, \Sigma_2)$：

$$D_{\text{KL}}(P \| Q) = \frac{1}{2}\!\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1}(\mu_2 - \mu_1) - k + \ln\frac{|\Sigma_2|}{|\Sigma_1|}\right]$$

**VAE 中的特殊情况**：$Q(z) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$，$P(z) = \mathcal{N}(0, I)$：

$$D_{\text{KL}}(Q \| P) = -\frac{1}{2}\sum_{j=1}^d \!\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

### 1.12 变分推断与 ELBO / Variational Inference & ELBO

目标：推断后验 $P(z \mid x) = P(x, z) / P(x)$，但 $P(x) = \int P(x, z)\, dz$ 不可解。

引入变分分布 $Q(z)$，用 ELBO 下界代替：

$$\log P(x) \geq \underbrace{E_{Q}\!\left[\log \frac{P(x, z)}{Q(z)}\right]}_{\text{ELBO}} = E_Q[\log P(x \mid z)] - D_{\text{KL}}(Q(z) \| P(z))$$

等价关系：$\log P(x) = \text{ELBO} + D_{\text{KL}}(Q \| P(z \mid x))$。最大化 ELBO $\Leftrightarrow$ 最小化 $D_{\text{KL}}(Q \| P(z \mid x))$。

---

## Part 2: PyTorch 代码片段 / PyTorch Snippets

### 2.1 数值稳定的 Softmax / Numerically Stable Softmax

```python
import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)          # 减去最大值防止溢出
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

### 2.2 从零实现交叉熵损失 / Cross-Entropy Loss from Scratch

```python
import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (N, C) — 未归一化的分数
    targets: (N,)   — 整数类标签
    返回: 标量 loss
    """
    # 数值稳定的 log-softmax
    log_probs = logits - logits.max(dim=-1, keepdim=True).values
    log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    # 取出目标类对应的 log 概率
    nll = -log_probs[torch.arange(len(targets)), targets]
    return nll.mean()
```

### 2.3 KL 散度（VAE 场景）/ KL Divergence for VAE

```python
import torch

def kl_divergence_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    计算 KL( N(mu, exp(log_var)) || N(0, I) )
    mu, log_var: (N, d)
    返回: (N,) 每个样本的 KL 值
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
```

### 2.4 SVD 低秩近似 / SVD Low-Rank Approximation

```python
import torch

def low_rank_approx(W: torch.Tensor, r: int) -> torch.Tensor:
    """
    返回 W 的秩为 r 的最优 Frobenius 范数近似。
    W: (m, n), r: 目标秩
    """
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    # 保留前 r 个奇异值
    return (U[:, :r] * S[:r]) @ Vt[:r, :]

# 示例
W = torch.randn(128, 64)
W_approx = low_rank_approx(W, r=8)
print(f"近似误差 (Frobenius): {(W - W_approx).norm():.4f}")
```

### 2.5 LoRA 风格的低秩前向传播 / LoRA-Style Forward Pass

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """简化的 LoRA 线性层，冻结预训练权重，添加可训练低秩旁路。"""

    def __init__(self, in_dim: int, out_dim: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        # 预训练权重（冻结）
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W.weight.requires_grad_(False)
        # 低秩旁路 ΔW = B @ A
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        # 初始化：A 用 Kaiming，B 初始化为零 → 开始时 ΔW = 0
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x) + self.B(self.A(x)) * self.scaling
```

### 2.6 高斯 MLE / Gaussian MLE from Scratch

```python
import torch

def gaussian_mle(data: torch.Tensor):
    """
    data: (N, d)
    返回: (mu, var) — MLE 估计的均值和方差
    """
    mu = data.mean(dim=0)
    var = data.var(dim=0, unbiased=False)   # MLE 用 N（非 N-1）
    return mu, var

def gaussian_log_likelihood(data: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """计算高斯对数似然。"""
    d = data.shape[-1]
    return -0.5 * (d * torch.log(2 * torch.pi * var) + ((data - mu) ** 2) / var).sum(dim=-1).mean()
```

### 2.7 Monte Carlo 估计 KL 散度 / MC Estimation of KL Divergence

```python
import torch

def mc_kl_divergence(log_p, log_q, samples, num_samples: int = 10000):
    """
    通过 Monte Carlo 估计 KL(P || Q)。
    log_p, log_q: 可调用函数，输入样本返回 log 概率
    samples: 从 P 采样的样本
    """
    return (log_p(samples) - log_q(samples)).mean()

# 示例：估计 N(0,1) || N(1, 0.5) 的 KL
def log_p(x): return -0.5 * (x ** 2 + torch.log(torch.tensor(2 * 3.14159)))
def log_q(x): return -0.5 * ((x - 1) ** 2 / 0.5 + torch.log(torch.tensor(2 * 3.14159 * 0.5)))

samples = torch.randn(100000)
print(f"MC KL estimate: {mc_kl_divergence(log_p, log_q, samples):.4f}")
```

---

## Part 3: 面试题库 / Interview Questions (25 Questions)

### L1 — 基础 / Basic

---

<details>
<summary>Q1. 条件概率、联合概率、边缘概率之间的关系是什么？</summary>

**A**：联合概率 $P(A,B) = P(A \mid B) P(B) = P(B \mid A) P(A)$；边缘概率通过对联合概率求和（或积分）得到：$P(A) = \sum_B P(A, B)$。贝叶斯定理 $P(A \mid B) = P(B \mid A) P(A) / P(B)$ 将三者联系起来。

**追问**：联合概率的链式法则如何推广到三个以上变量？条件独立 $A \perp B \mid C$ 意味着什么？
> **答**：$P(A,B,C) = P(A)P(B|A)P(C|A,B)$。$A \perp B \mid C$ 意味着给定 $C$ 后，知道 $A$ 不改变对 $B$ 的信念：$P(A,B|C) = P(A|C)P(B|C)$。在概率图模型中对应 $C$ 阻断了 $A$ 到 $B$ 的所有路径。

</details>

---

<details>
<summary>Q2. 期望和方差有哪些基本运算法则？</summary>

**A**：
- 线性：$E[aX+b] = aE[X]+b$
- 方差：$\text{Var}(X) = E[X^2] - (E[X])^2$
- 缩放：$\text{Var}(aX+b) = a^2\text{Var}(X)$
- 独立变量之和的方差等于方差之和（无协方差项）

**追问**：协方差矩阵的对角元素是什么？为什么协方差矩阵一定是半正定的？
> **答**：对角元素是各维度的方差 $\text{Var}(X_i)$。对称性显然；半正定性是因为 $\forall w: w^\top \Sigma w = \text{Var}(w^\top X) \geq 0$（方差非负）。

</details>

---

<details>
<summary>Q3. 列举至少四种常见概率分布及其均值、方差和典型应用场景。</summary>

**A**：

| 分布 | 均值 | 方差 | 应用 |
|------|------|------|------|
| 伯努利 $p$ | $p$ | $p(1-p)$ | 二分类标签 |
| 高斯 $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | 噪声模型、权重先验 |
| 泊松 $\lambda$ | $\lambda$ | $\lambda$ | 稀有事件计数 |
| 均匀 $[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ | 参数初始化 |

**追问**：多维高斯分布的 PDF 如何写？
> **答**：$\mathcal{N}(x|\mu,\Sigma) = (2\pi)^{-d/2}|\Sigma|^{-1/2}\exp\!\big(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$。

</details>

---

<details>
<summary>Q4. 什么是最大似然估计 (MLE)？请给出数学定义。</summary>

**A**：

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log P(x_i \mid \theta)$$

即选择使观测数据出现概率最大的参数。等价于最小化负对数似然 (NLL)。对于高斯假设下的回归，MLE 等价于最小化 MSE。

**追问**：MLE 在什么条件下会给出有偏估计？
> **答**：典型例子是高斯方差的 MLE（除以 $N$ 而非 $N-1$）是有偏的（低估方差）。小样本下 MLE 对复杂模型容易过拟合。

</details>

---

<details>
<summary>Q5. 矩阵的秩 (Rank) 是什么？低秩矩阵有什么特性？</summary>

**A**：矩阵的秩是其线性无关行（或列）的最大数目，等于列空间的维数。低秩矩阵 $W \in \mathbb{R}^{m \times n}$（秩 $r \ll \min(m,n)$）可分解为 $W = AB$（$A \in \mathbb{R}^{m \times r}, B \in \mathbb{R}^{r \times n}$），参数量从 $mn$ 降到 $r(m+n)$。

**追问**：低秩分解与参数高效微调 (PEFT) 有什么关系？
> **答**：LoRA 假设预训练权重的更新 $\Delta W$ 是低秩的，用 $\Delta W = BA$ 的低秩分解来近似，只训练 $A, B$ 而冻结原始权重，大幅减少可训练参数和显存开销。

</details>

---

<details>
<summary>Q6. 熵 (Entropy) 的定义和直觉是什么？</summary>

**A**：$H(X) = -\sum_x P(x) \log P(x)$，度量随机变量的不确定性。均匀分布时熵最大（$= \log |X|$），确定性分布时熵为 0。单位取决于对数底：$\log_2$ → bits，$\ln$ → nats。

**追问**：为什么最大熵原理可以推导出高斯分布？
> **答**：在给定均值和方差的约束下，最大化熵 $H(X)$ 受约束 $\int P = 1$、$E[X] = \mu$、$\text{Var}(X) = \sigma^2$，用拉格朗日乘子法求解，得到的最优分布恰好是高斯分布。

</details>

---

<details>
<summary>Q7. 点积 (Dot Product) 和余弦相似度 (Cosine Similarity) 的区别？</summary>

**A**：
- 点积：$u \cdot v = \|u\|\|v\|\cos\theta$，同时受向量幅度和方向影响
- 余弦相似度：$\cos\theta = \frac{u \cdot v}{\|u\|\|v\|}$，只反映方向（角度），幅度归一化

**追问**：Attention 中为什么用缩放点积而不用余弦相似度？
> **答**：缩放点积 $QK^\top / \sqrt{d_k}$ 计算高效（矩阵乘法），且 Q/K 的 norm 大小也参与注意力权重计算，增加了表达能力。RoPE 旋转保持 norm 不变，只编码相对位置信息。用余弦需要额外的 L2 归一化，会限制表达力。

</details>

---

<details>
<summary>Q8. Jacobian 矩阵和 Hessian 矩阵分别是什么？</summary>

**A**：
- **Jacobian** $J \in \mathbb{R}^{m \times n}$：向量函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 的一阶导数矩阵，$J_{ij} = \partial f_i / \partial x_j$
- **Hessian** $H \in \mathbb{R}^{n \times n}$：标量函数的二阶导数矩阵，$H_{ij} = \partial^2 f / \partial x_i \partial x_j$，对称。正定 → 局部极小；不定 → 鞍点

**追问**：为什么在大模型优化中很少直接用 Hessian？
> **答**：Hessian 大小为 $n \times n$（$n$ 为参数量），大模型参数量达数十亿，存储和计算 Hessian 不现实。K-FAC、Shampoo 等二阶方法用 Kronecker 分解或对角块近似 Hessian 以降低开销。

</details>

---

### L2 — 中等 / Intermediate

---

<details>
<summary>Q9. MLE 和 MAP 的区别是什么？正则化如何对应贝叶斯先验？</summary>

**A**：MLE 最大化似然 $P(D|\theta)$；MAP 最大化后验 $P(\theta|D) \propto P(D|\theta) P(\theta)$，多了一项先验 $\log P(\theta)$。Gaussian 先验 → L2 正则化（Weight Decay）；Laplace 先验 → L1 正则化（Lasso）。

**追问**：AdamW 的 weight decay 从贝叶斯角度看相当于假设了什么先验？
> **答**：等价于各向同性高斯先验 $P(\theta) \propto \exp(-\lambda \|\theta\|^2 / 2)$。AdamW 将 weight decay 从梯度中解耦（decoupled），与带 L2 正则的 Adam 有细微差异，但贝叶斯解释相同。

</details>

---

<details>
<summary>Q10. 偏差-方差分解 (Bias-Variance Decomposition) 是什么？</summary>

**A**：预测误差 = 偏差² + 方差 + 不可约噪声。高偏差 → 欠拟合；高方差 → 过拟合。增加模型容量降低偏差但可能增加方差，反之亦然，这就是 bias-variance tradeoff。

**追问**：集成方法（Bagging、Dropout as ensemble）主要降低的是偏差还是方差？
> **答**：主要降低方差。Bagging 对多个高方差模型取平均，方差按 $1/M$ 缩小（$M$ 为模型数），偏差几乎不变。Boosting 则主要降低偏差。

</details>

---

<details>
<summary>Q11. SVD 分解是什么？如何用于低秩近似？</summary>

**A**：任意矩阵 $W = U\Sigma V^\top$，其中 $U, V$ 正交，$\Sigma$ 为奇异值对角阵。Eckart–Young 定理表明，保留前 $r$ 个奇异值的截断 SVD $\hat{W}_r = U_r \Sigma_r V_r^\top$ 是秩为 $r$ 的最优 Frobenius 范数近似。

**追问**：Frobenius 范数和谱范数分别由奇异值如何确定？
> **答**：$\|W\|_F = \sqrt{\sum_i \sigma_i^2}$（所有奇异值的平方和的根），$\|W\|_2 = \sigma_1$（最大奇异值）。

</details>

---

<details>
<summary>Q12. 特征值分解 (Eigendecomposition) 和 SVD 有什么区别？</summary>

**A**：特征值分解仅适用于方阵（且需可对角化），$A = Q\Lambda Q^{-1}$；SVD 适用于任意形状矩阵，$W = U\Sigma V^\top$，奇异值始终非负。对称正定矩阵的特征值分解与 SVD 等价。

**追问**：协方差矩阵的特征向量有什么几何意义？
> **答**：协方差矩阵的特征向量是数据分布的主轴方向（principal directions），对应 PCA 的主成分方向。特征值大小表示该方向的方差。

</details>

---

<details>
<summary>Q13. KL 散度的定义和关键性质是什么？</summary>

**A**：$D_{\text{KL}}(P\|Q) = E_P[\log(P/Q)]$。性质：(1) $D_{\text{KL}} \geq 0$；(2) 非对称，不是距离度量；(3) $D_{\text{KL}} = 0 \Leftrightarrow P = Q$。

**追问**：Forward KL 和 Reverse KL 在拟合行为上有什么不同？
> **答**：Forward KL ($P \| Q$) 是均值追逐 (mean-seeking)，$Q$ 倾向覆盖 $P$ 的所有高概率区域，适合保持多样性。Reverse KL ($Q \| P$) 是模式追逐 (mode-seeking)，$Q$ 倾向集中在 $P$ 的单一峰上，可能忽略其他模式。变分推断通常用 Reverse KL。

</details>

---

<details>
<summary>Q14. 交叉熵、KL 散度、熵之间有什么关系？</summary>

**A**：$H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$。当 $P$ 是 one-hot 分布时 $H(P) = 0$，最小化交叉熵 = 最小化 KL 散度 = MLE。

**追问**：语言模型的困惑度 (PPL) 和交叉熵的关系是什么？
> **答**：$\text{PPL} = \exp(H(P, Q))$，其中 $H$ 是交叉熵。PPL 可理解为模型在每个 token 位置平均的有效选择数；PPL 越低，模型越好。

</details>

---

<details>
<summary>Q15. Jensen 不等式是什么？在 ML 中有哪些应用？</summary>

**A**：对凸函数 $f$：$f(E[X]) \leq E[f(X)]$。应用：(1) 证明 KL 散度非负；(2) 推导变分下界 ELBO；(3) EM 算法的收敛性证明。

**追问**：什么是变分推断 (VI)？ELBO 和 KL 散度的关系？
> **答**：VI 用简单分布 $Q(z)$ 近似复杂后验 $P(z|x)$。$\log P(x) = \text{ELBO} + D_{\text{KL}}(Q \| P(z|x))$，最大化 ELBO 等价于最小化近似后验与真实后验的 KL 散度。

</details>

---

<details>
<summary>Q16. 正定矩阵 (Positive Definite Matrix) 的充要条件有哪些？</summary>

**A**：对称矩阵 $A$ 正定的充要条件：(1) 所有特征值 $> 0$；(2) 所有顺序主子式 $> 0$（Sylvester 准则）；(3) 存在 Cholesky 分解 $A = LL^\top$；(4) $\forall x \neq 0: x^\top Ax > 0$。

**追问**：为什么协方差矩阵一定是半正定的？
> **答**：对任意 $w$，$w^\top \Sigma w = \text{Var}(w^\top X) \geq 0$（方差非负）。正定的额外要求是方差严格大于零（没有确定性线性关系）。

</details>

---

<details>
<summary>Q17. 矩阵乘法的计算复杂度是多少？LLM 训练的 FLOPs 如何估算？</summary>

**A**：$A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$：$AB$ 需 $O(mkn)$ FLOPs。LLM 训练估算：参数量 $N$，训练 token 数 $D$，总 FLOPs $\approx 6ND$（前向约 2ND，反向约 4ND；推理/仅前向时为 2ND）。

**追问**：为什么 LLM 推理的 decode 阶段通常是 memory-bound 而非 compute-bound？
> **答**：自回归 decode 每步只生成 1 个 token，batch 维度退化，矩阵乘法变成矩阵-向量乘，FLOPs 很小但需加载全部模型权重（IO 密集），GPU 算力未被充分利用。

</details>

---

### L3 — 深度 / Deep

---

<details>
<summary>Q18. 中心极限定理 (CLT) 与 SGD 的噪声有什么关系？SGD 噪声为什么有正则化效果？</summary>

**A**：CLT 保证 mini-batch 梯度 $\hat{g}_B = \frac{1}{B}\sum_i \nabla L(x_i)$ 是总体梯度的无偏估计，方差 $\propto 1/B$。SGD 的梯度噪声帮助模型逃离尖锐极小值 (sharp minima)，倾向于收敛到平坦极小值 (flat minima)。平坦极小值通常泛化更好，这相当于一种隐式正则化 (implicit regularization)。

**追问**：增大 batch size 对训练有什么影响？
> **答**：减小梯度方差（$\propto 1/B$），优化更稳定但可能丧失正则化效果，导致泛化能力下降。需要配合学习率调整（线性缩放规则）。

</details>

---

<details>
<summary>Q19. Forward KL 和 Reverse KL 的拟合行为有什么根本差异？各用在什么场景？</summary>

**A**：
- **Forward KL** ($D_{\text{KL}}(P \| Q)$，I-projection)：最小化时 $Q$ 试图在 $P > 0$ 的所有区域都有质量 → mean-seeking / zero-avoiding。适合保持多模态覆盖。
- **Reverse KL** ($D_{\text{KL}}(Q \| P)$，M-projection)：最小化时 $Q$ 在 $P \approx 0$ 的区域尽量没有质量 → mode-seeking / zero-forcing。适合集中在主模式上。

变分推断 (VI) 默认用 Reverse KL（因为 ELBO 推导自然产生）；RLHF 中 KL 约束 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 按上面的约定是 **Reverse KL**（mode-seeking）：把策略 $\pi_\theta$ 拉向参考、惩罚其偏离 $\pi_{\text{ref}}$ 高概率区域，抑制 reward hacking 式的分布漂移。

**追问**：混合 Forward/Reverse KL 有什么实际应用？
> **答**：如 Rényi-$\alpha$ 散度家族在 $\alpha \in (0,1)$ 时介于二者之间，被用于变分推断和强化学习中，在覆盖率和集中度之间做折中。

</details>

---

<details>
<summary>Q20. Attention 中为什么用缩放点积 (Scaled Dot-Product) 而不用余弦相似度？</summary>

**A**：缩放点积 $\text{softmax}(QK^\top / \sqrt{d_k})$ 优点：(1) 矩阵乘法高效；(2) Q/K 的 norm 大小参与注意力权重计算，增加表达能力；(3) 缩放因子 $\sqrt{d_k}$ 防止点积值过大导致 softmax 饱和。余弦相似度会丢失幅度信息，且需额外归一化计算。

RoPE (Rotary Position Embedding) 旋转编码只改变 Q/K 的方向（角度），保持 norm 不变，因此位置信息通过点积中的角度差自然融入。

**追问**：如果对 Q, K 做 L2 归一化后再算点积，效果如何？
> **答**：等价于余弦相似度（乘以温度缩放后）。一些工作（如 Normalized Attention）探索了这个方向，优点是注意力权重与 norm 无关、更稳定，但可能限制了不同 token 之间差异化表达的能力。CosFormer 等方法在保持线性注意力效率的同时引入余弦重加权。

</details>

---

<details>
<summary>Q21. 反向自动微分 (Reverse-Mode AD) 和前向自动微分 (Forward-Mode AD) 各在什么场景更高效？</summary>

**A**：
- **反向模式** (reverse-mode AD)：一次前向 + 一次反向可计算标量损失对所有参数的梯度，复杂度 $O(1) \times$ 前向计算。适合**参数多、输出少**（即 $f: \mathbb{R}^n \to \mathbb{R}$）的场景，是深度学习反向传播的基础。
- **前向模式** (forward-mode AD)：每次沿一个输入方向传播，得到输出对一个输入的梯度。适合**输入少、输出多**（即 $f: \mathbb{R} \to \mathbb{R}^m$）的场景，如计算 Jacobian-vector product (JVP)。

**追问**：HVP (Hessian-vector product) 可以不显式构造 Hessian 来计算吗？
> **答**：可以。$\nabla^2 f \cdot v = \nabla(\nabla f \cdot v)$，先对 $\nabla f \cdot v$ 做前向模式微分（或对 $\nabla f$ 做反向传播），只需两次微分运算，复杂度与一次梯度计算相当。这是 CG (Conjugate Gradient) 等隐式二阶方法的基础。

</details>

---

<details>
<summary>Q22. 重要性采样 (Importance Sampling) 在 PPO 中如何应用？PPO-Clip 中的 $\epsilon$ 限制的是什么？</summary>

**A**：PPO 复用旧策略 $\pi_{\theta_{\text{old}}}$ 收集的轨迹数据训练新策略 $\pi_\theta$，通过 importance ratio $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ 修正分布偏移。PPO-Clip 的目标函数：

$$L^{\text{CLIP}} = E_t\!\Big[\min\!\big(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]$$

$\epsilon$（通常 0.1~0.2）限制的是 importance ratio 的范围（而非权重本身）。当 $r_t$ 超出 $[1-\epsilon, 1+\epsilon]$ 时梯度被截断，防止过大的策略更新导致训练不稳定。

**追问**：如果 importance ratio 方差过大，有什么替代方案？
> **答**：可用 V-trace（在 IMPALA 中提出）对 ratio 做截断，或使用 Retrace($\lambda$) 等方法控制方差。核心思想是限制有效 importance weight 以保证收敛稳定性。

</details>

---

<details>
<summary>Q23. 变分推断 (Variational Inference) 和 ELBO 是什么？如何推导？</summary>

**A**：用简单分布 $Q(z)$ 近似真实后验 $P(z|x)$。推导：

$$\log P(x) = \log \int P(x,z)\,dz = \log E_Q\!\left[\frac{P(x,z)}{Q(z)}\right] \geq E_Q\!\left[\log\frac{P(x,z)}{Q(z)}\right] = \text{ELBO}$$

不等式由 Jensen 不等式给出（$\log$ 是凹函数）。ELBO = 重构项 $E_Q[\log P(x|z)]$ − 正则项 $D_{\text{KL}}(Q(z) \| P(z))$。$\beta$-VAE 通过超参数 $\beta$ 调节 KL 项权重，控制解耦程度和重构质量之间的 tradeoff。

**追问**：ELBO 为什么要最大化（而不是最小化）？
> **答**：ELBO 是 $\log P(x)$ 的下界。最大化 ELBO → 更紧的下界 → $Q(z)$ 更接近真实后验 $P(z|x)$。同时 $\log P(x) = \text{ELBO} + D_{\text{KL}}(Q \| P(z|x))$，因为 KL $\geq 0$，最大化 ELBO 相当于最小化变分后验与真实后验的 KL 散度。

</details>

---

<details>
<summary>Q24. 低秩假设在参数高效微调 (PEFT) 中的理论和实验依据是什么？</summary>

**A**：理论依据——内在维度 (Intrinsic Dimensionality) 的研究表明，预训练模型的 fine-tuning 实际上在一个远低于参数空间维度的低维子空间中进行。LoRA 将权重更新参数化为 $\Delta W = BA$（$B \in \mathbb{R}^{m \times r}, A \in \mathbb{R}^{r \times n}$），只需训练 $r(m+n)$ 个参数（$r \ll \min(m,n)$），比原始 $mn$ 参数量少几个数量级。实验上，$r = 4 \sim 64$ 即可在多种下游任务上达到接近全参微调的效果。

**追问**：如何选择 LoRA 的秩 $r$ 和哪些层应用 LoRA？
> **答**：秩 $r$ 通常通过在验证集上做小规模搜索确定。实践中应用到注意力层的 Q/K/V/O 投影矩阵效果较好。更进阶的方法（如 AdaLoRA）根据各层奇异值分布自适应分配不同的秩。秩 $r$ 的选择需要在表达能力和参数效率之间权衡。

</details>

---

<details>
<summary>Q25. 正态分布 KL 散度的闭合解是什么？在 VAE 中如何使用？</summary>

**A**：对于两个多元高斯 $P = \mathcal{N}(\mu_1, \Sigma_1)$，$Q = \mathcal{N}(\mu_2, \Sigma_2)$：

$$D_{\text{KL}}(P \| Q) = \frac{1}{2}\!\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^\top\Sigma_2^{-1}(\mu_2-\mu_1) - k + \ln\frac{|\Sigma_2|}{|\Sigma_1|}\right]$$

VAE 中 $Q(z) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$，$P(z) = \mathcal{N}(0, I)$，简化为：

$$D_{\text{KL}} = -\frac{1}{2}\sum_j (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

这个 KL 项作为正则项出现在 ELBO 中，鼓励编码器输出接近标准正态先验。

**追问**：$\beta$-VAE 中 $\beta > 1$ 有什么效果？
> **答**：增大 KL 项权重 → 编码器被迫更紧密地匹配标准正态先验 → 学到的隐变量 $z$ 更解耦 (disentangled)，但重构质量可能下降。$\beta < 1$ 则相反，允许更灵活的隐空间但可能牺牲解耦性。

</details>

---

*Last updated: 2025 | 本手册仅供学习参考，公式以原始文献为准。*

## 更多 L3 深挖 / Extended L3

<details>
<summary>Q26: Fisher 信息矩阵 (Fisher Information Matrix, FIM) 与自然梯度 (Natural Gradient) 有什么内在联系？为什么说 natural gradient 是 KL 意义下的最速下降？</summary>

  **答**：Fisher 信息矩阵定义为对数似然梯度的协方差：$F(\theta) = E_{x \sim p_\theta}\!\left[\nabla_\theta \log p_\theta(x)\, \nabla_\theta \log p_\theta(x)^\top\right]$。它同时也是对数似然 Hessian 的负期望（在 MLE 处），并与 KL 散度的局部二次展开密切相关：$D_{\text{KL}}(p_\theta \| p_{\theta + \delta}) \approx \frac{1}{2}\delta^\top F(\theta)\,\delta$。因此 FIM 是 KL 散度在参数空间的局部度量张量 (metric tensor)。自然梯度定义为 $\tilde{\nabla} = F^{-1}\nabla L$，即在 KL 球约束 $\{δ : D_{\text{KL}}(p_\theta \| p_{\theta+\delta}) \leq \epsilon\}$ 下使损失下降最快的方向。直观理解：普通梯度在欧氏空间中最速下降，但参数空间中等距移动并不对应等概率分布变化；natural gradient 用 FIM 修正了这个"扭曲"，使优化步长在分布空间中是均匀的。

  **追问**：为什么实际训练中通常不直接使用 natural gradient？近似方法有哪些？（提示：K-FAC、EKFAC 对 FIM 进行分块对角近似，降低求逆的计算开销。）

</details>

---

<details>
<summary>Q27: 重参数化技巧 (Reparameterization Trick) 的数学本质是什么？为什么它能降低梯度估计的方差？</summary>

  **答**：在 VAE 中，ELBO 的期望 $E_{z \sim q_\phi(z|x)}[\cdot]$ 的梯度无法直接通过采样节点反向传播（采样操作不可微）。重参数化技巧将 $z \sim q_\phi = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$ 改写为 $z = \mu_\phi + \sigma_\phi \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。这样随机性来源从参数 $\phi$ 转移到了无参数的噪声 $\epsilon$，而 $z$ 关于 $\phi$ 是确定性可微的，梯度可以通过标准反向传播传递到 $\mu_\phi$ 和 $\sigma_\phi$。相比 score function estimator（REINFORCE）$\nabla_\phi E_q[f(z)] = E_q[f(z)\nabla_\phi \log q_\phi(z)]$，重参数化将梯度信息直接编码到计算图中，不依赖 $f(z)$ 作为标量乘子，因此方差显著更低。本质上，score function estimator 只用 $f(z)$ 的标量值作为权重，而重参数化利用了 $f$ 对 $z$ 的局部梯度 $\partial f / \partial z$，提供了更丰富的信号。

  **追问**：对于离散隐变量（如离散 VAE），重参数化技巧不直接适用，有哪些替代方案？（提示：Gumbel-Softmax / Concrete 分布用连续松弛近似离散采样，保持可微性。）

</details>

---

<details>
<summary>Q28: 矩阵的谱范数 (Spectral Norm) $\|W\|_2 = \sigma_1$ 如何控制神经网络的 Lipschitz 常数？这对训练稳定性有何意义？</summary>

  **答**：一个函数 $f$ 的 Lipschitz 常数 $L$ 满足 $\|f(x) - f(y)\| \leq L\|x - y\|$。对于线性层 $f(x) = Wx$，Lipschitz 常数恰好等于 $\|W\|_2 = \sigma_1$（最大奇异值）。对于 $k$ 层网络 $f = W_k \cdots W_1$，整体 Lipschitz 常数 $L \leq \prod_i \|W_i\|_2$，即各层谱范数之积。如果 $L$ 过大，输入的微小扰动会在前向传播中指数放大（exploding activations），梯度也会在反向传播中指数放大（exploding gradients），导致训练不稳定。Spectral normalization（将每层权重除以其谱范数：$\hat{W} = W / \sigma_1$）将每层 Lipschitz 常数约束为 1，广泛用于 GAN 的判别器训练以防止模式崩溃 (mode collapse)。谱范数可通过幂迭代 (power iteration) 高效近似，无需完整 SVD。

  **追问**：为什么 GAN 训练中对判别器施加 Lipschitz 约束（如 spectral normalization 或 gradient penalty）是必要的？如果不约束会出现什么问题？（提示：Wasserstein 距离要求判别器属于 1-Lipschitz 函数类，否则目标无界。）

</details>

---

<details>
<summary>Q29: 矩阵求迹技巧 (Trace Trick) 的核心等式是什么？它在推导哪些 ML 公式时不可或缺？</summary>

  **答**：核心等式是 $\text{tr}(AB) = \text{tr}(BA)$（循环置换不变性）以及标量等于其自身的迹：$x^\top A x = \text{tr}(x^\top A x) = \text{tr}(A x x^\top)$。这使得对含矩阵乘积的标量函数求导变得可行。典型应用：(1) 线性回归的矩阵微分 $\nabla_W \|Y - XW\|_F^2 = \nabla_W \text{tr}((Y-XW)^\top(Y-XW))$，展开后利用 $\nabla_W \text{tr}(AW) = A^\top$ 得到闭合解；(2) 多元高斯对数似然中 $\log |\Sigma|$ 和 $x^\top \Sigma^{-1} x$ 的求导；(3) 协方差矩阵的 MLE 推导 $\hat{\Sigma} = \frac{1}{N}\sum_i (x_i - \mu)(x_i - \mu)^\top$；(4) 线性动态系统 (Kalman filter) 中矩阵 Riccati 方程的推导。总之，任何涉及矩阵二次型求导的场景，trace trick 都是核心工具。

  **追问**：试用 trace trick 推导多元高斯 $\mathcal{N}(\mu, \Sigma)$ 关于 $\Sigma^{-1}$（精度矩阵 $\Lambda$）的 MLE。提示：对数似然中含 $\text{tr}(\Lambda S)$ 和 $\log|\Lambda|$ 两项。（答：$\hat{\Lambda} = S^{-1}$，即 MLE 精度矩阵为样本协方差的逆。）

</details>

---

<details>
<summary>Q30: Schur 补 (Schur Complement) 与多元高斯的条件分布 (Conditional Distribution) 之间有什么精确的数学对应？</summary>

  **答**：设联合分布 $(x_a, x_b) \sim \mathcal{N}(\mu, \Sigma)$，将协方差矩阵分块为 $\Sigma = \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}$。条件分布 $P(x_a \mid x_b)$ 仍为高斯，其精度矩阵（逆协方差）中 $\Sigma_{aa|b}^{-1}$ 的左上角块恰好是联合精度矩阵 $\Sigma^{-1}$ 的对应块 $\Sigma^{aa}$，这就是 Schur 补：$\Sigma^{aa} = (\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})^{-1}$。条件均值为 $\mu_{a|b} = \mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b)$。这个对应有深远意义：(1) 高斯图模型 (Gaussian Markov Random Field) 中，精度矩阵的零元素对应条件独立关系，比协方差矩阵更直接编码了条件结构；(2) 高斯过程 (GP) 的后验预测本质上就是条件高斯，利用 Schur 补给出闭合解；(3) Kalman filter 的更新步也可以从 Schur 补角度理解。

  **追问**：为什么在高斯图模型中用精度矩阵而不是协方差矩阵来编码图结构？条件独立与精度矩阵零元素的关系是什么？（答：$x_i \perp x_j \mid \text{rest}$ 当且仅当 $\Sigma^{-1}_{ij} = 0$，即精度矩阵的非零模式直接对应图的边结构。）

</details>

---

<details>
<summary>Q31: Wasserstein 距离 (Wasserstein Distance) 与 KL 散度在度量分布差异时有何本质区别？为什么 WGAN 要用 Wasserstein 而非 KL？</summary>

  **答**：KL 散度 $D_{\text{KL}}(P \| Q)$ 在两个分布的支撑不重叠时为无穷大（即使它们在几何上很"近"），这在高维空间中几乎总是成立，因为两个低维流形支撑集重叠的概率极低。Wasserstein 距离（推土机距离, Earth Mover's Distance）基于最优传输 (optimal transport)，定义为 $W(P, Q) = \inf_{\gamma \in \Pi(P,Q)} E_{(x,y)\sim\gamma}[\|x - y\|]$，即使支撑不重叠也能给出有意义的有限距离。直观类比：KL 只看比值 $P/Q$（如果 $Q$ 在 $P$ 有质量的地方为 0，就"爆炸"了），Wasserstein 看把一堆"土"搬到另一堆需要的最少"工作量"。因此 Wasserstein 在分布不重叠时仍有平滑的梯度信号，而 KL 或 JS 散度会导致梯度消失。WGAN 用 Wasserstein-1 距离作为生成器目标，通过 Kantorovich-Rubinstein 对偶转化为对 1-Lipschitz 判别器的优化，解决了经典 GAN 训练不稳定的问题。

  **追问**：Wasserstein 距离的计算代价通常高于 KL 散度，在高维情况下如何近似？（提示：Sinkhorn 算法在最优传输中加入熵正则化，将线性规划转化为可并行化的矩阵缩放迭代；Sliced Wasserstein 通过随机投影将高维问题降为多个一维问题。）

</details>

---

<details>
<summary>Q32: Hessian 矩阵的谱性质 (spectrum) 如何刻画损失函数的局部几何结构？这对理解优化中的鞍点和逃离策略有何帮助？</summary>

  **答**：在临界点（梯度为零处），Hessian $H$ 的特征值分布决定了局部几何：(1) 所有特征值 $> 0$（正定）→ 局部极小，曲面呈碗状；(2) 所有特征值 $< 0$（负定）→ 局部极大；(3) 有正有负（不定）→ 鞍点，沿负曲率方向是下降方向。在高维参数空间中，鞍点远多于局部极小（因为特征值全部为正的"概率"指数衰减），这是非凸优化的主要障碍。逃离鞍点的策略：(1) SGD 的梯度噪声自然提供沿负曲率方向的扰动，帮助逃离——这是 SGD 噪声的隐式正则化效应的几何解释；(2) 动量 (momentum) 帮助穿越平坦区域；(3) 显式方法如添加 Hessian 负曲率方向的扰动（尚不常用但理论上有效）。Hessian 的特征值谱还与泛化相关：极小值处 Hessian 的特征值越大（曲面越"尖锐"），该极小值的泛化能力通常越差——这启发了 sharpness-aware minimization (SAM) 等方法。

  **追问**：为什么在极高维空间中，鞍点问题比局部极小问题更严重？如何从 Hessian 的特征值分布来解释？（答：在 $n$ 维空间中，Hessian 有 $n$ 个特征值；在随机临界点处，每个特征值等概率正或负，全部为正的概率为 $2^{-n}$，指数级小，因此几乎任何临界点都是鞍点而非极小值。）

</details>