# Coding & ML 实现面试速查手册
# Coding & ML Implementation — Interview Cheat Sheet

> **面向读者：** Research / ML Intern 候选人
> **使用说明：** 中文为主，关键术语附英文；公式用 LaTeX；代码均为从零实现

---

## 一、核心概念与公式推导 | Concepts & Formula Derivations

### 1.1 Softmax 函数

将任意向量映射为概率分布（probability distribution）：

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

**数值稳定版（Numerically Stable）：** 令 $m = \max(z)$，则

$$
\text{softmax}(z_i) = \frac{e^{z_i - m}}{\sum_{j=1}^{C} e^{z_j - m}}
$$

数学上等价，但避免了 $e^{z_i}$ 的浮点溢出（overflow）。

### 1.2 交叉熵损失 | Cross-Entropy Loss

给定 logits $\mathbf{z} \in \mathbb{R}^C$ 和真实标签 $y$：

$$
\mathcal{L} = -\log \text{softmax}(z_y) = -z_y + \log \sum_{j=1}^{C} e^{z_j}
$$

上式右侧即 **LogSumExp** trick：$\log \sum e^{z_j} = m + \log \sum e^{z_j - m}$，其中 $m = \max(\mathbf{z})$。

PyTorch 中 `nn.CrossEntropyLoss` 内部组合了 `LogSoftmax` + `NLLLoss`，直接接收 **logits** 而非概率。

### 1.3 Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

| 符号 | 含义 | 维度 |
|------|------|------|
| $Q$ | Query 查询矩阵 | $(N, d_k)$ |
| $K$ | Key 键矩阵 | $(N, d_k)$ |
| $V$ | Value 值矩阵 | $(N, d_v)$ |
| $d_k$ | Key 维度 | 标量 |

**为什么要除以 $\sqrt{d_k}$？** 当 $d_k$ 较大时，$QK^\top$ 的方差随 $d_k$ 线性增长，导致 softmax 输出趋向 one-hot，梯度趋近于零。缩放使方差稳定在 $\mathcal{O}(1)$。

### 1.4 Multi-Head Attention | 多头注意力

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$
$$
\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V)
$$

其中 $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$，$d_k = d_v = d_{\text{model}} / h$。

**多头的意义：** 不同 head 可以关注不同子空间的模式（如语法关系、语义关系），总计算量与单头相当。

### 1.5 Layer Normalization

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$

其中 $\mu = \frac{1}{d}\sum_{i=1}^d x_i$，$\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$，沿 **特征维度** $d$ 计算。

**RMSNorm（Root Mean Square Normalization）：** 去掉均值中心化，仅保留缩放：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}, \quad y_i = \gamma \cdot \frac{x_i}{\text{RMS}(x) + \epsilon}
$$

实践中效果接近 LayerNorm，但计算更快（省去均值计算）。

### 1.6 LoRA（Low-Rank Adaptation）| 低秩适配

冻结预训练权重 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，注入可训练的低秩分解：

$$
W = W_0 + \Delta W = W_0 + BA
$$

其中 $A \in \mathbb{R}^{r \times d_{\text{in}}}$，$B \in \mathbb{R}^{d_{\text{out}} \times r}$，$r \ll \min(d_{\text{in}}, d_{\text{out}})$。

带缩放因子 $\alpha$ 的前向传播：

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

**初始化策略：** $A \sim \mathcal{N}(0, \sigma^2)$，$B = \mathbf{0}$。这样训练开始时 $\Delta W = BA = \mathbf{0}$，模型行为与预训练完全一致。

### 1.7 Top-p（Nucleus）采样

对 logits $\mathbf{z}$ 经温度缩放后的概率分布 $p_i = \text{softmax}(z_i / T)$：

1. 将 token 按概率降序排列
2. 计算累积概率（cumulative probability）
3. 保留累积概率首次超过阈值 $p$ 的所有 token（含第一个超出的）
4. 在保留集合中重新归一化后采样

**Temperature 的作用：** $T \to 0$ 退化为贪心（greedy），$T = 1$ 保持原分布，$T > 1$ 使分布更平坦。

### 1.8 K-Means 聚类

**目标函数（WCSS, Within-Cluster Sum of Squares）：**

$$
\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
$$

**算法：** 交替执行 (1) 分配步（将每个点分给最近的 centroid）和 (2) 更新步（将 centroid 移至簇内均值）。

- 时间复杂度：$\mathcal{O}(nKdi)$，其中 $n$ = 样本数，$K$ = 簇数，$d$ = 维度，$i$ = 迭代次数
- **K-Means++** 初始化：按概率 $\propto D(x)^2$ 顺序选初始 centroid（$D(x)$ 为到最近已选 centroid 的距离），改善收敛速度

---

## 二、从零实现 PyTorch 代码 | From-Scratch Snippets

### S1: 数值稳定 Softmax（NumPy）

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    x_shifted = x - x.max(axis=-1, keepdims=True)   # 防溢出
    e = np.exp(x_shifted)
    return e / e.sum(axis=-1, keepdims=True)
```

### S2: 交叉熵损失（NumPy）

```python
def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    logits : (B, C) 未归一化分数
    labels : (B,)    整数类别索引
    """
    # LogSumExp trick
    m = logits.max(axis=1, keepdims=True)
    log_probs = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    N = logits.shape[0]
    return -log_probs[np.arange(N), labels].mean()
```

### S3: Scaled Dot-Product Attention（PyTorch）

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,           # (B, h, N, d_k)
    K: torch.Tensor,           # (B, h, N, d_k)
    V: torch.Tensor,           # (B, h, N, d_v)
    mask: torch.Tensor = None  # (B, 1, N, N), True = masked out
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V    # (B, h, N, d_v)
```

### S4: Multi-Head Attention（PyTorch）

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask=None):
        B, N, D = x.shape
        qkv = self.W_qkv(x).view(B, N, 3, self.n_heads, self.d_k)
        Q, K, V = qkv.unbind(dim=2)           # each (B, N, h, d_k)
        Q = Q.transpose(1, 2)                  # (B, h, N, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)
```

> **`contiguous()` 提示：** `transpose` 后内存不连续，直接 `.view()` 会报错。`.reshape()` 内部会自动拷贝，但 `.view()` 不会。

### S5: LoRA 线性层（PyTorch）

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        # 冻结的预训练权重
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight)
        # 可训练 LoRA 参数
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = x @ self.weight.T                       # 冻结路径
        lora_out = (x @ self.A.T @ self.B.T) * self.scale  # 可训练路径
        return base_out + lora_out
```

### S6: Layer Normalization（NumPy）

```python
import numpy as np

def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,   # (d,)
    beta: np.ndarray,    # (d,)
    eps: float = 1e-6,
) -> np.ndarray:
    """x: (..., d)"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta
```

### S7: K-Means（NumPy）

```python
import numpy as np

def kmeans(
    X: np.ndarray,       # (N, d)
    k: int,
    n_iter: int = 100,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(n_iter):
        # 分配：每个样本归入最近 centroid
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=-1)
        labels = dists.argmin(axis=1)
        # 更新：centroid 移至簇内均值
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if (labels == j).any()
            else centroids[j]
            for j in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids
```

### S8: Top-p 采样（PyTorch）

```python
import torch

def top_p_sample(
    logits: torch.Tensor,  # (vocab_size,)
    p: float = 0.9,
    temperature: float = 1.0,
) -> int:
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=0)
    # 累积概率超出 p 之前的 token 保留
    cutoff = cum_probs - sorted_probs > p
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_idx[idx].item()
```

---

## 三、面试题库 | Interview Questions (25 题)

### L1 — 基础 | Basic

<details>
<summary>Q1. Softmax 的作用是什么？为什么要在计算时减去最大值？</summary>

**答：** Softmax 将实数向量映射为概率分布（非负且和为 1）。减去最大值是为了数值稳定：$e^{z_i}$ 在 $z_i$ 较大时会导致浮点溢出得到 `inf`，softmax 输出变成 `NaN`。数学上减去常数不改变结果（分子分母同时约掉），但使指数运算始终在安全范围内。

> **追问：** 如果输入 logits 全为 0，softmax 输出是什么？
> 答：均匀分布 $[1/C, 1/C, \dots]$。

</details>

---

<details>
<summary>Q2. PyTorch 的 <code>nn.CrossEntropyLoss</code> 接收的是 logits 还是概率？为什么？</summary>

**答：** 接收 **logits**（未归一化分数）。内部先做 `LogSoftmax` 再做 `NLLLoss`，数学上等价于先 softmax 再取 log 再算交叉熵，但数值更稳定（避免 `log(0)` 的情况）。

> **追问：** 如果你手动先做了 softmax 再传入 `CrossEntropyLoss`，会发生什么？
> 答：效果变差，因为 CrossEntropyLoss 会把 softmax 输出（已在 (0,1) 内）当作 logits 再做一次 log_softmax，此时值域被压缩（e^{p_i}≈1–2.7），导致损失语义错误、梯度偏差，而非真正意义上的"锐化"。

</details>

---

<details>
<summary>Q3. Attention 机制中 $Q$、$K$、$V$ 分别代表什么？直觉上如何理解？</summary>

**答：** 可以类比图书馆检索：$Q$（Query）是你的查询问题，$K$（Key）是每本书的索引标签，$V$（Value）是书的内容。Attention 计算 $Q$ 和每个 $K$ 的相似度作为权重，对 $V$ 加权求和，得到"与查询最相关的信息聚合"。

> **追问：** 在 self-attention 中，$Q$、$K$、$V$ 从何而来？
> 答：均由同一输入 $X$ 通过不同的线性投影得到：$Q = XW^Q$，$K = XW^K$，$V = XW^V$。

</details>

---

<details>
<summary>Q4. 什么是 LoRA？其核心思想是什么？</summary>

**答：** LoRA（Low-Rank Adaptation）是一种参数高效微调（PEFT）方法。核心思想：微调时权重变化量 $\Delta W$ 是低秩的，因此可分解为两个小矩阵的乘积 $\Delta W = BA$。训练时冻结原始权重 $W_0$，仅训练 $A$ 和 $B$，大幅减少可训练参数量。

> **追问：** LoRA 的可训练参数量是多少？
> 答：对于 $d_{\text{in}} \times d_{\text{out}}$ 的线性层，LoRA 引入 $r \times d_{\text{in}} + d_{\text{out}} \times r$ 个参数，当 $r \ll d$ 时远小于 $d_{\text{in}} \times d_{\text{out}}$。

</details>

---

<details>
<summary>Q5. LayerNorm 与 BatchNorm 的核心区别是什么？</summary>

**答：** BatchNorm 沿 **batch 维度** 归一化（同一特征在不同样本间统计均值/方差），依赖 batch size。LayerNorm 沿 **特征维度** 归一化（同一样本内不同特征间统计），不依赖 batch size。Transformer 用 LayerNorm 是因为序列长度可变且推理时 batch size 可能为 1。

> **追问：** BatchNorm 在推理时如何处理？
> 答：使用训练时累积的 running mean 和 running variance，而非当前 batch 的统计量。

</details>

---

<details>
<summary>Q6. 滑动窗口（Sliding Window）算法的核心思路是什么？</summary>

**答：** 维护一个窗口 $[l, r]$，右指针 $r$ 逐元素扩展，当窗口满足某种条件时尝试收缩左指针 $l$，在扩展/收缩过程中更新答案。关键在于定义清楚窗口"合法"的条件和收缩策略。时间复杂度通常为 $O(n)$，因为每个元素最多进出窗口各一次。

> **追问：** 如何判断一个问题适合用滑动窗口？
> 答：问题具有"单调性"——当 $[l, r]$ 不满足条件时，$[l, r+1]$ 可能满足；当 $[l, r]$ 满足条件时，$[l+1, r]$ 一定不满足。

</details>

---

<details>
<summary>Q7. 二分搜索中"左闭右开"和"左闭右闭"模板的主要区别？</summary>

**答：** 左闭右开：搜索区间 $[lo, hi)$，初始 $hi = n$，循环条件 $lo < hi$，`hi = mid`（不减 1）。左闭右闭：搜索区间 $[lo, hi]$，初始 $hi = n - 1$，循环条件 $lo \leq hi$，`hi = mid - 1`。左闭右开在处理"找左边界/右边界"时更简洁，不容易出 off-by-one 错误。

> **追问：** 如何用二分搜索找"第一个 $\geq$ target 的位置"？
> 答：使用左闭右开模板，当 `arr[mid] < target` 时 `lo = mid + 1`，否则 `hi = mid`，最终 `lo` 即答案。

</details>

---

<details>
<summary>Q8. Top-p 采样与 Top-k 采样有什么区别？</summary>

**答：** Top-k 固定保留概率最高的 $k$ 个 token。Top-p（Nucleus）保留累积概率首次超过 $p$ 的 token 集合——集合大小自适应：分布集中时保留少，分布平坦时保留多。Top-p 更灵活，避免了 Top-k 在分布尖锐时引入噪声、平坦时截断过多的问题。

> **追问：** 实际推理中 top-p 和 top-k 可以同时使用吗？通常哪个先过滤？
> 答：可以同时使用，通常先 top-k 再 top-p（缩小候选集后再做累积概率截断）。

</details>

---

<details>
<summary>Q9. 什么是 BFS 和 DFS？在图搜索中各有什么适用场景？</summary>

**答：** BFS（广度优先搜索）用队列，逐层扩展，适合求最短路径（无权图）。DFS（深度优先搜索）用栈（或递归），沿一条路走到底再回溯，适合求连通分量、拓扑排序、检测环。时间复杂度均为 $O(V + E)$。

> **追问：** 拓扑排序可以用 BFS 实现吗？
> 答：可以，即 Kahn 算法：维护入度为 0 的节点队列，每次取出一个节点，将其邻居入度减 1。

</details>

---

### L2 — 中级 | Intermediate

<details>
<summary>Q10. Scaled Dot-Product Attention 中为什么除以 $\sqrt{d_k}$？如果不除会怎样？</summary>

**答：** 假设 $Q$ 和 $K$ 的每个元素独立同分布，均值为 0，方差为 1，则 $QK^\top$ 每个元素的方差为 $d_k$。当 $d_k$ 较大时，点积值量级很大，softmax 梯度趋近于零（输出趋向 one-hot），训练困难。除以 $\sqrt{d_k}$ 将方差稳定在 1 附近。

> **追问：** 有没有其他缩放方式？
> 答：Cosine attention 用 $\frac{QK^\top}{\|Q\|\|K\|}$；也有工作探索可学习的温度参数。

</details>

---

<details>
<summary>Q11. LoRA 中 $B$ 初始化为零、$A$ 随机初始化的策略有什么好处？</summary>

**答：** $B = 0$ 使得训练开始时 $\Delta W = BA = 0$，模型行为与预训练完全一致，保证初始状态不被破坏。如果 $A$ 也初始化为零，则 $\Delta W$ 永远为零（梯度为零），$A$ 和 $B$ 无法学到任何东西。因此必须至少一个矩阵非零初始化。

> **追问：** LoRA 的缩放因子 $\alpha / r$ 起什么作用？
> 答：$\alpha$ 控制 LoRA 更新的绝对幅度。$\alpha$ 固定时，增大 $r$ 会使更新幅度相对减小（除以更大的 $r$），有助于训练稳定性。实践中常设 $\alpha = r$ 或 $\alpha = 2r$。

</details>

---

<details>
<summary>Q12. Multi-Head Attention 中为什么需要多个 head？与单 head + 更大维度有何不同？</summary>

**答：** 多个 head 可以关注不同类型的模式（如语法、语义、位置关系），相当于一种隐式正则化。单 head + 大维度理论上表达能力等价，但优化更困难——模型倾向于只学到一种模式。实验表明多头在相同总计算量下效果更好。

> **追问：** 不同 head 学到的 pattern 有多大差异？有没有可视化工具？
> 答：BERTViz 等工具可视化表明，不同 head 确实关注不同类型的关系（如有的 head 关注相邻词，有的关注动宾关系）。但也有研究发现部分 head 是冗余的，可以被裁剪（head pruning）。

</details>

---

<details>
<summary>Q13. K-Means 的时间复杂度是什么？K-Means++ 初始化如何改善收敛？</summary>

**答：** 每次迭代需计算 $n$ 个样本到 $K$ 个 centroid 的距离（$O(nKd)$），设迭代 $i$ 次，总复杂度 $O(nKdi)$。K-Means++ 按概率 $\propto D(x)^2$ 顺序选取初始 centroid（$D(x)$ 为到最近已选 centroid 的距离），使得初始 centroid 尽量分散。理论上保证 WCSS 的期望值在 $O(\log K)$ 近似比内。

> **追问：** 如何选择 $K$？
> 答：常用 Elbow Method（画 WCSS vs $K$ 曲线，找拐点）或 Silhouette Score（衡量簇内紧密度与簇间分离度）。

</details>

---

<details>
<summary>Q14. RMSNorm 相比 LayerNorm 去掉了什么？为什么在大模型中常用 RMSNorm？</summary>

**答：** RMSNorm 去掉了均值中心化（re-centering）步骤，仅做缩放（re-scaling）。省去了均值计算，减少了约 10-15% 的计算量。实验表明去掉中心化对效果影响很小，尤其在大模型中，因此 LLaMA、Qwen 等主流模型均采用 RMSNorm。

> **追问：** Pre-Norm 和 Post-Norm 有什么区别？为什么现代模型多用 Pre-Norm？
> 答：Pre-Norm 在子层之前做归一化，Post-Norm 在子层之后。Pre-Norm 梯度更稳定、训练更易收敛，但可能损失一些表达能力。Post-Norm 理论上表达力更强，但需要更细致的超参调节（如 warmup）。

</details>

---

<details>
<summary>Q15. <code>torch.Tensor.view()</code> 和 <code>torch.Tensor.reshape()</code> 有什么区别？</summary>

**答：** `.view()` 要求 tensor 内存连续（contiguous），否则报错；返回的是共享内存的视图。`.reshape()` 更灵活：如果内存连续则等价于 `.view()`，否则会自动拷贝数据使其连续。性能敏感场景优先用 `.view()`（零拷贝），但需要确保先调用 `.contiguous()`。

> **追问：** 什么操作会导致 tensor 不连续？
> 答：`transpose()`、`permute()`、某些 `expand()` 操作。这些操作改变了 stride 但不移动数据，导致逻辑顺序与物理内存顺序不一致。

</details>

---

<details>
<summary>Q16. Temperature 参数如何影响采样分布？$T \to 0$ 和 $T \to \infty$ 分别等价于什么？</summary>

**答：** Temperature $T$ 用于缩放 logits：$p_i \propto e^{z_i / T}$。$T \to 0$ 时分布退化为 one-hot（贪心解码 greedy decoding）；$T \to \infty$ 时分布趋向均匀分布（完全随机）。$T < 1$ 使分布更尖锐（更确定），$T > 1$ 使分布更平坦（更多样）。

> **追问：** 实践中 temperature 通常设为多少？不同任务有何区别？
> 答：代码生成等确定性任务常用 $T \approx 0$–$0.2$；创意写作等多样性任务常用 $T \approx 0.7$–$1.0$。一般不超过 2.0，否则输出质量下降。

</details>

---

<details>
<summary>Q17. 动态规划（DP）的核心三要素是什么？</summary>

**答：** (1) **状态定义**——$dp[i]$ 代表什么（如前 $i$ 个元素的最优解）；(2) **转移方程**——$dp[i]$ 如何由之前的状态推出；(3) **初始条件与边界**——$dp[0]$ 的值。设计 DP 时最关键的是选对状态，好的状态定义使转移方程自然且高效。

> **追问：** 记忆化搜索（top-down）和自底向上填表（bottom-up）有什么区别？各自优缺点？
> 答：记忆化用递归 + 缓存，只计算需要的状态，代码更直观。bottom-up 用循环填表，无递归开销，易于空间优化（滚动数组）。两者时间复杂度相同。

</details>

---

### L3 — 深度 | Deep

<details>
<summary>Q18. Softmax 的梯度是什么？为什么说 Cross-Entropy + Softmax 的梯度形式简洁？</summary>

**答：** 令 $p_i = \text{softmax}(z_i)$，则 $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$（Jacobian 矩阵）。当与 Cross-Entropy $\mathcal{L} = -\log p_y$ 组合时，梯度简化为：

$$
\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbb{1}[i = y]
$$

即预测概率减去真实标签（one-hot），形式极为简洁。这也是为什么 PyTorch 将两者合并为一个算子的原因。

> **追问：** Label Smoothing 会如何改变这个梯度？
> 答：Label smoothing 将 one-hot 改为 $(1 - \epsilon)$ 在正确类、$\epsilon / (C-1)$ 在其他类，梯度变为 $p_i - \hat{y}_i$，防止模型过度自信。

</details>

---

<details>
<summary>Q19. Self-Attention 的计算复杂度是 $O(N^2 d)$，有哪些降低复杂度的方法？</summary>

**答：** 主要方向包括：
- **Sparse Attention：** 只计算局部窗口（如 Longformer 的 sliding window）或特定 pattern（如 BigBird 的 random + global + local）
- **Linear Attention：** 用核函数近似 softmax，将 $QK^\top V$ 的计算顺序改为 $Q(K^\top V)$，复杂度降为 $O(Nd^2)$
- **Flash Attention：** 不改变数学结果，通过 tiling 和 recomputation 优化 GPU 的 HBM/SRAM 访问模式，减少内存占用和实际运行时间
- **Low-rank 近似：** 如 Linformer 将 $K, V$ 投影到低维

> **追问：** Flash Attention 改变了注意力的数学结果吗？
> 答：不改变。Flash Attention 是一个 IO-aware 的精确算法，通过分块计算（tiling）和在线 softmax 技巧得到与标准 attention 数学上等价的结果，只是更高效地利用了 GPU 内存层级。

</details>

---

<details>
<summary>Q20. LoRA 的秩（rank）$r$ 应该如何选择？$\alpha$ 和 $r$ 的关系是什么？</summary>

**答：** rank $r$ 控制表达能力与参数量的权衡。常用范围 $r \in [4, 64]$。任务越复杂或目标层越多，可能需要更大的 $r$。$\alpha$ 是缩放因子，有效学习率大致正比于 $\alpha / r$。常见做法是固定 $\alpha$（如 $\alpha = 16$）然后调节 $r$，或者设 $\alpha = r$ 让缩放因子为 1。

> **追问：** 可以对不同的层使用不同的 rank 吗？有什么好处？
> 答：可以，这是 AdaLoRA 等方法的思路。不同层的"重要性"不同（通过 SVD 分析 $\Delta W$ 的奇异值谱），对重要层分配更高 rank，不重要的层分配更低 rank，实现参数预算的最优分配。

</details>

---

<details>
<summary>Q21. K-Means 一定能收敛吗？收敛到的一定是全局最优吗？</summary>

**答：** K-Means 保证收敛（WCSS 单调递减，有下界 0），但只收敛到 **局部最优**。分配步和更新步都会降低或保持 WCSS，所以 WCSS 不增。但目标函数非凸，不同初始值可能导致不同的局部最优。因此实践中常多次随机初始化取最优（multi-start），或使用 K-Means++ 改善初始化质量。

> **追问：** K-Means 的簇形状有什么限制？有没有替代方法？
> 答：K-Means 假设簇为球形（isotropic），对非球形簇效果差。替代方法包括 GMM（高斯混合模型，允许椭圆簇）、DBSCAN（基于密度，可发现任意形状簇）、Spectral Clustering。

</details>

---

<details>
<summary>Q22. 在 Transformer 推理中，KV Cache 是什么？为什么需要它？</summary>

**答：** 自回归生成时，每生成一个新 token 需要计算所有历史 token 的 attention。但之前 token 的 $K, V$ 不会因新 token 改变，因此可以缓存起来复用。KV Cache 存储已计算的 $K, V$ 矩阵，每步只需计算新 token 的 $Q, K, V$，将新 $K, V$ 追加到 cache。这将每步的计算量从 $O(n^2 d)$ 降为 $O(nd)$，是推理加速的关键。

> **追问：** KV Cache 的内存开销有多大？有什么压缩方法？
> 答：对于 $L$ 层、$h$ 头、head dim $d_k$、序列长度 $n$ 的模型，KV Cache 大小为 $2 \times L \times h \times d_k \times n \times \text{bytes}$。压缩方法包括：GQA（分组查询注意力，多个 Q head 共享 K/V）、量化（将 KV 从 FP16 量化到 INT8/INT4）、稀疏化（丢弃不重要的 KV 对）。

</details>

---

<details>
<summary>Q23. 手写 Attention 时，causal mask 用 <code>float('-inf')</code> 而不是一个很大的负数，为什么？</summary>

**答：** `float('-inf')` 经 softmax 后精确为 0（$e^{-\infty} = 0$），不会有任何信息泄漏。而大负数（如 $-10^9$）经 softmax 后会得到一个极小但非零的概率，理论上仍有微量信息泄漏。此外，`-inf` 在数值上更安全——不同精度的"大负数"阈值不同，而 `-inf` 在所有精度下行为一致。

> **追问：** 除了 causal mask，attention 中还有哪些常用 mask？
> 答：Padding mask（屏蔽填充 token）、Local/Sliding window mask（限制注意力范围）、Prefix mask（prefix 部分双向，后续 causal）、Block diagonal mask（多序列独立注意力）。

</details>

---

<details>
<summary>Q24. 为什么 softmax 需要数值稳定技巧，但 <code>torch.nn.functional.cross_entropy</code> 内部不需要用户关心这个问题？</summary>

**答：** PyTorch 的 `F.cross_entropy` 使用了更高级的数值稳定实现——在 log 空间直接计算 LogSumExp，全程避免显式的 `exp()` 和 `log()` 组合，数学上：

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}
$$

在单个 kernel 中完成，全程在 log 空间操作，避免了中间结果溢出。而手写 `softmax` + `log` + `nll` 是分步的，中间的 `exp` 可能溢出。

> **追问：** `torch.log_softmax` 的实现和 `log(softmax(x))` 有什么区别？
> 答：数学等价但数值行为不同。`torch.log_softmax` 内部直接计算 LogSumExp trick，避免中间的 `softmax` 溢出。`log(softmax(x))` 如果 `softmax` 已经溢出为 `inf`，`log(inf)` 结果虽然是 `inf`，但梯度为 `NaN`。

</details>

---

<details>
<summary>Q25. 在分布式训练（如联邦学习）场景中使用 LoRA，与全参数微调相比，通信开销有什么变化？</summary>

**答：** 全参数微调需要通信整个 $\Delta W$（$d_{\text{in}} \times d_{\text{out}}$），而 LoRA 只需通信 $A$（$r \times d_{\text{in}}$）和 $B$（$d_{\text{out}} \times r$），通信量为 $r \times (d_{\text{in}} + d_{\text{out}})$，当 $r \ll d$ 时大幅减少。这对于带宽受限的场景（如跨设备联邦学习）尤为重要。

> **追问：** LoRA 在联邦学习中聚合时，是聚合 $A, B$ 还是聚合 $\Delta W = BA$？
> 答：两种方式都可以。直接聚合 $A, B$ 通信量小但数学上不等价于聚合 $\Delta W$（$BA$ 的平均 $\neq$ 平均的 $B \times$ 平均的 $A$）。聚合 $\Delta W$ 数学上更正确但通信量大。实践中需要根据场景权衡。

</details>

---

## 附录：算法模式速查 | Algorithm Pattern Quick Reference

### 双指针 | Two Pointers

```python
# 对向双指针（有序数组找 pair）
l, r = 0, len(arr) - 1
while l < r:
    s = arr[l] + arr[r]
    if s == target:
        # found
        l += 1; r -= 1
    elif s < target:
        l += 1
    else:
        r -= 1
```

### 滑动窗口 | Sliding Window（可变大小）

```python
l = 0
for r in range(len(arr)):
    # 扩展窗口：加入 arr[r]
    ...
    while 窗口不满足条件:   # 收缩
        # 移除 arr[l]
        l += 1
    # 更新答案
    ans = max(ans, r - l + 1)
```

### 二分搜索 | Binary Search（左闭右开，找左边界）

```python
lo, hi = 0, len(arr)      # 搜索区间 [lo, hi)
while lo < hi:
    mid = (lo + hi) // 2
    if arr[mid] < target:
        lo = mid + 1
    else:
        hi = mid            # 保留 mid
# lo == hi 即第一个 >= target 的位置
```

### BFS 层序遍历 | Level-Order BFS

```python
from collections import deque
q = deque([root])
while q:
    for _ in range(len(q)):       # 逐层处理
        node = q.popleft()
        for child in node.children:
            q.append(child)
```

### DP 自底向上 | Bottom-Up DP

```python
dp = [0] * (n + 1)
dp[0] = base_case
for i in range(1, n + 1):
    dp[i] = f(dp[i-1], dp[i-2], ...)
```

---

## 练习日志模板 | Practice Log Template

| 题号 | 主题 | 难度 | 状态 | 复习日 |
|------|------|------|------|--------|
| LC 1 | 哈希表 / Two Sum | Easy | | |
| LC 3 | 滑动窗口 | Medium | | |
| LC 15 | 双指针 / 三数之和 | Medium | | |
| LC 33 | 二分搜索 | Medium | | |
| LC 56 | 区间合并 | Medium | | |
| LC 102 | 树 BFS 层序 | Medium | | |
| LC 207 | 图 / 拓扑排序 | Medium | | |
| LC 300 | DP / LIS | Medium | | |
| LC 295 | 堆 / 数据流中位数 | Hard | | |
| LC 42 | 单调栈 / 接雨水 | Hard | | |

> **状态标记：** ✅ 熟练 | ⚠️ 模糊 | ❌ 不会 — 仅对 ⚠️/❌ 安排复习。


## 更多 L3 深挖 / Extended L3

<details>
<summary>Q26. LoRA 的低秩假设（low-rank assumption）在理论上基于什么前提？哪些场景下该假设可能不成立？</summary>

**A:** LoRA 假设权重更新 $\Delta W$ 的有效秩（effective rank）远小于参数维度——即任务适配信息集中在少数方向上。理论基础是"内在维度（intrinsic dimensionality）"概念：预训练模型在高维参数空间中，任务适配实际只发生在低维子流形上。

失效场景：
- 下游任务与预训练分布差距大，需修改高秩结构（如跨语言、跨模态迁移）
- 多任务场景中各任务的低秩子空间不重叠，叠加后秩升高
- 模型本身较小，低秩约束成为容量瓶颈

**追问：** 除了单纯增大 $r$，还有哪些结构化方案突破低秩限制？
**答：** AdaLoRA 按层自适应分配秩（重要层更高秩）；Adapter 在 FFN 中插入带非线性激活的 bottleneck，突破纯线性低秩的表达限制；(IA)^3 等向量缩放方法以不同的效率-容量曲线提供额外表达力。

</details>

---

<details>
<summary>Q27. Transformer 中为什么用 LayerNorm 而非 BatchNorm？从统计量计算的角度分析二者的根本区别。</summary>

**A:** BatchNorm 沿 **batch 维度** 计算均值和方差，LayerNorm 沿 **特征维度** 计算。根本区别在于：

1. **序列长度可变性：** NLP 中同 batch 内序列长度不同，padding 位置对 BatchNorm 统计量引入噪声
2. **自回归生成：** 推理时 batch size 常为 1，BatchNorm 的统计量极不稳定
3. **每个 token 独立归一化：** LayerNorm 使各层输入分布稳定，不依赖 batch 中其他样本

**追问：** Pre-LN 和 Post-LN 有什么区别？为什么现代大模型多采用 Pre-LN？
**答：** Post-LN：$\text{LN}(x + \text{Sublayer}(x))$；Pre-LN：$x + \text{Sublayer}(\text{LN}(x))$。Pre-LN 使梯度通过残差路径直接回传（residual stream），不经过 LN 的非线性变换，训练更稳定，通常不需要 learning rate warmup；Post-LN 在深层网络中梯度需经过 LN，容易导致训练不稳定。

</details>

---

<details>
<summary>Q28. FlashAttention 的核心思想是什么？为什么它能在不改变数学结果的前提下显著提升效率？</summary>

**A:** 核心是 **IO-aware 分块计算（tiling）**：将 $Q, K, V$ 分成小块（tile），在 SRAM（片上缓存）中完成 softmax 与矩阵乘的融合计算，避免将 $N \times N$ 注意力矩阵写回 HBM（高带宽内存）。

关键技巧：
- **Online softmax：** 利用 softmax 的可分解性，逐块计算时维护 running max 和 running sum，每块结果通过缩放因子校正，最终与全局 softmax 数学等价
- **重计算（recomputation）：** 反向传播时不存储注意力矩阵，从 $Q, K, V$ 重新计算——以计算换内存

**追问：** FlashAttention 的 recomputation 与通用 gradient checkpointing 有什么异同？
**答：** 二者都是"以计算换内存"思想。Gradient checkpointing 是通用策略：选择性不保存中间激活，反向时重新前向计算。FlashAttention 的 recomputation 更特化：专门针对注意力矩阵这一大张量，且与 tiling 策略结合，不仅减少内存还减少了 HBM 访问次数（IO 复杂度从 $O(N^2)$ 降至 $O(N^2 d^2 / M)$，$M$ 为 SRAM 大小）。二者可组合使用。

</details>

---

<details>
<summary>Q29. RMSNorm 去掉了 LayerNorm 中的均值中心化（mean centering），为什么这样做在实践中仍然有效？</summary>

**A:** LayerNorm 的完整操作：$y_i = \gamma \cdot \frac{x_i - \mu}{\sigma} + \beta$；RMSNorm 简化为：$y_i = \gamma \cdot \frac{x_i}{\text{RMS}(x)}$，去掉 $\mu$ 和 $\beta$。

仍然有效的理论直觉：
- 深度网络中，经大量线性变换和激活后，特征均值往往已在合理范围，或可被后续层的 bias 补偿
- 归一化对训练稳定性起关键作用的是 **re-scaling**（控制方差），而非 re-centering（减均值）
- 去掉均值计算减少了一个 reduce 操作，在大规模模型中累积起来有显著效率提升

**追问：** 什么情况下去掉均值中心化可能有害？
**答：** 若某层输入有系统性偏移（systematic bias）且后续层无法轻易补偿，则偏移会传播。在小模型或浅层网络中影响可能更明显。但大规模 Transformer 的残差连接和深层堆叠提供了足够容量来补偿，实践中几乎未观察到退化。

</details>

---

<details>
<summary>Q30. Multi-Head Attention 中如果多个 head 学到了近似的 pattern（head collapse），会有什么后果？如何检测和缓解？</summary>

**A:** Head collapse 导致多头表达冗余——虽有 $h$ 个头的参数，但有效头数远少于 $h$，浪费了计算和模型容量，相当于降低了注意力的"有效秩"。

检测方法：
- 计算不同 head 注意力分布之间的 KL 散度或余弦相似度
- 分析 $W^Q, W^K$ 投影矩阵之间的子空间重叠度（如 principal angle）

**追问：** 如何从训练层面缓解 head collapse？
**答：**
- **Diversity regularization：** 在损失函数中加入鼓励不同 head 注意力分布差异的正则项（如惩罚相似度）
- **Attention dropout：** 对注意力权重施加 dropout，防止某些 head 过早主导
- **Head pruning + retraining：** 训练后裁剪冗余 head 再微调，迫使剩余 head 承担更多职责；这也是一种隐式的正则化

</details>

---

<details>
<summary>Q31. 对 Cross-Entropy Loss 施加 label smoothing 后，梯度形式有什么变化？为什么能提高泛化？</summary>

**A:** 标准 CE 中目标 $q$ 为 one-hot（$q_y = 1, q_{j \neq y} = 0$），label smoothing 令 $q_y = 1 - \epsilon, q_{j \neq y} = \frac{\epsilon}{C-1}$。

梯度变化（$\nabla_{z_i} \mathcal{L} = p_i - q_i$）：
- 标准 CE：$\nabla_{z_y} \mathcal{L} = p_y - 1$，$\nabla_{z_j} \mathcal{L} = p_j$
- Label smoothing：$\nabla_{z_y} \mathcal{L} = p_y - (1-\epsilon)$，$\nabla_{z_j} \mathcal{L} = p_j - \frac{\epsilon}{C-1}$

即模型不再被鼓励将 logits 推向无穷大，防止 over-confidence。

泛化改善：防止模型对训练标签过于自信（过拟合噪声标签），隐式地对 logits 施加了 soft 约束。

**追问：** Label smoothing 和 knowledge distillation 有什么联系？
**答：** 二者本质都是用"软目标（soft target）"替代 hard target 训练。Label smoothing 用均匀分布软化目标；distillation 用 teacher 输出分布作为软目标。可以认为 label smoothing 是"无 teacher 的、目标为均匀分布的 distillation"。Distillation 的优势在于 teacher 的 soft target 包含类别间相似性的结构信息（如猫 vs 狗的 logit 高于猫 vs 飞机），而非无结构的均匀分布。

</details>

---

<details>
<summary>Q32. K-Means 和 Gaussian Mixture Model (GMM) 的 EM 算法之间有什么数学联系？</summary>

**A:** K-Means 是 GMM-EM 在"各向同性、等协方差、$\sigma \to 0$"极限下的硬指派特例。

对 GMM $p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ 施加约束：所有分量共享固定的各向同性协方差 $\Sigma_k = \sigma^2 I$、等权重 $\pi_k = 1/K$。此时 E 步的软责任是温度为 $\sigma^2$ 的 softmax（等权重、等协方差使归一化常数约掉），令 $\sigma \to 0$ 即退化为硬指派：

$$
\gamma_{ik} = \frac{\exp\!\big(-\|x_i - \mu_k\|^2 / 2\sigma^2\big)}{\sum_j \exp\!\big(-\|x_i - \mu_j\|^2 / 2\sigma^2\big)} \;\xrightarrow{\;\sigma \to 0\;}\; \mathbf{1}\big[\,k = \arg\min_j \|x_i - \mu_j\|^2\,\big]
$$

由此两步一一对应：
- **E 步：** GMM 的软责任 $\gamma_{ik}$ → K-Means"分到最近质心"的硬指派（上式 $\sigma \to 0$ 极限）。
- **M 步：** GMM 的加权均值 $\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$ 在硬 $\gamma$ 下退化为簇内点的算术平均——即 K-Means 的质心更新；两者都只更新均值。
- **目标函数：** 该极限下 GMM 的负对数似然（去掉与 $\sigma$ 相关的常数）正比于 K-Means 的簇内平方和 $\sum_i \min_k \|x_i - \mu_k\|^2$。

**追问：** 既然 K-Means 是 GMM 的特例，什么时候该用 GMM？
**答：** 当簇非球形、大小/密度差异大、或需要软（概率化）成员归属时。GMM 额外学习每个分量的协方差 $\Sigma_k$ 与权重 $\pi_k$，能拟合不同朝向/尺度的椭圆簇，并用 $\gamma_{ik}$ 给出软归属与不确定性；K-Means 因假设各向同性等协方差，只能产生球形（Voronoi）硬划分。代价是 GMM 参数更多、对初始化与奇异协方差（某分量塌缩到单点使似然发散）更敏感，常需协方差下限或正则化。

</details>
