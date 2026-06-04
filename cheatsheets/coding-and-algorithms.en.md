# Coding & ML Implementation — Interview Cheat Sheet

> **Audience:** Research / ML Intern candidates
> **Usage:** English; key math terms kept as-is; formulas in LaTeX; all code is from-scratch implementation

---

## 1. Core Concepts & Formula Derivations

### 1.1 Softmax Function

Maps an arbitrary vector to a probability distribution:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

**Numerically Stable Version:** Let $m = \max(z)$, then

$$
\text{softmax}(z_i) = \frac{e^{z_i - m}}{\sum_{j=1}^{C} e^{z_j - m}}
$$

Mathematically equivalent, but avoids floating-point overflow in $e^{z_i}$.

### 1.2 Cross-Entropy Loss

Given logits $\mathbf{z} \in \mathbb{R}^C$ and ground-truth label $y$:

$$
\mathcal{L} = -\log \text{softmax}(z_y) = -z_y + \log \sum_{j=1}^{C} e^{z_j}
$$

The right-hand side is the **LogSumExp** trick: $\log \sum e^{z_j} = m + \log \sum e^{z_j - m}$, where $m = \max(\mathbf{z})$.

PyTorch's `nn.CrossEntropyLoss` internally combines `LogSoftmax` + `NLLLoss` and accepts **logits** directly, not probabilities.

### 1.3 Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $Q$ | Query matrix | $(N, d_k)$ |
| $K$ | Key matrix | $(N, d_k)$ |
| $V$ | Value matrix | $(N, d_v)$ |
| $d_k$ | Key dimension | scalar |

**Why divide by $\sqrt{d_k}$?** When $d_k$ is large the variance of $QK^\top$ grows linearly with $d_k$, pushing softmax outputs toward one-hot and gradients toward zero. Scaling keeps variance at $\mathcal{O}(1)$.

### 1.4 Multi-Head Attention

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$
$$
\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V)
$$

where $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $d_k = d_v = d_{\text{model}} / h$.

**Why multiple heads?** Different heads can attend to different subspace patterns (e.g., syntactic vs. semantic relationships); total compute is comparable to a single head.

### 1.5 Layer Normalization

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$

where $\mu = \frac{1}{d}\sum_{i=1}^d x_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$, computed along the **feature dimension** $d$.

**RMSNorm (Root Mean Square Normalization):** Removes mean-centering, keeping only re-scaling:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}, \quad y_i = \gamma \cdot \frac{x_i}{\text{RMS}(x) + \epsilon}
$$

Performance is close to LayerNorm in practice but faster (no mean computation).

### 1.6 LoRA (Low-Rank Adaptation)

Freeze pretrained weights $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and inject trainable low-rank decomposition:

$$
W = W_0 + \Delta W = W_0 + BA
$$

where $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $r \ll \min(d_{\text{in}}, d_{\text{out}})$.

Forward pass with scaling factor $\alpha$:

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

**Initialization strategy:** $A \sim \mathcal{N}(0, \sigma^2)$, $B = \mathbf{0}$. This ensures $\Delta W = BA = \mathbf{0}$ at the start of training, so the model behaves identically to the pretrained checkpoint.

### 1.7 Top-p (Nucleus) Sampling

Given the probability distribution $p_i = \text{softmax}(z_i / T)$ after temperature scaling of logits $\mathbf{z}$:

1. Sort tokens in descending order by probability
2. Compute cumulative probability
3. Retain all tokens up to and including the first one that pushes the cumulative probability past threshold $p$
4. Re-normalize within the retained set and sample

**Effect of Temperature:** $T \to 0$ degenerates to greedy decoding; $T = 1$ preserves the original distribution; $T > 1$ flattens the distribution.

### 1.8 K-Means Clustering

**Objective (WCSS, Within-Cluster Sum of Squares):**

$$
\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
$$

**Algorithm:** Alternates between (1) an assignment step (assign each point to its nearest centroid) and (2) an update step (move each centroid to the mean of its cluster).

- Time complexity: $\mathcal{O}(nKdi)$, where $n$ = number of samples, $K$ = number of clusters, $d$ = dimension, $i$ = number of iterations
- **K-Means++** initialization: select initial centroids sequentially with probability $\propto D(x)^2$ ($D(x)$ = distance to the nearest already-chosen centroid), improving convergence speed

---

## 2. From-Scratch PyTorch Snippets

### S1: Numerically Stable Softmax (NumPy)

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    x_shifted = x - x.max(axis=-1, keepdims=True)   # prevent overflow
    e = np.exp(x_shifted)
    return e / e.sum(axis=-1, keepdims=True)
```

### S2: Cross-Entropy Loss (NumPy)

```python
def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    logits : (B, C) unnormalized scores
    labels : (B,)   integer class indices
    """
    # LogSumExp trick
    m = logits.max(axis=1, keepdims=True)
    log_probs = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    N = logits.shape[0]
    return -log_probs[np.arange(N), labels].mean()
```

### S3: Scaled Dot-Product Attention (PyTorch)

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

### S4: Multi-Head Attention (PyTorch)

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

> **`contiguous()` note:** Memory is non-contiguous after `transpose`; calling `.view()` directly will error. `.reshape()` copies automatically when needed, but `.view()` does not.

### S5: LoRA Linear Layer (PyTorch)

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
        # frozen pretrained weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight)
        # trainable LoRA parameters
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = x @ self.weight.T                       # frozen path
        lora_out = (x @ self.A.T @ self.B.T) * self.scale  # trainable path
        return base_out + lora_out
```

### S6: Layer Normalization (NumPy)

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

### S7: K-Means (NumPy)

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
        # assignment: each sample goes to its nearest centroid
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=-1)
        labels = dists.argmin(axis=1)
        # update: move centroid to cluster mean
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

### S8: Top-p Sampling (PyTorch)

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
    # retain tokens before cumulative probability exceeds p
    cutoff = cum_probs - sorted_probs > p
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_idx[idx].item()
```

---

## 3. Interview Question Bank (25 Questions)

### L1 — Basic

<details>
<summary>Q1. What does Softmax do? Why subtract the maximum value during computation?</summary>

**A:** Softmax maps a real-valued vector to a probability distribution (non-negative and sums to 1). Subtracting the maximum is for numerical stability: when $z_i$ is large, $e^{z_i}$ causes floating-point overflow, giving `inf` and making the softmax output `NaN`. Subtracting a constant does not change the result mathematically (it cancels in numerator and denominator), but keeps the exponentiation in a safe range.

> **Follow-up:** If all input logits are 0, what does softmax output?
> A: The uniform distribution $[1/C, 1/C, \dots]$.

</details>

---

<details>
<summary>Q2. Does PyTorch's <code>nn.CrossEntropyLoss</code> accept logits or probabilities? Why?</summary>

**A:** It accepts **logits** (unnormalized scores). Internally it applies `LogSoftmax` followed by `NLLLoss`, which is mathematically equivalent to softmax → log → cross-entropy, but numerically more stable (avoids `log(0)`).

> **Follow-up:** If you manually apply softmax first and then pass the result into `CrossEntropyLoss`, what happens?
> A: Performance degrades because `CrossEntropyLoss` treats the softmax output (already in (0,1)) as logits and applies `log_softmax` again; the compressed range ($e^{p_i}\approx 1$–$2.7$) produces semantically wrong loss values and biased gradients — it is not strictly a "sharpening" effect.

</details>

---

<details>
<summary>Q3. What do $Q$, $K$, $V$ represent in the Attention mechanism? What is the intuition?</summary>

**A:** Think of a library search: $Q$ (Query) is your search question, $K$ (Key) is the index label on each book, $V$ (Value) is the book's content. Attention computes the similarity between $Q$ and each $K$ as weights, then takes a weighted sum of $V$ to produce "an aggregation of information most relevant to the query."

> **Follow-up:** In self-attention, where do $Q$, $K$, $V$ come from?
> A: All three are derived from the same input $X$ through different linear projections: $Q = XW^Q$, $K = XW^K$, $V = XW^V$.

</details>

---

<details>
<summary>Q4. What is LoRA? What is its core idea?</summary>

**A:** LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) method. Core idea: the weight change $\Delta W$ during fine-tuning is low-rank, so it can be factored as the product of two small matrices $\Delta W = BA$. During training, the original weights $W_0$ are frozen; only $A$ and $B$ are trained, drastically reducing the number of trainable parameters.

> **Follow-up:** How many trainable parameters does LoRA introduce?
> A: For a linear layer of shape $d_{\text{in}} \times d_{\text{out}}$, LoRA adds $r \times d_{\text{in}} + d_{\text{out}} \times r$ parameters, which is far fewer than $d_{\text{in}} \times d_{\text{out}}$ when $r \ll d$.

</details>

---

<details>
<summary>Q5. What is the key difference between LayerNorm and BatchNorm?</summary>

**A:** BatchNorm normalizes along the **batch dimension** (computing mean/variance of the same feature across different samples) and depends on batch size. LayerNorm normalizes along the **feature dimension** (computing statistics across different features within the same sample) and does not depend on batch size. Transformers use LayerNorm because sequence lengths vary and inference batch size may be 1.

> **Follow-up:** How does BatchNorm behave at inference time?
> A: It uses the running mean and running variance accumulated during training rather than the statistics of the current batch.

</details>

---

<details>
<summary>Q6. What is the core idea of the Sliding Window algorithm?</summary>

**A:** Maintain a window $[l, r]$; the right pointer $r$ advances element by element, and when the window satisfies some condition the left pointer $l$ is shrunk; the answer is updated during expansion/contraction. The key is to clearly define the condition for the window being "valid" and the shrinking strategy. Time complexity is typically $O(n)$ because each element enters and exits the window at most once.

> **Follow-up:** How do you tell whether a problem is suitable for a sliding window?
> A: The problem has "monotonicity" — when $[l, r]$ does not satisfy the condition, $[l, r+1]$ might; when $[l, r]$ satisfies the condition, $[l+1, r]$ definitely does not.

</details>

---

<details>
<summary>Q7. What is the main difference between the "closed-open" and "closed-closed" binary search templates?</summary>

**A:** Closed-open: search interval $[lo, hi)$, initialize $hi = n$, loop condition $lo < hi$, `hi = mid` (no minus 1). Closed-closed: search interval $[lo, hi]$, initialize $hi = n - 1$, loop condition $lo \leq hi$, `hi = mid - 1`. The closed-open variant is cleaner for finding left/right boundaries and less prone to off-by-one errors.

> **Follow-up:** How do you use binary search to find "the first position $\geq$ target"?
> A: Use the closed-open template: when `arr[mid] < target` set `lo = mid + 1`, otherwise set `hi = mid`; the final value of `lo` is the answer.

</details>

---

<details>
<summary>Q8. What is the difference between Top-p and Top-k sampling?</summary>

**A:** Top-k retains the $k$ tokens with the highest probability. Top-p (Nucleus) retains the smallest set of tokens whose cumulative probability first exceeds $p$ — the set size is adaptive: small when the distribution is concentrated, large when flat. Top-p is more flexible and avoids the problems of Top-k introducing noise when the distribution is sharp or truncating too aggressively when it is flat.

> **Follow-up:** Can top-p and top-k be used together in practice? Which filter is typically applied first?
> A: Yes, they can be combined; typically top-k is applied first (to narrow the candidate set) and then top-p (cumulative probability cutoff).

</details>

---

<details>
<summary>Q9. What are BFS and DFS? What are their respective use cases in graph search?</summary>

**A:** BFS (Breadth-First Search) uses a queue, expands level by level, and is well-suited for finding shortest paths in unweighted graphs. DFS (Depth-First Search) uses a stack (or recursion), goes as deep as possible before backtracking, and is well-suited for finding connected components, topological sorting, and cycle detection. Time complexity of both is $O(V + E)$.

> **Follow-up:** Can topological sorting be implemented with BFS?
> A: Yes — this is Kahn's algorithm: maintain a queue of nodes with in-degree 0; repeatedly dequeue a node and decrement the in-degree of its neighbors.

</details>

---

### L2 — Intermediate

<details>
<summary>Q10. Why divide by $\sqrt{d_k}$ in Scaled Dot-Product Attention? What happens if you don't?</summary>

**A:** Assuming each element of $Q$ and $K$ is i.i.d. with mean 0 and variance 1, each element of $QK^\top$ has variance $d_k$. When $d_k$ is large, dot-product values are large in magnitude, pushing softmax gradients toward zero (outputs approach one-hot), making training difficult. Dividing by $\sqrt{d_k}$ stabilizes the variance near 1.

> **Follow-up:** Are there other scaling approaches?
> A: Cosine attention uses $\frac{QK^\top}{\|Q\|\|K\|}$; some works also explore learnable temperature parameters.

</details>

---

<details>
<summary>Q11. What is the benefit of initializing $B$ to zero and $A$ randomly in LoRA?</summary>

**A:** $B = 0$ ensures that $\Delta W = BA = 0$ at the start of training, so the model behaves identically to the pretrained checkpoint and the initial state is not disrupted. If $A$ were also initialized to zero, $\Delta W$ would remain zero forever (zero gradient), and neither $A$ nor $B$ could learn anything. Therefore at least one matrix must be initialized non-zero.

> **Follow-up:** What role does the scaling factor $\alpha / r$ play?
> A: $\alpha$ controls the absolute magnitude of the LoRA update. With $\alpha$ fixed, increasing $r$ relatively reduces the update magnitude (dividing by a larger $r$), which aids training stability. In practice $\alpha = r$ or $\alpha = 2r$ are common choices.

</details>

---

<details>
<summary>Q12. Why are multiple heads needed in Multi-Head Attention? How does it differ from a single head with larger dimension?</summary>

**A:** Multiple heads can attend to different types of patterns (e.g., syntactic, semantic, positional relationships), acting as a form of implicit regularization. A single head with large dimension is theoretically equivalent in expressive power, but harder to optimize — the model tends to learn only one type of pattern. Experiments show that multi-head performs better at the same total compute.

> **Follow-up:** How different are the patterns learned by different heads? Are there visualization tools?
> A: Tools like BERTViz show that different heads do attend to different relationship types (e.g., some focus on adjacent words, others on subject-verb relationships). However, research also finds that some heads are redundant and can be pruned (head pruning).

</details>

---

<details>
<summary>Q13. What is the time complexity of K-Means? How does K-Means++ initialization improve convergence?</summary>

**A:** Each iteration requires computing distances from $n$ samples to $K$ centroids ($O(nKd)$); with $i$ iterations the total complexity is $O(nKdi)$. K-Means++ selects initial centroids sequentially with probability $\propto D(x)^2$ ($D(x)$ = distance to the nearest already-chosen centroid), making the initial centroids as spread out as possible. Theoretically it guarantees an expected WCSS within an $O(\log K)$ approximation ratio.

> **Follow-up:** How do you choose $K$?
> A: Common methods are the Elbow Method (plot WCSS vs $K$ and find the elbow) or the Silhouette Score (measures intra-cluster compactness vs. inter-cluster separation).

</details>

---

<details>
<summary>Q14. What does RMSNorm remove compared to LayerNorm? Why is RMSNorm commonly used in large models?</summary>

**A:** RMSNorm removes the mean-centering (re-centering) step, keeping only re-scaling. This eliminates the mean computation, reducing compute by roughly 10–15%. Experiments show that removing centering has little effect on performance, especially in large models; as a result mainstream models such as LLaMA and Qwen all adopt RMSNorm.

> **Follow-up:** What is the difference between Pre-Norm and Post-Norm? Why do modern large models mostly use Pre-Norm?
> A: Pre-Norm applies normalization before the sub-layer; Post-Norm applies it after. Pre-Norm provides more stable gradients and typically converges more easily without requiring learning rate warmup. Post-Norm is theoretically more expressive but requires more careful hyperparameter tuning (e.g., warmup).

</details>

---

<details>
<summary>Q15. What is the difference between <code>torch.Tensor.view()</code> and <code>torch.Tensor.reshape()</code>?</summary>

**A:** `.view()` requires the tensor to be contiguous in memory; otherwise it raises an error; it returns a view sharing the same memory. `.reshape()` is more flexible: if memory is already contiguous it is equivalent to `.view()`; otherwise it automatically copies data to make it contiguous. In performance-sensitive code prefer `.view()` (zero-copy), but ensure `.contiguous()` is called first.

> **Follow-up:** What operations can make a tensor non-contiguous?
> A: `transpose()`, `permute()`, certain `expand()` operations. These change the stride without moving data, so the logical order no longer matches the physical memory layout.

</details>

---

<details>
<summary>Q16. How does the Temperature parameter affect the sampling distribution? What do $T \to 0$ and $T \to \infty$ correspond to?</summary>

**A:** Temperature $T$ scales the logits: $p_i \propto e^{z_i / T}$. As $T \to 0$ the distribution degenerates to one-hot (greedy decoding); as $T \to \infty$ the distribution approaches uniform (fully random). $T < 1$ sharpens the distribution (more deterministic); $T > 1$ flattens it (more diverse).

> **Follow-up:** What temperature values are typically used in practice? Does the choice vary by task?
> A: Deterministic tasks such as code generation often use $T \approx 0$–$0.2$; creative writing tasks often use $T \approx 0.7$–$1.0$. Values above 2.0 are generally avoided as output quality degrades.

</details>

---

<details>
<summary>Q17. What are the three core elements of Dynamic Programming (DP)?</summary>

**A:** (1) **State definition** — what does $dp[i]$ represent (e.g., the optimal solution over the first $i$ elements); (2) **Transition equation** — how $dp[i]$ is derived from previous states; (3) **Base case and boundary** — the value of $dp[0]$. The most critical step is choosing the right state; a good state definition makes the transition natural and efficient.

> **Follow-up:** What is the difference between memoized search (top-down) and bottom-up tabulation? What are the pros and cons of each?
> A: Memoization uses recursion + caching, computes only needed states, and produces more intuitive code. Bottom-up uses iterative table-filling, has no recursion overhead, and is easier to space-optimize (rolling arrays). Time complexity is the same for both.

</details>

---

### L3 — Deep

<details>
<summary>Q18. What is the gradient of Softmax? Why is the gradient of Cross-Entropy + Softmax said to be elegant?</summary>

**A:** Let $p_i = \text{softmax}(z_i)$; then $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$ (the Jacobian matrix). When combined with Cross-Entropy $\mathcal{L} = -\log p_y$, the gradient simplifies to:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbb{1}[i = y]
$$

That is, predicted probability minus the true label (one-hot) — an extremely clean form. This is also why PyTorch fuses the two into a single operator.

> **Follow-up:** How does Label Smoothing change this gradient?
> A: Label smoothing replaces one-hot with $(1 - \epsilon)$ on the correct class and $\epsilon / (C-1)$ on other classes; the gradient becomes $p_i - \hat{y}_i$, preventing the model from becoming over-confident.

</details>

---

<details>
<summary>Q19. The computational complexity of Self-Attention is $O(N^2 d)$. What methods reduce this complexity?</summary>

**A:** Main directions include:
- **Sparse Attention:** Compute attention only within a local window (e.g., Longformer's sliding window) or specific patterns (e.g., BigBird's random + global + local)
- **Linear Attention:** Approximate softmax with a kernel function and reorder the computation of $QK^\top V$ as $Q(K^\top V)$, reducing complexity to $O(Nd^2)$
- **Flash Attention:** Does not change the mathematical result; optimizes GPU HBM/SRAM access patterns via tiling and recomputation, reducing memory usage and wall-clock time
- **Low-rank approximation:** E.g., Linformer projects $K, V$ to a lower dimension

> **Follow-up:** Does Flash Attention change the mathematical result of attention?
> A: No. Flash Attention is an IO-aware exact algorithm that uses block-wise computation (tiling) and an online softmax trick to produce a result mathematically equivalent to standard attention, while using GPU memory hierarchy more efficiently.

</details>

---

<details>
<summary>Q20. How should the rank $r$ in LoRA be chosen? What is the relationship between $\alpha$ and $r$?</summary>

**A:** Rank $r$ governs the tradeoff between expressive capacity and parameter count. Common range: $r \in [4, 64]$. More complex tasks or more target layers may require larger $r$. $\alpha$ is the scaling factor; the effective learning rate is roughly proportional to $\alpha / r$. A common approach is to fix $\alpha$ (e.g., $\alpha = 16$) and tune $r$, or to set $\alpha = r$ so the scaling factor is 1.

> **Follow-up:** Can different layers be assigned different ranks? What is the benefit?
> A: Yes — this is the idea behind methods like AdaLoRA. Different layers have different "importance" (analyzed via the singular value spectrum of $\Delta W$ through SVD); allocating higher rank to important layers and lower rank to unimportant ones achieves optimal allocation of the parameter budget.

</details>

---

<details>
<summary>Q21. Is K-Means guaranteed to converge? Is the solution guaranteed to be globally optimal?</summary>

**A:** K-Means is guaranteed to converge (WCSS decreases monotonically and is lower-bounded by 0), but it converges only to a **local optimum**. Both the assignment and update steps decrease or preserve WCSS, so WCSS never increases. However, the objective is non-convex, and different initializations can lead to different local optima. In practice, multiple random restarts (multi-start) are common to pick the best result, or K-Means++ is used to improve initialization quality.

> **Follow-up:** What shape constraints does K-Means impose on clusters? Are there alternative methods?
> A: K-Means assumes spherical (isotropic) clusters and performs poorly on non-spherical clusters. Alternatives include GMM (Gaussian Mixture Models, which allow elliptical clusters), DBSCAN (density-based, can discover arbitrarily shaped clusters), and Spectral Clustering.

</details>

---

<details>
<summary>Q22. What is the KV Cache in Transformer inference? Why is it needed?</summary>

**A:** During autoregressive generation, computing attention for a new token requires attending over all previous tokens. But the $K, V$ of previous tokens do not change when a new token is added, so they can be cached and reused. The KV Cache stores already-computed $K, V$ matrices; each step only needs to compute $Q, K, V$ for the new token and append the new $K, V$ to the cache. This reduces per-step compute from $O(n^2 d)$ to $O(nd)$ and is a key inference optimization.

> **Follow-up:** How large is the KV Cache memory footprint? What compression methods exist?
> A: For a model with $L$ layers, $h$ heads, head dimension $d_k$, and sequence length $n$, the KV Cache size is $2 \times L \times h \times d_k \times n \times \text{bytes}$. Compression methods include: GQA (Grouped Query Attention — multiple Q heads share K/V), quantization (reducing KV from FP16 to INT8/INT4), and sparsification (discarding unimportant KV pairs).

</details>

---

<details>
<summary>Q23. When writing Attention from scratch, why use <code>float('-inf')</code> in the causal mask rather than a large negative number?</summary>

**A:** `float('-inf')` becomes exactly 0 after softmax ($e^{-\infty} = 0$), so there is zero information leakage. A large negative number (e.g., $-10^9$) produces an extremely small but non-zero probability after softmax, theoretically leaking a trace of information. Additionally, `-inf` is numerically safer — the appropriate "large negative" threshold differs across precisions, whereas `-inf` behaves consistently at all precisions.

> **Follow-up:** Besides the causal mask, what other masks are commonly used in attention?
> A: Padding mask (to mask out padding tokens), Local/Sliding window mask (to limit the attention range), Prefix mask (bidirectional for the prefix portion, causal thereafter), Block diagonal mask (independent attention for multiple sequences).

</details>

---

<details>
<summary>Q24. Why does softmax require numerical stability tricks, yet users of <code>torch.nn.functional.cross_entropy</code> don't need to worry about this?</summary>

**A:** PyTorch's `F.cross_entropy` uses a more advanced numerically stable implementation — computing LogSumExp directly in log space, avoiding any explicit `exp()` + `log()` combination:

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}
$$

This is done in a single kernel, operating entirely in log space and avoiding overflow in intermediate results. A hand-written `softmax` + `log` + `nll` pipeline is step-by-step, and the intermediate `exp` can overflow.

> **Follow-up:** What is the difference between `torch.log_softmax` and `log(softmax(x))`?
> A: Mathematically equivalent but numerically different. `torch.log_softmax` directly applies the LogSumExp trick internally, avoiding an intermediate `softmax` that could overflow. With `log(softmax(x))`, if `softmax` already overflows to `inf`, then `log(inf)` is `inf`, but the gradient becomes `NaN`.

</details>

---

<details>
<summary>Q25. In distributed training (e.g., federated learning), how does using LoRA change communication overhead compared to full-parameter fine-tuning?</summary>

**A:** Full-parameter fine-tuning must communicate the entire $\Delta W$ ($d_{\text{in}} \times d_{\text{out}}$), whereas LoRA only needs to communicate $A$ ($r \times d_{\text{in}}$) and $B$ ($d_{\text{out}} \times r$), totaling $r \times (d_{\text{in}} + d_{\text{out}})$ parameters — far less when $r \ll d$. This is especially important in bandwidth-constrained settings such as cross-device federated learning.

> **Follow-up:** In federated learning aggregation with LoRA, do you aggregate $A, B$ or $\Delta W = BA$?
> A: Both approaches are viable. Aggregating $A, B$ directly has lower communication cost but is mathematically inexact (the mean of $BA$ does not equal the mean $B$ times the mean $A$). Aggregating $\Delta W$ is mathematically more correct but has higher communication cost. In practice the tradeoff depends on the use case.

</details>

---

## Appendix: Algorithm Pattern Quick Reference

### Two Pointers

```python
# opposing two pointers (find pair in sorted array)
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

### Sliding Window (variable size)

```python
l = 0
for r in range(len(arr)):
    # expand window: add arr[r]
    ...
    while window_condition_not_met:   # shrink
        # remove arr[l]
        l += 1
    # update answer
    ans = max(ans, r - l + 1)
```

### Binary Search (closed-open, find left boundary)

```python
lo, hi = 0, len(arr)      # search interval [lo, hi)
while lo < hi:
    mid = (lo + hi) // 2
    if arr[mid] < target:
        lo = mid + 1
    else:
        hi = mid            # retain mid
# lo == hi is the first position >= target
```

### Level-Order BFS

```python
from collections import deque
q = deque([root])
while q:
    for _ in range(len(q)):       # process level by level
        node = q.popleft()
        for child in node.children:
            q.append(child)
```

### Bottom-Up DP

```python
dp = [0] * (n + 1)
dp[0] = base_case
for i in range(1, n + 1):
    dp[i] = f(dp[i-1], dp[i-2], ...)
```

---

## Practice Log Template

| Problem | Topic | Difficulty | Status | Review Date |
|---------|-------|------------|--------|-------------|
| LC 1 | Hash Table / Two Sum | Easy | | |
| LC 3 | Sliding Window | Medium | | |
| LC 15 | Two Pointers / 3Sum | Medium | | |
| LC 33 | Binary Search | Medium | | |
| LC 56 | Interval Merge | Medium | | |
| LC 102 | Tree BFS Level-Order | Medium | | |
| LC 207 | Graph / Topological Sort | Medium | | |
| LC 300 | DP / LIS | Medium | | |
| LC 295 | Heap / Median of Data Stream | Hard | | |
| LC 42 | Monotonic Stack / Trapping Rain Water | Hard | | |

> **Status markers:** ✅ Fluent | ⚠️ Shaky | ❌ Unknown — schedule review only for ⚠️/❌.


## Extended L3

<details>
<summary>Q26. What is the theoretical basis for LoRA's low-rank assumption? In which scenarios might this assumption fail?</summary>

**A:** LoRA assumes that the effective rank of the weight update $\Delta W$ is far smaller than the parameter dimensions — that is, task-adaptation information is concentrated in a small number of directions. The theoretical basis is the concept of "intrinsic dimensionality": in the high-dimensional parameter space of a pretrained model, task adaptation actually occurs on a low-dimensional submanifold.

Failure scenarios:
- The downstream task is far from the pretraining distribution, requiring modification of high-rank structure (e.g., cross-lingual or cross-modal transfer)
- In multi-task settings, the low-rank subspaces of different tasks do not overlap, so their superposition raises the effective rank
- The model itself is small, making the low-rank constraint a capacity bottleneck

**Follow-up:** Beyond simply increasing $r$, what structured approaches can break the low-rank limitation?
**A:** AdaLoRA adaptively allocates rank per layer (higher rank for important layers); Adapter inserts a bottleneck with a nonlinear activation in the FFN, breaking the limitation of purely linear low-rank expressiveness; methods like (IA)^3 use vector scaling to provide additional expressiveness along a different efficiency-capacity curve.

</details>

---

<details>
<summary>Q27. Why does the Transformer use LayerNorm instead of BatchNorm? Analyze the fundamental difference between the two from the perspective of how statistics are computed.</summary>

**A:** BatchNorm computes mean and variance along the **batch dimension**; LayerNorm computes them along the **feature dimension**. The fundamental differences are:

1. **Variable sequence length:** In NLP, sequences within the same batch have different lengths; padding positions introduce noise into BatchNorm's statistics.
2. **Autoregressive generation:** At inference time, batch size is often 1, making BatchNorm's statistics extremely unstable.
3. **Each token normalized independently:** LayerNorm stabilizes the input distribution at each layer without depending on other samples in the batch.

**Follow-up:** What is the difference between Pre-LN and Post-LN? Why do modern large models mostly use Pre-LN?
**A:** Post-LN: $\text{LN}(x + \text{Sublayer}(x))$; Pre-LN: $x + \text{Sublayer}(\text{LN}(x))$. Pre-LN allows gradients to flow back directly through the residual path (residual stream) without passing through the nonlinear transformation of LN, making training more stable and typically eliminating the need for learning rate warmup. In Post-LN, deep-network gradients must pass through LN, which can cause training instability.

</details>

---

<details>
<summary>Q28. What is the core idea of FlashAttention? Why can it significantly improve efficiency without changing the mathematical result?</summary>

**A:** The core is **IO-aware block computation (tiling)**: divide $Q, K, V$ into small tiles, perform fused softmax + matrix multiplication on SRAM (on-chip cache), and avoid writing the $N \times N$ attention matrix back to HBM (high-bandwidth memory).

Key techniques:
- **Online softmax:** Exploit the decomposability of softmax — maintain a running max and running sum while processing each tile, scale results with a correction factor; the final output is mathematically equivalent to global softmax.
- **Recomputation:** During backward pass, do not store the attention matrix; recompute it from $Q, K, V$ — trading compute for memory.

**Follow-up:** How does FlashAttention's recomputation compare to general gradient checkpointing?
**A:** Both embody the "trade compute for memory" principle. Gradient checkpointing is a general strategy: selectively not saving intermediate activations, then recomputing them via a forward pass during backprop. FlashAttention's recomputation is more specialized: it targets specifically the large attention matrix tensor and combines with the tiling strategy to reduce not only memory but also HBM access count (IO complexity drops from $O(N^2)$ to $O(N^2 d^2 / M)$, where $M$ is SRAM size). The two can be used together.

</details>

---

<details>
<summary>Q29. RMSNorm removes the mean-centering step from LayerNorm. Why does this still work well in practice?</summary>

**A:** Full LayerNorm: $y_i = \gamma \cdot \frac{x_i - \mu}{\sigma} + \beta$; RMSNorm simplifies to: $y_i = \gamma \cdot \frac{x_i}{\text{RMS}(x)}$, removing $\mu$ and $\beta$.

Theoretical intuition for why this still works:
- After many linear transformations and activations in a deep network, feature means are often already in a reasonable range, or can be compensated by biases in subsequent layers.
- The aspect of normalization critical to training stability is **re-scaling** (controlling variance), not re-centering (subtracting the mean).
- Removing the mean computation eliminates one reduce operation; accumulated over a large-scale model this yields a meaningful efficiency gain.

**Follow-up:** In what situations might removing mean-centering be harmful?
**A:** If a layer's input has a systematic bias that subsequent layers cannot easily compensate, the bias will propagate. The effect may be more pronounced in small or shallow networks. However, the residual connections and deep stacking in large-scale Transformers provide sufficient capacity to compensate, and degradation has almost never been observed in practice.

</details>

---

<details>
<summary>Q30. In Multi-Head Attention, if multiple heads learn approximately the same pattern (head collapse), what are the consequences? How can it be detected and mitigated?</summary>

**A:** Head collapse leads to redundant multi-head representations — despite having $h$ heads' worth of parameters, the effective number of heads is far fewer than $h$, wasting compute and model capacity and effectively reducing the "effective rank" of the attention.

Detection methods:
- Compute KL divergence or cosine similarity between the attention distributions of different heads.
- Analyze the subspace overlap between $W^Q, W^K$ projection matrices (e.g., principal angles).

**Follow-up:** How can head collapse be mitigated at training time?
**A:**
- **Diversity regularization:** Add a regularization term to the loss that encourages diversity across head attention distributions (e.g., penalizing similarity).
- **Attention dropout:** Apply dropout to attention weights to prevent certain heads from dominating too early.
- **Head pruning + retraining:** Prune redundant heads after training and then fine-tune, forcing remaining heads to take on more responsibility; this also acts as implicit regularization.

</details>

---

<details>
<summary>Q31. After applying label smoothing to Cross-Entropy Loss, how does the gradient change? Why does it improve generalization?</summary>

**A:** In standard CE the target $q$ is one-hot ($q_y = 1, q_{j \neq y} = 0$); label smoothing sets $q_y = 1 - \epsilon, q_{j \neq y} = \frac{\epsilon}{C-1}$.

Gradient change ($\nabla_{z_i} \mathcal{L} = p_i - q_i$):
- Standard CE: $\nabla_{z_y} \mathcal{L} = p_y - 1$, $\nabla_{z_j} \mathcal{L} = p_j$
- Label smoothing: $\nabla_{z_y} \mathcal{L} = p_y - (1-\epsilon)$, $\nabla_{z_j} \mathcal{L} = p_j - \frac{\epsilon}{C-1}$

The model is no longer incentivized to push logits toward infinity, preventing over-confidence.

Generalization improvement: prevents the model from being overly confident in training labels (overfitting noisy labels) and implicitly imposes a soft constraint on logits.

**Follow-up:** What is the connection between label smoothing and knowledge distillation?
**A:** Both fundamentally replace hard targets with "soft targets" during training. Label smoothing softens targets with a uniform distribution; distillation uses the teacher's output distribution as soft targets. Label smoothing can be viewed as "distillation without a teacher, targeting a uniform distribution." The advantage of distillation is that the teacher's soft targets encode structural information about inter-class similarity (e.g., the cat vs. dog logit gap is smaller than cat vs. airplane), rather than the unstructured uniform distribution.

</details>

---

<details>
<summary>Q32. What is the mathematical connection between K-Means and the EM algorithm for a Gaussian Mixture Model (GMM)?</summary>

**A:** K-Means is the hard-assignment special case of GMM-EM in the limit of isotropic, equal-covariance, $\sigma \to 0$.

Apply these constraints to GMM $p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$: all components share a fixed isotropic covariance $\Sigma_k = \sigma^2 I$ and equal weights $\pi_k = 1/K$. The soft responsibilities in the E-step become a softmax with temperature $\sigma^2$ (equal weights and covariances cancel normalization constants); letting $\sigma \to 0$ degenerates to hard assignment:

$$
\gamma_{ik} = \frac{\exp\!\big(-\|x_i - \mu_k\|^2 / 2\sigma^2\big)}{\sum_j \exp\!\big(-\|x_i - \mu_j\|^2 / 2\sigma^2\big)} \;\xrightarrow{\;\sigma \to 0\;}\; \mathbf{1}\big[\,k = \arg\min_j \|x_i - \mu_j\|^2\,\big]
$$

The two steps correspond one-to-one:
- **E-step:** GMM's soft responsibility $\gamma_{ik}$ → K-Means hard assignment to nearest centroid (the $\sigma \to 0$ limit above).
- **M-step:** GMM's weighted mean $\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$ degenerates under hard $\gamma$ to the arithmetic mean of points in the cluster — exactly K-Means centroid update; both update only the mean.
- **Objective:** In this limit, the GMM negative log-likelihood (up to constants involving $\sigma$) is proportional to the K-Means WCSS $\sum_i \min_k \|x_i - \mu_k\|^2$.

**Follow-up:** Since K-Means is a special case of GMM, when should you use GMM?
**A:** When clusters are non-spherical, vary in size or density, or when soft (probabilistic) membership is needed. GMM additionally learns per-component covariance $\Sigma_k$ and weight $\pi_k$, fitting elliptical clusters of different orientations and scales and providing soft membership and uncertainty via $\gamma_{ik}$. K-Means, by assuming isotropic equal covariance, can only produce spherical (Voronoi) hard partitions. The tradeoff is that GMM has more parameters, is more sensitive to initialization and singular covariances (a component collapsing to a single point causes the likelihood to diverge), and often requires covariance lower bounds or regularization.

</details>
