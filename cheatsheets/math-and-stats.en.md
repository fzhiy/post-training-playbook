# Math & Statistics for ML Cheat Sheet

> Covers probability, linear algebra, calculus, information theory, MLE / MAP, and common distributions. Bilingual notation: primary language is English; established Chinese terms are referenced parenthetically where helpful.

> **Use cases**: ML/DL interview prep, course-note quick reference, technical blog reference.

---

## Part 1: Concepts & Formula Derivations

### 1.1 Probability Fundamentals

Basic relationships among **Conditional Probability**, **Joint Probability**, and **Marginal Probability**:

$$P(A, B) = P(A \mid B)\, P(B) = P(B \mid A)\, P(A)$$

$$P(A) = \sum_B P(A, B) \quad \text{(Marginalization)}$$

**Bayes' Theorem**:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

> posterior ∝ likelihood × prior.

**Chain Rule of Probability**:

$$P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid X_1, \ldots, X_{i-1})$$

**Conditional Independence**: $A \perp B \mid C$ means that given $C$, $A$ and $B$ are independent, i.e., $P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$.

---

### 1.2 Expectation, Variance, Covariance

| Property | Formula |
|------|------|
| Linearity | $E[aX + b] = a\,E[X] + b$ |
| Variance definition | $\text{Var}(X) = E[X^2] - (E[X])^2$ |
| Scaling | $\text{Var}(aX + b) = a^2\,\text{Var}(X)$ |
| Variance of a sum | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X,Y)$ |
| Covariance | $\text{Cov}(X,Y) = E[XY] - E[X]\,E[Y]$ |

**Covariance Matrix** $\Sigma$: diagonal entry $\Sigma_{ii} = \text{Var}(X_i)$; the matrix is always **positive semi-definite (PSD)** because $\forall w: w^\top \Sigma w = \text{Var}(w^\top X) \geq 0$.

---

### 1.3 Common Distributions

| Distribution | Parameters | Mean | Variance | Typical Use |
|------|------|------|------|----------|
| Bernoulli | $p$ | $p$ | $p(1-p)$ | Binary classification labels |
| Binomial | $n, p$ | $np$ | $np(1-p)$ | Count of successes in $n$ independent trials |
| Poisson | $\lambda$ | $\lambda$ | $\lambda$ | Counting rare events |
| Uniform | $[a, b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Parameter initialization |
| Gaussian | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | Noise models, weight priors |
| Exponential | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | Waiting-time modeling |

**Multivariate Gaussian PDF**:

$$\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\!\Big(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\Big)$$

**68-95-99.7 Rule**: For $\mathcal{N}(\mu, \sigma^2)$, approximately 68% of data falls within $\mu \pm 1\sigma$, 95% within $\mu \pm 2\sigma$, and 99.7% within $\mu \pm 3\sigma$.

---

### 1.4 Central Limit Theorem (CLT)

Let $X_1, \ldots, X_n$ be i.i.d. with mean $\mu$ and variance $\sigma^2$. As $n \to \infty$:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

**ML connection**: In mini-batch SGD, the batch gradient $\hat{g}_B$ is an unbiased estimator of the full-data gradient, with variance $\propto 1/B$ ($B$ = batch size). SGD noise provides implicit regularization in non-convex optimization, helping the model escape sharp minima.

---

### 1.5 Maximum Likelihood & Maximum A Posteriori

**MLE (Maximum Likelihood Estimation)**:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log P(x_i \mid \theta)$$

Purely data-driven; no prior information.

**MAP (Maximum A Posteriori)**:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \Big[\sum_i \log P(x_i \mid \theta) + \log P(\theta)\Big]$$

**Regularization equivalence**:

| Prior distribution | Corresponding regularization |
|----------|-----------|
| Gaussian prior $P(\theta) \propto e^{-\lambda \|\theta\|^2}$ | L2 regularization (Weight Decay) |
| Laplace prior $P(\theta) \propto e^{-\lambda \|\theta\|_1}$ | L1 regularization (Lasso) |

---

### 1.6 Bias-Variance Decomposition

$$E\!\Big[\big(\hat{f}(x) - f^*(x)\big)^2\Big] = \underbrace{\big(E[\hat{f}(x)] - f^*(x)\big)^2}_{\text{Bias}^2} + \underbrace{E\!\Big[\big(\hat{f}(x) - E[\hat{f}(x)]\big)^2\Big]}_{\text{Variance}} + \underbrace{\sigma_\epsilon^2}_{\text{Noise}}$$

- **Bias**: systematic deviation of the model → underfitting
- **Variance**: sensitivity of the model to perturbations in training data → overfitting
- **Noise**: irreducible error intrinsic to the data

Ensemble methods such as Bagging and Dropout primarily reduce Variance.

---

### 1.7 Core Linear Algebra

#### 1.7.1 Matrix Rank and Low-Rank Factorization

The rank $r = \text{rank}(W)$ of a matrix $W \in \mathbb{R}^{m \times n}$ is the maximum number of linearly independent rows (or columns), i.e., the dimension of the column space.

**Parameter efficiency of low-rank matrices**: If $\text{rank}(W) = r \ll \min(m, n)$, then $W$ can be factored as $W = AB$, where $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{r \times n}$.

- Original parameter count: $mn$
- Low-rank factorization parameter count: $r(m + n)$
- When $r \ll \min(m, n)$, parameter count is dramatically reduced

#### 1.7.2 Singular Value Decomposition (SVD)

Any matrix $W \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$W = U \Sigma V^\top$$

where $U \in \mathbb{R}^{m \times m}$ is orthogonal (left singular vectors), $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots)$ (singular values, $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$), and $V \in \mathbb{R}^{n \times n}$ is orthogonal (right singular vectors).

**Optimal Low-Rank Approximation (Eckart–Young–Mirsky Theorem)**:

$$\hat{W}_r = U_r \Sigma_r V_r^\top = \arg\min_{\text{rank}(M) = r} \|W - M\|_F$$

where $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{n \times r}$ are formed from the top $r$ singular values/vectors.

**Two norms expressed in terms of singular values**:

- **Frobenius norm**: $\|W\|_F = \sqrt{\sum_i \sigma_i^2}$
- **Spectral norm**: $\|W\|_2 = \sigma_1$ (largest singular value)

**ML applications**:
- PCA: right singular vectors of the covariance matrix = principal component directions
- Parameter compression: truncated SVD for model compression
- LoRA initialization strategy: truncated SVD of pretrained weights to initialize low-rank factors (see LoftQ and similar methods)

#### 1.7.3 Eigendecomposition vs. SVD

| | Eigendecomposition | SVD |
|--|-----------|-----|
| Applicability | Square matrices (must be diagonalizable) | Any $m \times n$ matrix |
| Factored form | $A = Q \Lambda Q^{-1}$ | $W = U \Sigma V^\top$ |
| Symmetric matrices | $A = Q \Lambda Q^\top$, $Q$ orthogonal | Same form; singular values = \|eigenvalues\| (eigenvalues = singular values when PSD) |
| Value range | Eigenvalues can be negative | Singular values $\geq 0$ |

When $A$ is symmetric positive definite (SPD), eigendecomposition and SVD are equivalent.

#### 1.7.4 Positive Definite Matrix

A symmetric matrix $A$ is positive definite if and only if:

1. All eigenvalues $> 0$
2. All leading principal minors $> 0$ (Sylvester's criterion)
3. A Cholesky decomposition $A = LL^\top$ exists
4. $\forall x \neq 0: x^\top A x > 0$

#### 1.7.5 Moore–Penrose Pseudoinverse

Given by SVD: $A^+ = V \Sigma^+ U^\top$, where $\Sigma^+$ replaces each nonzero singular value with its reciprocal. Used to compute the minimum-norm least-squares solution of $Ax = b$.

Closed-form solution of linear regression: $\hat{\theta} = (X^\top X)^{-1} X^\top y = X^+ y$. When $X^\top X$ is singular, add L2 regularization (Ridge regression): $\hat{\theta} = (X^\top X + \lambda I)^{-1} X^\top y$, which also improves the condition number.

#### 1.7.6 Matrix Multiplication Complexity

$A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$: computing $AB$ requires $O(mkn)$ FLOPs.

**LLM FLOPs estimate**: For a Transformer model with $N$ parameters trained on $D$ tokens, the total compute is approximately:

$$\text{FLOPs} \approx 6ND$$

(forward pass ≈ 2ND, backward pass ≈ 4ND; inference / forward-only ≈ 2ND.) This is the canonical estimate used in the scaling law literature.

---

### 1.8 Calculus & Gradients

#### 1.8.1 Jacobian and Hessian

**Jacobian matrix**: first-order derivative of a vector function $f: \mathbb{R}^n \to \mathbb{R}^m$:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}, \quad J \in \mathbb{R}^{m \times n}$$

**Hessian matrix**: second-order derivative of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}, \quad H \in \mathbb{R}^{n \times n}$$

$H$ is symmetric. Positive definite → local minimum; negative definite → local maximum; indefinite → saddle point.

#### 1.8.2 Chain Rule and Backpropagation

For a composite function $L = f(g(h(x)))$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

Matrix form (vector → vector), using the **vector-Jacobian product (VJP)**:

$$\frac{\partial L}{\partial x} = J_h^\top J_g^\top \frac{\partial L}{\partial f}$$

Backpropagation is essentially the efficient computation of the VJP chain product from output to input, making use of activations cached during the forward pass.

---

### 1.9 Information Theory

#### 1.9.1 Entropy

$$H(X) = -\sum_x P(x) \log P(x)$$

Measures the uncertainty of a random variable. $H$ is maximized under a uniform distribution ($ = \log |X|$) and equals 0 for a deterministic distribution. Units: $\log_2$ → bits; $\ln$ → nats (deep learning frameworks typically use nats).

#### 1.9.2 Kullback–Leibler Divergence

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = E_P\!\left[\log \frac{P}{Q}\right]$$

**Key properties**:
- $D_{\text{KL}} \geq 0$ (proved via Jensen's inequality); equality holds iff $P = Q$
- **Asymmetric**: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$; not a distance metric
- **Forward KL** ($P \| Q$): mean-seeking; $Q$ tends to cover all of $P$'s support
- **Reverse KL** ($Q \| P$): mode-seeking; $Q$ tends to concentrate on the dominant mode of $P$

**ML applications**: The KL penalty $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ in RLHF constrains the policy from deviating too far from the reference model; the DPO objective implicitly contains a KL constraint.

#### 1.9.3 Cross-Entropy

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

Relationship to entropy and KL divergence:

$$\boxed{H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)}$$

When $P$ is a deterministic distribution (one-hot label), $H(P) = 0$, so minimizing cross-entropy $\Leftrightarrow$ minimizing $D_{\text{KL}}(P \| Q)$ $\Leftrightarrow$ MLE.

**Perplexity (PPL)** and its relationship to cross-entropy:

$$\text{PPL} = \exp\!\big(H(P, Q)\big) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^N \log Q(x_i)\right)$$

#### 1.9.4 Mutual Information

$$I(X; Y) = D_{\text{KL}}\!\big(P(X,Y) \| P(X) P(Y)\big) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$$

Measures the amount of information shared between $X$ and $Y$. $I(X;Y) = 0$ iff $X \perp Y$.

**ML applications**: Maximizing mutual information in representation learning (e.g., contrastive learning with InfoNCE loss); using MI for feature selection to find features most relevant to the label.

#### 1.9.5 Jensen's Inequality

For a convex function $f$: $f(E[X]) \leq E[f(X)]$.

**Key applications**:
1. Proving non-negativity of KL divergence: $D_{\text{KL}}(P\|Q) = E_P[-\log(Q/P)] \geq -\log E_P[Q/P] = 0$
2. Deriving the ELBO: $\log P(x) = \log E_{z \sim Q}\!\left[\frac{P(x,z)}{Q(z)}\right] \geq E_Q\!\left[\log \frac{P(x,z)}{Q(z)}\right] = \text{ELBO}$

---

### 1.10 Importance Sampling

When direct sampling from a target distribution $P$ is difficult, sample from a proposal distribution $Q$ and correct with importance weights:

$$E_P[f(x)] = E_Q\!\left[f(x) \frac{P(x)}{Q(x)}\right] \approx \frac{1}{N}\sum_{i=1}^N f(x_i) \frac{P(x_i)}{Q(x_i)}, \quad x_i \sim Q$$

The weight $w_i = P(x_i)/Q(x_i)$ is the importance weight. When $Q$ and $P$ diverge too much, variance can explode.

---

### 1.11 Closed-Form KL for Gaussians

For $P = \mathcal{N}(\mu_1, \Sigma_1)$ and $Q = \mathcal{N}(\mu_2, \Sigma_2)$:

$$D_{\text{KL}}(P \| Q) = \frac{1}{2}\!\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^\top \Sigma_2^{-1}(\mu_2 - \mu_1) - k + \ln\frac{|\Sigma_2|}{|\Sigma_1|}\right]$$

**Special case in VAE**: $Q(z) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$, $P(z) = \mathcal{N}(0, I)$:

$$D_{\text{KL}}(Q \| P) = -\frac{1}{2}\sum_{j=1}^d \!\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

### 1.12 Variational Inference & ELBO

Goal: infer the posterior $P(z \mid x) = P(x, z) / P(x)$, but $P(x) = \int P(x, z)\, dz$ is intractable.

Introduce a variational distribution $Q(z)$ and replace with the ELBO lower bound:

$$\log P(x) \geq \underbrace{E_{Q}\!\left[\log \frac{P(x, z)}{Q(z)}\right]}_{\text{ELBO}} = E_Q[\log P(x \mid z)] - D_{\text{KL}}(Q(z) \| P(z))$$

Equivalence: $\log P(x) = \text{ELBO} + D_{\text{KL}}(Q \| P(z \mid x))$. Maximizing ELBO $\Leftrightarrow$ minimizing $D_{\text{KL}}(Q \| P(z \mid x))$.

---

## Part 2: PyTorch Snippets

### 2.1 Numerically Stable Softmax

```python
import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)          # subtract max to prevent overflow
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

### 2.2 Cross-Entropy Loss from Scratch

```python
import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (N, C) — unnormalized scores
    targets: (N,)   — integer class labels
    returns: scalar loss
    """
    # numerically stable log-softmax
    log_probs = logits - logits.max(dim=-1, keepdim=True).values
    log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    # extract log-probability of the target class
    nll = -log_probs[torch.arange(len(targets)), targets]
    return nll.mean()
```

### 2.3 KL Divergence for VAE

```python
import torch

def kl_divergence_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Computes KL( N(mu, exp(log_var)) || N(0, I) )
    mu, log_var: (N, d)
    returns: (N,) per-sample KL values
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
```

### 2.4 SVD Low-Rank Approximation

```python
import torch

def low_rank_approx(W: torch.Tensor, r: int) -> torch.Tensor:
    """
    Returns the optimal rank-r Frobenius-norm approximation of W.
    W: (m, n), r: target rank
    """
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    # keep top r singular values
    return (U[:, :r] * S[:r]) @ Vt[:r, :]

# example
W = torch.randn(128, 64)
W_approx = low_rank_approx(W, r=8)
print(f"Approximation error (Frobenius): {(W - W_approx).norm():.4f}")
```

### 2.5 LoRA-Style Forward Pass

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Simplified LoRA linear layer: freezes pretrained weights, adds a trainable low-rank bypass."""

    def __init__(self, in_dim: int, out_dim: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        # pretrained weight (frozen)
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W.weight.requires_grad_(False)
        # low-rank bypass ΔW = B @ A
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        # init: A with Kaiming, B with zeros → ΔW = 0 at start
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x) + self.B(self.A(x)) * self.scaling
```

### 2.6 Gaussian MLE from Scratch

```python
import torch

def gaussian_mle(data: torch.Tensor):
    """
    data: (N, d)
    returns: (mu, var) — MLE estimates of mean and variance
    """
    mu = data.mean(dim=0)
    var = data.var(dim=0, unbiased=False)   # MLE uses N (not N-1)
    return mu, var

def gaussian_log_likelihood(data: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Computes Gaussian log-likelihood."""
    d = data.shape[-1]
    return -0.5 * (d * torch.log(2 * torch.pi * var) + ((data - mu) ** 2) / var).sum(dim=-1).mean()
```

### 2.7 MC Estimation of KL Divergence

```python
import torch

def mc_kl_divergence(log_p, log_q, samples, num_samples: int = 10000):
    """
    Estimates KL(P || Q) via Monte Carlo.
    log_p, log_q: callable functions returning log-probabilities given samples
    samples: samples drawn from P
    """
    return (log_p(samples) - log_q(samples)).mean()

# example: estimate KL between N(0,1) and N(1, 0.5)
def log_p(x): return -0.5 * (x ** 2 + torch.log(torch.tensor(2 * 3.14159)))
def log_q(x): return -0.5 * ((x - 1) ** 2 / 0.5 + torch.log(torch.tensor(2 * 3.14159 * 0.5)))

samples = torch.randn(100000)
print(f"MC KL estimate: {mc_kl_divergence(log_p, log_q, samples):.4f}")
```

---

## Part 3: Interview Questions (25 Questions)

### L1 — Basic

---

<details>
<summary>Q1. What is the relationship among conditional probability, joint probability, and marginal probability?</summary>

**A**: Joint probability $P(A,B) = P(A \mid B) P(B) = P(B \mid A) P(A)$; marginal probability is obtained by summing (or integrating) over the joint: $P(A) = \sum_B P(A, B)$. Bayes' theorem $P(A \mid B) = P(B \mid A) P(A) / P(B)$ ties all three together.

**Follow-up**: How does the chain rule of probability generalize to more than three variables? What does conditional independence $A \perp B \mid C$ mean?
> **A**: $P(A,B,C) = P(A)P(B|A)P(C|A,B)$. $A \perp B \mid C$ means that given $C$, knowing $A$ does not change beliefs about $B$: $P(A,B|C) = P(A|C)P(B|C)$. In a probabilistic graphical model this corresponds to $C$ blocking all paths from $A$ to $B$.

</details>

---

<details>
<summary>Q2. What are the basic rules for expectation and variance?</summary>

**A**:
- Linearity: $E[aX+b] = aE[X]+b$
- Variance: $\text{Var}(X) = E[X^2] - (E[X])^2$
- Scaling: $\text{Var}(aX+b) = a^2\text{Var}(X)$
- Variance of a sum of independent variables equals the sum of variances (no covariance term)

**Follow-up**: What do the diagonal entries of a covariance matrix represent? Why is a covariance matrix always positive semi-definite?
> **A**: The diagonal entries are the per-dimension variances $\text{Var}(X_i)$. Symmetry is immediate; PSD holds because $\forall w: w^\top \Sigma w = \text{Var}(w^\top X) \geq 0$ (variance is non-negative).

</details>

---

<details>
<summary>Q3. Name at least four common probability distributions with their mean, variance, and typical use cases.</summary>

**A**:

| Distribution | Mean | Variance | Use |
|------|------|------|------|
| Bernoulli $p$ | $p$ | $p(1-p)$ | Binary classification labels |
| Gaussian $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | Noise models, weight priors |
| Poisson $\lambda$ | $\lambda$ | $\lambda$ | Counting rare events |
| Uniform $[a,b]$ | $(a+b)/2$ | $(b-a)^2/12$ | Parameter initialization |

**Follow-up**: How is the multivariate Gaussian PDF written?
> **A**: $\mathcal{N}(x|\mu,\Sigma) = (2\pi)^{-d/2}|\Sigma|^{-1/2}\exp\!\big(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$.

</details>

---

<details>
<summary>Q4. What is Maximum Likelihood Estimation (MLE)? Give the mathematical definition.</summary>

**A**:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log P(x_i \mid \theta)$$

Select the parameters that maximize the probability of the observed data. Equivalent to minimizing the negative log-likelihood (NLL). Under a Gaussian assumption, MLE for regression is equivalent to minimizing MSE.

**Follow-up**: Under what conditions does MLE produce a biased estimate?
> **A**: A classic example: the MLE for the Gaussian variance (dividing by $N$ instead of $N-1$) is biased (underestimates variance). With small samples, MLE is prone to overfitting on complex models.

</details>

---

<details>
<summary>Q5. What is matrix rank? What are the properties of a low-rank matrix?</summary>

**A**: The rank of a matrix is the maximum number of linearly independent rows (or columns), equal to the dimension of the column space. A low-rank matrix $W \in \mathbb{R}^{m \times n}$ (rank $r \ll \min(m,n)$) can be factored as $W = AB$ ($A \in \mathbb{R}^{m \times r}, B \in \mathbb{R}^{r \times n}$), reducing the parameter count from $mn$ to $r(m+n)$.

**Follow-up**: What is the connection between low-rank factorization and parameter-efficient fine-tuning (PEFT)?
> **A**: LoRA assumes that the weight update $\Delta W$ during fine-tuning is low-rank, and approximates it using the factorization $\Delta W = BA$. Only $A$ and $B$ are trained while the original weights are frozen, dramatically reducing the number of trainable parameters and GPU memory usage.

</details>

---

<details>
<summary>Q6. What is the definition of entropy and its intuition?</summary>

**A**: $H(X) = -\sum_x P(x) \log P(x)$, measuring the uncertainty of a random variable. Entropy is maximized under a uniform distribution ($= \log |X|$) and is 0 for a deterministic distribution. The unit depends on the base of the logarithm: $\log_2$ → bits; $\ln$ → nats.

**Follow-up**: Why does the maximum entropy principle yield the Gaussian distribution?
> **A**: Under the constraints $\int P = 1$, $E[X] = \mu$, $\text{Var}(X) = \sigma^2$, maximizing $H(X)$ via Lagrange multipliers yields the Gaussian distribution as the optimal solution.

</details>

---

<details>
<summary>Q7. What is the difference between dot product and cosine similarity?</summary>

**A**:
- Dot product: $u \cdot v = \|u\|\|v\|\cos\theta$, influenced by both magnitude and direction
- Cosine similarity: $\cos\theta = \frac{u \cdot v}{\|u\|\|v\|}$, reflects only direction (angle), magnitude is normalized out

**Follow-up**: Why does Attention use scaled dot-product rather than cosine similarity?
> **A**: Scaled dot-product $QK^\top / \sqrt{d_k}$ is computationally efficient (matrix multiplication), and the norms of Q and K participate in the attention weights, increasing expressiveness. RoPE rotations preserve norms and encode only relative position information. Cosine similarity requires extra L2 normalization and limits expressiveness.

</details>

---

<details>
<summary>Q8. What are the Jacobian matrix and the Hessian matrix?</summary>

**A**:
- **Jacobian** $J \in \mathbb{R}^{m \times n}$: first-order derivative matrix of a vector function $f: \mathbb{R}^n \to \mathbb{R}^m$, $J_{ij} = \partial f_i / \partial x_j$
- **Hessian** $H \in \mathbb{R}^{n \times n}$: second-order derivative matrix of a scalar function, $H_{ij} = \partial^2 f / \partial x_i \partial x_j$, symmetric. Positive definite → local minimum; indefinite → saddle point

**Follow-up**: Why is the Hessian rarely used directly in large-model optimization?
> **A**: The Hessian has size $n \times n$ ($n$ = number of parameters); storing and computing it is infeasible for models with billions of parameters. Second-order methods like K-FAC and Shampoo use Kronecker factorization or block-diagonal approximations of the Hessian to reduce cost.

</details>

---

### L2 — Intermediate

---

<details>
<summary>Q9. What is the difference between MLE and MAP? How does regularization correspond to Bayesian priors?</summary>

**A**: MLE maximizes the likelihood $P(D|\theta)$; MAP maximizes the posterior $P(\theta|D) \propto P(D|\theta) P(\theta)$, adding a prior term $\log P(\theta)$. Gaussian prior → L2 regularization (Weight Decay); Laplace prior → L1 regularization (Lasso).

**Follow-up**: From a Bayesian perspective, what prior does AdamW's weight decay assume?
> **A**: It is equivalent to an isotropic Gaussian prior $P(\theta) \propto \exp(-\lambda \|\theta\|^2 / 2)$. AdamW decouples weight decay from the gradient update (decoupled weight decay), which differs subtly from L2-regularized Adam, but the Bayesian interpretation is the same.

</details>

---

<details>
<summary>Q10. What is the Bias-Variance Decomposition?</summary>

**A**: Prediction error = Bias² + Variance + irreducible noise. High bias → underfitting; high variance → overfitting. Increasing model capacity reduces bias but may increase variance, and vice versa — this is the bias-variance tradeoff.

**Follow-up**: Do ensemble methods (Bagging, Dropout as ensemble) primarily reduce bias or variance?
> **A**: Primarily reduce variance. Bagging averages multiple high-variance models, reducing variance by a factor of $1/M$ ($M$ = number of models), with almost no change in bias. Boosting mainly reduces bias.

</details>

---

<details>
<summary>Q11. What is SVD? How is it used for low-rank approximation?</summary>

**A**: Any matrix $W = U\Sigma V^\top$, where $U, V$ are orthogonal and $\Sigma$ is a diagonal matrix of singular values. The Eckart–Young theorem states that the truncated SVD $\hat{W}_r = U_r \Sigma_r V_r^\top$, retaining only the top $r$ singular values, is the optimal rank-$r$ Frobenius-norm approximation.

**Follow-up**: How are the Frobenius norm and spectral norm each determined by the singular values?
> **A**: $\|W\|_F = \sqrt{\sum_i \sigma_i^2}$ (square root of the sum of squared singular values); $\|W\|_2 = \sigma_1$ (largest singular value).

</details>

---

<details>
<summary>Q12. What is the difference between eigendecomposition and SVD?</summary>

**A**: Eigendecomposition applies only to square matrices (and requires diagonalizability), $A = Q\Lambda Q^{-1}$; SVD applies to any matrix of any shape, $W = U\Sigma V^\top$, with singular values always non-negative. For symmetric positive definite matrices, eigendecomposition and SVD are equivalent.

**Follow-up**: What is the geometric meaning of the eigenvectors of a covariance matrix?
> **A**: The eigenvectors of the covariance matrix are the principal axis directions (principal directions) of the data distribution, corresponding to the principal component directions in PCA. The magnitude of each eigenvalue represents the variance along that direction.

</details>

---

<details>
<summary>Q13. What is the definition of KL divergence and its key properties?</summary>

**A**: $D_{\text{KL}}(P\|Q) = E_P[\log(P/Q)]$. Properties: (1) $D_{\text{KL}} \geq 0$; (2) asymmetric, not a distance metric; (3) $D_{\text{KL}} = 0 \Leftrightarrow P = Q$.

**Follow-up**: How do Forward KL and Reverse KL differ in their fitting behavior?
> **A**: Forward KL ($P \| Q$) is mean-seeking: $Q$ tends to cover all high-probability regions of $P$, preserving diversity. Reverse KL ($Q \| P$) is mode-seeking: $Q$ tends to concentrate on a single peak of $P$ and may ignore other modes. Variational inference typically uses Reverse KL.

</details>

---

<details>
<summary>Q14. What is the relationship among cross-entropy, KL divergence, and entropy?</summary>

**A**: $H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$. When $P$ is a one-hot distribution, $H(P) = 0$, so minimizing cross-entropy = minimizing KL divergence = MLE.

**Follow-up**: What is the relationship between a language model's perplexity (PPL) and cross-entropy?
> **A**: $\text{PPL} = \exp(H(P, Q))$, where $H$ is the cross-entropy. PPL can be understood as the average number of effective choices the model has at each token position; lower PPL means a better model.

</details>

---

<details>
<summary>Q15. What is Jensen's inequality? What are its applications in ML?</summary>

**A**: For a convex function $f$: $f(E[X]) \leq E[f(X)]$. Applications: (1) proving non-negativity of KL divergence; (2) deriving the variational lower bound ELBO; (3) proving convergence of the EM algorithm.

**Follow-up**: What is variational inference (VI)? What is the relationship between the ELBO and KL divergence?
> **A**: VI approximates the complex posterior $P(z|x)$ with a simple distribution $Q(z)$. $\log P(x) = \text{ELBO} + D_{\text{KL}}(Q \| P(z|x))$; maximizing ELBO is equivalent to minimizing the KL divergence between the variational posterior and the true posterior.

</details>

---

<details>
<summary>Q16. What are the necessary and sufficient conditions for a positive definite matrix?</summary>

**A**: For a symmetric matrix $A$ to be positive definite: (1) all eigenvalues $> 0$; (2) all leading principal minors $> 0$ (Sylvester's criterion); (3) a Cholesky decomposition $A = LL^\top$ exists; (4) $\forall x \neq 0: x^\top Ax > 0$.

**Follow-up**: Why is a covariance matrix always positive semi-definite?
> **A**: For any $w$, $w^\top \Sigma w = \text{Var}(w^\top X) \geq 0$ (variance is non-negative). Being strictly positive definite requires the variance to be strictly greater than zero (no deterministic linear relationship exists).

</details>

---

<details>
<summary>Q17. What is the computational complexity of matrix multiplication? How are LLM training FLOPs estimated?</summary>

**A**: $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$: $AB$ requires $O(mkn)$ FLOPs. LLM training estimate: with $N$ parameters and $D$ training tokens, total FLOPs $\approx 6ND$ (forward ≈ 2ND, backward ≈ 4ND; inference / forward-only ≈ 2ND).

**Follow-up**: Why is the decode phase of LLM inference typically memory-bound rather than compute-bound?
> **A**: Autoregressive decoding generates only 1 token per step, collapsing the batch dimension. Matrix multiplication degenerates to matrix-vector multiplication, which has very few FLOPs but requires loading the entire model weights (IO-intensive), leaving GPU compute underutilized.

</details>

---

### L3 — Deep

---

<details>
<summary>Q18. What is the relationship between the Central Limit Theorem (CLT) and SGD noise? Why does SGD noise have a regularization effect?</summary>

**A**: The CLT guarantees that the mini-batch gradient $\hat{g}_B = \frac{1}{B}\sum_i \nabla L(x_i)$ is an unbiased estimator of the full-data gradient, with variance $\propto 1/B$. SGD gradient noise helps the model escape sharp minima and tends to converge to flat minima. Flat minima generally generalize better, which amounts to implicit regularization.

**Follow-up**: What is the effect of increasing batch size during training?
> **A**: Gradient variance decreases ($\propto 1/B$), making optimization more stable but potentially losing the regularization effect and degrading generalization. Requires paired learning rate adjustment (linear scaling rule).

</details>

---

<details>
<summary>Q19. What is the fundamental difference in fitting behavior between Forward KL and Reverse KL? In what scenarios is each used?</summary>

**A**:
- **Forward KL** ($D_{\text{KL}}(P \| Q)$, I-projection): minimizing it forces $Q$ to have mass wherever $P > 0$ → mean-seeking / zero-avoiding. Suitable for preserving multi-modal coverage.
- **Reverse KL** ($D_{\text{KL}}(Q \| P)$, M-projection): minimizing it pushes $Q$ to have negligible mass where $P \approx 0$ → mode-seeking / zero-forcing. Suitable for concentrating on the dominant mode.

Variational inference (VI) uses Reverse KL by default (it arises naturally from the ELBO derivation); the KL constraint $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ in RLHF is **Reverse KL** (mode-seeking): it pulls policy $\pi_\theta$ toward the reference and penalizes deviation from high-probability regions of $\pi_{\text{ref}}$, suppressing reward-hacking distribution drift.

**Follow-up**: Are there practical applications of mixing Forward/Reverse KL?
> **A**: The Rényi-$\alpha$ divergence family interpolates between the two for $\alpha \in (0,1)$ and has been used in variational inference and reinforcement learning to trade off coverage against concentration.

</details>

---

<details>
<summary>Q20. Why does Attention use scaled dot-product rather than cosine similarity?</summary>

**A**: Scaled dot-product $\text{softmax}(QK^\top / \sqrt{d_k})$ advantages: (1) efficient matrix multiplication; (2) the norms of Q and K participate in attention weight computation, increasing expressiveness; (3) the scaling factor $\sqrt{d_k}$ prevents dot products from becoming too large and causing softmax saturation. Cosine similarity loses magnitude information and requires extra normalization.

RoPE (Rotary Position Embedding) only changes the direction (angle) of Q and K, keeping norms unchanged, so positional information is naturally incorporated through the angular difference in the dot product.

**Follow-up**: What happens if Q and K are L2-normalized before computing the dot product?
> **A**: This is equivalent to cosine similarity (with temperature scaling). Some work (e.g., Normalized Attention) has explored this direction; the advantage is that attention weights are norm-independent and more stable, but it may limit the ability to differentiate expressiveness among tokens. Methods like CosFormer introduce cosine reweighting while maintaining linear attention efficiency.

</details>

---

<details>
<summary>Q21. In what scenarios is reverse-mode automatic differentiation (AD) more efficient, and when is forward-mode AD preferred?</summary>

**A**:
- **Reverse-mode AD**: one forward pass + one backward pass computes gradients of a scalar loss with respect to all parameters, with complexity $O(1) \times$ the forward computation. Best for **many parameters, few outputs** (i.e., $f: \mathbb{R}^n \to \mathbb{R}$); this is the foundation of backpropagation in deep learning.
- **Forward-mode AD**: each pass propagates along one input direction to obtain the gradient of all outputs with respect to one input. Best for **few inputs, many outputs** (i.e., $f: \mathbb{R} \to \mathbb{R}^m$), such as computing Jacobian-vector products (JVPs).

**Follow-up**: Can a Hessian-vector product (HVP) be computed without explicitly forming the Hessian?
> **A**: Yes. $\nabla^2 f \cdot v = \nabla(\nabla f \cdot v)$; first apply forward-mode differentiation to $\nabla f \cdot v$ (or backpropagate through $\nabla f$), requiring only two differentiation passes with complexity comparable to a single gradient computation. This is the basis for implicit second-order methods such as conjugate gradient (CG).

</details>

---

<details>
<summary>Q22. How is importance sampling applied in PPO? What does the $\epsilon$ in PPO-Clip constrain?</summary>

**A**: PPO reuses trajectories collected under the old policy $\pi_{\theta_{\text{old}}}$ to train the new policy $\pi_\theta$, correcting for the distributional shift via the importance ratio $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$. The PPO-Clip objective:

$$L^{\text{CLIP}} = E_t\!\Big[\min\!\big(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\Big]$$

$\epsilon$ (typically 0.1–0.2) constrains the range of the importance ratio (not the weight itself). When $r_t$ falls outside $[1-\epsilon, 1+\epsilon]$, the gradient is cut off, preventing excessively large policy updates that would destabilize training.

**Follow-up**: If the importance ratio variance is too large, what are the alternatives?
> **A**: V-trace (introduced in IMPALA) truncates the ratio, or Retrace($\lambda$) and similar methods control variance. The core idea is to limit the effective importance weight to ensure convergence stability.

</details>

---

<details>
<summary>Q23. What is variational inference (VI) and the ELBO? How is it derived?</summary>

**A**: Approximate the true posterior $P(z|x)$ with a simple distribution $Q(z)$. Derivation:

$$\log P(x) = \log \int P(x,z)\,dz = \log E_Q\!\left[\frac{P(x,z)}{Q(z)}\right] \geq E_Q\!\left[\log\frac{P(x,z)}{Q(z)}\right] = \text{ELBO}$$

The inequality follows from Jensen's inequality ($\log$ is concave). ELBO = reconstruction term $E_Q[\log P(x|z)]$ − regularization term $D_{\text{KL}}(Q(z) \| P(z))$. $\beta$-VAE uses a hyperparameter $\beta$ to reweight the KL term, controlling the tradeoff between disentanglement and reconstruction quality.

**Follow-up**: Why is the ELBO maximized (rather than minimized)?
> **A**: The ELBO is a lower bound on $\log P(x)$. Maximizing ELBO → tighter lower bound → $Q(z)$ closer to the true posterior $P(z|x)$. Since $\log P(x) = \text{ELBO} + D_{\text{KL}}(Q \| P(z|x))$ and KL $\geq 0$, maximizing ELBO is equivalent to minimizing the KL divergence between the variational and true posteriors.

</details>

---

<details>
<summary>Q24. What is the theoretical and empirical basis for the low-rank assumption in parameter-efficient fine-tuning (PEFT)?</summary>

**A**: Theoretical basis — research on intrinsic dimensionality shows that fine-tuning a pretrained model actually takes place in a low-dimensional subspace far smaller than the full parameter space. LoRA parameterizes weight updates as $\Delta W = BA$ ($B \in \mathbb{R}^{m \times r}, A \in \mathbb{R}^{r \times n}$), requiring only $r(m+n)$ parameters ($r \ll \min(m,n)$) — orders of magnitude fewer than the original $mn$. Empirically, $r = 4 \sim 64$ is sufficient to approach full fine-tuning performance across a variety of downstream tasks.

**Follow-up**: How should the rank $r$ and which layers to apply LoRA to be chosen?
> **A**: Rank $r$ is typically selected via a small-scale search on the validation set. In practice, applying LoRA to the Q/K/V/O projection matrices of the attention layers works well. More advanced methods such as AdaLoRA adaptively allocate different ranks to each layer based on the singular value distribution. Choosing $r$ requires balancing expressiveness against parameter efficiency.

</details>

---

<details>
<summary>Q25. What is the closed-form KL divergence for Gaussian distributions? How is it used in a VAE?</summary>

**A**: For two multivariate Gaussians $P = \mathcal{N}(\mu_1, \Sigma_1)$ and $Q = \mathcal{N}(\mu_2, \Sigma_2)$:

$$D_{\text{KL}}(P \| Q) = \frac{1}{2}\!\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^\top\Sigma_2^{-1}(\mu_2-\mu_1) - k + \ln\frac{|\Sigma_2|}{|\Sigma_1|}\right]$$

In a VAE with $Q(z) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $P(z) = \mathcal{N}(0, I)$, this simplifies to:

$$D_{\text{KL}} = -\frac{1}{2}\sum_j (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

This KL term appears as a regularization term in the ELBO, encouraging the encoder output to stay close to the standard normal prior.

**Follow-up**: What is the effect of $\beta > 1$ in $\beta$-VAE?
> **A**: Increasing the KL term weight forces the encoder to match the standard normal prior more tightly, so the learned latent variable $z$ becomes more disentangled, but reconstruction quality may suffer. $\beta < 1$ has the opposite effect: a more flexible latent space but potentially at the cost of disentanglement.

</details>

---

*Last updated: 2025 | This cheat sheet is for study reference only; consult original literature for authoritative formulas.*

## Extended L3

<details>
<summary>Q26: What is the intrinsic connection between the Fisher Information Matrix (FIM) and the natural gradient? Why is the natural gradient the steepest descent in the KL sense?</summary>

  **A**: The Fisher information matrix is defined as the covariance of the log-likelihood gradient: $F(\theta) = E_{x \sim p_\theta}\!\left[\nabla_\theta \log p_\theta(x)\, \nabla_\theta \log p_\theta(x)^\top\right]$. It is also the negative expected Hessian of the log-likelihood (at the MLE), and is closely related to the local quadratic expansion of the KL divergence: $D_{\text{KL}}(p_\theta \| p_{\theta + \delta}) \approx \frac{1}{2}\delta^\top F(\theta)\,\delta$. Thus the FIM is the local metric tensor of the KL divergence in parameter space. The natural gradient is defined as $\tilde{\nabla} = F^{-1}\nabla L$, i.e., the direction of steepest descent in loss under the KL-ball constraint $\{δ : D_{\text{KL}}(p_\theta \| p_{\theta+\delta}) \leq \epsilon\}$. Intuitively: the ordinary gradient gives steepest descent in Euclidean space, but equal Euclidean steps in parameter space do not correspond to equal changes in the distribution; the natural gradient uses the FIM to correct this "distortion" so that optimization steps are uniform in distribution space.

  **Follow-up**: Why is the natural gradient not used directly in practice? What approximations exist? (Hint: K-FAC and EKFAC use block-diagonal Kronecker factorization of the FIM to reduce the cost of inversion.)

</details>

---

<details>
<summary>Q27: What is the mathematical essence of the reparameterization trick? Why does it reduce the variance of the gradient estimator?</summary>

  **A**: In a VAE, the gradient of the ELBO expectation $E_{z \sim q_\phi(z|x)}[\cdot]$ cannot be backpropagated through the sampling node directly (sampling is not differentiable). The reparameterization trick rewrites $z \sim q_\phi = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$ as $z = \mu_\phi + \sigma_\phi \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$. This shifts the randomness away from the parameters $\phi$ to a parameter-free noise variable $\epsilon$, while $z$ is deterministically differentiable with respect to $\phi$, so gradients can flow back to $\mu_\phi$ and $\sigma_\phi$ via standard backpropagation. Compared to the score function estimator (REINFORCE) $\nabla_\phi E_q[f(z)] = E_q[f(z)\nabla_\phi \log q_\phi(z)]$, the reparameterization trick directly encodes gradient information into the computation graph without relying on the scalar value $f(z)$ as a multiplicative weight, resulting in substantially lower variance. In essence, the score function estimator uses only the scalar value of $f(z)$ as a weight, while reparameterization exploits the local gradient $\partial f / \partial z$, providing a richer signal.

  **Follow-up**: For discrete latent variables (e.g., discrete VAE), the reparameterization trick does not directly apply. What are the alternatives? (Hint: Gumbel-Softmax / Concrete distribution uses a continuous relaxation to approximate discrete sampling while maintaining differentiability.)

</details>

---

<details>
<summary>Q28: How does the spectral norm $\|W\|_2 = \sigma_1$ of a matrix control the Lipschitz constant of a neural network? What does this imply for training stability?</summary>

  **A**: A function $f$ has Lipschitz constant $L$ if $\|f(x) - f(y)\| \leq L\|x - y\|$. For a linear layer $f(x) = Wx$, the Lipschitz constant equals exactly $\|W\|_2 = \sigma_1$ (largest singular value). For a $k$-layer network $f = W_k \cdots W_1$, the overall Lipschitz constant is $L \leq \prod_i \|W_i\|_2$, the product of spectral norms across layers. If $L$ is too large, small perturbations to inputs are amplified exponentially in the forward pass (exploding activations), and gradients are amplified exponentially in the backward pass (exploding gradients), destabilizing training. Spectral normalization (dividing each layer's weight by its spectral norm: $\hat{W} = W / \sigma_1$) constrains the per-layer Lipschitz constant to 1 and is widely used in GAN discriminator training to prevent mode collapse. The spectral norm can be efficiently approximated via power iteration without requiring a full SVD.

  **Follow-up**: Why is it necessary to impose a Lipschitz constraint on the GAN discriminator (e.g., via spectral normalization or gradient penalty)? What goes wrong without it? (Hint: The Wasserstein distance requires the discriminator to belong to the class of 1-Lipschitz functions; otherwise the objective is unbounded.)

</details>

---

<details>
<summary>Q29: What is the core identity of the trace trick (matrix trace trick)? In which ML derivations is it indispensable?</summary>

  **A**: The core identities are $\text{tr}(AB) = \text{tr}(BA)$ (cyclic permutation invariance) and the fact that a scalar equals its own trace: $x^\top A x = \text{tr}(x^\top A x) = \text{tr}(A x x^\top)$. These make it feasible to differentiate scalar functions of matrix products. Typical applications: (1) matrix differentiation in linear regression $\nabla_W \|Y - XW\|_F^2 = \nabla_W \text{tr}((Y-XW)^\top(Y-XW))$, expanded using $\nabla_W \text{tr}(AW) = A^\top$ to yield the closed-form solution; (2) differentiating $\log |\Sigma|$ and $x^\top \Sigma^{-1} x$ in the multivariate Gaussian log-likelihood; (3) deriving the MLE of the covariance matrix $\hat{\Sigma} = \frac{1}{N}\sum_i (x_i - \mu)(x_i - \mu)^\top$; (4) deriving the matrix Riccati equation in linear dynamical systems (Kalman filter). In short, the trace trick is the core tool for any setting involving differentiating matrix quadratic forms.

  **Follow-up**: Use the trace trick to derive the MLE for the precision matrix $\Lambda$ (inverse covariance) of a multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$. Hint: the log-likelihood contains $\text{tr}(\Lambda S)$ and $\log|\Lambda|$ terms. (Answer: $\hat{\Lambda} = S^{-1}$, i.e., the MLE precision matrix is the inverse of the sample covariance.)

</details>

---

<details>
<summary>Q30: What is the precise mathematical correspondence between the Schur complement and the conditional distribution of a multivariate Gaussian?</summary>

  **A**: Let $(x_a, x_b) \sim \mathcal{N}(\mu, \Sigma)$ and partition the covariance matrix as $\Sigma = \begin{pmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{pmatrix}$. The conditional distribution $P(x_a \mid x_b)$ is again Gaussian, and the top-left block of the joint precision matrix $\Sigma^{-1}$ is exactly $\Sigma^{aa}$, which is the Schur complement: $\Sigma^{aa} = (\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})^{-1}$. The conditional mean is $\mu_{a|b} = \mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b)$. This correspondence has far-reaching implications: (1) in Gaussian Markov Random Fields, zero entries in the precision matrix correspond to conditional independence relationships, encoding conditional structure more directly than the covariance matrix; (2) the posterior predictive in Gaussian Processes (GP) is fundamentally a conditional Gaussian, yielding a closed-form solution via the Schur complement; (3) the Kalman filter update step can also be understood from the Schur complement perspective.

  **Follow-up**: Why are Gaussian graphical models encoded using the precision matrix rather than the covariance matrix? What is the relationship between conditional independence and zero entries in the precision matrix? (Answer: $x_i \perp x_j \mid \text{rest}$ iff $\Sigma^{-1}_{ij} = 0$; the non-zero pattern of the precision matrix directly corresponds to the edge structure of the graph.)

</details>

---

<details>
<summary>Q31: What is the fundamental difference between Wasserstein distance and KL divergence for measuring distributional differences? Why does WGAN use Wasserstein rather than KL?</summary>

  **A**: KL divergence $D_{\text{KL}}(P \| Q)$ is infinite when the supports of $P$ and $Q$ do not overlap (even if they are geometrically "close"); in high-dimensional spaces this is almost always the case, since two low-dimensional manifold supports almost never overlap. Wasserstein distance (Earth Mover's Distance), based on optimal transport, is defined as $W(P, Q) = \inf_{\gamma \in \Pi(P,Q)} E_{(x,y)\sim\gamma}[\|x - y\|]$, and provides a meaningful finite distance even when supports do not overlap. Intuitive analogy: KL looks at the ratio $P/Q$ (and "explodes" if $Q = 0$ where $P > 0$); Wasserstein measures the minimum "work" required to transport one pile of "dirt" to another. Therefore, Wasserstein provides smooth gradient signals even when distributions do not overlap, while KL or JS divergence leads to vanishing gradients. WGAN uses the Wasserstein-1 distance as the generator objective, converting it via Kantorovich–Rubinstein duality to an optimization over 1-Lipschitz discriminators, resolving the training instability of classical GANs.

  **Follow-up**: The computational cost of Wasserstein distance is typically higher than that of KL divergence; how can it be approximated in high dimensions? (Hint: the Sinkhorn algorithm adds entropic regularization to optimal transport, converting a linear program into parallelizable matrix-scaling iterations; Sliced Wasserstein reduces high-dimensional problems to multiple one-dimensional problems via random projections.)

</details>

---

<details>
<summary>Q32: How does the spectral structure (spectrum) of the Hessian characterize the local geometry of the loss landscape? What does this imply for understanding saddle points and escape strategies in optimization?</summary>

  **A**: At a critical point (where the gradient is zero), the eigenvalue distribution of the Hessian $H$ determines the local geometry: (1) all eigenvalues $> 0$ (positive definite) → local minimum, the surface is bowl-shaped; (2) all eigenvalues $< 0$ (negative definite) → local maximum; (3) a mix of positive and negative eigenvalues (indefinite) → saddle point, with descent directions along negative-curvature directions. In high-dimensional parameter spaces, saddle points far outnumber local minima (because the probability of all eigenvalues being positive decays exponentially), making them the main obstacle in non-convex optimization. Escape strategies: (1) SGD gradient noise naturally provides perturbations along negative-curvature directions, helping escape saddle points — the geometric interpretation of SGD's implicit regularization effect; (2) momentum helps traverse flat regions; (3) explicit methods such as adding perturbations along Hessian negative-curvature directions (not yet common but theoretically effective). The eigenvalue spectrum of the Hessian is also related to generalization: the larger the eigenvalues of the Hessian at a minimum (the "sharper" the surface), the worse that minimum tends to generalize — this motivates sharpness-aware minimization (SAM) and related methods.

  **Follow-up**: Why is the saddle point problem more severe than the local minimum problem in extremely high-dimensional spaces? How does this follow from the eigenvalue distribution of the Hessian? (Answer: in an $n$-dimensional space, the Hessian has $n$ eigenvalues; at a random critical point each eigenvalue is equally likely positive or negative, so the probability that all are positive is $2^{-n}$, exponentially small — almost any critical point is a saddle point rather than a local minimum.)

</details>
