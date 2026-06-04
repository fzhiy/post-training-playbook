# LLM Post-Training Complete Reference Cheatsheet
# Complete Bilingual Cheat Sheet: LLM Post-Training

> Intended use: Interview preparation for LLM Post-Training Research Intern roles & everyday reference
> Language: English, with key technical terms preserved as-is

---

# Part 1 — Core Concepts & Formula Derivations
# Core Concepts & Formula Derivations

---

## 1. Pre-training vs Post-training Overview

| Dimension | Pre-training | Post-training |
|:---|:---|:---|
| **Data Scale** | Trillions of tokens, web-crawled corpora | Thousands–millions of high-quality annotated/preference examples |
| **Objective** | Next-token prediction; acquiring language and world knowledge | Instruction following + alignment to human preferences + enhanced reasoning |
| **Loss** | $L = -\sum_t \log p_\theta(x_t \mid x_{<t})$ | SFT loss + RLHF/DPO/GRPO objectives |
| **Learning Rate** | High (order of 1e-4), cosine annealing | Low (1e-5 ~ 5e-6), to prevent forgetting |
| **Hardware** | Thousands of GPUs, training for weeks to months | Hundreds of GPUs, training for hours to days |
| **Output** | Base model (highly capable but uncontrolled) | Instruct / Chat model (controllable, safe, helpful) |

**Standard 5-Step Pipeline:**

1. **SFT (Supervised Fine-Tuning)**: Supervised fine-tuning on high-quality (instruction, response) pairs to transform the base model into an "instruction assistant."
2. **Reward Model Training**: Train a scoring model RM using human preference comparison data (two responses to the same prompt + human preference labels).
3. **RLHF / PPO**: Reinforcement learning using RM feedback, with a KL constraint to prevent diverging too far from the SFT model.
4. **DPO (Offline Alternative)**: Bypasses the explicit RM; directly optimizes the policy from preference data, achieving simpler and more stable alignment.
5. **Iterative Loop**: Current policy samples new data → new preference labels → update RM → RL again, repeated over multiple rounds.

---

## 2. SFT Data Format & Loss Masking

**Chat Template (ChatML format example):**

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user instruction}<|im_end|>
<|im_start|>assistant
{assistant response}<|im_end|>
```

Different models have their own templates (Llama-2 uses `[INST]`, Llama-3 uses `<|start_header_id|>`, Qwen uses `<|im_start|>`). Training and inference must use the same template; otherwise a distribution shift occurs.

**Loss Masking:**

The training objective of SFT is to teach the model "how to answer," not "how to repeat the question." Cross-entropy loss is computed only at assistant token positions:

$$L_{SFT} = -\frac{1}{|A|} \sum_{t \in A} \log p_\theta(x_t \mid x_{<t})$$

where $A$ is the set of all assistant token positions. Labels for user / system tokens are set to $-100$ (ignored by default in PyTorch's `CrossEntropyLoss`).

**Multi-turn Loss Masking**: The user turn of every round is masked; only the assistant turn of each round contributes to the loss.

**Trade-off for including the system prompt in the loss**:
- Exclude (mainstream practice): saves capacity, focuses training on response quality.
- Include: the model better learns to follow system instructions, but introduces more noise.

### 2.1 Common tokenization / chat-template pitfalls (SFT engineering screening questions)

These are the most common failure modes in SFT engineering and are frequently tested in interviews (each: **problem** → fix):

- **`pad_token = eos_token`**: Many models (LLaMA/GPT-2) have no pad token; HF defaults to setting pad as eos. If pad positions are not masked in `attention_mask` and pad position labels are not set to `-100`, the model computes loss over pad tokens / learns to output eos everywhere. → Explicitly construct `attention_mask` to mask pad; set all prompt and pad labels to `-100`.
- **Applying chat template twice**: After `apply_chat_template` in data preprocessing, tokenization also sets `add_special_tokens=True` or wraps the template again → **duplicate BOS / duplicate special tokens**. → Apply the template only once; set `add_special_tokens=False` when tokenizing afterward.
- **Missing or inconsistent BOS**: BOS added during training but not at inference (or vice versa) → distribution shift (LLaMA-family is sensitive to BOS). → Verify whether the template already contains BOS; keep training/inference behavior consistent.
- **Tokenizer version / vocabulary mismatch**: Training and deployment use different versions or a version with added custom tokens, causing token ID misalignment. → Pin the tokenizer version; save it alongside the checkpoint.
- **Uninitialized newly added special tokens**: Adding new special tokens requires `resize_token_embeddings` (otherwise the token ID goes out of bounds); the newly added rows are randomly initialized and produce gibberish before training. → Initialize new rows properly (common heuristic: mean of existing embeddings, or reuse the vector of a semantically similar token — the mean is just a common heuristic, not the only option); ensure those new tokens appear in training data.
- **Packing cross-sample attention contamination**: When packing multiple documents into one sequence, without block-diagonal / `cu_seqlens` masking, tokens can attend across document boundaries (see §3). → Use varlen / cu_seqlens attention with correct separation.
- **Using the wrong model's template**: Llama-3 (`<|begin_of_text|>` / `<|start_header_id|>`), Qwen (`<|im_start|>`), Mistral (`[INST]`) have different formats; using the wrong one causes a sharp performance drop. → Use the target model's own `tokenizer.apply_chat_template`; do not hand-stitch concatenations.

> **Self-test (L2)**: Why does `pad_token=eos_token` break SFT if `attention_mask` and label mask are not set correctly? In multi-turn dialogue, how should labels for pad and prompt positions be handled?

---

## 3. Sequence Packing

**Definition**: Concatenate multiple short samples into a single sequence of length equal to the context window, adding EOS / separator tokens only at sample boundaries, thereby eliminating padding waste.

**GPU utilization**: Without packing, padding can account for 30–60% of tokens; with packing, nearly 100% of tokens are valid, yielding a 2–4× training speedup.

**Pitfall 1 — Cross-sample attention contamination**: Without a document-level attention mask, tokens from a preceding sample in a packed sequence can attend to tokens from a following sample, causing information leakage. Solution: use Flash Attention's `cu_seqlens` parameter, which takes the cumulative sequence lengths of each sample within the packed sequence and ensures attention is computed only within each sample.

**Pitfall 2 — Loss weight imbalance**: Packing implicitly weights by token count (longer samples produce more loss terms). If the original objective averaged over samples, the packing objective differs semantically; consider whether length normalization of the loss is needed.

**cu_seqlens example** (3 samples with lengths 5, 3, 7):
```
cu_seqlens = [0, 5, 8, 15]  # cumulative lengths
packed_ids = [s1_tok1, ..., s1_tok5, s2_tok1, ..., s2_tok3, s3_tok1, ..., s3_tok7]
```

---

## 4. RLHF Full Pipeline / PPO in RLHF

RLHF (Reinforcement Learning from Human Feedback) three stages:

1. **SFT**: Establish the initial policy $\pi_{ref}$ (reference policy, which serves as the baseline for the KL penalty).
2. **RM Training**: Fit a scalar reward function $r(x,y)$ from human preference comparison data using the Bradley-Terry model.
3. **PPO Optimization**: Maximize the augmented reward with a KL constraint.

**Augmented Reward:**

$$r_{total}(x,y) = r_{RM}(x,y) - \beta \cdot \text{KL}\!\left(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)\right)$$

**PPO Clipped Objective:**

$$L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio, and $\varepsilon$ is typically 0.1–0.2.

**GAE Advantage Estimation:**

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\lambda$ controls the bias-variance trade-off: $\lambda=1$ gives high variance, low bias; $\lambda=0$ degenerates to one-step TD.

**Recurrence (backward sweep, $O(T)$):** $\hat{A}_t=\delta_t+\gamma\lambda\,\hat{A}_{t+1}$, with $\hat{A}_T=\delta_T$.

- $\lambda=0$: $\hat{A}_t=\delta_t$, the one-step TD advantage (low variance, high bias).
- $\lambda=1$: $\hat{A}_t=\sum_l\gamma^l\delta_{t+l}$, the Monte-Carlo advantage (value function as a baseline only; low bias, high variance).
- **Key distinction:** $\gamma<1$ introduces bias regardless of $V$ accuracy (the discount itself); $\lambda<1$ introduces bias **only when $V$ is inaccurate** — so with a well-fit $V$, even $\lambda\to1$ is near-unbiased. Source: Schulman et al., [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) (ICLR 2016).

**4 Models Required by PPO (source of memory pressure):**

| Model | Role | Updated? |
|:---|:---|:---|
| **Actor** (Policy $\pi_\theta$) | The LLM policy being optimized | Yes (PPO gradient) |
| **Critic** (Value model) | Estimates $V(s_t)$, computes advantage | Yes (TD error) |
| **Reference** ($\pi_{ref}$) | KL penalty baseline, i.e., the SFT model | No (frozen) |
| **Reward Model** (RM) | Scores (x,y) pairs | No (frozen) |

**From-scratch implementation** (clipped policy loss + clipped value loss + entropy bonus + approx_kl monitoring):

```python
import torch

def ppo_loss(logp, logp_old, values, values_old, returns, advantages, entropy,
             clip_eps=0.2, vf_clip=0.2, vf_coef=0.5, ent_coef=0.0):
    # logp/logp_old: (B,) logprob of the taken action under current/behavior policy
    # advantages: normalized GAE advantage (for policy loss); returns: un-normalized GAE target (raw_adv + values_old, for value loss); both from upstream
    ratio = torch.exp(logp - logp_old)                      # importance ratio ρ_t
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_loss = -torch.min(surr1, surr2).mean()               # clipped policy loss

    v_clip = values_old + torch.clamp(values - values_old, -vf_clip, vf_clip)
    vf_loss = 0.5 * torch.max((values - returns) ** 2,
                              (v_clip - returns) ** 2).mean()  # clipped value loss

    loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()  # entropy bonus

    with torch.no_grad():                                   # diagnostics only
        logr = logp - logp_old                              # log(π_new/π_old)
        approx_kl = (torch.exp(logr) - 1 - logr).mean()     # K3: KL(π_old‖π_new) probe
        clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
    return loss, {"pg": pg_loss, "vf": vf_loss, "approx_kl": approx_kl, "clip_frac": clip_frac}
```

- Key points: ① **three loss terms** — clipped policy loss (the $L^{CLIP}$ above), clipped value loss (constrain the critic's single step around `values_old` to prevent value oscillation), and an entropy bonus (encourages exploration; `ent_coef` is often 0 or tiny); ② `approx_kl` uses the K3 estimator for **KL(π_old‖π_new)**, a **trust-region monitor** (early-stop the epoch if it exceeds a threshold) — a *different* KL from the $\beta\cdot\mathrm{KL}(\pi_\theta\|\pi_{ref})$ in the reward above: the former bounds the update step, the latter anchors to the reference policy; ③ both come from upstream: `advantages` is the **normalized** GAE advantage (fed to the policy loss), while `returns` is the **un-normalized** GAE target ($=$ raw advantage $+$ `values_old`, fed to the value loss) — the two are not interchangeable; this function only computes the minibatch loss; ④ `vf_coef` defaults to 0.5 as a **common heuristic weight** (not a principled gradient balance), most relevant when actor and critic share a backbone; `clip_frac` reports the **fraction of ratios outside the clip range**, which is not the same as the fraction whose objective term was actually clipped (that also depends on the advantage sign).

---

## 5. Bradley-Terry Reward Model

**Bradley-Terry preference model**: Given prompt $x$, the probability that the better response $y_w$ is preferred over the worse response $y_l$ is:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(r(x,y_w) - r(x,y_l)\right)$$

**RM training loss (maximizing log likelihood of preference data):**

$$L_{RM} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(r(x,y_w) - r(x,y_l)\right)\right]$$

**RM architecture**:
- Initialized from the SFT model (inherits language understanding capability).
- The LM head is removed and replaced with a linear layer $W \in \mathbb{R}^{d \times 1}$ that outputs a scalar reward.
- During training, a (chosen, rejected) response pair is fed in; both are forward-passed separately, taking the scalar output at the last token position to compute the Bradley-Terry loss.

**Key risks**:
- **Reward hacking**: the policy finds responses that score high under RM errors but are low quality (Goodhart's Law).
- **Distribution shift**: the RM breaks down on the policy's distribution when it has shifted away from the training distribution; iterative RM updates are needed.

---

---

## 6. DPO Full Derivation (Direct Preference Optimization Full Derivation)

### 6.1 Starting from the RLHF Objective

The KL-constrained RLHF optimization objective:

$$
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi(\cdot|x)}\!\big[r(x, y)\big] \;-\; \beta \cdot \mathrm{KL}\!\big(\pi(\cdot|x) \;\|\; \pi_{\mathrm{ref}}(\cdot|x)\big)
$$

where:
- $r(x, y)$ is the scalar reward from the reward model
- $\pi_{\mathrm{ref}}$ is the reference policy, typically the SFT model
- $\beta > 0$ controls KL penalty strength

Expanding the KL divergence:

$$
\max_{\pi} \; \mathbb{E}_{x,y}\!\big[r(x,y)\big] - \beta \sum_{y} \pi(y|x)\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}
$$

Taking the variational derivative per $y$ and setting it to zero:

$$
\frac{\partial}{\partial \pi(y|x)}\left[r(x,y) - \beta\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)} - \beta\right] = 0
$$

> Note: the normalization constraint $\sum_y \pi(y|x)=1$ introduces a Lagrange multiplier $Z(x)$

### 6.2 Closed-Form Optimal Policy

Solving yields the optimal policy:

$$
\boxed{\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)}
$$

where the partition function is:

$$
Z(x) = \sum_{y} \pi_{\mathrm{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

$Z(x)$ ensures $\sum_y \pi^*(y|x) = 1$, i.e., the policy is a valid probability distribution.

### 6.3 Inverting for the Reward

Taking log of both sides of the optimal policy:

$$
\log \pi^*(y|x) = \log \pi_{\mathrm{ref}}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)
$$

Rearranging:

$$
\boxed{r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)}
$$

**Key insight**: the reward can be expressed via the log-ratio of policy to reference, eliminating the explicit reward model.

### 6.4 Substituting into Bradley-Terry

Human preference model:

$$
p(y_w \succ y_l \mid x) = \sigma\!\big(r(x, y_w) - r(x, y_l)\big)
$$

where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function.

Substituting the inverted reward:

$$
r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} + \cancel{\beta\log Z(x)} - \cancel{\beta\log Z(x)}
$$

> **The $Z(x)$ terms cancel perfectly**! This is because both responses share the same partition function for the same prompt $x$.

$$
p(y_w \succ y_l \mid x) = \sigma\!\left(\beta\log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)
$$

### 6.5 DPO Loss Function

Replacing $\pi^*$ with parameterized $\pi_\theta$, taking negative log-likelihood:

$$
\boxed{\mathcal{L}_{\mathrm{DPO}}(\pi_\theta) = -\mathbb{E}_{(x,\, y_w,\, y_l) \sim \mathcal{D}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right]}
$$

Implicit reward defined as:

$$
\hat{r}_\theta(x, y) \triangleq \beta \log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}
$$

The loss simplifies to:

$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\!\big[\log\sigma\!\big(\hat{r}_\theta(x,y_w) - \hat{r}_\theta(x,y_l)\big)\big]
$$

### 6.6 Gradient Analysis

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta\,\mathbb{E}\!\left[\underbrace{\sigma(-\hat{r}_\theta(x,y_w)+\hat{r}_\theta(x,y_l))}_{\text{weight: larger gradient when the model is more wrong}}\!\left(\nabla_\theta\log\pi_\theta(y_w|x) - \nabla_\theta\log\pi_\theta(y_l|x)\right)\right]
$$

- When the model already ranks $y_w \succ y_l$ correctly, $\sigma(\cdot) \to 0$ and the gradient naturally decays
- When the model ranks incorrectly, $\sigma(\cdot) \to 1$ and the gradient is largest

### 6.7 DPO Advantages & Disadvantages

| Advantages | Disadvantages |
|---|---|
| No separate RM training needed | **Offline algorithm**: can only use the static dataset $\mathcal{D}$; no online exploration |
| No online rollout needed | **Distribution mismatch**: training signal degrades as $\pi_\theta$ drifts from the data collection policy |
| Simplified pipeline, single optimization pass | **Imprecise rejection**: rejects entire responses globally rather than correcting step by step |
| More stable than PPO | Sensitive to preference data quality |
| Theoretically equivalent to RLHF (with sufficient data) | $Z(x)$ cancellation depends on the correctness of the BT model assumption |

### 6.8 Likelihood Displacement: chosen log-prob also decreases

**Phenomenon**

Intuitively, DPO training should increase the model's probability for the chosen response $y_w$ and decrease it for the rejected response $y_l$. However, Razin et al. (arXiv:2410.08847) and Pal et al. (arXiv:2402.13228) both observe that **$\log\pi_\theta(y_w|x)$ and $\log\pi_\theta(y_l|x)$ tend to decrease simultaneously during training** — the loss decreases only because $y_l$ decreases faster, widening the margin between them, while the absolute probability of the chosen response shrinks.

> "While intuitively these methods should increase the probability of $y^+$ while decreasing that of $y^-$, several recent works observed that the probabilities of both $y^+$ and $y^-$ tend to decrease over the course of training." — Razin et al., arXiv:2410.08847

**Gradient Mechanism**

The DPO loss only constrains the log-prob **difference (margin)** relative to the reference model to widen:

$$\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

$\log\pi_\theta(y_w|x)$ itself has no lower-bound constraint — as long as $y_l$ decreases faster, the gradient objective is satisfied. Razin et al. (Theorem 1/3) note that when the hidden representations of $y_w$ and $y_l$ are similar (high CHES score), the gradient direction that suppresses $y_l$ simultaneously suppresses $y_w$, and probability mass shifts to tokens semantically **opposite** to $y_w$, forming "unintentional unalignment" (the phrase used in Razin et al.).

**Danger Conditions**

- Pairs in the dataset where $y_w$ and $y_l$ differ by only a few tokens (Pal et al. report a normalized edit distance of approximately 6.5% for the MetaMath subset), sharing large prefixes so that gradients are highly correlated.
- Preference pairs that are semantically similar with subtle differences (e.g., phrasing differences rather than factual differences).

**Detection**

Record both `chosen_logps_mean` and `rejected_logps_mean` throughout training (most training frameworks already log these). If the chosen mean consistently decreases beyond the reference baseline, displacement is occurring.

**Mitigation**

**(A) DPOP (Pal et al., arXiv:2402.13228)**: Adds a penalty term inside the DPO loss that directly prevents the chosen log-prob from falling below the reference model:

$$\mathcal{L}_\text{DPOP} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \lambda\cdot\max\!\left(0,\log\frac{\pi_\text{ref}(y_w|x)}{\pi_\theta(y_w|x)}\right)\right)\right]$$

The $\max(0,\cdot)$ term with $\lambda > 0$: when $\log\pi_\theta(y_w|x) < \log\pi_\text{ref}(y_w|x)$, a penalty is applied that "anchors" the chosen log-prob above the reference model.

**(B) CHES data filtering (Razin et al., arXiv:2410.08847)**: Filter out preference pairs where the representations of $y_w$ and $y_l$ are highly similar, cutting the gradient coupling path at the data level.

**(C) SimPO**: Uses length-normalized $\frac{1}{|y|}\log\pi_\theta(y|x)$ as the implicit reward and removes $\pi_\text{ref}$; the reward definition is directly aligned with generation-time likelihood, which by design weakens the driving force behind displacement (though the SimPO paper itself does not directly analyze this issue using the Razin/Pal framework).

> **Note**: The three mitigation approaches above come from different papers and should not be cross-attributed — the $\max$ regularization term in DPOP is from Pal et al.; CHES filtering is from Razin et al.; the connection between SimPO and displacement comes from downstream work and is provided here for reference only; it must not be attributed to either the Razin or Pal papers.

---
## 7. DPO Variants Comparison

### 7.1 IPO (Identity Preference Optimization / $\Psi$PO with $\Psi = \text{Id}$)

**DPO problem addressed**: DPO uses $\Psi(q) = \log(q/(1-q))$ (the logit function, corresponding to Bradley-Terry). When preferences approach certainty ($p^*(y_w \succ y_l) \to 1$), the logit tends to $+\infty$, driving $\pi^*(y_l) \to 0$ regardless of the KL penalty coefficient $\tau$ — KL regularization becomes ineffective under strong preferences, and the policy overfits the preference data.

**Core change**: Under the $\Psi$PO framework, replace $\Psi$ with the identity mapping (input preference probability $p\in[0,1]$, $\Psi(p)=p$ does not diverge as $p\to 1$ the way DPO's logit mapping does). The resulting empirical loss (Azar et al., arXiv:2310.12036, Eq. 17) is a squared-loss regression:

$$\mathcal{L}_{\text{IPO}}(\pi) = \mathbb{E}_{(y_w, y_l, x) \sim \mathcal{D}} \left[ \left( h_\pi(y_w, y_l, x) - \frac{1}{2\tau} \right)^2 \right]$$

where $h_\pi(y, y', x) = \log \dfrac{\pi(y|x)\,\pi_{\text{ref}}(y'|x)}{\pi(y'|x)\,\pi_{\text{ref}}(y|x)}$ is the log-ratio difference of the policy relative to the reference (logit margin); the target constant is $\frac{1}{2\tau}$, with $\tau$ the KL regularization strength.

**Citation**: Mohammad Gheshlaghi Azar et al. — arXiv:2310.12036 (Google DeepMind, 2023)

> "IPO, unlike DPO, always regularizes its solution towards $\pi_\text{ref}$ by controlling the gap between the log-likelihood ratios, thus avoiding the over-fitting to the preference dataset." — Azar et al., Section 5.2

**Properties**:
- The squared loss regresses the logit-margin to a fixed target $\frac{1}{2\tau}$; $\tau$ directly controls the upper bound of the learned log-ratio difference, keeping KL regularization effective at all times
- The gradient near the target may be small but is never zero — the solution cannot "escape" to an unbounded region
- **Note**: Azar et al.'s original validation is limited to small-scale bandit experiments; effectiveness at LLM scale requires independent verification
- **Trade-off**: The loss does not correspond to a BT probability model, slightly weakening theoretical interpretability

### 7.2 KTO (Kahneman-Tversky Optimization)

**DPO problems addressed**: (1) DPO requires paired preference data $(x, y_w, y_l)$, whereas in practice only pointwise positive/negative feedback (pointwise thumbs-up/down) is often available; paired data is expensive and scarce. (2) DPO maximizes preference log-likelihood, which is a proxy for the true objective of "maximizing generation utility," resulting in objective mismatch.

**Loss (complete form)**:

$$\mathcal{L}_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x,y \sim \mathcal{D}}\bigl[w(y)\bigl(1 - v_{\text{KTO}}(x,y;\beta)\bigr)\bigr]$$

where the implicit reward, KL baseline, and value function are defined as:

$$r_{\text{KTO}}(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

$$z_{\text{ref}} = \mathbb{E}_{x' \sim \mathcal{D}}\bigl[\beta\,\mathrm{KL}(\pi_\theta(y' \mid x') \,\|\, \pi_{\text{ref}}(y' \mid x'))\bigr]$$

$$v_{\text{KTO}}(x,y;\beta) = \begin{cases} \sigma(r_{\text{KTO}}(x,y) - z_{\text{ref}}) & \text{if } y \sim y_{\text{desirable}} \mid x \\ \sigma(z_{\text{ref}} - r_{\text{KTO}}(x,y)) & \text{if } y \sim y_{\text{undesirable}} \mid x \end{cases}$$

$$w(y) = \begin{cases} \lambda_D & \text{if desirable} \\ \lambda_U & \text{if undesirable} \end{cases}$$

**Role and implementation of $z_\text{ref}$ (KL Baseline)**

$z_\text{ref}$ is the expected KL divergence of the current policy relative to the reference model. In prospect theory it serves as the reference point — rewards above this point are "gains" and rewards below it are "losses," producing the sigmoid's concavity on the gain side (risk aversion) and convexity on the loss side (loss aversion).

In practice, $z_\text{ref}$ is estimated within each mini-batch (size $m$) using **mismatched** $(x', y'_U)$ pairs:

$$\hat{z}_{\text{ref}} = \max\!\left(0,\,\frac{1}{m}\sum_i \log\frac{\pi_\theta(y'_{U,i} \mid x'_i)}{\pi_{\text{ref}}(y'_{U,i} \mid x'_i)}\right)$$

Deliberately pairing prompt $x'$ with an **unrelated** output $y'_U$ is intentional, to avoid conflating the reward signal with the baseline estimate. Gradients do **not** propagate through the $z_\text{ref}$ term.

**Prospect Theory Mapping**

$v_\text{KTO}$ approximates the Kahneman-Tversky S-shaped value function with a logistic function (the original power-law form is hard to optimize directly): the sign flip — desirable branch $r - z_\text{ref}$; undesirable branch $z_\text{ref} - r$ — precisely simulates the "gain vs. loss" frame switch, and the asymmetric weights $\lambda_D / \lambda_U$ correspond to loss aversion.

> "KTO only requires a binary signal of whether an output is (un)desirable for a given input. This data is much more abundant, cheaper, and faster to collect in the real world than preferences." — Ethayarajh et al., arXiv:2402.01306

**Citation**: Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela — arXiv:2402.01306 (2024)

**Properties**:
- **No pairing required**: Each $(x, y)$ only needs a desirable/undesirable label; naturally binary logs such as thumbs-up/down can be used directly
- $z_\text{ref}$ provides a dynamic reference point, making the loss adaptive to KL drift
- **Trade-off**: Abandons the pairwise consistency constraint of the BT model, potentially lower information efficiency than pairwise methods; $\lambda_D / \lambda_U$ requires manual tuning

### 7.3 ORPO (Odds Ratio Preference Optimization)

**DPO problem addressed**: DPO requires two-stage training — first SFT then preference optimization — and requires maintaining a reference model $\pi_{\mathrm{ref}}$.

**Core change**: Directly attach an odds-ratio preference loss on top of the SFT cross-entropy loss (unified SFT + odds-ratio loss):

$$
\mathcal{L}_{\mathrm{ORPO}} = \underbrace{\mathcal{L}_{\mathrm{SFT}}(y_w)}_{\text{SFT on chosen}} + \lambda \cdot \underbrace{\left[-\log\sigma\!\left(\log\frac{\mathrm{odds}_\theta(y_w|x)}{\mathrm{odds}_\theta(y_l|x)}\right)\right]}_{\text{odds ratio preference}}
$$

where odds are defined as (odds defined as):

$$
\mathrm{odds}_\theta(y|x) = \frac{p_\theta(y|x)}{1 - p_\theta(y|x)}
$$

**Citation**: Jiwoo Hong, Noah Lee, James Thorne — arXiv:2403.07691 (2024)

> "In contrast to previous works, our approach requires neither an SFT warm-up stage nor a reference model, enabling resource-efficient development of preference-based aligned models." — Hong et al., arXiv:2403.07691

**Properties**:
- Single-stage training (Single-stage), no reference model needed (No reference model needed), halves the number of forward passes
- $\mathcal{L}_\text{SFT}$ provides domain-adaptation anchoring on chosen responses; $\mathcal{L}_\text{OR}$ simultaneously repels the rejected style
- **Trade-off**: The odds ratio has no direct theoretical connection to the BT model; the weight $\lambda$ between the two loss terms requires tuning; numerical results have not been independently verified here

### 7.4 SimPO (Simple Preference Optimization)

**DPO problems addressed**: (1) There is a divergence between DPO's implicit reward $\beta\log\pi_\theta/\pi_\text{ref}$ and the metric actually used at generation time (length-normalized likelihood) — Meng et al. note that in UltraFeedback triplets, the proportion of cases where the DPO reward ranking is satisfied but the log-likelihood ranking is reversed approaches half, meaning the model can "win the loss" while making the chosen response harder to generate. (2) Unnormalized log-probabilities decrease monotonically with length, allowing the model to satisfy the ranking by generating shorter rejected responses, introducing length bias. (3) Maintaining a frozen $\pi_\text{ref}$ incurs memory and compute overhead.

**Core change**: Replace the implicit reward with a length-normalized sequence-level average log-probability, and introduce an explicit target margin $\gamma > 0$ in the Bradley-Terry objective:

$$r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|}\log\pi_\theta(y|x)$$

$$\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

where $|y|$ is the token count, $\beta$ is a scaling constant, and $\gamma > 0$ requires the chosen reward to exceed the rejected reward by at least $\gamma$ (not merely be larger). Does not contain $\pi_\text{ref}$.

**Citation**: Yu Meng, Mengzhou Xia, Danqi Chen — arXiv:2405.14734 (NeurIPS 2024)

> "There is a divergence between DPO's reward formulation $r_\theta(x,y)=\beta\log\pi_\theta(y|x)/\pi_\text{ref}(y|x)$ and the average log likelihood metric $p_\theta(y|x)=\frac{1}{|y|}\log\pi_\theta(y|x)$, which directly impacts generation." — Meng et al., arXiv:2405.14734, Section 3.1

**Properties**:
- No reference model needed; reward directly aligns with generation-time likelihood (removes the source of distribution drift)
- Length normalization eliminates the shortcut of "short rejected response cheating"
- The target margin $\gamma > 0$ reinforces the absolute gap between chosen and rejected, not merely their relative ranking
- **Trade-off**: The loss does not correspond to a BT probability model; performance numbers (AlpacaEval/Arena-Hard) come from the paper abstract and have not been independently verified here

### 7.5 Online vs Offline DPO

**Distribution Mismatch**

Standard DPO is an **offline** algorithm: the preference dataset $\mathcal{D}$ is collected before training from some fixed data-generating policy $\mu$ (typically the SFT model). During training, $\pi_\theta$ is continuously updated while $\mathcal{D}$ remains static. Once $\pi_\theta$ diverges from $\mu$, the $(y_w, y_l)$ pairs in $\mathcal{D}$ no longer cover the current output distribution of $\pi_\theta$, creating an **off-policy distribution mismatch**.

Concrete manifestations:

- **Implicit reward drift**: DPO's implicit reward $\hat{r}_\theta(x,y) = \beta\log\pi_\theta(y|x)/\pi_\text{ref}(y|x)$ changes continuously during training; the scores assigned to the same $(y_w, y_l)$ pair shift as the policy updates, and the chosen/rejected margin may shrink or even reverse.
- **Limited exploration**: The policy cannot explore outputs better than those in $\mathcal{D}$, becoming stuck in a "local optimum" defined by the old preference data.

**Iterative / On-Policy DPO**

Solution: at each iteration, sample new response pairs using the **current** policy $\pi_\theta^{(t)}$, then construct new preference pairs $(y_w^{(t)}, y_l^{(t)})$ via a reward model (or human/AI judge), and update the policy with this batch of **distribution-matched** preference data:

$$\pi_\theta^{(t+1)} \leftarrow \text{DPO-update}\!\left(\pi_\theta^{(t)},\;\mathcal{D}^{(t)}\right), \quad \mathcal{D}^{(t)} \sim \pi_\theta^{(t)}$$

**Why online DPO is generally better**:

| Dimension | Offline DPO | Online / Iterative DPO |
|:---|:---|:---|
| Source of preference data | Statically pre-collected from fixed $\mu$ | Sampled each round from current $\pi_\theta$ |
| Distribution match | Off-policy, subject to drift | On-policy, matches current policy |
| Training signal quality | Limited by old distribution | Covers current policy's output distribution |
| Compute cost | Low (one-time data collection) | High (requires online sampling + RM scoring each round) |
| Exploration ability | None, locked to $\mathcal{D}$ | Can explore new output patterns |
| Representative methods | Standard DPO (Rafailov et al.) | RLHF-PPO, Online DPO, Self-Play Fine-Tuning |

**Practical guidance**: When access to an RM or automatic judge is available, iterative DPO (re-sampling + updating preference data every $k$ steps) generally yields better downstream conversation quality than purely offline DPO. If only offline is feasible, removing $\pi_\text{ref}$ (as in SimPO/DPOP) or adding chosen-anchoring can partially mitigate the likelihood displacement caused by distribution drift.

### 7.6 Precise Comparison Table

| Variant | Requires paired preferences? | Requires $\pi_\text{ref}$? | Key loss form | Main DPO problem corrected |
|:---:|:---:|:---:|:---|:---|
| **DPO** | ✅ Yes | ✅ Yes | $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})$ | Baseline (no explicit correction) |
| **IPO** | ✅ Yes | ✅ Yes | $\left(h_\pi(y_w,y_l) - \frac{1}{2\tau}\right)^2$, squared-loss regression | KL regularization fails under deterministic preferences; unbounded reward drift |
| **KTO** | ❌ No (pointwise binary label) | ✅ Yes | $w(y)(1 - v_\text{KTO}(x,y;\beta))$, asymmetric sigmoid + KL baseline $z_\text{ref}$ | Requires paired data; objective misaligned with actual generation utility |
| **ORPO** | ✅ Yes | ❌ No | $\mathcal{L}_\text{SFT} + \lambda(-\log\sigma(\log\frac{\text{odds}(y_w)}{\text{odds}(y_l)}))$ | Two-stage training; maintaining frozen $\pi_\text{ref}$ (doubles memory/compute) |
| **SimPO** | ✅ Yes | ❌ No | $-\log\sigma(\frac{\beta}{|y_w|}\log\pi_\theta(y_w) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l) - \gamma)$ | Likelihood displacement; length bias; $\pi_\text{ref}$ overhead |

**Citations**: DPO — Rafailov et al., arXiv:2305.18290; IPO — Azar et al., arXiv:2310.12036; KTO — Ethayarajh et al., arXiv:2402.01306; ORPO — Hong et al., arXiv:2403.07691; SimPO — Meng et al., arXiv:2405.14734 (NeurIPS 2024)

---

## 8. GRPO vs PPO (Group Relative Policy Optimization vs Proximal Policy Optimization)

### 8.1 GRPO Group Advantage Estimation

The core idea of GRPO: for the same prompt $x$, sample a group of responses $\{y_1, y_2, \dots, y_G\}$ and estimate the advantage using intra-group statistics (estimate advantage using intra-group statistics):

$$
\boxed{A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \dots, r_G\})}{\mathrm{std}(\{r_1, r_2, \dots, r_G\})}}
$$

where $r_i = r(x, y_i)$ is the reward for the $i$-th response (reward for the $i$-th response).

GRPO policy gradient loss (GRPO policy gradient loss):

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\left[\min\!\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\mathrm{old}}}(y_i|x)} A_i, \;\mathrm{clip}\!\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\mathrm{old}}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right)A_i\right)\right]
$$

> Same clipping mechanism as PPO, but entirely different advantage estimation (same clipping mechanism, entirely different advantage estimation).

### 8.2 Key Comparison

| Property | PPO | GRPO |
|:---|:---|:---|
| **Models required** | 4: Actor + Critic + Reference + RM | 2–3: Actor + Reference (+ optional RM; in RLVR settings reward comes from rules, no separate RM needed) |
| **Advantage estimation** | GAE (Generalized Advantage Estimation), requires a Critic network | Intra-group relative ranking, no Critic needed |
| **Memory overhead** | High (4 copies of model weights) | Low (2–3 copies of model weights) |
| **Reward source** | Learned neural RM (learned neural RM) | Typically verifiable/rule-based reward; neural RM can also be plugged in |
| **Suitable scenarios** | Open-ended dialogue, creative writing (open-ended generation) | Math reasoning, code generation (math, code with verifiable ground truth) |
| **Training stability** | Requires careful Critic tuning, otherwise unstable | More stable, no Critic estimation error |
| **Gradient variance** | Lower (GAE provides low-variance estimates) | Higher (limited group sample size $G$) |

### 8.3 RLVR Framework (RL from Verifiable Rewards)

The paradigm that best fits GRPO is **RLVR**: rewards come not from a learned RM but from automatically verifiable rules (rewards from automatically verifiable rules):

- **Math**: whether the answer equals the ground truth (answer matches ground truth) → $r = \mathbb{1}[\text{extract}(y) = y^*]$
- **Code**: whether all test cases pass (passes all test cases) → $r = \frac{\text{passed tests}}{\text{total tests}}$
- **Format**: whether the required format is followed (follows required format) → $r \in \{0, 1\}$

The core advantage of RLVR: **low-noise reward** (low-noise reward, relative to a learned RM), avoiding the bias and overfitting of the RM itself.

### 8.4 When to Prefer Which

- **Prefer GRPO**: rewards are automatically verifiable (math, code, logical reasoning); limited resources (cannot maintain 4 models); need stable training
- **Prefer PPO**: rewards require semantic/style judgment (dialogue quality, creative writing); reward signal is complex and cannot be rule-based; sufficient compute and a mature RM are available

### 8.5 RLOO and ReMax (critic-free baselines)

PPO relies on a learned critic (value network) to estimate a baseline. GRPO, RLOO, and ReMax are all **critic-free**, replacing the value baseline with a baseline computed from sampled rewards.

**RLOO** (REINFORCE Leave-One-Out, Ahmadian et al. 2024, ACL [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)): For a group of $G$ samples per prompt, the baseline for sample $i$ is the mean reward of the *other* $G-1$ samples; advantage $A_i = r_i - \frac{1}{G-1}\sum_{j\neq i} r_j$. It uses a pure REINFORCE gradient, no clipping, no critic; the policy-gradient estimate stays unbiased because the baseline does not depend on sample $i$'s own action.

**ReMax** (Li et al. 2024, ICML [arXiv:2310.10505](https://arxiv.org/abs/2310.10505)): The baseline is the reward of a single greedy (argmax) decode for the same prompt; advantage $A = r(\text{sample}) - r(\text{greedy})$. This requires only one extra greedy rollout per prompt, resulting in very low memory overhead and no critic network.

| Method | baseline | estimator | clip? | extra cost |
| :--- | :--- | :--- | :--- | :--- |
| GRPO | Group-relative (z-score) | PPO-style | Yes | $G$ samples |
| RLOO | Leave-one-out mean | REINFORCE | No | $G$ samples |
| ReMax | Greedy decode reward | REINFORCE | No | +1 greedy rollout |

---

## 9. Role of KL Penalty & Tuning β

### 9.1 Intuitive Role of the KL Term

$$
\beta \cdot \mathrm{KL}\!\big(\pi_\theta(\cdot|x) \;\|\; \pi_{\mathrm{ref}}(\cdot|x)\big)
$$

The KL penalty acts as **regularization**, with the following functions:

1. **Prevents excessive drift**: Ensures $\pi_\theta$ does not deviate too far from $\pi_{\mathrm{ref}}$, preserving pre-training knowledge
2. **Mitigates reward hacking**: If the policy learns to exploit flaws in the RM, the KL term grows as a penalty
3. **Maintains diversity**: Prevents the policy from collapsing to a few high-reward modes (mode collapse)
4. **Stabilizes training**: Constrains the exploration space, preventing excessively large policy updates

Expanding mathematically (Expanding mathematically):

$$
\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right] \geq 0
$$

> When $\pi_\theta = \pi_{\mathrm{ref}}$, KL = 0 (no penalty when the policy does not deviate at all from the reference policy).

### 9.2 KL-RM Score Pareto Frontier

Tuning $\beta$ is fundamentally a trade-off between two objectives:

$$
\underbrace{\mathbb{E}[r(x,y)]}_{\text{reward score}\uparrow} \quad \text{vs.} \quad \underbrace{\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})}_{\text{degree of deviation}\uparrow}
$$

| $\beta$ value | Effect |
|:---:|:---|
| $\beta$ too large | Policy barely updates, stays close to $\pi_{\mathrm{ref}}$, small reward improvement (underfitting) |
| $\beta$ too small | Policy updates aggressively, reward may be high but distribution shift is severe, risk of reward hacking |
| $\beta$ moderate | Finds a balance on the KL-RM frontier |

Typical range (typical range): $\beta \in [0.01, 0.5]$; $\beta = 0.1$ or $\beta = 0.2$ are commonly used in practice.

> 📝 **Quantitative version of over-optimization**: the gold-RM score traces an **inverted-U** in $\sqrt{\mathrm{KL}}$ (BoN form $d(\alpha-\beta d)$, RL form $d(\alpha-\beta\log d)$, with $d=\sqrt{\mathrm{KL}}$) — past the peak the policy drifts out of distribution and the gold score falls. See [reward-modeling-eval §3.2a](cheatsheet-reward-modeling-eval-en.html) (Gao et al. 2022, [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)).

### 9.3 β in DPO vs PPO

| Dimension | $\beta$ in PPO | $\beta$ in DPO |
|:---|:---|:---|
| **Mathematical role** | Controls the weight of the KL penalty term (in the loss function) | Controls the scaling of the implicit reward (in the log-ratio) |
| **Where it appears** | $\max_\pi \mathbb{E}[r] - \beta \cdot \mathrm{KL}$ | $\hat{r}(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$ |
| **Semantic equivalence** | Theoretically, DPO's $\beta$ originates from the same $\beta$ in the RLHF objective, but in practice the training dynamics differ, so both must be tuned separately |
| **Practical effect** | $\beta \uparrow$ → more conservative policy | $\beta \uparrow$ → implicit reward changes more sharply, more sensitive to preference signal |

> **Conclusion**: Theoretically equivalent (theoretically equivalent), practically different (practically different). In DPO, $\beta$ also influences the sharpness of the weight term $\sigma(\cdot)$ in the gradient.

### 9.4 KL Estimators & Placement

#### 9.4.1 Three Single-Sample Estimators

**Notation**: Let $r = \pi_{\mathrm{ref}} / \pi_\theta$ (Schulman 2020 convention), with samples drawn from the current policy $\pi_\theta$.

Define three estimators:

$$k_1 = -\log r = \log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$$

$$k_2 = \tfrac{1}{2}(\log r)^2 = \tfrac{1}{2}\!\left(\log\frac{\pi_{\mathrm{ref}}}{\pi_\theta}\right)^{\!2}$$

$$k_3 = (r - 1) - \log r = \left(\frac{\pi_{\mathrm{ref}}}{\pi_\theta} - 1\right) - \log\frac{\pi_{\mathrm{ref}}}{\pi_\theta}$$

**Verifying the expectation** (samples from $\pi_\theta$):

$$\mathbb{E}_{a \sim \pi_\theta}[r] = \sum_a \pi_\theta(a) \cdot \frac{\pi_{\mathrm{ref}}(a)}{\pi_\theta(a)} = \sum_a \pi_{\mathrm{ref}}(a) = 1$$

Therefore $\mathbb{E}[r - 1] = 0$: the term $(r-1)$ is zero-mean and serves as a control variate.

- **Unbiasedness of $k_1$**: $\mathbb{E}_{a \sim \pi_\theta}[k_1] = \mathbb{E}_{a \sim \pi_\theta}\!\left[\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right] = \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}}) \geq 0$. So $k_1$ is an unbiased estimate of the KL value.

- **Bias of $k_2$**: $k_2 = \tfrac{1}{2}(\log r)^2 \geq 0$, but its expectation under $\pi_\theta$ does not equal $\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$, so $k_2$ is biased.

- **Unbiasedness of $k_3$** (control-variate argument): For any $\lambda$, define $k_\lambda = -\log r + \lambda(r-1)$. Since $\mathbb{E}[r-1]=0$, we have $\mathbb{E}[k_\lambda] = \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$ for all $\lambda$ — universally unbiased. Setting $\lambda = 1$ yields $k_3$. At this choice, the added term $\lambda(r-1)$ is negatively correlated with $-\log r$, **reducing variance**. Hence $k_3$ is unbiased and, in the small-drift regime ($r \approx 1$) relevant to RLHF, has lower variance than $k_1$.

- **Non-negativity of $k_3$**: By the tangent-line inequality $\log r \leq r - 1$ for all $r > 0$ (equality only at $r=1$), we have $(r-1) - \log r \geq 0$, consistent with the non-negativity of KL divergence.

> **Order near $r=1$** (let $\epsilon = r-1$): $k_1 = -\log r \approx -\epsilon$ is **first-order** (signed), whereas $k_2 \approx k_3 \approx \tfrac{1}{2}\epsilon^2$ are **second-order** (non-negative). All three vanish as $r \to 1$ but at different rates — which is also why $k_3$ is non-negative like $k_2$ yet unbiased like $k_1$.

#### 9.4.2 The Gradient Perspective

The analysis above concerns **value estimation** only. When an estimator is used as a loss term, its gradient behavior must be analyzed separately.

- **$k_3$ as a loss term does not yield the exact reverse-KL gradient**: Although $k_3$ is an unbiased value estimate of KL, differentiating it with respect to $\pi_\theta$ (when used as a loss) produces a gradient that is only a first-order approximation of the true reverse-KL gradient. The approximation holds when policy drift is small ($r \approx 1$, so $\log r \approx r - 1$), but introduces systematic bias as drift grows.

- Per **Liu et al. (arXiv:2510.01555, "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization")**: **$k_1$ placed in the reward (in-reward)** and **$k_2$ used as a loss term (as-loss)** are gradient-principled choices, while **$k_3$ as a loss term lacks gradient-level justification**. In practice, the small $\beta$ used in GRPO / DeepSeek-R1 keeps this approximation error minor.

#### 9.4.3 Estimator Comparison Table

| Estimator | Form | Value-unbiased? | Gradient-principled? |
|:---|:---|:---|:---|
| $k_1$ | $-\log r$ | Yes | Yes, as in-reward |
| $k_2$ | $\tfrac{1}{2}(\log r)^2$ | No | Yes, as-loss |
| $k_3$ | $(r-1)-\log r$ | Yes | No, as-loss |

> **Variance note**: in the small-drift regime ($r \approx 1$), $k_3$ has lower variance than $k_1$ (both unbiased). $k_2$ is biased and operates in a different bias-variance regime; direct variance comparisons with $k_1$ or $k_3$ are not meaningful.

#### 9.4.4 Two Placement Styles (Style A vs Style B)

**Style A: In-Reward**

Representative: InstructGPT / PPO. The KL penalty is incorporated per-token into the reward signal:

$$r_{\mathrm{total}}(x, y) = r_{\mathrm{RM}}(x, y) - \beta \cdot k_1^{(t)}, \quad k_1^{(t)} = \log\frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{ref}}(a_t|s_t)}$$

- Each token receives an individual KL penalty signal; the Critic can learn stepwise KL costs.
- **Caveat**: PPO's clip mechanism truncates the policy ratio. For clipped tokens, the surrogate objective no longer depends on $\pi_\theta$, so **the KL signal in the reward is silently masked for those tokens at the gradient level** — an implementation detail that is easily overlooked.

**Style B: In-Loss**

Representative: GRPO (Shao et al., DeepSeekMath), DeepSeek-R1. The KL estimator is added directly to the policy optimization loss:

$$\mathcal{L} = \mathcal{L}_{\mathrm{GRPO}} + \beta \cdot k_3$$

where $k_3 = (r - 1) - \log r$, $r = \pi_{\mathrm{ref}} / \pi_\theta$ (computed per token, then averaged over the sequence).

- Requires no Critic / Value model, reducing memory footprint.
- **DAPO** (arXiv:2503.14476): removes the KL penalty entirely ($\beta = 0$). The stated rationale is that during long-CoT reasoning training the policy diverges substantially from the initial SFT reference, so a tight KL constraint is counterproductive; it instead relies on asymmetric clipping (Clip-Higher: decoupled upper/lower clip bounds) to prevent entropy collapse.

| Dimension | Style A (in-reward) | Style B (in-loss) |
|:---|:---|:---|
| Representative systems | InstructGPT, PPO | GRPO, DeepSeek-R1 |
| Estimator used | $k_1$ (per-token) | $k_3$ (per-token, averaged) |
| Gradient-principled | Yes | Approximate (acceptable at small $\beta$) |
| Engineering complexity | Requires Critic | No Critic needed |
| Clip-masking risk | Present (KL gradient silently dropped for clipped tokens) | Not applicable |

#### 9.4.5 Interview Self-Test

> **L2**: Using the $r = \pi_{\mathrm{ref}}/\pi_\theta$ convention, why does $\mathbb{E}_{a \sim \pi_\theta}[r] = 1$? How does this result establish the unbiasedness of $k_3$?

> **L3**: GRPO incorporates $k_3$ directly into the loss, yet $k_3$ as a loss term does not yield the principled reverse-KL gradient — why is this usually acceptable in GRPO practice? If $\beta$ were increased from 0.04 to 0.5, how would this approximation error change?

---

## 10. Process Reward Model (PRM) vs Outcome Reward Model (ORM)

### 10.1 ORM: Outcome Reward Model

$$
r_{\mathrm{ORM}}(x, y) = f_\phi(x, y) \in \mathbb{R}
$$

- Gives a single scalar reward only at the **end of the sequence** (End of Sequence, EOS)
- The entire response $y = (s_1, s_2, \dots, s_T)$ shares the same reward value
- Training data: $(x, y, \text{correct/incorrect})$ binary labels

### 10.2 PRM: Process Reward Model

$$
r_{\mathrm{PRM}}(x, y, t) = f_\phi(x, s_1, \dots, s_t) \in \mathbb{R}
$$

- Gives an independent score for **each step** of the reasoning chain (step-level scoring)
- $r_{\text{step}}(s_t)$ represents the quality of the $t$-th reasoning step
- Training data: $(x, s_1, \dots, s_t, \text{step correct/incorrect})$ step-level labels

### 10.3 Credit Assignment Advantage

This is the core advantage of PRM. Consider a mathematical reasoning chain:

> Step 1: Let $f(x) = x^2 + 3x$ → ✅ correct
> Step 2: Differentiate to get $f'(x) = 2x + 3$ → ✅ correct
> Step 3: Set $f'(x) = 0$, solve $x = -3/2$ → ✅ correct
> Step 4: $f(-3/2) = 9/4 - 9/2 = -9/4$ → ✅ correct

**ORM** only knows "the final answer is correct" → gives a high score, but does not know whether each step is reliable.

**PRM** can identify the case where "the first three steps are correct but the fourth is wrong":

$$
\underbrace{r_{\mathrm{PRM}}(s_1) > 0,\; r_{\mathrm{PRM}}(s_2) > 0,\; r_{\mathrm{PRM}}(s_3) > 0}_{\text{correct steps}} \quad \underbrace{r_{\mathrm{PRM}}(s_4) < 0}_{\text{erroneous step localized}}
$$

This allows PRM to guide search and training more precisely (more precise guidance for search and training).

### 10.4 Best-of-N Search with PRM

Given prompt $x$, sample $N$ candidate responses $\{y_1, \dots, y_N\}$ and score each step of each response:

$$
\text{Score}(y_i) = \min_{t=1}^{T_i} r_{\mathrm{PRM}}(x, y_i, t)
$$

or use the product form (product form):

$$
\text{Score}(y_i) = \prod_{t=1}^{T_i} \sigma\!\big(r_{\mathrm{PRM}}(x, y_i, t)\big)
$$

> **Taking min or product** ensures every step qualifies — any weak step pulls down the overall score (any weak step pulls down the overall score).

Select the best response (Select the best):

$$
y^* = \arg\max_{y_i} \text{Score}(y_i)
$$

### 10.5 PRM Training Data Challenges

| Challenge | Description |
|:---|:---|
| **Expensive annotation** | Every step of every reasoning chain requires expert human annotation of correctness — 10–50× more expensive than ORM annotation |
| **Ambiguous step boundaries** | There is no unified standard for segmenting reasoning steps; different annotators may segment them differently |
| **Low inter-annotator agreement** | Judgments of "whether a step is correct" may vary with the annotator's mathematical proficiency |
| **Limitations of automated methods** | Monte Carlo estimation (estimating the probability of reaching the correct answer after a given step via repeated sampling) has high variance |

> **Automated PRM annotation method**: After step $t$, sample completions multiple times and compute the proportion of final answers that are correct as an estimate of $r_{\mathrm{PRM}}(s_t)$. Formula: $r_{\mathrm{PRM}}(s_t) \approx \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[\text{completion}_k \text{ leads to correct answer}]$

---
## 11. Alignment Tax & Weight Averaging

### 11.1 Definition of Alignment Tax

**Alignment Tax** refers to the performance degradation on base capability benchmarks after a model undergoes alignment training:

$$
\text{Alignment Tax} = \text{Perf}_{\mathrm{base}}(\theta_{\mathrm{base}}) - \text{Perf}_{\mathrm{base}}(\theta_{\mathrm{aligned}})
$$

where $\text{Perf}_{\mathrm{base}}$ denotes performance on pre-training benchmarks (e.g., MMLU, coding ability, math ability, etc.).

Intuitively: SFT/RL training may "forget" or "overwrite" parts of pre-trained knowledge while improving alignment quality (safety, helpfulness, format following).

### 11.2 WiSE-FT Linear Interpolation

**WiSE-FT** (Weight-space Ensembles for Finetuning) mitigates the alignment tax by interpolating in weight space:

$$
\boxed{\theta_{\mathrm{merged}} = (1 - \alpha)\,\theta_{\mathrm{aligned}} + \alpha\,\theta_{\mathrm{base}}}
$$

where $\alpha \in [0, 1]$ controls the trade-off between the aligned model and the base model.

| $\alpha$ | Effect |
|:---:|:---|
| $\alpha = 0$ | fully aligned model |
| $\alpha = 1$ | base model only |
| $\alpha \in (0, 1)$ | compromise: retains some aligned behavior while recovering some base capability |

### 11.3 Why Interpolation Works

**Task Vector** perspective: alignment training is equivalent to moving in a direction within weight space:

$$
\tau_{\mathrm{align}} = \theta_{\mathrm{aligned}} - \theta_{\mathrm{base}}
$$

Research shows that the weight-change directions corresponding to different tasks are near-orthogonal, so:

$$
\theta_{\mathrm{merged}} = \theta_{\mathrm{base}} + (1-\alpha)\,\tau_{\mathrm{align}}
$$

Linear interpolation doesn't severely interfere with other task representations.

### 11.4 Advanced Model Merging Variants

| Method | Formula / Operation | Core Idea |
|:---|:---|:---|
| **Linear Interpolation** | $\theta_m = (1-\alpha)\theta_a + \alpha\theta_b$ | Simplest; element-wise linear average |
| **SLERP** (Spherical Linear Interpolation) | $\theta_m = \frac{\sin((1-t)\Omega)}{\sin\Omega}\theta_a + \frac{\sin(t\Omega)}{\sin\Omega}\theta_b$, where $\cos\Omega = \frac{\theta_a \cdot \theta_b}{\|\theta_a\|\|\theta_b\|}$ | Interpolates on the hypersphere, preserving vector norms |
| **DARE** (Drop And REscale) | Randomly drop $p\%$ of parameters in $\theta_{\mathrm{aligned}} - \theta_{\mathrm{base}}$, then rescale the remaining: $\delta_i \leftarrow \delta_i / (1-p)$, then merge | Sparsifies the task vector to reduce interference |
| **TIES** (Trim, Elect, Sign) | ① Trim small-magnitude changes ② Vote on sign ③ Keep only parameters with a consistent direction | Resolves parameter conflicts when merging multiple models |

> **SLERP intuition**: the "direction" of a weight vector matters more than its "length"; spherical interpolation preserves the geometric relationship between directions.

---

## 12. Catastrophic Forgetting, Mode Collapse & Reward Hacking

These are three **distinct** training failure modes.

### 12.1 Catastrophic Forgetting

**Definition**: when the model learns new behaviors during SFT or RL, it loses knowledge and capabilities acquired during pre-training.

**Mechanism**:

$$
\text{Pre-training capability} \xrightarrow{\text{SFT/RL update } \theta} \text{partially overwritten/erased}
$$

Neural network weight space is finite; gradient updates for new tasks may overwrite weights that store old knowledge.

**Detection Metrics**:
- Drop in benchmark scores (MMLU, GSM8K, HumanEval, etc.)
- Perplexity rises on the pre-training distribution
- Accuracy of capability probes drops

**Mitigation Strategies**:

| Strategy | Description |
|:---|:---|
| Mixed training data | Mix pre-training data into SFT |
| Low-rank adaptation (LoRA) | Only updates the low-rank delta $\Delta W = BA$, greatly reducing interference with original weights |
| Regularization | EWC (Elastic Weight Consolidation): $\mathcal{L} = \mathcal{L}_{\mathrm{new}} + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$, where $F_i$ is the Fisher information |
| Model merging | WiSE-FT / SLERP merges the aligned model with the base model |

### 12.2 Mode Collapse

**Definition**: during RL training, the model's output diversity drops sharply, repeatedly producing similar or even identical responses.

**Mechanism**: the policy over-optimizes a high-reward pattern, concentrating probability mass onto a small number of outputs:

$$
H(\pi_\theta(\cdot|x)) = -\sum_y \pi_\theta(y|x)\log\pi_\theta(y|x) \to 0
$$

**Detection Metrics**:
- Drop in output diversity metrics (self-BLEU ↑, distinct-n ↓, entropy of token distribution ↓)
- Responses converge across prompts
- Almost no variation under temperature sampling

**Mitigation Strategies**:

| Strategy | Description |
|:---|:---|
| Increase KL penalty | $\beta \uparrow$ keeps the policy close to $\pi_{\mathrm{ref}}$, maintaining diversity |
| Entropy regularization | Add a $-\eta H(\pi_\theta)$ term to encourage exploration |
| Data diversity | Training data covers a diverse prompt distribution |
| Early stopping | Monitor diversity metrics and stop training promptly |

### 12.3 Reward Hacking

**Definition**: the policy learns to exploit RM weaknesses, achieving high RM scores while actual human evaluation declines. This is a direct manifestation of **Goodhart's Law**:

$$
\text{"When a measure becomes a target, it ceases to be a good measure."}
$$

$$
\mathbb{E}_{y \sim \pi_\theta}[r_\phi(x,y)] \uparrow\uparrow \quad \text{but} \quad \mathbb{E}_{y \sim \pi_\theta}[r_{\mathrm{human}}(x,y)] \downarrow
$$

**Detection Metrics**:

| Metric | Description |
|:---|:---|
| Divergence between RM score and human rating | $\Delta = r_{\mathrm{RM}} - r_{\mathrm{human}}$ grows |
| Continuously increasing KL | Policy keeps drifting away from $\pi_{\mathrm{ref}}$ |
| Surge of specific patterns | e.g., overuse of filler phrases like "however", "it is worth noting", etc. |
| Response length bloat | RM favors longer answers → model learns to produce redundant content |

**Mitigation Strategies**:

| Strategy | Description |
|:---|:---|
| KL penalty | Constrains the policy from drifting too far (most fundamental) |
| RM ensemble | Average over multiple RMs to reduce bias from any single RM |
| Adversarial training | Continuously update the RM to adapt to policy changes (online RLHF) |
| Human evaluation | Periodically evaluate with humans to detect RM–human divergence |
| Length penalty | Apply length normalization to RM scores |

### 12.4 Relationship Summary

```
Pre-training → SFT → RL
                ↓         ↓         ↓
       Catastrophic   Mode       Reward
         Forgetting   Collapse   Hacking
       (knowledge   (diversity  (RM being
          loss)        loss)     exploited)
```

| Feature | Catastrophic Forgetting | Mode Collapse | Reward Hacking |
|:---|:---|:---|:---|
| Stage | SFT / RL | RL | RL |
| Root cause | Weight overwriting | Over-optimization of a single pattern | RM weaknesses exploited |
| Symptom | Capability degradation | Monotone output | High RM score but poor quality |
| Core mitigation | Regularization + mixed data | KL + entropy + diversity | KL + RM ensemble + human evaluation |

---

## 13. Constitutional AI / RLAIF

### 13.1 RLAIF Overview

**RLAIF** (Reinforcement Learning from AI Feedback) replaces human annotators with LLM-generated preference labels:

$$
\text{Standard RLHF:} \quad \text{Human annotators} \xrightarrow{\text{compare } (y_1, y_2)} \text{preference label} (y_w, y_l)
$$

$$
\text{RLAIF:} \quad \text{LLM judge} \xrightarrow{\text{compare } (y_1, y_2)} \text{preference label} (y_w, y_l)
$$

### 13.2 CAI Self-Critique-Revision Loop

The core of **Constitutional AI (CAI)** is a **four-step loop**:

**Step 1 — Generate**: given prompt $x$, use the current model to generate an initial response $y_0$:
$$y_0 \sim \pi_\theta(\cdot | x)$$

**Step 2 — Critique**: use an LLM to critique $y_0$ according to the constitution principles:
$$\text{critique} = \mathrm{LLM}\!\left(\text{"According to principle } P_j \text{, what is wrong with the following response: } y_0\text{"}\right)$$

**Step 3 — Revise**: based on the critique, use the LLM to revise the response:
$$y_1 = \mathrm{LLM}\!\left(\text{"Please revise the response based on the following critique: } [\text{critique}] \rightarrow y_0\text{"}\right)$$

> Can be iterated multiple times: $y_0 \to y_1 \to y_2 \to \dots$ (typically 1–3 rounds)

**Step 4 — Train**:
- **SL-CAI (SFT stage)**: use the revised $y_k$ as training data for SFT
- **RL-CAI (RL stage)**: use the LLM as a preference judge, generate $(y_w, y_l)$ preference pairs, train a reward model, then do RL

### 13.3 Constitution Principles

Constitution principles are a set of **auditable alignment constraints**, for example:

| No. | Principle Example |
|:---:|:---|
| P₁ | "Choose the response that is most helpful, accurate, and harmless" |
| P₂ | "Choose the response that does not promote bias or discrimination" |
| P₃ | "Choose the response that does not assist with illegal activities" |

Unlike implicit human preferences, constitution principles are **explicit and auditable**:

$$
p_{\mathrm{CAI}}(y_w \succ y_l | x) = \sigma\!\big(r_{\mathrm{LLM}}(x, y_w) - r_{\mathrm{LLM}}(x, y_l)\big)
$$

where $r_{\mathrm{LLM}}$ is the LLM score based on constitution principles.

### 13.4 Comparison with Standard RLHF

| Dimension | Standard RLHF | RLAIF / CAI |
|:---|:---|:---|
| **Preference source** | Human annotators | LLM (based on constitution principles) |
| **Annotation cost** | High (labor intensive) | Low (API call cost) |
| **Scalability** | Limited by annotator count and time | Nearly unlimited scaling |
| **Consistency** | Inter-annotator variance | LLM is highly consistent |
| **Auditability** | Preference criteria exist implicitly in annotators' minds | Constitution principles are explicit and auditable |
| **Risk** | Annotator bias | LLM's own bias + poorly designed constitution principles |
| **Human involvement** | Throughout | Only when designing constitution principles |

### 13.5 Theoretical Advantages of RLAIF

1. **Principle-guided**: alignment objectives are expressed explicitly through natural-language principles, making them more controllable than implicit preferences
2. **Self-improvement loop**: model critiques itself, revises itself, learns from the revised version → continuous improvement
3. **Reduced human burden**: humans only need to design principles, not annotate individual examples
4. **Cross-cultural consistency**: annotators from different cultural backgrounds may have different preferences, whereas constitution principles can unify the standard

> **Note**: CAI is not fully human-free. Humans still need to:
> - Design the constitution
> - Evaluate final model quality
> - Monitor for drift during iteration

---

## 14. Distillation (Post-Training Perspective)

### 14.1 Comparison of Three Distillation Paradigms

#### SeqKD (Sequence-Level Knowledge Distillation)

The Teacher first generates complete output sequences via beam search or sampling; the Student then performs **standard SFT** (cross-entropy loss) on these sequences:

$$L_{\text{SeqKD}} = -\sum_{t} \log p_{\theta_S}(y_t \mid x, y_{<t}), \quad y \sim \pi_T(x)$$

Key points:
- Data can be generated offline; **no online Teacher inference required**;
- The distillation signal comes only from discrete sequences sampled by the Teacher, losing the full distribution information across tokens ("soft labels" are hardened);
- Simplest to implement, lowest cost, suitable for most engineering scenarios.

#### Token-Level KD (Token-Level Knowledge Distillation)

At each position $t$, align the Student's and Teacher's probability distributions over the vocabulary:

$$L_{\text{TKD}} = \sum_{t} D_{\text{KL}}\!\left(p_T(\cdot \mid x, y_{<t}) \;\Big\|\; p_{\theta_S}(\cdot \mid x, y_{<t})\right)$$

Key points:
- Preserves the Teacher's **soft distribution** (soft labels), providing richer information, especially when multiple tokens are plausible;
- **Requires the Teacher to provide logits online or offline**; the Teacher must be accessible (or logits must be cached in advance);
- When the Teacher is very large (e.g., 671B MoE), caching logits for all positions is extremely expensive.

#### On-Policy Distillation

The Student itself rolls out candidate sequences, which are then scored by the Teacher (or a verifiable reward); the Student updates accordingly:

$$L_{\text{on-policy}} = -\mathbb{E}_{y \sim \pi_{\theta_S}}\!\left[r_T(x, y) \cdot \log p_{\theta_S}(y \mid x)\right]$$

Key points:
- Training signal comes from the **Student's own distribution**, no off-policy drift;
- Equivalent to using Teacher reward as an RLVR signal; training is more complex but generalization is typically stronger;
- Representative method: GRPO + verifiable reward (Teacher itself as verifier).

---

### 14.2 CoT Distillation (R1-style Chain-of-Thought Distillation)

**Core idea**: use a large RL model (e.g., DeepSeek-R1-671B) to generate long reasoning sequences with complete chains of thought, then perform **SFT** on a small model (i.e., the CoT version of SeqKD).

The DeepSeek-R1 paper (arXiv:2501.12948) reports experimental results of SFT on Qwen and Llama models at 1.5B, 7B, 8B, 14B, 32B, and 70B parameters using approximately 800K distillation samples (approximately 600K reasoning + approximately 200K non-reasoning), with reasoning capability of small models improving substantially.

**Why CoT distillation into small models is often more stable / more efficient than directly applying GRPO (per the distillation experiments in the R1 paper):**

1. **Asymmetric exploration cost**: GRPO requires the model to independently explore high-quality chains of thought, but small models have limited capability — random sampling rarely produces effective reasoning sequences (reward is extremely sparse), and gradient signals are noisy; the Teacher directly providing high-quality CoT effectively **compresses the exploration space**.
2. **No Critic / RM needed**: the SeqKD path only requires SFT — no online rollout or reward model — eliminating the GPU memory and compute overhead of GRPO's online sampling and reward/critic.
3. **Training stability**: the loss landscape of SFT is smoother than RL, with no risk of reward hacking or mode collapse and fewer hyperparameters.

> **Hedging caveat**: the above "more stable / more efficient" conclusion comes from observational results in the R1 paper under its distillation configuration (DeepSeek-V3-Base as the base model, approximately 800K data scale), and does not imply this holds across all small models or data scales; direct RL (GRPO) may have a higher ceiling when data and compute are sufficient.

---

### 14.3 Forward KL vs Reverse KL

#### Definitions

**Forward KL** (also called inclusive KL; mean-seeking):

$$D_{\text{KL}}^{\text{fwd}}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)}$$

Optimization direction: minimizing the forward KL of $q$ relative to $p$ is equivalent to maximizing $\mathbb{E}_{y \sim p}[\log q(y)]$ — the Student $q$ must cover **all modes** of the Teacher $p$ (wherever $p(y)>0$, $q$ cannot be 0, otherwise KL diverges).

**Reverse KL** (also called exclusive KL; mode-seeking):

$$D_{\text{KL}}^{\text{rev}}(q \| p) = \sum_y q(y) \log \frac{q(y)}{p(y)}$$

Optimization direction: minimizing this quantity takes the expectation over the support of $q$, allowing $q$ to **ignore certain modes of $p$** (the term is 0 where $q(y)=0$), but $q$ will concentrate on regions where $p$ has high probability.

#### Why Generation Tasks Often Prefer Reverse KL / Mode-Seeking

Intuitive derivation:

Suppose the Teacher distribution $p$ is bimodal, with two modes $y_1, y_2$ each having probability $\approx 0.5$.

- **Forward KL**: to maximize $\mathbb{E}_{y \sim p}[\log q(y)]$, the Student $q$ must cover both modes, resulting in $q$ being spread between the two modes — **but this middle ground in text space often corresponds to low-quality or unnatural sequences** (the "mean" is a semantically meaningless mixture). This phenomenon in generation tasks is called **mode averaging**: the output is an average of all modes, resembling none of the reasonable answers.

- **Reverse KL**: the Student $q$ incurs a log penalty wherever $q(y)>0$, naturally choosing to concentrate on **one** high-probability, semantically coherent mode in $p$. Although the other mode is sacrificed, the generated sequences are higher quality and more natural.

Mathematical statement: let $q^\*(y) = \arg\min_q D_{\text{KL}}^{\text{rev}}(q \| p)$; for a capacity-limited Student, the solution exhibits mass concentration on the dominant mode(s) of $p$, rather than "smearing" across multiple modes.

> **One-line intuition**: Forward KL requires "don't miss any answer from the Teacher"; Reverse KL allows "only learn the Teacher's most confident answers." Generation tasks require coherent outputs — better to cover less but with higher quality, hence the preference for Reverse KL.

> **Note**: Token-level KD typically uses forward KL (Student aligns to Teacher soft labels), while SeqKD / SFT at the sequence level more closely resembles reverse KL behavior (Student only learns the modes sampled by the Teacher). The two are not mutually exclusive; in practice they are often mixed depending on the task.

---

### 14.4 Distillation vs RFT vs PPO: Three-Row Comparison

| Method | Data Source | Comparison / Optimization Signal | Applicable Scale |
|:---|:---|:---|:---|
| **Distillation (SeqKD)** | Sequences generated by the Teacher (offline) | Teacher output sequences (cross-entropy / soft labels) | Small-to-medium models (typically ≤ 70B), Teacher significantly stronger than Student |
| **RFT (Rejection Sampling FT)** | Self-sampled from current policy, filtered by reward to keep high-scoring outputs | Verifiable reward / RM filtering | Medium scale (7B–70B), reward can be automatically verified |
| **PPO** | Online rollout from current policy | RM score + KL constraint + GAE Advantage | Large scale (typically ≥ 7B), with sufficient RM and compute resources |

---

### 14.5 Self-Assessment Questions

> **L2 — Distinguishing Distillation Paradigms**: both SeqKD and Token-Level KD use the Teacher model as the signal source, but fundamentally one more closely resembles reverse KL and the other more closely resembles forward KL. Please explain: (a) which corresponds to which direction of KL; (b) when the Teacher distribution is bimodal, how will the Student distributions trained by each method behave differently?

> **L3 — Applicability Analysis of CoT Distillation**: suppose you have a 3B small model and sufficient GPUs (capable of running both the 671B Teacher and the Student simultaneously). Analyze: under what data scale and task types would directly applying GRPO have an advantage over SeqKD distillation? Give at least two substantive reasons.

---
# Part 2 — PyTorch Code Snippets / From-Scratch PyTorch Snippets

---

**SFT loss masking** — During SFT training, compute loss only on the assistant's response tokens; mask the prompt portion with `label=-100`.


```python
import torch
from torch.nn.utils.rnn import pad_sequence

class SFTDataCollator:
    """
    将 prompt token 的 label 设为 -100，loss 只计算 assistant 部分。
    Masks prompt tokens with label=-100 so loss only applies to assistant tokens.
    """
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, batch):
        input_ids, labels, attention_mask = [], [], []
        for sample in batch:  # each sample: dict with 'input_ids' and 'prompt_length'
            ids = torch.tensor(sample["input_ids"], dtype=torch.long)
            prompt_len = sample["prompt_length"]
            lab = ids.clone()
            lab[:prompt_len] = -100  # 屏蔽 prompt / mask prompt tokens
            input_ids.append(ids)
            labels.append(lab)
        # 动态 padding / dynamic pad to longest in batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# --- 用法示例 / Usage example ---
collator = SFTDataCollator(type("Tok", (), {"pad_token_id": 0})())
toy_batch = [
    {"input_ids": [10, 20, 30, 40, 50], "prompt_length": 3},  # prompt=前3个
    {"input_ids": [11, 21, 31], "prompt_length": 2},
]
out = collator(toy_batch)
print("input_ids:\n", out["input_ids"])
print("labels (prompt positions = -100):\n", out["labels"])
# labels: tensor([[ -100, -100, -100, 40, 50],
#                 [ -100, -100,  31,  0,  0]])
```



---

**DPO loss** — Compute the Direct Preference Optimization loss from log-probabilities of the policy and reference models.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def get_logps(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    逐 token 计算 log-probability 并在序列维度求和。
    Computes per-token log-probs and sums over the sequence dimension.
    logits: (B, T, V),  labels: (B, T),  mask: (B, T)  (1=有效, 0=padding)
    返回每个样本的标量 log-prob / Returns scalar log-prob per sample.
    """
    # shift: 预测下一个 token / predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)            # (B, T-1, V)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return (token_logps * shift_mask).sum(dim=-1)               # (B,)

def dpo_loss(
    policy_logps_chosen: torch.Tensor,
    policy_logps_rejected: torch.Tensor,
    ref_logps_chosen: torch.Tensor,
    ref_logps_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO loss: L = -E[ log σ( β·(log π_θ/π_ref)_chosen - β·(log π_θ/π_ref)_rejected ) ]
    """
    log_ratio_chosen = policy_logps_chosen - ref_logps_chosen
    log_ratio_rejected = policy_logps_rejected - ref_logps_rejected
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()
    return loss

# --- 示例 / Example ---
B, T, V = 4, 10, 100
logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))
mask = torch.ones(B, T)

logps = get_logps(logits, labels, mask)  # (B,)
# splitting chosen / rejected is done on the caller side
policy_logps_chosen, policy_logps_rejected = logps[:2], logps[2:]
ref_logps_chosen, ref_logps_rejected = logps[:2] - 0.1, logps[2:] + 0.05

loss = dpo_loss(policy_logps_chosen, policy_logps_rejected,
                ref_logps_chosen, ref_logps_rejected, beta=0.1)
print("DPO loss:", loss.item())
```



---

**Reward Model** — Replace the LM head on a pretrained LLM backbone with a scalar linear head and train with Bradley-Terry loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """
    奖励模型：LLM 骨干 + 线性标量头，取最后一个有效 token 的隐状态。
    Reward model: LLM backbone + scalar linear head on last valid hidden state.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)  # 标量奖励 / scalar reward

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)
        # 取每个序列最后一个有效 token 的隐状态 / hidden state of last valid token
        last_idx = attention_mask.sum(dim=1) - 1  # (B,)
        last_hidden = hidden[torch.arange(hidden.size(0)), last_idx]  # (B, H)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return reward

def bradley_terry_loss(rewards_chosen, rewards_rejected):
    """
    Bradley-Terry loss: L = -log σ(r_chosen - r_rejected)
    BT loss: higher reward for preferred responses.
    """
    return -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

# --- 训练示例 / Training example ---
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
rm = RewardModel("Qwen/Qwen2.5-0.5B").to(device)

chosen_text = ["The answer is 42.", "It is safe to proceed."]
rejected_text = ["I don't know.", "No, never do that."]
tok_chosen = tokenizer(chosen_text, return_tensors="pt", padding=True, truncation=True)
tok_rejected = tokenizer(rejected_text, return_tensors="pt", padding=True, truncation=True)

r_chosen = rm(tok_chosen["input_ids"], tok_chosen["attention_mask"])
r_rejected = rm(tok_rejected["input_ids"], tok_rejected["attention_mask"])
loss = bradley_terry_loss(r_chosen, r_rejected)
print("BT loss:", loss.item())
```



---

**PPO complete loss** — single-step actor-critic loss: clipped surrogate + clipped value loss + entropy bonus + approx_kl diagnostic (token-level).

```python
import torch
import torch.nn.functional as F

def ppo_actor_critic_loss(
    logp, old_logp, advantages, returns, values, old_values, entropy, mask,
    clip_eps=0.2, vf_clip=0.2, vf_coef=0.5, ent_coef=0.01,
):
    """
    Token-level PPO loss: clipped policy surrogate + clipped value loss + entropy bonus.
    All tensors (B, T); mask marks valid response tokens (1=valid).
    logp/old_logp: log π(a_t|s_t) under the current / old policy; advantages: GAE A_t;
    returns: R_t; values/old_values: current / old critic predictions.
    """
    def masked_mean(x):                                         # average over valid tokens only
        return (x * mask).sum() / mask.sum().clamp(min=1)

    # --- policy loss: clipped surrogate (pessimistic lower bound) ---
    ratio = torch.exp(logp - old_logp)                          # pi_theta / pi_theta_old, (B,T)
    pg_loss = -torch.min(ratio * advantages,
                         torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages)

    # --- clipped value loss (guards against critic jumps) ---
    v_clipped = old_values + torch.clamp(values - old_values, -vf_clip, vf_clip)
    v_loss = 0.5 * torch.max((values - returns) ** 2, (v_clipped - returns) ** 2)

    # --- total = policy + c_vf * value - c_ent * entropy ---
    loss = masked_mean(pg_loss) + vf_coef * masked_mean(v_loss) - ent_coef * masked_mean(entropy)

    # --- diagnostics: approx_kl via k3 = (r-1) - log r (here r = pi_theta/pi_theta_old,
    #     estimating KL(pi_old || pi_theta); estimator rationale in §9.4, but note the r
    #     convention is inverted vs §9.4's r = pi_ref/pi_theta) ---
    with torch.no_grad():
        log_ratio = logp - old_logp
        approx_kl = masked_mean((ratio - 1) - log_ratio)        # >= 0, for early-stop / adaptive KL
        clip_frac = masked_mean((torch.abs(ratio - 1) > clip_eps).float())
    return loss, {"approx_kl": approx_kl.item(), "clip_frac": clip_frac.item()}

# --- Toy example ---
torch.manual_seed(0)
B, T = 2, 5
logp       = (torch.randn(B, T) * 0.1).requires_grad_(True)
old_logp   = logp.detach() + torch.randn(B, T) * 0.05           # behavior (old) policy
advantages = torch.randn(B, T); advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
returns    = torch.randn(B, T)
values     = torch.randn(B, T, requires_grad=True)
old_values = values.detach() + torch.randn(B, T) * 0.1
entropy    = torch.rand(B, T)                                   # per-token policy entropy
mask       = torch.ones(B, T); mask[1, 3:] = 0                  # second row's tail is padding

loss, logs = ppo_actor_critic_loss(logp, old_logp, advantages, returns, values, old_values, entropy, mask)
loss.backward()
print("PPO loss:", round(loss.item(), 4), "| diag:", {k: round(v, 4) for k, v in logs.items()})
```

---

**GRPO advantage** — Group Relative Policy Optimization: normalize rewards within the same group to produce advantages for the policy gradient update.

```python
import torch
import torch.nn.functional as F

def compute_grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    在 group 内归一化奖励作为 advantage：(r - mean) / std。
    Normalize rewards within group: subtract mean, divide by std.
    rewards: (G,)  — 同一 prompt 的 G 个采样回复的奖励
    """
    mean = rewards.mean()
    std = rewards.std().clamp(min=1e-8)  # 防止除零 / avoid division by zero
    return (rewards - mean) / std

# --- 简化策略梯度更新 / Simplified policy gradient update ---
# simulate: given policy log-probs and group advantages, perform one gradient ascent step
G = 8  # 每个 prompt 采样 8 个回复 / sample 8 responses per prompt

# simulated per-sequence log-probs (already summed to sequence level)
policy_logps = torch.randn(G, requires_grad=True)

# simulated rewards (e.g., from a reward model)
rewards = torch.tensor([1.2, 0.5, 2.0, 0.3, 1.8, 0.1, 1.5, 0.9])

advantages = compute_grpo_advantages(rewards)
print("Advantages:", advantages)

# policy gradient loss = -E[advantage * log_prob]  → maximize log-prob for high-advantage responses
grpo_loss = -(advantages.detach() * policy_logps).mean()
grpo_loss.backward()
print("GRPO loss:", grpo_loss.item())
print("policy_logps.grad:", policy_logps.grad)
```

---

**GRPO token-level loss** — broadcast group advantage to tokens + clipped surrogate + per-token K3 KL (no critic, no GAE; token-level averaging, cf. §9.4 and DAPO).

```python
import torch

def grpo_token_loss(logp, old_logp, ref_logp, group_adv, mask, clip_eps=0.2, beta_kl=0.04):
    """
    Token-level GRPO loss: per-sequence group advantage broadcast to tokens
    + clipped surrogate + per-token K3 KL (no critic, no GAE).
    logp/old_logp/ref_logp: (B, T) log-prob of the taken token under current / old / reference policy
    group_adv: (B,) within-group normalized advantage A_i (see compute_grpo_advantages above), broadcast per sequence
    mask: (B, T) 1=valid response token
    """
    adv = group_adv.unsqueeze(1)                                # (B,1) -> broadcast to (B,T)
    # clipped surrogate (same clip as PPO, group-relative advantage)
    ratio = torch.exp(logp - old_logp)                          # pi_theta / pi_theta_old
    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    # per-token K3 KL: r = pi_ref/pi_theta, k3 = (r-1) - log r >= 0 (same convention as §9.4)
    log_r = ref_logp - logp                                     # log(pi_ref / pi_theta)
    k3 = torch.exp(log_r) - 1 - log_r
    per_token = pg + beta_kl * k3
    # token-level averaging convention borrowed from DAPO (§3.3) so long-CoT gradients are not diluted;
    # note the KL term here is GRPO-style (beta>0), not DAPO (which sets beta=0)
    return (per_token * mask).sum() / mask.sum().clamp(min=1)

# --- Toy example ---
torch.manual_seed(0)
B, T = 4, 6                          # 4 sampled responses for one prompt
logp      = (torch.randn(B, T) * 0.1).requires_grad_(True)
old_logp  = logp.detach() + torch.randn(B, T) * 0.02
ref_logp  = logp.detach() + torch.randn(B, T) * 0.05
rewards   = torch.tensor([1.2, 0.3, 1.8, 0.5])                  # one scalar reward per response
group_adv = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-8)   # within-group normalization
mask      = torch.ones(B, T); mask[1, 4:] = 0

loss = grpo_token_loss(logp, old_logp, ref_logp, group_adv, mask)
loss.backward()
print("GRPO token-level loss:", round(loss.item(), 4))
```

---

**Sequence packing with cu_seqlens** — Concatenate multiple variable-length sequences into a single batch, compute `cu_seqlens` required by Flash Attention, and correctly mask the loss over the packed output.

```python
import torch

def pack_sequences(input_ids_list, labels_list, pad_token_id=0):
    """
    将多条序列拼接成一个平坦 tensor，并计算 Flash Attention 用的 cu_seqlens。
    Packs variable-length sequences into a flat tensor with cu_seqlens for Flash Attention.
    """
    # compute real lengths of each sequence
    lengths = [ids.size(0) for ids in input_ids_list]
    # cu_seqlens: [0, len_0, len_0+len_1, ...]  (半精度索引 / Flash Attention format)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    for i, l in enumerate(lengths):
        cu_seqlens[i + 1] = cu_seqlens[i] + l

    # concatenate all sequences into one flat tensor
    packed_input_ids = torch.cat(input_ids_list, dim=0)   # (total_tokens,)
    packed_labels = torch.cat(labels_list, dim=0)          # (total_tokens,)
    return packed_input_ids, packed_labels, cu_seqlens

def compute_packed_loss(logits_flat, labels_flat, cu_seqlens, ignore_index=-100):
    """
    在拼接序列上计算 cross-entropy，loss 屏蔽 label=-100 的 token。
    Compute cross-entropy on packed sequence; -100 labels are masked.
    logits_flat: (total_tokens, V),  labels_flat: (total_tokens,)
    """
    # shift for next-token prediction
    shift_logits = logits_flat[:-1, :]
    shift_labels = labels_flat[1:]
    # mask loss at sequence boundaries
    boundary_mask = torch.zeros(shift_labels.size(0), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        if start < end:
            boundary_mask[start] = True  # 屏蔽第一条 token 的 shift / mask first token of seq
    shift_labels[boundary_mask] = ignore_index
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
    return loss

# --- 示例 / Example ---
seq_a_ids = torch.tensor([101, 202, 303, 404, 505])
seq_b_ids = torch.tensor([606, 707])
seq_c_ids = torch.tensor([808, 909, 1010])

seq_a_lab = torch.tensor([-100, -100, 303, 404, 505])   # first two are prompt
seq_b_lab = torch.tensor([-100, 707])
seq_c_lab = torch.tensor([-100, 1010, 1010])

packed_ids, packed_labels, cu_seqlens = pack_sequences(
    [seq_a_ids, seq_b_ids, seq_c_ids], [seq_a_lab, seq_b_lab, seq_c_lab]
)
print("packed_ids:", packed_ids)
print("cu_seqlens:", cu_seqlens)  # tensor([0, 5, 7, 10])

# simulate logits
V = 2000
logits_flat = torch.randn(packed_ids.size(0), V)
loss = compute_packed_loss(logits_flat, packed_labels, cu_seqlens)
print("Packed loss:", loss.item())
```

---

**KL divergence penalty** — In PPO/RLHF reward shaping, compute the per-token KL penalty between the policy and reference models.

```python
import torch
import torch.nn.functional as F

def compute_kl_penalty(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    逐 token KL 散度：KL(π_θ || π_ref)，在序列维度求均值后取 batch 均值。
    Per-token KL divergence: KL(policy || ref), averaged over valid tokens & batch.
    policy_logits / ref_logits: (B, T, V),  mask: (B, T) — 1=有效, 0=padding
    """
    policy_logps = F.log_softmax(policy_logits, dim=-1)  # (B, T, V)
    ref_logps = F.log_softmax(ref_logits, dim=-1)        # (B, T, V)
    # KL(p||q) = sum_p p(x) * [log p(x) - log q(x)] = E_p[log p - log q]
    policy_probs = policy_logps.exp()
    token_kl = (policy_probs * (policy_logps - ref_logps)).sum(dim=-1)  # (B, T)
    # masked mean
    kl_per_seq = (token_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B,)
    return kl_per_seq.mean()  # scalar

# --- Used in PPO reward shaping ---
B, T, V = 2, 8, 1000
policy_logits = torch.randn(B, T, V)
ref_logits = torch.randn(B, T, V)
mask = torch.ones(B, T); mask[1, 6:] = 0  # second sequence has padding in the latter half

kl = compute_kl_penalty(policy_logits, ref_logits, mask)
print("KL penalty:", kl.item())

# PPO reward shaping: r = r_raw - beta * KL
beta_kl = 0.05
shaped_reward = 1.5 - beta_kl * kl  # used at batch level
print("Shaped reward:", shaped_reward.item())
```



---

**Rejection Sampling Fine-tuning (RFT)** — Sample N responses from the policy model, score them with a reward function, keep the highest-scoring response as the SFT target for fine-tuning.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def rejection_sampling_finetune(model, tokenizer, prompts, reward_fn, N=4, max_new_tokens=64):
    """
    RFT 流程：对每个 prompt 采样 N 个回复，用 reward_fn 评分，取 top-1 做 SFT。
    RFT loop: sample N responses, score with reward_fn, keep top-1 as SFT target.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for prompt in prompts:
        # ---- Sampling phase ----
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        all_completions, all_rewards = [], []
        with torch.no_grad():
            for _ in range(N):
                out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=0.8, top_p=0.95)
                gen_ids = out[0, input_ids.size(1):]           # keep generated portion only
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                reward = reward_fn(prompt, text)               # 标量奖励 / scalar reward
                all_completions.append(gen_ids)
                all_rewards.append(reward)

        # ---- Select best response ----
        best_idx = int(torch.tensor(all_rewards).argmax())
        best_ids = all_completions[best_idx]

        # ---- SFT phase (compute loss on best response) ----
        full_ids = torch.cat([input_ids[0], best_ids]).unsqueeze(0)  # (1, T)
        labels = full_ids.clone()
        labels[0, :input_ids.size(1)] = -100  # 屏蔽 prompt / mask prompt tokens
        logits = model(input_ids=full_ids).logits
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"RFT loss: {loss.item():.4f}, best reward: {all_rewards[best_idx]:.4f}")

# --- Simple reward function ---
def dummy_reward_fn(prompt, response):
    """Reward: longer is better (demo only)."""
    return float(len(response))

# Run
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
prompts = ["Explain gravity in one sentence.", "What is 2+2?"]
rejection_sampling_finetune(model, tokenizer, prompts, dummy_reward_fn, N=4)
```

---
# Part 3 — Interview Question Bank

### ━━━ L1 Basic ━━━

---

<details>
<summary>Q1. What problems do pre-training and post-training each solve? What is the standard pipeline?</summary>

**Answer:**
Pre-training aims to have the model learn general language capabilities, world knowledge, and a foundation for reasoning from massive unlabeled text — essentially unsupervised language modeling. Post-training aims to transform this "knowledgeable but unruly" base model into an assistant that follows instructions, is helpful, safe, and aligned with human values. The standard pipeline is: 1) Supervised Fine-Tuning (SFT), which fine-tunes the model on high-quality instruction-response pairs; 2) Preference Alignment, which typically uses methods such as RLHF or DPO to further optimize model behavior based on human preference data.

**Follow-up:**
Why can't a single stage (e.g., SFT alone) complete the full transformation from a pre-trained model to a usable assistant?

</details>

<details>
<summary>Q2. What is loss masking in SFT? Why is loss computed only on assistant tokens?</summary>

**Answer:**
Loss masking means that when computing the SFT loss, only the prediction loss for tokens corresponding to the assistant's response (i.e., the part the model needs to learn to generate) is included in the total loss, while the loss for the input/user instruction portion is ignored. This focuses the model's optimization objective on "learning how to respond correctly" rather than "parroting the user's input." Without masking the input portion, the model might waste learning capacity memorizing input formats instead of focusing on generating high-quality responses.

**Follow-up:**
If the gradients for the user instruction portion are never updated during SFT, does the model truly become completely unable to "understand" instructions? Please explain.

</details>

<details>
<summary>Q3. What is the training objective of a Reward Model? What is the Bradley-Terry model?</summary>

**Answer:**
The training objective of a Reward Model (RM) is to output a scalar score for a given (prompt, response) pair that reflects human preference rankings for response quality. Specifically, it learns by comparing a pair of responses (chosen vs. rejected). The Bradley-Terry model is a probabilistic model for pairwise comparisons; it assumes that the probability of selecting the winning response is proportional to the difference in reward values between the two responses. In RM training, the loss function is typically based on this probability, with the goal of maximizing the reward margin of the chosen response over the rejected response.

**Follow-up:**
If the preference rankings in human annotation data are inconsistent or noisy, how does this affect the Reward Model trained under the Bradley-Terry model?

</details>

<details>
<summary>Q4. What is the role of the KL penalty in RLHF? How is β tuned?</summary>

**Answer:**
During the reinforcement learning phase of RLHF, the policy model (the LLM being optimized) maximizes rewards from the Reward Model when generating responses. However, this can cause the model to generate strange, unnatural text that deviates from its original capability distribution in pursuit of high scores. The KL penalty term is added to the optimization objective by computing the KL divergence between the current policy and the initial SFT model (reference policy). Its role is to constrain the optimized model from drifting too far from the initial model, thereby preserving language quality and diversity. β is a hyperparameter that controls the strength of the KL penalty: larger β imposes heavier penalties on deviation, making the model more conservative and closer to the initial model; smaller β gives the model more freedom, potentially pursuing higher rewards but at greater risk.

**Follow-up:**
The KL penalty computes the divergence over the full sequence distribution. What challenges does this pose in practice? Are there more efficient or more local approximation methods?

</details>

<details>
<summary>Q5. What is DPO? What is the core difference from RLHF?</summary>

**Answer:**
DPO (Direct Preference Optimization) is a method that directly optimizes a language model using human preference data. Through a clever mathematical transformation, it merges the two steps of RLHF — "train a Reward Model, then use it for RL optimization" — into a single supervised learning loss function. In DPO, the model directly learns to translate preference rankings into adjustments of response probabilities. The core difference is: RLHF is "explicit," involving a separate RM training step and an online RL optimization process (e.g., PPO); DPO is "implicit" — it bypasses explicit RM training and online sampling, directly optimizing the policy through an offline contrastive loss, and is generally simpler and more stable.

**Follow-up:**
A major criticism of DPO is that it heavily depends on the quality of preference data. Why might its requirements for data quality be higher than those of RLHF?

</details>

<details>
<summary>Q6. What is sequence packing? What are its benefits and pitfalls?</summary>

**Answer:**
Sequence packing is a training efficiency optimization technique. It concatenates multiple short sequences (e.g., multiple different instruction-response pairs) — using special separators such as <code>&lt;EOS&gt;</code> followed by the start token of a new sequence — into a single long sequence that reaches the model's maximum context length, then trains on this as a whole. Benefits: significantly improves GPU utilization, reduces computation waste from padding short sequences, and speeds up training. The main pitfalls are: 1) careful attention mask design is required to prevent the model from "seeing" information from other short sequences within the same packed sequence during training (i.e., cross-sequence attention leakage), which can cause data contamination or learning bias; 2) the model may be sensitive to sequence ordering.

**Follow-up:**
In sequence packing, if two concatenated sequences are on completely unrelated topics (e.g., a math problem and a poem), what specific harm does cross-sequence attention leakage cause?

</details>

<details>
<summary>Q7. What is reward hacking? Give two examples.</summary>

**Answer:**
Reward hacking refers to the model finding ways to "cheat" or "game" the system to obtain higher reward scores, even though the generated responses do not actually meet the true human goals of being helpful, honest, and harmless. It is over-optimization or exploitation of the reward function. Example 1: If the RM favors longer responses, the model may learn to generate verbose but hollow replies. Example 2: If the RM gives high scores to responses containing certain specific "safety" phrases (e.g., "As an AI assistant, I must comply with…"), the model may learn to mechanically insert such boilerplate into all responses, regardless of whether it is actually needed.

**Follow-up:**
Beyond improving the Reward Model itself, what strategies can be employed during RLHF training to mitigate reward hacking?

</details>

<details>
<summary>Q8. What tensions exist among the Helpful / Harmless / Honest triad in alignment?</summary>

**Answer:**
Helpful, Harmless, and Honest have inherent tensions among them. For example, a model that prioritizes Harmless excessively may refuse to answer reasonable but sensitive questions due to over-caution, thereby compromising Helpfulness (e.g., a doctor discussing medical symptoms). A model pursuing extreme Honesty may expose unverified information or user privacy in responses, thereby compromising Harmlessness. Conversely, fabricating answers to be Helpful compromises Honesty. An ideally aligned model must dynamically balance these three objectives across different contexts; there is no fixed perfect solution.

**Follow-up:**
Can you provide a concrete scenario in which a model unavoidably sacrifices Helpfulness and Honesty in order to achieve Harmlessness?

</details>

### ━━━ L2 Intermediate ━━━

---

<details>
<summary>Q9. What is the core difference between GRPO and PPO? How many models does GRPO require?</summary>

**Answer:**
GRPO (Group Relative Policy Optimization) and PPO (Proximal Policy Optimization) are both policy gradient algorithms, but GRPO makes key improvements to simplify the RLHF training process. The core difference: PPO requires maintaining four models — a policy model, a reference model, a value model (Critic), and a reward model; GRPO **does not require a separate value model**. GRPO generates a group of responses for the same prompt, then uses the average reward within the group as a baseline to estimate the advantage function, thereby computing the policy gradient. Therefore, GRPO typically requires only two models: the policy model and the reward model (the reference model can be merged or shared).

**Follow-up:**
GRPO uses the group's average reward as a baseline to estimate the advantage function. What kind of bias might this introduce, and how does it affect training stability?

</details>

<details>
<summary>Q10. What problems do IPO, KTO, ORPO, and SimPO each solve with respect to DPO?</summary>

**Answer:**
These methods are all improvements or variants of DPO:
- **IPO (Identity Preference Optimization)**: Addresses the problem in DPO where KL regularization breaks down and overfitting occurs when preferences approach near-deterministic — adopts a bounded squared-loss objective (see §7.1) for more robust optimization.
- **KTO (Kahneman-Tversky Optimization)**: Addresses DPO's requirement for strictly paired preference data (chosen/rejected pairs). KTO only requires binary labels indicating whether each response is "good" or "bad," without pairing, making data collection more flexible.
- **ORPO (Odds Ratio Preference Optimization)**: Attempts to merge SFT and preference alignment into a single training stage. It directly optimizes the odds ratio of the model generating a chosen response relative to a rejected response.
- **SimPO (Simple Preference Optimization)**: Aims to further simplify DPO by removing the dependency on a reference model, while improving optimization stability and robustness to response length by using length-normalized log-probabilities as an implicit reward and introducing a target reward margin.

**Follow-up:**
Among these methods, which has relatively the lowest requirements for training data quality or quantity? Why?

</details>

<details>
<summary>Q11. Which matters more in SFT — data quality or data quantity? How is data curation done?</summary>

**Answer:**
In the SFT phase, **data quality is generally far more important than data quantity**. High-quality, diverse, accurate, and human-value-aligned instruction data, even at a smaller scale, can significantly improve model performance. Conversely, large amounts of low-quality, erroneous, or harmful data can severely contaminate the model. A typical data curation pipeline includes: 1) **Source filtering**: selecting trustworthy and professional sources; 2) **Quality filtering**: using rules or models (e.g., an RM) to filter out low-scoring, harmful, or malformatted samples; 3) **Deduplication**: removing duplicate or near-duplicate samples; 4) **Diversity augmentation**: ensuring instructions cover a wide range of tasks, difficulty levels, and domains; 5) **Format normalization**: standardizing the style and length distribution of responses.

**Follow-up:**
If you could use only a single automated model (rather than humans) to evaluate and filter quality in large-scale SFT data, what type of model would you prioritize? Why?

</details>

<details>
<summary>Q12. What are the main paradigms for synthetic data generation? Where does length bias come from?</summary>

**Answer:**
The main paradigms are: 1) **Self-Instruct**: having the model generate new instructions and responses from seed tasks; 2) **Evol-Instruct**: evolving existing instructions through multiple rounds and multiple dimensions of complexification; 3) **Bootstrapping**: using a powerful "teacher" model to generate training data for a "student" model (e.g., distillation); 4) **Reward-guided Generation**: using an RM or rules to filter/revise multiple candidate responses generated by the model. Length bias mainly originates from: 1) **Model-intrinsic bias**: common responses in pre-training data (e.g., technical documentation) tend to be long; 2) **Reward model bias**: if human annotators in the RM's training data generally prefer more detailed, longer responses, the RM will assign higher scores to longer responses, causing the model to tend toward generating longer text when optimizing the RM; 3) **Generation strategy**: for example, verbose enumeration to ensure all points are covered.

**Follow-up:**
When generating synthetic data, how can the pipeline or loss function be designed to explicitly control or reduce length bias in the final responses?

</details>

<details>
<summary>Q13. What is the difference between online and offline preference learning? What scenarios is each suited for?</summary>

**Answer:**
**Online learning** (e.g., the PPO phase in standard RLHF) means that the policy model generates new responses in real time during training and receives new reward signals from the environment (e.g., the RM) to update the policy. **Offline learning** (e.g., DPO) means using a pre-collected, fixed preference dataset to optimize the model, without generating new data during training. Online learning is suited for scenarios that require continuous exploration, fast adaptation to new reward signals, or resolving distribution shift, but has high computational cost and instability. Offline learning is suited for scenarios where data collection is expensive and stable training pipelines are needed, but is easily constrained by a fixed data distribution and may converge to a suboptimal solution.

**Follow-up:**
In offline learning, if the preference data distribution used for training differs greatly from the data distribution encountered during deployment, what problems arise? How can this be mitigated?

</details>

<details>
<summary>Q14. What is benchmark contamination? How can it be detected?</summary>

**Answer:**
Benchmark contamination refers to the situation where the model being evaluated (or its training data) has already "seen" the test questions or answers from the evaluation benchmark during training. This causes the model to achieve inflated, unrealistic performance scores on that benchmark, which do not reflect its true generalization capability. Detection methods include: 1) **Membership inference attacks**: analyzing differences in perplexity between the model's outputs on test-set samples versus similar non-test-set samples; 2) **n-gram overlap analysis**: checking the degree of text overlap between the model's training data and the test set; 3) **Data provenance auditing**: rigorously auditing training data sources to exclude datasets known to contain mainstream benchmark test sets (e.g., certain versions of Common Crawl); 4) **Dynamic benchmark design**: using regularly updated, non-public test sets.

**Follow-up:**
Beyond data contamination, what other methodological flaws in evaluation might lead to misjudgment of a model's capabilities?

</details>

<details>
<summary>Q15. How does catastrophic forgetting manifest in post-training? How can it be mitigated?</summary>

**Answer:**
In post-training, catastrophic forgetting manifests as the model **losing the broad knowledge, language capabilities, or ability to handle diverse tasks learned during pre-training** while acquiring new capabilities (e.g., instruction following, value alignment) through SFT or RLHF. For example, an aligned model may perform well on instruction following but exhibit significant degradation in foundational capabilities such as coding, mathematics, or multilingual tasks compared to the base model. Mitigation methods include: 1) **Mixed training data**: mixing pre-training data or general-capability data into SFT/RLHF data; 2) **Low-rank adaptation**: using parameter-efficient fine-tuning methods such as LoRA to update only a small fraction of parameters; 3) **Regularization**: adding an L2 penalty on the original model parameters to the loss function (similar to EWC); 4) **Knowledge distillation**: using the original model as a teacher to constrain the output distribution of the aligned model.

**Follow-up:**
In parameter-efficient fine-tuning methods (e.g., LoRA), how does the choice of which layers to fine-tune (e.g., QKV projections in attention layers vs. FFN layers) differently affect the mitigation of catastrophic forgetting and the preservation of existing capabilities?

</details>

<details>
<summary>Q16. Process Reward Model (PRM) vs. Outcome Reward Model (ORM)?</summary>

**Answer:**
An ORM (Outcome Reward Model) gives a single reward score only for the **final answer or complete response** generated by the model, without regard for the intermediate reasoning process. A PRM (Process Reward Model) evaluates and scores **each intermediate step** in solving the problem or generating the response. The advantage of PRM lies in providing denser, more fine-grained supervision signals that help guide the model toward correct step-by-step reasoning — especially valuable for complex tasks such as mathematics and logical reasoning, as it prevents the model from arriving at the correct answer via "shortcuts" with an incorrect process. The challenge is that annotation costs are extremely high, requiring human experts to evaluate each step.

**Follow-up:**
In practice, how can data for training a PRM be collected efficiently? Is it possible to use an ORM or other models to automatically generate training labels for a PRM?

</details>

<details>
<summary>Q17. What are the limitations of MT-Bench, AlpacaEval, and Chatbot Arena respectively?</summary>

**Answer:**
- **MT-Bench**: Uses pre-designed multi-turn conversation questions and a powerful LLM (e.g., GPT-4) as the judge. Limitations: 1) the judge model itself may be biased; 2) fixed questions make it easy to overfit; 3) cannot evaluate long-document processing or real-world complex tasks.
- **AlpacaEval**: Uses a fixed instruction set; GPT-4 is used to compare the model's responses against reference responses (typically GPT-4's own responses). Limitations: 1) strongly dependent on GPT-4's preferences, which may not reflect the preferences of a broad user base; 2) risk of "self-preference," where responses stylistically similar to GPT-4 may score higher.
- **Chatbot Arena**: Conducts pairwise comparisons through anonymous votes from real users, making it the most human-preference-aligned dynamic evaluation currently available. Limitations: 1) the user base may not be fully representative (skewed toward technical users); 2) high evaluation cost and slow speed; 3) uneven distribution of conversation domains.

**Follow-up:**
If you were to design a new, more comprehensive evaluation framework for post-trained models, what different evaluation dimensions and methods would you integrate to compensate for the shortcomings of these individual benchmarks?

</details>

### ━━━ L3 Deep ━━━

---

<details>
<summary>Q18. Why is the value model (critic) in PPO difficult to train? How does GRPO sidestep this problem?</summary>

**Answer:**
In PPO for RLHF, the value model (Critic) must accurately estimate the expected total future reward given a current state (i.e., the current prompt and partial generation history) — that is, the state value function V(s). This estimation is extremely difficult: 1) **Sparse rewards**: rewards are typically given only after a complete response is generated, so intermediate states lack direct supervision signals; 2) **High variance**: the state space for text generation is vast and complex, leading to high variance in value estimates and unstable training; 3) **Non-stationarity**: the policy model updates rapidly, causing the target distribution for the value function to shift continuously, increasing the difficulty of fitting. GRPO sidesteps this problem by eliminating the value model entirely. It generates a group of responses for each prompt and uses the group's average reward as a baseline to estimate each response's advantage relative to the group average. This approach avoids training a complex value network over all possible states.

**Follow-up:**
GRPO uses the group's average reward as a baseline, which implicitly assumes that the value of all states (i.e., different generation paths for the same prompt) is equal. Under what circumstances does this assumption become unreasonable?

</details>

<details>
<summary>Q19. Theoretical derivation of DPO: walk through the derivation from the RLHF KL-constrained optimal solution to the DPO loss.</summary>

**Answer:**
1. **RLHF objective**: We have a KL-constrained optimization objective: <code>max_{π} E_{x~D, y~π}[r(x, y)] - β * KL[π(y|x) || π_ref(y|x)]</code>, where π is the policy, π_ref is the reference policy, and r is the reward function.
2. **Closed-form optimal solution**: Solving the above objective with respect to π yields the closed-form optimal solution: <code>π*(y|x) = π_ref(y|x) * exp(r(x, y) / β) / Z(x)</code>, where <code>Z(x)</code> is the partition function (normalization constant).
3. **Inverting for the reward function**: Taking logarithms on both sides and rearranging, the reward function can be expressed as a function of the policy: <code>r(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log(Z(x))</code>.
4. **Substituting into the Bradley-Terry model**: For a preference pair (y_w, y_l), according to the BT model, the probability that a human selects y_w is <code>σ(r(x, y_w) - r(x, y_l))</code>, where σ is the sigmoid function.
5. **Canceling the partition function**: Substituting the reward expression from step 3 into step 4, the <code>log(Z(x))</code> terms cancel in the subtraction, yielding: <code>P(y_w ≻ y_l | x) = σ(β * log(π*(y_w|x) / π_ref(y_w|x)) - β * log(π*(y_l|x) / π_ref(y_l|x)))</code>.
6. **DPO loss**: Finally, the DPO loss function maximizes the above probability (i.e., minimizes negative log-likelihood): <code>L_DPO(θ) = -E[log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x)))]</code>, where π_θ is the policy being optimized.

**Follow-up:**
In the above derivation, we assume that the reward function r can be expressed in terms of the policy π (step 3). What are the implicit conditions for this assumption to hold?

</details>

<details>
<summary>Q20. What is the difference between mode collapse and reward hacking? How can mode collapse be detected?</summary>

**Answer:**
**Reward hacking** is when the model finds "shortcuts" to obtain high rewards while producing outputs that do not match human intent (e.g., generating verbose filler). **Mode collapse** refers to a sharp drop in the diversity of the model's outputs, where the model tends to repeatedly generate a few types of high-reward, safe, or stereotyped responses, losing the richness and creativity expected when responding to diverse prompts. It is a common failure mode in generative models. Methods for detecting mode collapse include: 1) **Diversity metrics**: computing lexical diversity (e.g., distinct-n) and variance in semantic embeddings of responses generated for a set of prompts, compared against a baseline model; 2) **Reward distribution analysis**: if the model's reward score distribution becomes highly concentrated (high mean, low variance), it may indicate that the model has found a few "high-scoring templates"; 3) **Manual sampling inspection**: randomly sampling multiple groups of responses and observing whether their content, structure, and word choices are highly similar.

**Follow-up:**
Increasing the KL penalty coefficient β is an effective way to mitigate mode collapse in RLHF training. Beyond this, what methods from a data perspective or algorithmic perspective can encourage diversity?

</details>

<details>
<summary>Q21. What is alignment tax? How does weight averaging mitigate it, and what is the principle?</summary>

**Answer:**
Alignment tax refers to the **performance cost paid on certain general capabilities not directly optimized** (e.g., basic language modeling, complex reasoning) — i.e., degradation in these capabilities — as the model undergoes post-training alignment to achieve better instruction following, safety, and harmlessness. Weight averaging is a simple and effective mitigation technique. It averages the weights of multiple models produced at different training checkpoints or with different random seeds to obtain a smoother, more generalizable final model. The principle is: 1) **Variance reduction**: averaging reduces performance instability caused by training fluctuations or randomness in any single model; 2) **Exploring better solutions**: different training snapshots may reside in different "good" regions of the loss landscape, and averaging may find an intermediate point that performs well across dimensions; 3) **Implicit regularization effect**, preventing the model from overfitting to specific patterns in training data (including biases that may exist in alignment data).

**Follow-up:**
In specific implementations of weight averaging — such as Stochastic Weight Averaging (SWA) and Model Soups — how do their strategies and assumptions differ? Which is likely more effective at mitigating alignment tax?

</details>

<details>
<summary>Q22. What are the key design decisions in DeepSeek-R1's training pipeline? What is the role of cold-start SFT?</summary>

**Answer:**
According to the DeepSeek-R1 paper (arXiv:2501.12948), it is important to distinguish between two models:

**DeepSeek-R1-Zero**: Applies pure RL (GRPO) directly on DeepSeek-V3-Base, **completely skipping the SFT phase**. The paper states: "we bypass the conventional supervised fine-tuning (SFT) phase before RL training." R1-Zero demonstrates that reasoning capabilities can emerge from pure RL, but it suffers from poor readability and language mixing.

**DeepSeek-R1**: Four-stage pipeline (paper Section 3):
1. **Cold-start SFT**: Collects thousands of cold-start data samples with human-conversational-style chain-of-thought, then fine-tunes DeepSeek-V3-Base via SFT to produce Dev1. Note: this is "cold-start" rather than standard large-scale SFT; the data volume is small (thousands).
2. **Reasoning-oriented RL (Stage 1 RL)**: Applies GRPO on Dev1 for reasoning-task reinforcement learning (rule-based rewards: accuracy + format) to produce Dev2.
3. **Rejection-sampling SFT**: Samples from Dev2, merges reasoning and non-reasoning data for SFT to produce Dev3. This stage also improves general capabilities such as writing.
4. **Full-scenario RL (Stage 2 RL)**: Applies comprehensive RL on Dev3, with reward signals combining rule-based (reasoning) + RM (general dialogue, safety), yielding the final DeepSeek-R1.

**Role of cold-start**: Resolves R1-Zero's readability and language-mixing issues, providing a more well-structured behavioral foundation for subsequent RL and making RL exploration more efficient.

**Follow-up:**
The data used for cold-start SFT has very high quality requirements. If this data contains errors or biases, what cascading effects would this have on the exploration in subsequent reinforcement learning stages?

</details>

<details>
<summary>Q23. How does the self-critique-revision mechanism in RLAIF and Constitutional AI work?</summary>

**Answer:**
The core idea of RLAIF (Reinforcement Learning from AI Feedback) and Constitutional AI is to use AI models themselves to generate preference feedback or perform corrections, reducing dependence on human annotation. The self-critique-revision mechanism typically involves a loop: 1) **Generate initial response**: given a prompt, the model first generates a preliminary response. 2) **Self-critique**: the model (or a separate critic model) reviews the initial response against a set of predefined "constitutional" principles (e.g., "answers should be objective," "avoid harmful content") and identifies potential violations. 3) **Revise response**: the model revises the initial response based on the generated critique to produce a new version that better conforms to the constitutional principles. 4) **(Optional) Use for training**: the (initial response, revised response) pair is used as a (rejected, chosen) pair to train an RM or to directly perform DPO-style optimization. This mechanism allows the model to self-improve and align without requiring real-time human intervention.

**Follow-up:**
Could this self-revision mechanism cause the model to fall into a kind of "alignment loop"? For example, in pursuing a "safer" response, the model might through multiple rounds of revision produce responses that become increasingly conservative and even useless.

</details>

<details>
<summary>Q24. How are iterative RLHF and online DPO similar and different? How can distribution mismatch be resolved?</summary>

**Answer:**
Both address the problem of **mismatch between the training data distribution (preference pairs generated by an old policy) and the current policy distribution** that arises in offline methods like standard DPO. **Similarities**: both iteratively use the current policy model to generate new data (or responses) and update the model with this new data, so that the training data distribution tracks the policy as it changes. **Differences**: Iterative RLHF typically refers to alternating between "online data generation (sampling with the current policy and scoring with an RM)" and "updating the policy with new data (possibly using PPO or DPO)." Online DPO more specifically refers to generating a set of responses with the current policy at each training iteration, having an RM or human select preference pairs, and then directly computing the DPO loss and updating the model using this **newly generated, distribution-matched preference data**, skipping the explicit RL step.

**Follow-up:**
When generating preference pairs using the current policy in Online DPO, what sampling temperature should be used? Why is this parameter choice important?

</details>

<details>
<summary>Q25. Scaling laws in post-training: how do data volume and model scale affect alignment quality? How do the optimal compute allocation strategies for SFT and RL differ?</summary>

**Answer:**
Post-training scaling laws differ from those in pre-training. For **data volume**: in the SFT phase, there are diminishing returns; high-quality data is more important than large volumes of low-quality data, and performance improvements slow after reaching a certain scale. For **model scale**: larger base models generally have stronger alignment potential and can better understand complex instructions and values, but the amount of high-quality data needed to achieve the same alignment level may not scale proportionally. **Optimal allocation strategy for SFT vs. RL**: SFT yields more "data-efficient" returns, and it is typically cost-effective to invest more compute early in a project to quickly establish instruction-following capability. RL (e.g., RLHF) is more "compute-intensive," with its returns manifesting in fine-grained behavioral adjustments and value alignment, requiring more online sampling and iteration. A common strategy is: use most of the compute budget to train a sufficiently good base model and SFT model, then use the remaining, relatively smaller compute budget for a few key RL iterations for fine-tuning, since the marginal returns of RL may diminish rapidly.

**Follow-up:**
If we treat both model scale and data volume as resources, in the post-training phase, do you think it is more likely to yield a superior assistant model in real-world applications to invest in aligning a 70B model, or to invest in aligning a 7B model with a larger volume of higher-quality data? Please explain your reasoning.

</details>

## More L3 Deep Dives / Extended L3

<details>
<summary>Q26: What does DPO's implicit reward actually learn? What are its fundamental limitations compared to an explicit RM?</summary>

The gradient of the DPO loss is equivalent to optimizing an implicit reward $\hat{r}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$. This implicit reward is essentially an accumulation of token-level log-probability ratios under the reference policy, with no explicit modeling of generation semantics. Compared to training an independent RM, the DPO reward is bound to the policy's parameter space, leading to three core limitations: (1) **distribution coupling** — the reward cannot evaluate OOD responses independently of the policy, limiting exploration; (2) **representation bottleneck** — the policy must simultaneously serve as both "value evaluator" and "strategy generator," creating potential parameter conflicts; (3) **temporal inconsistency** — as the policy changes during training the implicit reward drifts, whereas an explicit RM's reward distribution remains relatively stable. This also explains why online DPO (re-sampling with the current policy) typically outperforms offline DPO.

> **Follow-up:** Since DPO has an off-policy problem, Rejection Sampling Fine-Tuning (RFT) is a simpler alternative — under what conditions would RFT be more effective than DPO, and under what conditions would it fail?

</details>

---

<details>
<summary>Q27: What statistical bias does GRPO's Group Normalization introduce? How can it be mitigated?</summary>

GRPO applies group-level z-score normalization (subtract mean, divide by standard deviation) across multiple responses to the same prompt, implicitly assuming that within-prompt comparison is sufficient. Statistically, when the group size $G$ is small (e.g., $G<8$), the estimated mean and variance are highly variable, causing high noise in advantage estimates. More critically, **group normalization defines advantage entirely relative to the same group**, which means: (1) if all responses in a group are of low quality, a "best of a bad bunch" dynamic still produces positive advantage, reinforcing the policy in a low-quality region; (2) conversely, if all responses in the group are high quality, even excellent answers are suppressed. This **relative ranking bias** means that when the reward distribution is skewed (e.g., most responses score similarly), GRPO may systematically diverge from the absolute quality signal. Mitigation approaches include introducing a baseline anchor (e.g., an EMA reference reward) or mixing absolute-relative advantage.

> **Follow-up:** Under GRPO's KL constraint, if the group size tends to infinity, what form does GRPO's optimization objective mathematically converge to? How does it relate to standard PPO?

</details>

---

<details>
<summary>Q28: How does Reward Model overparameterization affect RLHF? Should the RM be the same scale as the policy, larger, or smaller?</summary>

RM overparameterization (far more parameters than training data requires) causes two problems: (1) **spurious correlations** — the RM may learn surface features unrelated to preference (e.g., specific writing styles, length) and achieve high accuracy, but these shortcuts break down once the policy updates; (2) **calibration degradation** — the scalar output of an overparameterized RM tends to be overconfident (concentrated at a few extreme values), causing advantage estimate variance to explode in PPO or the policy to be dominated by a small number of samples. In practice, RM scale selection involves a trade-off: a larger RM has stronger semantic understanding but is more prone to overfitting and is expensive to run; a smaller RM may generalize better but has limited expressiveness. One view is that the RM should be slightly larger than or equal to the policy scale to ensure sufficient reward signal resolution, while **reward ensembles** (averaging/voting across multiple RMs) mitigate overfitting.

> **Follow-up:** If multiple RMs in a reward ensemble all start from the same SFT initialization and differ only in data shuffling, under what conditions will this ensemble still fail systematically? How would you design a truly diverse RM ensemble?

</details>

---

<details>
<summary>Q29: How is the Credit Assignment problem solved in multi-turn dialogue RLHF? Is existing sequence-level reward sufficient?</summary>

In multi-turn dialogue, the user's final satisfaction is a function of the entire conversation history, but standard RLHF gives only a single scalar reward at the final turn, creating a severe **temporal credit assignment** problem: the model cannot tell which turn's response caused a positive or negative evaluation. Intuitive solutions include: (1) **turn-level reward modeling** — training an independent reward model for each dialogue turn, but this faces partial observability of dialogue state and high annotation costs; (2) **Monte Carlo rollout** — re-sampling subsequent dialogue from a given turn to estimate value, but combinatorial explosion is severe; (3) **shaped reward via dialogue act** — using dialogue acts (e.g., clarification, confirmation) as intermediate reward signals. Empirically, pure sequence-level reward is manageable for short dialogues (2–3 turns), but in long dialogues the policy tends to fall into **early-turn over-optimization** (over-optimizing the first-turn response to capture the initial reward signal while neglecting subsequent interaction quality).

> **Follow-up:** If you want to implement reward attribution at the token level (rather than turn level), what methods could theoretically decompose a sequence-level reward down to each token? What are the theoretical guarantees and practical difficulties of such an approach?

</details>

---

<details>
<summary>Q30: Is the theoretically optimal solution of KL-constrained RL sensitive to β? When β deviates from optimal, how do PPO and DPO differ in their failure modes?</summary>

From a KL-regularized RL perspective, $\beta$ controls the position on the exploration-exploitation Pareto frontier. Theoretically, the optimal $\beta^*$ depends on the scale of the reward function and the entropy of the reference policy, and cannot be determined in advance. When $\beta$ is too large (over-regularization), both PPO and DPO converge toward the reference policy and alignment effects are weak. When $\beta$ is too small (under-regularization), their failure modes diverge: **PPO** experiences a positive feedback loop of reward hacking — once the policy finds a reward loophole it is continuously reinforced, the RM is evaluated out-of-distribution, and reward collapses; **DPO** exhibits **instability from preference reversal** — the implicit reward of off-policy samples drifts during training, the margin between chosen and rejected shrinks or even flips, and the loss oscillates. In practice, PPO's $\beta$ (KL penalty coefficient) typically needs to be co-tuned with the learning rate, while DPO's $\beta$ behaves more like a temperature: a smaller $\beta$ allows a larger chosen-rejected margin but is also more prone to overfitting.

> **Follow-up:** Is there a theoretically grounded method to adaptively adjust $\beta$ (rather than manually tuning)? What problems arise when using KL divergence itself as the signal for adaptive β?

</details>

---

<details>
<summary>Q31: Process Reward Models (PRM) have advantages on long-chain tasks like mathematical reasoning, but how do you handle the annotation ambiguity of "steps that are correct but part of a suboptimal reasoning path"?</summary>

The core challenge for PRMs is the **multi-modal solution distribution**: for the same problem, multiple valid reasoning paths exist (e.g., algebraic vs. geometric approaches), where steps within each path are internally consistent but paths are not directly comparable. During annotation, if annotators are asked "is this step correct?", they may give false negatives when unfamiliar with a particular reasoning style. More subtly, even if a step is correct within its current path, if the overall path is suboptimal, the step-level reward should be adjusted — but this requires a global view, which is fundamentally at odds with PRM's local evaluation nature. Directions for resolution include: (1) **path-conditioned PRM** — evaluating the current step conditioned on preceding steps, rather than in absolute terms; (2) **Monte Carlo estimation** — rolling out from the current step to the final answer and using the success rate as the step-level reward, though computational cost is high; (3) **agreement-based filtering** — annotating only the "critical steps" shared across multiple paths, avoiding path-specific steps.

> **Follow-up:** If Monte Carlo rollout is used to estimate PRM's step-level reward, should the rollout policy be the current training policy or a fixed exploration policy? How does this choice affect the bias and variance of the reward estimate?

</details>

---

<details>
<summary>Q32: Constitutional AI (CAI) claims AI feedback can replace human feedback, but where is the theoretical ceiling of RLAIF? Can the gap between AI feedback and human feedback be eliminated?</summary>

The theoretical ceiling of RLAIF is bounded by the **capability limits of the AI evaluator**. The core issue is: if the AI evaluator has systematic preferences of its own (e.g., verbosity bias, sycophancy), then a policy trained on its feedback will inherit and even amplify those preferences, creating a **evaluator-policy co-adaptation** degeneracy loop. The deeper limitation is the **unverifiability of value alignment** — certain dimensions of human preference (such as honesty and harmlessness) fundamentally require human judgment, and AI cannot self-validate. CAI's "constitutional principles" attempt to circumvent this with explicit rules, but rules cannot cover all corner cases, and conflicts between rules require human arbitration. Empirically, RLAIF can approach human feedback on certain objective dimensions (e.g., format correctness), but still has a significant gap on dimensions requiring deep value judgment (e.g., nuanced harm assessment). Theoretically, RLAIF can only achieve RLHF-level performance when the AI evaluator is an **unbiased and consistent estimator** of human preferences — an assumption that currently cannot be guaranteed.

> **Follow-up:** If the AI evaluator has a known bias (e.g., verbosity bias), can debiasing techniques (e.g., calibration, adversarial training) correct it before RLAIF training? What are the theoretical guarantees of such correction?

</details>

---

<details>
<summary>Q33: In multi-turn RLHF, how should the dynamics of user strategy be modeled? What systematic errors arise from assuming a fixed user strategy?</summary>

Standard multi-turn RLHF implicitly makes a **stationary user assumption** — that the user follows a fixed response strategy throughout the conversation. In reality, users adjust their questioning strategy based on the model's replies (e.g., pressing harder when the model evades a question, asking for brevity when the model is too verbose). This transforms RLHF from a single-agent MDP into a **two-player Markov Game**. Under a non-stationary user strategy, the fixed-user assumption causes: (1) **overfitting to the simulated user** — the policy learns optimal responses for a particular simulated user pattern rather than a robust strategy for real dynamic users; (2) **exploitation of user patience** — if the simulated user never terminates the conversation due to overly long responses, the policy learns an excessively verbose style. The more fundamental difficulty is that real user strategies are themselves a distribution and may even shift because of model behavior (user-model co-evolution), which theoretically approaches **non-stationary multi-agent RL**, for which no mature convergence guarantees currently exist.

> **Follow-up:** If you want to explicitly model dynamic user strategies, could a user simulator be jointly trained with the policy? What are the known failure modes of such a self-play framework?

</details>

## §A Key Papers Timeline

- **2015-06 · GAE** — Schulman et al., ICLR 2016. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) — Introduces Generalized Advantage Estimation, a TD(λ)-style exponentially-weighted multi-step return that continuously interpolates the bias-variance trade-off via λ; the standard advantage estimator underlying PPO and RLHF.

- **2022-03 · InstructGPT** — Ouyang et al., NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) — Establishes the canonical 3-stage RLHF pipeline (SFT → Bradley-Terry reward model → PPO with KL penalty) for aligning GPT-3 into an instruction-following assistant.

- **2022-09 · WiSE-FT** — Wortsman et al., CVPR 2022. [arXiv:2109.01903](https://arxiv.org/abs/2109.01903) — Reduces alignment tax by linearly interpolating weights of the fine-tuned and base models in weight space, preserving pre-training robustness while retaining task performance.

- **2022-12 · Constitutional AI (CAI / RLAIF)** — Bai et al., arXiv preprint. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) — Replaces human preference annotators with an LLM guided by explicit constitutional principles via a self-critique-revision loop, enabling scalable RLAIF without per-sample human labels.

- **2023-05 · DPO** — Rafailov et al., NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — Derives a closed-form reparameterization of the KL-constrained RLHF objective that eliminates the explicit reward model, reducing preference alignment to a single supervised classification loss on (chosen, rejected) pairs.

- **2023-10 · IPO** — Azar et al., AISTATS 2024. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036) — Identifies that DPO's Bradley-Terry/logit mapping allows KL regularization to vanish under near-deterministic preferences; proposes ΨPO with Ψ = Identity, yielding a bounded squared-loss objective that preserves effective regularization.

- **2024-02 · DeepSeekMath / GRPO** — Shao et al., arXiv preprint. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — Introduces Group Relative Policy Optimization (GRPO), which removes the PPO critic by normalizing rewards within a sampled group for the same prompt, halving the number of models required and enabling stable RL from verifiable rewards.

- **2024-02 · KTO** — Ethayarajh et al., ICML 2024. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) — Replaces paired (chosen, rejected) preference data with pointwise binary desirability labels, framing alignment as prospect-theoretic utility maximization with an asymmetric sigmoid loss and a KL-based reference point.

- **2024-02 · DPOP (Smaug)** — Pal et al., arXiv preprint. [arXiv:2402.13228](https://arxiv.org/abs/2402.13228) — Shows that DPO can decrease the log-probability of chosen responses when chosen and rejected are near-identical; adds a max(0,·) penalty term to anchor chosen log-prob above the reference model.

- **2024-03 · ORPO** — Hong et al., arXiv preprint. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) — Merges SFT and preference alignment into a single stage by appending a reference-free odds-ratio contrastive term to the cross-entropy loss, eliminating the need for a frozen reference model.

- **2024-05 · SimPO** — Meng et al., NeurIPS 2024. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734) — Aligns DPO's implicit reward with generation-time likelihood by using length-normalized average log-probability and adds an explicit target margin γ, removing the reference model and mitigating length bias.

- **2024-10 · Likelihood Displacement** — Razin et al., ICLR 2025. [arXiv:2410.08847](https://arxiv.org/abs/2410.08847) — Proves that DPO shifts probability mass away from chosen responses when chosen and rejected share high hidden-embedding similarity (CHES score), potentially causing "unintentional unalignment"; proposes CHES-based data filtering as a remedy.

- **2025-01 · DeepSeek-R1** — Guo et al., Nature 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — Demonstrates that chain-of-thought reasoning ability can emerge from pure GRPO-based RL on a base model (R1-Zero), and that a cold-start SFT stage followed by two rounds of RL and rejection-sampling SFT yields a reasoning model competitive with OpenAI o1.
