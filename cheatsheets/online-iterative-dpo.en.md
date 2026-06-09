# Online & Iterative DPO Cheatsheet
## From offline DPO's distribution shift to on-policy sampling, self-rewarding, self-play, and game-theoretic preference optimization

---

## 1. Overview

Standard **DPO** (Rafailov et al., [arXiv:2305.18290](https://arxiv.org/abs/2305.18290), NeurIPS 2023) is an **offline** algorithm: the preference dataset $\mathcal{D}$ is collected once before training, drawn from some fixed generating policy $\mu$ (usually the SFT model). During training $\pi_\theta$ keeps updating, but $\mathcal{D}$ stays frozen — this is **off-policy**. **Online / iterative DPO** changes exactly one thing: **every round, re-sample responses and rebuild preference pairs with the current policy $\pi_\theta^{(t)}$**, so the training signal always hugs the current policy's output distribution (**on-policy**).

```
Offline DPO (off-policy)                   Online / iterative DPO (on-policy)
─────────────────────                      ──────────────────────────
data sampled once from fixed μ (SFT)       re-sample each round from current π_θ^(t)
once π_θ drifts from μ, (y_w,y_l)          preference pairs always match the
no longer cover its outputs                current policy distribution
cheap, reproducible, no online infra  ⇄    expensive (sample+score+train each round), needs annotator
locked to the stale distribution,     ⇄    can explore new outputs, keep approaching
hard to surpass the data                   the optimal policy
```

### 1.1 A family tree: two orthogonal axes

Almost all differences between online preference-optimization methods fall on two **orthogonal questions**: **① Where do responses come from? ② Who labels the preferences?**

| Axis | Spectrum | Examples |
|------|--------|------|
| **① Response source** | fixed $\mu$ (offline) → current $\pi_\theta$ (on-policy) → $\pi_\theta$ with exploration | DPO → Online DPO → XPO |
| **② Preference labeling** | human / external RM / LLM-as-judge / **the model itself** / game equilibrium | RLHF·OAIF / Self-Rewarding / Nash-PO·SPPO |

- **§3** covers the minimal loop for "making DPO on-policy" + how preference pairs are constructed (RM vs LLM-judge vs human).
- **§4** pulls the annotator **into the model itself**: self-rewarding (judge yourself), self-play (use your own outputs as negatives).
- **§5** upgrades "passive on-policy sampling" into **active exploration** and **game equilibrium** (no BT transitivity assumption needed).
- **§6** lands on production recipes (Llama-3 six rounds, Tülu-3) and the pitfalls specific to the loop.

### 1.2 Scope of this page

> ✅ This page only covers **the "make preference optimization on-policy / iterative" layer**. The following is **not repeated here** — follow the cross-links:
> - **DPO loss derivation, BT model, implicit reward** → [llm-post-training §6](cheatsheet-llm-post-training-en.html)
> - **Loss comparison of offline variants IPO / KTO / ORPO / SimPO** → [llm-post-training §7](cheatsheet-llm-post-training-en.html) (§7.5 already gives an online-vs-offline summary table; this page deepens it)
> - **How to train / evaluate reward models, process rewards** → [reward-modeling-eval](cheatsheet-reward-modeling-eval-en.html)
> - **Policy-gradient details of online RLHF (PPO / GRPO)** → [llm-post-training §8](cheatsheet-llm-post-training-en.html) and [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html)

---

## 2. The Case for On-Policy

### 2.1 Three failure modes of offline DPO

**(a) Distribution mismatch.** Once $\pi_\theta$ leaves $\mu$ during training, the $(y_w, y_l)$ in $\mathcal{D}$ fall into regions $\pi_\theta$ now rarely produces. DPO widens the margin on these **stale samples**, while giving no supervision at all on the outputs the current policy actually generates — the gradient is spent on "paths it will never take again."

**(b) Likelihood displacement.** The DPO loss only constrains the **difference of log-ratios** between chosen and rejected, $\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}$; it does **not** directly constrain the absolute value of $\log\pi_\theta(y_w)$. Note this is not inevitable for an isolated pair update — in the single-step softmax update of §3.1, $y_w$'s probability actually rises. But in **real-LM shared-parameter aggregate training** (massive conflicting preference pairs, sequence-level normalization, optimizer dynamics stacked together), a "shortcut to satisfaction" emerges: push $y_l$ down harder while $y_w$'s probability **is dragged down too** — as long as it drops more slowly than $y_l$, the margin still grows. The squeezed-out probability mass often flows toward a **third class of OOD outputs** (neither $y_w$ nor $y_l$). On-policy data makes $y_w$ come from the current distribution to begin with, mitigating this "push the good answer down too" degeneration.

**(c) OOD over-optimization.** DPO's implicit reward $\hat r_\theta = \beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ is a **reparameterization** of the current/reference policy ratio, not an independent RM that "extrapolates." The problem: regions the data does not cover **lack preference supervision to calibrate** this ratio, offline training can't reach them, and the policy may drift toward "implicit reward inflated but actually bad." On-policy re-sampling keeps pulling "where the policy really goes now" back into the labeling loop.

> 🚨 The three **compound and amplify inside the iterative loop**: this is exactly why §6 stresses "refresh the RM each round / add a chosen-NLL anchor / control length."

**From-scratch implementation**: batch-level DPO loss with mask, β, and reference logps (interview hand-tear standard):

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp, logp_ref, mask, beta=0.1):
    """DPO loss (batch version with mask, interview hand-tear standard).
    logp/logp_ref: (B, T) per-token log-prob under π_θ and π_ref
    mask: (B, T) valid tokens (excludes padding); first half of B is chosen, second half rejected
    Returns a scalar loss."""
    # seq logp: total log-prob per sequence
    seq_logp   = (logp * mask).sum(dim=1)      # (B,) π_θ
    seq_logp_r = (logp_ref * mask).sum(dim=1)  # (B,) π_ref
    # Split back into chosen / rejected (first B/2 are chosen)
    B = logp.shape[0] // 2
    logp_c, logp_r_c = seq_logp[:B], seq_logp_r[:B]       # chosen
    logp_j, logp_r_j = seq_logp[B:], seq_logp_r[B:]       # rejected
    # Implicit reward difference β·[(log π_c/π_ref_c) - (log π_j/π_ref_j)]
    log_ratio_diff = beta * ((logp_c - logp_r_c) - (logp_j - logp_r_j))
    loss = -F.logsigmoid(log_ratio_diff).mean()            # -log σ(Δ)
    return loss
# Key points:
# ① logp_ref for both chosen and rejected must be computed under π_ref, not π_θ
# ② β controls how strongly to learn from preferences (large = conservative, small = aggressive)
# ③ This is the base DPO formulation; the iterative version simply calls this loss with fresh pairs each round
# ④ Variants: IPO (loss = (log_ratio_diff - 1/(2β))²), KTO (single-sample, no pairs needed)
```

### 2.2 Online vs offline: where the performance gap comes from (Tang et al. 2024)

**Tang et al.** (["Understanding the Performance Gap between Online and Offline Alignment Algorithms"](https://arxiv.org/abs/2405.08448), [arXiv:2405.08448](https://arxiv.org/abs/2405.08448), preprint) ran a set of **controlled empirical / mechanistic studies** (experiments + ablations, not a theorem proof), systematically asking "why is online consistently better than offline":

- **Phenomenon**: under controlled settings, **online algorithms consistently beat offline algorithms**, and the gap **cannot** be closed merely by "feeding offline methods more data / expanding coverage."
- **Ruled out**: they checked and **rejected** several naive explanations one by one — the gap is not simply due to insufficient **discriminative accuracy** of offline methods, nor determined by the **loss-function form** (contrastive offline loss vs online RL); even when offline methods use a strong contrastive loss, the gap persists.
- **Core attribution**: **on-policy sampling itself** (data generated by the current policy) is the key driver, not the "offline/online algorithm" label. Offline methods underperform precisely on the responses "they should have learned to distinguish."

> 💡 In one line: in their setting, **making the data on-policy matters more than swapping the loss function**. This gives "online / iterative DPO" empirical support independent of the specific loss — you can **prioritize** changing DPO's data source to on-policy without necessarily swapping the loss first; but this does not mean "swapping the loss / going to RL is never necessary" (see the caveat below).

> ⚠ Caveat: the above is Tang et al.'s conclusion under their specific setting; **do not extrapolate it to "online is necessarily better on any task."** Online's costs (sampling + scoring + training cost, reward-hacking risk) are real — see §6.

---

## 3. Online DPO Algorithms

### 3.1 The Iterative Loop

To turn offline DPO into on-policy, the minimal loop is just three steps, repeated round by round:

```
for t in 0..T-1:
   1) Generate: for each prompt x, sample K responses from the current policy π_θ^(t)   ← on-policy
   2) Label:    use RM / LLM-judge / human to rank the K, build preference pairs (y_w,y_l) → D^(t)
   3) Update:   π_θ^(t+1) ← DPO-update(π_θ^(t), D^(t);  ref = π_ref)
```

Formally, as given in [llm-post-training §7.5](cheatsheet-llm-post-training-en.html):

$$\pi_\theta^{(t+1)} \leftarrow \text{DPO-update}\!\left(\pi_\theta^{(t)},\;\mathcal{D}^{(t)}\right), \quad \mathcal{D}^{(t)} \sim \pi_\theta^{(t)}$$

The toy code below uses a **discrete response space** to show why on-policy preference pairs keep pushing the policy toward high-reward regions, while one-shot offline data "goes stale." DPO's gradient on a softmax policy has a clean closed form — the logsumexp term cancels, so the gradient of the margin w.r.t. the logits is just $\beta(e_w - e_l)$:

```python
import numpy as np

# ===== DPO on a toy categorical policy =====
# logπ_i = θ_i - logsumexp(θ);  d(logπ_w - logπ_l)/dθ_j = [w==j] - [l==j]
# so the gradient of the margin w.r.t. logits is independent of logsumexp -- a clean closed form.

def log_softmax(theta):
    m = theta.max()
    return theta - m - np.log(np.exp(theta - m).sum())

def softmax(theta):
    z = theta - theta.max()
    e = np.exp(z)
    return e / e.sum()

def dpo_step(theta, theta_ref, w, l, beta=0.5, lr=0.3):
    """One DPO gradient-descent step on a preference pair (w wins, l loses); returns updated logits."""
    lp, lpr = log_softmax(theta), log_softmax(theta_ref)
    margin = beta * ((lp[w] - lpr[w]) - (lp[l] - lpr[l]))
    sig = 1.0 / (1.0 + np.exp(-margin))      # σ(margin)
    coef = (sig - 1.0) * beta                # dL/dmargin = σ(margin) - 1 < 0
    grad = np.zeros_like(theta)
    grad[w] += coef                          # raise w
    grad[l] -= coef                          # lower l
    return theta - lr * grad

def expected_reward(theta, r):
    return float((softmax(theta) * r).sum())

def make_pair(samples, r):
    """From a batch of samples, take (best, worst) by true reward as the preference pair."""
    s = sorted(samples, key=lambda i: r[i])
    return s[-1], s[0]

# true reward: 5 discrete responses, index 4 is best
r = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
theta0 = np.zeros(5)                          # SFT start: uniform
rng = np.random.default_rng(0)

# --- offline: preference pairs sampled once from π0, then reused repeatedly ---
off = theta0.copy()
pool = [make_pair(rng.choice(5, size=2, p=softmax(theta0), replace=False), r)
        for _ in range(40)]
for w, l in pool:
    off = dpo_step(off, theta0, w, l)

# --- online: re-sample preference pairs from the current policy at every step ---
on = theta0.copy()
for _ in range(40):
    w, l = make_pair(rng.choice(5, size=2, p=softmax(on), replace=False), r)
    on = dpo_step(on, theta0, w, l)           # ref fixed to SFT

print("offline E[r] =", round(expected_reward(off, r), 3))
print("online  E[r] =", round(expected_reward(on, r), 3))
```

> 💡 Toy intuition: both widen the margin, but **online** feeds back "the pairs the policy really samples now" each round, continuously concentrating mass on high-reward responses; **offline**'s pool drifts further from the current policy the more it is reused, and its marginal returns decay. In a real system this gap is widened further by likelihood displacement and over-optimization.

### 3.2 How to build preference pairs: RM vs LLM-judge vs human

After sampling $K$ on-policy responses, **who decides $(y_w, y_l)$** determines the method's cost and bias:

| Annotator | How it labels | Pros | Risks |
|--------|--------|------|------|
| **External RM** | score → take highest/lowest, or sample by score | cheap, batchable, reuses an existing RM | RM gets hacked; score bias (length, etc.) amplifies over rounds |
| **LLM-as-judge** | have a strong model judge pairs (**OAIF**) | no RM training, online instant labeling | inherits the judge model's preference/style bias; the judge errs too |
| **Human** | manual preference labeling | most trustworthy signal, most expensive | slow, costly, hard to do every round (Llama-3 uses a human+RM mix) |

**OAIF** (Guo et al., ["Direct Language Model Alignment from Online AI Feedback"](https://arxiv.org/abs/2402.04792), [arXiv:2402.04792](https://arxiv.org/abs/2402.04792), preprint) is the representative of "LLM-judge online labeling": for each step's two responses sampled from the current policy, an **online annotator LLM** judges which is better on the spot, then a DPO update follows — replacing offline DPO's "static preference set" with "online AI feedback," getting both on-policy and RM-free.

> 📝 Pairing is not only "highest vs lowest." **RSO** (Liu et al., ["Statistical Rejection Sampling Improves Preference Optimization"](https://arxiv.org/abs/2309.06657), [arXiv:2309.06657](https://arxiv.org/abs/2309.06657), ICLR 2024) points out: the ideal preference pair should be sampled from the distribution of the **optimal policy $\pi^\*$** (note: the target optimal policy, not the current one), so it uses **rejection sampling** to approximately draw samples close to $\pi^\*$ from $\pi_\text{ref}$ and then label them, nudging the "offline data source" toward the **ideal $\pi^\*$ distribution** — a step bridging offline sampling and that ideal.

### 3.3 Reference policy and β: two knobs in the loop

In the iterative loop, the settings of $\pi_\text{ref}$ and $\beta$ directly determine stability:

- **Reset $\pi_\text{ref}$ each round vs fix it to SFT.**
  - **Reset each round** $\pi_\text{ref} \leftarrow \pi_\theta^{(t)}$: a trust-region-like step, the KL constraint is always "relative to the previous round," updates are more stable; the cost is **losing the anchor to SFT**, so after many rounds the whole thing may drift away.
  - **Fix SFT** as $\pi_\text{ref}$: always anchored to the start; but the more $\pi_\theta$ drifts, the larger $\frac{\pi_\theta}{\pi_\text{ref}}$ becomes, the **looser** the effective constraint, and the easier over-optimization gets late on.
  - Production practice (Llama-3 / Tülu-3) mostly **fixes ref within a round** and swaps the baseline between rounds.
- **$\beta$ (KL strength).** Larger $\beta$ hugs $\pi_\text{ref}$ more, more conservative; smaller $\beta$ dares to drift more, easier to over-optimize. For the semantic contrast between $\beta$ and ref, see [llm-post-training §9.3](cheatsheet-llm-post-training-en.html).

---

## 4. Self-Rewarding & Self-Play

§3's annotators are all "external." This section pulls the annotator **into the model itself** — the benefit is escaping the external RM/human bottleneck; the risk is that the signal and the policy are **homologous**, easily self-reinforcing bias.

### 4.1 Self-Rewarding LMs

**Self-Rewarding LMs** (Yuan et al., [arXiv:2401.10020](https://arxiv.org/abs/2401.10020), ICML 2024) make **one model serve as both policy and judge**: an *LLM-as-a-Judge* prompt has the model score its own sampled responses, builds preference pairs from that, then runs iterative DPO. The key narrative is that **judging ability rises together with policy ability** — the paper observes instruction-following and "being a judge" improving in lockstep across iterations, forming a self-improvement loop (the paper runs three iterations, $M_1\!\to\!M_2\!\to\!M_3$).

```python
# ===== Self-Rewarding: the model judges itself and builds preference pairs (§4.1) =====
def self_reward_pairs(prompt, gen_fn, judge_fn, k=4):
    """Sample k, score them yourself, take (highest, lowest) as the pair; if all tie, no signal -> skip."""
    cands = [gen_fn(prompt) for _ in range(k)]
    scored = sorted(((judge_fn(prompt, c), c) for c in cands),
                    key=lambda t: t[0], reverse=True)
    if scored[0][0] == scored[-1][0]:
        return None                          # no discrimination, skip this prompt
    return scored[0][1], scored[-1][1]       # (y_w, y_l)
```

> 🚨 Failure mode: judge and policy are homologous → **reward hacking / self-preference** gets amplified by the loop (the model favors its own style, gives itself inflated scores), and after many rounds it may **saturate or degenerate**. In practice, fall back on "fix a portion of external / verifiable signal + periodic human review."

### 4.2 Self-play: SPIN and SPPO

**SPIN** (Self-Play Fine-Tuning, Chen et al., [arXiv:2401.01335](https://arxiv.org/abs/2401.01335), ICML 2024) **needs no preference labels and no external reward**: treat the **SFT human data as $y_w$ (positive)** and the **model's own current generations as $y_l$ (negative)**, training the model with a DPO-style contrastive objective to distinguish "human data" from "its own outputs." This is a discriminator/generator self-play — it converges when the model's generations become **indistinguishable in distribution** from the SFT data ($\pi \to p_\text{data}$).

```python
# ===== SPIN: self-play pairing -- human data = win, model's own sample = lose (§4.2) =====
def spin_pairs(prompts, human_responses, model_gen_fn):
    """y_w taken from SFT human data, y_l from the current model; if identical, no signal -> skip."""
    pairs = []
    for x, y_human in zip(prompts, human_responses):
        y_self = model_gen_fn(x)
        if y_self != y_human:
            pairs.append((x, y_human, y_self))   # (prompt, y_w, y_l)
    return pairs
```

> ⚠ SPIN's optimization target (fixed point) is the SFT data distribution: it learns to "approach the human data," and with **no external reward** it cannot push the target beyond that distribution — this is the essential difference from "online DPO with an external reward."

**SPPO** (Self-Play Preference Optimization, Wu et al., [arXiv:2405.00675](https://arxiv.org/abs/2405.00675), preprint / NeurIPS 2024 Workshop) models alignment as a **two-player constant-sum game**, targeting the **Nash equilibrium** of preferences: each round it estimates the win-rate between self-sampled responses with a preference model and takes a **multiplicative-weights / quadratic** update toward the equilibrium policy. It does not assume preferences are explained by a single scalar reward (BT) — which leads us to §5's game-theoretic view.

---

## 5. Exploration & Game-Theoretic PO

### 5.1 Why "game-theoretic": preferences may be intransitive

BT / reward-based assumptions hold that **preference probability is explained by a difference of scalar rewards** (one scalar reward per response), so preferences are **transitive** in expectation. But real human preferences may be **intransitive**: a cycle $A\succ B\succ C\succ A$ appears, which no scalar reward can express. **Game-theoretic preference optimization** sidesteps this assumption: rather than finding a "reward-maximizing" policy, it finds the **Nash equilibrium** of the two-player preference game (maximizing the minimum win-rate against all opponents).

**Nash-LHF / Nash-MD** (Munos et al., ["Nash Learning from Human Feedback"](https://arxiv.org/abs/2312.00886), [arXiv:2312.00886](https://arxiv.org/abs/2312.00886), ICML 2024): first learn a **preference model** $\mathcal{P}(y\succ y'\mid x)$ (rather than a reward model), then use **Nash-MD** (mirror descent) to iteratively solve for the **Nash equilibrium of the regularized game**, with provable convergence in the tabular / regularized setting. It generalizes RLHF from "reward maximization" to "preference-game equilibrium"; §4.2's SPPO is a self-play instance of the same idea.

### 5.2 Active exploration: XPO

Passive on-policy sampling just "samples randomly from the current policy," and does **not** deliberately explore high-potential but uncertain regions. **XPO** (Exploratory Preference Optimization, Xie et al., [arXiv:2405.21046](https://arxiv.org/abs/2405.21046), ICLR 2025) **adds just one optimism bonus** to the DPO objective, encouraging the policy to explore responses "whose implicit reward might be high but is currently uncertain"; via implicit $Q^\*$-approximation it is provably **sample-efficient** under its theoretical assumptions. In one line: XPO = online DPO + one line of optimistic exploration, upgrading "lucky-dip sampling" into "directed exploration."

> 💡 Spectrum: **offline DPO** (passively use stale data) → **online DPO** (passive on-policy sampling) → **XPO** (active exploration). Well-designed active exploration is better at escaping the small distribution "the current policy already knows."

### 5.3 Active Querying

When the labeling budget is limited, **which prompts / which pairs to label** can also be optimized: prioritize labeling pairs where the **RM is uncertain / information gain is large**, rather than labeling uniformly. For combining this with RM uncertainty estimation, see [reward-modeling-eval](cheatsheet-reward-modeling-eval-en.html).

---

## 6. Practical Recipes & Pitfalls

### 6.1 Production-scale iterative recipes

- **Llama 3** (Grattafiori et al., ["The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783), [arXiv:2407.21783](https://arxiv.org/abs/2407.21783), preprint): post-training runs **six rounds** of iteration, each round = reward modeling + rejection sampling + SFT + **DPO**; each round's preference data is generated by **the best model from the previous round** and labeled by humans — a production-scale example of iterative DPO.
- **Tülu 3** (Lambert et al., ["Tülu 3: Pushing Frontiers in Open Language Model Post-Training"](https://arxiv.org/abs/2411.15124), [arXiv:2411.15124](https://arxiv.org/abs/2411.15124), COLM 2025): an open **SFT → DPO → RLVR** recipe, where DPO uses a large-scale **on-policy preference mix** (sample completions from the policy model, then label preferences with models such as GPT-4). For RLVR details see [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html).
- **Iterative Reasoning PO** (Pang et al., ["Iterative Reasoning Preference Optimization"](https://arxiv.org/abs/2404.19733), [arXiv:2404.19733](https://arxiv.org/abs/2404.19733), NeurIPS 2024): iterative DPO for **reasoning / CoT** — it sets $y_w$/$y_l$ by **whether the answer is correct** (a verifiable signal), and **adds an extra NLL/SFT term on $y_w$** in the loss to suppress likelihood displacement (§2.1b), improving round by round on GSM8K / MATH and the like. A template for "iterative DPO + verifiable signal + chosen anchor."

### 6.2 Pitfalls specific to the loop

| Pitfall | Mechanism | Mitigation |
|------|------|------|
| **Reward hacking / over-optimization** | the proxy RM gets exploited round by round, Goodhart; offline it is exposed once, in the loop it **compounds** | refresh/retrain the RM each round; KL anchor; keep verifiable/human-review signal |
| **Length explosion** | RM / judge prefer longer answers → the loop amplifies length drift | length normalization (SimPO-style), report length-controlled win-rate (LC) |
| **Likelihood-displacement compounding** | if likelihood displacement occurs, the effect of $\log\pi_\theta(y_w)$ being dragged down compounds round by round | add a chosen-NLL term (IRPO/RPO); fix the SFT ref anchor |
| **Diversity collapse** | on-policy repeatedly reinforces high-scoring patterns, sampling diversity drops, signal weakens | raise temperature / sample more candidates; active exploration (XPO); periodically inject new prompts |
| **Compute cost** | each round = sample + score + train, far costlier than one-shot offline DPO | control round count / per-round budget; reuse the previous round's samples |

> 🚨 **SimPO** (Meng et al., [arXiv:2405.14734](https://arxiv.org/abs/2405.14734), NeurIPS 2024) and its length normalization and $\pi_\text{ref}$-free design are often borrowed to mitigate length drift in the iterative loop; but SimPO itself is an offline loss variant — for the full comparison see [llm-post-training §7.4 / §7.6](cheatsheet-llm-post-training-en.html).

### 6.3 Online DPO vs online RLHF (PPO / GRPO)

Both are **on-policy**; the difference is "how the reward signal enters the update":

| Dimension | Online / iterative DPO | Online RLHF (PPO / GRPO) |
|:---|:---|:---|
| Reward | **implicit** (hidden in the DPO loss), via pairwise preference | **explicit** reward, fed into the policy gradient |
| Value network | not needed | PPO needs a critic; GRPO uses a within-group baseline, critic-free |
| Credit assignment | sequence-level (one contrast per whole response) | can do finer (token/step-level) credit + reward shaping |
| Engineering complexity | lower (no RL infra) | higher (rollout + optimizer + KL control) |
| Positioning | between offline DPO and full RLHF | most expressive, most flexible |

> 💡 One-line positioning: **online DPO captures RLHF's on-policy dividend while keeping DPO's simplicity**; the cost is giving up RLHF's fine-grained credit assignment and reward-shaping flexibility. For GRPO/PPO details see [llm-post-training §8](cheatsheet-llm-post-training-en.html).

---

## 7. Interview Questions

### L1 — Foundational

---

<details>
<summary>Q1: Is standard DPO on-policy or off-policy? Why?</summary>

**Answer:** **Off-policy (offline).** The preference dataset $\mathcal{D}$ is sampled once before training from a fixed policy $\mu$ (usually the SFT model); during training $\pi_\theta$ keeps updating while $\mathcal{D}$ stays frozen. Once $\pi_\theta$ drifts from $\mu$, the $(y_w,y_l)$ in the data no longer cover the outputs the current policy actually generates — that is off-policy distribution mismatch. Online / iterative DPO's change is to re-sample with the current $\pi_\theta^{(t)}$ each round, making it on-policy.

> **Follow-up:** Are "iterative DPO" and "online DPO" the same thing?
> They are often used interchangeably; the core of both is "re-sample preference pairs with the current policy." When distinguished, "iterative" stresses the discrete outer loop of sample-a-round-train-a-round, while "online" can mean finer-grained sample-and-train-as-you-go; this page uses both to mean on-policy preference optimization.

</details>

---

<details>
<summary>Q2: What are the three steps of iterative DPO's minimal loop?</summary>

**Answer:** ① **Generate**: for each prompt, sample $K$ responses from the current policy $\pi_\theta^{(t)}$ (on-policy); ② **Label**: use RM / LLM-judge / human to rank them and build preference pairs $(y_w,y_l)$, yielding $\mathcal{D}^{(t)}$; ③ **Update**: $\pi_\theta^{(t+1)}\leftarrow\text{DPO-update}(\pi_\theta^{(t)},\mathcal{D}^{(t)})$. Repeat round by round. The key is that step ① samples with the **current** policy, not from fixed data.

> **Follow-up:** Which step is most expensive?
> Usually generation + labeling: each round must sample and score (more expensive if labeled by humans / a strong model). This is exactly online's main cost over offline.

</details>

---

<details>
<summary>Q3: Online DPO and online RLHF (PPO) are both on-policy — what is the main difference?</summary>

**Answer:** The difference is **how the reward signal enters the update**. Online DPO's reward is **implicit** (hidden in the pairwise DPO loss), needs no explicit reward and no critic, and is sequence-level contrast; PPO/GRPO feed an **explicit reward** into the policy gradient, can do finer (token/step-level) credit assignment and reward shaping, but need RL infrastructure (PPO also needs a critic; GRPO uses a within-group baseline to avoid one). Online DPO sits between offline DPO and full RLHF: it gets the on-policy dividend while keeping DPO's simplicity.

> **Follow-up:** Then why not always use PPO?
> Engineering complexity, parameter sensitivity, high cost. Online DPO captures a good part of the on-policy gain at a smaller cost, a compromise many open-source recipes adopt (e.g., the DPO stage of Tülu-3).

</details>

---

### L2 — Intermediate

---

<details>
<summary>Q4: What is likelihood displacement? Why does on-policy data mitigate it?</summary>

**Answer:** DPO only optimizes the **difference of log-ratios** between chosen and rejected, not the absolute value of $\log\pi_\theta(y_w)$. So the model can push $y_l$ lower while letting $y_w$'s probability **also drop** (as long as it drops more slowly), and the margin still grows; the squeezed-out mass often flows toward a **third class of OOD outputs** — i.e., the "good answer gets pushed down too" degeneration. On-policy data makes $y_w$ come from the current distribution to begin with, and methods like IRPO **add an extra NLL term on $y_w$** to anchor its absolute probability, together mitigating the drift.

> **Follow-up:** Is adding a chosen-NLL term alone enough?
> It can significantly mitigate likelihood displacement, but it does not solve distribution mismatch or over-optimization — those two still need on-policy re-sampling. The two kinds of remedy are orthogonal and are often used together.

</details>

---

<details>
<summary>Q5: Self-Rewarding LM and SPIN are both "self-sufficient" — what is the essential difference?</summary>

**Answer:** **The signal source differs.** Self-Rewarding has the model **judge itself**, scoring its own sampled responses (LLM-as-judge), so **both winner and loser in the pair come from the model's generations**, and it can in principle surpass the initial data as the model gets stronger. SPIN **neither scores nor needs preference labels**: it fixes **SFT human data as the winner** and the model's own samples as the loser, doing discriminative self-play, converging to "generations indistinguishable from human data." Therefore SPIN's fixed point is the **SFT data distribution** (it cannot push beyond that distribution without an external reward), while Self-Rewarding may break through (but with the risk of self-preference amplification).

> **Follow-up:** What is each one's main risk?
> Self-Rewarding: judge and policy are homologous → reward hacking / self-preference amplified by the loop, may saturate. SPIN: limited by the quality and coverage of the SFT data — if the data is poor, the ceiling is low.

</details>

---

<details>
<summary>Q6: In the iterative loop, should π_ref be reset each round or fixed to SFT? What is the cost of each?</summary>

**Answer:** **Reset each round** $\pi_\text{ref}\leftarrow\pi_\theta^{(t)}$ is like a trust-region step, the KL constraint is relative to the previous round, updates are more stable, but it **loses the anchor to SFT** and the whole thing may drift away after many rounds. **Fix SFT** always anchors the start, but the more $\pi_\theta$ drifts, the larger $\pi_\theta/\pi_\text{ref}$ becomes, the looser the effective constraint, and the easier over-optimization gets late on. Production practice mostly **fixes ref within a round** and swaps the baseline between rounds; $\beta$ simultaneously tunes how tightly it hugs ref (large = conservative, small = drifts easily).

> **Follow-up:** Is this the same as PPO's KL control?
> Same spirit (both limit the policy from going too far from the reference), but DPO's KL is implicitly encoded in $\beta\log(\pi_\theta/\pi_\text{ref})$, while PPO uses an explicit KL penalty/clipping. See [llm-post-training §9.3](cheatsheet-llm-post-training-en.html).

</details>

---

<details>
<summary>Q7: What is Tang et al. 2024's core conclusion about "why online beats offline"?</summary>

**Answer:** In their controlled study: ① online **consistently beats** offline, and the gap **cannot** be closed by feeding offline more data / expanding coverage; ② the gap is **not** determined by insufficient discriminative accuracy of offline methods nor by the loss-function form (contrastive offline vs online RL) — even with a strong contrastive loss the gap remains; ③ the core attribution is that **on-policy sampling itself** (data generated by the current policy) is the key driver, not the "online/offline algorithm" label. Takeaway: **making the data on-policy matters more than swapping the loss**; you need not abandon DPO, just change its data source to on-policy.

> **Follow-up:** Can we conclude from this that "online is better on any task"?
> No. This is a conclusion under their specific setting, and online has real costs (sampling/labeling/training cost, reward hacking). It must be weighed against the task and budget.

</details>

---

### L3 — Advanced

---

<details>
<summary>Q8: Why "game-theoretic" preference optimization (Nash)? What flawed premise of BT/reward-based methods does it fix?</summary>

**Answer:** BT/reward-based methods assume preference probability is explained by a **difference of scalar rewards**, so preferences are **transitive** (in expectation). But real human preferences may be **intransitive** (the cycle $A\succ B\succ C\succ A$), and then no scalar reward can explain the preferences. The game-theoretic approach (NLHF / Nash-MD, Munos et al.) instead learns a **preference model** $\mathcal{P}(y\succ y'\mid x)$ and solves for the **Nash equilibrium** of the two-player game (maximizing the minimum win-rate against all opponents), with no transitivity assumption needed, and with provable convergence to the game equilibrium in the tabular / regularized setting. SPPO is a self-play instance of the same idea.

> **Follow-up:** Do the Nash-equilibrium policy and the "reward-maximizing policy" coincide when preferences are transitive?
> When preferences happen to be induced by a BT reward (transitive), the two tend to coincide; the Nash framework is the more general superset, still well-defined when intransitive — which is precisely its value.

</details>

---

<details>
<summary>Q9: Where do passive on-policy sampling and XPO's "active exploration" differ? Why does exploration bring sample efficiency?</summary>

**Answer:** Passive on-policy just "samples randomly from the current policy," with sampling concentrated in high-probability regions the policy **already knows**, and it **will not deliberately try** responses that are "unseen but possibly better"; iterate long enough and diversity collapses, the signal weakens. XPO (Xie et al. 2024) adds **one optimism bonus** to the DPO objective, actively favoring responses "whose implicit reward might be high but is currently uncertain," and via implicit $Q^\*$-approximation is provably **sample-efficient** under its theoretical assumptions. Intuition: exploration spends the labeling budget where **information gain is large** (uncertain regions), rather than repeatedly confirming known good answers — exactly the classic "optimism in the face of uncertainty" efficiency-gain logic in RL.

> **Follow-up:** How does online DPO without the exploration term degenerate?
> It easily falls into "self-confirmation": repeatedly reinforcing the current high-scoring pattern → sampling diversity drops → new preference pairs become less discriminative → improvement stalls. Active exploration or periodically injecting new prompts / raising temperature can counter this.

</details>

---

<details>
<summary>Q10: In the iterative loop, why are reward hacking and length explosion more dangerous than in one-shot offline DPO? How to mitigate systematically?</summary>

**Answer:** Offline exposes the proxy RM **once**; in the iterative loop, every round labels with the RM/judge and retrains, so the policy drifts **round by round** toward "where the RM overestimates," and the error **compounds** (Goodhart): RM prefers long answers → the loop amplifies length drift; the RM's systematic bias → gets exploited repeatedly. Mitigation is a combo: ① **refresh/retrain the RM each round** (don't let the policy chase a static proxy); ② **KL anchor + fixed SFT ref** to limit per-round drift; ③ **length normalization / report length-controlled win-rate (LC)** to treat length; ④ keep **verifiable signal / human review** as a ground-truth fallback (IRPO uses answer correctness); ⑤ add **chosen-NLL** to prevent likelihood-displacement compounding.

> **Follow-up:** Why is a "verifiable signal" especially valuable in the iterative loop?
> A rule-based verifier (exact-match / unit tests) ≈ ground truth, **harder to hack on well-defined tasks** (but you still must guard against data leakage and spec loopholes), and can cut the compounding-amplification chain of "proxy RM exploited round by round" — which is also why RLVR / IRPO prefer verifiable rewards on reasoning tasks.

</details>

---

<details>
<summary>Q11: Given only a single fixed offline preference set, can you approximate the on-policy gain? What are the means and their ceilings?</summary>

**Answer:** You can only **partially approximate** it, never fully replace. Means: ① **RSO** — use rejection sampling to approximately draw samples close to the optimal policy $\pi^\*$ from $\pi_\text{ref}$ and then label them, nudging the data source toward the on-policy ideal; ② $\pi_\text{ref}$-free / chosen-anchoring (SimPO, adding chosen-NLL) to mitigate likelihood displacement; ③ enlarge $\beta$ to limit drift and avoid over-optimizing outside the stale distribution. **Ceiling**: none of these change the fundamental fact that "the data is generated by a stale policy" — once $\pi_\theta$ drifts into regions the offline set does not cover, there is no supervision at all. Tang et al.'s conclusion makes exactly this point: **on-policy sampling itself** is the key that offline means cannot fully supply.

> **Follow-up:** Then what irreplaceable value does offline DPO still have?
> Cheap, reproducible, no online infrastructure, suited to cold-start / resource-constrained scenarios; many production recipes first lay a base with offline DPO, then stack a few online/iterative rounds on top — a cost-effective compromise.

</details>

---

## Appendix: Key Terms Glossary

| Term | Definition |
|----------|----------|
| Offline DPO | preference data sampled once from fixed μ; off-policy |
| Online / Iterative DPO | re-sample preference pairs each round with the current π_θ; on-policy |
| On-policy / Off-policy | whether the data comes from the policy currently being optimized |
| Distribution Mismatch | after π_θ drifts from μ, stale data no longer covers its outputs |
| Likelihood Displacement | margin grows but logπ(y_w) drops instead |
| Over-optimization | drifting toward OOD directions where the implicit reward is inflated |
| OAIF | online AI feedback: LLM-judge labels on-policy pairs on the spot |
| LLM-as-judge | use a strong model to judge pairs, replacing RM/human |
| RSO | statistical rejection sampling: approximate the optimal-policy distribution, then label |
| Self-Rewarding | one model serves as both policy and judge (scorer) |
| SPIN | self-play fine-tuning: discriminative self-play with human data = win, model's own sample = lose |
| SPPO | self-play preference optimization: a self-play update solving for the Nash equilibrium of preferences |
| NLHF / Nash-MD | Nash learning: learn a preference model, use mirror descent to solve the game equilibrium |
| Intransitive Preference | A≻B≻C≻A, expressible by no scalar reward |
| XPO | exploratory preference optimization: DPO + an optimism term, sample-efficient |
| Optimism Bonus | encourages exploring uncertain / high-potential responses |
| IRPO | iterative reasoning preference optimization: verifiable signal sets winner/loser + chosen-NLL anchor |
| Reward Hacking | exploiting proxy-RM flaws to inflate scores (compounds inside the loop) |
| LC win-rate | length-controlled win-rate, with length bias removed |

---

*This cheatsheet is for study reference only. Paper conclusions and figures defer to the original papers; benchmark scores are illustrative only and do not constitute a head-to-head comparison.*

## §A Key Papers Timeline

- **2023-05 · Direct Preference Optimization: Your Language Model is Secretly a Reward Model** — Rafailov et al., NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — reparameterizes RLHF's reward maximization into a pairwise classification loss on the policy, dispensing with an explicit RM and RL; **offline DPO** is the baseline for all online/iterative methods on this page.

- **2023-09 · Statistical Rejection Sampling Improves Preference Optimization** — Liu et al., ICLR 2024. [arXiv:2309.06657](https://arxiv.org/abs/2309.06657) — points out that the ideal preference pair should be sampled from the optimal-policy distribution, and uses rejection sampling to approximately draw samples close to π* from π_ref before labeling (RSO), nudging the offline data source toward the on-policy ideal.

- **2023-12 · Nash Learning from Human Feedback** — Munos et al., ICML 2024. [arXiv:2312.00886](https://arxiv.org/abs/2312.00886) — learns a preference model rather than a reward model, uses Nash-MD (mirror descent) to solve the Nash equilibrium of the regularized game, generalizing RLHF from reward maximization to a preference game, with no BT transitivity assumption.

- **2024-01 · Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models** — Chen et al., ICML 2024. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) — SPIN: discriminative self-play with SFT human data as winner and the model's own samples as loser, needing no preference labels or external reward; converges when generations are indistinguishable from the data, with the SFT data distribution as the fixed point.

- **2024-01 · Self-Rewarding Language Models** — Yuan et al., ICML 2024. [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) — one model serves as both policy and uses LLM-as-judge to score its own responses to build preference pairs, iterating DPO; judging ability rises in lockstep with policy ability (three rounds M1→M2→M3).

- **2024-02 · Direct Language Model Alignment from Online AI Feedback** — Guo et al., preprint. [arXiv:2402.04792](https://arxiv.org/abs/2402.04792) — OAIF: for each step's two responses sampled from the current policy, an online annotator LLM judges which is better on the spot before a DPO update, replacing the static preference set with online AI feedback, getting both on-policy and RM-free.

- **2024-04 · Iterative Reasoning Preference Optimization** — Pang et al., NeurIPS 2024. [arXiv:2404.19733](https://arxiv.org/abs/2404.19733) — iterative DPO for CoT reasoning: sets winner/loser by answer correctness (a verifiable signal), adds an NLL term on the chosen in the loss to suppress likelihood displacement, improving round by round on GSM8K/MATH and the like.

- **2024-05 · Self-Play Preference Optimization for Language Model Alignment** — Wu et al., preprint / NeurIPS 2024 Workshop (AFM, Oral). [arXiv:2405.00675](https://arxiv.org/abs/2405.00675) — SPPO: models alignment as a two-player constant-sum game, using preference win-rates for multiplicative-weights/quadratic updates toward the Nash equilibrium, not relying on a single scalar reward (BT).

- **2024-05 · Understanding the Performance Gap between Online and Offline Alignment Algorithms** — Tang et al., preprint. [arXiv:2405.08448](https://arxiv.org/abs/2405.08448) — controlled study: online consistently beats offline, the gap cannot be closed and is not determined by the loss form, the core attribution being **on-policy sampling itself** — making the data on-policy matters more than swapping the loss.

- **2024-05 · Exploratory Preference Optimization: Harnessing Implicit Q\*-Approximation for Sample-Efficient RLHF** — Xie et al., ICLR 2025. [arXiv:2405.21046](https://arxiv.org/abs/2405.21046) — XPO: adds one optimistic exploration bonus to the DPO objective, encouraging exploration of high-potential uncertain responses, and via implicit Q*-approximation proves sample efficiency under its theoretical assumptions; upgrades passive on-policy into active exploration.

- **2024-07 · The Llama 3 Herd of Models** — Grattafiori et al., preprint. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) — post-training runs six rounds of iteration (each round RM + rejection sampling + SFT + DPO), with preference data generated by the previous round's best model and labeled by humans; a production-scale example of iterative DPO.

- **2024-11 · Tülu 3: Pushing Frontiers in Open Language Model Post-Training** — Lambert et al., COLM 2025. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) — an open end-to-end recipe SFT → DPO → RLVR, with the DPO stage using a large-scale on-policy preference mix; fully public data/code/recipe.
