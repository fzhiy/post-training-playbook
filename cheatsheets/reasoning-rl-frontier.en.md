# Reasoning-RL Frontier

> The hottest track in post-training from 2024–2026: from PPO to critic-free GRPO / RLOO, on to long-CoT reasoning RL (DAPO / Dr.GRPO) and RLVR. Frequently asked at frontier labs (Seed / DeepSeek / Qwen / Moonshot).
> ⚠️ For specific paper numbers (benchmark scores, etc.) always defer to the **original paper**; this page focuses on **mechanisms and trade-offs** and deliberately avoids stacking numbers.

## 0. The evolution

`PPO` (actor + critic + ref + RM, GAE advantage) → **`GRPO`** (drops the critic, uses "group-relative" as the baseline) → **`DAPO` / `Dr.GRPO`** (fixes GRPO's bias and entropy collapse under long-CoT); side branch **`RLOO`** (leave-one-out baseline). Reward source: learned RM → (in verifiable domains) **`RLVR`** (rules / verifiers supply the reward).

## 1. PPO recap

- Four models: policy (actor), value (critic), reference, reward model.
- Advantage computed with **GAE**; objective is the clipped surrogate $L^{CLIP}=\mathbb{E}\big[\min(\rho A,\ \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)A)\big]$, $\rho=\pi_\theta/\pi_{\theta_{old}}$; plus $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})$.
- Pain points: memory (4 models), difficulty training the value network, sparse rewards for long sequences.

## 2. GRPO — dropping the critic / Group Relative Policy Optimization

- For each prompt, sample a **group** of $G$ responses with rewards $r_1..r_G$; use **within-group statistics** as the baseline in place of a value network:
$$A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+\varepsilon}$$
- Objective is the same clipped surrogate as PPO, but the advantage is $A_i$, with **no critic, no GAE**; KL penalty against the reference is retained (k1/k2/k3 estimators and in-reward vs in-loss placement: see [llm-post-training §9.4](cheatsheet-llm-post-training-en.html)).
- Benefits: saves one value model and avoids training a value function; especially stable for **verifiable rewards** (math / code). Used by the DeepSeek family.

**From-scratch implementation** (group z-score advantage + per-token clip + K3 KL, in-loss):

```python
import torch

def grpo_loss(logp, logp_old, logp_ref, rewards, mask, group_size,
              clip_eps=0.2, beta=0.04):
    # logp/logp_old/logp_ref: (B, T) per-token logprobs; B = n_prompts * group_size
    r = rewards.view(-1, group_size)                       # (n_prompts, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    adv = adv.reshape(-1, 1)                               # (B,1) group z-score advantage
    ratio = torch.exp(logp - logp_old)                     # importance ratio ρ
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy = torch.min(surr1, surr2)                       # clipped surrogate
    logr = logp_ref - logp                                 # log(π_ref/π_θ)
    kl = torch.exp(logr) - logr - 1                        # K3 estimator, always ≥ 0
    per_tok = policy - beta * kl                           # KL placed in the loss
    seq = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)  # 1/|o_i| length normalization
    return -seq.mean()
# Dr.GRPO de-bias: drop the /std in adv; replace 1/|o_i| with a constant (e.g. max length).
```

- Key points: ① the advantage is standardized **within the group** (z-score), replacing the value baseline; ② `min(surr1, surr2)` is the same clipping as PPO, but the ratio uses **per-token** logprobs; ③ K3 = $e^{\log r}-\log r-1\ge0$ is an unbiased, non-negative KL estimator (mind the direction of `logr`: $\log(\pi_{ref}/\pi_\theta)$); ④ the trailing `1/|o_i|` length normalization is the original GRPO formulation — Dr.GRPO shows it favors long wrong responses, so de-biasing replaces it with a constant.

## 3. RLOO — REINFORCE leave-one-out

- Also critic-free: the baseline for sample $i$ = **the mean reward of the other $G-1$ samples**, REINFORCE-style gradient.
- Simpler than PPO (no clip / critic), competitive with PPO on RLHF. Difference from GRPO lies in baseline construction (leave-one-out vs. within-group standardization) and whether clipping is applied.

**From-scratch implementation** (leave-one-out baseline + REINFORCE gradient):

```python
import torch

def rloo_loss(logp, rewards, mask, group_size):
    """RLOO: REINFORCE + leave-one-out baseline. rewards are grouped by prompt.
    logp: (B, T) per-token logprob; rewards: (B,) scalar reward per response; mask: (B, T) valid tokens."""
    r = rewards.view(-1, group_size)                        # (n_prompts, G)
    # Leave-one-out baseline: sample i's baseline = mean of the other G-1 rewards
    # Equivalent to (sum(r) - r_i) / (G - 1)
    baseline = (r.sum(1, keepdim=True) - r) / (group_size - 1)
    adv = r - baseline                                      # (n_prompts, G) REINFORCE advantage
    adv = adv.reshape(-1, 1)                                # (B, 1)
    seq_logp = (logp * mask).sum(1)                         # (B,) total log-prob per response
    loss = -(seq_logp * adv.squeeze()).mean()               # REINFORCE: minimize -log-likelihood × advantage
    return loss
# Core differences between RLOO and GRPO:
# 1. Baseline uses leave-one-out mean (G-1 samples) rather than full-group mean (G samples including self)
# 2. No std normalization — reward magnitude directly reflects advantage
# 3. No clipping — pure REINFORCE without PPO-style importance-sampling correction
```

## 4. DAPO — keeping GRPO stable under long-CoT / Decoupled-clip & Dynamic-sAmpling PO

ByteDance 2025 open-source recipe — four modifications targeting long-chain reasoning RL:
1. **Clip-Higher**: **decouple** the upper and lower clip bounds $\epsilon$ and raise the upper bound → gives low-probability tokens room to rise, **preventing entropy collapse** (policy becoming deterministic too early and ceasing to explore).
2. **Dynamic Sampling**: discard prompts where the entire group is correct or entirely wrong (within-group advantage is always 0, yielding zero gradient), ensuring every batch contributes useful signal.
3. **Token-level loss**: average by **token** rather than by **sequence**, preventing the gradient of long responses from being diluted (critical for long CoT).
4. **Overlong reward shaping**: soft penalty for excessively long responses, stabilizing training.

**From-scratch implementation** (three changes on top of the GRPO code):

```python
def dapo_loss(logp, logp_old, logp_ref, rewards, mask, group_size,
              clip_low=0.2, clip_high=0.28, beta=0.0):
    """DAPO: three changes on GRPO — clip-higher + token-level loss + optional KL removal.
    beta=0.0 matches DAPO's original recipe of dropping KL."""
    # --- Same advantage computation as GRPO ---
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    adv = adv.reshape(-1, 1)

    # --- Change 1: Clip-Higher — decoupled upper/lower bounds ---
    ratio = torch.exp(logp - logp_old)
    surr1 = ratio * adv
    # Upper bound clip_high > lower bound clip_low → room for low-prob tokens to rise
    surr2 = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * adv
    policy = torch.min(surr1, surr2)

    # --- Change 2: Token-level loss — all tokens weighted equally ---
    logr = logp_ref - logp
    kl = torch.exp(logr) - logr - 1
    per_tok = policy - beta * kl
    # Divide by total_tokens, not per-sequence average
    total_tokens = mask.sum().clamp(min=1)            # ∑|o_i|
    loss = -(per_tok * mask).sum() / total_tokens      # token-level average

    # --- Change 3 (not in code, handled by data loader): Dynamic Sampling ---
    # Discard prompts where all rewards in the group are identical (all-correct / all-wrong)
    # → guarantees every batch carries effective gradient

    return loss
# Compare to GRPO: ① ε symmetric→asymmetric (clip_high > clip_low);
# ② per-sequence 1/|o_i| → per-token 1/∑|o_i| normalization;
# ③ dynamic sampling filters at batch construction, not in the loss.
```

## 5. Dr.GRPO — fixing GRPO's optimization bias / GRPO Done Right

- Identifies two **biases** in GRPO: **std normalization** in the advantage (amplifies imbalance across problem difficulty) + **1/response-length** normalization in the loss (biases toward "longer wrong responses").
- Fix: **remove the std division + remove length normalization** (replace with a constant) → a more **unbiased** estimate; same performance with fewer tokens and no artificially inflated response length.

## 5.5 GSPO — sequence-level importance ratio / Group Sequence Policy Optimization

> 💡 GSPO (Qwen team, Zheng et al., [arXiv:2507.18071](https://arxiv.org/abs/2507.18071), 2025-07) lifts the granularity of importance-sampling (IS) correction from "each token" to "the whole sequence", mitigating GRPO's instability when training large-scale MoE models.

**Why GRPO's token-level ratio is unstable.** Following PPO, GRPO computes a separate ratio per token, $w_{i,t}=\pi_\theta(y_{i,t}\mid x,y_{i,<t})/\pi_{\theta_\text{old}}(\cdots)$:

- A single-token ratio is a single-sample estimate — intrinsically high variance — and the noise accumulates along long CoT sequences.
- Whenever some $w_{i,t}$ strays outside $[1-\epsilon,1+\epsilon]$, that token's gradient is clipped to zero — frequent in long sequences even when the overall policy shift is small.
- **Acute for MoE**: after an update the router may send the same token to a different set of experts, so numerator and denominator run through different compute paths; routing drift shows up as ratio spikes that trigger clipping, which the paper calls "catastrophic and irreversible model collapse" (its words).

**GSPO's fix: unit matching.** The reward is granted to the whole sequence, so the unit of IS correction should be the sequence too. The sequence-level ratio is the **length-normalized geometric mean**:

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_\text{old}}(y_i\mid x)}\right)^{1/|y_i|}=\exp\!\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_\text{old}}(\cdots)}\right)$$

The objective has the same PPO-clip form, but with the ratio replaced by $s_i$ and a sequence-level advantage $\hat A_i$ (within-group z-score, as in GRPO):

$$J_\text{GSPO}(\theta)=\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G\min\!\big(s_i\hat A_i,\ \mathrm{clip}(s_i,1{-}\varepsilon_l,1{+}\varepsilon_r)\,\hat A_i\big)\right]$$

The whole sequence is either used or clipped as a unit — a single token's routing jump can no longer trigger gradient zeroing on its own.

| Aspect | GRPO (token-level) | GSPO (sequence-level) |
|---|---|---|
| IS ratio | $w_{i,t}=\pi_\theta(y_{i,t})/\pi_{\theta_\text{old}}(y_{i,t})$ | $s_i=(\pi_\theta(y_i)/\pi_{\theta_\text{old}}(y_i))^{1/|y_i|}$ |
| clip range (paper's setup) | $\varepsilon_l{=}0.2,\ \varepsilon_r{=}0.27$ | $\varepsilon_l{=}3{\times}10^{-4},\ \varepsilon_r{=}4{\times}10^{-4}$ |
| clipping granularity | each token independently | whole sequence |
| MoE routing drift | ratio spikes → spurious clipping | geometric mean smooths most jitter |

> 📝 **Don't misread the order-of-magnitude gap in $\varepsilon$.** GSPO's $\varepsilon\sim10^{-4}$ is far smaller than GRPO's $\sim0.2$, but that is a design choice flowing from the **different ratio definitions** — **not** a mathematical inevitability of the geometric mean "compressing shifts to 1". If all tokens move the same direction, $s_i$ is the same order as the token ratios and is **not** compressed. The geometric mean only smooths within-sequence sign-mixed jitter (lower variance); GSPO uses a tiny $\varepsilon$ to impose a tighter sequence-level proximal constraint, so in practice clipping is active almost every step.

**Stability and engineering payoffs (paper's results, no independent replication):**
- MoE stability: sequence likelihood doesn't fluctuate wildly with per-token routing drift, removing the need for the earlier Routing Replay (an internal stopgap, first disclosed in this paper).
- Precision robustness: sequence-level aggregation is insensitive to per-token numerical precision, so one can feed log-probs straight from an inference engine (e.g. vLLM) without recomputing through the training engine.
- On Qwen3-30B-A3B-Base, GSPO's training curves (AIME'24 / LiveCodeBench / CodeForces Elo) beat GRPO; the paper credits it with contributing to Qwen3's performance gains (an association claim, no controlled ablation).

> ⚠️ GMPO ([arXiv:2507.20673](https://arxiv.org/abs/2507.20673)) argues sequence-level clipping is "too aggressive" and discards gradient information, advocating token-level clipping with geometric-mean weighting instead; the two trade off differently and there is no settled verdict yet.

**CISPO** (MiniMax, [arXiv:2506.13585](https://arxiv.org/abs/2506.13585), 2025-06) attacks the clip-zeroes-gradients problem from another angle: instead of clipping the probability ratio (which zeroes the gradient of out-of-range tokens), it clips the **scalar IS weight** itself while keeping every token's gradient. The paper reports ~2× training speedup over DAPO on Qwen2.5-32B. GSPO does unit-matching at the sequence level, CISPO preserves gradient integrity at the token level — two complementary ways to repair GRPO's clipping.

### 5.6 GMPO — token-level clip + geometric-mean weighting / Group Mean Policy Optimization

**GMPO** (Zhao et al., [arXiv:2507.20673](https://arxiv.org/abs/2507.20673), 2025-07, "Geometric-Mean Policy Optimization") argues that GSPO's sequence-level clipping is "too aggressive, discarding fine-grained token-level gradients." It replaces GRPO's arithmetic mean with a **geometric mean** of token-level rewards, rather than GSPO's sequence-level ratio — the paper reports improvements over GRPO on math reasoning. The core disagreement among GSPO vs. GMPO vs. CISPO is **"what should be the unit of clipping granularity"** — sequence (arguing token-level drift is a spurious signal), token (arguing sequence-level loses information), scalar weight (arguing that clipping the probability ratio itself is the wrong thing to clip). Each side rejects the others' premises; this is the most active theoretical debate in this direction as of 2025.

| Method | Clip granularity | Ratio definition | Core change |
|------|------|------|------|
| **GSPO** | Sequence-level | $s_i=(\pi_\theta/\pi_{old})^{1/\vert y_i\vert}$ | Clip the whole sequence together |
| **GMPO** | Token-level | $w_{i,t}=\pi_\theta/\pi_{old}$ | Token-level clip + geometric-mean weighting to reduce variance |
| **CISPO** | Token-level (scalar) | Scalar IS weight (not probability ratio) | Clip the weight itself, not the probability ratio |

```python
import torch
# Toy: G=3 responses of lengths 6/5/4; per-token logprobs under new vs old policy.
logp_new = torch.tensor([[-1.2,-0.8,-1.5,-0.4,-2.1,-1.0],
                         [-0.9,-1.3,-0.7,-1.8,-0.6, 0.0],
                         [-1.1,-0.5,-1.4,-0.9, 0.0, 0.0]])
logp_old = torch.tensor([[-1.3,-0.9,-1.4,-0.5,-2.0,-1.1],
                         [-1.0,-1.2,-0.8,-1.7,-0.7, 0.0],
                         [-1.0,-0.6,-1.3,-1.0, 0.0, 0.0]])
lengths = torch.tensor([6., 5., 4.])
mask = torch.arange(6)[None, :] < lengths[:, None].long()   # (G,T) real-token mask

log_ratio = logp_new - logp_old                             # per-token log-ratio
w_token = torch.exp(log_ratio)                              # GRPO: token-level ratio w_{i,t}
# GSPO: sequence-level ratio = length-normalized geometric mean of token ratios
mean_log_ratio = (log_ratio * mask.float()).sum(1) / lengths
s_seq = torch.exp(mean_log_ratio)                           # s_i = (pi_theta/pi_old)^(1/|y_i|)

eps = 0.2                      # GRPO token-level clip
eps_l, eps_r = 3e-4, 4e-4      # GSPO sequence-level clip (asymmetric)
grpo_clip = (((w_token < 1-eps) | (w_token > 1+eps)) & mask).sum().item()
gspo_clip = ((s_seq < 1-eps_l) | (s_seq > 1+eps_r)).sum().item()

for i in range(3):
    r = mask[i]
    print(f"resp{i} len={int(lengths[i])}  token-ratio[{w_token[i][r].min():.3f},{w_token[i][r].max():.3f}]  s_i={s_seq[i]:.4f}")
print(f"GRPO clipped {grpo_clip}/{int(mask.sum())} tokens (eps={eps})")
print(f"GSPO clipped {gspo_clip}/3 sequences (eps_l={eps_l}, eps_r={eps_r})")
# Note: s_i here (~1.02-1.03) already exceeds GSPO's tiny eps -> nearly every
# sequence is clipped in practice. The small eps is an intentional tight proximal
# constraint, NOT evidence that GSPO clips less than GRPO.
```

## 6. RLVR — RL from Verifiable Rewards

- Rewards come from **rules / verifiers** (math exact-match, code unit tests), not a learned neural RM.
- Advantages: almost no "neural RM being hacked" (verifier ≈ ground truth); disadvantage: **only applicable to verifiable domains**. This is the reward foundation for o1 / R1-style reasoning RL.

## 6.5 DeepSeek-R1 recipe
Chains the GRPO + RLVR pieces above into a full pipeline. R1 is not "RL all the way through" but **four stages alternating SFT and RL**:

| Stage | Name | What it does | Reward / data |
|---|---|---|---|
| 1 | Cold-start SFT | Fine-tune base on a small set of high-quality long-CoT samples | Supervised data (fixes readability / format / language mixing) |
| 2 | Reasoning RL | GRPO + RLVR to push reasoning on math/code | Rule rewards (answer exact-match + format + language consistency) |
| 3 | Rejection-sampling SFT | Sample heavily from the stage-2 policy, keep the correct ones, then SFT | Self-distilled data (~800k in the paper: reasoning + general mixed) |
| 4 | All-scenario RL | RL again over all prompts to align general preferences | Rule rewards for verifiable domains + helpful/harmless RM for general |

- **R1-Zero**: **pure RL, no SFT** (run GRPO + rule rewards directly from base). Proves reasoning can **emerge spontaneously** from RL (self-reflection / verification), but suffers **poor readability / language mixing** — precisely the motivation for the stage-1 cold-start SFT.
- **R1-Distill**: distill R1's generated reasoning data into smaller dense models (Qwen / Llama 1.5B–70B) via **SFT only** (no RL). Paper finding: **distillation > running RL directly on small models** — small models can't explore enough on their own, so feeding them the big model's reasoning traces wins.
- For the process-reward (PRM) vs outcome-reward (ORM) trade-off and Math-Shepherd-style rollout auto-labeling, see [reward-modeling-eval §2](cheatsheet-reward-modeling-eval-en.html); R1's main line uses a **rule-based ORM** (RLVR) rather than a neural PRM.

## 7. long chain-of-thought & test-time scaling

- RLVR trained on **long CoT** → the model learns to "think longer" (more reasoning tokens), and accuracy improves with **test-time compute** (inference-time scaling).
- Observed phenomena: self-reflection / backtracking / spontaneous "aha moments"; evaluation shifts from "single-pass accuracy" to "accuracy given a compute budget".

## 8. self-rewarding / self-play

- **Self-Rewarding LM**: the model acts as its own judge to generate preference data and iteratively applies preference optimization, reducing dependence on human annotation.
- **SPIN (self-play)**: the model's own previous outputs serve as "negative samples" for adversarial fine-tuning. Risk: self-preference gets amplified.

## 9. Algorithm cheat-sheet

| Scenario | Recommended | Why |
|------|------|------|
| Verifiable domain (math/code), want simplicity, enough compute | **GRPO** | No critic overhead, clean group-relative advantage; DeepSeek's mainstay, best ecosystem |
| Verifiable domain + long CoT + fear entropy collapse | **DAPO** | clip-higher prevents collapse + token-level loss + dynamic sampling; ByteDance open-source |
| Verifiable domain + want a more unbiased estimator | **Dr.GRPO** | Removes std and length-norm biases; better token efficiency, no artificial response lengthening |
| MoE training, need sequence-level stability | **GSPO** | Sequence-level IS ratio + tight proximal constraint; used for Qwen3, robust to routing drift |
| Non-verifiable domain (dialogue/writing quality), noisy rewards | **PPO** | Critic smooths noise through TD learning; GAE provides per-token signal; most stable but most expensive |
| Open domain + need robustness to reward noise | **RLOO** | Leave-one-out baseline is unbiased; simpler than PPO but more stable than GRPO in non-verifiable domains (no std to amplify noise) |
| Resource-constrained (small model / few GPUs) | **Distill SFT** | R1-Distill proved small-model RL loses to consuming large-model traces; SFT is cheap and stable |

> 📝 Core principle: **clean reward signal → critic-free (GRPO family); noisy / sparse reward signal → critic-based (PPO); small model / resource-constrained → distillation.**

---

## Stratified follow-ups

### L1 Fundamentals

<details>
<summary>Q1: What does GRPO save compared to PPO? How exactly is the "group-relative advantage" computed?</summary>

**A:** GRPO drops the **critic (value network)** — from PPO's four models (actor + critic + ref + RM) down to three (actor + ref + RM), saving the memory of one model's parameters and the training cost of the value function. Group-relative advantage: for each prompt, sample $G$ responses to obtain rewards $r_1..r_G$, then use the **within-group mean as a value-baseline substitute + standard deviation scaling**:
$$A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \varepsilon}$$
Intuition: $\mathrm{mean}(r)$ is an MC estimate of "how many points this prompt earns on average" — using it as the baseline factors out problem-difficulty effects; $\mathrm{std}(r)$ scaling keeps advantages from different prompts on the same scale. Why can the critic be dropped? Because rewards in verifiable domains (math exact-match, code unit tests) are clean and stable — no need to learn a value function to estimate "how many points this state will earn in the future."

> **Follow-up:** If rewards are noisy (e.g., dialogue-quality RM scores), can the within-group mean still serve as a baseline?
> Yes but with reduced effectiveness — when noise is high, the MC variance of $\mathrm{mean}(r)$ is large, diluting the within-group comparison signal; PPO's critic smooths noise through TD learning and is more stable. This is precisely why the GRPO mainline chooses verifiable domains (see §6).

</details>

<details>
<summary>Q2: Where does RLVR's reward come from? Why does it mitigate reward hacking? What are its limitations?</summary>

**A:** RLVR's reward comes from **rules / verifiers** rather than a learned neural RM: math uses exact-match (compare the final answer to the ground truth), code uses unit test pass/fail, format uses regex checks. It mitigates reward hacking because: **a rule-based verifier ≈ ground truth** — it does not suffer from systematic biases like "learned that long = good" which plague neural RMs, and is nearly impossible to hack (format gaming and repeated-token exploits for length rewards remain, but are much cleaner than neural RM issues). Limitation: **only applicable to verifiable domains** (math/code/machine-checkable formats) — open-ended domains (writing quality, dialogue helpfulness, safety) have no rule-based verifier and still require neural RMs or LLM-as-judge. The R1 mainline exploits exactly this property: RLVR does the heavy lifting in verifiable domains; a helpful/harmless RM handles general alignment in stage 4.

> **Follow-up:** Is a format reward (e.g., "+0.1 for putting the answer in \boxed{}") RLVR or a neural RM?
> It is RLVR — the format is a hard regex-checkable rule. But in practice format rewards easily become a reward-hack target (the model learns to output \boxed{wrong answer} to farm the format points), so the weight must be tuned carefully.

</details>

### L2 Intermediate

<details>
<summary>Q3: In long-CoT RL, why does "token-level vs. sequence-level" loss matter?</summary>

**A:** **Sequence-level loss** (GRPO's original formulation): $L = \frac{1}{N}\sum_i \frac{1}{|o_i|}\sum_t \ell_{i,t}$ — average within each sequence first, then average across sequences. The problem: each token in a long response contributes a gradient diluted by $|o_i|$ (large $|o_i|$ means tiny per-token gradient), so the early critical steps of a long reasoning chain receive almost no signal. **Token-level loss** (DAPO's fix): $L = \frac{1}{\sum_i |o_i|}\sum_i\sum_t \ell_{i,t}$ — all tokens have equal weight, each token contributes an equal gradient. **In long-CoT RL token-level is required** — otherwise the model is induced to "write fewer tokens for the same reward", compressing reasoning chains. Dr.GRPO notes further: $1/|o_i|$ not only dilutes correct tokens in long responses, but also **systematically favors long wrong responses** (a long wrong response has each wrong token's penalty diluted → the loss looks smaller), a key source of optimization bias.

> **Follow-up:** Won't token-level loss conversely give short correct responses disproportionately large gradients, causing training instability?
> Yes — each token in a short correct response gets a larger gradient; in practice this requires gradient clipping and an appropriate LR schedule to control.

</details>

<details>
<summary>Q4: What bias does std normalization of the advantage introduce in GRPO? How does Dr.GRPO fix it?</summary>

**A:** Std normalization scales the advantages from different prompts to the same range — but it introduces two biases: ① **Difficulty bias**: prompts with small within-group variance (e.g., all-correct / all-wrong, std→0) have their advantages artificially inflated (divided by a tiny number), giving these prompts a louder voice in the batch; ② **Interaction bias with length normalization**: std scaling combined with $1/|o_i|$ length normalization makes the optimization drift toward "longer but worse responses" (a long wrong response undergoes two operations that make its gradient look "nicer"). **Dr.GRPO's fixes**: ① Remove the std division — use just $A_i = r_i - \mathrm{mean}(r)$, keeping the baseline's variance reduction while removing the scaling bias; ② Remove the $1/|o_i|$ length normalization — replace it with constant normalization (e.g., divide by max length, or no normalization), eliminating the systematic preference for long wrong responses. The cost: the absolute magnitude of advantages varies by prompt, requiring a suitable LR schedule. The paper claims this yields better token efficiency and no artificial response-length inflation.

> **Follow-up:** After removing std, how do you handle prompts spanning very different difficulty levels (easy: rewards 0–1, hard: rewards 0–10)?
> Apply global reward normalization (e.g., batch-level z-score) or reward clipping — but this is a separate design choice, orthogonal to GRPO's within-group baseline construction.

</details>

<details>
<summary>Q5: What problem does DAPO's clip-higher solve? What is entropy collapse and why is it harmful?</summary>

**A:** **Entropy collapse**: the policy becomes **excessively deterministic** too early (some token probabilities → 1, others → 0), causing it to stop exploring new reasoning paths — the model gets "locked" into its current policy. **How PPO's symmetric clip causes entropy collapse**: the clip range $[1-\varepsilon, 1+\varepsilon]$ symmetrically constrains the importance ratio $\rho=\pi_\theta/\pi_{old}$. But for **newly learned low-probability reasoning tokens** (e.g., specific token sequences for backtracking or switching approaches), the initial probability is extremely low, so under a positive advantage $\rho$ rises from far below 1 — and the upper bound $1+\varepsilon$ quickly caps the growth of these tokens. Meanwhile, high-probability tokens (connective words like "therefore", "so") have $\rho\approx1$ and are barely affected by the clip. The result: **the upward channel for low-probability innovative tokens is symmetrically capped, while high-probability tokens become further entrenched**. **Clip-higher's solution**: decouple the upper and lower bounds and raise the upper bound (e.g., $\varepsilon_{low}=0.2, \varepsilon_{high}=0.28$), leaving room for low-probability tokens to rise and maintaining exploration.

> **Follow-up:** Besides clip-higher, what other methods prevent entropy collapse?
> Add an entropy bonus (reward $+\lambda H(\pi)$), temperature annealing (high temperature early in training to encourage exploration), EMA reference (slow-anchor against the SFT checkpoint). DAPO chose clip-higher because it is the lightest weight and does not alter the reward signal.

</details>

### L3 Deep Dive

<details>
<summary>Q6: Derive: why can GRPO's advantage be seen as "approximating the value baseline with the within-group mean"? How are bias and variance balanced?</summary>

**A:** Let $V(s) = \mathbb{E}[r|s]$ be the expected reward given the state (prompt). GRPO's within-group mean $\hat\mu=\frac{1}{G}\sum_i r_i$ is an **MC estimate** of $V$ — as $G\to\infty$, $\hat\mu\to V(s)$. The centered advantage $r_i-\hat\mu$ is exactly doing variance reduction with an MC baseline — keeping the unbiased gradient of REINFORCE while reducing variance. **The bias–variance spectrum**: ① When $G$ is small, the MC variance of $\hat\mu$ is high (the baseline itself is noisy), the advantage estimate fluctuates more, but the mean remains **unbiased** (no systematic shift); ② std normalization introduces **mild bias** (dividing by $\hat\sigma$ shifts the expected value of $A_i$), but substantially reduces variance (protecting gradient-scale stability); GRPO chooses this end of the trade-off; ③ Dr.GRPO removes std → returns to the unbiased $r_i-\hat\mu$, with higher variance, requiring larger $G$ or a more conservative LR as compensation; ④ PPO's value baseline $\hat V(s)$ is **biased** (function approximation error), but continually improves its estimate quality through TD learning. Thus in verifiable domains with clean rewards and sufficiently large $G$, critic-free's unbiased baseline suffices; in noisy-reward, long-trajectory open-ended domains, PPO's biased but lower-variance critic is more stable.

> **Follow-up:** Can you take a weighted blend of GRPO's and PPO's baselines?
> In principle yes (an ensemble baseline), but in practice virtually nobody does this — the two baselines require different training paradigms (critic-free vs. critic-based), and the engineering cost of the blend is high for unclear gain.

</details>

<details>
<summary>Q7: Why does dynamic sampling (discarding all-correct / all-wrong groups) improve efficiency? What is its relationship to curriculum / difficulty sampling?</summary>

**A:** In an all-correct or all-wrong group, all samples have the same reward → $A_i\equiv0$ → **zero gradient**, purely consuming compute with no parameter update. Dynamic sampling discarding them not only saves compute but **guarantees every batch carries effective gradient signal**, preventing optimization from being disrupted by zero-gradient batches. Relation to curriculum learning: both filter training samples by difficulty, but in **opposite directions** — curriculum is a preset "easy → hard" ordering of difficulty, while dynamic sampling filters in real time based on the **model's current capability**: all-correct prompts are too easy (the model already knows them), all-wrong prompts are too hard (the model still can't do them), leaving the difficulty band where "the model can just barely do them but isn't certain" — essentially **online difficulty filtering**. DAPO's dynamic sampling discards prompts with group accuracy 0 or 1 and resamples until the batch is filled.

> **Follow-up:** What if most prompts in a batch are filtered out (difficulty converges late in training)?
> Down-sample the frequency rather than fully discarding — lower the sampling rate of all "ineffective prompts" without zeroing them out; or expand the prompt pool with new problems. DAPO takes the former approach.

</details>

<details>
<summary>Q8: The trade-offs between reasoning RL and "pure SFT distillation of long CoT"? What does the R1-Distill conclusion imply?</summary>

**A:** Reasoning RL (GRPO + RLVR): lets the model **explore on its own** how to use more tokens effectively (reflection, re-checking, switching approaches), acquiring long-reasoning behaviors during training. Distillation (SFT from teacher traces): directly imitates reasoning traces that "the teacher already thought through." Their respective pros and cons: ① Reasoning RL has a higher ceiling — it can discover reasoning patterns **not present** in distillation data (R1-Zero's spontaneous reflection/re-checking proves this), but training is unstable and only works in verifiable domains; ② Distillation is cheaper, more stable, and broadly applicable (any domain with teacher traces works), but the ceiling is capped by teacher quality ("the teacher can't teach what it doesn't know"). **R1-Distill's core conclusion**: **distillation > running RL directly on small models** — small models can't explore effectively on their own (in such a vast search space, the small model's initialization isn't good enough and within-group diversity is poor), so directly consuming the large model's reasoning traces wins. This doesn't mean distillation is absolutely better than RL — rather, in the **large-model-does-RL + small-model-distills** combo, the division of labor is clear: the large model explores, the small model learns the exploration results.

> **Follow-up:** What does this conclusion imply for resource-constrained academic research?
> If you have only modest compute, prioritize the distillation route (use a strong open-source reasoning model to produce traces → SFT into a small model), rather than running GRPO from scratch on the small model — the cost-effectiveness ratio is far higher.

</details>

<details>
<summary>Q9: Under what conditions does critic-free (GRPO / RLOO) actually underperform value-based PPO?</summary>

**A:** Critic-free methods assume "**clean reward signal + effective within-group comparison**." Four failure modes: ① **Noisy rewards** (e.g., dialogue-quality RM scores with high fluctuation): lacking a value function for temporal smoothing, within-group z-score amplifies noise, and the gradient direction becomes unreliable; ② **Poor within-group diversity**: when the sampling temperature is too low or the model is overfitted, the G responses are nearly identical → std→0 → the advantage estimate collapses (division by zero) — PPO's critic is unaffected by sampling diversity; ③ **Sparse rewards + long trajectories**: only the final answer's correctness (0/1) is given, with no signal for the 1000 intermediate tokens — GRPO can only rely on the final reward's within-group contrast, while PPO's GAE back-propagates future rewards step by step via TD, providing a smoother per-token signal; ④ **Non-stationary reward distributions**: when the neural RM itself is changing during training, PPO's critic can adapt to reward-distribution drift, while critic-free's within-group baseline only reflects the current batch. Mnemonic: **verifiable domains = critic-free is optimal; non-verifiable domains / long trajectories / noisy environments = PPO is more stable**.

> **Follow-up:** Could you introduce a lightweight critic as a hybrid in "verifiable domain + sparse reward" settings?
> This is an open research question — a lightweight critic (e.g., initialized from a PRM, doing only outcome-level value estimation) could provide a cross-prompt global baseline for each group, improving sparse-reward settings while retaining critic-free simplicity. There is no standard approach yet.

</details>


## Extended L3

<details>
<summary>Q: GRPO retains the KL penalty $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})$, but in long-CoT training the model needs to explore long reasoning paths far beyond the reference distribution. How should we understand this tension? What happens if KL is removed?</summary>

**A:** The role of the KL penalty is **policy anchoring** — preventing the policy from drifting under reward hacking (collapsing onto some reward shortcut). But in long-CoT settings, precisely what the model needs to learn is the long-chain reflection behavior that the reference model **cannot do**, so the KL is inherently penalizing "novel reasoning paths." Practical trade-offs: $\beta$ too large → model fails to learn long CoT, reasoning capacity is capped by the reference; $\beta$ too small → policy may degrade into reward hacking (e.g., repeating tokens to fool the verifier). DAPO's original recipe actually **removes KL** and instead relies on clip-higher + dynamic sampling to prevent collapse; GRPO retains KL but typically sets it to a low value. At its core this is a tightrope walk between **"not collapsing"** and **"being able to explore"**.
  **Follow-up:** If you want to remove KL to open up exploration while still preventing policy collapse, what alternative anchoring mechanisms are feasible beyond clip-higher? (e.g., EMA reference, regularization toward the SFT checkpoint, etc.)

</details>

---

<details>
<summary>Q: From a variance reduction perspective, what are the theoretical pros and cons of GRPO's within-group standardization baseline vs. RLOO's leave-one-out baseline?</summary>

**A:** Both methods are variance reduction variants of REINFORCE; the difference lies in baseline construction. GRPO uses $\mathrm{mean}(r)$ and divides by $\mathrm{std}(r)$ (i.e., a z-score); RLOO gives sample $i$ a baseline of $\frac{1}{G-1}\sum_{j\neq i}r_j$. RLOO's leave-one-out baseline is unbiased (because $r_i$ is excluded from its own baseline), whereas GRPO's $\mathrm{mean}(r)$ includes $r_i$ itself, introducing a mild self-correlation bias (negligible when $G$ is large). However, GRPO's **std normalization** simultaneously applies **variance scaling**, making it more robust when reward magnitudes are uncertain — at the cost of the problem-difficulty bias that Dr.GRPO identifies. RLOO does not apply std normalization: it is more sensitive to reward scale but more unbiased. The choice depends on the stability of the task's reward distribution.
  **Follow-up:** What would happen if you combined RLOO's leave-one-out baseline with GRPO's std normalization? Are there known problems with this hybrid?

</details>

---

<details>
<summary>Q: DAPO's token-level loss divides the sequence gradient by the total number of tokens $T$, i.e., $L_{\text{token}}=\frac{1}{T}\sum_t \ell_t$. Does this in turn introduce an implicit bias toward "short correct responses"?</summary>

**A:** There is some effect, but the direction is more complex than intuition suggests. Token-level loss ensures every token contributes an equally weighted gradient, which does mean **each token in a short correct response receives a larger gradient** (total gradient is divided among $T$ tokens; smaller $T$ means larger per-token gradient). But the key factor is the **sign** of the gradient: correct responses receive positive reinforcement, wrong responses receive negative penalization. Token-level loss therefore makes **short wrong responses** receive more concentrated, stronger penalty — which is not necessarily bad for training efficiency. The real risk arises when **rewards are binary 0/1**: a long correct response and a short correct response receive the same total reward, but under token-level loss the per-token reinforcement signal for the long response is weaker, which may gradually push the model to compress correct reasoning chains.
  **Follow-up:** If DAPO token-level loss is combined with outcome reward (only the final answer's correctness), how do "compress reasoning chain length" and "compress to correct answer" compete with each other?

</details>

---

<details>
<summary>Q: RLVR currently only works in verifiable domains (math/code). Can a process reward model (PRM) be used as a "soft verifier" and integrated into the GRPO/DAPO framework? What are the technical obstacles?</summary>

**A:** The idea is feasible in principle: use the PRM to score each step of the CoT, aggregate step-level scores into a sequence-level reward, and feed this into GRPO's within-group advantage computation. But there are three obstacles: ① **PRM annotation and training** — step-level human annotations or automated labels (e.g., Monte Carlo rollout estimation) are required, which is expensive and of limited accuracy; ② **Reward alignment** — the PRM scores "reasoning step quality," which may be inconsistent with final answer correctness (good steps but wrong answer vs. rough steps but correct answer), producing conflicting RL signals; ③ **Temporal credit assignment** — when aggregating step-level scores into a sequence reward, the choice of weighting scheme (mean? final step? worst step?) directly affects learning dynamics. A simple mean blurs the contribution of critical steps; using only the final step degenerates into outcome reward.
  **Follow-up:** Within the GRPO framework, is it possible to use different reward aggregation strategies (adaptive weighting) for different samples in the group, rather than a uniform scheme?

</details>

---

<details>
<summary>Q: In reasoning RL, 0/1 binary rewards are common (correct answer = 1, wrong = 0). When prompt difficulty varies widely, a group of samples may be all correct or all wrong. Beyond dynamic sampling (discarding such prompts), what methods can extract useful training signal from an "all-wrong group"?</summary>

**A:** The core problem with an all-wrong group is that all rewards are identical → advantage is always zero → zero gradient. Several approaches: ① **Introduce process rewards** — even if all final answers are wrong, the quality of intermediate reasoning steps may differ; use a PRM or auxiliary signals such as reasoning length and format compliance to create within-group variation; ② **Mixed reward design** — layer format rewards, reasoning completeness rewards, and other soft signals on top of the outcome reward so that even "all-wrong" groups can still distinguish better from worse responses; ③ **Cross-prompt baseline** — rather than restricting to within-group comparison, use a batch-level or moving-average global baseline to provide gradient direction even for all-wrong groups; ④ **Difficulty bucketing + resampling** — mark all-wrong prompts as "too hard," downsample their frequency without fully discarding them, avoiding a training-set bias toward easy problems. Each approach has different costs: process rewards require extra annotation or models; cross-prompt baselines may introduce high variance.
  **Follow-up:** How would a cross-prompt baseline (e.g., using an EMA global mean as the baseline) be implemented concretely within GRPO? How should the weight be tuned when mixing with the within-group baseline?

</details>

---

<details>
<summary>Q: How does the choice of group size $G$ in GRPO theoretically and practically affect training? What does GRPO degenerate into at $G=1$ and $G\to\infty$?</summary>

**A:** At $G=1$ there is only one sample, $\mathrm{mean}(r)=r_1$, and the advantage is always zero — no gradient at all; GRPO is completely ineffective. At $G=2$ it degenerates into pairwise comparison, essentially an online version of pairwise preference learning. As $G\to\infty$ the within-group mean and standard deviation converge to the **population expectation and standard deviation**, so the baseline approximates a Monte Carlo estimate of the global value function; in theory GRPO then approaches REINFORCE with a global baseline. Practical trade-offs: $G$ too small → high baseline variance, noisy advantage estimates; $G$ too large → high sampling cost per prompt (inference cost grows linearly) and diversity may actually decrease (many similar responses in repeated sampling). DeepSeek's practice uses a moderate value of $G$. There is also coupling between $G$ and the clip range and learning rate — with larger $G$ the advantage estimate is more accurate, so more aggressive update steps are tolerable.
  **Follow-up:** Is there an adaptive-$G$ strategy — sampling more from "hard" prompts and fewer from "easy" ones? How would this coordinate with dynamic sampling?

</details>

---

<details>
<summary>Q: In the self-rewarding paradigm the model generates preference data using its own judgments and trains iteratively. From the perspective of online learning theory, under what conditions does this self-play converge, and under what conditions does it mode-collapse?</summary>

**A:** The core condition for convergence is that **the reward signal must continuously provide effective discrimination** — the model must be able to distinguish the quality of its own outputs. When model capability is far below task difficulty, self-judgment is noisy but not systematically biased; training may be slow but won't collapse. Typical triggers for mode collapse: ① **Amplified anchoring effect** — the model prefers responses that resemble its own style; a positive feedback loop continuously reduces output diversity until it collapses to a narrow "self-preferred" mode; ② **Judgment saturation** — when model outputs become too similar in quality, the reward signal degrades to noise and training loses direction; ③ **Reward hacking its own judge** — the model learns to "convince" its own judge rather than genuinely improving, analogous to Goodhart's Law applied to itself. Mitigations include: retaining an SFT anchor, periodically introducing external verification signals, and limiting the number of iterations.
  **Follow-up:** If a fixed external verifier (e.g., code unit tests) is introduced into the self-rewarding loop as an "anchor" to calibrate self-scoring, to what extent can it prevent mode collapse?

</details>

---

<details>
<summary>Q: DAPO's clip-higher raises the upper clip bound to give low-probability tokens room to increase. From an information geometry perspective, why does standard PPO's symmetric clip systematically suppress valuable low-probability reasoning tokens in long CoT?</summary>

**A:** Standard PPO's symmetric clip $[1-\epsilon, 1+\epsilon]$ acts on the importance ratio $\rho=\pi_\theta/\pi_{\theta_{old}}$. The key insight: in long CoT, already high-probability tokens (e.g., common reasoning connectives) have $\rho$ close to 1, so the symmetric clip barely affects them; but **newly learned low-probability reasoning patterns** (e.g., specific token sequences for backtracking or self-correction) start at very low probability, and when the advantage is positive $A>0$, $\rho$ rises from below 1 — and hits the upper bound $1+\epsilon$ quickly, hard-capping the growth of these tokens. Meanwhile, when $A<0$, the lower bound $1-\epsilon$ equally limits the decline of high-probability tokens, but high-probability tokens have ample room to decrease anyway. Therefore the symmetric clip creates an **asymmetric learning dynamic** in information-geometric terms: the "downward channel" for high-probability tokens is wider than the "upward channel" for low-probability tokens. Clip-higher breaks this asymmetry by raising the upper bound.
  **Follow-up:** Apart from decoupling the clip bounds, is it possible to address this problem more elegantly from a trust-region perspective (e.g., using a KL constraint instead of a hard clip)? What is the computational cost of doing so in the long-CoT setting?

</details>

<details>
<summary>Q: Walk through DeepSeek-R1's full training pipeline (four stages). Why is each stage where it is? Could the order be swapped?</summary>

**A:** R1's four-stage ordering is carefully designed, not an arbitrary stack:

1. **Cold-start SFT** (→ readability foundation): Running RL directly from a base model produces R1-Zero's problems — long-CoT direction is correct but answers are unreadable and mix Chinese with English. First, fine-tune on a small set of high-quality long-CoT samples to lay a "format / language" foundation, so subsequent RL starts from a policy that "speaks properly."

2. **Reasoning RL** (→ core reasoning capability): Without adding dialogue preferences, use GRPO + RLVR (rule rewards) to focus on pulling up reasoning ability. Why before preference RL? If you do preference alignment first and reasoning RL later, preference RL may suppress exploration (KL anchors to a "safe but mediocre" policy), and reasoning RL needs a large exploration space.

3. **Rejection-sampling SFT** (→ knowledge distillation + capability consolidation): Heavily sample from the stage-2 policy, filter for reasoning-correct + format-clean traces, then SFT on them (the R1 paper uses ~800k SFT samples, a mixture of reasoning and general). This step: ① "bakes" the good reasoning patterns discovered by RL back into the SFT distribution, providing a stable starting point for the next stage's general RL; ② produces a large quantity of high-quality reasoning data for distillation (R1-Distill uses the data produced by this stage, post-processed for SFT).

4. **All-scenario RL** (→ general alignment): Finally run RL over all prompts (reasoning + general) — rule rewards for verifiable domains, helpful/harmless RM for general domains. Placed last so that preference alignment is a light touch-up on top of "reasoning is already good enough," without damaging reasoning capability.

**Could the order be swapped?** The alternation of stages 2 and 3 (SFT→RL→SFT→RL) is the core innovation; the R1 paper describes this multi-stage design with two rounds of alternation. Stages 1 and 2 cannot be swapped (must lay the readability foundation before RL, otherwise R1-Zero's language-mixing problem recurs). Stage 4 must be last (preference alignment suppresses exploration; placing it earlier would lock down reasoning RL's exploration space).

> **Follow-up:** What if you kept only one alternation (dropped stages 3–4)?
> The R1 paper doesn't directly report this ablation, but looking at R1-Zero (RL only, no SFT): dropping stage 1 severely damages readability; dropping stages 3–4 leaves strong reasoning but poor dialogue experience — essentially R1-Zero + cold-start SFT, which is usable but falls short on practical utility.

</details>

## §A Key Papers Timeline

- **2017-07 · PPO** — Schulman et al., arXiv preprint. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) — Clipped surrogate objective + GAE advantage estimation; establishes the LLM-RL baseline (actor + critic + ref + RM).
- **2024-01 · Self-Rewarding LM** — Yuan et al., Preprint (Meta). [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) — Model acts as its own judge via LLM-as-a-Judge to generate preference data and iterate; risk: self-preference amplification.
- **2024-01 · SPIN** — Chen et al., ICML 2024. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) — Self-play fine-tuning using the model's own older outputs as negatives.
- **2024-02 · GRPO / DeepSeekMath** — Shao et al., Preprint. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — Removes the critic; group-relative reward (z-score) replaces the value baseline; keeps KL to ref; core DeepSeek algorithm.
- **2024-02 · RLOO** — Ahmadian et al., ACL 2024. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740) — Critic-free; baseline for sample i = mean reward of the other G-1 samples (leave-one-out); pure REINFORCE, no clip; competitive with PPO on RLHF.
- **2025-01 · DeepSeek-R1 / RLVR** — Guo et al., Nature 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — Rule/verifier rewards (math exact-match, code unit tests) replace the neural RM, nearly eliminating neural-RM hacking (verifier hacking / format gaming remain); GRPO + long-CoT RL induces self-reflection; opens inference-time scaling.
- **2025-03 · DAPO** — Yu et al., Preprint (ByteDance Seed / Tsinghua AIR). [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) — Four long-CoT-RL fixes: Clip-Higher (anti entropy-collapse), Dynamic Sampling, token-level loss, overlong reward shaping.
- **2025-03 · Dr.GRPO** — Liu et al., Preprint. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) — Fixes two GRPO biases (std normalization, 1/length normalization); removing both yields an unbiased estimator with better token efficiency.
- **2025-06 · CISPO** — MiniMax team, Preprint (MiniMax-M1 tech report). [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) — Clips the scalar IS weight itself rather than the probability ratio, keeping every token's gradient signal; the paper reports ~2× training speedup over DAPO on Qwen2.5-32B.
- **2025-07 · GSPO** — Zheng et al., Preprint (Alibaba Qwen). [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) — Lifts IS correction from token-level to sequence-level (length-normalized geometric mean), mitigating GRPO's collapse when training large-scale MoE models; used for Qwen3 training.
