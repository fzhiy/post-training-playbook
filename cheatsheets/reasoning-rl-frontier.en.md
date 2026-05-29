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
- Objective is the same clipped surrogate as PPO, but the advantage is $A_i$, with **no critic, no GAE**; KL penalty against the reference is retained.
- Benefits: saves one value model and avoids training a value function; especially stable for **verifiable rewards** (math / code). Used by the DeepSeek family.

## 3. RLOO — REINFORCE leave-one-out

- Also critic-free: the baseline for sample $i$ = **the mean reward of the other $G-1$ samples**, REINFORCE-style gradient.
- Simpler than PPO (no clip / critic), competitive with PPO on RLHF. Difference from GRPO lies in baseline construction (leave-one-out vs. within-group standardization) and whether clipping is applied.

## 4. DAPO — keeping GRPO stable under long-CoT / Decoupled-clip & Dynamic-sAmpling PO

ByteDance 2025 open-source recipe — four modifications targeting long-chain reasoning RL:
1. **Clip-Higher**: **decouple** the upper and lower clip bounds $\epsilon$ and raise the upper bound → gives low-probability tokens room to rise, **preventing entropy collapse** (policy becoming deterministic too early and ceasing to explore).
2. **Dynamic Sampling**: discard prompts where the entire group is correct or entirely wrong (within-group advantage is always 0, yielding zero gradient), ensuring every batch contributes useful signal.
3. **Token-level loss**: average by **token** rather than by **sequence**, preventing the gradient of long responses from being diluted (critical for long CoT).
4. **Overlong reward shaping**: soft penalty for excessively long responses, stabilizing training.

## 5. Dr.GRPO — fixing GRPO's optimization bias / GRPO Done Right

- Identifies two **biases** in GRPO: **std normalization** in the advantage (amplifies imbalance across problem difficulty) + **1/response-length** normalization in the loss (biases toward "longer wrong responses").
- Fix: **remove the std division + remove length normalization** (replace with a constant) → a more **unbiased** estimate; same performance with fewer tokens and no artificially inflated response length.

## 6. RLVR — RL from Verifiable Rewards

- Rewards come from **rules / verifiers** (math exact-match, code unit tests), not a learned neural RM.
- Advantages: almost no "neural RM being hacked" (verifier ≈ ground truth); disadvantage: **only applicable to verifiable domains**. This is the reward foundation for o1 / R1-style reasoning RL.

## 7. long chain-of-thought & test-time scaling

- RLVR trained on **long CoT** → the model learns to "think longer" (more reasoning tokens), and accuracy improves with **test-time compute** (inference-time scaling).
- Observed phenomena: self-reflection / backtracking / spontaneous "aha moments"; evaluation shifts from "single-pass accuracy" to "accuracy given a compute budget".

## 8. self-rewarding / self-play

- **Self-Rewarding LM**: the model acts as its own judge to generate preference data and iteratively applies preference optimization, reducing dependence on human annotation.
- **SPIN (self-play)**: the model's own previous outputs serve as "negative samples" for adversarial fine-tuning. Risk: self-preference gets amplified.

---

## Stratified follow-ups

### L1 Fundamentals
- What does GRPO save compared to PPO? How exactly is the "group-relative advantage" computed?
- Where does RLVR's reward come from? Why does it mitigate reward hacking? What are its limitations?

### L2 Advanced
- In long-CoT RL, why does "token-level vs. sequence-level" loss matter? (Gradient dilution for long responses.)
- What bias does std normalization of the advantage introduce in GRPO? How does Dr.GRPO fix it?
- What problem does DAPO's clip-higher solve? What is entropy collapse and why is it harmful?

### L3 Deep Dive
- Derive: why can GRPO's advantage be seen as "approximating the value baseline with the within-group mean"? How are bias and variance balanced?
- Why does dynamic sampling (discarding all-correct / all-wrong groups) improve efficiency? What is its relationship to curriculum / difficulty sampling?
- What does test-time scaling imply for the paradigm of "capabilities are acquired at training time"? What are the trade-offs between reasoning RL and "pure SFT distillation of long CoT"?
- Under what conditions does critic-free (GRPO / RLOO) actually underperform value-based PPO?


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

## Timeline

- **2017-07** — PPO (Schulman et al., arXiv:1707.06347): Introduces the clipped surrogate objective and GAE advantage estimation; establishes the four-model framework (actor + critic + ref + RM) as the baseline for LLM RL.
- **2024-01** — Self-Rewarding LM (Yuan et al., arXiv:2401.10020): The model acts as its own judge via LLM-as-a-Judge prompting to generate preference data and iterate; reduces reliance on human annotation but risks amplifying self-preference.
- **2024-01** — SPIN (Chen et al., arXiv:2401.01335): Self-play fine-tuning using the model's own older outputs as negatives for adversarial optimization; accepted at ICML 2024.
- **2024-02** — GRPO / DeepSeekMath (Shao et al., arXiv:2402.03300): Removes the critic; samples a group of G responses per prompt and replaces the value baseline with group-relative reward (z-score normalization); retains KL penalty against ref; core algorithm of the DeepSeek line.
- **2024-02** — RLOO (Ahmadian et al., arXiv:2402.14740): Also critic-free; baseline for sample i = mean reward of the other G-1 samples (leave-one-out, unbiased); pure REINFORCE gradient with no clip; competitive with PPO on RLHF tasks.
- **2025-01** — DeepSeek-R1 / RLVR (Guo et al., arXiv:2501.12948): Replaces the neural RM with rule-based verifiers (math exact-match, code unit tests), nearly eliminating reward hacking; GRPO + long-CoT RL induces emergent self-reflection and backtracking; opens the inference-time scaling paradigm; published in Nature 2025.
- **2025-03** — DAPO (Yu et al., arXiv:2503.14476): Four fixes for long-CoT RL — Clip-Higher (decouple upper/lower clip bounds to prevent entropy collapse), Dynamic Sampling (discard all-correct/all-wrong groups), token-level loss (prevent gradient dilution for long answers), and overlong reward shaping (soft penalty for over-length responses); fully open-sourced by ByteDance Seed and Tsinghua AIR.
- **2025-03** — Dr.GRPO (Liu et al., arXiv:2503.20783): Identifies two biases in GRPO — std normalization amplifies question-difficulty imbalance, and 1/length normalization favors longer wrong answers; removing both yields an unbiased estimator with better token efficiency.

## §A Key References

1. **PPO** — Schulman et al., 2017, arXiv preprint. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
2. **GRPO / DeepSeekMath** — Shao et al., 2024, Preprint. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
3. **RLOO** — Ahmadian et al., 2024, ACL 2024. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)
4. **Self-Rewarding Language Models** — Yuan et al., 2024, Preprint (Meta). [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
5. **SPIN** — Chen et al., 2024, ICML 2024. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335)
6. **DeepSeek-R1** — Guo et al., 2025, Nature 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
7. **DAPO** — Yu et al., 2025, Preprint (ByteDance Seed / Tsinghua AIR). [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
8. **Dr.GRPO** — Liu et al., 2025, Preprint. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
