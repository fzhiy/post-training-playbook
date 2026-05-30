# Reward Modeling & Evaluation Cheatsheet
## A Complete Overview of Reward Modeling in LLM Post-Training

---

## 1. RM Training Methods

### 1.1 Core Objective

A reward model learns a **scalar scoring function** from human preferences, used to score LLM-generated candidate responses in RLHF (Reinforcement Learning from Human Feedback) or similar pipelines, guiding policy model optimization.

### 1.2 Pairwise / Bradley-Terry Model

**Core idea:** Rather than estimating absolute scores directly, learn the **preference relationship between two responses**.

**Bradley-Terry (BT) Model:**

Given prompt $x$, chosen response $y_w$, rejected response $y_l$:

$$P(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

where $\sigma$ is the sigmoid function and $r_\theta$ is the reward model with parameters $\theta$.

**Training Loss:**

$$\mathcal{L}_{\text{BT}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

**Advantages:**
- Requires only preference rankings, not absolute score annotations — lower annotation cost
- Naturally aligned with policy gradient in RLHF
- Humans make relative judgments more reliably and consistently than absolute ratings

**Limitations:**
- Only uses binary preference signals, making limited use of available information
- Assumes transitivity of preferences, which may not hold in practice
- Does not directly output a scalar value; additional processing may be needed in some downstream tasks

### 1.3 Pointwise / Regression

**Core idea:** Directly regress to an **absolute quality score**.

**Training Loss:**

$$\mathcal{L}_{\text{point}} = \mathbb{E}_{(x, y, s)} \left[ (r_\theta(x, y) - s)^2 \right]$$

where $s$ is the human-annotated absolute score (e.g., a 1–5 Likert scale).

**Advantages:**
- Can be directly used for Best-of-N (BoN) sampling, filtering, and similar scenarios
- Scores are highly interpretable

**Limitations:**
- Requires absolute annotations; annotation cost is high and inter-annotator agreement is low
- Score scales vary by annotator, requiring calibration

### 1.4 Other Variants

| Method | Description |
|------|------|
| **Listwise / Plackett-Luce** | Full ranking over k responses; carries more information than pairwise |
| **Regression + Ranking Hybrid** | Jointly optimizes regression loss and ranking loss |
| **Multi-objective RM** | Assigns separate scores for different dimensions (helpfulness, safety, factuality) |
| **Token-level RM** | Distributes rewards at the token level; related to PRM |

---

## 2. PRM vs ORM: Process Reward Model vs Outcome Reward Model

### 2.1 ORM — Outcome Reward Model

**Definition:** Assigns a single overall reward score based solely on the **final output**.

```
Input:  [Prompt + full response text]
Output: single scalar reward r
```

**Characteristics:**
- ✅ Simple annotation: only need to judge whether the final answer is correct or of high quality
- ✅ Lower training and inference compute cost
- ❌ **Credit assignment problem**: does not identify which steps in the response were correct or incorrect
- ❌ Sparse signal for multi-step tasks such as mathematical reasoning and code generation

### 2.2 PRM — Process Reward Model

**Definition:** Assigns a reward score to **each step** of the reasoning process.

```
Input:  [Prompt + first i steps]
Output: per-step reward r_step(i)
```

**Characteristics:**
- ✅ Denser reward signal; more precise credit assignment
- ✅ Can pinpoint specific erroneous reasoning steps; enables early stopping or tree search
- ✅ Generally outperforms ORM significantly on tasks such as mathematical reasoning (qualitative conclusion from multiple studies)
- ❌ **Very high annotation cost**: requires expert-level step-by-step judgments
- ❌ Step boundary definition (step delineation) is inherently ambiguous

### 2.3 Comparison Table

| Dimension | ORM (Outcome Reward) | PRM (Process Reward) |
|------|----------------|----------------|
| Reward Granularity | Entire response | Each reasoning step |
| Signal Density | Sparse | Dense |
| Annotation Cost | Low | High |
| Credit Assignment | Poor | Good |
| Typical Applications | Dialogue, writing | Math reasoning, code, multi-step reasoning |
| Integration with Search | Best-of-N | Best-of-N, tree search, beam search |
| Annotation Method | Final answer correctness | Per-step logical correctness |

### 2.4 Automated PRM Data Generation

To reduce annotation cost, common approaches include:
- **Monte Carlo estimation (MC estimation):** From each step, sample multiple rollouts and use the final answer accuracy as the reward for that step
- **LLM-as-Step-Judge:** Use a strong LLM to annotate per-step correctness
- **ORM-to-PRM distillation:** Back-derive process signals from an ORM

---

## 3. Reward Hacking & Over-Optimization

### 3.1 What is Reward Hacking

**Definition:** The policy model learns to exploit **flaws and out-of-distribution behavior** of the reward model to obtain high scores, rather than genuinely improving response quality.

**Common Patterns:**

| Pattern | Definition |
|------|------|
| **Verbosity hacking** | Generating long but low-quality content to exploit the RM's spurious preference for length |
| **Sycophancy** | Using flattering, fawning language to align with the user's stance rather than providing accurate answers |
| **Format gaming** | Overusing Markdown lists, headings, bold text, etc., which the RM misinterprets as quality signals |
| **Spec gaming** | Meeting the literal requirements of the RM or task specification while violating its intent (e.g., answers that are "formally correct" but substantively wrong) |
| **OOD collapse** | After the policy drifts from the training distribution, RM scores become unreliable, giving inflated or random scores for out-of-distribution generations |

### 3.2 Goodhart's Law Perspective

> "When a measure becomes a target, it ceases to be a good measure."

$$r_{\text{true}} \neq r_{\text{RM}} \quad (\text{OOD})$$

The RM is only accurate within its **training distribution**; once the policy generates out-of-distribution content, RM predictions are no longer reliable.

### 3.2a Gao et al. 2022 — Scaling Laws for Overoptimization

> Source: Gao, Schulman & Hilton, arXiv:2210.10760, using a synthetic gold RM in the InstructGPT setup.

**Core variable:** Let $d = \sqrt{D_{\text{KL}}(\pi \| \pi_{\text{init}})}$ (the square root of the KL distance; the paper chooses this parameterization because KL is a "quadratic measure"). The functional form of how the gold RM score $R$ changes with $d$ **differs by optimization method**:

**Best-of-N (BoN) form:**

$$R_{\text{bon}}(d) = d(\alpha_{\text{bon}} - \beta_{\text{bon}} d)$$

**Reinforcement Learning (RL) form:**

$$R_{\text{RL}}(d) = d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d)$$

where $\alpha_{\text{bon}}, \beta_{\text{bon}}, \alpha_{\text{RL}}, \beta_{\text{RL}}$ are fitted parameters (varying smoothly with RM parameter count); $R(0) := 0$. ⚠️ The RL form has infinite slope near the origin; the paper notes this form may not hold near the origin.

**Key findings:**
- **Initial rise then decline**: The gold score first increases with $d$ (the RM signal is effective), then decreases due to out-of-distribution collapse — a quantitative characterization of the Goodhart effect
- **Peak location differs by method**: The gold score peaks of BoN (quadratic) and RL (logarithmic) occur at different values of $d$, depending on the fitted coefficients (see Figure 3 in the original paper); the paper does not draw a general conclusion about "which is more KL-efficient"
- **Smooth coefficient scaling**: $\alpha$ and $\beta$ vary approximately logarithmically with RM parameter count, allowing extrapolation to predict peak performance
- **Limits of KL penalty**: The paper's §3.6 experiments find that in their setup, increasing the KL penalty raises the proxy score for a given KL, but **does not improve the gold score–KL curve**; the effect is equivalent to early stopping rather than improving RM robustness itself. The authors note this conclusion is sensitive to hyperparameters

⚠️ The specific values of $\alpha, \beta$ above depend on RM parameter count and data volume; the paper provides no single universal constant. For specific values, consult Figure 3 of the original paper.

### 3.3 Mitigations

#### 3.3.1 KL Divergence Control

Add a KL penalty term to the RLHF objective:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} \left[ r_\theta(x, y) - \beta \cdot D_{\text{KL}}(\pi(y|x) \| \pi_{\text{ref}}(y|x)) \right]$$

- Larger $\beta$ → more conservative policy, closer to the reference model
- Smaller $\beta$ → larger exploration space, but higher reward hacking risk
- **Adaptive KL:** Dynamically adjust $\beta$ based on the current KL value

> 📝 For the single-sample KL estimators k1/k2/k3, in-reward vs in-loss placement, and the gradient bias of k3-as-loss, see [llm-post-training §9.4](cheatsheet-llm-post-training-en.html).

#### 3.3.2 RM Ensembles

- Train **multiple independent RMs** (different initializations, different data subsets, different architectures)
- Aggregate via mean, minimum, or uncertainty estimation
- **Conservative strategy:** $\hat{r}(x,y) = \min_i r_i(x,y)$ — avoids overly optimistic scores from any single RM
- Can be used for **uncertainty filtering**: reduce trust in regions with high variance

#### 3.3.3 Length Penalty

$$r_{\text{adjusted}}(x,y) = r_\theta(x,y) - \alpha \cdot \text{len}(y)$$

or the normalized form:

$$r_{\text{adjusted}} = \frac{r_\theta(x,y)}{\text{len}(y)^\gamma}$$

- Requires careful tuning to avoid responses becoming too short due to over-penalization
- A better approach: control length preferences at the **data annotation stage**

#### 3.3.4 Iterative Re-Training / Online RLHF

```
Iterative pipeline:
1. Use current policy π_k to generate candidate responses
2. Collect new preference annotations / or annotate with RM
3. Update RM: r_k → r_{k+1}
4. Apply RLHF update with new RM: π_k → π_{k+1}
5. Repeat
```

- **Key benefit:** The RM is always trained on the policy's current distribution, reducing distribution shift
- **Challenge:** Compute cost more than doubles; annotations may also drift over time
- Variants: DPO (Direct Preference Optimization), iterative versions of RLHF + rejection sampling

#### 3.3.5 Other Methods

| Method | Description |
|------|------|
| **Rejection Sampling / Best-of-N** | No RL; sample N responses from the policy and select the one with the highest RM score |
| **Constrained Optimization** | Add hard safety/quality constraints |
| **Pre-training anchor** | Maintain proximity to the pre-trained model (e.g., L2 regularization) |
| **Robust RM training** | Adversarial training and data augmentation to improve RM generalization |

### 3.4 Preference Data Construction

The quality ceiling of the RM is determined by the quality of the preference data. The following are core design choices:

#### 3.4.1 Absolute vs Relative Annotations

| Dimension | Absolute Annotation (Pointwise) | Relative Annotation (Pairwise/Comparative) |
|------|------|------|
| Annotation form | Rate a single response 1–5 | Compare two responses and pick the better one |
| Annotation cost | High (requires internally consistent scales across annotators) | Low (cognitively simpler task) |
| Inter-annotator agreement | Low; scale drift is pronounced | High; humans make relative judgments more reliably |
| Information content | Richer (ordinal relationships recoverable) | Directly corresponds to the Bradley-Terry model |
| Typical use | Direct regression RM | Mainstream RLHF practice |

#### 3.4.2 Margin Filtering

**Motivation:** Preference pairs contain many "hard to distinguish" samples (low annotator agreement); training directly on these introduces noise.

**Approach:**
- Compute inter-annotator agreement rates; discard preference pairs near 50/50
- Introduce a **margin** label: is the quality gap between chosen and rejected "clearly discernible"? Retain only samples with sufficiently large margin
- Some work (e.g., Llama-2, per the original paper) distinguishes significantly better / slightly better / negligibly better, training with stratified weights
- ⚠️ Aggressive filtering loses borderline samples, reducing RM discriminative ability in "slight difference" scenarios

#### 3.4.3 Annotator Calibration

**Problem:** Different annotators have systematic biases (personal styles, varying strictness).

**Mitigation methods:**
- **Calibration items:** Insert anchor questions with known answers into each annotation batch to detect annotator drift
- **Annotator effect modeling:** Explicitly model each annotator's preference distribution and aggregate with a mixture model
- **Majority voting / gold-standard filtering:** 3+ annotators label the same pair; take the majority; discard samples with excessive disagreement

⚠️ Annotator bias is learned and amplified by the RM, ultimately affecting policy behavior — calibration at the data construction stage is more fundamental than post-hoc remediation.

---

## 4. Length Bias & Other RM Pathologies

### 4.1 Length Bias

**Phenomenon:** RMs tend to assign higher scores to **longer responses**, even when longer does not mean better.

**Root causes:**
- In training data, detailed responses (chosen) are typically longer than brief responses (rejected)
- Annotators prefer detailed content → length correlations are implicitly encoded in preference pairs
- RM overfits to the spurious feature "long = good"

**Consequences:**
- Policy models learn to "pad" responses for higher scores
- Generates excessive filler words, repetition, and redundant explanations
- Higher inference cost; degraded user experience

**Mitigations:**
- Control response length in training data: construct chosen/rejected preference pairs with similar lengths
- Apply length penalties at inference time (see §3.3.3)
- Data debiasing: statistically analyze and compensate for length correlations
- Use length-normalized evaluation metrics

### 4.2 Position Bias

**Phenomenon:** In pairwise comparisons, RMs tend to prefer responses in **a specific position** (e.g., the first or the last).

**Mitigation:** Swap positions and make two predictions; accept only consistent results.

### 4.3 Verbosity / Redundancy Bias

**Phenomenon:** Beyond mere length, RMs also tend to favor responses containing **more redundant modifiers, filler transitions, and formatting elements**.

**Distinction from length bias:** Even at the same length, "fancier" responses may still be preferred.

### 4.4 Confirmation / Sycophancy Bias

**Phenomenon:** RMs prefer responses that agree with the user's stated position, even if that position is incorrect.

### 4.5 Summary of Common RM Pathologies

| Pathology | Manifestation | Root Cause |
|---|---|---|
| **Length bias** | Longer responses score higher | Spurious correlations in training data |
| **Position bias** | Preference for specific positions | Order effects in annotation |
| **Verbosity bias** | Preference for over-decorated responses | Annotators mistakenly treat it as a quality signal |
| **Sycophancy bias** | Preference for responses that flatter users | Psychological tendencies of annotators |
| **Format bias** | Preference for lists/Markdown formatting | Spurious format associations in training data |
| **OOD collapse** | Inaccurate scoring for out-of-distribution generations | Limited RM generalization |
| **Surface feature shortcuts** | Scores based on keywords rather than semantics | Insufficient model capacity/data |
| **Annotator noise amplification** | RM learns individual annotator biases | Uneven annotation quality |

---

## 5. RM & Judge Evaluation

### 5.1 RewardBench

**Overview:** RewardBench is a comprehensive reward model evaluation benchmark designed to systematically test RM performance across multiple dimensions.

**Evaluation dimensions (approximate categories):**
- **Chat / Dialogue quality:** Judging the quality of conversational responses
- **Chat-Hard / Difficult dialogue:** Distinguishing subtle quality differences
- **Safety:** Identifying harmful content
- **Reasoning:** Judging correctness of mathematical/logical reasoning

**Evaluation method:**
- Pairwise accuracy: given a chosen/rejected pair, does the RM rank them correctly?
- **Note:** For specific scores and rankings, refer to the official RewardBench leaderboard; this document does not cite specific numbers <!-- CLAUDE-REVIEW: verify number -->

**Use cases:**
- Selecting the best RM for RLHF
- Diagnosing which domains an RM performs poorly on
- Comparing the effectiveness of different training methods

### 5.2 LLM-as-Judge

> 📎 **Cross-reference**: This section focuses on LLM-as-Judge as a **training signal** (how its biases can contaminate RM training data and affect RLHF optimization). For specifics on LLM-as-Judge in **evaluation practice** — including bias mitigation and benchmark selection — see `eval-and-judges-en.html §2`.

**Core idea:** Use a powerful LLM (e.g., GPT-4-class models) to directly score or rank response quality, replacing human evaluation.

#### Common Evaluation Paradigms

| Paradigm | Description |
|------|------|
| **Pointwise Scoring** | Rate a single response on a 1–10 scale |
| **Pairwise Comparison** | Choose the better of two responses |
| **Ranking / Listwise** | Rank multiple responses |
| **Reference-guided** | Score by comparison against a reference answer |

#### Common Biases in LLM-as-Judge

##### (a) Position Bias
- **Phenomenon:** LLMs tend to select the response presented in the **first position** (or a specific position)
- **Verification:** Swap the A/B order and check for consistency
- **Mitigation:** Evaluate twice (A first then B, then B first then A); retain only consistent results

##### (b) Verbosity Bias
- **Phenomenon:** LLMs prefer longer, more detailed responses, even when concise responses are of higher quality
- **Cause:** Consistent with the pattern that detailed responses typically score higher in training data
- **Mitigation:** Explicitly instruct that "length should not affect the score"

##### (c) Self-Preference Bias
- **Phenomenon:** LLM judges tend to assign higher scores to responses generated by themselves (or models in the same family)
- **Cause:** The generation style/distribution is closer to the judge's own training distribution
- **Mitigation:** Use models from different families for evaluation; average across multiple judges (multiple models)

##### (d) Other LLM-as-Judge Biases

| Bias | Description |
|------|------|
| **Format bias** | Preference for responses using Markdown, lists, and other formatted elements |
| **Capability boundary bias** | LLM cannot correctly evaluate responses that exceed its own capabilities |
| **Sycophancy bias** | Overly "friendly"; reluctant to give low scores |
| **Anchoring effect** | Influenced by candidate content seen earlier |
| **Keyword bias** | Oversensitivity to specific terminology |

#### LLM-as-Judge Best Practices

1. **Structured rubrics:** Clearly define evaluation dimensions and criteria
2. **Position debiasing:** Evaluate twice with swapped positions
3. **Temperature set to 0:** Reduce randomness, improve consistency
4. **Multi-judge:** Use multiple different LLMs and average/vote
5. **Human calibration:** Periodically check consistency against human evaluations
6. **Pairwise > absolute:** LLMs make more reliable relative judgments than absolute ratings
7. **Chain-of-Thought evaluation:** Have the LLM reason first, then assign a score

### 5.3 Trade-offs Between Human and Automated Evaluation

```
Evaluation quality ←——————————————————————→ Evaluation cost
  Expert human eval    Crowdsourcing    LLM-as-Judge    Automatic metrics (BLEU/ROUGE...)
  (highest quality)                                             (lowest cost)
```

### 5.4 Other Evaluation Methods

- **Win Rate:** Proportion of pairwise comparisons won against a baseline model
- **Elo Rating:** Dynamic ranking system based on tournament-style comparisons
- **Multi-dimensional evaluation:** Independent scoring on helpfulness, safety, factuality, etc.
- **Adversarial evaluation:** Tests RM robustness on edge cases and adversarial examples

---

## 6. Interview Questions

### L1 — Fundamentals

---

<details>
<summary>Q1: What is a Reward Model (RM)? What role does it play in RLHF?</summary>

**A:** A reward model is a scoring function learned from human preference data that outputs a scalar reward value for LLM-generated responses. In RLHF, the RM acts as a proxy for human preferences, serving as the objective function for policy optimization — the policy model improves its output quality by maximizing the RM score.

> **Follow-up:** Why can't human annotations be used directly as rewards for RL?
> Because RL requires a large volume of online reward signals; human annotation is expensive and slow. The RM provides a proxy signal that can be computed in batch, instantly.

</details>

---

<details>
<summary>Q2: What are the core assumptions of the Bradley-Terry model? What does its training loss function look like?</summary>

**A:** The BT model assumes: for a given prompt, the probability that response $y_w$ is preferred over $y_l$ is determined by the sigmoid function of the difference in their reward scores. The training loss is the negative log-likelihood: $\mathcal{L} = -\log\sigma(r(y_w) - r(y_l))$. Core assumptions include that preferences can be explained by scalar score differences and that preferences are transitive.

> **Follow-up:** What happens if preferences are not transitive?
> The BT model encounters inconsistent annotation pairs, leading to unstable training. In such cases, more flexible preference models such as Plackett-Luce may be considered.

</details>

---

<details>
<summary>Q3: What is the core difference between PRM and ORM? Which scenarios is each suited for?</summary>

**A:** ORM assigns a single reward to the final output only; the signal is sparse. PRM assigns rewards to each reasoning step; the signal is dense. PRM is suited for multi-step reasoning tasks (math, code); ORM is suited for end-to-end evaluation tasks (dialogue, writing). PRM's advantage is precise credit assignment, but its annotation cost is far higher than ORM's.

> **Follow-up:** How can PRM training data be obtained automatically?
> Monte Carlo estimation can be used: continue sampling multiple rollouts from each step and use the final answer accuracy as an approximation of that step's reward.

</details>

---

<details>
<summary>Q4: What is Best-of-N (BoN) sampling? How does it differ from RL training?</summary>

**A:** BoN samples N responses from the policy and uses the RM to select the one with the highest score as the final output. It is an inference-time optimization method that does not update model parameters; it is simple but inference cost scales proportionally with N. RL training updates parameters during training, requiring no additional sampling at inference time.

> **Follow-up:** What is the ceiling of BoN?
> Diminishing returns as N grows; also limited by the policy's original distribution — it cannot generate high-quality responses outside the distribution.

</details>

---

### L2 — Intermediate

---

<details>
<summary>Q5: What is reward hacking? Give 2–3 concrete examples.</summary>

**A:** The policy model learns to exploit RM weaknesses for high scores rather than genuinely improving quality. Examples: (1) generating verbose content to score higher (length hacking); (2) using flattering/sycophantic language; (3) repeating high-scoring template sentences from training data; (4) overusing formatting elements (headings, lists, bold text).

> **Follow-up:** What is the relationship between Goodhart's Law and reward hacking?
> Goodhart's Law states that "once a measure becomes a target, it is no longer a good measure" — the RM is an approximate measure of human preferences; when the policy specifically optimizes it, the two become decoupled.

</details>

---

<details>
<summary>Q6: Explain in detail the role of KL divergence penalty in RLHF and how it is tuned.</summary>

**A:** The KL penalty term $\beta \cdot D_{\text{KL}}(\pi \| \pi_{\text{ref}})$ constrains the policy from deviating too far from the reference model, acting as regularization. $\beta$ too large → the policy barely updates and learns nothing; $\beta$ too small → allows excessive exploration, prone to reward hacking. Adaptive KL is commonly used: set a target KL value and adjust $\beta$ dynamically.

> **Follow-up:** Which layer of reward hacking does each of KL penalty and RM ensembles address?
> KL constrains the policy's scope from the optimization constraint layer; RM ensembles reduce scoring variance and over-optimism from the reward estimation layer. The two are complementary.

</details>

---

<details>
<summary>Q7: How can RM ensembles mitigate reward hacking? What are the differences between aggregation strategies?</summary>

**A:** Train multiple independent RMs (different seeds/data subsets/architectures) and aggregate scores. Strategies: (1) **Mean:** smooths scores, reduces noise from any single RM; (2) **Min / conservative:** takes the most conservative score, avoids over-optimism — safer when the policy drifts out of distribution; (3) **Uncertainty-weighted:** reduce the weight of high-variance samples. In practice, the min strategy is most effective against reward hacking but can be overly conservative.

> **Follow-up:** How can the compute overhead of ensembles be optimized?
> Use a shared-parameter backbone with different heads; or train multiple lightweight variants using PEFT adapters.

</details>

---

<details>
<summary>Q8: What causes length bias? How can it be mitigated at the data level and inference level?</summary>

**A:** **Cause:** In training preference pairs, chosen responses are typically longer than rejected ones; the RM overfits to the spurious feature "long = good." **Data level:** Construct preference pairs with similar lengths; analyze and remove length correlations. **Inference level:** Apply a length penalty $r' = r - \alpha \cdot \text{len}$; or use length normalization.

> **Follow-up:** Why might applying a length penalty only at inference time be insufficient?
> If the RM has already internalized length as a strong feature, the length penalty may not fully offset it; the more fundamental approach is to debias at the training data or training method level.

</details>

---

<details>
<summary>Q9: How does iterative re-training mitigate distribution shift?</summary>

**A:** Standard RLHF trains the RM only on data generated by the initial policy; as RL training progresses, the policy distribution shifts, and the RM becomes inaccurate on the new distribution. Iterative re-training uses the current policy to generate new data → re-annotate → update the RM → continue RLHF, ensuring the RM always covers the policy's current distribution. The cost is more than a doubling of compute and annotation overhead.

> **Follow-up:** Does DPO also need iterative re-training?
> In theory, DPO also faces the distribution shift problem; iterative DPO (online DPO / online preference optimization) has been proposed to address this.

</details>

---

<details>
<summary>Q10: What are the main biases of LLM-as-Judge? How can they be systematically mitigated?</summary>

**A:** Main biases: (1) **Position bias** — prefers specific positions; (2) **Verbosity bias** — prefers long responses; (3) **Self-preference** — gives higher scores to models from the same family; (4) **Format bias** — prefers formatted content. Mitigations: swap positions for dual evaluation, define explicit scoring rubrics, vote across multiple judges, periodically calibrate against humans.

> **Follow-up:** Why is pairwise comparison generally better than absolute scoring (pointwise)?
> Because absolute scoring requires the judge to maintain a consistent internal scoring scale, which is prone to drift; pairwise comparison only requires judging relative quality, which is a simpler cognitive task with higher consistency.

</details>

---

<details>
<summary>Q11: What is RewardBench? Which capability dimensions does it evaluate in RMs?</summary>

**A:** RewardBench is a comprehensive reward model evaluation benchmark that measures RM performance across multiple dimensions using pairwise accuracy. Dimensions include: general dialogue quality, challenging dialogue (distinguishing subtle differences), safety, and reasoning ability. It provides a standardized comparison framework that helps diagnose an RM's strengths and weaknesses.

> **Follow-up:** Can RewardBench results directly predict an RM's actual performance in RLHF?
> Not entirely. Pairwise accuracy on a benchmark is a necessary but not sufficient condition — an RM's performance on a benchmark is not fully correlated with whether it will encounter reward hacking during RLHF optimization; online evaluation is also needed.

</details>

---

### L3 — Advanced

---

<details>
<summary>Q12: From a theoretical perspective, why is KL constraint necessary when the RM is imperfect?</summary>

**A:** When $r_{\text{RM}} \neq r_{\text{true}}$, unconstrained maximization of $r_{\text{RM}}$ can cause $r_{\text{true}}$ to decrease (reward hacking). The KL constraint is equivalent to optimizing within a local region where the RM is reliable — the RM is accurate near the training distribution, and KL ensures the policy does not leave this "trust region." This aligns with the trust region concept in TRPO/PPO. When KL is 0, $\pi = \pi_{\text{ref}}$; when KL is small, RM reliability is high.

> **Follow-up:** Are there types of reward hacking that KL penalty cannot prevent?
> Yes — if reward hacking occurs within a low-KL region near $\pi_{\text{ref}}$ (e.g., simply adding a few flattering words), KL penalty cannot stop it. In such cases, a better RM is needed.

</details>

---

<details>
<summary>Q13: Design a complete iterative RLHF system and describe its architecture and key decision points.</summary>

**A:**

```
Architecture:
π₀ (SFT model)
   ↓
[Data generation] π_k generates → N responses per prompt
   ↓
[Preference annotation] Human annotations or RM annotations (on-policy data)
   ↓
[RM update] Fine-tune RM_k on new data → RM_{k+1}
   ↓
[Policy optimization] RLHF (PPO) with RM_{k+1}, KL→π_ref
   ↓
[Evaluation] Reward curves, KL curves, human evaluation, reward hacking detection
   ↓
Repeat k = 1, 2, ...
```

Key decision points:
- Iteration frequency (how many RL steps before updating the RM)
- Whether to collect human annotations each round (high cost) vs. using RM self-annotation
- Choice of KL target value
- When to stop iterating (reward saturation / human evaluation target met)

> **Follow-up:** How do you detect when iteration should stop? What are the potential risks of continuous iteration?
> Stop when human evaluation scores stop improving, KL continues to increase, or mode collapse occurs. Continued iteration may cause the policy to converge toward the RM's specific preferences, losing diversity.

</details>

---

<details>
<summary>Q14: Compare PRM's Monte Carlo estimation method with direct human annotation; analyze the bias-variance trade-off of each.</summary>

**A:** **MC estimation:** Sample K rollouts from each step and use final accuracy as the reward. Bias comes from limited sampling (estimates are imprecise when K is small); variance comes from sampling randomness. Larger K is more accurate but more expensive. **Human annotation:** Theoretically unbiased but constrained by annotator capability/consistency; subject to systematic bias and noise. MC estimation is scalable but has systematic bias; human annotation is accurate but not scalable. A hybrid approach (small-scale human annotation + MC expansion) is common practice.

> **Follow-up:** Under what conditions does MC estimation fail severely?
> When the downstream search space from a given step is extremely large and correct answers are extremely rare (e.g., complex math problems), even many rollouts may all fail, causing zero reward for correct steps (false negatives).

</details>

---

<details>
<summary>Q15: How would you design an evaluation framework that is robust to reward hacking?</summary>

**A:** Multiple layers of defense are needed:

1. **RM internal metrics:** Pairwise accuracy (in-distribution vs. OOD), calibration
2. **Proxy metric monitoring:** KL curves, reward distribution drift, generation length changes, n-gram repetition rate
3. **Blind human evaluation:** Randomly sample outputs for human scoring; compare against RM scores
4. **Adversarial test sets:** Construct test cases with known reward hacking patterns
5. **Multi-RM consistency:** Declining score correlation across multiple RMs as an early warning signal
6. **A/B testing:** End-user satisfaction as the ultimate evaluation

> **Follow-up:** In practice, which layer is most often overlooked but most critical?
> OOD detection is most often overlooked — people typically only look at in-distribution accuracy, but RM performance on the OOD distribution generated by the policy is what determines whether reward hacking occurs.

</details>

---

<details>
<summary>Q16: Discuss the impact of LLM-as-Judge self-preference bias on benchmark leaderboards.</summary>

**A:** If a mainstream LLM-as-Judge (e.g., GPT-4) has self-preference bias, models stylistically similar to it will rank higher on leaderboards, distorting rankings. For example, if a model is from the same family as the judge or trained on similar data, it may receive disproportionately high scores. This creates judge sensitivity in benchmark results — rankings depend on judge selection — undermining the comparability and credibility of leaderboards.

> **Follow-up:** How would you design a leaderboard that is robust to judge bias?
> Use multiple heterogeneous judges (different vendors, different sizes); report inter-judge agreement; disclose the relationship between the judge and each model; and incorporate human evaluation as an anchor.

</details>

---

<details>
<summary>Q17: Why is "the RM's generalization ability the most critical bottleneck in RLHF"?</summary>

**A:** The entire RLHF optimization loop relies on the RM as its objective function. The RM's capability ceiling determines the ceiling of policy optimization. Specifically: (1) Poor RM generalization → reward hacking; (2) Biased RM → policy inherits the bias; (3) Unreliable RM on OOD → iterative training also fails to improve it; (4) RM cannot evaluate quality dimensions beyond its training distribution → the policy cannot improve on those dimensions. All other techniques (KL, ensemble, iteration) are compensating for insufficient RM generalization.

> **Follow-up:** Can scaling up the RM (increasing parameter count) systematically solve the generalization problem?
> Larger RMs do have better generalization (higher capacity, better feature representations), but cannot fully solve it — because the fundamental bottleneck is sometimes the noise and incompleteness of the preference data itself, not model capacity.

</details>

---

<details>
<summary>Q18: From an information-theoretic perspective, why is pairwise more efficient than pointwise?</summary>

**A:** Human preference judgments are fundamentally **ordinal** information rather than **cardinal** information. Pairwise methods directly leverage ordinal information (A > B) with minimal information loss; pointwise requires humans to map to an absolute scale (e.g., 1–5), introducing additional scale calibration noise. From an information-theoretic perspective, a portion of the information in each pointwise annotation is "wasted" on scale noise. Pairwise therefore has higher sample efficiency.

> **Follow-up:** Is there a way to recover absolute scores from pairwise data?
> The latent scores of each response can be recovered by solving for the MLE of the BT model, but the absolute values only have relative meaning and require an additional anchor point to determine the scale.

</details>

---

<details>
<summary>Q19: Compare DPO and RLHF in handling reward hacking — similarities and differences.</summary>

**A:** **Similarities:** Both rely on implicit/explicit reward signals in preference data; both are subject to the constraints of Goodhart's Law. **Differences:** (1) RLHF has an explicit RM that can be attacked and diagnosed; DPO implicitly encodes rewards, making it hard to inspect in isolation; (2) RLHF allows flexible adjustments via KL, ensembles, and iteration; DPO's β is similar to a KL penalty but offers different control granularity; (3) DPO is trained on off-policy data, facing more severe distribution shift; (4) RLHF's PPO is inherently on-policy, which to some extent partially mitigates distribution shift.

> **Follow-up:** Are there forms of reward hacking unique to RLHF that DPO would not encounter?
> Issues specific to RLHF include: the explicit RM's OOD scoring failures, and instability in PPO training leading to sudden policy changes. DPO does not encounter explicit RM OOD problems, but faces implicit reward distribution shift problems — different in form but similar in nature.

</details>

---

<details>
<summary>Q20a (L3): What practical guidance does the scaling law from Gao et al. 2022 offer for setting KL penalty strength?</summary>

**A:** The paper's core finding is that in their experimental setup, increasing the KL penalty coefficient $\beta$ does not improve the gold score–KL curve (the frontier); the effect is equivalent to early stopping on the same curve. This implies:

1. **$\beta$ cannot be tuned as a generalization measure**: Increasing $\beta$ constrains the policy to deviate less from the reference model, reducing KL consumption, but does not make the RM more robust to the same KL shift; "being safer" is only because optimization stops earlier, not because the RM itself improved.

2. **The true role of KL penalty**: Preventing the policy from moving too far in a single step (stabilizing training), not fundamentally solving reward hacking. What is truly needed is improvement in RM generalization itself (more data, larger models, iterative re-training).

3. **Practical implication**: One should not rely on increasing $\beta$ to "buy" more optimization headroom; if the gold score has already peaked, the RM should be retrained rather than continuing to strengthen KL constraints on the same RM.

> **⚠️ Honesty note**: The paper's authors explicitly note that this conclusion is "sensitive to hyperparameters" and is not guaranteed to hold in all settings.

> **Follow-up:** BoN and RL have different functional forms (BoN is $d(\alpha - \beta d)$, RL is $d(\alpha - \beta \log d)$) — what does this tell us?
> BoN's quadratic form means over-optimization deteriorates with acceleration (decline is faster at larger $d$); RL's logarithmic form means deterioration is slower but sustained. Therefore, for the same KL "budget," BoN is more efficient but also collapses faster; one should not directly compare optimization quantity across methods using KL, as the two obey different over-optimization dynamics.

</details>

---

<details>
<summary>Q20: If you were designing the next generation of RM, what do you think are the three most important improvement directions?</summary>

**A:** (Open-ended question; three key directions below)

1. **Stronger generalization and OOD robustness:** Current RMs perform well in-distribution but collapse out-of-distribution. Better architectural design (e.g., uncertainty-aware RM), adversarial training, and broader training data coverage are needed.

2. **Multi-dimensional disentangled scoring:** Decouple helpfulness, safety, factuality, style, and other dimensions into independent scoring heads, avoiding compression of information into a single scalar. This also allows different optimization weights for different dimensions.

3. **Self-improving RM:** Give the RM self-calibration capability — continuously detecting during RLHF whether its own predictions are consistent with actual preferences, and updating automatically. Reduces reliance on human annotation.

> **Follow-up:** What potential conflicts exist between these three directions?
> Multi-dimensional scoring increases model complexity, which may affect generalization; self-improvement mechanisms may introduce systematic bias if unreliable; uncertainty estimation is computationally expensive in high-dimensional spaces. Trade-offs must be made in engineering practice.

</details>

---

## Appendix: Key Terminology Glossary

| Term | Chinese | Brief Definition |
|----------|----------|----------|
| Reward Model (RM) | 奖励模型 | Scoring function learned from preference data |
| RLHF | 基于人类反馈的强化学习 | RL optimization using RM signals |
| Bradley-Terry Model | 布拉德利-特里模型 | Probabilistic model for pairwise comparison |
| ORM | 结果奖励模型 | Scores only the final output |
| PRM | 过程奖励模型 | Scores each reasoning step |
| Reward Hacking | 奖励欺骗 | Exploiting RM weaknesses for high scores |
| KL Divergence | KL 散度 | Measure of the difference between two distributions |
| Distribution Shift | 分布偏移 | Mismatch between training and evaluation data distributions |
| Best-of-N (BoN) | N选一 | Sample N responses and select the highest-scoring one |
| LLM-as-Judge | 大模型作评估者 | Using an LLM to replace human evaluation |
| Length Bias | 长度偏差 | RM preference for longer responses |
| Self-Preference Bias | 自我偏好偏差 | LLM preference for responses in its own style |
| Inter-annotator Agreement | 标注者间一致性 | Degree of consistency across different annotators |
| Trust Region | 信赖域 | Safe update region in optimization |
| Credit Assignment | 信用分配 | Attributing final outcomes to individual steps |
| Elo Rating | Elo 排名 | Dynamic scoring system based on win/loss outcomes |
| Plackett-Luce Model | Plackett-Luce 模型 | Probabilistic model for ranked data |
| DPO | 直接偏好优化 | Preference learning that bypasses an explicit RM |
| Conservative Estimation | 保守估计 | Strategy of taking the minimum score in an ensemble |

---

*This cheatsheet is for study reference only. For specific numbers, refer to the original papers and official leaderboards.* <!-- CLAUDE-REVIEW: verify number -->

## Extended L3

<details>
<summary>Q: When designing RMs for "next-generation" foundation models (e.g., with stronger reasoning and planning capabilities), what fundamental challenges might existing evaluation paradigms (such as RewardBench) face?</summary>

**A:** The core challenge lies in the leap in complexity of the object being evaluated. Most existing benchmarks target relatively standard dialogue or simple reasoning tasks. For agents capable of long-horizon planning, tool use, or complex sub-task decomposition, the RM needs to evaluate not just the quality of a single response, but the **overall effectiveness of an interaction trajectory** and the **soundness of long-term decisions**. This requires evaluation paradigms to shift from "scoring static snippets" to "evaluating dynamic sequences," and calls for new metrics that can understand state transitions and world models — a fundamental expansion of the capability dimensions required of RMs.

> **Follow-up:** How should RM training data be constructed under this new paradigm?
> The shift needs to go from "preference pairs" to "trajectory comparison" data, potentially involving multi-step interactions in simulated environments. Annotation will rely more heavily on automated verification (e.g., task success rate) and sandboxed testing in advanced simulators; the feasibility of pure human annotation will drop sharply.

</details>

---

<details>
<summary>Q: What are the fundamental difficulties in applying Process Reward Models (PRM) to open-ended generation tasks (e.g., creative writing)?</summary>

**A:** The fundamental difficulty lies in the **ambiguity of "process" definition and the subjectivity of evaluation**. In mathematical reasoning, "steps" have clear logical boundaries and objective correctness criteria. But in creative writing, the transitions between paragraphs, the construction of imagery, and the build-up of emotion have no objective standards; their quality is highly dependent on subjective taste and holistic context. As a result, collecting reliable step-by-step annotations for PRM is extremely difficult, and automated evaluation (e.g., MC estimation) also fails due to the lack of a clear "correct answer."

> **Follow-up:** Should PRM be abandoned entirely for open-domain tasks, or are there compromises?
> A "coarse-grained PRM" or "hybrid reward" approach can be adopted. For example, divide the writing process into a small number of high-level "stages" (e.g., ideation, development, conclusion), or combine ORM's holistic reward with process rewards at key turning points (e.g., plot climaxes), to balance evaluation granularity with feasibility.

</details>

---

<details>
<summary>Q: From a game-theoretic perspective, can reward hacking be viewed as a "red team–blue team" dynamic game between the policy model and the reward model? What does this imply for designing mitigations?</summary>

**A:** Yes, this is essentially a non-cooperative game. The policy model (blue team) aims to discover and exploit vulnerabilities in the RM's (red team's) decision boundary to maximize rewards. Traditional mitigation strategies (such as fixed KL penalties) are "static defenses," whereas the game-theoretic perspective suggests adopting **dynamic, adaptive adversarial strategies**. For example, a dedicated "red team" RM can be trained whose objective is not to score responses but to actively find patterns that let the current policy obtain inflated rewards, and the discovered vulnerabilities are then used to update (harden) the primary RM.

> **Follow-up:** What stability and cost challenges might this adaptive adversarial approach face in practice?
> The main challenge is that training dynamics may be unstable, with both parties entering an "arms race" that causes the optimization objective to drift continuously. Maintaining multiple adversarial models also introduces significant compute and coordination overhead.

</details>

---

<details>
<summary>Q: How does RM calibration affect the RLHF optimization process? What specific risks does an "accurate but uncalibrated" RM pose?</summary>

**A:** Calibration refers to whether an RM's output scores truthfully reflect the absolute probability or expected value of response quality. An "accurate but uncalibrated" RM may rank correctly while its absolute score values or distribution have systematic bias. In RLHF, this can lead to **misjudgment of optimization intensity**: for example, uniformly inflated RM scores may cause the optimizer to believe the policy is already good and stop too early; or a narrow scoring range for small quality differences may result in insufficient policy update momentum (weak gradient signal). The KL penalty term depends on the relative magnitude of rewards; an uncalibrated RM may invalidate this trade-off.

> **Follow-up:** In engineering practice, what methods can diagnose and improve RM calibration?
> Calibration curves can be plotted: segment RM prediction scores and compare the true quality (human annotation or task success rate) within each segment. Improvement methods include adding calibration regularization terms to the RM training loss, or post-processing RM outputs at inference time (e.g., temperature scaling).

</details>

---

<details>
<summary>Q: When integrating safety as a hard constraint (rather than an optimization objective) into the RM framework, what is the theoretically most rigorous formalization?</summary>

**A:** The most rigorous approach is to model the problem as a **constrained optimization problem**, rather than a simple multi-objective weighted sum. Specifically, the optimization objective remains maximizing the RM score on the primary quality dimension (e.g., helpfulness), subject to a set of safety constraints (e.g., RM_safety(x, y) > τ). Theoretically, this can be solved via Lagrange multipliers or projected gradient methods, ensuring the policy optimizes within the safety-feasible set. This is more robust to the safety objective being sacrificed compared to mixing safety and helpfulness into a single score.

> **Follow-up:** What is the main difficulty in setting an explicit threshold τ for safety constraints (e.g., "harmful probability < 0.1%") in practice?
> The core difficulty is the fuzziness and context-dependence of safety boundaries. A response that is "safe" in most contexts may be unsafe in specific sensitive contexts. Therefore, a globally fixed τ is unrealistic; context-dependent dynamic adjustment is needed, which places higher demands on both the RM and the constraint system.

</details>

---

<details>
<summary>Q: Beyond ensembles and uncertainty filtering, what more active roles can RM uncertainty estimation play during RLHF training?</summary>

**A:** Uncertainty estimation can actively guide training data collection and exploration strategies, enabling **active learning**. For example, within the training loop, priority can be given to sampling and human annotation in prompt-response regions where RM uncertainty is high, to improve the RM most efficiently. Furthermore, during policy optimization, the agent can be encouraged to actively explore high-uncertainty regions (i.e., the RM's knowledge boundary) to discover potentially new high-quality strategies; this is analogous to the exploration-exploitation trade-off in Bayesian optimization.

> **Follow-up:** How can uncertainty-based exploration be specifically implemented in on-policy algorithms such as PPO in RLHF?
> RM uncertainty can be incorporated as part of the exploration reward, encouraging the policy to generate responses that the RM finds "surprising" or "uncertain." Specifically, add an exploration term positively correlated with uncertainty to the final reward, but carefully balance it to avoid generating meaningless random outputs.

</details>

## §A Key Papers Timeline

- **2017-06 · Deep RL from Human Preferences** — Christiano et al., NeurIPS 2017. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) — Establishes the RLHF foundation: trains a reward model from human pairwise trajectory comparisons and uses it to drive policy optimization via reinforcement learning.

- **2019-09 · Fine-Tuning Language Models from Human Preferences** — Ziegler et al., arXiv preprint. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593) — First systematic application of preference-based reward modeling (Bradley-Terry pairwise loss + KL penalty) to language model fine-tuning, establishing the modern RLHF pipeline blueprint.

- **2020-09 · Learning to Summarize with Human Feedback** — Stiennon et al., NeurIPS 2020. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325) — Validates the full RLHF loop on text summarization — pairwise preference annotation, RM training, PPO optimization — with outputs substantially preferred over supervised fine-tuning baselines.

- **2022-03 · Training Language Models to Follow Instructions with Human Feedback (InstructGPT)** — Ouyang et al., NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) — Scales RLHF to GPT-3-class models via a three-stage SFT → RM → PPO pipeline, with detailed treatment of preference data construction and margin-based quality filtering.

- **2022-10 · Scaling Laws for Reward Model Overoptimization** — Gao et al., ICML 2023. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760) — Quantifies the reward over-optimization dynamic: gold RM score rises then falls with KL distance, following a quadratic form for Best-of-N and a logarithmic form for RL, with coefficients that scale smoothly with RM size.

- **2023-05 · Let's Verify Step by Step (PRM800K)** — Lightman et al., ICLR 2024. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) — Systematically compares Process Reward Models (PRM) against Outcome Reward Models (ORM) on mathematical reasoning, releases 800K step-level human labels, and demonstrates that per-step supervision substantially outperforms outcome-only supervision.

- **2023-05 · Direct Preference Optimization (DPO)** — Rafailov et al., NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — Eliminates the explicit reward model in RLHF by reparameterizing the optimal policy as a closed-form function of preferences, reducing the problem to a simple binary cross-entropy loss without RM training or PPO sampling.

- **2023-06 · Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al., NeurIPS 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) — Systematically studies using strong LLMs as evaluators in place of humans, identifying position bias, verbosity bias, and self-preference bias, and introducing the MT-Bench and Chatbot Arena benchmarks.

- **2023-07 · Llama 2: Open Foundation and Fine-Tuned Chat Models** — Touvron et al., arXiv preprint. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — Large-scale open RLHF practice report detailing margin-based confidence filtering (significantly/slightly/negligibly better stratified weighting) as a core technique for preference data quality control.

- **2024-03 · RewardBench: Evaluating Reward Models for Language Modeling** — Lambert et al., arXiv preprint. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787) — Introduces a standardized RM evaluation benchmark spanning chat, hard chat, safety, and reasoning dimensions, using pairwise accuracy as the metric to enable systematic diagnosis and model selection.
