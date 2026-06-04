# Test-Time Scaling Cheatsheet
## From sequential/parallel scaling, search, and verifiers to compute-optimal allocation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Sequential Scaling](#2-sequential-scaling)
3. [Parallel Scaling](#3-parallel-scaling)
4. [Search-Guided Generation](#4-search-guided-generation)
5. [Verifiers](#5-verifiers)
6. [Compute-Optimal & RL-Learned Reasoning](#6-compute-optimal--rl-learned-reasoning)
7. [Interview Questions](#7-interview-questions)

---

## 1. Overview

**Test-time scaling (TTS)**, also called inference-time scaling, means **trading more compute spent at inference for higher accuracy, without changing the model weights**. It is the core of the o1 / R1 paradigm: the model learns to "think longer", and the evaluation lens shifts from "single-shot accuracy" to "accuracy under a given compute budget".

```
Train-time compute                       Test-time compute
─────────────────────                    ─────────────────────
+params / +data / +steps                 Sequential: longer chains (long CoT, budget forcing)
one-time cost, fixed               ⇄      Parallel: multi-sample + aggregate (BoN, self-consistency)
inference cost unchanged                  Search: verifier-guided tree/beam search (beam, MCTS)
                                          cost grows linearly/super-linearly with budget
```

### 1.1 Paradigm shift: train-time compute → test-time compute

Classic scaling laws focus on **train-time** investment (params, data, steps); TTS brings another knob to the fore — **how much inference compute to spend on a single query**. **Snell et al.** ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314), ICLR 2025) argue systematically that under a fixed FLOPs budget, **optimally allocating test-time compute** (search on hard problems, few samples on easy ones) is, **in some settings, more cost-effective than simply scaling up parameters**. This pries open the assumption that "capability is fully decided at training time": a portion of capability can be "cashed in on the spot" at inference.

> 💡 One-line intuition: train-time compute raises the model's "potential ceiling", while test-time compute raises "how much of that potential gets realized on a single problem". The two are complementary — a model that *can* solve a problem but often gets it wrong can gain a lot from TTS; a problem the model simply *cannot* do won't be solved no matter how much you sample.

### 1.2 Two axes: sequential (depth) vs parallel (width)

| Axis | Mechanism | Representative methods | Cost | When it works |
|------|------|----------|------|----------|
| **Sequential** | think longer / self-correct on one chain | long CoT, budget forcing (s1), self-refine | high serial latency | multi-step reasoning, self-correctable |
| **Parallel** | sample many independent chains, then aggregate | best-of-N, self-consistency, repeated sampling | parallelizable, throughput-friendly | answer space is aggregatable/verifiable |
| **Search** | tree expansion + verifier pruning | beam search, ToT, MCTS (rStar-Math) | highest | reliable process signal (PRM) available |

Search can be viewed as a hybrid of sequential and parallel — expanding step by step (depth) over multiple partial paths (width), with a verifier deciding which to keep.

### 1.3 Core question: how to optimally allocate a given compute budget

The central question of TTS is not "can we be more accurate", but **"given the same N× compute, how do we spend it best"**:

- **Difficulty adaptivity**: one of Snell's key findings is that the optimal strategy **varies with problem difficulty** — easy problems suit a small amount of sequential correction, hard problems suit wider parallel sampling + search.
- **Sequential vs parallel ratio**: parallel sampling buys coverage, sequential correction buys depth, and the optimal ratio between them depends on the task.
- **The verifier is the ceiling**: BoN / weighted voting / search **ultimately depend on verifier quality to "pick the right one" from the candidates** (see §5) — without a reliable verifier, sampling more doesn't help you select better (pure majority-vote self-consistency is the exception, relying on answer aggregatability and whether the majority answer is correct).

Below, §2–§4 unpack the three axes one by one, §5 covers the verifier that runs through all of them, and §6 covers compute-optimal allocation and "how RL teaches a model to use test-time compute well".

---

## 2. Sequential Scaling

Sequential scaling means **spending more tokens on a single reasoning chain**: thinking longer, writing out intermediate steps in full, and going back to fix errors once they're found.

### 2.1 long CoT and "think longer"

The most naive sequential scaling is simply **letting the model generate a longer chain of thought**. The core observation of o1 / R1-style models: after RL training, the model **spontaneously produces longer reasoning** on hard problems (self-reflection, case enumeration, re-checking), and on hard problems, within a suitable budget range, **accuracy rises with the number of reasoning tokens** (over-long reasoning on easy problems instead hurts, see §6.4). Note there are two layers here: ① whether the model **has the ability** to use more tokens well (decided by training); ② whether at inference you **let it** use more tokens (decided by decoding control). §2.2 handles the latter.

### 2.2 Budget forcing — s1

**s1** (Muennighoff et al., [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)) proposes a minimalist sequential-scaling control — **budget forcing**: directly manipulate the "thinking budget" during decoding.

- **Wants to stop early but below the lower bound** → suppress the end token, inject `"Wait"` to make the model keep thinking (often triggers self-correction);
- **Exceeds the upper bound** → forcibly truncate thinking and require an answer.

s1 uses only **1000** curated reasoning samples for SFT + budget forcing, yet achieves competitive results on math reasoning, demonstrating that "test-time scaling can be very simple".

```python
# Budget forcing (s1, Muennighoff et al. 2025): control the "thinking budget" at decode time.
# - wants to stop early but below the min token count → inject "Wait" to keep going, forcing
#   the model to think longer (often triggers self-correction);
# - exceeds the max → forcibly stop thinking and switch to answering. Uses a mock decoder to
#   demo the [control logic], with no real model.
def budget_forcing(model_step, min_think, max_think, wait_token="Wait"):
    """model_step(prefix)->(token, wants_stop): one mock decode step, returns the next
       token text and whether "the model wants to stop thinking"."""
    think_tokens, n_wait = [], 0
    while len(think_tokens) < max_think:
        tok, wants_stop = model_step(think_tokens)
        if wants_stop and len(think_tokens) < min_think:
            think_tokens.append(wait_token)   # suppress stop, force continuation
            n_wait += 1
            continue
        if wants_stop:
            break
        think_tokens.append(tok)
    return think_tokens, n_wait
```

### 2.3 Self-correction / refinement (self-refine) and its limits

**Self-Refine** (Madaan et al., [arXiv:2303.17651](https://arxiv.org/abs/2303.17651), NeurIPS 2023): use **the same LLM** in a loop to "generate → self-critique → revise per feedback", with no extra training or supervision data, improving quality on a range of generation tasks.

> ⚠️ **The ceiling of self-correction**: whether self-correction truly fixes errors depends on whether the model **can reliably tell that it was wrong**. A body of follow-up work finds that under purely intrinsic self-correction with **no external signal** (no verifier, no ground truth), LLMs often **change right to wrong** or "self-persuade" into errors, with limited or even negative net gains. Reliable sequential correction usually needs **external verification signals** (unit tests, calculators, retrieval) as a backstop — which is exactly the point of §5 verifiers. A common interview point: "is self-correction reliable?" — answer "it depends on whether there's a reliable external verification signal".

---

## 3. Parallel Scaling

Parallel scaling means **independently sampling many complete solutions, then aggregating them in some way**. Naturally parallelizable and throughput-friendly, it is the most easily deployable TTS in practice.

### 3.1 best-of-N and self-consistency (majority voting)

The two most basic aggregations:

- **Self-Consistency** (Wang et al., [arXiv:2203.11171](https://arxiv.org/abs/2203.11171), ICLR 2023): sample many CoTs for the same problem and **take a majority vote over the final answers**. No verifier needed, only that answers can be normalized and compared; significantly improves math/commonsense reasoning.
- **best-of-N (BoN)**: sample N solutions, use a **verifier/RM to score each, and take the single highest-scoring one**. Depends on verifier quality (§5).

The difference between the two: majority voting relies on "the mode of the answer distribution", BoN relies on "verifier selection". **Weighted self-consistency** is the in-between form — weight each vote by the verifier score before voting.

```python
from collections import defaultdict

# Three parallel aggregations (self-consistency from Wang et al. 2022 + BoN variants):
# - majority_vote: take a majority vote over the [final answers] (no verifier needed);
# - weighted_vote: weight each vote by verifier/RM score (weighted self-consistency);
# - best_of_n: directly take the single highest verifier-scoring solution (no voting).
def majority_vote(answers):
    counts = defaultdict(float)
    for a in answers:
        counts[a] += 1.0
    return max(counts, key=counts.get)

def weighted_vote(answers, scores):
    counts = defaultdict(float)        # scores assumed non-negative (e.g. verifier confidence in [0,1])
    for a, s in zip(answers, scores):
        counts[a] += s                 # accumulate weighted by verifier score
    return max(counts, key=counts.get)

def best_of_n(answers, scores):
    return max(zip(answers, scores), key=lambda x: x[1])[0]  # take the single highest-scoring one
```

### 3.2 Weighted voting / verifier reranking

When a verifier is available, pure majority voting wastes the signal: a **minority but high-confidence** correct answer can be voted out by a majority of wrong answers. Weighted voting / verifier reranking lets the verifier score participate in the decision, usually beating pure majority voting — provided the verifier itself is reliable (otherwise it amplifies noise). This is also why §5 verifier quality is the core bottleneck of TTS.

### 3.3 Coverage vs precision: the ceiling of pass@k

**Large Language Monkeys** (Brown et al., [arXiv:2407.21787](https://arxiv.org/abs/2407.21787)) systematically studies "repeated sampling": it defines **coverage = the fraction of problems where at least one sample is correct**, and finds coverage grows approximately **log-linearly** with the number of samples across four orders of magnitude. But beware the distinction between two notions:

$$\text{pass@}k = \mathbb{E}\left[1 - \binom{n-c}{k}\big/\binom{n}{k}\right]$$

- **pass@k / coverage**: "win as long as one correct sample is drawn" — this is the **upper bound given an oracle verifier** (knowing which one is correct).
- **Actually achievable accuracy**: limited by whether the real verifier can **pick that correct one out of the N**.

> ❌ **Pitfall**: treating coverage / pass@k as "the score the model actually gets". High coverage only means "the answer is among the candidates", not "it can be selected". **The real gain of sampling-scaling = improvement in coverage × the verifier's selection ability**; when the verifier is weak, coverage rises while actual accuracy barely moves. This mirrors §5. For "sample-filter-retrain" self-training routes like RFT, see [data-pipeline §3](cheatsheet-data-pipeline-en.html).

---

## 4. Search-Guided Generation

Search refines "sampling" from "sampling a whole complete solution" to "**step-by-step expansion + mid-course pruning**": use a verifier to score **partial reasoning paths**, keeping only the promising branches to continue. It is a hybrid of sequential (depth) and parallel (width).

### 4.1 step-level beam search and lookahead

Cut reasoning into "steps", expand several candidate next-steps per step, use a PRM (process reward model, §5) to score the **current partial path**, and keep top-B to continue — i.e. **step-level beam search**. **lookahead** goes further: before scoring, first simulate a rollout a few steps forward, using "future" information to estimate the value of the current step (more accurate but more expensive).

```python
# PRM-guided step-level beam search: expand candidate next-steps per step, use the process
# reward model (PRM) to score the [partial reasoning path], keep top-B to keep expanding.
# A path is represented as a list of step texts.
def prm_beam_search(root, expand, prm_score, beam=2, depth=4, n_expand=3):
    """expand(path)->list[next_step]; prm_score(path)->float. Returns the highest-scoring complete path."""
    beams = [(prm_score([root]), [root])]
    for _ in range(depth):
        cands = []
        for _, path in beams:
            for step in expand(path)[:n_expand]:
                new_path = path + [step]
                cands.append((prm_score(new_path), new_path))
        if not cands:
            break
        cands.sort(key=lambda x: x[0], reverse=True)
        beams = cands[:beam]                 # prune: keep only top-B branches
    return max(beams, key=lambda x: x[0])
```

### 4.2 Tree of Thoughts / MCTS

- **Tree of Thoughts (ToT)** (Yao et al., [arXiv:2305.10601](https://arxiv.org/abs/2305.10601), NeurIPS 2023): organize reasoning into a **tree** — nodes are coherent "thoughts", and the model can generate multiple candidate thoughts, **self-evaluate**, and search with BFS/DFS **backtracking**, suited to problems needing trial-and-error / planning.
- **MCTS (Monte Carlo Tree Search)**: use the selection-expansion-simulation-backup four steps to allocate the search budget over the reasoning tree, concentrating compute on more promising branches. **rStar-Math** (Guan et al., [arXiv:2501.04519](https://arxiv.org/abs/2501.04519)) uses MCTS + a process reward model for **self-evolution**, getting 1.5B–7B small models to strong math performance **without relying on distillation from frontier models**.

### 4.3 PRM-guided search

Whether beam or MCTS, **pruning/selection depends on a signal that can score "partial paths"** — exactly where the PRM (process reward model) comes in (an ORM can only score complete solutions and cannot directly guide mid-course pruning; it can only estimate partial-path value indirectly by rolling out to the end). The strength of search methods is **highly coupled to PRM quality**: when the PRM is noisy, search wastes compute on branches that were wrongly over-estimated. For PRM and ORM training details, see §5 and [reward-modeling-eval §2](cheatsheet-reward-modeling-eval-en.html).

---

## 5. Verifiers

The verifier is the **shared bottleneck** of TTS: parallel relies on it to rerank, search relies on it to prune. Here we only cover **how the verifier is consumed at test time**; for its training (RM loss, data construction), see [reward-modeling-eval §2](cheatsheet-reward-modeling-eval-en.html).

### 5.1 ORM vs PRM

| | ORM (outcome reward) | PRM (process reward) |
|------|------|------|
| What it scores | only the **complete solution / final answer** | each **reasoning step** |
| Can it guide search | no (cannot directly score partial paths, only estimate indirectly by rolling out to the end) | **yes** (the basis of beam/MCTS pruning) |
| Annotation cost | low (only need outcome correctness) | high (needs step-level labels → see §5.2 auto-labeling) |
| Signal granularity | coarse | fine, can locate "which step started going wrong" |

**Let's Verify Step by Step** (Lightman et al., [arXiv:2305.20050](https://arxiv.org/abs/2305.20050), ICLR 2024) builds **PRM800K** (800K human step-level annotations) and argues that **on BoN reranking of MATH solutions, process-supervised verifiers beat outcome-supervised ones**.

### 5.2 Automatic process annotation — Math-Shepherd

The pain point of PRMs is that step-level annotation is expensive. **Math-Shepherd** (Wang et al., [arXiv:2312.08935](https://arxiv.org/abs/2312.08935), ACL 2024) proposes **annotation-free** auto-labeling: for a given step, **roll out to the end multiple times** from it, and use "the fraction that reaches the correct answer after this step" as the process label for this step, then train a PRM. This PRM can be used for BoN reranking and also as a PPO reward.

### 5.3 Generative verifiers

**Generative Verifiers (GenRM)** (Zhang et al., [arXiv:2408.15240](https://arxiv.org/abs/2408.15240), ICLR 2025) change verification from "discriminative scoring" to **"next-token prediction"** — let the verifier **generate** a CoT judging "is this solution correct", then read out the yes/no probability as the score. Benefits: ① reuse the LLM's generation/reasoning ability for verification; ② the verifier can itself scale verification compute with CoT + majority voting (the verifier can also do TTS). This is the "LLM-as-judge" idea applied to verification.

```python
import numpy as np

# A PRM aggregates "the correctness probability of each step" into "a whole-path score". Three common aggregations:
# - min : the weakest step decides the whole (strictest, common for step-level verification);
# - prod: multiply the per-step independent correctness (equivalent to summing log-probs);
# - last: look only at the last step (approximates outcome-style scoring, not a strict ORM).
def aggregate_prm(step_probs, mode="min"):
    p = np.asarray(step_probs, dtype=float)
    if mode == "min":
        return float(p.min())
    if mode == "prod":
        return float(p.prod())
    if mode == "last":
        return float(p[-1])
    raise ValueError(f"unknown mode: {mode}")
```

### 5.4 The fragility of verifiers

> ⚠️ A verifier is not ground truth. It can be **reward-hacked** — e.g. learning "long solution = good solution", being fooled by specific formats, or being miscalibrated on out-of-distribution samples. TTS pushes the verifier to the core of the decision loop, and its bias gets **amplified by the number of samples** (the more you sample, the more likely you draw a solution that "fools the verifier but is actually wrong"). Verifiable domains (math exact-match, code unit tests) can bypass this problem with **rule-based verifiers** — which is also why RLVR / R1 mainline choose verifiable domains (see §6.3). For details on RM over-optimization and length bias, see [reward-modeling-eval §3–§4](cheatsheet-reward-modeling-eval-en.html).

---

## 6. Compute-Optimal & RL-Learned Reasoning

### 6.1 Compute-optimal allocation — Snell et al.

**Snell et al.** ([arXiv:2408.03314](https://arxiv.org/abs/2408.03314), ICLR 2025) propose **compute-optimal** test-time scaling: given a test-time compute budget, **adaptively choose the strategy by problem difficulty** (sequential correction vs parallel sampling vs search) rather than one-size-fits-all. Conclusions: ① difficulty-adaptive allocation significantly beats fixed strategies; ② in some settings, a small model + optimal TTS **can match or even surpass the single-shot inference of a model several times larger** — but this has limits and is not universal.

### 6.2 Reasoning scaling law: the test-time vs add-parameters tradeoff

TTS and scaling up parameters are **two ways to spend compute**, each with its applicable regime:

- **TTS is more cost-effective**: when the base model **already has the ability but often errs** (high coverage, low single-shot accuracy) and the task is **verifiable/aggregatable**.
- **Adding parameters is more cost-effective**: when hard problems are **fundamentally beyond the model's ability** (low coverage, no amount of sampling finds the right one), or there's **no reliable verifier** to pick from candidates.
- **Essence**: TTS cashes in "potential"; when potential is insufficient, the returns of test-time compute saturate quickly.

### 6.3 RL-learned test-time scaling — o1 / R1

TTS methods (§2–§4) are **inference-time** decoding/search strategies; but **whether the model can use this compute well is shaped at training time**. This is exactly the point of o1 / R1:

- **OpenAI o1** ("Learning to Reason with LLMs", OpenAI, 2024-09, no arXiv paper) uses large-scale RL to train the model to do long chain-of-thought reasoning before answering, publicly showing curves of **accuracy rising with test-time compute**.
- **DeepSeek-R1** (DeepSeek-AI, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948), later published in Nature 2025) trains long CoT with GRPO + verifiable rewards (RLVR), and **long reasoning behaviors (self-reflection, re-checking) emerge spontaneously through RL**; R1-Zero further shows that in its verifiable-task setting, pure RL (no SFT) can trigger them.

In other words: **RLVR makes the model acquire effective long-reasoning behaviors at training time**, while TTS cashes in that ability and allocates compute at inference time. For **training-side** details such as GRPO / RLVR / the R1 four-stage recipe, see [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html); this page focuses on the inference side.

### 6.4 Open problems

- **over-thinking**: forcibly lengthening CoT on easy problems instead hurts and wastes compute; how to **adaptively decide how long to think** is still an open problem (budget forcing is a coarse-grained attempt).
- **Verifier reliability**: non-verifiable domains (open-ended writing, multi-turn dialogue) lack reliable verifiers, limiting TTS gains (§5.4).
- **Generalization beyond verifiable domains**: RL-learned reasoning is mostly trained on math/code; whether it generalizes to general reasoning is still under study.
- **The economics of test-time compute**: latency and cost rise with the budget, and real deployment must trade off accuracy against per-query cost.

---

## 7. Interview Questions

### L1 — Fundamentals

---

<details>
<summary>Q1: What is test-time scaling? How does it differ from train-time scaling?</summary>

**A:** TTS means **not changing the weights, and spending more compute at inference to trade for accuracy** (think longer / sample more / search). Train-time scaling adds params/data/steps, a one-time investment with unchanged inference cost; TTS invests on demand at inference, with cost growing with the budget. Intuition: training decides the "potential ceiling", TTS decides "how much of that potential is realized per problem".

> **Follow-up:** Does piling on test-time compute improve accuracy on any problem?
> No. For a problem the base model **fundamentally can't do** (extremely low coverage), more sampling won't draw a correct one; TTS mainly helps problems the model "can do but often gets wrong".

</details>

---

<details>
<summary>Q2: What are sequential scaling and parallel scaling? Their representative methods?</summary>

**A:** **Sequential**: think longer / self-correct on one chain — long CoT, budget forcing (s1), self-refine, with high serial latency. **Parallel**: sample many independent chains then aggregate — best-of-N, self-consistency, repeated sampling, parallelizable and throughput-friendly. **Search** is a hybrid of the two (expand step by step over width).

> **Follow-up:** Which to deploy first in practice?
> Usually **parallel** first (easy to implement, parallelizable). Sequential correction has high latency; search is strongest but depends on a reliable PRM.

</details>

---

<details>
<summary>Q3: What is self-consistency? How does it differ from best-of-N?</summary>

**A:** **Self-Consistency (Wang et al. 2022)** samples many CoTs for the same problem and **takes a majority vote over the final answers**, no verifier needed. **best-of-N** samples N solutions then uses a **verifier/RM to score and takes the single highest-scoring one**. Difference: majority voting relies on the mode of the answer distribution, BoN relies on verifier selection; **weighted self-consistency** is the in-between form (weight votes by verifier score).

> **Follow-up:** With no verifier, which can you use?
> Majority voting (self-consistency), which only needs answers to be normalizable and comparable; BoN/weighted voting both need verifier scores.

</details>

---

<details>
<summary>Q4: What is budget forcing? How does s1 use it?</summary>

**A:** Budget forcing is the minimalist sequential-scaling control of **s1 (Muennighoff et al. 2025)**: manipulate the "thinking budget" at decode time — if it wants to stop early but is below the lower bound, inject `"Wait"` to force the model to keep thinking (often triggers self-correction); if it exceeds the upper bound, forcibly truncate and switch to answering. s1 achieves competitive results on math reasoning with only 1000 SFT samples + budget forcing.

> **Follow-up:** Why does "Wait" improve accuracy?
> It suppresses premature stopping, giving the model a chance to **review and correct** earlier reasoning; but forcibly lengthening easy problems can cause over-thinking and instead hurt.

</details>

---

<details>
<summary>Q5: What's the difference between ORM and PRM? Why do search methods favor PRM?</summary>

**A:** **ORM (outcome reward)** scores only the complete solution / final answer; **PRM (process reward)** scores each reasoning step. Search (beam/MCTS) needs to score **partial paths** to prune — ORM can't provide that, only PRM can guide mid-course selection. The cost: PRM step-level annotation is more expensive (see Math-Shepherd auto-labeling).

> **Follow-up:** Is PRM always better than ORM?
> When you need to guide search / locate the wrong step, PRM is better (Lightman et al. 2023); but PRM is harder to train, and when noisy it wastes search compute on over-estimated branches.

</details>

---

<details>
<summary>Q6: What is coverage / pass@k? Why is it an "upper bound" rather than actual accuracy?</summary>

**A:** coverage = **the fraction of problems where at least one sample is correct**; pass@k is its unbiased estimate $1-\binom{n-c}{k}/\binom{n}{k}$. It assumes an **oracle verifier** (knowing which one is correct), so it's an **upper bound**. Actually achievable accuracy is limited by whether the real verifier can **pick that correct one out of the N** — when the verifier is weak, coverage rises while actual accuracy barely moves.

> **Follow-up:** The main finding of Large Language Monkeys?
> coverage grows approximately **log-linearly** with the number of samples across about four orders of magnitude; "repeated sampling + verifier" yields large gains on code/math — but the real gain is limited by verifier quality.

</details>

---

<details>
<summary>Q7: Is self-refine reliable?</summary>

**A:** It depends on **whether there's a reliable external verification signal**. Self-Refine (Madaan et al. 2023) can improve quality when feedback is available; but under **purely intrinsic** self-correction (no verifier, no ground truth), the model often "changes right to wrong" or self-persuades, with limited or even negative net gains. Reliable sequential correction usually needs external signals like unit tests / calculators / retrieval as a backstop.

> **Follow-up:** What does this imply for agent design?
> Giving an agent **real executable feedback** (run code, check tool results) is more reliable than letting it "reflect in a vacuum".

</details>

---

<details>
<summary>Q8: Test-time scaling vs scaling up model parameters — which should you choose?</summary>

**A:** It depends on the task. **TTS is more cost-effective**: the base model already has the ability but often errs (high coverage, low single-shot), task verifiable/aggregatable. **Adding parameters is more cost-effective**: hard problems beyond the model's ability (low coverage), or no reliable verifier to pick from candidates. Essence: TTS cashes in "potential", and when potential is insufficient the returns saturate quickly (Snell et al. 2024's difficulty-adaptive conclusion).

> **Follow-up:** Can a small model + lots of TTS replace a large model?
> On some verifiable tasks it can match or even surpass (Snell); but this has limits and isn't universal — problems beyond the base model's ability can't be made up by TTS.

</details>

---

### L2 — Intermediate

---

<details>
<summary>Q9: Why is "the verifier the ceiling of TTS"? Give a concrete failure chain.</summary>

**A:** Parallel relies on the verifier to rerank, search relies on the verifier to prune — **the final "pick the right one" step depends on the verifier**. Failure chain: the verifier is biased (e.g. prefers long solutions) → the more you sample, the more likely you draw a solution that's "long but wrong, yet over-estimated by the verifier" → BoN/weighted voting selects it → accuracy drops instead of rising. That is, **the verifier's bias is amplified by the number of samples**. So high coverage ≠ high actual score, with verifier quality standing between the two.

> **Follow-up:** How to mitigate?
> Use rule-based verifiers (verifiable domains: exact-match / unit tests) to bypass neural-RM hacking; or use stronger / generative verifiers, and evaluate the verifier itself (see [reward-modeling-eval §5](cheatsheet-reward-modeling-eval-en.html)).

</details>

---

<details>
<summary>Q10: step-level beam search vs best-of-N on complete solutions — the essential difference and use cases?</summary>

**A:** BoN **first samples N complete solutions, then reranks with ORM/PRM at the end** (the decision is only at the tail); beam search **prunes at every step, concentrating compute on promising branches** (the decision runs throughout). beam needs a PRM (to score partial paths) and is theoretically more compute-efficient (prunes bad branches early), but is **more sensitive to PRM noise** — one mis-prune can lose the only correct branch. BoN is more robust (keeps full diversity) but costs more compute. When step counts are high and the PRM is reliable, beam/MCTS win; otherwise BoN is more stable.

> **Follow-up:** What does lookahead solve?
> Before scoring, roll out a few steps forward, using "future" information to correct the myopic estimate of the current step, reducing mis-pruning; the cost is being more expensive.

</details>

---

<details>
<summary>Q11: How does Math-Shepherd train a PRM without human step-level annotations?</summary>

**A:** For a given intermediate step, **roll out from it multiple times until a final answer**, and use "the fraction that rolls out to the correct answer after this step" as the **process label** for this step (a Monte Carlo estimate), then train a PRM on these auto labels. This PRM can be used for BoN reranking and also as a PPO reward. The core is using "outcome verifiability" to back-distill a "process signal", bypassing expensive human step-level annotation (compare PRM800K's 800K human annotations).

> **Follow-up:** What are the noise sources of these auto labels?
> Limited rollout sampling brings estimation variance; and "lucky hits" (a wrong step happening to reach the correct answer) can give a high score to a wrong step — needing enough rollouts and consistency checks.

</details>

---

<details>
<summary>Q12: The advantages of generative verifiers (GenRM) over discriminative RMs?</summary>

**A:** GenRM (Zhang et al. 2024) models verification as **next-token prediction**: let the verifier first **generate a CoT judging** correctness, then read the yes/no probability as the score. Advantages: ① reuse the LLM's generation/reasoning ability (verification can also "think a bit"); ② **the verifier itself can do TTS** — scale verification compute with CoT + majority voting; ③ unified with "LLM-as-judge". The cost: more expensive than discriminative scoring (it must generate).

> **Follow-up:** Won't this bring the generator's flaws into the verifier?
> Yes — GenRM can also be induced into wrong judgments by adversarial solution surfaces; and self-verification has a "model prefers its own style" bias, requiring independent evaluation of verifier quality.

</details>

---

<details>
<summary>Q13: What is compute-optimal test-time scaling? Snell et al.'s core conclusion?</summary>

**A:** Given a test-time compute budget, **adaptively choose the strategy by problem difficulty** (easy problems get a small amount of sequential correction, hard problems get wider parallel sampling + search) rather than fixing one. Snell et al. 2024's conclusion: ① difficulty-adaptive allocation significantly beats fixed strategies; ② in some settings, a small model + optimal TTS can match or even surpass the single-shot inference of a model several times larger. But there are limits — for problems beyond the base model's ability, TTS returns saturate quickly.

> **Follow-up:** In practice, how to estimate "problem difficulty"?
> Use the model's own uncertainty / the consistency of a few early samples / the verifier distribution to estimate online, and allocate the remaining budget accordingly.

</details>

---

<details>
<summary>Q14: What's the relationship between o1 / R1 and the decoding methods of §2–§4? What role does RL play here?</summary>

**A:** §2–§4 are **inference-time** decoding/search strategies; but **whether the model can use this compute well is shaped at training time**. o1 / R1 use large-scale RL (R1 uses GRPO + RLVR) to train long CoT, making **behaviors like self-reflection/re-checking emerge spontaneously** (R1-Zero shows that in its verifiable-task setting, pure RL can trigger them). So: **RLVR makes the model acquire effective long-reasoning behaviors at training time, and TTS cashes in and allocates that compute at inference time**. For training-side details, see [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html).

> **Follow-up:** Why do the o1/R1 mainline choose verifiable domains?
> RLVR uses rule-based verifiers (exact-match / unit tests) ≈ ground truth, with almost no neural-RM-hacking problem; the cost is being applicable only to verifiable domains.

</details>

---

### L3 — Advanced

---

<details>
<summary>Q15: From an information-theoretic / selection angle, why does actual accuracy get capped by verifier quality as the number of samples N → ∞, rather than tending to coverage?</summary>

**A:** Let the oracle accuracy (= coverage) be $C$, but we only have a **noisy verifier** $v$. Final accuracy = P(the selected solution is exactly correct). As $N$ grows, both the **number of correct solutions** and the **number of solutions that "look better but are actually wrong"** grow among the candidates; if the verifier has a systematic preference for "wrong but high-scoring" solutions (a nonzero false-positive rate), the max score of these distractors rises with $N$ and **overtakes** the score of the truly correct solution. So the probability that $\arg\max_v$ selects the correct solution is capped by the verifier's discrimination margin and FP-rate bound, and **does not tend to $C$ as N grows**. Under these assumptions ("fixed noisy verifier + single scoring + argmax selection"), only a **noiseless oracle** (a rule-based verifier) lets actual accuracy → coverage (with stronger assumptions like repeated verification / calibrated uncertainty / consistency verification, you can partly approach it). This explains why verifiable domains (RLVR) are the cleanest setting for TTS.

> **Follow-up:** What does this imply for the "weighted voting vs BoN" choice?
> Weighted voting is more robust to single-point verifier noise (votes smooth it out), while BoN is more sensitive to verifier tail errors (single-point argmax). When the verifier is noisy, voting is more stable; when it's near-oracle, BoN has a higher ceiling.

</details>

---

<details>
<summary>Q16: Work through the coupling between budget forcing and RL-trained long-CoT: why does forcibly applying budget forcing to a model without long-CoT RL barely help?</summary>

**A:** Budget forcing only **stops the model from halting at decode time**; it creates no capability — it can only **elicit long-reasoning behaviors the model has already learned**. For a model without long-CoT RL, the marginal content of "more tokens" is mostly **repetition / off-topic / self-persuasion** (because the distribution has no "high-quality continued reflection" pattern), so injecting `"Wait"` continues with low-value tokens and accuracy barely moves or even drops from the noise. A model trained with RLVR has compressed "reflect/re-check/switch approach" into the distribution, so `"Wait"` can truly trigger effective extra reasoning. Essence: **the gain of TTS ∝ the model's "elicitable ability" in long-horizon reasoning**, which is decided by training. This also echoes §6.2 — TTS cashes in potential, and potential is shaped by training.

> **Follow-up:** Then why is s1 effective with only 1000 SFT samples?
> s1's 1000 samples are **curated high-quality long-reasoning traces** that distill the pattern of "how to continue thinking effectively" into the model, giving budget forcing something to elicit; it takes the SFT-distill-long-CoT route, not from-scratch RL.

</details>

---

<details>
<summary>Q17: Why might PRM-guided beam search be [worse than] best-of-N on complete solutions when the PRM is biased? Give the bias-propagation mechanism.</summary>

**A:** beam search does argmax pruning with the PRM at **every step**, so errors **compound step by step**: if the PRM has a systematic bias on a certain class of intermediate steps, the correct branch may be mis-pruned **early**, and once pruned it **cannot recover** (a beam with no backtracking) — the final pool may contain no correct solution at all, with coverage collapsing inside the search. BoN instead keeps N **complete** solutions, with the PRM acting only **once at the end**; the correct solution (if already sampled) always stays in the candidate pool, and the verifier bias only affects "the final ranking", not "whether it exists". So when PRM bias is large: beam's **irreversible early-pruning** risk > BoN's **end-stage mis-ranking** risk. MCTS, with backtracking/exploration terms (UCT), mitigates part of the early-pruning problem and sits between the two.

> **Follow-up:** How to make beam more robust?
> Increase beam width (keep more branches), introduce randomness/temperature to avoid deterministic early pruning, use lookahead to reduce myopia, or soft pruning (sample by score rather than a hard top-B). The essence is hedging PRM bias with diversity.

</details>

---

<details>
<summary>Q18: The over-thinking phenomenon: why does "thinking longer" hurt on easy problems? How to design an adaptive thinking budget?</summary>

**A:** The correct solution to an easy problem is usually **short and direct**; forcibly lengthening CoT makes the model **introduce unnecessary steps**, each extra step carrying a nonzero error probability (error accumulation), and possibly "talking itself out of" an originally correct intuition — i.e. **the marginal return of reasoning is negative**. Ideas for an adaptive budget: ① use the **consistency of a few early samples** to estimate difficulty — if the first few samples are highly consistent (low entropy), judge it easy and stop early; ② use the **verifier distribution** — stop if the top candidate's score already far exceeds the rest; ③ train a **meta-controller / router** to predict each problem's optimal budget (learn "how long to think" itself). budget forcing's fixed bounds are a coarse-grained approximation; the ideal is per-query adaptivity.

> **Follow-up:** How to combine an adaptive budget with the RL training objective?
> You can **penalize the token count in the RL reward** (length regularization), making the model learn to "stop when enough"; but too heavy a penalty suppresses necessary long reasoning — back to the tension of §6.4.

</details>

---

<details>
<summary>Q19: Does test-time scaling change the methodology of "model evaluation"? When two models are compared under different test-time budgets, what counts as fair?</summary>

**A:** Yes. A single "pass@1" no longer characterizes a model that uses TTS — you must report an **accuracy-compute curve** (accuracy vs test-time FLOPs / tokens / samples). The key to a fair comparison is **aligning the compute axis**: ① fix **total inference FLOPs** and compare accuracy (not fixed sample count — a large model's single shot is more expensive); ② report the **whole curve** rather than a single point (curves may cross: A wins at small budgets, B wins at large ones); ③ distinguish **with vs without an oracle verifier** (coverage/pass@k is an upper bound; real deployment uses the accuracy of the real verifier). Otherwise comparisons like "small model + lots of sampling vs large model single shot" are distorted by a misaligned compute axis.

> **Follow-up:** What does this mean for leaderboard design?
> You need to publish the test-time budget and decoding protocol used in evaluation (sample count / whether search / verifier), otherwise scores aren't comparable; ideally compare at iso-FLOPs or report the Pareto front.

</details>

---

## Glossary

| Term | Brief definition |
|----------|----------|
| Test-Time Scaling (TTS) | Spend more compute at inference for accuracy, without changing weights |
| Inference-Time Scaling | A synonym for TTS |
| Sequential Scaling | Think longer / self-correct on one chain |
| Parallel Scaling | Sample many independent chains, then aggregate |
| long CoT | A longer step-by-step reasoning token sequence |
| Budget Forcing | Control the thinking budget at decode time (s1): inject "Wait" / hard-truncate |
| Self-Refine | Same-LLM generate→self-evaluate→revise loop |
| Self-Consistency | Majority vote over final answers across many CoTs |
| best-of-N (BoN) | Sample N, take the single highest-scoring one by verifier |
| Weighted Vote | Weight each vote by verifier score, then vote |
| Coverage / pass@k | Fraction of problems with at least one correct sample (oracle upper bound) |
| Repeated Sampling | Trade lots of sampling for coverage (Large Language Monkeys) |
| Beam Search (step-level) | Expand per step + PRM-prune to keep top-B |
| Tree of Thoughts (ToT) | Tree of thoughts + self-evaluation + backtracking search |
| MCTS | select-expand-simulate-backup to allocate the search budget |
| ORM | Outcome reward model: scores only the complete solution / final answer |
| PRM | Process reward model: scores each reasoning step, can guide search pruning |
| Math-Shepherd | Auto-construct step-level labels via rollout to train a PRM, no human annotation |
| Generative Verifier (GenRM) | Model verification as next-token prediction (generate a CoT judging correctness) |
| Compute-Optimal TTS | Adaptively allocate test-time compute by problem difficulty |
| RLVR | RL with verifiable rewards: use a rule-based verifier (exact-match/unit tests) as reward |
| over-thinking | Forcibly lengthening CoT on easy problems instead hurts |

---

*For study reference only. Paper conclusions and figures follow the original papers; benchmark scores are illustrative and not head-to-head comparisons.*

## §A Key Papers Timeline

- **2022-03 · Self-Consistency Improves Chain of Thought Reasoning in Language Models** — Wang et al., ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) — Replaces greedy decoding with sampling many CoTs + a majority vote over final answers, significantly improving arithmetic/commonsense reasoning; one of the earliest parallel test-time scaling methods.

- **2023-03 · Self-Refine: Iterative Refinement with Self-Feedback** — Madaan et al., NeurIPS 2023. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) — Uses the same LLM in a "generate→self-critique→revise" loop, improving quality without extra training; follow-up work notes that purely intrinsic self-correction yields limited gains without external signals.

- **2023-05 · Tree of Thoughts: Deliberate Problem Solving with Large Language Models** — Yao et al., NeurIPS 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) — Organizes reasoning into a tree: nodes are coherent "thoughts", supporting self-evaluation and BFS/DFS backtracking search, suited to problems needing trial-and-error / planning (there is also a same-named different work, Long et al. 2305.08291).

- **2023-05 · Let's Verify Step by Step** — Lightman et al., ICLR 2024. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) — Builds PRM800K (800K human step-level annotations), arguing that process-supervised verifiers beat outcome-supervised ones on BoN reranking of math solutions, establishing the value of PRMs.

- **2023-12 · Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations** — Wang et al., ACL 2024. [arXiv:2312.08935](https://arxiv.org/abs/2312.08935) — Auto-constructs step-level labels from rollout "the fraction reaching the correct answer after this step" to train a PRM, with no human annotation, usable for BoN reranking and as a PPO reward.

- **2024-07 · Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** — Brown et al., preprint. [arXiv:2407.21787](https://arxiv.org/abs/2407.21787) — Systematically studies repeated sampling: coverage grows approximately log-linearly with the number of samples across about four orders of magnitude; "sampling + verifier" yields large gains on code/math, but the real gain is limited by verifier quality.

- **2024-08 · Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** — Snell et al., ICLR 2025. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314) — Proposes compute-optimal TTS: adaptively allocate test-time compute by problem difficulty; in some settings a small model + optimal TTS can match the single-shot inference of a model several times larger.

- **2024-08 · Generative Verifiers: Reward Modeling as Next-Token Prediction** — Zhang et al., ICLR 2025. [arXiv:2408.15240](https://arxiv.org/abs/2408.15240) — Changes reward modeling/verification into next-token prediction: the verifier generates a CoT judging correctness and reads out the yes/no probability, reusing generation ability while the verifier itself can do TTS.

- **2024-09 · Learning to Reason with LLMs** — OpenAI, blog (no arXiv paper, o1 series release). [openai.com](https://openai.com/index/learning-to-reason-with-llms/) — Releases the o1 series: uses large-scale RL to train long chain-of-thought reasoning before answering, showing curves of accuracy rising with test-time compute, igniting the "inference-time scaling" paradigm.

- **2025-01 · rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking** — Guan et al., preprint. [arXiv:2501.04519](https://arxiv.org/abs/2501.04519) — Uses MCTS + a process reward model for self-evolution, getting 1.5B–7B small models to strong math performance without relying on distillation from frontier models.

- **2025-01 · s1: Simple test-time scaling** — Muennighoff et al., preprint. [arXiv:2501.19393](https://arxiv.org/abs/2501.19393) — With only 1000 curated reasoning samples for SFT + budget forcing (inject "Wait" / hard-truncate to control the thinking budget), achieves competitive results on math reasoning, arguing TTS can be very simple.

- **2025-01 · DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek-AI, Nature 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — Trains long CoT with GRPO + verifiable rewards, with long reasoning behaviors emerging spontaneously through RL (R1-Zero triggers them with pure RL); for the training-side recipe see [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html), this page focuses on the inference side.
