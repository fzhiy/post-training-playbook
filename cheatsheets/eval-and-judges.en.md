# Evaluation & LLM-as-Judge

> In post-training, **evaluation** is often the real bottleneck: training can run, but whether the model actually improved — or where it regressed — is entirely determined by evaluation. This page covers how to evaluate an aligned model and the pitfalls of various evaluation approaches.
> ⚠️ No concrete scores are listed here (they go stale quickly and are easy to misremember); for specific numbers, refer to official benchmark/leaderboard sources.

## 0. TL;DR

- In post-training, evaluation is the bottleneck: **capability benchmarks** (ground-truth, automatic), **preference/chat evals** (judge-scored), and **RM evals** each cover one facet.
- **LLM-as-judge** is cheap but has systematic biases (position / verbosity / self-preference / format) — **swapping order and judging twice** is the cheapest effective debias.
- **Data contamination** inflates scores: detect via n-gram overlap / Min-k% / canaries / paraphrase-drop; defend via dynamic private sets + temporal isolation.
- Reporting discipline: fix the prompt, report variance over seeds, check regressions (alignment tax), and remember a judge ≠ ground truth.

## 1. Three Families of Evaluation

| Type | Measures | Examples |
|---|---|---|
| **Capability benchmarks** (automatic, ground-truth answers) | Knowledge / reasoning / code | MMLU, GSM8K, MATH, HumanEval/MBPP, BBH, IFEval (instruction following) |
| **Preference / dialogue evaluation** (judge scoring) | Subjective quality of responses | AlpacaEval (LLM-judge win-rate), MT-Bench (multi-turn, judge scores), Chatbot Arena (human pairwise → Elo) |
| **Reward model evaluation** | Whether the RM aligns with human preferences | RewardBench (chat / safety / reasoning categories, etc.), agreement rate with human annotations |

### 1.1 Benchmark cheat-sheet

> ⚠️ The tables below list only **mechanism and structure** (signal source, scoring method, pitfalls), not concrete scores.

**Preference / dialogue evals** (judge or human):

| Eval | Signal source | Scoring | Most prominent bias | Debias / control |
|---|---|---|---|---|
| **AlpacaEval 2.0** | Single LLM-judge (model vs. reference answer) | Win-rate (vs. the reference) | Verbosity bias | Length-controlled win-rate (regress out length) |
| **MT-Bench** | LLM-judge | Multi-turn, 1–10 scalar score or pairwise | Position / verbosity / self-preference | Average the two orderings in pairwise mode |
| **Chatbot Arena** | Human blind pairwise battles | Bradley-Terry / Elo → ranking + confidence intervals | User distribution / style preference, crowd noise | Massive vote volume + fresh dynamic questions resist contamination |

**Capability benchmarks** (ground-truth, automatic):

| Benchmark | Measures | Format | Scoring | Known pitfalls |
|---|---|---|---|---|
| **MMLU** | 57-subject multiple-choice knowledge | 4-way multiple choice | Option accuracy | Option / letter-order bias; heavily contaminated |
| **MATH** | Competition math (7 subjects / 5 difficulty levels) | Free-form solutions | Final-answer match (verifiable) | Brittle answer parsing; partly contaminated |
| **HumanEval** | Python code generation | Function completion + unit tests | pass@k (unit-test pass rate) | Only 164 problems, high variance, easy to overfit |
| **IFEval** | Verifiable instruction following | Constrained instructions (length / format) | Programmatic verification (no judge needed) | Covers only machine-checkable constraints |
| **GPQA** (Diamond) | Graduate-level physics/biology/chemistry, designed so that **skilled non-experts with unrestricted web access cannot achieve 100%** ("Google-proof") | 4-way multiple choice | Option accuracy | Domain expert accuracy ~65% (Diamond subset higher, varies by subject); options require expert-level knowledge to distinguish, very high discriminability |
| **MMLU-Pro** | MMLU strengthened — removes easy/contaminated questions, uses 10-way choice for greater difficulty | 10-way multiple choice | Option accuracy | Lower random baseline (10% vs 25%); significantly mitigates MMLU's ceiling effect |

**2024 evolutions of preference / dialogue evals**:

| **Arena-Hard** | Automatically selects **500 "model-stumping" questions** from Chatbot Arena data, scored by GPT-4 pairwise judge | Win-rate (vs. the reference) | Judge bias (fully inherited) | Spearman ~94.1% with Arena Elo rankings; fast automatic proxy for human preference; hard questions amplify discriminability |
| **MixEval** | Matches **real web user queries** to existing benchmark items, evaluates with **ground-truth answers** (not a judge), dynamically weighted aggregation | Normalized multi-dimension average | Sampling weights are sensitive; stale contamination from subset components can leak through | Multi-round weight calibration + decontamination filtering, aiming to reflect capability + preference holistically; fundamentally a ground-truth benchmark mixture, not a judge benchmark |

### 1.1a Unbiased pass@k

**Naive estimator** $1-(1-\hat p)^k$ ($\hat p=c/n$): for $k\ge2$ it **systematically underestimates** the true pass@k at any finite $n$ (Jensen's inequality: $(1-\hat p)^k$ is convex in $\hat p$, so $\mathbb{E}[(1-\hat p)^k]\ge(1-p)^k$); the bias shrinks as $n$ grows but is never zero; for $k=1$, $c/n$ is already unbiased.

**Unbiased estimator** (Chen et al., [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) §3): sample $n\ge k$ candidates per problem, $c$ pass the unit tests, then

$$\widehat{\text{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}$$

Intuition: $\binom{n-c}{k}/\binom{n}{k}$ is the probability that $k$ randomly drawn candidates are **all wrong**; $1$ minus it is "at least one correct". Numerically stable implementation (avoids large-binomial overflow):

```python
import numpy as np
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

> 📝 Standard protocol (Chen et al.): $n=200$ per problem, report $k\in\{1,10,100\}$.

### 1.2 Bradley-Terry / Elo: pairwise wins → ranking

Arena turns a massive pile of "A vs B, which is better?" pairwise votes into a total-order ranking via the **Bradley-Terry (BT)** model (Elo is its online approximation).

**BT model**: give each item $i$ a latent score $s_i$; then

$$P(i \succ j) = \sigma(s_i - s_j) = \frac{1}{1+e^{-(s_i-s_j)}}$$

Only score **differences** are identifiable (a global shift leaves probabilities unchanged), so fix an anchor (e.g. $s_0=0$). Fitting = **logistic-regression MLE**: the log-likelihood $\sum_{i,j} w_{ij}\log\sigma(s_i-s_j)$ ($w_{ij}$ = times $i$ beats $j$) is **concave** (equivalently the negative log-likelihood is a convex loss), a convex optimization with a global optimum; the solution is finite and unique when the win graph is connected.

**Core assumptions** (and where they break):
1. **Unidimensional strength**: quality is summarized by a single scalar $s_i$ → items are totally orderable;
2. **(Stochastic) transitivity**: if A tends to beat B and B tends to beat C, then A tends to beat C — **no rock-paper-scissors cyclic preferences** ($A\succ B\succ C\succ A$);
3. **Independent comparisons**: matches are mutually independent, no order/learning effects;
4. Basic BT does not model ties (Rao-Kupper and other extensions handle ties).

→ When preferences are genuinely **non-transitive cycles** (each wins on a different dimension), a single scalar Elo flattens the cycle into one order, and the ranking's "objectivity" is overstated.

**Elo = online approximation of BT**: each match does one fixed-step update on the prediction error, viewable as an online gradient update on the BT logistic loss:

```python
# Elo: the online/streaming version of Bradley-Terry, one fixed-step gradient update per match
def elo_update(r_a, r_b, score_a, K=32, scale=400):
    e_a = 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))  # BT/logistic predicted P(A wins)
    r_a = r_a + K * (score_a - e_a)                  # score_a in {1, 0.5, 0}
    r_b = r_b + K * ((1 - score_a) - (1 - e_a))
    return r_a, r_b
```

> 📝 In the batch setting, fitting BT by MLE (logistic regression) over all pairwise results is more stable than per-match Elo and yields **confidence intervals** (Arena uses BT + bootstrap for intervals).

### 1.3 SWE-bench: From Function Completion to Real-World Software Engineering

> 📎 **Cross-reference**: SWE-bench represents the evolution of code evaluation from "complete a function" to "fix a real bug". For the unbiased estimator of pass@k, see §1.1a; for the upper-bound analysis of sampling-scaling gains (coverage vs. verifier quality), see [test-time-scaling §3.3](cheatsheet-test-time-scaling-en.html). SWE-bench trajectories are also commonly used as training data for agent SFT/RL (see [long-horizon-agents](cheatsheet-long-horizon-agents-en.html)).

**SWE-bench** (Jimenez et al., [arXiv:2310.06770](https://arxiv.org/abs/2310.06770), ICLR 2024) upgrades code evaluation from "completing a function" to **"fixing a real GitHub issue"** — one of the most influential new benchmarks in 2024 code evaluation.

- **Task**: give the model a **real GitHub issue description** + the corresponding codebase **snapshot** → model generates a **patch** (`git diff`, potentially touching multiple files and lines)
- **Evaluation**: inside a Docker container, apply the patch → run the instance's associated **FAIL_TO_PASS** and **PASS_TO_PASS** tests → counted as resolved only if both pass
- **Scale**: ~2,294 issues across 12 popular Python repositories (Django, Flask, SymPy, scikit-learn, matplotlib, etc.)
- **Difficulty**: requires understanding large codebase structure, locating the bug's file/function, writing cross-file changes — far beyond single-function completion (compare HumanEval's 164 isolated function problems)

**Key subsets**:

| Subset | Description | Instance count |
|------|------|--------|
| **SWE-bench Lite** | Curated "easy-to-get-started" subset from the original SWE-bench, for rapid iteration | ~300 |
| **SWE-bench Verified** (OpenAI 2024-08, [blog](https://openai.com/index/introducing-swe-bench-verified/)) | Human review of 500 instances: confirmed issue description quality + test reliability + excluded unreproducible instances; the fixed-up subset | 500 |

**Key result trajectory** (% resolved; illustration of relative progress only, not a head-to-head comparison):
- 2024-03 (original SWE-bench): Devin (Cognition AI) ~13.86% (first to break 10%)
- 2024-08 (original SWE-bench): SWE-agent + GPT-4 combination ~18–20%
- 2024-10 (Verified subset): Claude 3.5 Sonnet ~49% (then-SOTA)
- Early 2025 (Verified subset): multi-agent systems >50%

> ⚠️ SWE-bench's core constraint is **evaluation cost** — each instance requires building a full Docker container + running the repo's test suite, potentially taking 5–30 min per instance.

**Three tiers of code evaluation** (HumanEval → BigCodeBench → SWE-bench):

| Tier | Benchmark | Files | Scoring | Capability tested |
|------|------|------|------|------|
| Single-function completion | HumanEval, MBPP | 1 file | Unit-test pass@k | Basic code generation |
| Complex func + library calls | **BigCodeBench** ([arXiv:2406.15877](https://arxiv.org/abs/2406.15877)) | Function-level + diverse library calls | Unit-test pass@k | Complex API usage |
| Real-world software engineering | **SWE-bench** | Full repo (12 real repositories) | Apply patch → run original test suite | Code understanding + localization + editing |

**Contamination-resistant complement** — **LiveCodeBench** (Jain et al., [arXiv:2403.07974](https://arxiv.org/abs/2403.07974), ICLR 2025): automatically builds code evaluations from **new** LeetCode/AtCoder/Codeforces problems (temporal isolation), covering diverse tasks including code generation, execution prediction, and test generation, updated monthly on a rolling basis, addressing HumanEval's small size and overfitting issues.

### 1.4 Agent Evaluation

> 📎 **Cross-reference**: This section focuses on agent **evaluation** (how to tell whether an agent is doing well). For agent training methods (self-evolution RL, tool-use SFT, computer-use recipes), see [long-horizon-agents](cheatsheet-long-horizon-agents-en.html).

Post-training is evolving from "single-turn dialogue" to "multi-step agents", and evaluation must follow. Agent evaluation is more complex than traditional benchmarks — it requires **simulating real interaction environments** (browsers, file systems, APIs), and success depends on **multi-step decision chains** rather than single-generation quality.

Three representative agent benchmarks:

| Benchmark | Environment | Typical tasks | Scoring method | Core limitation |
|------|------|------|------|------|
| **WebArena** (Zhou et al., [arXiv:2307.13854](https://arxiv.org/abs/2307.13854), ICLR 2024) | Self-built websites (e-commerce/forum/map/CMS) | "Find the highest-rated review on a given product and reply" | Programmatic verification (page state/URL/element presence) | Self-built environment has limited coverage, not the real internet |
| **TAU-bench** (Yao et al., [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)) | Conversational tool-agent-user interaction (airline/retail/finance DB) | "Help the user reschedule the flight to Friday night, window seat, mileage upgrade" | Database state comparison + pass^k | Multi-turn dialogue with simulated users; user behavior choices shape evaluation signal |
| **OSWorld** (Xie et al., [arXiv:2404.07972](https://arxiv.org/abs/2404.07972), NeurIPS 2024 D&B) | Real OS VMs (Ubuntu/Windows/macOS) | "Create a table in LibreOffice → export PDF → email it" | Screen state / file system / window state checks | Most realistic but most expensive — each task needs a VM snapshot + VNC |

**Three distinctive dimensions of agent evaluation** (vs. traditional benchmarks):

1. **Step-count dependence**: the result is the success rate "given a maximum step budget" — **accuracy-vs-steps curves** carry more information than a single-point number (analogous to test-time scaling's accuracy-budget curves). The same model's success rate at 5 steps vs. 30 steps can differ by a factor of several.
2. **Non-reproducibility**: environment state and tool-call results can vary with network/time — needs Docker/VM snapshots to lock the environment for reproducibility.
3. **Trajectory quality ≠ success**: a trajectory that "took 20 detour steps to succeed" and one that "finished directly in 3 steps" are equivalent on success rate, but differ by multiples in efficiency and cost — must report **step / token efficiency**.

> ⚠️ The biggest unresolved problem in agent evaluation: **no consistent cross-benchmark ranking** — an agent that is SOTA on WebArena may be mediocre on OSWorld, and vice versa. This reflects the fragmented state of agent evaluation; there is not yet an "Arena for agents".

### 1.5 Long-Context Evaluation

"Claiming 128K/1M support" and "actually using it well" are two different things. Long-context evaluation needs to answer three questions: ① **Can it find it?** (retrieval); ② **Can it understand it?** (reasoning/synthesis); ③ **Can it use it?** (downstream tasks).

| Benchmark | What it measures | Context length | Core finding |
|------|------|------|------|
| **Needle-in-a-Haystack (NIAH)** (Kamradt 2023, [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)) | Insert a single fact at a **random position** in a long document → ask the model about that fact | Extensible to any length | Most models score near 100% for "needles" at the beginning or end, but **drop significantly in the middle** (U-shaped curve) — position bias also exists in long-context settings |
| **RULER** (Hsieh et al., [arXiv:2404.06654](https://arxiv.org/abs/2404.06654), COLM 2024) | **Multi-needle** retrieval + aggregation (answers across multiple needles must be synthesized) + multi-hop QA | 4K–128K+ | High scores on single-needle NIAH **do not equal real capability**; under multi-needle + multi-hop tests, many models that "support 128K" have effective context far below the claimed value |
| **LongBench** (Bai et al., [arXiv:2308.14508](https://arxiv.org/abs/2308.14508), ACL 2024) | 21 datasets across 6 task categories (single/multi-doc QA, summarization, code, few-shot learning, synthetic, dialogue) | 1K–18K bilingual (CN+EN) | Long-context ability across different tasks **is not a single dimension** — code long-context and summarization long-context are two types of ability, not summarizable by a single scalar |

**Key connections between long-context evaluation and post-training**:
- **A special form of alignment tax**: long-context ability often **degrades** during RLHF/DPO alignment training — safety/dialogue training data is mostly short-context, causing silent drops in long-context recall. This must be explicitly checked during evaluation.
- **The judge's own long-context capability**: if LLM-as-judge is used to evaluate multi-turn or other long-context outputs, whether the **judge itself** can accurately judge in long contexts is also a factor (§2.2 judge-human agreement does not directly answer this).

> ⚠️ "Supports 128K" ≠ "can use all of 128K well". At minimum, cross-validate with RULER (multi-needle) and LongBench (multi-task) — do not rely on single-needle NIAH heatmaps alone.

## 2. LLM-as-Judge: How to Use It + Biases

> 📎 **Cross-reference**: This section focuses on LLM-as-Judge from the perspective of **evaluation practice** (how to select a judge, operational details, benchmark applications). For how LLM-as-Judge biases affect RM training and reward hacking when used as a **RLHF training signal**, see `cheatsheet-reward-modeling-eval-en.html §5.2`.

Use a strong model as a judge to score responses or do pairwise comparisons. **Cheaper and faster, but has systematic biases**:
- **Position bias**: tends to favor the "first" answer → mitigation: **evaluate both orderings and average**.
- **Verbosity / length bias**: tends to favor longer answers → mitigation: length debiasing, controlling response length.
- **Self-preference**: the judge favors outputs that stylistically resemble its own.
- **Format / style bias**: markdown formatting and confident tone are rated more favorably.
General mitigations: **reference-guided** judging (provide a reference answer), **rubrics / scoring scales**, **multi-judge voting**, and **calibration against human annotations**.

**Formalizing position debiasing:** let the judge's preference on an ordered pair be $J(a,b)\in\{A,B\}$. With no position bias, $J(a,b)$ and $J(b,a)$ should pick the **same** answer (consistent). Debiased rule: count a win only if both orders pick the same answer, else call it a **tie** — making position bias an explicit tie rather than letting it leak into the win-rate. Note: AlpacaEval uses a single fixed ordering by default (model response first); MT-Bench pairwise mode uses swap augmentation (average the two orderings) — not all tools apply order averaging.

### 2.1 Position-debiased judge harness (code)

```python
# LLM-as-judge: pairwise comparison + position debiasing (order-swap) harness.
# In practice judge() calls a strong model and parses its verdict; here a stub shows the protocol.

def judge_debiased(question, ans1, ans2, judge):
    """Judge each ordering once to cancel position bias.
    judge(q, A, B) returns 'A' or 'B' (which position's answer is better).
    Returns 'ans1' / 'ans2' / 'tie' (orders disagree -> tie)."""
    v1 = judge(question, ans1, ans2)            # order (A=ans1, B=ans2)
    v2 = judge(question, ans2, ans1)            # swapped (A=ans2, B=ans1)
    pick1 = 'ans1' if v1 == 'A' else 'ans2'     # answer actually chosen, call 1
    pick2 = 'ans2' if v2 == 'A' else 'ans1'     # in the swapped call A=ans2, so v2=='A' means ans2 was picked
    return pick1 if pick1 == pick2 else 'tie'   # count only if consistent

def win_rate(questions, model_answers, ref_answers, judge):
    """Debiased win-rate of model vs ref; tie counts 0.5."""
    s = 0.0
    for q, m, r in zip(questions, model_answers, ref_answers):
        out = judge_debiased(q, m, r, judge)    # ans1=model, ans2=ref
        s += 1.0 if out == 'ans1' else (0.5 if out == 'tie' else 0.0)
    return s / len(questions)

# --- Demo: an extreme "always pick the first" position-biased judge is exposed as a tie ---
def biased_judge(q, a, b):
    return 'A'                                  # extreme position bias
print(judge_debiased("q", "model", "ref", biased_judge))   # -> 'tie'
print("win_rate:", win_rate(["q"], ["model"], ["ref"], biased_judge))  # -> 0.5
```

### 2.2 Measuring judge–human agreement

"Calibration / agreement with humans" recurs throughout (the §1 RM row, the mitigations above, Q31), but **raw agreement alone overstates "the real agreement after removing chance"**: when one label dominates, two annotators agree by pure guessing with high probability. **Cohen's κ** removes this chance agreement:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- $p_o$: observed agreement (the fraction on which the two agree);
- $p_e$: **chance** agreement $=\sum_c p^{(1)}_c\,p^{(2)}_c$ (sum over categories of the product of the two annotators' marginal proportions).

$\kappa=1$ is perfect, $\kappa=0$ means agreement is only at the level of "guessing independently from each annotator's own marginals", $\kappa<0$ is worse than guessing. Intuition: if both annotators label **independently** with 90/10 marginals, raw agreement is about $0.9^2+0.1^2=0.82$ — looks high, yet $\kappa\approx0$. **So report κ, not just raw agreement, for judge–human consistency.**

```python
import numpy as np
def cohens_kappa(labels_a, labels_b):
    cats = sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(cats)}
    n, K = len(labels_a), len(cats)
    conf = np.zeros((K, K))
    for a, b in zip(labels_a, labels_b):
        conf[idx[a], idx[b]] += 1
    p_o = np.trace(conf) / n                            # observed agreement
    p_e = (conf.sum(0) * conf.sum(1)).sum() / n ** 2    # chance agreement
    return (p_o - p_e) / (1 - p_e)
```

> More than two annotators (nominal categories): use **Fleiss' κ**; ordinal scores: use **weighted κ** or **Krippendorff's α** (supports ordinal distances and missing data).

### 2.3 Length-controlled win-rate: the regression form

The §1 table lists AlpacaEval 2.0's debiasing as "regress out length" — concretely (Dubois et al., [arXiv:2404.04475](https://arxiv.org/abs/2404.04475)) you fit a **generalized linear model** predicting the judge's preference, with the **length difference** as an explicit term (simplified form):

$$\text{logit}\,P(\text{model}\succ\text{ref}) = \theta_m + \gamma\cdot\Delta_{\text{len}} + \dots$$

where $\Delta_{\text{len}}$ is the two answers' length difference. To report, set the length term $\Delta_{\text{len}}=0$ and take the expectation over the instruction distribution, giving the **length-controlled win-rate** — intuitively "the win-rate the model should have if both answers were the same length", removing the "length association" the GLM models.

⚠️ Note: it removes the **length association the model fits**, so genuine quality signal correlated with length may be removed along with it, and unmodeled style effects are not removed; empirically it markedly improves the Spearman correlation with Chatbot Arena (human Elo) rankings.

## 3. Data Contamination

The training set contains test-set examples → inflated scores that do not reflect generalization.

### 3.1 Types of Contamination

| Type | Description |
|------|------|
| **Direct overlap** | Training data contains the exact questions or answers from the evaluation set |
| **Temporal leakage** | Training data cutoff is later than the evaluation set creation date; the model has "seen" test-period content |
| **Near-overlap** | Paraphrased or translated versions of test questions appear in the training set, undetected by string matching |
| **Membership inference contamination** | Training set does not contain the exact questions, but contains highly related in-distribution samples, causing score inflation |

### 3.2 Detection Methods

- **n-gram / substring overlap**: compute n-gram Jaccard similarity between the training set and the evaluation set; filter above a threshold
- **Min-k% Prob (MIA)**: Membership Inference Attack — compute the minimum k% token probability of a test sample under the model; member samples typically have higher minimum token probabilities than non-members (per Shi et al. 2024 and related work; exact effectiveness varies by setting)
- **Canary strings**: insert random strings into training data; if the model can "recall" them, the data was indeed used for training
- **Anomalously low perplexity**: compute perplexity on test samples; if significantly lower than comparable new samples, this suggests the model may have seen them
- **Score drop after paraphrase**: semantically rephrase test questions; if the model scores higher on originals than on rephrased versions, this suggests memorization rather than generalization

### 3.3 n-gram Deduplication

- Deduplicating within training data (exact + near-dedup) reduces memorization risk and limits contamination propagation to benchmarks
- Common tools: MinHash LSH, suffix array deduplication (per Lee et al. 2022 and related work)
- ⚠️ Deduplication is not the same as decontamination: deduplication targets repeated entries within the training set; decontamination (去污染) specifically targets overlap between the training set and the evaluation set

### 3.4 Contamination-Resistant Benchmark Design

- **Dynamic / private evaluation sets**: use new questions each evaluation round (e.g., Chatbot Arena's continuous collection, LiveBench's monthly updates)
- **Programmatically generated questions**: automatically generate questions with verifiable answers (math, code); novelty is guaranteed by the generation procedure
- **Temporal isolation**: explicitly report the training data cutoff date and the evaluation set creation date, so readers can assess temporal leakage risk
- **Contamination audit reports**: publicly disclose the decontamination pipeline and filtering rates in papers/reports to increase credibility

## 4. Pitfalls in Preference Evaluation

- **Goodhart's Law**: once a benchmark becomes an optimization target, it ceases to be a good measure (leaderboard gaming ≠ genuine improvement).
- **Prompt sensitivity**: the same model can score very differently with different prompt templates → fix evaluation prompts, report variance.
- **Judge ≠ ground truth**: LLM-judge win-rate only reflects "what another model thinks is good," not human preference or real-world utility.
- **Single-pass accuracy vs. compute budget**: for reasoning models, report accuracy under a given test-time compute budget, not just single-pass accuracy.

## 5. A Practical Evaluation Protocol (Post-Training)

1. **Capability**: GSM8K/MATH (math), HumanEval → SWE-bench (code), MMLU/MMLU-Pro/GPQA (knowledge), IFEval (instruction following).
2. **Alignment quality**: AlpacaEval / MT-Bench / Arena-Hard (judge, with position debiasing); Chatbot Arena when necessary.
3. **Safety / refusal**: harmful-prompt refusal rate, over-refusal rate.
4. **Agent**: WebArena / TAU-bench (multi-step interaction, report accuracy-vs-steps curves).
5. **Long-context**: RULER (multi-needle retrieval) + LongBench (multi-task understanding); don't rely on single-needle NIAH alone.
6. **Regression**: compare against baselines to confirm no dimensions degraded (alignment tax).
7. **Contamination audit** + **multiple seeds/prompts** to report variance.

---

## Stratified Follow-ups

### L1 Basics
1. Why is evaluation considered the bottleneck of post-training? What does each of capability benchmarks and preference evaluation measure?
2. What is LLM-as-judge? What known biases does it have?
3. What is data contamination? Why does it make scores unreliable?
4. What is the fundamental difference between capability benchmarks and preference evaluation? What is each suited to measure?
5. How is AlpacaEval's win-rate computed? Against what reference is the rate measured?
6. How do MT-Bench and Chatbot Arena differ in their evaluation signal (who does the scoring)?
7. What does pass@k mean? Why do code evals often use it instead of single-pass accuracy?
8. What is alignment tax? Why specifically check for regressions after post-training?
9. Why fix the prompt template and report variance over multiple seeds?
10. How do IFEval and MMLU fundamentally differ in *how* they score (programmatic verification vs. option accuracy)?

### L2 Intermediate
11. How is position bias mitigated? Why does "evaluate both orderings and average" work?
12. How do AlpacaEval / MT-Bench / Chatbot Arena differ in their evaluation signals (automatic judge vs. human Elo)?
13. How do you detect whether training data has contaminated a given evaluation set?
14. Why is verbosity bias so stubborn? How does length-controlled win-rate factor out the length effect?
15. The order-swap rule calls a tie when the two orderings disagree — compared to majority voting, what robustness advantage does this have?
16. Chatbot Arena uses Bradley-Terry / Elo to turn pairwise outcomes into rankings — what is the core assumption of this model?
17. List several contamination-detection methods (n-gram / Min-k% / canary / paraphrase-drop) — what does each actually catch?
18. Why is "dedup ≠ decontamination"?
19. Why is self-preference bias especially dangerous when using a homologous model as the judge?
20. Why does a high judge win-rate not equal stronger human preference? Where do the two systematically diverge?

### L3 Deep Dive
21. How does Goodhart's Law manifest in leaderboard gaming? How do you design "gaming-resistant" evaluations?
22. How is a reward model evaluated (the RewardBench approach)? What is the relationship between RM evaluation and final policy performance?
23. For reasoning models, why should evaluation shift from "single-pass accuracy" to "accuracy under a compute budget"? What does this demand of the evaluation protocol?
24. If online metrics (user retention) conflict with offline evaluation (judge win-rate), which do you trust, and how do you investigate the discrepancy?
25. When summarizing a model with a single scalar score, how can it mask Pareto-style regressions like "capability ↑ but safety ↓"? How should evaluation design expose this?

## 2024 Frontier Supplement

> The following questions cover the 2024 evaluation frontiers added to this page (SWE-bench, Agent Evaluation, Long-Context Evaluation, GPQA/MMLU-Pro).

<details>
<summary>Qa. Why is GPQA called "Google-proof"? How does it fundamentally differ from MMLU / MMLU-Pro in discriminability?</summary>

    **A:** GPQA (Graduate-Level Google-Proof Q&A) questions are **graduate-level** physics/biology/chemistry multiple-choice questions designed by domain experts with the explicit goal that "even a skilled non-expert with unrestricted web access cannot achieve 100% within 30 minutes" (hence "Google-proof"). The fundamental difference from MMLU: ① MMLU's human ceiling is high (~90%+), so discriminability is low (strong models cluster near the ceiling); ② GPQA's domain expert accuracy is ~65% (varies by subject for Diamond), spreading out model differences. MMLU-Pro mitigates MMLU's ceiling effect via 10-way choice (rather than 4-way) + removing overfitted/contaminated questions — it is the stepping stone between MMLU and GPQA.

    **Follow-up:** Why do reasoning models (o1/R1) often choose GPQA Diamond as a core evaluation?
    Because GPQA requires deep domain knowledge + reasoning (cannot be answered directly by retrieval alone), making it natural for measuring "whether longer reasoning can compensate for knowledge gaps" — exactly the selling point of reasoning models.

</details>

<details>
<summary>Qb. What is the essential difference between the code capability measured by SWE-bench vs. HumanEval? Why is pass@k alone insufficient to describe SWE-bench performance?</summary>

    **A:** HumanEval measures **single-function completion** (given signature + docstring → write the function body); SWE-bench measures **real-world software engineering** (given a GitHub issue + full codebase → locate the bug → cross-file patch → pass the instance's FAIL_TO_PASS and PASS_TO_PASS tests). Essential difference: ① the former tests "generation"; the latter tests the compound capability of "understanding + localization + editing"; ② the former has isolated problems; the latter requires understanding cross-file dependencies in a large codebase. pass@k is insufficient to describe SWE-bench performance because each SWE-bench problem has only one correct patch (unlike code generation where you can sample k candidates and check whether one is correct) — pass@k's "at least one of k is correct" logic does not apply; SWE-bench uses **% resolved** (associated tests all pass), a single-submission pass/fail.

    **Follow-up:** Why is SWE-bench Verified important?
    The original SWE-bench had some issues with unclear descriptions or unreliable tests — models could fail due to "description ambiguity" or "bugs in the tests themselves" rather than lack of capability. The Verified subset (released by OpenAI 2024-08) underwent human review of 500 instances to exclude such noise, making the evaluation signal cleaner.

</details>

<details>
<summary>Qc. What are the three distinctive challenges of agent evaluation? Why is "success rate" alone insufficient as a metric?</summary>

    **A:** ① **Step-count dependence**: the same model's success rate at a 5-step vs. 30-step budget can differ by multiples — you must report accuracy-vs-steps curves, not single-point numbers. ② **Non-reproducibility**: network/time/environment-state changes cause the same agent to produce different results on two runs — needs Docker/VM snapshots to lock the environment. ③ **Trajectory quality ≠ success**: a 20-detour-step success and a 3-direct-step success are equivalent on success rate, but totally different on efficiency and cost — must also report step / token efficiency. "Success rate" is insufficient because it only answers "did it eventually succeed", not "at what cost" and "under what step budget" — the latter two are more important for real deployment.

    **Follow-up:** What is the biggest unresolved problem in agent evaluation today?
    The lack of consistent cross-benchmark ranking — an agent that is SOTA on WebArena may be mediocre on OSWorld (and vice versa). This reflects the fragmented state of agent evaluation; there is no "Chatbot Arena for agents" yet.

</details>

<details>
<summary>Qd. Why does a good single-needle NIAH heatmap ≠ strong long-context capability? What gaps do RULER and LongBench respectively fill?</summary>

    **A:** Single-needle NIAH only tests "can you find one fact in a pile of irrelevant text" — this is a **necessary but insufficient** condition for long-context capability. RULER fills the "multi-needle retrieval + multi-hop aggregation" gap (real usage often requires synthesizing information from multiple locations), exposing many models that score perfectly on single-needle NIAH but collapse on multi-needle. LongBench fills the "multi-task diversity" gap (21 datasets across 6 categories, spanning summarization, code, few-shot learning, etc.), revealing that long-context capability is **not a single dimension** — a model may be strong at code long-context but weak at summarization long-context. The relationship: NIAH = basic retrieval probe, RULER = stress test, LongBench = ecological validity (real-world utility).

    **Follow-up:** How to check whether post-training has damaged long-context capability?
    Run RULER + LongBench once before and once after post-training; RLHF/DPO training data is mostly short-context, so silent degradation of long-context capability is an often-overlooked form of alignment tax.

</details>

<details>
<summary>Qe. How do Arena-Hard and MixEval respectively address the problems of "benchmark overfitting/contamination" and "single-benchmark narrowness"? What limitations remain?</summary>

    **A:** **Arena-Hard** automatically selects 500 "model-stumping" questions from Chatbot Arena's massive human-vote data and uses a GPT-4 pairwise judge — it inherits Arena's contamination resistance (dynamic question pool) while compressing evaluation from "crowd voting" into an automatic pipeline, with Spearman ~94.1% against human Elo. Limitations: it fully inherits judge bias, and a fixed set of 500 questions can still be targeted for optimization. **MixEval** matches real web user queries to existing benchmark items, evaluates with ground-truth answers (not a judge), and uses dynamic weighted aggregation — fundamentally a "multi-source benchmark mixture" rather than a judge benchmark. Limitations: sampling weights are sensitive (tuning weights can "change the ranking"), and contamination from stale subset components can leak through.

    **Follow-up:** How does this relate to Goodhart's Law in §4?
    Both Arena-Hard and MixEval are anti-gaming mechanisms designed under the premise that "once a benchmark becomes an optimization target, it ceases to be a good measure" — the former relies on dynamic + hard-question filtering, the latter on multi-source mixing to dilute any single benchmark's weight.

</details>


## Extended L3

<details>
<summary>Q26. When using LLM-as-judge to evaluate multi-turn, long-context conversations, what are the common difficulties? What methodological improvements exist?</summary>

    The core challenge in evaluating multi-turn conversations is that the judge model tends to exhibit **context forgetting** or **local bias** — focusing only on the quality of the most recent one or two turns while ignoring overall conversational coherence and task completion. A key improvement is designing **process-oriented rubrics** that explicitly require evaluating each turn's contribution to the final goal, and introducing a **segment summarization** mechanism that forces the judge to summarize before scoring, thereby partially mitigating its short-sightedness.
    **Follow-up**: Beyond improving the rubric, can the evaluation protocol itself be changed to reduce judging difficulty — for example, decomposing it into a series of simpler subtask evaluations?

</details>

<details>
<summary>Q27. What are the main limitations of LLM-as-judge in assessing factual accuracy and logical soundness? How can they be mitigated?</summary>

    The main limitations are that the judge model's own **knowledge boundary** and **reasoning flaws** can lead to incorrect verdicts. It may fail to detect factual errors, or mistakenly accept an answer with logical gaps as sound. Mitigation typically involves a **hybrid evaluation** approach: for fact-checking, combine **retrieval-augmented verification** — retrieve authoritative information first, then compare; for logical evaluation, attempt to use **formal verification** tools or design **step-by-step verification prompts** specifically targeting reasoning chains.
    **Follow-up**: Given limited resources, which judge capabilities (breadth of knowledge, reasoning ability, tool use) should be prioritized for improvement to most effectively increase evaluation accuracy?

</details>

<details>
<summary>Q28. How do you evaluate a model's "emergent capabilities"? How does this fundamentally differ from evaluating conventional capabilities?</summary>

    The key difference in evaluating emergent capabilities lies in their **unpredictability** and **non-smoothness**. Conventional capabilities typically improve predictably on a benchmark as model scale or training data increases. Emergent capabilities, by contrast, appear suddenly past some threshold and are often not directly reflected in standard benchmarks. As a result, evaluation methods must shift from **fixed test sets** to **open-ended, programmatically generated probe tasks**, and must focus on detecting **behavioral pattern shifts** when the model faces entirely novel, complex task combinations.
    **Follow-up**: Can one design an evaluation framework that not only discovers emergent capabilities but also, to some degree, predicts the conditions under which they will appear?

</details>

<details>
<summary>Q29. Why does calibrating the confidence of LLM-as-judge outputs matter? How is it achieved in practice?</summary>

    Calibrating judge output confidence gives its scores or comparison results **interpretable probabilistic meaning**. For example, when a judge says "90% confident that A is better than B," that number should, in the long run, approximate the true frequency with which A is actually preferred over B. In practice, calibration requires a **human-annotated calibration set**. By repeatedly evaluating the judge on this set, one can analyze the distribution of its scoring deviations from human consensus, then apply **post-hoc calibration algorithms** (such as Platt Scaling or Isotonic Regression) to adjust raw scores so they better match the statistical patterns of human judgment.
    **Follow-up**: If the human-annotated data used for calibration is itself low-quality or very small in scale, what effect does this have on the calibrated judge? What are the alternatives?

</details>

<details>
<summary>Q30. When evaluating the safety of conversational models, why is "over-refusal" an important metric? How do you analyze the trade-off it forms with the harmful-prompt refusal rate?</summary>

    Over-refusal measures the degree to which a model incorrectly refuses **benign or borderline queries**, directly affecting user experience and model **utility**. A model with a very high over-refusal rate may be safe but becomes effectively useless. Analyzing this trade-off cannot simply pursue Pareto optimality across both metrics; instead, **risk tiering** should be introduced. Categorize harmful prompts by severity and set different refusal strictness thresholds for each tier. Evaluation should separately report refusal rates for each tier and use **cost-sensitive analysis** to assess the model's balance between overall risk exposure and user experience loss.
    **Follow-up**: How do you construct a high-quality adversarial safety evaluation set that can automatically generate prompts spanning various risk tiers and "gray area" cases?

</details>

<details>
<summary>Q31. How do you conduct "meta-evaluation" — that is, how do you judge whether an evaluation benchmark or an LLM-as-judge is itself valid and reliable?</summary>

    Meta-evaluation of a benchmark primarily examines its **discriminability** (can it effectively distinguish models at different capability levels), **robustness** (is it sensitive to minor prompt changes), and **ecological validity** (does the capability it measures relate to real-world needs). Meta-evaluation of an LLM-as-judge focuses on its **agreement** with human judgments (e.g., Cohen's Kappa) and its **fairness** across different subgroups. A key method is **cross-validation**: have multiple distinct, high-quality judges (or humans) evaluate the same set of data, and check whether the target judge or benchmark agrees with the consensus.
    **Follow-up**: After discovering that a widely used benchmark likely has serious biases or is outdated, what responsibilities and feasible actions does a researcher have to promote its iteration or warn the community?

</details>

<details>
<summary>Q32. In domain-specific settings (e.g., medical, legal), what unique challenges arise for general-purpose LLM-as-judge evaluation? What are the key steps in building a domain-expert evaluation pipeline?</summary>

    The core challenges are the **domain knowledge barrier** and the **specialized evaluation criteria** required. A general-purpose judge may not understand the nuances of domain terminology or the rigor of professional logic. Key steps in building a domain evaluation pipeline are, first, **co-defining evaluation dimensions** with domain experts — jointly determining dimensions such as "conservatism of medical advice" or "accuracy of legal citations." Second, **building a domain gold standard** — a set of authoritative reference answers or judgments annotated by experts. Finally, designing a **human-AI collaborative evaluation process** in which the AI judge handles initial screening while human experts handle edge-case review and final adjudication.
    **Follow-up**: When domain experts themselves disagree on the same response (e.g., physicians from different schools of thought), how do you design a system that accommodates reasonable expert disagreement while still enabling effective automated evaluation?

</details>

## §A Key Papers Timeline

- **2020-09 · MMLU** — Hendrycks et al., ICLR 2021. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) — 57-subject four-way multiple-choice knowledge eval; established the "broad-coverage MCQ" paradigm. Its main weaknesses are option/letter-order bias and heavy later-stage contamination.

- **2021-03 · MATH** — Hendrycks et al., NeurIPS 2021. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) — Competition mathematics (7 subjects / 5 difficulty levels) with free-form solutions and verifiable final-answer matching; became the standard math-reasoning benchmark.

- **2021-07 · HumanEval** — Chen et al., arXiv preprint. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) — 164 function-completion problems with hidden unit tests; introduced pass@k as the code-generation metric. Small size means high variance.

- **2021-10 · GSM8K** — Cobbe et al., arXiv preprint. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) — 8.5K grade-school math word problems plus a trained verifier for reranking; grounded the chain-of-thought + answer-verification paradigm for eval and training.

- **2023-05 · AlpacaFarm** — Dubois et al., NeurIPS 2023. [arXiv:2305.14387](https://arxiv.org/abs/2305.14387) — Uses an LLM-judge to cheaply simulate RLHF human-preference labeling, making win-rate-style alignment evaluation reproducible and iterable.

- **2023-06 · MT-Bench / LLM-as-a-Judge** — Zheng et al., NeurIPS 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) — Multi-turn dialogue judge scoring plus systematic quantification of position/verbosity/self-preference bias; set the methodological baseline for LLM-as-judge.

- **2023-07 · WebArena** — Zhou et al., ICLR 2024. [arXiv:2307.13854](https://arxiv.org/abs/2307.13854) — Self-built web environment (e-commerce/forum/map/CMS) for agent evaluation with programmatic verification of page state; one of the earliest systematic agent benchmarks.

- **2023-08 · LongBench** — Bai et al., ACL 2024. [arXiv:2308.14508](https://arxiv.org/abs/2308.14508) — 21 datasets across 6 task categories (single/multi-doc QA, summarization, code, few-shot, synthetic, dialogue) in a bilingual (CN+EN) benchmark, revealing that long-context ability is not a single dimension.

- **2023-10 · Min-K% Prob** — Shi et al., ICLR 2024. [arXiv:2310.16789](https://arxiv.org/abs/2310.16789) — Uses a sample's lowest k% token probabilities for pretraining-data membership inference, probing whether an eval set has been "seen" (contamination probe).

- **2023-10 · SWE-bench** — Jimenez et al., ICLR 2024. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) — Evaluates code capability via real GitHub issues → patch → run FAIL_TO_PASS+PASS_TO_PASS tests, upgrading code evaluation from "function completion" to "real-world software engineering"; one of the most influential new benchmarks of 2024.

- **2023-11 · IFEval** — Zhou et al., arXiv preprint. [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) — Instruction-following eval built on programmatically verifiable instructions (word count / format), requiring no judge and objectively recomputable.

- **2023-11 · GPQA** — Rein et al., NeurIPS 2024 D&B. [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) — Graduate-level "Google-proof" physics/biology/chemistry multiple-choice questions; domain experts score only 65–74%, providing a high-discriminability evaluation target for reasoning models.

- **2024-03 · Chatbot Arena** — Chiang et al., ICML 2024. [arXiv:2403.04132](https://arxiv.org/abs/2403.04132) — Human blind pairwise battles converted to rankings via Bradley-Terry/Elo; massive votes plus fresh prompts make it the contamination-resistant human-preference gold standard.

- **2024-03 · RewardBench** — Lambert et al., arXiv preprint. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787) — First systematic RM evaluation benchmark (chat / safety / reasoning categories), measuring reward-model quality via pairwise-preference accuracy.

- **2024-04 · Length-Controlled AlpacaEval** — Dubois et al., COLM 2024. [arXiv:2404.04475](https://arxiv.org/abs/2404.04475) — Regresses "length" out of win-rate, substantially reducing verbosity bias and improving correlation with human rankings.

- **2024-04 · OSWorld** — Xie et al., NeurIPS 2024 D&B. [arXiv:2404.07972](https://arxiv.org/abs/2404.07972) — Evaluates agents in real OS VMs (LibreOffice/browser/file management), verifying task completion via screen state + file system changes; the most realistic agent benchmark but extremely expensive.

- **2024-06 · LiveBench** — White et al., ICLR 2025. [arXiv:2406.19314](https://arxiv.org/abs/2406.19314) — Monthly rolling updates with objective, verifiable answers for contamination-resistant evaluation, reducing inflation from data leakage.

- **2024-06 · MMLU-Pro** — Wang et al., arXiv preprint. [arXiv:2406.01574](https://arxiv.org/abs/2406.01574) — MMLU strengthened: 10-way choice + removal of overfitted/contaminated questions, addressing MMLU's ceiling effect with significantly improved discriminability among strong models.

- **2024-06 · Arena-Hard** — Li et al., arXiv preprint. [arXiv:2406.11939](https://arxiv.org/abs/2406.11939) — Automatically selects 500 hard questions from Chatbot Arena data for GPT-4 pairwise judging, with Spearman ~0.9 against human Elo; fast automatic proxy for human preference.

- **2024-06 · MixEval** — Ni et al., NeurIPS 2024. [arXiv:2406.06565](https://arxiv.org/abs/2406.06565) — Mixed sampling from multiple benchmarks with dynamic weighted aggregation, using multi-source mixing to dilute individual benchmark weight against narrowness + contamination.

- **2024-04 · RULER** — Hsieh et al., COLM 2024. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654) — Multi-needle retrieval + multi-hop synthesis as a long-context stress test, exposing the effective-context inadequacy masked by single-needle NIAH heatmaps.

- **2024-06 · BigCodeBench** — Zhuo et al., arXiv preprint. [arXiv:2406.15877](https://arxiv.org/abs/2406.15877) — Function-level code evaluation with diverse library calls, filling the evaluation gap between HumanEval (simple functions) and SWE-bench (full repo).

- **2024-06 · TAU-bench** — Yao et al., arXiv preprint. [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) — Conversational tool-agent-user interaction benchmark, verifying agent task completion via database state comparison; another important dimension of agent evaluation.

- **2024-08 · SWE-bench Verified** — OpenAI, [blog](https://openai.com/index/introducing-swe-bench-verified/). — Human review of 500 SWE-bench instances: confirmed issue description quality + test reliability, excluding noise to produce a cleaner evaluation signal.

- **2024-08 · LiveCodeBench** — Jain et al., ICLR 2025. [arXiv:2403.07974](https://arxiv.org/abs/2403.07974) — Automatically constructs code evaluations from new LeetCode/AtCoder/Codeforces problems, with monthly rolling updates for temporal isolation, addressing HumanEval's small size and overfitting.
