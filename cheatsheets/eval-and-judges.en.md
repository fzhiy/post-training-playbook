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

1. **Capability**: GSM8K/MATH (math), HumanEval (code), MMLU (knowledge), IFEval (instruction following).
2. **Alignment quality**: AlpacaEval / MT-Bench (judge, with position debiasing); Chatbot Arena when necessary.
3. **Safety / refusal**: harmful-prompt refusal rate, over-refusal rate.
4. **Regression**: compare against baselines to confirm no dimensions degraded (alignment tax).
5. **Contamination audit** + **multiple seeds/prompts** to report variance.

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

- **2023-10 · Min-K% Prob** — Shi et al., ICLR 2024. [arXiv:2310.16789](https://arxiv.org/abs/2310.16789) — Uses a sample's lowest k% token probabilities for pretraining-data membership inference, probing whether an eval set has been "seen" (contamination probe).

- **2023-11 · IFEval** — Zhou et al., arXiv preprint. [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) — Instruction-following eval built on programmatically verifiable instructions (word count / format), requiring no judge and objectively recomputable.

- **2024-03 · Chatbot Arena** — Chiang et al., ICML 2024. [arXiv:2403.04132](https://arxiv.org/abs/2403.04132) — Human blind pairwise battles converted to rankings via Bradley-Terry/Elo; massive votes plus fresh prompts make it the contamination-resistant human-preference gold standard.

- **2024-03 · RewardBench** — Lambert et al., arXiv preprint. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787) — First systematic RM evaluation benchmark (chat / safety / reasoning categories), measuring reward-model quality via pairwise-preference accuracy.

- **2024-04 · Length-Controlled AlpacaEval** — Dubois et al., COLM 2024. [arXiv:2404.04475](https://arxiv.org/abs/2404.04475) — Regresses "length" out of win-rate, substantially reducing verbosity bias and improving correlation with human rankings.

- **2024-06 · LiveBench** — White et al., ICLR 2025. [arXiv:2406.19314](https://arxiv.org/abs/2406.19314) — Monthly rolling updates with objective, verifiable answers for contamination-resistant evaluation, reducing inflation from data leakage.
