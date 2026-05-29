# Evaluation & LLM-as-Judge

> In post-training, **evaluation** is often the real bottleneck: training can run, but whether the model actually improved — or where it regressed — is entirely determined by evaluation. This page covers how to evaluate an aligned model and the pitfalls of various evaluation approaches.
> ⚠️ No concrete scores are listed here (they go stale quickly and are easy to misremember); for specific numbers, refer to official benchmark/leaderboard sources.

## 1. Three Families of Evaluation

| Type | Measures | Examples |
|---|---|---|
| **Capability benchmarks** (automatic, ground-truth answers) | Knowledge / reasoning / code | MMLU, GSM8K, MATH, HumanEval/MBPP, BBH, IFEval (instruction following) |
| **Preference / dialogue evaluation** (judge scoring) | Subjective quality of responses | AlpacaEval (LLM-judge win-rate), MT-Bench (multi-turn, judge scores), Chatbot Arena (human pairwise → Elo) |
| **Reward model evaluation** | Whether the RM aligns with human preferences | RewardBench (chat / safety / reasoning categories, etc.), agreement rate with human annotations |

## 2. LLM-as-Judge: How to Use It + Biases

> 📎 **Cross-reference**: This section focuses on LLM-as-Judge from the perspective of **evaluation practice** (how to select a judge, operational details, benchmark applications). For how LLM-as-Judge biases affect RM training and reward hacking when used as a **RLHF training signal**, see `cheatsheet-reward-modeling-eval-en.html §5.2`.

Use a strong model as a judge to score responses or do pairwise comparisons. **Cheaper and faster, but has systematic biases**:
- **Position bias**: tends to favor the "first" answer → mitigation: **evaluate both orderings and average**.
- **Verbosity / length bias**: tends to favor longer answers → mitigation: length debiasing, controlling response length.
- **Self-preference**: the judge favors outputs that stylistically resemble its own.
- **Format / style bias**: markdown formatting and confident tone are rated more favorably.
General mitigations: **reference-guided** judging (provide a reference answer), **rubrics / scoring scales**, **multi-judge voting**, and **calibration against human annotations**.

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

### L2 Intermediate
4. How is position bias mitigated? Why does "evaluate both orderings and average" work?
5. How do AlpacaEval / MT-Bench / Chatbot Arena differ in their evaluation signals (automatic judge vs. human Elo)?
6. How do you detect whether training data has contaminated a given evaluation set?

### L3 Deep Dive
7. How does Goodhart's Law manifest in leaderboard gaming? How do you design "gaming-resistant" evaluations?
8. How is a reward model evaluated (the RewardBench approach)? What is the relationship between RM evaluation and final policy performance?
9. For reasoning models, why should evaluation shift from "single-pass accuracy" to "accuracy under a compute budget"? What does this demand of the evaluation protocol?
10. If online metrics (user retention) conflict with offline evaluation (judge win-rate), which do you trust, and how do you investigate the discrepancy?


## Extended L3

<details>
<summary>Q11. When using LLM-as-judge to evaluate multi-turn, long-context conversations, what are the common difficulties? What methodological improvements exist?</summary>

    The core challenge in evaluating multi-turn conversations is that the judge model tends to exhibit **context forgetting** or **local bias** — focusing only on the quality of the most recent one or two turns while ignoring overall conversational coherence and task completion. A key improvement is designing **process-oriented rubrics** that explicitly require evaluating each turn's contribution to the final goal, and introducing a **segment summarization** mechanism that forces the judge to summarize before scoring, thereby partially mitigating its short-sightedness.
    **Follow-up**: Beyond improving the rubric, can the evaluation protocol itself be changed to reduce judging difficulty — for example, decomposing it into a series of simpler subtask evaluations?

</details>

<details>
<summary>Q12. What are the main limitations of LLM-as-judge in assessing factual accuracy and logical soundness? How can they be mitigated?</summary>

    The main limitations are that the judge model's own **knowledge boundary** and **reasoning flaws** can lead to incorrect verdicts. It may fail to detect factual errors, or mistakenly accept an answer with logical gaps as sound. Mitigation typically involves a **hybrid evaluation** approach: for fact-checking, combine **retrieval-augmented verification** — retrieve authoritative information first, then compare; for logical evaluation, attempt to use **formal verification** tools or design **step-by-step verification prompts** specifically targeting reasoning chains.
    **Follow-up**: Given limited resources, which judge capabilities (breadth of knowledge, reasoning ability, tool use) should be prioritized for improvement to most effectively increase evaluation accuracy?

</details>

<details>
<summary>Q13. How do you evaluate a model's "emergent capabilities"? How does this fundamentally differ from evaluating conventional capabilities?</summary>

    The key difference in evaluating emergent capabilities lies in their **unpredictability** and **non-smoothness**. Conventional capabilities typically improve predictably on a benchmark as model scale or training data increases. Emergent capabilities, by contrast, appear suddenly past some threshold and are often not directly reflected in standard benchmarks. As a result, evaluation methods must shift from **fixed test sets** to **open-ended, programmatically generated probe tasks**, and must focus on detecting **behavioral pattern shifts** when the model faces entirely novel, complex task combinations.
    **Follow-up**: Can one design an evaluation framework that not only discovers emergent capabilities but also, to some degree, predicts the conditions under which they will appear?

</details>

<details>
<summary>Q14. Why does calibrating the confidence of LLM-as-judge outputs matter? How is it achieved in practice?</summary>

    Calibrating judge output confidence gives its scores or comparison results **interpretable probabilistic meaning**. For example, when a judge says "90% confident that A is better than B," that number should, in the long run, approximate the true frequency with which A is actually preferred over B. In practice, calibration requires a **human-annotated calibration set**. By repeatedly evaluating the judge on this set, one can analyze the distribution of its scoring deviations from human consensus, then apply **post-hoc calibration algorithms** (such as Platt Scaling or Isotonic Regression) to adjust raw scores so they better match the statistical patterns of human judgment.
    **Follow-up**: If the human-annotated data used for calibration is itself low-quality or very small in scale, what effect does this have on the calibrated judge? What are the alternatives?

</details>

<details>
<summary>Q15. When evaluating the safety of conversational models, why is "over-refusal" an important metric? How do you analyze the trade-off it forms with the harmful-prompt refusal rate?</summary>

    Over-refusal measures the degree to which a model incorrectly refuses **benign or borderline queries**, directly affecting user experience and model **utility**. A model with a very high over-refusal rate may be safe but becomes effectively useless. Analyzing this trade-off cannot simply pursue Pareto optimality across both metrics; instead, **risk tiering** should be introduced. Categorize harmful prompts by severity and set different refusal strictness thresholds for each tier. Evaluation should separately report refusal rates for each tier and use **cost-sensitive analysis** to assess the model's balance between overall risk exposure and user experience loss.
    **Follow-up**: How do you construct a high-quality adversarial safety evaluation set that can automatically generate prompts spanning various risk tiers and "gray area" cases?

</details>

<details>
<summary>Q16. How do you conduct "meta-evaluation" — that is, how do you judge whether an evaluation benchmark or an LLM-as-judge is itself valid and reliable?</summary>

    Meta-evaluation of a benchmark primarily examines its **discriminability** (can it effectively distinguish models at different capability levels), **robustness** (is it sensitive to minor prompt changes), and **ecological validity** (does the capability it measures relate to real-world needs). Meta-evaluation of an LLM-as-judge focuses on its **agreement** with human judgments (e.g., Cohen's Kappa) and its **fairness** across different subgroups. A key method is **cross-validation**: have multiple distinct, high-quality judges (or humans) evaluate the same set of data, and check whether the target judge or benchmark agrees with the consensus.
    **Follow-up**: After discovering that a widely used benchmark likely has serious biases or is outdated, what responsibilities and feasible actions does a researcher have to promote its iteration or warn the community?

</details>

<details>
<summary>Q17. In domain-specific settings (e.g., medical, legal), what unique challenges arise for general-purpose LLM-as-judge evaluation? What are the key steps in building a domain-expert evaluation pipeline?</summary>

    The core challenges are the **domain knowledge barrier** and the **specialized evaluation criteria** required. A general-purpose judge may not understand the nuances of domain terminology or the rigor of professional logic. Key steps in building a domain evaluation pipeline are, first, **co-defining evaluation dimensions** with domain experts — jointly determining dimensions such as "conservatism of medical advice" or "accuracy of legal citations." Second, **building a domain gold standard** — a set of authoritative reference answers or judgments annotated by experts. Finally, designing a **human-AI collaborative evaluation process** in which the AI judge handles initial screening while human experts handle edge-case review and final adjudication.
    **Follow-up**: When domain experts themselves disagree on the same response (e.g., physicians from different schools of thought), how do you design a system that accommodates reasonable expert disagreement while still enabling effective automated evaluation?

</details>
