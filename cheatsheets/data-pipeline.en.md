# Post-Training Data Pipeline & Curation Cheatsheet
## From instruction synthesis, rejection sampling, and preference construction to decontamination and data mixing

---

## 1. Pipeline Overview

The ceiling of post-training (SFT / preference alignment / RLVR) is **largely set by the data pipeline** — with the same algorithm, different data quality and mixtures give wildly different results. A typical post-training data pipeline:

```
Sources                  Generate                    Clean                    Mix → Consume
─────────────            ──────────────────          ─────────────            ────────────────────
human annotation         Self-Instruct / Evol        dedup                    mixing
open datasets      ─→    rejection sampling    ─→    decontamination    ─→    curriculum          ─→  SFT
distillation             preference pairs            quality filter           quality/difficulty       DPO / RLHF / RLVR
production logs          Magpie / synth textbooks    format normalize         per-stage mixing
```

### 1.1 Three main tracks

| Stage | Data form | Typical sources | Consuming algorithm |
|-------|-----------|-----------------|---------------------|
| **SFT (instruction tuning)** | (instruction, response) | human / synthetic / distilled / rejection sampling | supervised cross-entropy |
| **Preference alignment** | (prompt, chosen, rejected) | human preference / AI feedback | DPO / RLHF (PPO/GRPO) |
| **RLVR (verifiable reward)** | (prompt, verifier) | math/code with ground truth | GRPO + rule-based reward |

### 1.2 Core principle: quality > quantity

**LIMA** (Zhou et al., [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)) fine-tunes a 65B LLaMA on only **1,000** carefully curated SFT samples (no RLHF) and matches models trained on orders of magnitude more data. Its claim: **alignment is mostly a "surface behavior/format" learned atop pretraining knowledge, and a small set of high-quality samples suffices to activate it**.

> 💡 LIMA's "Less is More" does not mean less data is always better, but: **in SFT, sample quality, diversity, and alignment with the target distribution matter more than raw count**. This directly drives all the "filter-first" pipeline designs downstream.

Implication: the pipeline's value is not in "piling up volume" but in **synthesizing diverse high-quality candidates → strictly filtering/dedup/decontaminating → mixing by target capability**. §2–§6 unpack each stage.

---

## 2. Instruction Data Synthesis

Writing instruction data by hand is expensive and hard to scale. The mainstream approach is to **synthesize with a strong LLM**; the core challenges are **diversity** and **quality**.

### 2.1 Self-Instruct — seed bootstrapping

**Self-Instruct** (Wang et al., [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)): starting from a small set of human seed tasks, few-shot prompt the model to **generate new instructions + inputs + outputs**, then filter out samples too similar to the existing pool or low quality, iterate, and finally fine-tune **the same model** on the expanded data. Stanford **Alpaca** (no arXiv paper; blog/repo) used the Self-Instruct idea to distill 52k instructions from `text-davinci-003`.

**De-duplication filtering** is key: Self-Instruct uses a **ROUGE-L similarity < 0.7** threshold to drop new instructions too similar to existing ones, preventing the synthetic data from collapsing into a few templates.

```python
import random
from difflib import SequenceMatcher

# Self-Instruct-style bootstrap: few-shot prompt an LLM to generate new [instructions],
# then filter by similarity against the existing pool to de-duplicate. (The original also
# generates input/output instances per instruction; here we only demonstrate the
# "instruction generation + dedup" step.)
def sim_ratio(a: str, b: str) -> float:
    # Cheap similarity: difflib sequence-match ratio as a ROUGE-L proxy (original uses ROUGE-L < 0.7)
    return SequenceMatcher(None, a, b).ratio()

def self_instruct_round(pool, llm_generate, k=20, sim_thresh=0.7):
    """pool: existing instructions (list); llm_generate(prompt) -> new instruction text."""
    new_items = []
    for _ in range(k):
        demos = random.sample(pool, min(6, len(pool)))           # few-shot seeds
        prompt = "Generate a new task instruction different from these examples:\n" + "\n".join(demos)
        cand = llm_generate(prompt).strip()
        # Filter 1: too similar to any existing instruction -> drop
        if any(sim_ratio(cand, p) > sim_thresh for p in pool + new_items):
            continue
        # Filter 2: heuristic quality gate (character length; works for CN & EN)
        if not (5 <= len(cand) <= 300):
            continue
        new_items.append(cand)
    pool.extend(new_items)
    return new_items
```

### 2.2 Evol-Instruct — instruction evolution

**WizardLM / Evol-Instruct** (Xu et al., [arXiv:2304.12244](https://arxiv.org/abs/2304.12244), ICLR 2024): instead of flat generation, use an LLM to **"evolve" seed instructions to be more complex**, in two directions:

- **In-depth**: add constraints, add reasoning steps, complicate inputs, deepen difficulty.
- **In-breadth**: generate brand-new instructions covering new domains from existing ones.

After evolving, filter out "failed evolutions" (didn't get harder / became empty), yielding an instruction set with a steeper difficulty gradient.

### 2.3 Magpie — "zero-prompt" extraction from aligned models

**Magpie** (Xu et al., [arXiv:2406.08464](https://arxiv.org/abs/2406.08464), ICLR 2025): exploits the autoregressive nature of instruction models (e.g. Llama-3-Instruct) — **feed only the "pre-user-turn template" (no seed instruction)** and the model continues by writing a user instruction itself; then have it answer, yielding an (instruction, response) pair. No seeds, no human curation, scalable extraction of alignment data.

### 2.4 Distillation

Using a **stronger teacher model** to generate data for training a student is one of the most common synthesis routes.

- **Sequence-Level KD (SeqKD)** (Kim & Rush, [arXiv:1606.07947](https://arxiv.org/abs/1606.07947), EMNLP 2016): the classic approach — train the student on **sequences decoded by the teacher** (replacing word-level soft targets with sequence-level outputs); the student is faster and competitive.
- **Synthetic textbooks**: **Textbooks Are All You Need** (Gunasekar et al., [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)) trains the 1.3B phi-1 on "textbook-quality" filtered web data + GPT-3.5-synthesized textbooks and exercises; the paper reports **HumanEval pass@1 = 50.6%** — showing **high-quality synthetic data can drastically lower the scale requirement**.

> ⚠️ Two red lines for distillation/synthetic data: ① **inherited capability ceiling** — the student can hardly exceed the teacher on that distribution (good for leveling up, hard for pushing frontiers); ② **license/compliance** — using closed-model (e.g. GPT-4) outputs for training may violate its terms of service; check the license before publishing.

### 2.5 Risks of synthetic data: homogenization and mode collapse

> ❌ **Pitfall**: unfiltered synthetic data **homogenizes** — a few high-frequency templates, phrasings, and topics recur, and diversity plummets. Recursively "training a model on model-generated data" can further trigger **model collapse** (loss of distribution tails, variance shrinkage). Mitigations: similarity dedup (§2.1), multi-teacher / multi-temperature sampling, mixing in real human data, and explicit constraints on coverage (domain/difficulty/length).

---

## 3. Rejection Sampling & Self-Training

When a task has a **verifiable correctness signal** (math answers, unit tests, format), you can let the model **generate and filter its own data**, then train on it supervised — a cost-effective middle road between SFT and RL.

### 3.1 RFT — rejection sampling fine-tuning

**Rejection sampling Fine-Tuning (RFT)** (Yuan et al., [arXiv:2308.01825](https://arxiv.org/abs/2308.01825)): for each prompt, sample **N reasoning solutions** from the current (SFT) model, **keep only those with the correct answer**, and **dedup the reasoning paths** (keep several "distinct" correct paths per question to add diversity); use these self-generated correct samples as augmented SFT data.

```python
import re

# RFT: for each prompt sample N CoT solutions, keep only those with a CORRECT answer
# and a DISTINCT reasoning path, as augmented SFT data (Yuan et al. 2023).
def canonical(y: str) -> str:
    # Very crude toy fingerprint: digits->#, collapse whitespace
    # (the RFT paper dedups by distinct equation/reasoning paths — far finer).
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", y)).strip()

def rejection_sampling_ft_data(prompts, policy_sample, verify, gold,
                               N=64, max_keep=4):
    """policy_sample(x) -> one solution; verify(y, gold) -> bool."""
    sft_data = []
    for x in prompts:
        seen, kept = set(), []
        for _ in range(N):
            y = policy_sample(x)
            if not verify(y, gold[x]):          # wrong answer -> reject
                continue
            key = canonical(y)
            if key in seen:                      # same reasoning path -> dedup
                continue
            seen.add(key); kept.append(y)
            if len(kept) >= max_keep:            # at most max_keep per prompt
                break
        sft_data += [(x, y) for y in kept]
    return sft_data                              # then run standard SFT on these
```

### 3.2 STaR — bootstrapping reasoning with reasoning

**STaR** (Zelikman et al., [arXiv:2203.14465](https://arxiv.org/abs/2203.14465), NeurIPS 2022): have the model generate CoT reasoning, **keep the reasoning that reaches the correct answer** for fine-tuning; for wrong questions, give the correct answer and let the model **"rationalize"** a reasoning path that reaches it, also adding it to the training set. Reasoning ability bootstraps upward over iterations.

### 3.3 ReST — Grow / Improve double loop

**ReST** (Gulcehre et al., [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)): writes self-training as two alternating steps —

- **Grow**: generate a batch of samples **offline** from the current policy (expand the dataset);
- **Improve**: fine-tune on samples that pass a **quality threshold** (which can be raised gradually).

More compute-efficient than online RLHF: sampling and training are decoupled, and the generated data can be reused across rounds.

### 3.4 Relationship to RL

> 💡 RFT/STaR/ReST can be understood as **reward-weighted behavior cloning / iterated Best-of-N distillation**: maximum likelihood only on the self-sampled "reward=1 (correct)" samples. Under on-policy, single-step, small-update conditions it **approximates** one step of a "binary-reward policy gradient", but it is usually **not full RL** — no online exploration, continuous reward, or step-wise credit assignment. Its upside is simplicity, stability, and reusable data. For the link to RLVR/GRPO see [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html) and [llm-post-training](cheatsheet-llm-post-training-en.html).

---

## 4. Preference Data Construction

Preference triples (prompt, chosen, rejected) are the fuel for RLHF/DPO; their quality caps the RM/policy.

### 4.1 Where preference signals come from

| Source | How | Representative |
|--------|-----|----------------|
| **Human feedback** | annotators pick the better of a pair | InstructGPT / Llama-2 |
| **AI feedback (RLAIF)** | a strong LLM judge produces preference labels | Constitutional AI / RLAIF |
| **Hybrid** | AI pre-filter + human verification of key subset | most industrial pipelines |

- **Constitutional AI** (Bai et al., [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)): two stages — first SFT on the model's **self-critiques + revisions**, then RLHF with **AI-generated preference labels** driven by a set of "constitutional principles" (RLAIF), with harmlessness barely relying on human preferences.
- **RLAIF vs RLHF** (Lee et al., [arXiv:2309.00267](https://arxiv.org/abs/2309.00267), ICML 2024): shows that off-the-shelf LLM preference labels (RLAIF) can match RLHF on summarization/dialogue, and proposes **direct-RLAIF** (obtain the reward directly from the LLM during RL, no separate RM).

### 4.2 UltraFeedback / Zephyr — scaled AI-feedback recipes

- **UltraFeedback** (Cui et al., [arXiv:2310.01377](https://arxiv.org/abs/2310.01377), ICML 2024): for about **64k instructions**, generate about **256k candidate responses** with multiple models, then have **GPT-4 give numerical scores + textual critiques across several quality dimensions** (over 1M feedback entries in total), constructing large-scale preference/RM data.
- **Zephyr** (Tunstall et al., [arXiv:2310.16944](https://arxiv.org/abs/2310.16944)): **dDPO** (distilled DPO) runs DPO directly on UltraFeedback's AI feedback, distilling alignment with no human annotation.

### 4.3 Building preference pairs from scored candidates

```python
import itertools, random

# Build preference pairs from scored candidates: chosen = higher score, rejected = lower,
# keeping only pairs with a large enough score margin to avoid "hard-to-tell" noise
# (UltraFeedback-style).
def build_preference_pairs(prompt, candidates, scores, margin=1.0, max_pairs=4):
    # candidates: list[str]; scores: list[float] (GPT-4 / RM scores)
    assert len(candidates) == len(scores), "candidates and scores must align (zip truncates silently)"
    ranked = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)
    pairs = []
    for (y_w, s_w), (y_l, s_l) in itertools.combinations(ranked, 2):
        if s_w - s_l < margin:           # margin too small -> hard to tell -> drop
            continue
        pairs.append({"prompt": prompt, "chosen": y_w,
                      "rejected": y_l, "margin": s_w - s_l})
    random.shuffle(pairs)
    return pairs[:max_pairs]             # cap pairs per prompt to avoid imbalance
```

### 4.4 on-policy vs off-policy preferences

> ⚠️ Whether the responses in a preference pair **come from the current policy** affects the outcome: **off-policy** (pairs from other models or historical data) is a **standard, common** practice — DPO was designed for offline preferences, and Zephyr/UltraFeedback are both offline; **on-policy/online** (sampled from the model being optimized) is closer to the current distribution with better coverage. The risk is not that "the objective is inherently invalid", but that when preference data is **too far from the current policy**, coverage gaps and distribution shift (plus length/style bias) hurt. Iterative/online DPO mitigates this.

> 📝 Design choices for preference data such as **margin confidence filtering, annotator calibration, length debiasing** are covered in [reward-modeling-eval §3.4](cheatsheet-reward-modeling-eval-en.html); this section focuses on "where pairs come from".

---

## 5. Dedup & Decontamination

### 5.1 Why deduplicate

**Deduplicating Training Data Makes Language Models Better** (Lee et al., [arXiv:2107.06499](https://arxiv.org/abs/2107.06499), ACL 2022): removing near-duplicate spans/documents from training corpora reduces **verbatim memorization by roughly an order of magnitude**, while reaching comparable or better perplexity with fewer training steps. Same for post-training: duplicate samples are over-weighted, amplifying memorization and bias.

### 5.2 exact vs near-dup

- **Exact dedup**: hash whole spans/documents, drop identical ones.
- **Near-dup**: use **MinHash + LSH** to detect text with high Jaccard similarity.

**Jaccard similarity**: $J(A,B)=\dfrac{|A\cap B|}{|A\cup B|}$, where $A,B$ are the **k-shingles** (sets of k consecutive tokens) of two documents.

**Key MinHash property**: for a random hash (permutation) $h$,

$$\Pr[\,\min_{x\in A} h(x)=\min_{x\in B} h(x)\,]=J(A,B)$$

i.e. "the probability that two documents' signatures match at a given slot = their Jaccard". With $m$ independent hashes forming a signature, the fraction of matching slots is an unbiased estimate of $J$, reducing the $O(|A||B|)$ pairwise comparison to a constant-length signature comparison.

```python
import hashlib

# MinHash near-duplicate detection: k-shingles + min over multiple hashes form a signature;
# the fraction of matching signature slots approximates the Jaccard similarity.
def shingles(text, k=5):
    toks = text.split()
    return {" ".join(toks[i:i+k]) for i in range(max(1, len(toks) - k + 1))}

def minhash_sig(sh_set, num_perm=128):
    sig = []
    for s in range(num_perm):                       # each s simulates an independent hash
        m = min(int(hashlib.md5((str(s) + sh).encode()).hexdigest(), 16)
                for sh in sh_set)                   # min value under this hash
        sig.append(m)
    return sig

def est_jaccard(sig_a, sig_b):
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)

def is_near_dup(text, corpus_sigs, k=5, num_perm=128, thresh=0.8):
    sig = minhash_sig(shingles(text, k), num_perm)
    # brute-force linear scan over corpus_sigs here; in production add LSH banding to compare only same-bucket candidates
    dup = any(est_jaccard(sig, s) >= thresh for s in corpus_sigs)
    return dup, sig                                 # drop only true dups; otherwise add sig to the index
```

### 5.3 Decontamination

**Definition**: **remove training samples that overlap with evaluation benchmarks (test sets)**; otherwise the model "has seen the exam questions" and benchmark scores are inflated and meaningless.

- **n-gram overlap**: if a training sample shares a long enough n-gram with a benchmark test item (e.g. GPT-3 used 13-grams), flag it as contaminated and remove it. Simple, but insensitive to paraphrase/translation leakage.
- **Membership inference**: **Min-K% Prob** (Shi et al., [arXiv:2310.16789](https://arxiv.org/abs/2310.16789), ICLR 2024) — training-free, judges whether a text was seen during pretraining: take the **average log-likelihood of the lowest $k\%$ tokens**; unseen text tends to contain a few extremely-low-probability outlier tokens that drag the mean down, whereas member (seen) text has relatively higher likelihood even on its lowest $k\%$ tokens, so **a higher mean → more likely seen**. Useful for **post-hoc auditing** whether a model was contaminated by a benchmark.

> 🚨 **Pitfall**: decontamination must be done **before training, against all target eval sets**; contamination discovered afterward is often unfixable and forces you to switch eval sets. When reporting scores, state the decontamination protocol (n-gram order, whether paraphrase detection was used), otherwise "high scores" are not trustworthy.

---

## 6. Data Mixing & Curriculum

### 6.1 Data mixing

**The proportions** in which different sources/domains/tasks are mixed significantly affect final capabilities. **How Far Can Camels Go? (Tülu)** (Wang et al., [arXiv:2306.04751](https://arxiv.org/abs/2306.04751), NeurIPS 2023) systematically ablates 12 instruction-tuning datasets and concludes **no single dataset is best on all capabilities — mixing complementary sources is needed for breadth**. **Tülu 3** (Lambert et al., [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) further provides an open end-to-end SFT→DPO→RLVR recipe with full data.

### 6.2 DoReMi — learning the optimal mixture automatically

**DoReMi** (Xie et al., [arXiv:2305.10429](https://arxiv.org/abs/2305.10429), NeurIPS 2023): first train a small **proxy model**, use **Group DRO** (distributionally robust optimization) over domains to learn sampling weights, then resample the full dataset with those weights to train the large model; the paper reports reaching comparable performance about **2.6× faster**.

**Group DRO objective**: a min-max over the simplex of domain weights $\alpha$,

$$\min_\theta \max_{\alpha\in\Delta}\ \sum_{i=1}^{D}\alpha_i\, L_i(\theta)$$

DoReMi uses the **excess loss** (the proxy's loss minus a reference model's) to drive exponentiated-gradient updates, up-weighting "harder / higher-headroom" domains.

```python
import numpy as np

# DoReMi: learn per-domain sampling weights with a small proxy + Group DRO (exponentiated gradient).
# Key: each domain's "excess loss" must be clamped at the *token* level BEFORE averaging:
#   excess_i = mean_token( max(proxy_token_loss - ref_token_loss, 0) );
# clamping at the domain level ("average first, then clamp") wrongly cancels +/- token diffs.
def domain_excess(proxy_tok_loss, ref_tok_loss):
    # proxy_tok_loss[i] / ref_tok_loss[i]: per-token loss array of domain i
    return np.array([np.maximum(np.asarray(p) - np.asarray(r), 0.0).mean()
                     for p, r in zip(proxy_tok_loss, ref_tok_loss)])

def doremi_update(weights, excess, lr=1.0, smooth=1e-3):
    weights, excess = np.asarray(weights, float), np.asarray(excess, float)
    log_w = np.log(weights) + lr * excess                # exponentiated gradient ascent
    w = np.exp(log_w - log_w.max())                      # numerically stable softmax
    w = w / w.sum()
    return (1 - smooth) * w + smooth / len(w)            # smoothing (paper c≈1e-3), avoid starving a domain

# Average the per-step weights from proxy training as the final mixture, then resample to train the large model.
def sample_domain(weights, rng):
    return rng.choice(len(weights), p=weights)
```

### 6.3 Curriculum

Order data by **difficulty/length/quality** and feed it easy-to-hard, which often converges more stably and faster. Common in post-training:

- **Difficulty curriculum**: easy questions first, then hard (especially math/code).
- **Length curriculum**: short sequences first, then long (paired with context-window extension).
- **Quality curriculum**: large-scale medium quality first, then small-scale high quality (quality annealing).

### 6.4 Quality filtering

Score samples with **perplexity, reward-model scores, heuristic rules, or classifiers** and drop low-quality ones — but beware the filter's own bias (e.g. preferring long text) leaking into the data.

```python
# Quality filter pipeline: multi-layer filters screen samples in tiers;
# low-quality samples are eliminated at different stages.
# In practice all thresholds must be ablated per domain; over-filtering harms coverage and diversity.
def quality_filter(samples, ppl_model=None, rm=None, heuristic_rules=None):
    """samples: list[{'text': str, ...}]; returns samples that pass all filters.
    Filter layers: ① heuristic rules → ② PPL threshold → ③ RM score → ④ (optional) classifier."""
    passed = []
    for s in samples:
        t = s['text']
        # ① Heuristic: empty / too short / too long / excessive n-gram repetition → discard
        if not t or len(t) < 20:
            continue
        words = t.split()
        if len(words) > 2048:                               # too long (tune threshold by task)
            continue
        # Repetition check: most frequent 4-gram fraction > 30% → suspected template / filler
        from collections import Counter
        ngrams = Counter(zip(*(words[i:] for i in range(4))))
        if ngrams and ngrams.most_common(1)[0][1] / len(words) > 0.3:
            continue
        # ② PPL threshold: per-token perplexity moderate (too low = memorized, too high = gibberish)
        if ppl_model is not None:
            ppl = ppl_model.perplexity(t)
            if not (5 < ppl < 500):                          # tune thresholds per domain
                continue
        # ③ RM / quality model scoring: drop low-scoring samples
        if rm is not None:
            score = rm.score(t).item()
            if score < 0.0:                                  # RM threshold (tune per domain)
                continue
        # ④ Custom heuristic rules (e.g. must contain keywords, format checks, etc.)
        if heuristic_rules is not None:
            if not heuristic_rules(t):
                continue
        passed.append(s)
    return passed
# Key design choices:
# - PPL thresholds: too high → miss hard-but-valid long-tail samples; too low → admit repetitive/filler data
# - RM threshold: too low → valid samples of different style wrongly killed; too high → filter is toothless
# - Filter ordering: cheap heuristics first (save PPL/RM inference), expensive model-based ones later
# - Anti-bias monitoring: periodically check the distribution of filtered samples by domain/length/source
#   to prevent systematic over-filtering of any group
```

> ⚠️ Mixing and curriculum are largely **empirical** and require ablation against the target capabilities. Two common anti-patterns: ① **an overly narrow source mix / over-deduplication** → capability collapse, loss of diversity; ② **blindly piling on one high-scoring domain** → other capabilities regress (catastrophic forgetting), see [continual-post-training](cheatsheet-continual-post-training-en.html).

---

## 7. Interview Questions

### L1 — Fundamentals

---

<details>
<summary>Q1: What are the main stages of a post-training data pipeline?</summary>

**A:** Roughly: **sources** (human/open/distilled/logs) → **synthesize or collect** (Self-Instruct/Evol-Instruct/rejection sampling/preference pairs) → **clean** (dedup/decontamination/quality filtering/format normalization) → **mixing and curriculum** (domain ratios/difficulty ordering) → **consume** (SFT / DPO·RLHF / RLVR). The core idea is "synthesize diverse candidates → filter strictly → mix by target capability", not merely piling up volume.

> **Follow-up:** How do SFT data and preference data differ in form?
> SFT is a single supervised sample (instruction, response); preference data is a paired comparison (prompt, chosen, rejected) for RM/DPO to learn relative quality.

</details>

---

<details>
<summary>Q2: What is the core idea of Self-Instruct? Why is filtering essential?</summary>

**A:** Starting from a few human seed tasks, few-shot prompt the model to generate new "instruction + input + output", iteratively expand, then fine-tune the same model. Filtering is essential because unconstrained self-generation **homogenizes badly** (a few templates recur); Self-Instruct drops new instructions with ROUGE-L similarity > 0.7 to existing ones to preserve diversity, plus a basic quality gate (length/noise).

> **Follow-up:** How is Alpaca related to Self-Instruct?
> Alpaca (no arXiv paper) used the Self-Instruct idea to distill 52k instructions from `text-davinci-003` and fine-tune LLaMA-7B — a famous engineering instance of Self-Instruct.

</details>

---

<details>
<summary>Q3: What is rejection sampling fine-tuning (RFT)? How does it differ from plain SFT?</summary>

**A:** RFT samples N reasoning solutions per prompt from the current model, **keeps only the correct ones and dedups the reasoning paths**, then runs SFT on these self-generated correct samples. Difference from plain SFT: the data comes from the **model's own sampling + correctness filtering** (self-training), needs no extra human annotation, and is naturally on-policy (within the model's current distribution).

> **Follow-up:** What is RFT's precondition?
> An **automatically verifiable correctness signal** (math exact-match, code unit tests, format checks); otherwise you cannot "reject" wrong samples.

</details>

---

<details>
<summary>Q4: Why deduplicate training data? What does dedup buy you?</summary>

**A:** Duplicate samples get over-weighted, amplifying **verbatim memorization** and bias, and wasting compute. Lee et al. 2021 found removing near-duplicates cuts verbatim memorization by ~an order of magnitude and reaches comparable/better perplexity with fewer steps. In post-training, duplicate instructions also overfit the model to specific templates.

> **Follow-up:** What is the difference between exact dedup and near-dup?
> Exact removes only identical text; near-dup uses MinHash/LSH to catch high-Jaccard similarity (paraphrases, light edits too) — broader coverage but more compute.

</details>

---

<details>
<summary>Q5: What is decontamination? What happens if you skip it?</summary>

**A:** Decontamination removes training samples overlapping with **evaluation benchmark test sets**. If skipped, the model effectively "has seen the exam", so benchmark scores are inflated, fail to reflect true generalization, and become incomparable across models. Common methods: n-gram overlap removal (e.g. 13-gram), membership-inference auditing (Min-K% Prob).

> **Follow-up:** At which stage should decontamination be done?
> **Before training, against all target eval sets.** Contamination found afterward usually cannot be fixed — you can only switch eval sets.

</details>

---

<details>
<summary>Q6: What is distillation data? What is SeqKD's core approach?</summary>

**A:** Using a stronger teacher to generate data for training a student. **Sequence-Level KD (SeqKD, Kim & Rush 2016)** core: instead of distilling on word-level soft labels, train the student on **entire sequences decoded by the teacher**, yielding a faster, competitive model. The LLM-era instruction distillation (Alpaca/Zephyr) is its spiritual successor.

> **Follow-up:** What is the fundamental limitation of distillation data?
> **Inherited capability ceiling** — the student can hardly exceed the teacher on that distribution; plus license/compliance risk from training on closed-model outputs.

</details>

---

<details>
<summary>Q7: Where do chosen/rejected preference pairs come from? Human vs AI feedback?</summary>

**A:** Three routes: ① **human feedback** — annotators pick the better of a pair (InstructGPT/Llama-2); ② **AI feedback (RLAIF)** — a strong LLM judge produces preference labels (Constitutional AI, RLAIF); ③ **hybrid** — AI pre-filter + human verification of a key subset. AI feedback is cheap and scalable but inherits the judge model's biases.

> **Follow-up:** How does UltraFeedback construct preference data?
> For about 64k instructions, generate about 256k candidates with multiple models, then have GPT-4 score across several dimensions + write critiques, building preference/RM data; Zephyr runs dDPO directly on it.

</details>

---

<details>
<summary>Q8: What does LIMA's "Less is More" conclusion tell us?</summary>

**A:** LIMA fine-tunes a 65B LLaMA on 1,000 carefully curated samples (no RLHF) and rivals models trained on far more data. Conclusion: **alignment is mostly learning a surface behavior/format atop pretraining knowledge**, so in SFT the sample **quality, diversity, and alignment with the target distribution** matter more than raw count. It drives the "filter-first, quality-first" pipeline philosophy.

> **Follow-up:** Does this mean less data is always better?
> No. It means **high-quality few > low-quality many**; quantity still matters but with diminishing returns, and quality/diversity are prerequisite gates.

</details>

---

### L2 — Intermediate

---

<details>
<summary>Q9: Differences among Self-Instruct / Alpaca / Evol-Instruct / Magpie?</summary>

**A:** All generate instruction data, but differ in mechanism: **Self-Instruct** seed bootstrap + similarity filtering; **Alpaca** is a Self-Instruct instance distilled from davinci; **Evol-Instruct (WizardLM)** uses an LLM to "evolve" instructions to be harder (depth) or broader (breadth); **Magpie** exploits an aligned model's autoregressive nature, feeding only the pre-user-turn prefix to make the model emit instruction-response pairs, no seeds needed. Complexity gradient: Magpie/Self-Instruct (breadth) → Evol-Instruct (depth/difficulty).

> **Follow-up:** Why does Evol-Instruct improve complex instruction following?
> Because it explicitly creates **harder, more-constrained** instructions, shifting the dataset's difficulty distribution upward and forcing the model to handle multi-constraint/multi-step tasks.

</details>

---

<details>
<summary>Q10: Similarities and differences among RFT, STaR, ReST?</summary>

**A:** Common: all are **self-training** — the model generates, filters by correctness/quality, then trains supervised. Differences: **RFT** keeps multiple deduped correct reasoning paths per question for augmented SFT; **STaR** additionally "rationalizes" reasoning for wrong questions given the correct answer; **ReST** writes it explicitly as Grow (offline generation) / Improve (fine-tune on threshold-passing samples), raising the threshold over rounds and reusing data.

> **Follow-up:** What problem does STaR's rationalization solve?
> The coverage problem of "hard questions never sampling a correct rationale" — given the answer, the model back-derives a rationale so hard questions can also contribute training signal.

</details>

---

<details>
<summary>Q11: What are the main risks of synthetic data? How to mitigate?</summary>

**A:** ① **Homogenization** — a few templates/topics/phrasings recur, diversity plummets; ② **model collapse** — recursively training a model on model data loses distribution tails and collapses variance; ③ **error amplification** — the teacher's systematic errors get copied en masse; ④ **capability ceiling** — hard to exceed the teacher. Mitigations: similarity dedup, multi-teacher/multi-temperature sampling, mixing in real human data, explicit coverage constraints (domain/difficulty/length), and keeping a human-verified subset.

> **Follow-up:** Why does recursive training cause model collapse?
> Each generation's sampling favors high-probability regions and discards low-probability tails; errors accumulate across generations, the distribution narrows, and eventually diversity and coverage of the true distribution collapse.

</details>

---

<details>
<summary>Q12: Why is on-policy preference data important? What's the problem with off-policy?</summary>

**A:** off-policy (using other models or historical data) is DPO's **standard usage** (DPO was designed for offline preferences; Zephyr/UltraFeedback are all like this) and is not invalid; but when preference data is **far from the current policy**, coverage gaps and distribution shift reduce update efficiency and pick up the data source's length/style bias. **on-policy/online** (sampled from the model being optimized) puts the signal in the region the model actually generates, improving coverage and reducing post-iteration mismatch. Iterative/online DPO mitigates this by re-sampling preference pairs with the new policy.

> **Follow-up:** Is this the same as the RM "distribution shift" problem?
> Same root — "training-signal distribution ≠ current policy distribution". For RMs it shows up as OOD scoring drift; for DPO as implicit-reward/gradient distortion; both are mitigated by iteratively refreshing with on-policy data.

</details>

---

<details>
<summary>Q13: How does MinHash approximate Jaccard? Why does it speed up near-dup detection?</summary>

**A:** For a random hash $h$, the probability that two sets' min-hashes are equal equals the Jaccard: $\Pr[\min h(A)=\min h(B)]=J(A,B)$. With $m$ independent hashes forming a length-$m$ signature, the **fraction of matching slots** is an unbiased estimate of $J$. This reduces the $O(|A||B|)$ per-shingle comparison to a **constant-length signature comparison**; with LSH bucketing, only candidates in the same bucket are compared, avoiding all-pairs comparison.

> **Follow-up:** How do you choose the shingle size k?
> k too small → short documents look similar (high noise); k too large → over-sensitive to small rewrites (low recall). Text dedup commonly uses k=5–10 tokens, tuned to corpus granularity.

</details>

---

<details>
<summary>Q14: How is decontamination actually done? What are the pitfalls?</summary>

**A:** Two kinds: ① **n-gram overlap** — remove a training sample if it shares a long enough n-gram with a benchmark test item (e.g. GPT-3's 13-grams); simple but **insensitive to paraphrase/translation leakage**; ② **membership inference** — e.g. Min-K% Prob, training-free, to audit post-hoc whether text was seen in pretraining. Pitfalls: missing paraphrase/multilingual leakage, too-loose/too-strict thresholds, checking only some benchmarks, and not stating the decontamination protocol so scores are incomparable.

> **Follow-up:** Why is paraphrase-style leakage especially dangerous?
> It bypasses exact/n-gram matching (synonyms, reordering, translation) yet lets the model effectively see the question's semantics; n-gram decontamination misses it, requiring semantic-level detection or stricter source control.

</details>

---

<details>
<summary>Q15: Why does data mixing matter? How does DoReMi learn the mixture automatically?</summary>

**A:** Mixing proportions of different domains/tasks significantly shape the final capability profile (Tülu's ablation shows no single dataset is all-round; complementary mixing is needed). **DoReMi** first trains a small proxy model, uses **Group DRO** over domains to learn sampling weights — for each domain it computes the **excess loss** relative to a reference model and up-weights high-excess (harder/higher-headroom) domains via exponentiated gradient; then resamples with the learned weights to train the big model, reporting ~2.6× speedup.

> **Follow-up:** Why use a small proxy model instead of searching the mixture on the big model directly?
> Repeatedly trying mixtures on the big model is extremely expensive; the small proxy is cheap, and the learned domain weights transfer to the big model's data resampling, drastically cutting the search cost.

</details>

---

<details>
<summary>Q16: How is curriculum learning used in post-training?</summary>

**A:** Order data by difficulty/length/quality and feed easy-to-hard. Common: **difficulty curriculum** (easy then hard, especially math/code), **length curriculum** (short then long, paired with context extension), **quality annealing** (large-scale medium quality first, small-scale high quality to finish). The goal is more stable, faster convergence and avoiding early derailment by hard samples.

> **Follow-up:** Is curriculum learning always effective?
> Not necessarily. Effectiveness depends on the task and difficulty metric; with an inaccurate difficulty metric or ample model capacity, curriculum may not differ much from random shuffling — ablate to verify.

</details>

---

<details>
<summary>Q17: How do Constitutional AI / RLAIF replace human labels with AI feedback?</summary>

**A:** **Constitutional AI** has two stages: first the model **self-critiques and revises** its own harmful responses per a set of "constitutional principles" for SFT, then **generates preference labels per the principles** for RLHF (i.e. RLAIF), with harmlessness barely depending on human preferences. **RLAIF vs RLHF** shows off-the-shelf LLM preference labels can match human labels, and proposes direct-RLAIF (obtain reward directly from the LLM during RL, no separate RM).

> **Follow-up:** What are the main risks of AI feedback?
> It inherits the judge model's biases (verbosity, self-preference, format preference) and may amplify them; key dimensions still need human calibration and heterogeneous-judge cross-checks (see reward-modeling-eval).

</details>

---

### L3 — Advanced

---

<details>
<summary>Q18: What is the mathematical relationship between RFT and RL (policy gradient)? Is RFT RL?</summary>

**A:** RFT does maximum likelihood on self-sampled "reward=1 (correct)" samples and can be seen as **reward-weighted behavior cloning / iterated Best-of-N distillation**. It is linked to policy gradient: in the REINFORCE gradient $\mathbb{E}_{y\sim\pi}[r(y)\nabla\log\pi(y)]$, when $r\in\{0,1\}$ and only $r=1$ samples are kept, this is exactly "raising the log-likelihood of correct samples" — **but this approximation only holds under on-policy, single-step, small updates**; RFT usually samples data once and trains offline, lacking online exploration, KL constraint, and step-wise credit assignment, so it is **generally not called full RL** and has a lower optimization ceiling than online GRPO/PPO.

> **Follow-up:** Since RFT is simpler, why still use RLVR/GRPO?
> RFT only hard-filters by correct/incorrect, discarding the gradient information in wrong samples, and has no online exploration; GRPO updates online with intra-group relative advantage, leverages negative samples and continuous rewards, and has a higher ceiling on hard tasks.

</details>

---

<details>
<summary>Q19: Why can training on synthetic data cause model collapse? Theoretical hazard of recursive training?</summary>

**A:** Recursively "training the next generation on the previous generation's outputs" progressively **loses distribution tails**: each sampling favors high-probability modes and systematically under-samples low-probability (rare but real) examples; finite-sampling statistical error + function-approximation error accumulate across generations, the distribution's variance narrows and its mode concentrates, eventually collapsing diversity and diverging from the true distribution. Mitigation: **always mix in enough real human data** as an anchor, cap the synthetic fraction, and constrain coverage explicitly.

> **Follow-up:** How big a threat is this to one-off distillation like "distill GPT-4 into an open model"?
> One-off distillation (fixed, strong teacher) is far safer than "self-consuming recursive training" — the main risks are inherited capability ceiling and teacher-bias copying, not typical recursive collapse; but if the distilled model's outputs are fed back into training in a loop, collapse risk rises.

</details>

---

<details>
<summary>Q20: Quality vs quantity: how to understand the tension between LIMA and scaling?</summary>

**A:** Scaling laws say "more data → lower loss"; LIMA says "1,000 samples suffice for alignment" — they don't conflict, acting at **different stages and goals**: pretraining acquires knowledge via scale (quantity-dominated); the SFT alignment stage mainly **activates/formats** capabilities already in pretraining, where high-quality, diverse, target-aligned small samples have the highest marginal value and quantity returns diminish fast. So "quality-first" targets post-training alignment and does not negate the pretraining scaling law.

> **Follow-up:** So does SFT data ignore quantity entirely?
> No, but there is a "sufficient coverage" inflection: once diversity covering the target capabilities/formats is reached, piling on more homogeneous data yields little, and may even hurt via noise/repetition; the key is **diversity coverage**, not raw count.

</details>

---

<details>
<summary>Q21: Design a complete SFT data pipeline. What are the key decision points?</summary>

**A:**

```
[Sources] human seeds + open data + teacher distillation + rejection-sampled self-production
   ↓
[Synthesize] Self-Instruct (breadth) + Evol-Instruct (difficulty) + Magpie (scale)
   ↓
[Filter] similarity dedup (MinHash) + quality scoring (PPL/RM/classifier) + heuristic rules
   ↓
[Decontaminate] n-gram + semantic-level detection against ALL target benchmarks
   ↓
[Mix/Curriculum] domain mixing (DoReMi/manual ablation) + difficulty/length curriculum
   ↓
[Train] SFT → evaluate → diagnose weak capabilities → loop back to add data
```

Key decision points: teacher choice & license, synthetic-vs-real ratio, dedup threshold (recall vs over-deletion), decontamination protocol, per-capability domain ratios, whether to use a curriculum, loop-back iteration frequency.

> **Follow-up:** Which step is most easily overlooked but most costly?
> **Decontamination** — once contamination is left uncleaned, all benchmark conclusions become untrustworthy and hard to trace; next is **diversity/coverage monitoring**, since homogenization often surfaces only when eval scores drop.

</details>

---

<details>
<summary>Q22: How do margin filtering and annotator calibration of preference data affect RM/DPO?</summary>

**A:** **Margin filtering** drops pairs whose chosen/rejected score gap is too small (hard to tell, noisy) — it denoises and stabilizes training; too aggressive, it loses boundary samples and weakens RM/DPO discrimination on "slight differences". **Annotator calibration** (anchor items, annotator-effect modeling, majority voting) reduces systematic labeling bias, otherwise bias is learned by the RM and amplified through RLHF into the policy. Both are "fundamental investments at the data-construction stage", more curative than post-hoc KL/ensemble tuning.

> **Follow-up:** What to watch when computing the margin from AI-judge scores?
> AI scores have their own bias and scale drift; absolute gaps are not comparable across prompts — compare within the same prompt, take stable averages (temperature=0, repeated) when needed, and calibrate against humans. See reward-modeling-eval §3.4.

</details>

---

<details>
<summary>Q23: What does incomplete decontamination mean for benchmark evaluation? How to prevent leakage systematically?</summary>

**A:** Incomplete = the model has partially seen the exam, so scores are **systematically inflated** and incomparable, even misleading model selection and scientific conclusions. Systematic prevention: ① decontaminate against **all** target eval sets before training with n-gram + semantic-level detection; ② audit post-hoc with membership inference like Min-K%; ③ source control (avoid crawling pages/question banks containing benchmarks); ④ **state the decontamination protocol** when reporting (n-gram order, whether paraphrase/multilingual was checked); ⑤ re-verify key conclusions on **newly released / private holdout** eval sets.

> **Follow-up:** Why can a "newly released eval set" serve as a contamination control?
> If a model's score drops sharply on an eval set released after its training cutoff, that strongly suggests its earlier high scores came from contamination; the temporal "could not have seen it" provides a natural control.

</details>

---

<details>
<summary>Q24: What is DoReMi's Group DRO objective? Why use a proxy model to learn the mixture?</summary>

**A:** Group DRO optimizes the **worst case** of weighted per-domain loss: $\min_\theta\max_{\alpha\in\Delta}\sum_i\alpha_i L_i(\theta)$, with $\alpha$ on the domain-weight simplex. DoReMi uses a reference model to define each domain's **excess loss** and iteratively up-weights high-excess domains via exponentiated gradient, yielding a mixture robust to "hard domains". A **proxy (small model)** is used because repeatedly searching mixtures on the big model is extremely expensive; weights learned on the small model transfer to the big model's data resampling, with the paper reporting ~2.6× training speedup.

> **Follow-up:** Does Group DRO's "up-weight high-loss domains" over-favor noisy domains?
> There is such a risk — weighting purely by raw loss can amplify noisy/unlearnable domains. DoReMi uses **excess** loss relative to a reference (not absolute loss) plus smoothing/clipping to mitigate, but remains sensitive to the reference choice and noise.

</details>

---

<details>
<summary>Q25: License, copyright, and "inherited capability ceiling" issues of distillation/synthetic data?</summary>

**A:** ① **License/compliance**: most closed models' (e.g. GPT-4) terms **forbid using their outputs to train competing models**, so publishing distilled data has compliance risk — check terms or switch to a permissively-licensed teacher; ② **inherited capability ceiling**: the student can hardly exceed the teacher on the teacher's distribution — distillation suits "leveling up/democratizing" capability rather than "pushing the frontier"; ③ **bias/error copying**: the teacher's systematic errors and stylistic preferences get copied en masse into the student. In practice: multi-teacher, mix in real data, keep human verification, and record data provenance and licenses.

> **Follow-up:** If you want the student to exceed the teacher, what can you do at the data level?
> Move to **verifiable self-produced data** (RFT/RLVR, using environments/unit-tests/answers as the teacher rather than a model), on-policy preference iteration, and new real human data or tool feedback — so the supervision signal comes from "objective correctness" rather than another model's distribution.

</details>

---

## Glossary

| Term | Brief definition |
|------|------------------|
| Self-Instruct | Bootstrap instruction data from seed tasks + similarity filtering |
| Evol-Instruct | Use an LLM to evolve instructions harder/broader (WizardLM) |
| Distillation | Use a strong teacher to generate data for a student |
| SeqKD | Train the student on teacher-decoded sequences |
| Rejection Sampling FT (RFT) | Sample N, keep only correct + deduped, then SFT |
| STaR | Bootstrap reasoning; rationalize from answers for wrong questions |
| ReST | Grow (generate) / Improve (threshold fine-tune) self-training loop |
| RLAIF | RL from AI feedback — LLM preference labels replace human labels |
| Constitutional AI | Principle-driven self-critique + AI preference alignment |
| Preference Pair | (prompt, chosen, rejected) triple |
| on-policy data | Samples from the model currently being optimized |
| Dedup | Remove (near-)duplicate samples |
| MinHash / LSH | Approximate Jaccard with min-hash signatures; speed up near-dup search |
| Jaccard | $|A\cap B|/|A\cup B|$ |
| Decontamination | Remove training samples overlapping with eval sets |
| Min-K% Prob | Training-free membership inference: was the text seen in pretraining? |
| Data Mixing | Mixing proportions across domains/sources |
| DoReMi | Learn domain mixture with a proxy + Group DRO |
| Curriculum | Order data by difficulty/length/quality |
| Model Collapse | Recursive training loses distribution tails and collapses diversity |
| LIMA | "Less is More": few high-quality samples suffice for alignment |

---

*For study reference only. Paper conclusions and figures follow the original papers; benchmark scores are illustrative and not head-to-head comparisons.*

## §A Key Papers Timeline

- **2016-06 · Sequence-Level Knowledge Distillation** — Kim & Rush, EMNLP 2016. [arXiv:1606.07947](https://arxiv.org/abs/1606.07947) — Lifts knowledge distillation to the sequence level: train the student on teacher-decoded sequences instead of word-level soft labels, yielding a faster, competitive model — the conceptual root of LLM distillation data.

- **2021-07 · Deduplicating Training Data Makes Language Models Better** — Lee et al., ACL 2022. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) — Systematically shows removing near-duplicates from training corpora cuts verbatim memorization by ~an order of magnitude and reaches comparable/better perplexity with fewer steps, establishing dedup as a baseline action.

- **2022-03 · STaR: Bootstrapping Reasoning With Reasoning** — Zelikman et al., NeurIPS 2022. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) — Iteratively bootstraps on the model's own correct CoT rationales; for wrong questions it "rationalizes" a rationale from the given answer, requiring no large human rationale dataset.

- **2022-12 · Self-Instruct: Aligning LMs with Self-Generated Instructions** — Wang et al., ACL 2023. [arXiv:2212.10560](https://arxiv.org/abs/2212.10560) — Bootstraps instruction data from a few seed tasks + ROUGE-L similarity filtering for de-homogenization, pioneering low-cost scalable instruction-data synthesis (Alpaca is an instance).

- **2022-12 · Constitutional AI: Harmlessness from AI Feedback** — Bai et al., preprint. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) — Uses a set of "constitutional principles" to drive model self-critique + revision for SFT, then AI-generated preference labels for RLHF (RLAIF), with harmlessness barely relying on human preference annotation.

- **2023-04 · WizardLM: Empowering LLMs to Follow Complex Instructions** — Xu et al., ICLR 2024. [arXiv:2304.12244](https://arxiv.org/abs/2304.12244) — Proposes Evol-Instruct: use an LLM to evolve seed instructions in depth (add constraints/difficulty) and breadth (cover new domains), markedly improving complex instruction following.

- **2023-05 · LIMA: Less Is More for Alignment** — Zhou et al., NeurIPS 2023. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) — Aligns a 65B LLaMA with only 1,000 curated samples (no RLHF), arguing alignment mainly activates pretrained capabilities and that quality/diversity beat quantity.

- **2023-05 · DoReMi: Optimizing Data Mixtures** — Xie et al., NeurIPS 2023. [arXiv:2305.10429](https://arxiv.org/abs/2305.10429) — Uses a small proxy model + Group DRO to learn per-domain sampling weights, then resamples to train the big model, reporting ~2.6× speedup and turning data mixing into an optimizable objective.

- **2023-06 · How Far Can Camels Go? (Tülu)** — Wang et al., NeurIPS 2023. [arXiv:2306.04751](https://arxiv.org/abs/2306.04751) — Systematically ablates 12 instruction-tuning datasets, concluding no single dataset is all-round and complementary mixing is needed, and releases the Tülu open instruction-tuned model suite.

- **2023-06 · Textbooks Are All You Need (phi-1)** — Gunasekar et al., preprint. [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) — Trains the 1.3B phi-1 on "textbook-quality" filtered data + synthetic textbooks/exercises, reporting HumanEval pass@1 = 50.6%, arguing high-quality synthetic data can drastically lower the scale requirement.

- **2023-08 · Scaling Relationship on Math Reasoning (RFT)** — Yuan et al., preprint. [arXiv:2308.01825](https://arxiv.org/abs/2308.01825) — Proposes rejection sampling fine-tuning (RFT): sample multiple reasoning paths, keep only correct and deduped ones for augmented SFT, and studies how pretraining loss and data amount jointly govern math reasoning.

- **2023-08 · Reinforced Self-Training (ReST)** — Gulcehre et al., preprint. [arXiv:2308.08998](https://arxiv.org/abs/2308.08998) — Frames self-training as a Grow (offline sample generation) / Improve (fine-tune on threshold-passing samples) double loop; decoupling sampling from training is more compute-efficient than online RLHF and data is reusable.

- **2023-09 · RLAIF vs. RLHF: Scaling RLHF with AI Feedback** — Lee et al., ICML 2024. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267) — Shows off-the-shelf LLM preference labels (RLAIF) can match RLHF on summarization/dialogue, and proposes direct-RLAIF: obtain the reward directly from the LLM during RL, no separate RM.

- **2023-10 · UltraFeedback: Boosting LMs with Scaled AI Feedback** — Cui et al., ICML 2024. [arXiv:2310.01377](https://arxiv.org/abs/2310.01377) — For ~64k instructions, generates ~256k candidates with multiple models and GPT-4 multi-dimensional scores + critiques, building large-scale AI-feedback preference/RM data — a key fuel for open-source alignment.

- **2023-10 · Zephyr: Direct Distillation of LM Alignment** — Tunstall et al., preprint. [arXiv:2310.16944](https://arxiv.org/abs/2310.16944) — Uses dDPO (distilled DPO) directly on UltraFeedback's AI feedback to distill a strongly-aligned 7B model with no human annotation, validating the "AI feedback + DPO" recipe.

- **2023-10 · Detecting Pretraining Data (Min-K% Prob)** — Shi et al., ICLR 2024. [arXiv:2310.16789](https://arxiv.org/abs/2310.16789) — Proposes the training-free membership-inference Min-K% Prob: take the average log-likelihood of the lowest k% tokens — higher for members, lower for unseen text (which has low-probability outliers) — usable for benchmark contamination auditing.

- **2024-06 · Magpie: Alignment Data Synthesis from Scratch** — Xu et al., ICLR 2025. [arXiv:2406.08464](https://arxiv.org/abs/2406.08464) — Exploits an aligned model's autoregressive nature: feed only the pre-user-turn template to make the model emit instruction-response pairs — zero seeds, zero human curation, scalable alignment-data synthesis.

- **2024-11 · Tülu 3: Pushing Frontiers in Open Post-Training** — Lambert et al., preprint. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) — An open end-to-end post-training recipe (SFT→DPO→RLVR) with full data/code, introducing RLVR as a key component and showcasing the data pipeline and mixing engineering in detail.
