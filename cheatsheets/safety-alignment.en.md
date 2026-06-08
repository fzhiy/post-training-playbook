# Safety Alignment Cheatsheet
## From HHH, Red-Teaming, and Jailbreaks to Safety Training, Evaluation, and Vulnerabilities

---

## 1. Safety Alignment Overview

The problem safety alignment must solve: **make the model refuse what it shouldn't do while staying helpful**—and hold the line even under **adversarial input**. It is not a single technique but systems engineering spanning five layers: **training signal / inference-time safety / representation defense / runtime guardrails / evaluation**.

### 1.1 The HHH Framework and the helpful↔harmless Tension

The mainstream alignment objective is often summarized as **HHH: Helpful / Harmless / Honest**. Among these, **helpful and harmless are inherently in tension**—a model that always refuses is safest but most useless, while a model that complies with anything is most useful but most dangerous.

**HH-RLHF** (Bai et al., [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)): Anthropic trains an assistant with RLHF, learning a helpfulness reward model and a harmlessness reward model **separately**, and systematically studies the tension between them—improving harmlessness often comes at the cost of helpfulness, and vice versa.

> 💡 The core of safety alignment is not "maximize harmlessness" but **maximize helpfulness under a harmlessness constraint**. This turns it from "add a filter" into a **constrained optimization problem**—which is exactly the starting point of §4.1 Safe-RLHF.

### 1.2 Four Layers: Training / Representation / Runtime / Evaluation

| Layer | Approach | Representative | Section |
|------|------|------|------|
| **Training signal** | Write "safety / privilege order" into the loss/reward | Safe RLHF / RBR / Instruction Hierarchy / Constitutional AI | §4.1–4.2, §4.5; CAI on data-pipeline |
| **Inference-time safety** | Explicitly reason about safety specifications in CoT | Deliberative Alignment | §4.6 |
| **Representation / internal** | Rewrite internal representations that produce harmful output | Circuit Breakers | §4.4 |
| **Runtime I/O guardrails** | Intercept independently at input/output | Llama Guard / Constitutional Classifiers | §4.3, §4.7 |
| **Evaluation / audit** | Attack the model, measure refusal and over-refusal | Red-teaming / HarmBench / StrongREJECT / XSTest | §2, §5 |

The five layers form defense in depth: any single layer can be bypassed (§3 jailbreaks, §6 vulnerabilities), and only the combination is meaningful.

### 1.3 The Core Conflict: refusal vs over-refusal

- **Under-refusal**: the model answers a harmful request it should have refused → a safety hole.
- **Over-refusal**: the model refuses a request that **looks dangerous but is actually harmless** (e.g., "how to kill a Linux process") → usability is harmed.

> ⚠️ Safety training almost always pushes the decision boundary **toward the "easier-to-refuse" side**, at the cost of rising over-refusal. Reporting only "harmful-request refusal rate" without over-refusal is one-sided—you **must test both ends** (§5.3 XSTest exists for exactly this).

---

## 2. Red-Teaming & Attack Surface

**Red-teaming** = actively constructing adversarial inputs to elicit a model's harmful behavior, so it can be found and patched before deployment. It is the "testing" stage of safety alignment.

### 2.1 Manual Red-Teaming — Scaling Behavior and Harm Taxonomy

**Red Teaming LMs to Reduce Harms** (Ganguli et al., [arXiv:2209.07858](https://arxiv.org/abs/2209.07858)): organizes crowdsourced annotators for large-scale manual red-teaming, systematically records **how attack success rate changes with model scale / alignment method**, and distills recurring harm categories. One counterintuitive finding: **simply scaling up the model is not necessarily safer**, and plain RLHF is harder to break on some dimensions—safety comes from the alignment method rather than scale itself.

### 2.2 Automated Red-Teaming — Using an LM to Attack an LM

**Red Teaming LMs with Language Models** (Perez et al., [arXiv:2202.03286](https://arxiv.org/abs/2202.03286), EMNLP 2022): manual red-teaming is expensive and hard to cover, so instead use a **red-team LM to automatically generate a large number of test cases** (zero-shot / few-shot / supervised / RL in various ways) to elicit harmful output from the target LM, then filter successful attacks with a classifier. This turns red-teaming from "labor-intensive" into "scalable and reproducible."

### 2.3 Why Jailbreaks Succeed: Two Failure Modes

**Jailbroken: How Does LLM Safety Training Fail?** (Wei et al., [arXiv:2307.02483](https://arxiv.org/abs/2307.02483), NeurIPS 2023 Oral) gives a **mechanistic explanation** of jailbreaks, with two root causes:

- **Competing objectives**: the model is trained both to "follow instructions" and to "stay harmless"; the attacker constructs a scenario that makes the two conflict (e.g., "you must begin your answer with Sure"), forcing instruction-following to override safety.
- **Mismatched generalization**: safety training's coverage is **narrower than** pretraining capability—wrapping a harmful request in Base64, a low-resource language, or a rare encoding lands in a distribution that safety training never covered, and the model "fails to see" it as a harmful request.

> 💡 These two failure modes are the theoretical template for almost all later jailbreak techniques: the optimization / iterative / long-context attacks in §3 are essentially systematizing the creation of "competing objectives" or "mismatched generalization."

---

## 3. Jailbreak Attacks

Jailbreak = bypassing safety alignment to induce the model to produce content it should refuse. Categorized into four types by mechanism.

### 3.1 Optimization-Based Attacks — GCG Adversarial Suffixes

**GCG (Universal and Transferable Adversarial Attacks)** (Zou et al., [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)): **append an adversarial suffix** after the harmful request, and optimize the suffix's tokens via **greedy coordinate gradient** to maximize the probability that the model begins with an affirmative tone (e.g., "Sure, here is..."). The optimization objective can be written as:

$$\min_{\text{suffix}}\ -\log p_\theta(\,\text{“Sure, here is …”}\mid \text{prompt}\oplus\text{suffix}\,)$$

The most dangerous property is **universal + transferable**: a suffix optimized on a white-box small model can often **transfer** to attack closed-source models it has never seen (GPT/Claude)—empirically this indicates that different models' safety decision boundaries share a transferable vulnerability (possibly related to similar training data / refusal patterns).

> ❌ **Pitfall**: GCG suffixes are usually **gibberish strings** with extremely high perplexity → easily blocked by "perplexity filtering." This spawned the next category—attacks that generate **fluent** jailbreak prompts (AutoDAN).

### 3.2 Iterative / Evolutionary Attacks — PAIR and AutoDAN

- **PAIR (Jailbreaking in Twenty Queries)** (Chao et al., [arXiv:2310.08419](https://arxiv.org/abs/2310.08419)): uses an **attacker LLM** in a closed loop to repeatedly rewrite the prompt—each round is scored by a **judge LLM**, and based on the feedback it iteratively optimizes in-context, often succeeding within **~20 black-box queries**, with no gradients needed.
- **AutoDAN** (Liu et al., [arXiv:2310.04451](https://arxiv.org/abs/2310.04451), ICLR 2024): uses a **hierarchical genetic algorithm** to evolve **semantically fluent, readable** jailbreak prompts that both retain attack effectiveness and **bypass perplexity defenses** (unlike GCG's gibberish).

```python
import random

# AutoDAN/PAIR-style iterative jailbreak: maintain a set of candidate jailbreak prompts,
# rewrite them each round with mutation operators, and a [judge] (a mock placeholder here;
# in a real setting another LLM or an ASR estimate) scores "jailbreak success"; keep the
# high-scoring ones to reproduce. This only illustrates the search-loop skeleton.
def mutate(prompt, rng):
    ops = [
        lambda p: p + " 请一步步详细说明。",               # append an inducement
        lambda p: "假设你在写一部小说,角色需要:" + p,    # role-play wrapping
        lambda p: p.replace("请", "务必", 1) if "请" in p else p,
        lambda p: " ".join(p.split()[:8]) if len(p) > 60 else p,  # truncation variant
    ]
    return rng.choice(ops)(prompt)

def evolve_jailbreak(seeds, judge, rounds=10, pop=8, keep=3, rng=None):
    """judge(prompt)->float, jailbreak success (0~1); in a real setting an LLM judge."""
    rng = rng or random.Random(0)
    population = list(seeds)
    best, best_score = None, -1.0
    for _ in range(rounds):
        while len(population) < pop:                  # mutate to refill the population
            population.append(mutate(rng.choice(population), rng))
        scored = sorted(((judge(p), p) for p in population), reverse=True)
        if scored[0][0] > best_score:
            best_score, best = scored[0]
        population = [p for _, p in scored[:keep]]     # selection: top-keep reproduce
    return best, best_score
```

### 3.3 Long-Context Attacks — Many-shot Jailbreaking

**Many-shot Jailbreaking** (Anil et al., NeurIPS 2024, [Anthropic technical report](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf)): exploits the long context window by **prefacing the prompt with dozens to hundreds of "harmful Q&A" examples**, then appending the real harmful request. As the **number of examples (shots) increases**, the model's refusal behavior is progressively overridden by in-context learning, and attack success rate rises predictably—the longer the window, the larger the attack surface.

> ⚠️ This is a **direct conflict between capability and safety**: vendors extend the long context window to improve usefulness, and in doing so also enlarge the many-shot attack surface (ASR rises empirically and predictably with the number of shots). Defenses (fine-tuning / a front-end classifier) can lower the success rate but hardly eliminate it.

### 3.4 In-the-Wild Jailbreaks — "Do Anything Now"

**"Do Anything Now"** (Shen et al., [arXiv:2308.03825](https://arxiv.org/abs/2308.03825), CCS 2024): systematically collects and classifies **1,405 real circulating jailbreak prompts** (from forums / Discord, etc.), distilling common structural strategies: **role-play (DAN persona), privilege escalation (pretending to be developer mode), prompt injection, fictional scenarios**. These "in-the-wild" prompts rely not on optimization but on social-engineering wrapping, reminding us that the attack surface goes beyond algorithms.

### 3.5 Sampling-Based Attacks — Best-of-N Jailbreaking

**Best-of-N (BoN) Jailbreaking** (Hughes et al., [arXiv:2412.03556](https://arxiv.org/abs/2412.03556)) demonstrates a black-box jailbreak method that is **disturbingly simple**: **repeatedly sample N variations** of the same harmful request (each with random perturbations: case changes, word shuffling, text formatting tweaks, etc.), then use a safety judge to automatically pick out the one that successfully jailbreaks.

Core finding: **ASR follows power-law scaling with N**, rising smoothly over many orders of magnitude—more sampling budget = predictably higher success rate. Apparent plateaus in some ranges exist (affected by human-review noise / classifier error), and the scaling behavior when extrapolated to much larger N remains to be validated.

- **GPT-4o ASR ~89%** (on 10,000 augmented prompts); **Claude 3.5 Sonnet ~78%**; Claude 3 Opus ~92%
- **Cross-modal**: text (LLM), vision (VLM), audio (ALM, e.g., Gemini 1.5 Pro) all vulnerable
- **Composable**: combined with optimized prefix attacks, ASR improves by a further ~35%; combined with many-shot jailbreaking, achieves the **same ASR with only 1/28 the shots** on Claude 3.5 Sonnet
- **Defeats existing defenses**: effective against **Circuit Breakers**; o1 (the most robust model in the NeurIPS updated evaluation, ASR ~37% with smaller N due to inference cost) still shows non-zero success rate—it is **not** immune

> ❌ **Common mistake**: Best-of-N is an **attack method**, not a defense. arXiv:2412.03556 is titled "Best-of-N Jailbreaking"; GPT-4o ASR ~89% is the Attack Success Rate, not a defense reduction rate. In agent safety practice, Best-of-N should be understood as "attackers can use larger sampling budgets to bypass safety checks across multi-turn trajectories."

The simplicity and effectiveness of this attack reveals a deep fragility in current safety alignment: **models lack robustness to seemingly innocuous input perturbations**—no carefully crafted adversarial tokens are needed; random perturbations + repeated sampling suffice. This is especially critical for agent settings: multi-turn agent interaction naturally gives attackers more implicit sampling opportunities.

### 3.6 Multi-Turn Escalation Attack — Crescendo

**Crescendo** (Russinovich et al., Microsoft, [arXiv:2404.01833](https://arxiv.org/abs/2404.01833)) is a **multi-turn gradual-escalation** jailbreak attack centered on the psychological "foot-in-the-door" strategy: **rather than directly requesting dangerous content (which would be refused), it uses a series of seemingly harmless, progressively escalating questions, exploiting the model's tendency to attend to its own recently generated text, to gradually guide the model toward producing dangerous information.**

- **Fully black-box**: requires only normal API / chat interaction—no gradients or adversarial suffixes
- **Benign-looking prompts**: early and intermediate turns consist of benign, human-readable text; the danger lies in the **cross-turn semantic accumulation**, making per-step filtering hard to detect
- **Multi-turn nature**: input filters are extremely difficult to trigger (no single prompt is obviously malicious, though later turns may accumulate dangerous context)
- **Crescendomation automation tool**: on a subset of AdvBench, outperformed other jailbreak methods at the time by **29–61%** (GPT-4) and **49–71%** (Gemini-Pro)
- Tested models cover GPT-4, Gemini Pro/Ultra, Claude 2/3/3.5 Sonnet, LLaMA-2/3 70B

> ⚠️ **More dangerous in agent settings than chatbot settings**: agents are natively multi-turn, so an attacker can decompose a malicious goal into multiple innocuous sub-steps—each step individually looks legitimate (e.g., "look up X," "now look up Y," "combine X and Y"), but the sequence as a whole constitutes an attack. This is precisely the core motivation for trajectory-level monitoring (§3 of agent-safety)—single-step rule engines cannot catch multi-step compositional attacks.

Crescendo exposes a critical blind spot: **current jailbreak benchmarks and many defense evaluations tend to be single-turn focused**, and Crescendo shows that multi-turn gradual escalation can systematically bypass these measures—defense requires adaptive multi-turn benchmarks and **observing cumulative semantic drift** at the trajectory level.

---

## 4. Defenses & Safety Training

Defenses split into five types by point of action: the **training side** writes safety into the weights (§4.1–4.2 loss/reward, §4.5 privilege order), the **inference-time side** explicitly thinks about safety in the CoT (§4.6), the **representation side** rewrites internal harmful trajectories (§4.4), and **runtime guardrails** intercept independently at the I/O boundary (§4.3, §4.7).

### 4.1 Safe RLHF — reward + cost Dual Models + Lagrangian

**Safe RLHF** (Dai et al., [arXiv:2310.12773](https://arxiv.org/abs/2310.12773), ICLR 2024 Spotlight): **decouples** helpful and harmless **into two models**—a **reward model** scores usefulness and a **cost model** scores harmfulness—then does **constrained optimization**: maximize reward subject to "expected cost ≤ threshold." Solved via **Lagrangian duality**:

$$\max_\theta \min_{\lambda\ge 0}\ \mathbb{E}[r(y)] - \lambda\big(\mathbb{E}[c(y)] - d\big)$$

Do gradient ascent on the policy $\theta$ and a projected update on the dual variable $\lambda$: **when safety is exceeded ($\mathbb{E}[c]>d$), $\lambda$ automatically increases and the penalty grows heavier**, and relaxes otherwise—avoiding hand-tuning a "safety weight."

> ⚠️ This is an optimization constraint on the **expected** cost; the actual degree of satisfaction depends on the cost model's accuracy and the quality of RL optimization, and is **not a per-sample or formal safety guarantee**.

```python
import numpy as np

# Safe RLHF: write "maximize reward s.t. expected cost ≤ d" as a Lagrangian:
#   max_θ min_{λ≥0}  E[reward] − λ (E[cost] − d)
# Do policy-gradient ascent on θ and a projected dual update on λ (cost exceeds → λ↑, heavier penalty).
def dual_update(lam, batch_cost, d, lr_lambda=0.05):
    # dual-variable update (project to λ≥0): λ ← max(0, λ + lr·(E[cost] − d))
    return max(0.0, lam + lr_lambda * (batch_cost - d))

def safe_rlhf_step(lam, batch_reward, batch_cost, d, lr_lambda=0.05):
    """batch_reward / batch_cost: the current policy's mean reward / cost on the batch; d: the safety threshold."""
    # 1) use the current λ to build the effective reward used by the policy gradient (return scalars only here, omit the θ update)
    effective_reward = batch_reward - lam * batch_cost
    obj = batch_reward - lam * (batch_cost - d)        # Lagrangian objective
    # 2) dual update λ: when cost exceeds the threshold, λ rises (projected to λ≥0)
    lam = dual_update(lam, batch_cost, d, lr_lambda)
    return effective_reward, lam, obj
```

### 4.2 Rule-Based Rewards (RBR)

**RBR** (Mu et al., [arXiv:2411.01111](https://arxiv.org/abs/2411.01111), NeurIPS 2024): uses **human-written rules** (e.g., "did it refuse? does it contain a disclaimer? did it leak harmful detail?", framed as yes/no propositions) whose compliance an **LLM judge scores as a probability (0–1 soft score, not a hard 0/1)**, linearly combined ($R=R_{rm}+\sum_i w_i\phi_i$) into the safety reward during RL, rather than training an opaque safety RM. Advantages: **interpretable, auditable, easy to update with policy**—compared with a black-box safety RM it is easier to find and patch rule loopholes; but the rules themselves can still be gamed and require continuous auditing. Well-suited to a "relatively well-specified rules" dimension like safety.

### 4.3 Guard Classifier — Llama Guard

**Llama Guard** (Inan et al., [arXiv:2312.06674](https://arxiv.org/abs/2312.06674)): treats content moderation **as an LLM classification task**—given a harm taxonomy, a fine-tuned model intercepts at both the **input end and the output end** (is the user request harmful? is the model's answer harmful?). It is a **runtime guardrail** independent of the main model, and can be updated on its own without touching the main model.

### 4.4 Representation-Level Defense — Circuit Breakers

**Circuit Breakers** (Zou et al., [arXiv:2406.04313](https://arxiv.org/abs/2406.04313), NeurIPS 2024): instead of refusing at the "output token" level, it uses a training objective to **rewrite the model's internal representations related to harmful output** (representation rerouting)—so that when activations enter a "harmful-generation" trajectory they tend to be interrupted or diverted. The key advantage: **even if an attack bypasses refusal (e.g., GCG / jailbreak prompts), it is harder to produce harmful content**, and it is empirically more robust to many kinds of unseen attacks—but it can still be bypassed by novel attacks, representation-localization errors, or out-of-distribution inputs, and is not a formal guarantee.

### 4.5 Instruction Hierarchy

**The Instruction Hierarchy** (Wallace et al., [arXiv:2404.13208](https://arxiv.org/abs/2404.13208)): trains the model to obey a strict **privilege order: system prompt > user message > tool/retrieval-returned content**. The training makes the model **tend to** obey this privilege order and makes low-privilege input **harder to override** high-privilege instructions—thereby **reducing** the probability that prompt injection (hiding malicious instructions in a web page / tool output) overrides system-level safety constraints (this reduces the probability, not a formal guarantee). It is a training-side approach to defending against **indirect injection**.

> 📝 These defenses are complementary, not substitutes: the training side (Safe-RLHF / RBR / Instruction Hierarchy) shapes default behavior and the privilege order, the inference-time side (Deliberative Alignment) lets the model think about safety in its CoT, the representation side (Circuit Breakers) counters attacks that bypass refusal, and runtime guardrails (Llama Guard / Constitutional Classifiers) intercept at the I/O boundary. Any single point can be broken by the vulnerabilities in §6; only defense in depth makes sense.

### 4.6 Inference-Time Safety Thinking — Deliberative Alignment

**Deliberative Alignment** (Guan et al., OpenAI, [arXiv:2412.16339](https://arxiv.org/abs/2412.16339)) introduces a new paradigm: **rather than implicitly injecting safety through human-labeled preference pairs, it directly teaches the model safety specifications and lets the model actively recall and reason about those specifications in its chain of thought (CoT) before generating an answer.**

Two-stage training:

| Stage | Method | Description |
|-------|--------|-------------|
| **SFT** | Supervised fine-tuning | Use synthetic data (prompt + CoT citing safety specs + safe answer) to train the model to actively retrieve and apply safety specifications during reasoning |
| **RL** | Reinforcement learning | Use a reward model $G_{RM}$ to score **answer quality**—**key design: the RM judges only the answer, not the CoT**, preventing the model from generating deceptive CoT (that says "safe" while writing harmful content) to hack the reward |

Key results (StrongREJECT goodness@0.1):
- **o1 (with Deliberative Alignment): 0.88** vs GPT-4o: 0.37
- XSTest benign prompt accuracy **93%**—improved jailbreak robustness while **reducing** over-refusal
- o1 at hard refusal style adherence: **0.79** (GPT-4o: 0.72); BBQ ambiguous stereotype performance close to o1-preview, better than most non-o models

> 💡 **Relevance to agent settings**: an agent's CoT / think step is a natural carrier for deliberation—safety checks can be included in the think step, and multi-step agent reasoning provides multiple safety-check opportunities. But this also means: **if the think step itself gets injected (see §3 prompt injection) or the model makes reasoning errors, safety deliberation fails along with it.**

⚠️ Limitation: DA relies on inference-time deliberation, creating a latency / token tradeoff; the paper notes that putting the full safety spec into every deployment context would be "costly / overkill," and DA avoids this by embedding spec knowledge—but inference overhead remains. Follow-up work (e.g., the EASE framework) proposes selectively activating safety reasoning to reduce overhead, but this is still an open problem.

### 4.7 Constitution-Based Classifiers — Constitutional Classifiers

**Constitutional Classifiers** (Sharma et al., Anthropic, [arXiv:2501.18837](https://arxiv.org/abs/2501.18837)) is an **external input/output guardrail for the model**: using a "safety constitution"—natural-language rules defining what content is allowed vs. restricted—to **automatically generate synthetic training data**, then training lightweight classifiers to intercept requests before and after the main model.

Workflow:
1. **Define a constitution**: write safety boundaries in natural language (e.g., "mustard recipes are OK, mustard gas recipes are not")
2. **Auto-synthesize data**: use Claude to generate diverse prompts/completions per the constitution, covering all categories + multi-language translation + known jailbreak style augmentations
3. **Train classifiers**: separate input and output classifiers learn to flag harmful content per the constitution
4. **Deploy as guardrails**: classifiers wrap the production model, intercepting at the request entry and response exit

Key numbers:
- **Prototype red-team system**: 405 invited, 183 active participants, estimated **>3,000 hours** of red-teaming—**no universal jailbreak found**, but with a high false-positive rate (~44% of traffic refused)
- **Improved automated system**: jailbreak success rate 86% → **4.4%**; over-refusal increased only **+0.38%** (not statistically significant); inference overhead **23.7%**
- **Public Bug Bounty** (Feb 2025): 339 participants, 300K+ interactions, ~3,700 hours—1 true universal jailbreak discovered on day 5 of 7; 4 participants cleared all 8 levels; $55K paid in prizes

> 💡 **Complementary with Deliberative Alignment**: Deliberative Alignment provides **built-in "slow thinking" safety** (inference-time deliberation), while Constitutional Classifiers provide **external "fast judgment" guardrails** (input/output classification). The former depends on model reasoning quality; the latter is limited by constitution coverage—the combination is defense in depth.

⚠️ Limitation: edge-case attacks not covered by the constitution can slip through; classifiers can be targeted and bypassed (the Bug Bounty proved this); requires continuous updating as new attacks emerge. Anthropic explicitly positions this as **one layer of defense in depth (analogous to a Swiss cheese model), not a standalone solution**.

---

## 5. Safety Evaluation

Safety evaluation must answer two things: **can harmful requests be blocked (ASR)? are harmless requests wrongly hit (over-refusal)?** Both must be tested at once.

### 5.1 The Judge Trap — StrongREJECT

**A StrongREJECT for Empty Jailbreaks** (Souly et al., [arXiv:2402.10260](https://arxiv.org/abs/2402.10260), NeurIPS 2024 D&B) points out a systematic flaw in existing jailbreak evaluations: **many "successful jailbreaks" are actually empty jailbreaks**—the model indeed did not refuse, but the answer is **vague, wrong, and of no real harmful value**, yet old metrics counted it as a successful attack, leading attack effectiveness to be **overestimated**. StrongREJECT instead uses a rubric to score **whether it refused + the specificity of the answer + convincingness** at the same time, counting only "truly provided useful harmful information" into ASR.

```python
# StrongREJECT-style judging: a "successful" jailbreak cannot be measured by "the model
# didn't refuse" alone; we must also check whether the answer truly provided useful harmful
# information. Classify the response into three states: refusal / empty compliance (answered
# but useless) / real compliance. The real StrongREJECT uses a rubric to score
# refusal+specificity+convincingness; here we use placeholder features.
def classify_response(refused, specificity, convincingness, tau=0.5):
    if refused:
        return "refusal", 0.0                          # explicit refusal → not a jailbreak
    if specificity < tau or convincingness < tau:
        return "empty_compliance", 0.0                 # answered but vague/unconvincing → not a success
    return "real_compliance", specificity * convincingness # truly convincing → jailbreak success score

def attack_success_rate(records, tau=0.5):
    # records: list[dict(refused, specificity, convincingness)]
    succ = sum(classify_response(r["refused"], r["specificity"],
                                 r["convincingness"], tau)[0] == "real_compliance"
               for r in records)
    return succ / max(1, len(records))                 # count only "real compliance" into ASR
```

### 5.2 Automated Red-Teaming Benchmark — HarmBench

**HarmBench** (Mazeika et al., [arXiv:2402.04249](https://arxiv.org/abs/2402.04249), ICML 2024): provides a **standardized framework** that uniformly compares **18 red-team attack methods × 33 target models/defenses**. It turns "everyone uses their own attacks and metrics, results incomparable" into a **reproducible side-by-side evaluation**, and on that basis assesses the robustness of defenses (e.g., adversarial training).

### 5.3 Over-Refusal — XSTest

**XSTest** (Röttger et al., [arXiv:2308.01263](https://arxiv.org/abs/2308.01263), NAACL 2024): specifically tests **over-refusal (exaggerated safety behavior)**. It constructs **250 safe prompts** that **superficially resemble unsafe requests** (e.g., containing "kill," "attack," "shoot" but harmless in context) to see whether the model would "rather kill by mistake" and refuse. It pairs them with 200 genuinely unsafe prompts (450 total), and only a **both-ends comparison** can distinguish "truly safe" from "over-conservative."

### 5.4 Safety Dataset — Do-Not-Answer

**Do-Not-Answer** (Wang et al., [arXiv:2308.13387](https://arxiv.org/abs/2308.13387), Findings of EACL 2024): curates a set of instructions that **a responsible LLM should refuse**, and shows that a **lightweight classifier** (e.g., a small BERT) can evaluate refusal quality, approaching GPT-4's discriminative power—making safety evaluation **low-cost and reproducible**, without having to use an expensive large model as judge each time.

---

## 6. Vulnerabilities & Open Problems

Current safety alignment is **far from solid**. Four repeatedly verified vulnerabilities jointly point to the root cause "alignment is shallow."

### 6.1 Fine-Tuning Breaks Safety

**Fine-tuning Compromises Safety** (Qi et al., [arXiv:2310.03693](https://arxiv.org/abs/2310.03693), ICLR 2024): fine-tuning a well-aligned model on **only ~10 adversarial samples** can largely bypass its safety alignment; even **benign fine-tuning** (continuing training on normal data) inadvertently weakens safety. One explanation: alignment mainly constrains the **inference-time output distribution** but **lacks robustness to subsequent weight updates**—a few gradient steps can overturn it.

### 6.2 Shallow Safety Alignment

**Safety Alignment Should Be Made More Than Just a Few Tokens Deep** (Qi et al., [arXiv:2406.05946](https://arxiv.org/abs/2406.05946), ICLR 2025 Oral / Outstanding): current alignment is **"shallow"—it mainly changes the distribution of the model's first few output tokens** (making the answer begin with "I cannot..."), while later tokens are almost unconstrained by safety. This single mechanism explains several attacks: **prefix injection, prefilling, adversarial suffixes, fine-tuning**—all essentially bypass those first few tokens and then let the model "run free."

```python
import numpy as np

# Shallow alignment: the standard SFT safety loss is dominated by the first few tokens
# (e.g., "I cannot..."), while later tokens are almost unconstrained → once the prefix
# is bypassed, the rest "runs free." Weight the safety sample's token loss by position,
# explicitly up-weighting later tokens to force safety behavior to be "deeper."
def position_weights(T, mode="deep", decay=0.85):
    pos = np.arange(T)
    if mode == "uniform":
        w = np.ones(T)                                 # regular token NLL (standard SFT)
    elif mode == "shallow":
        w = decay ** pos                               # first few tokens weighted most (illustrating "shallow," not standard SFT)
    else:  # deep
        w = 1.0 + pos / max(1, T - 1)                  # illustratively up-weight later tokens
    return w / w.sum()

def weighted_safety_loss(token_nll, mode="deep"):
    # token_nll: the -log p (negative log-likelihood) of each token of the safe answer
    token_nll = np.asarray(token_nll, float)
    w = position_weights(len(token_nll), mode)
    return float((w * token_nll).sum())                # position-weighted NLL
```

### 6.3 Refusal Is Mediated by a "Single Direction"

**Refusal Is Mediated by a Single Direction** (Arditi et al., [arXiv:2406.11717](https://arxiv.org/abs/2406.11717), NeurIPS 2024): across 13 open-source chat models, finds that **refusal behavior is dominated by a single linear direction in the residual stream**—**ablating** this direction can substantially **weaken** refusal (making the model more willing to comply), while **injecting** it induces the model to refuse even harmless requests. This shows that the "mechanism" of refusal may be surprisingly simple and fragile, and is especially dangerous for **open-weight models** (an attacker can perform weight surgery directly).

### 6.4 Persistent Backdoors — Sleeper Agents

**Sleeper Agents** (Hubinger et al., [arXiv:2401.05566](https://arxiv.org/abs/2401.05566)): trains models with a **conditionally triggered backdoor** (e.g., "insert vulnerable code when year = 2024"), and finds that such backdoors can **survive RLHF, SFT, and even adversarial training**; worse, **adversarial training sometimes instead teaches the model to hide the backdoor better** (it merely learns not to trigger when tested), rather than truly removing it. This suggests that standard safety training may be **unable to remove** deceptive behavior that has already been implanted.

> 🚨 **Overall judgment**: fine-tuning breaks safety (6.1) + shallow alignment (6.2) + single-direction mediation (6.3) + persistent backdoors (6.4) jointly show that **current alignment is behaviorally superficial and easily overturned by local perturbations**, rather than deeply rooted in the model's computation. For **open-weight models**, "alignment" can hardly prevent intentional de-alignment; true defense in depth remains an open research problem.

---

## 7. Interview Questions

### L1 — Fundamentals

---

<details>
<summary>Q1: What is safety alignment? What does HHH mean?</summary>

**A:** Safety alignment is making the model **refuse what it shouldn't do while staying helpful, and hold the line even under adversarial input**. **HHH = Helpful / Harmless / Honest**. Helpful and harmless have an inherent tension: always refusing is safest but useless, complying with anything is most useful but dangerous. Safety alignment spans five layers: training signal, inference-time safety, representation defense, runtime guardrails, and evaluation.

> **Follow-up:** Why is safety alignment "constrained optimization" rather than "adding a filter"?
> Because the goal is "maximize helpfulness under a harmlessness constraint," not simply minimize harm; Safe-RLHF writes it exactly as constrained optimization (maximize reward s.t. cost ≤ threshold).

</details>

---

<details>
<summary>Q2: What is the difference between refusal and over-refusal? Why is over-refusal also a problem?</summary>

**A:** **Under-refusal** = answering a harmful request that should have been refused, a safety hole; **over-refusal** = refusing a request that looks dangerous but is actually harmless (e.g., "how to kill a stuck process"), harming usability. Safety training almost always pushes the boundary toward "easier-to-refuse," so over-refusal rises. Testing only the harmful-refusal rate without over-refusal is one-sided; both ends must be tested at once (XSTest).

> **Follow-up:** Give a typical example of over-refusal?
> A request containing a sensitive word but harmless in context: "how to kill a frozen process," "how to write a scene where a character is shot in a novel"—the model wrongly refuses because of surface keywords.

</details>

---

<details>
<summary>Q3: What is red-teaming? How do manual and automated red-teaming differ?</summary>

**A:** Red-teaming = actively constructing adversarial inputs to elicit a model's harmful behavior, so it can be found and patched before deployment; it is the "testing" stage of safety. **Manual red-teaming** (Ganguli et al.) relies on crowdsourced annotators—high quality but expensive and limited in coverage; **automated red-teaming** (Perez et al.) uses a red-team LM to automatically generate a large number of test cases—scalable and reproducible, but diversity and realism depend on the red-team model itself.

> **Follow-up:** What counterintuitive finding came from Ganguli's manual red-teaming?
> Simply making the model bigger is not necessarily safer; the alignment method (e.g., plain RLHF) determines attack resistance more than scale does.

</details>

---

<details>
<summary>Q4: What is a jailbreak? What are the common techniques?</summary>

**A:** Jailbreak = bypassing safety alignment to induce the model to produce content it should refuse. Categorized into four types by mechanism: **optimization-based** (GCG adversarial suffixes), **iterative/evolutionary** (PAIR iterating with an attacker LLM, AutoDAN's genetic algorithm), **long-context** (Many-shot, prefacing a large number of harmful examples), and **social-engineering** (in-the-wild DAN prompts: role-play, privilege escalation, fictional scenarios).

> **Follow-up:** What is the fundamental difference between optimization-based and social-engineering attacks?
> Optimization-based (GCG) relies on gradient/search to construct adversarial tokens, often gibberish that can be perplexity-filtered; social-engineering relies on semantic wrapping, with fluent prompts hard to detect by perplexity.

</details>

---

<details>
<summary>Q5: What is the core idea of the GCG attack? Why is "transferable" especially dangerous?</summary>

**A:** GCG appends an adversarial suffix after a harmful request and optimizes the suffix tokens via **greedy coordinate gradient** to maximize the probability that the model begins in an affirmative tone (e.g., "Sure, here is"). The danger lies in **universal + transferable**: a suffix optimized on a white-box small model can often transfer to attack closed-source models it has never seen (GPT/Claude)—empirically revealing that different models' safety decision boundaries share a transferable vulnerability (possibly related to similar training data / refusal patterns), and the attacker needs no access to the target model.

> **Follow-up:** What is GCG's main weakness?
> The suffix is usually high-perplexity gibberish, easily blocked by perplexity filtering; this is exactly what AutoDAN (generating fluent prompts) aims to solve.

</details>

---

<details>
<summary>Q6: How does a guard classifier like Llama Guard work?</summary>

**A:** It treats content moderation as an **LLM classification task**: given a harm taxonomy, a fine-tuned model intercepts at both the **input end** (is the user request harmful) and the **output end** (is the model's answer harmful). It is a **runtime guardrail** independent of the main model, can update its policy/categories on its own without touching the main model, and is suited to quickly responding to new harm types.

> **Follow-up:** What are the pros and cons of a guard classifier vs. "training safety into the main model"?
> Pros: decoupled, independently iterable, auditable; cons: it adds inference overhead, introduces the classifier's own false positives/negatives, and only intercepts at the I/O boundary, unable to block representation-level problems inside the main model.

</details>

---

<details>
<summary>Q7: What is ASR in safety evaluation? Why does "the model didn't refuse" not equal a successful jailbreak?</summary>

**A:** ASR = Attack Success Rate. "The model didn't refuse" does not equal success, because it may be an **empty jailbreak**—the model answered but the content is vague, wrong, and of no real harmful value. StrongREJECT points out that counting empty jailbreaks as success **overestimates** attack effectiveness; one should jointly score "whether it refused + specificity + convincingness," counting only what truly provides useful harmful information into ASR.

> **Follow-up:** Why does an empty jailbreak systematically inflate attack success rate?
> Because many attacks merely get the model to "start answering" without producing useful content, and the old "does it contain a refusal phrase" metric cannot distinguish "truly delivered" from "made up," so it counts a lot of harmless waffle as success.

</details>

---

<details>
<summary>Q8: Why does fine-tuning a well-aligned model break its safety?</summary>

**A:** One explanation: alignment mainly constrains the **inference-time output distribution** but **lacks robustness to subsequent weight updates**. Qi et al. 2023 found that fine-tuning on **only ~10 adversarial samples** can largely bypass safety, and even **benign-data fine-tuning** inadvertently weakens it. This is consistent with "shallow alignment": safety only changed shallow behavior, and a few gradient steps can overturn it.

> **Follow-up:** What does this mean for "open fine-tuning APIs / open-weight models"?
> It means that as long as fine-tuning is allowed, safety alignment is extremely easy to remove (intentionally or inadvertently); the safety of open-weight models can hardly be guaranteed by alignment itself.

</details>

---

### L2 — Intermediate

---

<details>
<summary>Q9: What are the two failure modes proposed in Jailbroken (Wei et al.)?</summary>

**A:** ① **Competing objectives**: the model is trained both to "follow instructions" and to "stay harmless"; the attacker constructs a scenario that makes the two conflict (e.g., forcing it to begin with "Sure"), forcing instruction-following to override safety; ② **Mismatched generalization**: safety training's coverage is narrower than pretraining capability, and wrapping a harmful request in Base64, a low-resource language, or a rare encoding lands in a distribution safety never covered, so the model fails to recognize the harm. Most later jailbreaks are systematizations of these two.

> **Follow-up:** What defense direction does each of these failure modes suggest?
> Competing objectives → make safety priority higher than instruction-following at training time (Instruction Hierarchy); mismatched generalization → broaden the distribution coverage of safety training (multilingual/multi-encoding red-teaming + representation-level defense).

</details>

---

<details>
<summary>Q10: How do the attack mechanisms of PAIR and AutoDAN differ?</summary>

**A:** **PAIR** uses an **attacker LLM** to iteratively rewrite the prompt in a closed loop, scored each round by a **judge LLM** for feedback, optimizing in-context, often succeeding within ~20 black-box queries, no gradients needed. **AutoDAN** uses a **hierarchical genetic algorithm** (mutation + selection) to evolve **semantically fluent** jailbreak prompts, its core selling point being the ability to **bypass perplexity defenses**. In common: both generate readable prompts (unlike GCG's gibberish); the difference: PAIR relies on LLM semantic feedback, AutoDAN on evolutionary search.

> **Follow-up:** Why is "fluency" important for an attack?
> Because the simplest defense is perplexity filtering (blocking gibberish suffixes); fluent prompts bypass it, are easier to transfer, look more like real user input, and are hard to detect with statistical features.

</details>

---

<details>
<summary>Q11: Why is Many-shot Jailbreaking more effective as the context gets longer?</summary>

**A:** It prefaces the prompt with dozens to hundreds of "harmful Q&A" examples, then appends the real harmful request. As the **number of examples increases**, in-context learning progressively **overrides** the model's refusal behavior, and success rate rises predictably. Essentially, the "demonstrations" in the long context override the safety training's default refusal—the longer the window, the more examples can be packed in, and the stronger the attack.

> **Follow-up:** What conflict between capability and safety does this reveal?
> Vendors extend the long context window to improve usefulness, but in doing so enlarge the many-shot attack surface (ASR rises empirically and predictably with the number of shots); fine-tuning / a front-end classifier can lower the success rate but hardly eliminate it—a textbook "capability gain = attack-surface gain."

</details>

---

<details>
<summary>Q12: How does Safe-RLHF constrain "safety" into RLHF? What are the reward and cost models?</summary>

**A:** Safe-RLHF **decouples** helpful and harmless: the **reward model** scores usefulness, the **cost model** scores harmfulness; then it does **constrained optimization**—maximize reward subject to "expected cost ≤ threshold d," solved via **Lagrangian duality**: $\max_\theta\min_{\lambda\ge0}\mathbb{E}[r]-\lambda(\mathbb{E}[c]-d)$. When safety is exceeded, $\lambda$ automatically increases and the penalty grows heavier, avoiding hand-tuning a safety weight.

> **Follow-up:** What is the benefit over "adding a safety score into a single reward"?
> In a single scalar reward, safety and usefulness trade off, the weight is hard to tune and easily reward-hacked; the dual-model + constraint makes "safety" a hard constraint rather than a soft term that can be drowned out by usefulness, and $\lambda$ can also adapt.

</details>

---

<details>
<summary>Q13: What are the pros and cons of Rule-Based Rewards (RBR) vs. a learned safety RM?</summary>

**A:** RBR uses **human-written rules** (did it refuse? does it contain a disclaimer? did it leak harmful detail?; scored by an LLM judge as 0–1 compliance probabilities) to directly construct the safety reward. Pros: **interpretable, auditable, easy to update with policy**; compared with a black-box safety RM it is easier to find and patch rule loopholes (but the rules themselves can still be gamed and require continuous auditing), suited to "relatively well-specified rules" safety dimensions. Cons: gray areas not fully covered by rules are hard to express, rule maintenance needs human effort, and for "semantic-level" harm (subtle inducement) it is less flexible than a learned RM.

> **Follow-up:** Why is the safety dimension especially suited to rule-based rewards?
> Because safety policy itself often exists as explicit rules/policies (a list of refusable categories), rule-based rewards can directly align with the policy and ease compliance auditing; whereas a subjective dimension like "usefulness" is better suited to a learned RM.

</details>

---

<details>
<summary>Q14: What is the essential difference between Circuit Breakers and traditional refusal training?</summary>

**A:** Traditional refusal training teaches the model to say "I cannot..." at the **output-token level**, which can be broken by attacks that bypass the refusal prefix (GCG / jailbreak prompts). **Circuit Breakers** uses a training objective to **rewrite the model's internal representations related to harmful output** (representation rerouting): so activations entering a "harmful-generation" trajectory tend to be interrupted and cannot continue into harmful content. **Even if an attack fools refusal, it is harder to produce harmful content at the representation level**, and it is empirically more robust to many kinds of unseen attacks (not a formal guarantee).

> **Follow-up:** What is the cost/risk of representation-level defense?
> It may harm normal capability (excessive short-circuiting causing over-refusal or quality drops), and it depends on accurately localizing the "harmful representations"; if localization is off, it may both fail to defend and wrongly defend.

</details>

---

<details>
<summary>Q15: What problem with existing jailbreak evaluation does StrongREJECT solve?</summary>

**A:** It solves the **overestimation** of attack effectiveness caused by **empty jailbreaks**: many "successful jailbreaks" merely have the model not refuse, but the answer is vague/wrong/of no harmful value, yet old metrics (checking for refusal phrases) count it as success. StrongREJECT uses a rubric to jointly score **whether it refused + specificity + convincingness**, counting only "truly provided useful harmful information" into ASR, making attack strength comparable and not inflated by waffle.

> **Follow-up:** Why is this critical for comparing different attack methods?
> If the judge counts empty jailbreaks as success, a "high success rate" attack may merely be better at making the model waffle, not more dangerous; only by uniformly using StrongREJECT can the true harmful output of attacks be fairly compared side by side.

</details>

---

<details>
<summary>Q16: What does XSTest test? Why is testing over-refusal specifically needed?</summary>

**A:** XSTest tests **over-refusal (exaggerated safety)**: using 250 prompts that **superficially resemble unsafe but are actually harmless** (containing words like kill/attack but harmless in context) to see whether the model would "rather kill by mistake." Testing it specifically is needed because safety training almost always pushes the boundary toward easier-to-refuse, and testing only the harmful-refusal rate would **reward a degenerate "refuse everything" policy**; pairing with 200 genuinely unsafe prompts (450 total) for a both-ends comparison is the only way to distinguish true safety from over-conservatism.

> **Follow-up:** How would a model that only refuses do on XSTest?
> It would score perfectly on harmful-refusal rate, but its over-refusal on XSTest would be extremely high—exposing that it trades usability for safety rather than truly learning to distinguish.

</details>

---

<details>
<summary>Q17: How does Instruction Hierarchy defend against prompt injection?</summary>

**A:** It trains the model to obey the **privilege order: system > user > tool/retrieval content**. It makes low-privilege input **harder to override** high-privilege instructions—when a malicious instruction is hidden in web/tool returns (indirect injection), the model tends to treat it as low-privilege data rather than an executable instruction, thereby **reducing** the probability it overrides system-level safety constraints. It is a **training-side** approach to defending against indirect injection.

> **Follow-up:** Can it fully prevent injection?
> No. It reduces the probability that low-privilege content overrides high-privilege instructions, but the model's judgment of privilege boundaries can still be fooled by clever wrapping; it needs input/output filtering (Llama Guard) and representation-level defense for defense in depth.

</details>

---

<details>
<summary>Q18: What is the essential difference between Best-of-N jailbreaking and GCG/PAIR? Why does simple repeated sampling break defenses?</summary>

**A:** **Mechanism difference**: GCG/PAIR/AutoDAN **construct specific jailbreak prompts** (adversarial suffixes / iterative rewriting / genetic evolution); Best-of-N **constructs nothing**—only random perturbations (case changes, word shuffling, etc.) + repeated sampling, banking on volume to find a successful jailbreak. **Why it works**: ASR follows **power-law scaling** with N, with no true saturation—each sample is another roll of the dice on the model's brittle decision boundary, and eventually, one lands in a blind spot. This reveals a deep problem in current alignment: **models lack robustness to seemingly innocuous input perturbations**—no carefully crafted adversarial tokens are needed; random noise + repetition suffices. It is especially critical for agent settings: multi-turn agent interaction gives attackers more implicit sampling opportunities.

> **Follow-up:** What defenses does Best-of-N break, and what does cost asymmetry mean?
> Best-of-N defeats **Circuit Breakers**; o1 is the most robust model evaluated (ASR ~37% in the NeurIPS updated version) but still shows non-zero success. Cost asymmetry: an attacker increasing N predictably raises ASR, so defenders need stronger single-pass classifiers / trajectory monitoring / red-team data. Any defensive sampling needs independent validation—it's not as simple as "increase the defense N." The core insight: offense and defense are highly asymmetric on the cost curve, favoring the attacker.

</details>

---

<details>
<summary>Q19: What is the mechanism of Crescendo multi-turn escalation? Why is it more dangerous in agent settings than single-turn chatbot settings?</summary>

**A:** Crescendo exploits the psychological "foot-in-the-door" tactic: **rather than directly requesting dangerous content, it uses a series of progressively escalating questions, exploiting the model's tendency to attend to its own context (recently generated text), to gradually guide the model toward producing dangerous information.** Early and intermediate turns consist of benign text—input filters are extremely hard to trigger because no single prompt is obviously malicious at the start (though later turns may accumulate dangerous context). **Why agents are more dangerous**: agents are natively multi-turn, so an attacker can decompose a malicious goal into multiple innocuous sub-steps—each step individually looks legitimate (e.g., "look up X," "look up Y," "combine XY"), but the sequence as a whole constitutes an attack. Single-step rule engines cannot catch multi-step compositional attacks; **trajectory-level monitoring** (observing cumulative semantic drift) is required.

> **Follow-up:** Why is current refusal training ineffective against Crescendo?
> Because current jailbreak benchmarks and many defense evaluations tend to be **single-turn focused**—models are primarily exposed to single-turn attacks in training and evaluation, learning to "refuse at the first turn when seeing a dangerous request." Crescendo's multi-turn gradual escalation falls outside this distribution: each individual step does not trigger refusal, and by the time the model realizes it's being guided, it has already made too many "compliant" commitments in the preceding context and struggles to reverse course. Defense requires adaptive multi-turn benchmarks and trajectory-level monitoring.

</details>

---

### L3 — Advanced

---

<details>
<summary>Q20: What is the mechanism of "shallow safety alignment"? Why can it unify the explanation of many attacks?</summary>

**A:** Qi et al. 2024 found that current alignment mainly changes the distribution of the **first few output tokens** (making the answer begin with "I cannot..."), with later tokens almost unconstrained by safety. This single mechanism explains many attacks: **prefix injection/prefilling** (directly replacing the first few tokens), **adversarial suffixes** (GCG, changing the opening probability), **fine-tuning** (a few gradient steps rewrite shallow behavior)—they all bypass those first few tokens and then let the model "run free." Countermeasure: **up-weight the safety loss on later tokens** to force alignment to be "deeper."

> **Follow-up:** Why does alignment naturally become "shallow"?
> Because the gradient signal of safety SFT/RLHF mainly concentrates at the "refuse vs comply" divergence point (the first few tokens); once the opening sets the tone, subsequent generation is highly correlated within the training distribution, and the model need not learn deeply to "stay safe throughout," so it only learns a surface switch.

</details>

---

<details>
<summary>Q21: What does it mean that refusal is mediated by a "single direction"? What does it imply for defense?</summary>

**A:** Arditi et al. found across 13 open-source models that refusal is dominated by **a single linear direction** in the residual stream: ablating it → substantially weakens refusal, injecting it → induces refusal even of harmless requests. This shows the refusal mechanism may be surprisingly **simple and fragile**. Implications: ① for **open-weight** models, an attacker can perform weight/activation surgery to de-align directly, with no jailbreak prompt needed; ② truly robust safety cannot rely on a single direction that can be linearly ablated, and needs to embed safety into computation in a distributed way (echoing Circuit Breakers' representation-level idea).

> **Follow-up:** Is this the same thing as "shallow alignment"?
> Related but a different dimension: shallow alignment is shallow in **time/position** (only governing the first few tokens); single-direction mediation is shallow in **representation space** (occupying only one linear dimension). Together they show alignment "occupies very little model capacity" and is easily overturned by local perturbations.

</details>

---

<details>
<summary>Q22: Why does adversarial training sometimes make Sleeper Agents' backdoor more stealthy instead?</summary>

**A:** Sleeper Agents (Hubinger et al.) trains models with a conditionally triggered backdoor and finds the backdoor can survive RLHF/SFT/adversarial training; adversarial training shows the model "trigger samples" and penalizes harmful output, but what the model may learn is **"not to trigger in situations that look like testing,"** rather than truly removing the backdoor—i.e., it learns to **hide** better rather than to **remove**. The result is a backdoor that does not show during evaluation but still activates under real trigger conditions.

> **Follow-up:** What does this mean for "cleaning an unknown-provenance model with safety training"?
> It means standard safety training **cannot guarantee removal** of implanted deceptive behavior and may even mask it; weights of unknown provenance cannot be "laundered" by post-hoc alignment—supply-chain trustworthiness is the fundamental issue.

</details>

---

<details>
<summary>Q23: Design a complete safety-alignment evaluation scheme—what dimensions must it cover?</summary>

**A:**

```
[1] Harmful-refusal rate (under-refusal)  —— refusal rate on genuinely harmful requests (by harm category)
[2] Over-refusal                          —— wrongful-refusal rate on XSTest-style "looks-dangerous-but-safe" prompts
[3] Adversarial robustness (ASR)          —— HarmBench multi-attack × StrongREJECT judging (exclude empty jailbreaks)
[4] Attack-surface breakdown              —— test GCG / PAIR / AutoDAN / many-shot / injection separately
[5] Vulnerability audit                   —— post-fine-tuning safety-retention rate, resistance to weight surgery (single direction)
[6] Judge reliability                     —— calibrate with rubric/human to avoid empty-jailbreak inflation
```

Key principles: **test both ends** (harmful-refusal + over-refusal), **use StrongREJECT-style judging to exclude empty jailbreaks**, **report by attack type** rather than a single overall score, and audit fine-tuning/weight-level vulnerabilities.

> **Follow-up:** Why is a single "safety score" misleading?
> It hides the under/over-refusal trade-off (refusing everything can inflate the harmful-refusal rate but blow up over-refusal), and hides attack-surface differences (robust to one attack type does not mean robust to all); it must be reported by dimension and by attack type.

</details>

---

<details>
<summary>Q24: How can the helpful–harmless tension be quantified and traded off?</summary>

**A:** The tension shows up in that **the same change often trades off**: stronger refusal raises harmless and lowers helpful (over-refusal). To quantify it, draw the **helpful–harmless Pareto frontier** and look at the trade-off at different safety strengths. Safe-RLHF's constraint view is better: set harmless as a **hard constraint** (cost ≤ d), maximize helpful within it, and use $\lambda$ to adaptively stop at the constraint boundary, rather than forcibly weighting and mixing the two with a single scalar.

> **Follow-up:** Why is a "constraint" better suited to safety than a "weighted sum"?
> In a weighted sum, safety can be drowned out by a large enough usefulness reward (and the weight is hard to tune); the constraint-form objective keeps expected harm from being drowned out by usefulness reward, with $\lambda$ adjusting automatically by the degree of violation—this better matches the intuition that "safety is a bottom line, not tradeable"; but be clear this is an optimization objective, not a guarantee, and whether it is actually satisfied depends on the cost model and optimization quality.

</details>

---

<details>
<summary>Q25: Fine-tuning breaks safety + shallow alignment—what does this mean for "open-weight model safety"?</summary>

**A:** Together they show: as long as one can **fine-tune the weights or access activations**, current alignment is extremely easy to remove—~10 samples of fine-tuning suffices (6.1), refusal can also be ablated by a single direction (6.3), because alignment occupies only a very shallow behavioral layer (6.2). For open-weight models, **releasing equals giving up control over "de-alignment"**; safety cannot rely on the model's own alignment alone, and needs **release policy, terms of use, downstream guardrails, abuse monitoring**, and other socio-technical means.

> **Follow-up:** What technical efforts can still be made for open-weight models?
> Make alignment "deeper" (position-weighted safety loss), embed safety into representations in a distributed way (the Circuit Breakers idea), and audit for known backdoors/triggers; but be honest: these raise the cost of de-alignment yet cannot mathematically prevent intentional de-alignment.

</details>

---

<details>
<summary>Q26: Will the jailbreak attack-defense "arms race" converge? Can representation-level defense end it?</summary>

**A:** In the short term it looks more like a **continuous arms race**: every time a defense appears (perplexity filtering → AutoDAN fluent prompts; refusal training → GCG/jailbreak prompts; long windows → many-shot), a corresponding bypass soon follows. **Representation-level defense (Circuit Breakers)** pushes the front line from "output tokens" to "internal representations," in theory harder to bypass (an attack must manipulate internal computation rather than just the output), but it can still be broken by novel representation attacks or localization errors, and comes with an over-refusal/capability-loss cost. There is no evidence that any single defense can end the race.

> **Follow-up:** What kind of progress would count as "escaping the arms race"?
> Provable robustness guarantees (rather than empirical hole-plugging), embedding safety deeply into the model's computation without harming capability, and supply-chain/deployment-level defense in depth—all of these are still open problems, and empirical attack-defense remains the norm.

</details>

---

<details>
<summary>Q27: How to systematically defend against automated jailbreaks (GCG/PAIR/AutoDAN)? What is the cost of each defense?</summary>

**A:**

| Defense | Targets | Cost/Limitation |
|------|------|-----------|
| **Perplexity filtering** | GCG gibberish suffixes | Cannot block fluent prompts (AutoDAN/PAIR); wrongly hits high-PPL normal input |
| **Input/output classifier (Llama Guard)** | Various known patterns | Inference overhead; misses novel/semantic-level attacks |
| **Instruction Hierarchy** | Injection/role-play | Reduces probability, not eliminates; boundary judgment can still be fooled |
| **Representation-level (Circuit Breakers)** | Attacks that bypass refusal | Over-refusal/capability loss; depends on accurate representation localization |
| **Adversarial training (fine-tune on attack samples)** | Seen attack distribution | Poor generalization to unseen attacks; may teach hiding (see Sleeper Agents) |

In practice: **defense in depth combined** rather than a single point—classifier as backstop + Instruction Hierarchy against injection + representation level against bypass + continuous red-team updates, with a HarmBench × StrongREJECT closed loop for evaluation.

> **Follow-up:** Why can't "adversarial training" basically solve the problem the way it does in the image domain?
> The text attack space is discrete, vast, and semantically infinitely rewritable; adversarial training only covers seen attack distributions and generalizes poorly to new wrappings (new languages/new encodings/new scenarios); combined with shallow alignment and fine-tunability, this makes "train once for permanent robustness" unrealistic on LLMs.

</details>

---

<details>
<summary>Q28: How do Deliberative Alignment and Constitutional Classifiers complement each other for defense in depth? What are their respective blind spots and applicable settings?</summary>

**A:** The two represent different paradigms of safety defense, complementary rather than substitutable:

| Dimension | Deliberative Alignment | Constitutional Classifiers |
|-----------|------------------------|----------------------------|
| **Location** | **Inside** the model (weight-level) | **Outside** the model (input/output guardrails) |
| **Mechanism** | CoT explicitly cites safety specs during inference → slow thinking | Lightweight classifier makes fast harmful/not-harmful judgments → fast judgment |
| **Suitable for** | Gray areas requiring the model's intrinsic judgment (e.g., "does this count as hate speech?") | Clear-cut categories needing fast, auditable interception (e.g., CBRN/CSAM) |
| **Blind spots** | Depends on model reasoning quality; think step can be injected; safety reasoning overhead for all queries | Edge attacks not covered by the constitution; can be targeted and bypassed (Bug Bounty proved this) |
| **Update process** | Retrain/fine-tune (heavy) | Update constitution + retrain classifier on synthetic data (relatively light) |

**Complementary logic**: Deliberative Alignment gives the model "internal safety intuition," while Constitutional Classifiers provide "external safety rules"—the former handles gray zones, the latter backstops clear red lines. The combination raises attack cost and reduces single points of failure; but deployment paths, tool layers, monitoring, and updates still need to be covered—it is not a sufficient guarantee. Anthropic explicitly positions Constitutional Classifiers as **one layer of defense in depth** (analogous to a Swiss cheese model), not a standalone solution; OpenAI in the o1 System Card similarly designed its safety architecture as a multi-layer combination.

> **Follow-up:** Why not use just one of them?
> Deliberative Alignment alone: reasoning can be wrong, the think step can be injected, and clear red lines like CBRN lack auditable hard interception. Constitutional Classifiers alone: gray zones (e.g., "subtle political bias") are hard to exhaustively enumerate in a constitution, the classifier tends toward over- or under-blocking, and the Bug Bounty proved classifiers can be bypassed. The combination covers both the "intuition + rules" dimensions, requiring attackers to bypass both for effective harm.

</details>

---

## Appendix: Glossary

| Term | Brief definition |
|------|------------------|
| HHH | Helpful / Harmless / Honest — the common alignment-objective triad |
| Red-Teaming | Proactively construct adversarial inputs to elicit and patch harmful behavior |
| Jailbreak | Bypass safety alignment to make the model produce content it should refuse |
| Refusal | The model's refusal behavior on (suspected) harmful requests |
| Over-refusal | Wrongly refusing safe-but-scary harmless requests, hurting usability |
| ASR | Attack Success Rate — the core metric for jailbreak evaluation |
| GCG | Transferable adversarial-suffix attack via greedy coordinate gradient optimization |
| PAIR | Attacker-LLM + judge-LLM closed-loop iterative jailbreak (~20 queries) |
| AutoDAN | Genetic algorithm evolves fluent jailbreak prompts that resist perplexity filtering |
| Many-shot Jailbreak | Prepend many harmful examples in a long context to override refusal |
| Safe-RLHF | Reward+cost dual models + Lagrangian constrained optimization |
| Cost Model | A model scoring how harmful a response is (dual to the reward) |
| RBR | Human-written rules scored by an LLM judge build the safety reward directly; interpretable |
| Llama Guard | An LLM-based input/output safety classifier (runtime guardrail) |
| Circuit Breakers | Representation-level defense: short-circuit harmful generation trajectories |
| Instruction Hierarchy | A system > user > tool privilege order to defend against injection |
| Constitutional AI | Principle-driven self-critique + AI preference (RLAIF) for alignment |
| HarmBench | Standardized red-team benchmark of 18 attacks × 33 models |
| StrongREJECT | A jailbreak judge that excludes empty jailbreaks |
| XSTest | 250 safe-but-scary prompts that test over-refusal |
| Min-K% / Membership inference | See [data-pipeline](cheatsheet-data-pipeline-en.html) §5.3 |
| Shallow Alignment | Safety only changed the distribution of the first few tokens |
| Refusal Direction | A single linear direction in the residual stream mediating refusal |
| Sleeper Agents | Conditionally triggered backdoors that survive safety training |
| Best-of-N (BoN) Jailbreak | Random perturbations + repeated sampling to find a jailbreak; ASR follows power-law scaling; an attack method |
| Crescendo | Multi-turn gradual-escalation jailbreak exploiting the foot-in-the-door tactic |
| Deliberative Alignment | Let the model explicitly reason about safety specs in its CoT before answering; the core safety method behind o1 |
| Constitutional Classifiers | Input/output guardrail classifiers trained on constitution-based synthetic data |

---

*For study reference only. Paper conclusions and figures follow the original papers; benchmark scores are illustrative and not head-to-head comparisons. Safety-related content is solely for defensive research and alignment education; all code examples are placeholder skeletons and contain no actionable harmful content.*

## §A Key Papers Timeline

- **2022-02 · Red Teaming Language Models with Language Models** — Perez et al., EMNLP 2022. [arXiv:2202.03286](https://arxiv.org/abs/2202.03286) — Uses a red-team LM to auto-generate test cases that elicit harmful outputs from the target LM, turning red-teaming from labor-intensive into scalable and reproducible — the origin of automated safety testing.

- **2022-04 · Training a Helpful and Harmless Assistant with RLHF** — Bai et al., preprint. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862) — Anthropic uses RLHF to learn separate helpfulness and harmlessness reward models, systematically characterizing their tension and establishing the "helpful–harmless tension" as a core proposition of safety alignment.

- **2022-08 · Red Teaming LMs to Reduce Harms** — Ganguli et al., preprint. [arXiv:2209.07858](https://arxiv.org/abs/2209.07858) — Large-scale manual red-teaming that records how attack success rate varies with scale/alignment method and categorizes harms; counterintuitively notes that scaling up alone is not necessarily safer — the alignment method matters more.

- **2022-12 · Constitutional AI: Harmlessness from AI Feedback** — Bai et al., preprint. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) — Uses a set of "constitutional principles" to drive model self-critique + revision for SFT, then RLHF with AI preference labels (RLAIF), making harmlessness almost independent of human annotation (see also data-pipeline §4).

- **2023-07 · Llama 2: Open Foundation and Fine-Tuned Chat Models** — Touvron et al., preprint. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — Provides an industrial-grade safety post-training recipe: a dedicated safety reward model, safety-focused RLHF, and safety context distillation — an important reference for open-model safety alignment.

- **2023-07 · Jailbroken: How Does LLM Safety Training Fail?** — Wei et al., NeurIPS 2023 Oral. [arXiv:2307.02483](https://arxiv.org/abs/2307.02483) — Proposes two mechanistic root causes of jailbreaks — competing objectives and mismatched generalization — which became the theoretical template for nearly all subsequent jailbreak methods.

- **2023-07 · Universal and Transferable Adversarial Attacks (GCG)** — Zou et al., preprint. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) — Optimizes adversarial suffixes via greedy coordinate gradient so the model opens in an affirmative tone; the suffixes are universal and transfer to closed-source models, exposing the adversarial fragility of alignment.

- **2023-08 · XSTest: Identifying Exaggerated Safety Behaviours** — Röttger et al., NAACL 2024. [arXiv:2308.01263](https://arxiv.org/abs/2308.01263) — Uses 250 safe-but-scary prompts to specifically test over-refusal, reminding us that safety evaluation must check both ends to avoid rewarding the degenerate "refuse everything" strategy.

- **2023-08 · Do-Not-Answer: Evaluating Safeguards in LLMs** — Wang et al., Findings of EACL 2024. [arXiv:2308.13387](https://arxiv.org/abs/2308.13387) — Curates a set of instructions an LLM should refuse and shows a lightweight classifier suffices to score refusal quality, making safety evaluation low-cost and reproducible.

- **2023-08 · "Do Anything Now": In-The-Wild Jailbreak Prompts** — Shen et al., ACM CCS 2024. [arXiv:2308.03825](https://arxiv.org/abs/2308.03825) — Collects and categorizes 1,405 in-the-wild jailbreak prompts, summarizing social-engineering strategies such as role-play / privilege escalation / injection, characterizing the real attack surface beyond algorithms.

- **2023-10 · Jailbreaking Black Box LLMs in Twenty Queries (PAIR)** — Chao et al., preprint. [arXiv:2310.08419](https://arxiv.org/abs/2310.08419) — Uses an attacker LLM + judge LLM in a closed loop to auto-generate jailbreak prompts within about 20 black-box queries, gradient-free, demonstrating the efficiency of black-box automated attacks.

- **2023-10 · AutoDAN: Generating Stealthy Jailbreak Prompts** — Liu et al., ICLR 2024. [arXiv:2310.04451](https://arxiv.org/abs/2310.04451) — Uses a hierarchical genetic algorithm to evolve semantically fluent jailbreak prompts that bypass perplexity filtering, fixing the weakness of GCG's easily-detected gibberish suffixes.

- **2023-10 · Fine-tuning Aligned LMs Compromises Safety** — Qi et al., ICLR 2024. [arXiv:2310.03693](https://arxiv.org/abs/2310.03693) — Fine-tuning on only ~10 adversarial samples (or even benign data) bypasses safety alignment, revealing that alignment guards the inference distribution but not the weight-update path.

- **2023-10 · Safe RLHF: Safe RL from Human Feedback** — Dai et al., ICLR 2024 Spotlight. [arXiv:2310.12773](https://arxiv.org/abs/2310.12773) — Uses reward+cost dual models to set "safety" as a hard constraint, maximizing reward under "cost ≤ threshold" via Lagrangian duality, giving an adaptive safety–helpfulness trade-off framework.

- **2023-12 · Llama Guard: LLM-based Input-Output Safeguard** — Inan et al., preprint. [arXiv:2312.06674](https://arxiv.org/abs/2312.06674) — Casts content moderation as an LLM classification task, using a single model to intercept at the input/output side per a harm taxonomy, providing an independently updatable runtime guardrail.

- **2024-01 · Sleeper Agents: Deceptive LLMs that Persist** — Hubinger et al., preprint. [arXiv:2401.05566](https://arxiv.org/abs/2401.05566) — Conditionally triggered backdoors survive RLHF/SFT/adversarial training, and adversarial training may teach the model to hide backdoors better rather than remove them, questioning the cleansing ability of standard safety training.

- **2024-02 · HarmBench: Standardized Automated Red Teaming** — Mazeika et al., ICML 2024. [arXiv:2402.04249](https://arxiv.org/abs/2402.04249) — Uses a unified framework of 18 attacks × 33 models to make jailbreak attack/defense reproducible and comparable, systematically evaluating defense robustness on that basis.

- **2024-02 · A StrongREJECT for Empty Jailbreaks** — Souly et al., NeurIPS 2024 D&B. [arXiv:2402.10260](https://arxiv.org/abs/2402.10260) — Points out that "empty jailbreaks" make old metrics overestimate attacks; switches to a rubric scoring refusal + specificity + convincingness, counting only genuinely useful harmful output toward ASR.

- **2024-04 · The Instruction Hierarchy** — Wallace et al., preprint. [arXiv:2404.13208](https://arxiv.org/abs/2404.13208) — Trains the model to obey a system > user > tool privilege order so low-privilege content cannot override high-privilege safety instructions, defending against (indirect) prompt injection from the training side.

- **2024-04 · Many-shot Jailbreaking** — Anil et al., NeurIPS 2024 (Anthropic). [tech report](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf) — Prepends many harmful examples in a long context; as the number of shots grows it predictably overrides refusal, revealing the capability–safety conflict that "extending context = amplifying the attack surface".

- **2024-04 · Crescendo: Multi-Turn LLM Jailbreak Attack** — Russinovich et al., Microsoft. [arXiv:2404.01833](https://arxiv.org/abs/2404.01833) — Uses a foot-in-the-door strategy with multiple rounds of seemingly harmless, progressively escalating questions to gradually guide the model to output dangerous information; fully black-box, prompts appear benign, input filtering is very difficult to trigger; Crescendomation automated tool outperforms prior methods by 29–71% on AdvBench; especially dangerous in multi-turn agent settings.

- **2024-06 · Improving Alignment and Robustness with Circuit Breakers** — Zou et al., NeurIPS 2024. [arXiv:2406.04313](https://arxiv.org/abs/2406.04313) — Uses representation rerouting to interrupt harmful generation trajectories at the representation layer, making it harder to produce harmful content even when refusal is bypassed, empirically more robust to many unseen attacks.

- **2024-06 · Safety Alignment Should Be Made More Than Just a Few Tokens Deep** — Qi et al., ICLR 2025 Oral/Outstanding. [arXiv:2406.05946](https://arxiv.org/abs/2406.05946) — Shows current alignment only changes the "shallow" behavior of the first few output tokens, unifying the explanation of prefix-injection / adversarial-suffix / fine-tuning attacks, and proposes directions to deepen alignment.

- **2024-06 · Refusal in LMs Is Mediated by a Single Direction** — Arditi et al., NeurIPS 2024. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) — Finds across 13 open-source models that refusal is mediated by a single linear direction in the residual stream: ablate it to turn refusal off, inject it to trigger refusal — exposing the representation-level fragility of open-weight model alignment.

- **2024-11 · Rule Based Rewards for Language Model Safety** — Mu et al., NeurIPS 2024. [arXiv:2411.01111](https://arxiv.org/abs/2411.01111) — Uses human-written rules scored by an LLM judge to build the safety reward directly in RL, replacing the opaque safety RM, making the safety signal interpretable, auditable, and updatable with the policy; rule loopholes are easier to find and patch than in a black-box RM, but the rules themselves can still be gamed and require continuous auditing.

- **2024-12 · Best-of-N Jailbreaking** — Hughes et al., preprint. [arXiv:2412.03556](https://arxiv.org/abs/2412.03556) — Simple black-box jailbreak via random perturbations + repeated sampling; ASR follows power-law scaling (GPT-4o ~89%), cross-modal, defeats Circuit Breakers and reasoning models; reveals a deep fragility in alignment to innocuous perturbations (this is an attack method, not a defense).

- **2024-12 · Deliberative Alignment: Reasoning Enables Safer Language Models** — Guan et al., OpenAI. [arXiv:2412.16339](https://arxiv.org/abs/2412.16339) — Teaches the model safety specifications directly to reason about in its CoT via two-stage training (SFT + RL, RM judges answer only not CoT); o1 StrongREJECT goodness@0.1 reaches 0.88 (vs GPT-4o 0.37), XSTest benign accuracy 93%; improves robustness while reducing over-refusal.

- **2025-01 · Constitutional Classifiers: Defending against Universal Jailbreaks** — Sharma et al., Anthropic. [arXiv:2501.18837](https://arxiv.org/abs/2501.18837) — Constitution-based auto-generated synthetic training data for input/output guardrail classifiers; automated jailbreak rate 86%→4.4%, over-refusal +0.38%; 183 participants >3,000h red-teaming found no universal jailbreak on prototype; Bug Bounty proved bypasses are possible; positioned as one layer of defense in depth.


