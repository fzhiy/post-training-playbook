# Long-Horizon / Self-Evolving Agents: Production State vs. Frontier (Interview-Oriented)

> **long-horizon** = multi-step, long-running tasks requiring sustained autonomous execution; **self-evolving** = enabling an agent to continuously improve via self-generated data / self-feedback.
> ⚠️ This page uses **strict column separation**: 【Production】= shipped products / official engineering guides; 【Frontier】= papers / technical reports, **not yet industry standards**. In interviews, do not treat frontier findings as production-standard answers.
> Integrity notice: the "interview question clusters" on this page are **high-frequency question clusters inferred from public papers / JDs**, **not verified real exam questions**; no unverified benchmark numbers are cited. Deep frontier topics (fully automated self-evolution, etc.) are out of scope for this playbook — only signals are given.

## 1. 【Production】What Long-Horizon Agentic Systems Look Like Today (Shipped Products)

Two organizations have shipped "long-horizon agentic" as a product — hard currency when discussing "agent deployment":

- **Anthropic computer use** (2024-10-22 public beta; Anthropic API / Amazon Bedrock / Google Vertex AI): Claude views screenshots → moves cursor / clicks / types, translating instructions into a sequence of computer operations; tasks "require tens, sometimes hundreds of steps." Officially described as **experimental and error-prone**; recommended to start with low-risk tasks.<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">Claude connected to the screen: screenshot → cursor/click/type, tens to hundreds of steps of computer operation (officially experimental).<a href="https://www.anthropic.com/news/3-5-models-and-computer-use">Anthropic 2024 ↗</a></span></span>
- **OpenAI Operator** (2025-01 research preview, ChatGPT Pro) → **ChatGPT agent** (2025-07-17, merging Operator + deep research): the underlying **CUA** (Computer-Using Agent) = GPT-4o vision + RL reasoning; loop = view screenshot → CoT reasoning for next step → click/scroll/type, until completion or human handoff required. Safety: password entry requires user takeover; high-risk tasks (e.g., bank transfers) are declined.<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">OpenAI's computer-use agent: GPT-4o vision + RL, screenshot → reasoning → action loop.<a href="https://openai.com/index/introducing-operator/">OpenAI 2025 ↗</a></span></span> ChatGPT agent operates on a **virtual computer** with visual browser + text browser + terminal + API.<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">Combines Operator (action) and deep research (synthesis) on a single virtual computer.<a href="https://openai.com/index/introducing-chatgpt-agent/">OpenAI 2025 ↗</a></span></span>

## 2. 【Production】Engineering Pillars of Long-Horizon Agents (Official Guides, High Interview Frequency)

**Anthropic《Building Effective Agents》(2024-12-19)**<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Anthropic's agent engineering classic: workflow vs agent, ACI, stopping conditions, per-step environment ground truth.<a href="https://www.anthropic.com/research/building-effective-agents">Anthropic 2024 ↗</a></span></span> ≈ the engineering "bible" of this field:

- **Workflow vs Agent** (must-know): workflow = LLM / tools follow **predefined code paths**; agent = LLM **dynamically decides** the process and tool usage itself (who owns the control flow is the key distinction).
- Common patterns: prompt chaining, routing, parallelization (sectioning / voting), orchestrator-workers, evaluator-optimizer; and **autonomous agents** (loop driven by environment feedback, planning autonomously to completion or until a stopping condition is triggered).
- **When to use an agent**: task is open-ended, number of steps is unpredictable, path cannot be hardcoded, and the higher latency/cost tradeoff for better performance is acceptable — **otherwise try a simpler solution first** (single call + retrieval + few-shot).
- Engineering essentials: craft the **ACI (agent-computer interface)** as carefully as HCI; set **stopping conditions** (e.g., max iterations); evaluate progress at every step with **environment ground truth** (tool results / code execution); sandbox + guardrails to prevent error accumulation.

**Claude Agent SDK**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">Long-horizon agent loop gather→act→verify→repeat + context management (compaction / files as memory / subagents).<a href="https://claude.com/blog/building-agents-with-the-claude-agent-sdk">Anthropic ↗</a></span></span> core loop (worth memorizing):

> **gather context → take action → verify work → repeat**

- **Context management** (avoid blowing context in long runs): compaction (auto-summarize old messages), **file system as memory** (grep / tail on demand), subagents (isolated context, parallel execution, return only summaries).
- **Self-verification**: rules-based (linter, precise), visual (screenshot, verifies UI), LLM-as-judge (fuzzy criteria, costs latency / stability).
- Reliability = appropriate tools + clear feedback + representative scenario testing + iterating on failure modes.

**Agent loop skeleton** (simplified, showing control flow; real implementations need error handling / compaction / sandbox):

```python
def agent_loop(task, tools, max_steps=50, token_limit=128_000):
    """Long-horizon agent core loop: gather context → take action → verify → repeat."""
    context = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    context.append({"role": "user", "content": task})

    for step in range(max_steps):
        # 1. gather: LLM reasons about the next action
        response = llm.chat(context, tools=tools)

        if response.has_tool_calls():
            for tc in response.tool_calls:
                result = execute_tool(tc)                # real tool execution (in sandbox)
                context.append(tool_result_msg(tc, result))
        else:
            break                                        # agent decides the task is done

        # 2. verify: check the current step with environment ground truth
        if not verify_step(response, tools):             # rules/visual/LLM-as-judge
            context.append({"role": "user",
                           "content": "Step seems wrong; review the result and retry."})

        # 3. context management: prevent token overflow
        if estimate_tokens(context) > token_limit:
            context = compact_context(context)           # auto-summarize old messages

    return response.content
# Key design points:
# - execute_tool must be sandboxed (code in isolated env, browser in virtual browser)
# - verify_step tiered from cheap to expensive: rules → visual → LLM-as-judge
# - compact_context keeps key decision-point markers, only compresses intermediate chat
```

## 3. 【Frontier】Training Paradigms for Long-Horizon / Agentic Systems (Research Context, Not Yet Industry Standards)

> ⚠️ The following is **research** context, not a production deployment claim for any specific product. In interviews you may say "I've been tracking direction X" — do not say "this is the industry standard."

### 3.1 Sparse / Long-Horizon Rewards + Difficulty-Band Design (High-Frequency System Design Question)
Long-horizon task rewards are sparse (often only a final pass/fail signal). A recurring engineering principle: **effective RL signal only exists in the intermediate difficulty band** — explicitly preventing training data from collapsing to either extreme is required.

**Self-Play SWE-RL**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">Self-play RL where the same LLM both injects and fixes bugs; segmented reward for difficulty band (this page only uses the reward design).<a href="https://arxiv.org/abs/2512.18552">Wei 2025 ↗</a></span></span>: the same LLM both injects bugs and fixes them, using a test suite as reward. The bug injection reward is a piecewise function ($s$ = fraction of fixers who solved the bug, i.e., solve rate):

$$
r_{\text{inject}} = \begin{cases} -\alpha, & s \in \{0, 1\} \\ 1-(1+\alpha)\,s, & 0 < s < 1 \end{cases}, \quad \alpha = 0.8
$$

Negative reward is given for both "too hard (no one solves it, $s{=}0$)" and "too easy (everyone solves it, $s{=}1$)"; only intermediate difficulty is rewarded. *(This page only uses the reward design; performance numbers are unverified and not cited.)*

**MiMo**<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">Xiaomi's released-model RL recipe: remove KL loss, Clip-Higher, dynamic sampling to filter pass-rate 0/1.<a href="https://arxiv.org/abs/2505.07608">Xiaomi 2025 ↗</a></span></span> (RL recipe for Xiaomi's released model): dynamic sampling **filters prompts with pass-rate$=0/1$**, and maintains a 10% easy-question pool to prevent instability in late-stage policy updates.

**The motivation of both is the same** = difficulty-adaptive curriculum: concentrate signal on problems the model "can almost solve but hasn't yet mastered."

```python
def difficulty_band_reward(solve_rate, alpha=0.8):
    """Self-Play SWE-RL piecewise reward: penalize too-hard/too-easy, reward intermediate.
    solve_rate: s ∈ [0,1] = fraction of fixers who solved the bug."""
    if solve_rate == 0.0 or solve_rate == 1.0:
        return -alpha                       # too hard (no one solves) or too easy (everyone solves) → negative
    return 1.0 - (1.0 + alpha) * solve_rate # intermediate s ∈ (0,1): lower s (harder) → higher reward
# Intuition: s=0→-0.8; s≈0.1 (extremely hard)≈0.82 max; s=0.5→0.1; s=1→-0.8.
# Encourages generating "extremely hard but not impossible" bugs — real hard bugs that most fixers can't solve.
# Note: the function peaks as s→0+; the paper also applies a consistency-validation penalty.
```

### 3.2 Three Core Challenges of Web / Agent RL (WebRL Framework)
Standard structure for answering "why is web/long-horizon agent training hard": ① **scarcity of training tasks**; ② **sparse feedback signals**; ③ **policy distribution shift**.<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">Three core challenges of web agent training: task scarcity / sparse feedback / policy shift (this page only uses this framework).<a href="https://arxiv.org/abs/2411.02337">Qi 2024 ↗</a></span></span> *(WebRL's core is "failed trajectories → self-evolving curriculum"; this page only uses its three-challenge framework and does not expand on that mechanism.)*

### 3.3 Connections to Other Pages
GRPO improvements (Clip-Higher, remove KL loss) originate from ByteDance **DAPO**<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">ByteDance's GRPO improvements: Clip-Higher, remove KL loss.<a href="https://arxiv.org/abs/2503.14476">ByteDance 2025 ↗</a></span></span>, adopted by MiMo — see [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html); not repeated here.

## 4. 【Frontier · Least Mature】Self-Evolving / Self-Evolving Agents (Requires Most Caution)

> ⚠️ The vast majority of this area is **research**; production evidence is weak. Do not claim this as an industry standard in interviews, and do not cite unverified numbers.

- Core idea: let the agent continuously improve via **self-generated data / self-play / generating new tasks from failures / reflection-self-correction**, bypassing human annotation.
- Current state of the field: papers exist exploring this direction (automated curriculum, self-play search, etc.), but strong claims of "unsupervised fully automatic scaling" **often do not hold up to verification**; treat as an **open research question**, not a mature solution. Deep treatment is left to the independent [agent-post-training-playbook](https://github.com/fzhiy/agent-post-training-playbook).

**Existing research attempts (framework-level, not performance claims)**:

| Direction | Representative work | Core mechanism | Known fragility |
|------|------|------|------|
| Failure → new tasks | **WebRL** ([arXiv:2411.02337](https://arxiv.org/abs/2411.02337)) | Reverse-construct new training tasks from failure trajectories ("the agent went wrong here → create a targeted drill") | Auto-generated tasks may be "noise" rather than "effective difficulty" |
| Self-play RL | **Self-Play SWE-RL** ([arXiv:2512.18552](https://arxiv.org/abs/2512.18552)) | Same LLM injects bugs → fixes bugs, using test suites as ground-truth reward | Bug injection may degenerate to "injecting too-trivial/too-absurd bugs" |
| Reflection self-correction | **Self-Refine** ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)) | Model generates → self-critiques → revises per feedback | Purely intrinsic self-correction often changes right to wrong (see [test-time-scaling §2.3](cheatsheet-test-time-scaling-en.html)) |
| Multi-agent mutual critique | **MULTI-AGENT Debate** line | Agent A generates, Agent B critiques, iteratively improve | Multiple agents share the same base-model biases → risk of groupthink |

**Four core failure modes** (standard answer framework for L3 interviews): ① **Mode collapse** — model prefers its own style → positive feedback → loss of diversity; ② **Reward collapse** — self-judgment quality degrades (increasingly "lenient" toward its own outputs); ③ **Difficulty collapse** — self-generated tasks trend simple (the model invents problems it already knows how to do); ④ **Distributional drift** — self-generated data diverges from the real task distribution. Root cause: **lack of an external ground-truth anchor** — no objective, model-independent verification signal to calibrate the direction. Verifiable domains (code/math) can provide anchors; open-ended domains have almost none. For interviews: a candid analysis of the boundaries is more persuasive than claiming "it can be done."

- **Honest connection (your research)**: your **Continual Agent (in progress) + Fed-TaLoRA anti-forgetting** perspective → you can say "I focus on continual learning / catastrophic forgetting in agents, which gives me an understanding of why self-evolution is still immature in production and where the boundaries are"; do **not** say "I have built production self-evolving agent systems." See [continual-post-training](cheatsheet-continual-post-training-en.html).

## §A Key Papers & Products Timeline

- **2023-03 · Self-Refine** — Madaan et al., NeurIPS 2023. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) — "Generate→self-critique→revise" loop with the same LLM; the seed of self-evolution ideas; follow-up work notes that purely intrinsic self-correction yields limited gains.

- **2023-07 · WebArena** — Zhou et al., ICLR 2024. [arXiv:2307.13854](https://arxiv.org/abs/2307.13854) — One of the earliest systematic web agent benchmarks; self-built websites (e-commerce/forum/CMS) with programmatic verification.

- **2024-05 · SWE-agent** — Yang et al., NeurIPS 2024. [arXiv:2405.15793](https://arxiv.org/abs/2405.15793) — Proposed ACI (Agent-Computer Interface) design principles; "well-designed CLI tools" over raw shell improve agent efficiency.

- **2024-10 · Anthropic Computer Use** — Anthropic, product release (public beta). [blog](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude views screenshots → operates a computer; tasks "require dozens, and sometimes even hundreds of steps"; the first frontier AI model to offer computer use in public beta.

- **2024-11 · WebRL** — Qi et al., ICLR 2025. [arXiv:2411.02337](https://arxiv.org/abs/2411.02337) — Proposes the three core challenges of web agent training (task scarcity/sparse feedback/policy shift) + a self-evolving curriculum framework; this page uses the framework without expanding the mechanism.

- **2024-12 · Building Effective Agents** — Anthropic, engineering guide. [blog](https://www.anthropic.com/research/building-effective-agents) — workflow vs agent, common patterns, when to use an agent, ACI/sandbox principles; the agent engineering "bible."

- **2025-01 · OpenAI Operator / CUA** — OpenAI, research preview. [blog](https://openai.com/index/introducing-operator/) — GPT-4o vision + RL reasoning computer-use agent, screenshot→CoT→action loop; integrated into ChatGPT agent 2025-07.

- **2025-03 · DAPO** — Yu et al. (ByteDance Seed), preprint. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) — Clip-Higher + remove KL loss; adopted by MiMo and other agent RL recipes (details at [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html)).

- **2025-05 · MiMo** — Xiaomi LLM-Core, tech report. [arXiv:2505.07608](https://arxiv.org/abs/2505.07608) — Complete RL recipe for a released model: remove KL loss + Clip-Higher + dynamic sampling filtering + difficulty-band curriculum.

- **2025-12 · Self-Play SWE-RL** — Wei et al. (Meta/FAIR), accepted to ICML 2026. [arXiv:2512.18552](https://arxiv.org/abs/2512.18552) — Self-play RL where the same LLM both injects and fixes bugs; piecewise difficulty-band reward; this page only uses the reward design, without unverified performance numbers.

## 5. Interview Question Clusters / Stratified Follow-ups

> Inferred from public papers / JDs as high-frequency clusters; **not real exam questions**.

### L1 Fundamentals

<details>
<summary>Q1: What is the difference between a workflow and an agent? When should you **not** use an agent?</summary>

**A:** **Workflow** = LLM/tools follow **predefined code paths** (control flow lives in the code — the developer hardcodes "call A → if X then B → else C" beforehand). **Agent** = LLM **dynamically decides** the process and tool usage itself (control flow lives in the model — it chooses what to do next based on the current state). Anthropic's decision criterion: an agent is appropriate when the **task is open-ended, the number of steps is unpredictable, and the path cannot be hardcoded** — and you're willing to trade higher latency/cost for better performance. When NOT to use an agent: ① the number of steps is predictable (e.g., "check weather → give advice" is two fixed segments) → prompt chaining + routing suffices; ② a single LLM call + retrieval handles it (e.g., FAQ answering); ③ latency/cost is critical and the agent's extra benefit doesn't justify the cost. The golden rule: **"Start with the simplest solution and only escalate to an agent when necessary."**

> **Follow-up:** What's the difference between orchestrator-workers and an autonomous agent?
> orchestrator-workers has an LLM-dynamic "orchestrator" (unlike hardcoded prompt chaining), but the "workers" are fixed sub-LLM calls — the control flow is in the orchestrator's hands, not in the code, so it sits closer to the agent side, as a "contained agent."

</details>

<details>
<summary>Q2: How does computer use / Operator work (the screenshot → reasoning → action loop)?</summary>

**A:** The core loop: **screenshot → visually understand the current screen → CoT reasoning about the next action → execute the action (click/scroll/type/hotkey) → wait for the environment to change → repeat**. Two implementations: ① **Anthropic computer use** (2024-10): Claude views screenshots and outputs structured tool-use instructions (`computer` tool type); tasks "require dozens, and sometimes even hundreds of steps"; officially experimental and error-prone. ② **OpenAI Operator / CUA** (2025-01→07): GPT-4o vision + RL reasoning; the same screenshot→CoT→action loop underneath; ChatGPT agent integrates this into a virtual computer (visual browser + terminal + API). Shared safety design: password/payment entry requires user handoff (not fully automated); high-risk tasks (bank transfers) are declined; sandbox isolation is used. In interviews, stress two points: ① visual understanding is the key bottleneck (screenshot resolution/latency/imprecise element targeting); ② many steps → error accumulation → per-step environment ground truth verification is essential.

> **Follow-up:** What is the key engineering difference between the two?
> Anthropic provides an API tool (developers integrate into their own systems), OpenAI provides a product (directly operates a virtual computer within ChatGPT) — the former is more flexible but requires self-assembly of the environment, the latter is ready out of the box but limited in customization.

</details>

### L2 Intermediate

<details>
<summary>Q3: How do you prevent context overflow in long-horizon agents (compaction / files as memory / subagents)?</summary>

**A:** Claude Agent SDK provides three combined strategies: ① **compaction**: auto-summarize old messages — when the prompt grows too long, compress early conversation into a brief summary to free tokens while keeping key information; ② **file system as memory**: don't cram all context into the prompt — write long outputs/intermediate results/history to the file system → grep/tail on demand when needed — essentially using the disk as "external memory"; ③ **subagents**: delegate sub-tasks to independent agents (isolated context, parallel execution), returning only the summary to the main agent — preventing sub-task details from contaminating the main context. In practice, the three are combined in tiers: compaction for chat history, the file system for long persistent memory, subagents for parallel independent sub-tasks. Additional measures: structured outputs (limit token waste), step counter (forced truncation).

> **Follow-up:** Can compaction lose critical information?
> Yes, there's a risk — a detail compressed away early in the conversation may later become critical. Mitigations: keep pointers to the original messages ("re-check the source if needed"), and set non-compaction markers on key decision points.

</details>

<details>
<summary>Q4: How do you handle sparse rewards in long-horizon tasks? Why use a difficulty band / dynamic sampling?</summary>

**A:** Long-horizon agent tasks typically have only a **final pass/fail** binary signal — with no reward feedback at all for the 100 intermediate steps. The core insight of the difficulty-band design: **effective RL signal only exists in the intermediate difficulty band where "the model can almost do it but isn't certain."** Both extremes yield zero gradient: too hard → always fails → all-wrong group advantage is zero; too easy → always succeeds → all-correct group advantage is zero. Solution: **dynamic sampling filters prompts with pass-rate=0/1** (MiMo's approach: evaluate current pass-rate on each prompt → discard all-correct and all-wrong → keep the "can-just-barely-do" difficulty band + maintain a 10% easy-question pool to prevent late-stage instability). Self-Play SWE-RL goes further: uses a **piecewise reward function** to explicitly penalize both difficulty extremes ($s=0$ too hard no one can solve it, $s=1$ too easy everyone solves it) → only rewards $0<s<1$, the intermediate difficulty band. The motivation of both is the same: **difficulty-adaptive curriculum — concentrate the training signal on problems that the model "can reach but hasn't yet mastered."**

> **Follow-up:** What if difficulty converges late in training (most prompts become either too easy or too hard)?
> Expand the prompt pool and inject new problems; meanwhile keep a tiny fraction of all-correct/all-wrong prompts as "anchors" — don't filter completely clean, preserve a minimal proportion of easy/hard samples to prevent distribution skew.

</details>

<details>
<summary>Q5: How does an agent self-verify (rules / visual / LLM-as-judge), and what are the costs of each?</summary>

**A:** Claude Agent SDK provides three verification methods, used in tier from cheap to expensive: ① **rules-based** (linter/unit tests/regex/state checks): most precise, zero latency, fully automatable — but only works for rule-checkable properties (code correctness, format compliance, page URL correctness); ② **visual** (screenshot verification): suited for UI agents — use a vision model / pixel comparison to judge "has the interface reached the desired state?" — more flexible than rules but requires additional visual understanding capability, and the screenshot/processing overhead is non-trivial; ③ **LLM-as-judge**: most flexible — can evaluate fuzzy criteria ("is the response helpful?", "is the interaction natural?") — but costs extra LLM call latency and expense + judge bias and error (position/verbosity/self-preference → see [eval-and-judges §2](cheatsheet-eval-and-judges-en.html)). In practice, use from cheap to expensive in tier: rules first → visual if insufficient → LLM-as-judge as the last resort.

> **Follow-up:** When must you use LLM-as-judge?
> When the evaluation criterion itself is inherently fuzzy — such as "is the dialogue helpful?" or "is the generated email tonally appropriate?" — these cannot be rule-checked and can only be approximated by human preference or an LLM judge. The cost is high but there is no alternative.

</details>

### L3 Deep Dive

<details>
<summary>Q6: Design the reward for a long-horizon coding agent: how do you prevent reward hacking and degeneration?</summary>

**A:** Core challenges: ① only a final test pass/fail binary signal → zero feedback for the N intermediate steps; ② the agent may reward-hack (loop empty actions to burn steps, hardcode answers to bypass real reasoning); ③ prompts that are too hard or too easy yield zero gradient. Design plan: **Primary reward** = final test pass rate (0/1 binary), plus **step-efficiency penalty** (deduct λ per extra step to prevent infinite looping); **Process reward** = small reward for key intermediate products (e.g., "located the bug's file") to alleviate the zero-signal problem in intermediate steps — but beware the "process reward being hacked" risk (the model learns to do actions that "look like finding a bug" without actually finding it); **Difficulty-band filtering** = discard all-correct/all-wrong prompts + keep a 10% easy-question anchor. The key to preventing reward hacking: coding agents choose **verifiable domains** — code unit tests ≈ ground truth, nearly impossible to fool with a neural RM — which is precisely why the coding-agent RL mainline and GRPO/RLVR take the same path (rule-based verifiers).

> **Follow-up:** Can intermediate rewards be hacked?
> Yes — this is exactly why the R1 mainline chooses outcome reward (only the final answer's correctness) rather than process reward. In coding-agent settings, intermediate verification signals (e.g., running a linter, checking whether a file exists) are more reliable than an AI model's PRM — prefer rule-based intermediate signals.

</details>

<details>
<summary>Q7: What are the three core challenges of web / long-horizon agent training and how is each mitigated?</summary>

**A:** The "three challenges" framework comes from WebRL (Qi et al. 2024): ① **Training task scarcity**: real web tasks are diverse but expensive to annotate. Mitigation: auto-generate task templates (derive new tasks from website structure) + reverse-construct new training tasks from failure trajectories ("the agent went wrong at this step → create a targeted drill for it"); ② **Sparse feedback signals**: web tasks typically have only a binary signal of final goal completion. Mitigation: provide shaping rewards for intermediate state changes (page URL jumps/DOM changes/form-filling progress) + set checkpoint rewards for key milestones; ③ **Policy distribution shift**: the agent's behavior changes the distribution of its subsequent observations — a new policy produces different browsing paths, and "what counts as a good action" for the same prompt differs under different policies. Mitigation: online RL (sample with the freshly updated policy each round, don't reuse old data) + first SFT to consolidate basic browsing behavior, then RL fine-tune. These three challenges are coupled: task scarcity limits training scale → limits the ability to cope with distribution shift → which in turn exacerbates the sparse-feedback difficulty.

> **Follow-up:** Why are web agents so much harder than math agents?
> A math agent's environment is static (a math problem doesn't change because the model tried it multiple times). A web agent's environment is interactive — the agent's actions change the page state, causing distribution shift. Coupled with the web's lack of a clean "correct answer" — success is ambiguous ("booked a flight but the price isn't optimal" — does that count as success?) — rewards are even harder to define.

</details>

<details>
<summary>Q8: What are the failure modes of self-evolving / self-play training? Why is full automation still not trusted in production?</summary>

**A:** Four core failure modes: ① **Mode collapse**: the model prefers its own style → a positive feedback loop → output diversity continuously decreases → degenerates into a narrow "self-preferred" policy; ② **Reward collapse**: self-judgment quality degrades round by round (the model becomes increasingly "lenient" toward its own outputs, assigning higher and higher scores while actual quality stays flat or drops) → training gets worse the more it runs; ③ **Difficulty collapse**: self-generated tasks trend simple — the model tends to invent problems it already knows how to do (because those earn positive reward) → training-set difficulty continuously falls → training becomes meaningless; ④ **Distributional drift**: self-generated data gradually diverges from the real task distribution → performance on real tasks actually drops ("getting better and better in its own world, getting worse and worse in the real world"). The root cause: **lack of an external ground-truth anchor** — no objective, model-independent verification signal to calibrate the self-evolution direction. Verifiable domains (code unit tests / math exact-match) can provide such anchors, but open-ended domains have virtually none. Production doesn't trust full automation because: without an external anchor, a self-evolving system looks good in the short term but inevitably collapses in the long term.

> **Follow-up:** Are there "semi-automated" compromise approaches being explored?
> ① Periodically bring in human evaluation to re-anchor the direction (a small amount of manual labeling re-anchors the self-evolution direction); ② Use verifiable domains as anchor tasks and mix in open-domain training (code/math ground-truth signals indirectly constrain open-domain behavior); ③ Multi-agent mutual critique (one agent generates, another critiques — no reliance on a single self-judgment). All three have been tried but none has a widely accepted mature solution — all remain at the research stage.

</details>

## References

> Click superscript `[N]` to jump here; click `↩` to return to the original text; on wide screens the gist appears as a margin note.

<ol class="refs">
<li id="ref-1">Anthropic — <em>Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku</em>(2024-10-22). <a href="https://www.anthropic.com/news/3-5-models-and-computer-use">anthropic.com</a> <a href="#fnref-1">↩</a></li>
<li id="ref-2">Anthropic — <em>Building Effective Agents</em>(2024-12-19). <a href="https://www.anthropic.com/research/building-effective-agents">anthropic.com</a> <a href="#fnref-2">↩</a></li>
<li id="ref-3">Anthropic — <em>Building agents with the Claude Agent SDK</em>. <a href="https://claude.com/blog/building-agents-with-the-claude-agent-sdk">claude.com</a> <a href="#fnref-3">↩</a></li>
<li id="ref-4">OpenAI — <em>Introducing Operator</em> / <em>Computer-Using Agent (CUA)</em>(2025-01). <a href="https://openai.com/index/introducing-operator/">openai.com</a> <a href="#fnref-4">↩</a></li>
<li id="ref-5">OpenAI — <em>Introducing ChatGPT agent</em>(2025-07-17). <a href="https://openai.com/index/introducing-chatgpt-agent/">openai.com</a> <a href="#fnref-5">↩</a></li>
<li id="ref-6">Wei et al.(Meta / FAIR) — <em>Toward Training Superintelligent Software Agents through Self-Play SWE-RL</em>. <a href="https://arxiv.org/abs/2512.18552">arXiv:2512.18552</a> — this page only uses the reward design; unverified performance numbers are not included. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Xiaomi LLM-Core — <em>MiMo Technical Report</em>. <a href="https://arxiv.org/abs/2505.07608">arXiv:2505.07608</a>. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Qi et al. — <em>WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum RL</em>. <a href="https://arxiv.org/abs/2411.02337">arXiv:2411.02337</a> — this page only uses the three-challenge framework; the self-evolving curriculum mechanism is not expanded. <a href="#fnref-8">↩</a></li>
<li id="ref-9">ByteDance Seed — <em>DAPO</em>. <a href="https://arxiv.org/abs/2503.14476">arXiv:2503.14476</a>. <a href="#fnref-9">↩</a></li>
</ol>
