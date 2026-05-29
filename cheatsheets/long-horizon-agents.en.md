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

### 3.2 Three Core Challenges of Web / Agent RL (WebRL Framework)
Standard structure for answering "why is web/long-horizon agent training hard": ① **scarcity of training tasks**; ② **sparse feedback signals**; ③ **policy distribution shift**.<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">Three core challenges of web agent training: task scarcity / sparse feedback / policy shift (this page only uses this framework).<a href="https://arxiv.org/abs/2411.02337">Qi 2024 ↗</a></span></span> *(WebRL's core is "failed trajectories → self-evolving curriculum"; this page only uses its three-challenge framework and does not expand on that mechanism.)*

### 3.3 Connections to Other Pages
GRPO improvements (Clip-Higher, remove KL loss) originate from ByteDance **DAPO**<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">ByteDance's GRPO improvements: Clip-Higher, remove KL loss.<a href="https://arxiv.org/abs/2503.14476">ByteDance 2025 ↗</a></span></span>, adopted by MiMo — see [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html); not repeated here.

## 4. 【Frontier · Least Mature】Self-Evolving / Self-Evolving Agents (Requires Most Caution)

> ⚠️ The vast majority of this area is **research**; production evidence is weak. Do not claim this as an industry standard in interviews, and do not cite unverified numbers.

- Core idea: let the agent continuously improve via **self-generated data / self-play / generating new tasks from failures / reflection-self-correction**, bypassing human annotation.
- Current state of the field: papers exist exploring this direction (automated curriculum, self-play search, etc.), but strong claims of "unsupervised fully automatic scaling" **often do not hold up to verification**; treat as an **open research question**, not a mature solution. Deep treatment is left to the independent [agent-post-training-playbook](https://github.com/fzhiy/agent-post-training-playbook).
- **Honest connection (your research)**: your **Continual Agent (in progress) + Fed-TaLoRA anti-forgetting** perspective → you can say "I focus on continual learning / catastrophic forgetting in agents, which gives me an understanding of why self-evolution is still immature in production and where the boundaries are"; do **not** say "I have built production self-evolving agent systems." See [continual-post-training](cheatsheet-continual-post-training-en.html).

## 5. Interview Question Clusters / Stratified Follow-ups

> Inferred from public papers / JDs as high-frequency clusters; **not real exam questions**.

### L1 Fundamentals
- What is the difference between a workflow and an agent? When should you **not** use an agent (when a workflow / single call is sufficient)?
- How does computer use / Operator work (the screenshot → reasoning → action loop)?

### L2 Intermediate
- How do you prevent context overflow in long-horizon agents (compaction / files as memory / subagents)?
- How do you handle sparse rewards in long-horizon tasks? Why use a difficulty band / dynamic sampling (filter pass-rate $0/1$)?
- How does an agent self-verify (rules / visual / LLM-as-judge), and what are the costs of each?

### L3 Deep Dive
- Design the reward for a long-horizon coding agent: how do you prevent reward hacking and degeneration (no signal when too hard / too easy)?
- What are the three core challenges of web / long-horizon agent training (task scarcity / sparse feedback / policy shift) and how is each mitigated?
- What are the failure modes of self-evolving / self-play training? Why is full automation still not trusted in production?

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
