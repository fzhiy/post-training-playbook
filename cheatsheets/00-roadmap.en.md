# 📍 Learning Path / Roadmap

> Recommended order for working through the playbook. For each topic: read the **cheatsheet** to build understanding → work through the corresponding **drill** → self-test with the in-page L1/L2/L3 questions. Links go directly to each page.

## 1 · Foundations
- [ml-dl-fundamentals](cheatsheet-ml-dl-fundamentals-en.html) — optimizers / regularization / normalization / backpropagation
- [math-and-stats](cheatsheet-math-and-stats-en.html) — probability / linear algebra (SVD → low-rank) / KL / information theory
- Drills: [cross-entropy](drill-cross-entropy.html) · [adamw](drill-adamw.html)

## 2 · LLM Architecture
- [llm-architecture](cheatsheet-llm-architecture-en.html) — Transformer / attention variants / positional encoding / KV cache / MoE
- Drills: [attention](drill-attention.html) · [rope](drill-rope.html) · [kv-cache](drill-kv-cache.html) · [gqa-mqa](drill-gqa-mqa.html) · [swiglu-ffn](drill-swiglu-ffn.html) · [rmsnorm](drill-rmsnorm.html) · [sampling](drill-sampling.html)

## 3 · PEFT
- [peft-methods](cheatsheet-peft-methods-en.html) — LoRA / DoRA / rank-spectrum budgeting / merge inference
- Drills: [lora-forward](drill-lora-forward.html) · [dora-forward](drill-dora-forward.html)

## 4 · Post-training Core
- [llm-post-training](cheatsheet-llm-post-training-en.html) — SFT / RM / RLHF / PPO / DPO full pipeline
- [reward-modeling-eval](cheatsheet-reward-modeling-eval-en.html) — RM training / PRM-ORM / reward hacking / evaluation
- [eval-and-judges](cheatsheet-eval-and-judges-en.html) — three evaluation categories / LLM-as-judge / evaluation bias / benchmarks and data contamination
- Drills: [sft-loss-mask](drill-sft-loss-mask.html) · [sequence-packing](drill-sequence-packing.html) · [dpo-loss](drill-dpo-loss.html) · [simpo-loss](drill-simpo-loss.html) · [ppo-clip](drill-ppo-clip.html) · [gae](drill-gae.html) · [reward-margin](drill-reward-margin.html)

## 5 · Reasoning-RL Frontier
- [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier-en.html) — PPO→GRPO→DAPO·Dr.GRPO / RLVR / long-CoT
- Drills: [grpo](drill-grpo.html) · [rloo](drill-rloo.html)

## 6 · Systems & Wrap-up
- [ml-system-design](cheatsheet-ml-system-design-en.html) — fine-tuning pipelines / distributed RLHF / evaluation infrastructure
- [coding-and-algorithms](cheatsheet-coding-and-algorithms-en.html) — algorithm problem types + ML implementation problems

## 7 · Continual / Lifelong
- [continual-post-training](cheatsheet-continual-post-training-en.html) — catastrophic forgetting / replay / model merging / KL · production-validated methods only

## 8 · Long-horizon / Agentic
- [long-horizon-agents](cheatsheet-long-horizon-agents-en.html) — computer use / agent engineering / difficulty-graded rewards / self-evolution (production vs. frontier)

---

**Review method**: after reviewing each item, mark it ✅ solid / ⚠️ fuzzy / ❌ unknown; then only re-drill ⚠️/❌. Before interviews, use the homepage search bar to jump directly to weak topics.
