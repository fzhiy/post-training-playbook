# 📍 学习路径 / Roadmap

> 建议按此顺序过 playbook。每个主题:先读 **cheatsheet** 理解 → 做对应 **drill** 手撕 → 用页内 L1/L2/L3 自测。链接直达各页。

## 1 · 打基础 / Foundations
- [ml-dl-fundamentals](cheatsheet-ml-dl-fundamentals.html) — 优化器 / 正则 / 归一化 / 反向传播
- [math-and-stats](cheatsheet-math-and-stats.html) — 概率 / 线代(SVD→低秩)/ KL / 信息论
- 手撕:[cross-entropy](drill-cross-entropy.html) · [adamw](drill-adamw.html)

## 2 · LLM 架构 / Architecture
- [llm-architecture](cheatsheet-llm-architecture.html) — Transformer / attention 变体 / 位置编码 / KV cache / MoE
- 手撕:[attention](drill-attention.html) · [rope](drill-rope.html) · [kv-cache](drill-kv-cache.html) · [gqa-mqa](drill-gqa-mqa.html) · [swiglu-ffn](drill-swiglu-ffn.html) · [rmsnorm](drill-rmsnorm.html) · [sampling](drill-sampling.html)

## 3 · PEFT
- [peft-methods](cheatsheet-peft-methods.html) — LoRA / DoRA / 秩谱预算 / 合并推理
- 手撕:[lora-forward](drill-lora-forward.html) · [dora-forward](drill-dora-forward.html)

## 4 · Post-training 主线 / Core
- [llm-post-training](cheatsheet-llm-post-training.html) — SFT / RM / RLHF / PPO / DPO 全流程
- [reward-modeling-eval](cheatsheet-reward-modeling-eval.html) — RM 训练 / PRM-ORM / reward hacking / 评测
- [eval-and-judges](cheatsheet-eval-and-judges.html) — 评测三类 / LLM-as-judge / 评测偏置 / 基准与数据污染
- 手撕:[sft-loss-mask](drill-sft-loss-mask.html) · [sequence-packing](drill-sequence-packing.html) · [dpo-loss](drill-dpo-loss.html) · [simpo-loss](drill-simpo-loss.html) · [ppo-clip](drill-ppo-clip.html) · [gae](drill-gae.html) · [reward-margin](drill-reward-margin.html)

## 5 · 推理-RL 前沿 / Frontier
- [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html) — PPO→GRPO→DAPO·Dr.GRPO / RLVR / long-CoT
- 手撕:[grpo](drill-grpo.html) · [rloo](drill-rloo.html)

## 6 · 工程 & 收尾 / Systems
- [ml-system-design](cheatsheet-ml-system-design.html) — 微调流水线 / RLHF 分布式 / 评测体系
- [coding-and-algorithms](cheatsheet-coding-and-algorithms.html) — 算法题型 + ML 实现题

## 7 · 持续 / 终身 / Continual
- [continual-post-training](cheatsheet-continual-post-training.html) — 灾难性遗忘 / replay / 模型合并 / KL · 只收生产验证方法

## 8 · 长程 / Agentic / Long-horizon
- [long-horizon-agents](cheatsheet-long-horizon-agents.html) — computer use / agent 工程 / 难度带奖励 / 自进化(生产 vs 前沿)

---

**复习法**:每题复习后标 ✅ 熟练 / ⚠️ 模糊 / ❌ 不会;之后只重刷 ⚠️/❌。面试前用首页搜索框直接跳到薄弱主题。
