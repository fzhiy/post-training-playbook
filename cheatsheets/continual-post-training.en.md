# Continual / Lifelong Post-training (production-validated methods only)

> Models are **iteratively updated** (new data / new capabilities / new alignment rounds) → catastrophic forgetting and capability regression are real problems at scale.
> ⚠️ This page **only covers methods validated in large-scale production**; classical academic CL algorithms are listed separately and explicitly marked "not production-validated" — do not cite them as industry-standard answers in interviews.

## 1. What "Continual" Looks Like in Production

Not the textbook online-streaming continual learning setup, but: **periodic retraining from a base / checkpoint with adjusted data mixing ratios**. Goal = add new capabilities / new alignment while not degrading existing capabilities (avoid alignment tax / regression).

## 2. ✅ Production-Validated Toolbox

### 2.1 Data Replay / Rehearsal — the workhorse
Mix a certain proportion of **old / general data** (instruction data mixing ratio) during continual fine-tuning. The simplest and most effective anti-forgetting measure; the engineering focus is on ratio, deduplication, and quality filtering.

### 2.2 Low Learning Rate + Few Epochs + PEFT
Small-step fine-tuning limits weight drift; LoRA / adapters enable **cheap incremental adaptation + change isolation** (a bad adapter can simply be discarded).

### 2.3 KL Regularization Against the Base
The RLHF term $\beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})$ is fundamentally about **anchoring the policy near the base** to prevent drift and forgetting.

### 2.4 Model Merging / Weight Averaging
- **Model soups** (averaging multiple fine-tuned checkpoints), **task arithmetic** (task-vector addition/subtraction), **EMA**, **WiSE-FT**.
- Averaging / soup is well-validated; **TIES / DARE** are newer and require your own validation before production use.

### 2.5 Distillation Consolidation
**Distill** multiple experts / updated teachers into a single model to consolidate capabilities and compress multiple iterative rounds.

## 3. ❌ Not Production-Validated (Academic — Not Industry Standard)

- Regularization-based: **EWC, SI, MAS**; gradient-projection-based: **GEM / A-GEM**; architecture-based: **PackNet, progressive networks**.
- These are **essentially unused** in LLM-scale production — high cost/complexity, worse results than plain replay + merging + PEFT + KL.
- Interview framing: you may mention "academically there's EWC etc.", but honestly add "**the production mainstream is replay / merging / PEFT / KL**".

## 4. Honestly Leveraging Your CL Background

Fed-TaLoRA (federated continual fine-tuning), Continual Agent → **transferable insights** (forgetting metrics, retention perspective, aggregation consistency).
- ✅ Honest framing: "I study continual learning, so I understand why simpler replay / merging suffices in production, and where its limits lie."
- ❌ Don't claim "I've done production-level continual post-training."

## Stratified Interview Follow-ups

### L1 Basics
- Why does continual fine-tuning cause forgetting? What is the simplest effective anti-forgetting method (replay)? Why does LoRA help reduce forgetting?

### L2 Intermediate
- How do you set the data mixing ratio for replay? Why does KL regularization prevent forgetting? Why does model soup / weight averaging work, and what are the prerequisites (same initialization / mode connectivity)?

### L3 Deep Dive
- Why are classical CL algorithms (EWC etc.) unpopular in LLM production? What are the assumptions and failure modes of task arithmetic?
- In a multi-round pipeline of continual SFT → DPO → RL, at which step is forgetting most severe, and how do you mitigate it?
- How do "continual alignment" and "retraining" trade off in terms of cost / effectiveness? When is true incremental learning worth it over full retraining?
