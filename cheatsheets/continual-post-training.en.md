# Continual / Lifelong Post-training (production-validated methods only)

> Models are **iteratively updated** (new data / new capabilities / new alignment rounds) → catastrophic forgetting and capability regression are real problems at scale.
> ⚠️ This page **only covers methods validated in large-scale production**; classical academic CL algorithms are listed separately and explicitly marked "not production-validated" — do not cite them as industry-standard answers in interviews.

## 1. What "Continual" Looks Like in Production

Not the textbook online-streaming continual learning setup, but: **periodic retraining from a base / checkpoint with adjusted data mixing ratios**. Goal = add new capabilities / new alignment while not degrading existing capabilities (avoid alignment tax / regression).

## 1.1 Mechanism & Measurement of Forgetting

**Mechanism:** gradient descent on new data often produces an update direction $-\nabla_\theta \mathcal{L}_{\text{new}}$ that **conflicts** with the old task's descent direction (gradient interference: $\langle \nabla\mathcal{L}_{\text{old}},\, \nabla\mathcal{L}_{\text{new}}\rangle < 0$), so weights **drift** toward "good-for-new, bad-for-old" (weight drift) and old capabilities are lost. Larger drift, higher LR, and longer training all worsen forgetting — which is exactly why "low LR + few epochs + PEFT" works (it limits drift).

**Measurement:** use **BWT (backward transfer)**. Let $R_{i,j}$ be the metric on task $j$ after learning task $i$; after the final task $T$:

$$\mathrm{BWT} = \frac{1}{T-1}\sum_{j=1}^{T-1}\big(R_{T,j} - R_{j,j}\big)$$

$\mathrm{BWT}<0$ means forgetting (more negative = worse). In production it is monitored alongside **retention** and **regression probes** on old benchmarks.

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
- **Why it works (linear mode connectivity, LMC):** models fine-tuned from the **same pretrained init** often land in the same loss basin, with a **low loss barrier along the linear interpolation** between them (linearly mode-connected), so the averaged point stays low-loss. **Premise:** shared init + small drift. **Failure:** different inits / overly divergent tasks → high barrier → averaging hurts.
- **Task arithmetic:** empirically, task vectors $\tau_i=\theta_{\text{ft},i}-\theta_0$ compose approximately **linearly** (implicitly occupying different subspaces / low interference), so $\theta_0+\sum_i\tau_i$ gains multiple tasks and $\theta_0-\tau$ can "forget" one. **Failure:** when task vectors are highly correlated/interfering or too large in magnitude, addition/subtraction no longer composes linearly (Ilharco et al. 2023 frame failure as "interference").

### 2.5 Distillation Consolidation
**Distill** multiple experts / updated teachers into a single model to consolidate capabilities and compress multiple iterative rounds.

### 2.6 Sequential Forgetting Across Stages (SFT → DPO → RL)
Later stages erode earlier ones: the policy drift in the DPO / RL stage **wipes out part of the capabilities and formatting learned in SFT**, usually worst at the **final RL step** (no labeled constraint, chasing reward, prone to over-optimization). Mitigations: keep a KL to the SFT reference during RL, mix in SFT replay, add verifier constraints on key capabilities, and run regression probes after each stage.

## 3. ❌ Not Production-Validated (Academic — Not Industry Standard)

- Regularization-based: **EWC, SI, MAS**; gradient-projection-based: **GEM / A-GEM**; architecture-based: **PackNet, progressive networks**.
- These are **essentially unused** in LLM-scale production — high cost/complexity, worse results than plain replay + merging + PEFT + KL.
- Interview framing: you may mention "academically there's EWC etc.", but honestly add "**the production mainstream is replay / merging / PEFT / KL**".

## 4. Honestly Leveraging Your CL Background

Fed-TaLoRA (federated continual fine-tuning), Continual Agent → **transferable insights** (forgetting metrics, retention perspective, aggregation consistency).
- ✅ Honest framing: "I study continual learning, so I understand why simpler replay / merging suffices in production, and where its limits lie."
- ❌ Don't claim "I've done production-level continual post-training."

## 5. Code: replay mixing + weight merging

```python
import torch, itertools

# (1) replay mixing: interleave old/general data into new data by ratio, to fight forgetting
def make_replay_stream(new_data, old_data, replay_ratio=0.3, seed=0):
    """After each new sample, insert one cyclically-reused old sample with prob replay_ratio."""
    g = torch.Generator().manual_seed(seed)
    old_cycle = itertools.cycle(old_data)
    stream = []
    for x in new_data:
        stream.append(("new", x))
        if torch.rand(1, generator=g).item() < replay_ratio:
            stream.append(("old", next(old_cycle)))
    return stream

# (2) model soup: equal-weight average of homogeneous checkpoints (must share init)
def model_soup(state_dicts):
    avg = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
    for sd in state_dicts:
        for k, v in sd.items():
            avg[k] += v / len(state_dicts)
    return avg

# (3) task arithmetic: theta0 + sum_i scale_i*(theta_ft_i - theta0); add to gain, subtract to forget
def task_arithmetic(theta0, finetuned, scales):
    merged = {k: v.clone() for k, v in theta0.items()}
    for sd, s in zip(finetuned, scales):
        for k in merged:
            merged[k] += s * (sd[k] - theta0[k])     # tau_i = theta_ft_i - theta0
    return merged

# --- Toy check ---
t0 = {"w": torch.zeros(3)}
a  = {"w": torch.tensor([1., 0., 0.])}
b  = {"w": torch.tensor([0., 2., 0.])}
print("soup:", model_soup([a, b])["w"])                                  # [0.5, 1.0, 0.0]
print("theta0+tau_a+tau_b:", task_arithmetic(t0, [a, b], [1.0, 1.0])["w"])   # [1., 2., 0.]
print("forget b (-tau_b):", task_arithmetic(t0, [a, b], [1.0, -1.0])["w"])   # [1., -2., 0.]
print("replay stream:", [tag for tag, _ in make_replay_stream(range(4), range(100, 103), 0.5)])
```

## Stratified Interview Follow-ups

### L1 Basics
- Why does continual fine-tuning cause forgetting? What is the simplest effective anti-forgetting method (replay)? Why does LoRA help reduce forgetting?

### L2 Intermediate
- How do you set the data mixing ratio for replay? Why does KL regularization prevent forgetting? Why does model soup / weight averaging work, and what are the prerequisites (same initialization / mode connectivity)?

### L3 Deep Dive
- Why are classical CL algorithms (EWC etc.) unpopular in LLM production? What are the assumptions and failure modes of task arithmetic?
- In a multi-round pipeline of continual SFT → DPO → RL, at which step is forgetting most severe, and how do you mitigate it?
- How do "continual alignment" and "retraining" trade off in terms of cost / effectiveness? When is true incremental learning worth it over full retraining?
