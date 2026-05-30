# 持续 / 终身 Post-training / Continual & Lifelong(只收生产验证方法)

> 模型是**迭代更新**的(加新数据/新能力/新对齐轮)→ 灾难性遗忘与能力回归是规模化时的真问题。
> ⚠️ 本页**只收在大规模生产中验证过的方法**;经典学术 CL 算法单列、明确标注「未经生产验证」,面试别当工业标准答。

## 1. 生产里「持续」长什么样

不是教科书式的在线流式持续学习,而是:**周期性从 base / checkpoint 重训 + 调数据配比**。目标 = 加新能力/新对齐,同时不退化已有能力(避免 alignment tax / 回归)。

## 1.1 遗忘的机制与度量

**机制**:在新数据上做梯度下降时,更新方向 $-\nabla_\theta \mathcal{L}_{\text{new}}$ 常与旧任务的下降方向**冲突**(gradient interference:$\langle \nabla\mathcal{L}_{\text{old}},\, \nabla\mathcal{L}_{\text{new}}\rangle < 0$),于是权重朝"对新好、对旧差"的方向**漂移**(weight drift),旧能力随之损失。漂移越大、LR 越高、训得越久,遗忘越重——这正是"低 LR + 少 epoch + PEFT"奏效的原因(限制漂移)。

**度量**:用 **BWT(backward transfer)** 量化遗忘。设学完任务 $i$ 后在任务 $j$ 上的指标为 $R_{i,j}$,训完最后任务 $T$ 后:

$$\mathrm{BWT} = \frac{1}{T-1}\sum_{j=1}^{T-1}\big(R_{T,j} - R_{j,j}\big)$$

$\mathrm{BWT}<0$ 即遗忘(越负越严重)。生产里常配合**保持率**(retention)与对旧 benchmark 的**回归探针**(regression probe)一起监控。

## 2. ✅ 生产验证过的工具箱

### 2.1 数据回放 / 混合(replay / rehearsal)—— 最主力
持续微调时混入一定比例的**旧/通用数据**(指令数据配比)。最朴素也最有效的防遗忘手段;工程重点是配比、去重、质量过滤。

### 2.2 低学习率 + 少 epoch + PEFT
小步微调限制权重漂移;LoRA / adapter 做**廉价增量适配 + 改动隔离**(改坏了可丢弃 adapter)。

### 2.3 对 base 的 KL 正则
RLHF 的 $\beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})$ 本质就是把策略**锚在 base 附近**、防漂移与遗忘。

### 2.4 模型合并 / 权重平均
- **model soups**(平均多个微调 checkpoint)、**task arithmetic**(任务向量加减)、**EMA**、**WiSE-FT**。
- 平均 / soup 验证充分;**TIES / DARE** 较新,生产用需自行验证。
- **为何 work(线性模式连通性 LMC)**:从**同一预训练 init** 微调出的多个模型,常落在损失面同一盆地、彼此间**线性插值的 loss barrier 很低**(linearly mode-connected),故平均点仍低损。**前提**:共享 init + 漂移不大。**失效**:不同 init / 任务差异过大 → barrier 高 → 平均反而更差。
- **task arithmetic**:经验上任务向量 $\tau_i=\theta_{\text{ft},i}-\theta_0$ 近似**可线性叠加**(隐含各向量落在不同子空间、相互干扰小),于是 $\theta_0+\sum_i\tau_i$ 同时获多任务、$\theta_0-\tau$ 可"遗忘"某任务。**失效**:任务向量高度相关/干扰、或幅度过大时,加减不再线性叠加(Ilharco et al. 2023 以"interference"刻画失败)。

### 2.5 蒸馏 consolidation
把多个专家 / 更新后的 teacher **蒸馏**成一个模型,巩固能力、压缩多轮迭代。

### 2.6 多阶段顺序遗忘(SFT → DPO → RL)
后阶段会侵蚀前阶段:DPO / RL 阶段的策略漂移会**抹掉部分 SFT 习得的能力与格式**,通常**最后的 RL 步**最严重(无标注约束、只追奖励,易过优化)。缓解:RL 阶段保留对 SFT-ref 的 KL、混入 SFT replay、对关键能力加 verifier 约束;并在每阶段后跑回归探针。

## 3. ❌ 未经生产验证(学术——别当工业标准)

- 正则化系:**EWC、SI、MAS**;梯度投影系:**GEM / A-GEM**;结构系:**PackNet、progressive networks**。
- 这些在 LLM 规模生产里**基本不用**——成本/复杂度高,效果不及朴素的 replay + 合并 + PEFT + KL。
- 面试口径:可提「学术上有 EWC 等」,但要诚实补一句「**生产主流是 replay / 合并 / PEFT / KL**」。

## 4. 把你的 CL 背景诚实地用上

Fed-TaLoRA(联邦持续微调)、Continual Agent → **可迁移的洞察**(遗忘度量、保持率视角、聚合一致性)。
- ✅ 诚实框架:「我研究持续学习,所以理解生产里为什么更朴素的 replay / 合并就够用、以及它们的边界」。
- ❌ 别声称「我做过生产级 continual post-training」。

## 5. 代码：replay 混合 + 权重合并

```python
import torch, itertools

# (1) replay 混合：按比例把旧/通用数据交错进新数据，防遗忘
def make_replay_stream(new_data, old_data, replay_ratio=0.3, seed=0):
    """每条新数据后，以 replay_ratio 概率插入一条循环复用的旧数据。"""
    g = torch.Generator().manual_seed(seed)
    old_cycle = itertools.cycle(old_data)
    stream = []
    for x in new_data:
        stream.append(("new", x))
        if torch.rand(1, generator=g).item() < replay_ratio:
            stream.append(("old", next(old_cycle)))
    return stream

# (2) model soup：等权平均多个同构 checkpoint（需同一 init）
def model_soup(state_dicts):
    avg = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
    for sd in state_dicts:
        for k, v in sd.items():
            avg[k] += v / len(state_dicts)
    return avg

# (3) task arithmetic：θ0 + Σ scale_i·(θ_ft_i − θ0)，加得能力 / 减则遗忘
def task_arithmetic(theta0, finetuned, scales):
    merged = {k: v.clone() for k, v in theta0.items()}
    for sd, s in zip(finetuned, scales):
        for k in merged:
            merged[k] += s * (sd[k] - theta0[k])     # τ_i = θ_ft_i − θ0
    return merged

# --- 玩具验证 ---
t0 = {"w": torch.zeros(3)}
a  = {"w": torch.tensor([1., 0., 0.])}
b  = {"w": torch.tensor([0., 2., 0.])}
print("soup:", model_soup([a, b])["w"])                                  # [0.5, 1.0, 0.0]
print("θ0+τa+τb:", task_arithmetic(t0, [a, b], [1.0, 1.0])["w"])         # [1., 2., 0.]
print("forget b (−τb):", task_arithmetic(t0, [a, b], [1.0, -1.0])["w"])  # [1., -2., 0.]
print("replay stream:", [tag for tag, _ in make_replay_stream(range(4), range(100, 103), 0.5)])
```

## 分层面试题 / Stratified follow-ups

### L1 基础
- 持续微调为什么会遗忘?最朴素有效的防遗忘方法是什么(replay)?为什么 LoRA 有助于减遗忘?

### L2 进阶
- replay 的数据配比怎么定?KL 正则为什么能防遗忘?model soup / 权重平均为什么 work、前提是什么(同一初始化/模式连通性)?

### L3 深挖
- 为什么经典 CL 算法(EWC 等)在 LLM 生产里不流行?task arithmetic 的假设与失效情形?
- 连续 SFT → DPO → RL 多轮里,遗忘最严重在哪一步、怎么缓解?
- 「持续对齐(continual alignment)」与「重训」在成本/效果上怎么权衡?什么时候值得真正做增量而非重训?

## §A 核心论文时间线 / Key Papers Timeline

- **2016-12 · EWC** — Kirkpatrick et al., PNAS 2017. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — 弹性权重巩固:用 Fisher 信息估计各参数对旧任务的重要性,对重要权重加二次惩罚以减遗忘,是正则化系持续学习的代表(LLM 生产少用)。

- **2017-06 · Gradient Episodic Memory** — Lopez-Paz & Ranzato, NeurIPS 2017. [arXiv:1706.08840](https://arxiv.org/abs/1706.08840) — 用情景记忆把新梯度投影到不增旧任务损失的方向;并提出 BWT/FWT 度量,成为量化遗忘的标准指标。

- **2019-12 · Linear Mode Connectivity** — Frankle et al., ICML 2020. [arXiv:1912.05671](https://arxiv.org/abs/1912.05671) — 揭示同一初始化训出的解常落在低 barrier 的连通区域,为权重平均 / model soup 的"为何 work"提供理论前提。

- **2021-09 · WiSE-FT** — Wortsman et al., CVPR 2022. [arXiv:2109.01903](https://arxiv.org/abs/2109.01903) — 微调权重与零样本权重线性插值:一行权重平均同时拿到分布内增益与分布外鲁棒,是抗遗忘式微调的简洁范例。

- **2022-03 · Model Soups** — Wortsman et al., ICML 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) — 把多个微调 checkpoint 直接等权平均("soup"),零额外推理成本下提升精度与鲁棒,确立权重平均合并的主力地位。

- **2022-12 · Task Arithmetic** — Ilharco et al., ICLR 2023. [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) — 提出任务向量 τ=θ_ft−θ0,经验上可线性加减以组合 / 遗忘能力;并用"interference"刻画其失效情形。

- **2023-06 · TIES-Merging** — Yadav et al., NeurIPS 2023. [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — 合并前先裁剪小幅参数、对齐符号、再取一致子集均值,缓解任务向量冲突,显著优于朴素平均。

- **2023-11 · DARE** — Yu et al., ICML 2024. [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) — Drop And REscale:随机丢弃大比例 delta 参数再重缩放,几乎无损地稀疏化任务向量,作为合并前处理可叠加 TIES。
