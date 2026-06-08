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

> 💡 **XRIGHT (Fernando et al., arXiv:2410.15483)** 首次给出理论证明:顺序 SFT → DPO/RLHF 的遗忘存在**非消失最优性差距 (non-diminishing optimality gap)**——即只靠多训几轮不能消除遗忘。提出 ALRIGHT / MAXRIGHT 联合优化框架,在 MMLU/HellaSwag/SORRYBench/XSTest 上做到最高 ~23% 提升。

### 2.7 重训 vs 增量——什么时候别省事

| 条件 | 偏向重训 | 偏向增量 |
|------|---------|---------|
| 新数据规模 | 大(>30% 原始数据) | 小(<10%) |
| 新旧任务重叠度 | 低(全新 domain) | 高(同 domain 微调) |
| 对遗忘的容忍度 | 零容忍(安全/合规) | 可接受小幅退化 |
| 计算预算 | 充足 | 受限 |
| 模型规模 | 大(合并/缓存可摊薄成本) | 小(重训便宜) |

**通信/算力成本模型 (粗略):** 重训成本 $C_{\text{retrain}} \propto D_{\text{total}} \times T$;增量成本 $C_{\text{incre}} \propto D_{\text{new}} \times T_{\text{incre}}$ (通常 $T_{\text{incre}} \ll T$ 但需加 replay 成本 $\alpha D_{\text{old}}$)。

**经验法则:** 如果 $D_{\text{new}} / D_{\text{total}} < 0.1$ 且任务相似 → 增量 + replay;如果多次增量后出现累积退化 → 重训。Google PaLM-2 合并研究表明**大模型对简单平均合并更鲁棒**,甚至简单平均就能接近多任务训练——所以"增量 → 定期重训 + 合并 checkpoint"是当前主流模式。



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

## 6. 面试题 / Interview Questions

### L1 — 基础 (Fundamentals)

---

<details class="qa"><summary>Q1: 持续微调为什么会遗忘?最朴素有效的防遗忘方法是什么?</summary>

**答：** 遗忘根源是**梯度冲突 + 权重漂移**:在新数据上梯度下降时,更新方向 $-\nabla_\theta \mathcal{L}_{\text{new}}$ 常与旧任务梯度冲突 ($\langle\nabla\mathcal{L}_{\text{old}},\nabla\mathcal{L}_{\text{new}}\rangle < 0$),权重朝"对新好、对旧差"的方向漂移,旧能力随之损失。漂移越大、LR 越高、训得越久,遗忘越重。**最朴素有效的方法是 replay(数据回放/混合)**:在新数据中混入一定比例旧/通用数据,从数据层面直接告诉模型"别忘了"。工程重点是配比、去重、质量过滤。

> **追问：** 为什么低 LR + 少 epoch 能减遗忘?
> 因为限制了每步权重漂移的幅度——小步微调 = 对旧知识扰动小。同理,LoRA 通过低秩 adapter 限制改动容量,改坏了可丢弃 adapter 回退。

</details>

---

<details class="qa"><summary>Q2: 为什么 LoRA / PEFT 有助于减缓遗忘?</summary>

**答：** 三层原因:① **改动隔离**——全参微调会修改所有权重,低秩 adapter 只改一小部分参数,对旧能力的物理扰动小;② **可丢弃**——adapter 改坏了可以直接丢弃回退到 base model,全参微调无法干净回退;③ **廉价增量适配**——每个新任务/新数据配一个独立 adapter,推理时按需组合,避免"一个模型同时满足所有任务"导致的干扰。代价:LoRA 的表达力有限,复杂新能力可能需要更高秩或全参微调。

> **追问：** LoRA 能完全避免遗忘吗?
> 不能。LoRA 限制的是"改动幅度"而非"改动方向"——如果新数据与旧知识强冲突,LoRA 仍会覆盖旧能力,只是幅度更小。防遗忘仍需配合 replay / KL 正则 / 合并等。

</details>

---

<details class="qa"><summary>Q3: BWT (backward transfer) 怎么度量遗忘?BTW vs 保持率的区别?</summary>

**答：** BWT = $\frac{1}{T-1}\sum_{j=1}^{T-1}(R_{T,j} - R_{j,j})$,其中 $R_{i,j}$ = 学完任务 $i$ 后在任务 $j$ 上的指标。BWT < 0 即遗忘(越负越严重)。**vs 保持率 (retention)**: BWT 直接量化绝对退化量($\Delta$),保持率 = $R_{T,j} / R_{j,j}$ 度量相对保留比例——两者互补。生产里常配合对旧 benchmark 的**回归探针**一起监控,不只看汇总的 BWT。

> **追问：** BWT 的局限是什么?
> 它只测"忘了多少"不测"学了多少"——搭配 FWT (forward transfer,新知识正向迁移)一起看才完整:$BWT+FWT$ 是增量更新是否"净赚"的判据。此外,BWT 假设任务可独立评测,对于交织的对话/推理能力,需精心选择 proxy benchmark。

</details>

---

### L2 — 中级 (Intermediate)

---

<details class="qa"><summary>Q4: replay 的数据配比怎么定?什么情况下 replay 不划算?</summary>

**答：** 配比没有万能公式,实践经验法则:① **规模定**:旧数据量 $\approx$ 新数据量的 10–50%(取决于新旧任务相似度);② **动态调**:监控旧 benchmark 上的保持率,退化超过阈值(如 2%)→ 增大 replay ratio;③ **质量优于数量**:少量高质量、代表性强的旧样本比大量低质旧数据更有效。**不划算的情况**:新旧任务差异极大(如从代码切到医疗)、旧数据量太大无法全部 replay、存储/带宽限制——此时优先考虑 LoRA 隔离或模型合并。

> **追问：** replay 和 KL 正则的关系?
> 互补而非替代。Replay 从**数据层面**直接提醒模型旧分布;KL 正则从**优化层面**约束策略不偏离 ref。RLHF 场景下,KL 是必备防漂移手段(replay 通常也加)。两者叠加通常比单用效果好。

</details>

---

<details class="qa"><summary>Q5: model soup / 权重平均为什么 work?前提与失效条件?</summary>

**答：** 理论基础是**线性模式连通性 (Linear Mode Connectivity, LMC)**:从同一预训练 init 微调出的多个模型,常落在损失面同一盆地,彼此间**线性插值的 loss barrier 很低**(linearly mode-connected),故平均点仍低损。**前提**:共享 init + 漂移不大(低 LR 微调)。**失效**:不同 init / 任务差异过大 → barrier 高 → 平均反而更差。**Task arithmetic** ($\theta_0 + \sum \tau_i$) 进一步假设任务向量可线性叠加(隐含各向量落在不同子空间、相互干扰小);失效时用 **TIES**(先裁剪小幅参数、对齐符号、再取一致子集平均)或 **DARE**(随机丢弃大比例 delta 再重缩放)做预处理,可显著优于朴素平均。

> **追问：** TIES-Merging 的三步在做什么?
> ① **Trim**: 重置变化微小的参数(稀疏化 delta,减少噪声);② **Elect Signs**: 跨模型投票确定每个参数的主导方向(解决正负冲突);③ **Disjoint Merge**: 只合并方向一致的参数。三步之后,参数冲突被消解,合并质量大幅提升。

</details>

---

<details class="qa"><summary>Q6: 为什么经典 CL 算法 (EWC / GEM / SI) 在 LLM 生产里不流行?</summary>

**答：** 四个核心原因:① **成本**:EWC 需要计算/存储 Fisher 信息矩阵(LLM 参数量级下不可承受),GEM 需要每步求解投影(计算量随 episodic memory 增大);② **复杂度**:超参多、难调,且对 LLM 这种"能力交织"的任务,正则化目标难以精确定义"哪些参数对哪些任务重要";③ **效果不及简单方法**:朴素的 replay + 合并 + PEFT + KL 在实践中往往持平或优于经典 CL 算法,且更简单;④ **规模不适配**:经典 CL 方法设计时面向小模型/小任务,LLM 的 100B+ 参数和千级任务使它们计算上不可行。

> **追问：** 学术上 EWC 等还有价值吗?
> 有——它们提供了遗忘的理论框架(重要性加权、梯度投影、情景记忆压缩),启发了 LLM 时代更轻量的变体。但在面试中:可以提「学术上有 EWC 等」,但要诚实补一句「生产主流是 replay / 合并 / PEFT / KL」。

</details>

---

### L3 — 高级 (Advanced)

---

<details class="qa"><summary>Q7: 连续 SFT → DPO → RL 多轮里,遗忘最严重在哪一步?怎么缓解?</summary>

**答：** 最严重在**最后的 RL 步**——无标注约束、只追奖励,极易过优化(策略漂移抹掉 SFT 习得的能力与格式)。**XRIGHT (Fernando et al., arXiv:2410.15483)** 首次理论证明顺序训练的遗忘存在**非消失最优性差距**(多训不能自动消除)。缓解:① RL 阶段保留对 SFT-ref 的 KL 约束;② 混入 SFT replay(数据回放);③ 对关键能力加 verifier 约束(不只是 reward 打分);④ 每阶段后跑**回归探针**——监控旧 benchmark 上的退化量,超阈值则回滚或调整配比;⑤ **模型平均**:Wang (ACL 2024) 发现对预训练和 RLHF 后的权重做 per-layer 插值,能在"对齐-能力"帕累托前沿上拿到最优权衡点。

> **追问：** 「持续对齐 (continual alignment)」vs「重训」怎么选?
> 如果新对齐需求相对旧的对齐增量小且同分布 → 增量 RL 微调 + KL 锚定 ref;如果多轮增量后累积退化明显(regression probe 报警) → 从 base checkpoint 重训并合并旧 adapter/checkpoint。Google PaLM-2 实证:**大模型对简单平均合并更鲁棒**——"增量 → 定期重训 + 合并 checkpoint"是当前主流模式。

</details>

---

<details class="qa"><summary>Q8: task arithmetic 的假设是什么?什么情况下会失效?</summary>

**答：** Task arithmetic 假设任务向量 $\tau_i = \theta_{\text{ft},i} - \theta_0$ 近似**可线性叠加**——隐含前提:① 各 $\tau_i$ 落在向量空间的不同子空间,相互干扰小;② 改动幅度不太大(避免非线性效应)。**失效条件**:① 任务高度相关/冲突 → 叠加后干扰(Ilharco et al. 用 "interference" 刻画失败);② 漂移幅度过大→线性叠加假设不再成立(损失面不再平坦);③ 不同 init 微调的模型不共享 LMC → 线性插值的 barrier 高。**缓解**:TIES-Merging(裁剪+符号对齐+合并)、DARE(随机丢弃 delta + 重缩放)作为预处理可显著降低合并时的参数冲突。**大模型实证 (Google, 64B PaLM-2)**:模型越大,不同合并方法(TIES/DARE-TIES/Averaging)的结果越接近——甚至简单平均就足够好。

> **追问：** 为什么大模型合并更容易?
> 大模型的参数冗余更多,不同微调产生的 delta 落在参数空间的 sparse/orthogonal 子空间的可能性更高,相互干扰小——所以简单平均就有效。这是"过参数化红利"在持续学习场景的体现。

</details>

---

<details class="qa"><summary>Q9: 设计一套生产级持续 post-training 监控方案,要覆盖哪些维度?</summary>

**答：**

| 维度 | 指标/工具 | 阈值/触发 |
|------|----------|----------|
| **遗忘监控** | BWT + 保持率 + 回归探针(旧 benchmark) | 保持率退化 >2% → 增 replay ratio 或回滚 |
| **新能力验收** | 新 benchmark 指标 vs 重训 baseline | 增量效果 < 重训的 80% → 考虑重训 |
| **对齐/安全** | 安全拒答率 / over-refusal / 越狱 ASR | 任一项退化 >1% → 暂停增量,审计 |
| **成本追踪** | 累积增量成本 vs 重训成本 | $C_{\text{incre,cumulative}} > C_{\text{retrain}}$ → 重训 |
| **漂移审计** | 权重 L2 距离 vs base / 激活分布 KL | 异常漂移 → 检查数据配比、LR、epoch |

**关键原则**:不要只看汇总指标——分能力维度监控(推理/代码/安全/对话各有什么趋势),汇总 BWT 会掩盖特定能力的严重退化。

> **追问：** COPR (Zhan et al., arXiv:2402.14228) 的 benchmark 贡献了什么?
> 建立了首个**持续人类偏好对齐 benchmark**——含 reward-based + GPT-4 + 人工三层评测,覆盖多 backbone、多 replay 配置、多学习顺序。提出拉格朗日对偶动态正则化(每步正则化当前策略 vs 历史上最优策略),在不同 backbone 和 replay size 下均鲁棒。

</details>

## §A 核心论文时间线 / Key Papers Timeline

- **2016-12 · EWC** — Kirkpatrick et al., PNAS 2017. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — 弹性权重巩固:用 Fisher 信息估计各参数对旧任务的重要性,对重要权重加二次惩罚以减遗忘,是正则化系持续学习的代表(LLM 生产少用)。

- **2017-06 · Gradient Episodic Memory** — Lopez-Paz & Ranzato, NeurIPS 2017. [arXiv:1706.08840](https://arxiv.org/abs/1706.08840) — 用情景记忆把新梯度投影到不增旧任务损失的方向;并提出 BWT/FWT 度量,成为量化遗忘的标准指标。

- **2019-12 · Linear Mode Connectivity** — Frankle et al., ICML 2020. [arXiv:1912.05671](https://arxiv.org/abs/1912.05671) — 揭示同一初始化训出的解常落在低 barrier 的连通区域,为权重平均 / model soup 的"为何 work"提供理论前提。

- **2021-09 · WiSE-FT** — Wortsman et al., CVPR 2022. [arXiv:2109.01903](https://arxiv.org/abs/2109.01903) — 微调权重与零样本权重线性插值:一行权重平均同时拿到分布内增益与分布外鲁棒,是抗遗忘式微调的简洁范例。

- **2022-03 · Model Soups** — Wortsman et al., ICML 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) — 把多个微调 checkpoint 直接等权平均("soup"),零额外推理成本下提升精度与鲁棒,确立权重平均合并的主力地位。

- **2022-12 · Task Arithmetic** — Ilharco et al., ICLR 2023. [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) — 提出任务向量 τ=θ_ft−θ0,经验上可线性加减以组合 / 遗忘能力;并用"interference"刻画其失效情形。

- **2023-06 · TIES-Merging** — Yadav et al., NeurIPS 2023. [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — 合并前先裁剪小幅参数、对齐符号、再取一致子集均值,缓解任务向量冲突,显著优于朴素平均。

- **2023-11 · DARE** — Yu et al., ICML 2024. [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) — Drop And REscale:随机丢弃大比例 delta 参数再重缩放,几乎无损地稀疏化任务向量,作为合并前处理可叠加 TIES。

- **2024-02 · COPR** — Zhan et al., arXiv. [arXiv:2402.14228](https://arxiv.org/abs/2402.14228) — 首个持续人类偏好对齐 benchmark(三层评测:reward + GPT-4 + 人工);提出拉格朗日对偶动态正则化,在不同 backbone 和 replay 配置下鲁棒。

- **2024-06 · Online DPO (OFS-DPO)** — Qi et al., arXiv. [arXiv:2406.05534](https://arxiv.org/abs/2406.05534) — fast-slow LoRA 模块对模拟种内竞争以缓解 DPO 的灾难性遗忘;扩展到跨域持续对齐(COFS-DPO),推导 online learning 的 regret 上界。

- **2024-08 · Model Merging Survey** — Yang et al., arXiv. [arXiv:2408.07666](https://arxiv.org/abs/2408.07666) — 最全面的模型合并综述:提出新分类法,覆盖 LLM/MLLM 及 10+ ML 子领域应用,含持续学习/多任务/少样本等。

- **2024-10 · XRIGHT (ALRIGHT/MAXRIGHT)** — Fernando et al., arXiv. [arXiv:2410.15483](https://arxiv.org/abs/2410.15483) — 首次理论证明顺序 SFT→DPO/RLHF 的遗忘存在非消失最优性差距;提出联合优化框架,在 MMLU/HellaSwag/SORRYBench/XSTest 上提升最高 ~23%。

- **2024-10 · What Matters for Model Merging at Scale** — Google/UNC. [arXiv:2410.03617](https://arxiv.org/abs/2410.03617) — 大规模合并实证(到 64B PaLM-2,合并 8 个 expert):大模型对简单平均更鲁棒,不同合并方法结果趋近;合并模型可超多任务训练。

- **2024-10 · UFT** — Wang et al., arXiv. [arXiv:2410.21438](https://arxiv.org/abs/2410.21438) — 将 SFT 和对齐(RLHF/DPO)统一为**单个训练阶段**以预防遗忘;在 IFEval 和 Truthful-QA 上取得增益,提出 pretraining→UFT 范式的可能性。
