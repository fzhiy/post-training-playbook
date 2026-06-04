# 测试时扩展 (Test-Time Scaling) 速查手册
## 从顺序/并行扩展、搜索、验证器到计算最优分配

---

## 目录 (Table of Contents)

1. [测试时扩展总览 (Overview)](#1-测试时扩展总览-overview)
2. [顺序扩展 (Sequential Scaling)](#2-顺序扩展-sequential-scaling)
3. [并行扩展 (Parallel Scaling)](#3-并行扩展-parallel-scaling)
4. [搜索类方法 (Search-Guided Generation)](#4-搜索类方法-search-guided-generation)
5. [验证器 (Verifiers)](#5-验证器-verifiers)
6. [计算最优与 RL 习得 (Compute-Optimal & RL-Learned Reasoning)](#6-计算最优与-rl-习得-compute-optimal--rl-learned-reasoning)
7. [面试题 (Interview Questions)](#7-面试题-interview-questions)

---

## 1. 测试时扩展总览 (Overview)

**测试时扩展 (test-time scaling, TTS)**,又称推理时扩展 (inference-time scaling),指**在不改变模型权重的前提下,通过在推理阶段投入更多计算来换取更高的准确率**。它是 o1 / R1 范式的核心:模型学会"想得更久",评测口径也从"单次准确率"转向"给定算力预算下的准确率"。

```
训练时算力 (train-time)                 测试时算力 (test-time)
─────────────────────                  ─────────────────────
加参数 / 加数据 / 加训练步              顺序:更长的思考链 (long CoT, budget forcing)
一次投入、固定成本                ⇄     并行:多次采样 + 聚合 (BoN, self-consistency)
推理成本不变                            搜索:验证器引导的树/束搜索 (beam, MCTS)
                                        成本随预算线性/超线性增长
```

### 1.1 范式转移:训练时算力 → 测试时算力

传统 scaling law 关注**训练时**投入(参数、数据、步数);TTS 把另一个旋钮推到台前——**单个 query 上花多少推理算力**。**Snell et al.**（[arXiv:2408.03314](https://arxiv.org/abs/2408.03314), ICLR 2025）系统论证:在固定 FLOPs 预算下,**最优分配测试时算力**(对难题搜索、对易题少采样)在某些设定下**比单纯把参数放大更划算**。这把"能力是否训练即得"的范式撕开一道口子:一部分能力可以在推理时"现场兑换"。

> 💡 一句话直觉:训练时算力提升的是模型的"潜力上限",测试时算力提升的是"单题把潜力兑现的程度"。两者互补——一个会做但常做错的模型,往往能靠 TTS 大幅提分;一个根本不会的题,再多采样也采不出来。

### 1.2 两个轴:顺序 (深度) vs 并行 (宽度)

| 轴 | 机制 | 代表方法 | 成本 | 何时有效 |
|------|------|----------|------|----------|
| **顺序 (sequential)** | 一条链上想得更久、自我修正 | long CoT、budget forcing (s1)、self-refine | 串行延迟高 | 需多步推理、可自我纠错 |
| **并行 (parallel)** | 独立采样多条、再聚合 | best-of-N、self-consistency、repeated sampling | 可并行、吞吐友好 | 答案空间可聚合/可验证 |
| **搜索 (search)** | 树状展开 + 验证器剪枝 | beam search、ToT、MCTS (rStar-Math) | 最高 | 有可靠过程信号 (PRM) |

搜索可视为顺序与并行的混合——在多条部分路径(宽度)上逐步展开(深度),用验证器决定保留哪些。

### 1.3 核心问题:给定算力预算如何最优分配

TTS 的中心问题不是"能不能更准",而是**"同样 N 倍算力,怎么花最值"**:

- **难度自适应**:Snell 的关键发现之一是最优策略**随题目难度变化**——易题适合少量顺序修正,难题适合更宽的并行采样 + 搜索。
- **顺序 vs 并行的配比**:并行采样吃覆盖率 (coverage),顺序修正吃深度,二者最优比例依赖任务。
- **验证器是天花板**:BoN / 加权投票 / 搜索**最终从候选里"挑对的"依赖验证器质量**(见 §5)——没有可靠验证器,采再多也选不准(纯多数投票 self-consistency 例外,它靠答案可聚合性与多数答案是否正确)。

下面 §2–§4 逐一展开三个轴,§5 讲贯穿其中的验证器,§6 讲计算最优分配与"RL 如何教会模型用好测试时算力"。

---

## 2. 顺序扩展 (Sequential Scaling)

顺序扩展指**在一条推理链上投入更多 token**:想得更久、把中间步骤写全、发现错误后回头修正。

### 2.1 long CoT 与 "think longer"

最朴素的顺序扩展就是**让模型生成更长的思维链**。o1 / R1 式模型的核心观察是:经过 RL 训练后,模型在难题上会**自发产生更长的推理**(自我反思、列举情形、验算),且在难题上、合适预算区间内**准确率随推理 token 数上升**(简单题上过长反而掉点,见 §6.4)。注意这里有两层:① 模型**有没有能力**用好更多 token(由训练决定);② 推理时**让不让它**用更多 token(由解码控制决定)。§2.2 处理后者。

### 2.2 预算强制 budget forcing — s1

**s1**（Muennighoff et al., [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)）提出极简的顺序扩展控制——**budget forcing(预算强制)**:在解码时直接操纵"思考预算"。

- **想提前停但没到下限** → 抑制结束符,注入 `"Wait"` 让模型继续想(往往能触发自我纠错);
- **超过上限** → 强行截断思考、要求给出答案。

s1 仅用 **1000 条**精选推理样本做 SFT + budget forcing,就在数学推理上取得有竞争力的结果,论证了"测试时扩展可以很简单"。

```python
# Budget forcing (s1, Muennighoff et al. 2025):解码时控制"思考预算"。
# - 想提前结束但还没到下限 token 数 → 注入 "Wait" 续写,逼模型想更久(常触发自我纠错);
# - 超过上限 → 强行停止思考、转入作答。这里用 mock 解码器演示【控制逻辑】,不含真实模型。
def budget_forcing(model_step, min_think, max_think, wait_token="Wait"):
    """model_step(prefix)->(token, wants_stop):mock 解码一步,
       返回下一个 token 文本与"模型是否想结束思考"。"""
    think_tokens, n_wait = [], 0
    while len(think_tokens) < max_think:
        tok, wants_stop = model_step(think_tokens)
        if wants_stop and len(think_tokens) < min_think:
            think_tokens.append(wait_token)   # 抑制结束、强制续写
            n_wait += 1
            continue
        if wants_stop:
            break
        think_tokens.append(tok)
    return think_tokens, n_wait
```

### 2.3 自我修正 / 修订 (self-refine) 及其局限

**Self-Refine**（Madaan et al., [arXiv:2303.17651](https://arxiv.org/abs/2303.17651), NeurIPS 2023）:用**同一个 LLM** 在循环里"生成 → 自我批判 → 据反馈修订",无需额外训练或监督数据,在多项生成任务上提升质量。

> ⚠️ **自我修正的天花板**:自我修正能否真正纠错,取决于模型**能否可靠地判断自己错了**。多项后续工作发现:在**无外部信号**(无 verifier、无 ground-truth)的纯内在自我纠正下,LLM 常常**改对为错**或在错误上"自我说服",净收益有限甚至为负。可靠的顺序修正通常需要**外部验证信号**(单测、计算器、检索)兜底——这正是 §5 验证器的意义。面试常考点:"自我纠正到底靠不靠谱"——答"取决于有无可靠的外部验证信号"。

---

## 3. 并行扩展 (Parallel Scaling)

并行扩展指**独立采样多条完整解,再用某种方式聚合**。天然可并行、吞吐友好,是工程上最易落地的 TTS。

### 3.1 best-of-N 与 self-consistency(多数投票)

两种最基础的聚合:

- **Self-Consistency**（Wang et al., [arXiv:2203.11171](https://arxiv.org/abs/2203.11171), ICLR 2023）:对同一题采样多条 CoT,**对最终答案做多数投票**。无需验证器,只要答案可归一化比较即可,显著提升数学/常识推理。
- **best-of-N (BoN)**:采 N 条,用**验证器/RM 给每条打分,取最高分的单条**。依赖验证器质量(§5)。

二者的区别:多数投票靠"答案分布的众数",BoN 靠"验证器选优"。**加权 self-consistency** 是中间形态——按验证器分数给每票加权再投。

```python
from collections import defaultdict

# 三种并行聚合(Wang et al. 2022 的 self-consistency + BoN 变体):
# - majority_vote:对【最终答案】做多数投票(无需验证器);
# - weighted_vote:用 verifier/RM 分数给每票加权(加权 self-consistency);
# - best_of_n:直接取验证器分最高的单条(不投票)。
def majority_vote(answers):
    counts = defaultdict(float)
    for a in answers:
        counts[a] += 1.0
    return max(counts, key=counts.get)

def weighted_vote(answers, scores):
    counts = defaultdict(float)        # scores 设为非负(如验证器置信度 ∈ [0,1])
    for a, s in zip(answers, scores):
        counts[a] += s                 # 按验证器分数加权累加
    return max(counts, key=counts.get)

def best_of_n(answers, scores):
    return max(zip(answers, scores), key=lambda x: x[1])[0]  # 取分数最高的单条
```

### 3.2 加权投票 / verifier 重排

当有验证器时,纯多数投票会浪费信号:一条**少数但高置信**的正确答案可能被多数错误答案投掉。加权投票 / verifier 重排让验证器分数参与决策,通常优于纯多数投票——前提是验证器本身可靠(否则把噪声放大)。这也是为什么 §5 验证器质量是 TTS 的核心瓶颈。

### 3.3 覆盖率 vs 精度:pass@k 的天花板

**Large Language Monkeys**（Brown et al., [arXiv:2407.21787](https://arxiv.org/abs/2407.21787)）系统研究"重复采样":定义 **coverage(覆盖率)= 至少有一条样本答对的题目比例**,发现 coverage 随采样数在四个数量级上近似**对数线性**增长。但要警惕两个口径的区别:

$$\text{pass@}k = \mathbb{E}\left[1 - \binom{n-c}{k}\big/\binom{n}{k}\right]$$

- **pass@k / coverage**:"只要采到一条对的就算赢"——是**有 oracle 验证器时的上界**(知道哪条对)。
- **实际可达准确率**:受限于真实验证器能否**从 N 条里挑出**那条对的。

> ❌ **陷阱**:把 coverage / pass@k 当成"模型实际能拿到的分"。coverage 高只说明"答案在候选里",不代表"能选出来"。**采样扩展的实际收益 = coverage 的提升 × 验证器的挑选能力**;验证器弱时,coverage 涨而实际准确率几乎不动。这与 §5 互为表里。RFT 等"采样-筛选-再训练"自训练路线见 [data-pipeline §3](cheatsheet-data-pipeline.html)。

---

## 4. 搜索类方法 (Search-Guided Generation)

搜索把"采样"从"采整条完整解"细化到"**逐步展开 + 中途剪枝**":用验证器对**部分推理路径**打分,只保留有希望的分支继续。它是顺序(深度)与并行(宽度)的混合。

### 4.1 step-level beam search 与 lookahead

把推理切成"步"(step),每步扩展若干候选下一步,用 PRM(过程奖励模型,§5)给**当前部分路径**打分,保留 top-B 继续——即 **step-level beam search**。**lookahead** 进一步:打分前先向前模拟 rollout 几步,用"未来"信息估计当前步的价值(更准但更贵)。

```python
# PRM 引导的 step-level beam search:每步扩展候选下一步,用过程奖励模型(PRM)给
# 【部分推理路径】打分,保留 top-B 继续展开。path 用 step 文本列表表示。
def prm_beam_search(root, expand, prm_score, beam=2, depth=4, n_expand=3):
    """expand(path)->list[next_step];prm_score(path)->float。返回得分最高的完整路径。"""
    beams = [(prm_score([root]), [root])]
    for _ in range(depth):
        cands = []
        for _, path in beams:
            for step in expand(path)[:n_expand]:
                new_path = path + [step]
                cands.append((prm_score(new_path), new_path))
        if not cands:
            break
        cands.sort(key=lambda x: x[0], reverse=True)
        beams = cands[:beam]                 # 剪枝:只留 top-B 分支
    return max(beams, key=lambda x: x[0])
```

### 4.2 Tree of Thoughts / MCTS

- **Tree of Thoughts (ToT)**（Yao et al., [arXiv:2305.10601](https://arxiv.org/abs/2305.10601), NeurIPS 2023）:把推理组织成**树**——节点是连贯的"想法"(thought),模型可生成多个候选想法、**自评估**、并用 BFS/DFS **回溯**搜索,适合需要试错/规划的题。
- **MCTS(蒙特卡洛树搜索)**:用 selection-expansion-simulation-backup 四步在推理树上分配搜索预算,把算力集中到更有希望的分支。**rStar-Math**（Guan et al., [arXiv:2501.04519](https://arxiv.org/abs/2501.04519)）用 MCTS + 过程奖励模型做**自演化**,让 1.5B–7B 的小模型在数学上达到很强水平,且**不依赖从前沿模型蒸馏**。

### 4.3 PRM 引导的搜索

无论 beam 还是 MCTS,**剪枝/选择都依赖一个能给"部分路径"打分的信号**——这正是 PRM(过程奖励模型)的用武之地(ORM 只能给完整解打分,无法直接指导中途剪枝;只能靠 rollout 到底间接估计部分路径价值)。搜索方法的强弱**高度耦合 PRM 的质量**:PRM 噪声大时,搜索会把算力浪费在被错误高估的分支上。PRM 与 ORM 的训练细节见 §5 与 [reward-modeling-eval §2](cheatsheet-reward-modeling-eval.html)。

---

## 5. 验证器 (Verifiers)

验证器是 TTS 的**公共瓶颈**:并行要靠它重排、搜索要靠它剪枝。这里只讲**验证器在测试时如何被消费**;其训练(RM loss、数据构造)见 [reward-modeling-eval §2](cheatsheet-reward-modeling-eval.html)。

### 5.1 ORM vs PRM

| | ORM(结果奖励) | PRM(过程奖励) |
|------|------|------|
| 打分对象 | 只对**完整解/最终答案** | 对**每一推理步** |
| 能否指导搜索 | 否(无法直接给部分路径打分,只能 rollout 到底间接估计) | **能**(beam/MCTS 剪枝的基础) |
| 标注成本 | 低(只需结果对错) | 高(需步级标签 → 见 §5.2 自动标注) |
| 信号粒度 | 粗 | 细,能定位"哪一步开始错" |

**Let's Verify Step by Step**（Lightman et al., [arXiv:2305.20050](https://arxiv.org/abs/2305.20050), ICLR 2024）构建了 **PRM800K**(80 万条人工步级标注),并论证**在 MATH 数学解的 BoN 重排上,过程监督的验证器优于结果监督**。

### 5.2 自动过程标注 — Math-Shepherd

PRM 的痛点是步级标注贵。**Math-Shepherd**（Wang et al., [arXiv:2312.08935](https://arxiv.org/abs/2312.08935), ACL 2024）提出**无需人工**的自动标注:对某一步,从它出发**多次 rollout 到底**,以"该步之后能走到正确答案的比例"作为这一步的过程标签,再训练 PRM。该 PRM 既可用于 BoN 重排,也可作为 PPO 的奖励。

### 5.3 生成式验证器 (generative verifiers)

**Generative Verifiers (GenRM)**（Zhang et al., [arXiv:2408.15240](https://arxiv.org/abs/2408.15240), ICLR 2025）把验证从"判别式打分"改为**"下一个 token 预测"**——让验证器**生成**一段 CoT 判断"这条解对不对",再读出 yes/no 概率作为分数。好处:① 复用 LLM 的生成/推理能力做验证;② 可自身用 CoT + 多数投票扩展验证算力(验证器也能 TTS)。这是"LLM-as-judge"思路在验证场景的体现。

```python
import numpy as np

# PRM 把"每一步的正确概率"聚合成"整条路径分"。三种常见聚合:
# - min :最弱一步决定全局(最严格,常用于步级验证);
# - prod:各步独立正确性连乘(等价于对 log 概率求和);
# - last:只看最后一步(近似结果式打分,非严格 ORM)。
def aggregate_prm(step_probs, mode="min"):
    p = np.asarray(step_probs, dtype=float)
    if mode == "min":
        return float(p.min())
    if mode == "prod":
        return float(p.prod())
    if mode == "last":
        return float(p[-1])
    raise ValueError(f"unknown mode: {mode}")
```

### 5.4 验证器的脆弱性

> ⚠️ 验证器不是 ground truth。它会被**奖励欺骗**(reward hacking)——比如学到"长解=好解"、被特定格式骗、对分布外样本失准。TTS 把验证器推到决策回路的核心,其偏差会被**采样数放大**(采得越多,越可能采到一条"骗过验证器但实际错"的解)。可验证域(数学 exact-match、代码单测)能用**规则化验证器**绕过这一问题——这也是 RLVR / R1 主线选择可验证域的原因(见 §6.3)。RM 的过度优化与长度偏差细节见 [reward-modeling-eval §3–§4](cheatsheet-reward-modeling-eval.html)。

---

## 6. 计算最优与 RL 习得 (Compute-Optimal & RL-Learned Reasoning)

### 6.1 计算最优分配 — Snell et al.

**Snell et al.**（[arXiv:2408.03314](https://arxiv.org/abs/2408.03314), ICLR 2025）提出**compute-optimal(计算最优)** test-time scaling:给定测试时算力预算,**按题目难度自适应地选择策略**(顺序修正 vs 并行采样 vs 搜索),而非一刀切。结论:① 难度自适应分配显著优于固定策略;② 在部分设定下,小模型 + 最优 TTS **可媲美甚至超过大数倍模型的单次推理**——但这有边界,不是普适。

### 6.2 推理 scaling law:test-time vs 加参数的权衡

TTS 与放大参数是**两种花算力的方式**,各有适用区:

- **TTS 更划算**:基模型**已有能力但常出错**(coverage 高、单次准确率低)、且任务**可验证/可聚合**时。
- **加参数更划算**:难题**根本超出模型能力**(coverage 低,采再多也采不到对的)、或**无可靠验证器**从候选里挑选时。
- **本质**:TTS 兑现的是"潜力";潜力不足时,测试时算力的回报会迅速饱和。

### 6.3 RL 习得的 test-time scaling — o1 / R1

TTS 的方法(§2–§4)是**推理时**的解码/搜索策略;但**模型能否用好这些算力,是训练时塑造的**。这正是 o1 / R1 的意义:

- **OpenAI o1**（"Learning to Reason with LLMs", OpenAI, 2024-09,无 arXiv 论文)用大规模 RL 训练模型在作答前进行长思维链推理,公开展示了**准确率随测试时算力提升**的曲线。
- **DeepSeek-R1**（DeepSeek-AI, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948),后发表于 Nature 2025)用 GRPO + 可验证奖励 (RLVR) 训练长 CoT,**长推理行为(自我反思、验算)随 RL 自发涌现**;R1-Zero 进一步表明:在其可验证任务设置下,纯 RL(无 SFT)即可触发。

换言之:**RLVR 在训练时让模型习得有效的长推理行为**,TTS 在推理时把这份能力兑现并分配算力。GRPO / RLVR / R1 四阶段配方等**训练侧**细节见 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html);本页聚焦推理侧方法。

### 6.4 开放问题

- **over-thinking(过度思考)**:简单题上强行拉长 CoT 反而掉点、且浪费算力;如何**自适应决定该想多久**仍是开放问题(budget forcing 是粗粒度尝试)。
- **验证器可靠性**:不可验证域(开放写作、多轮对话)缺乏可靠验证器,TTS 收益受限(§5.4)。
- **可验证域之外的泛化**:RL 习得的推理多在数学/代码上训练,能否泛化到一般推理仍在研究中。
- **测试时算力的经济学**:延迟与成本随预算上升,实际部署需在准确率与单 query 成本间权衡。

---

## 7. 面试题 (Interview Questions)

### L1 — 基础 (Fundamentals)

---

<details>
<summary>Q1: 什么是测试时扩展 (test-time scaling)?它和训练时 scaling 有何区别?</summary>

**答：** TTS 指**不改权重、在推理阶段投入更多计算来换准确率**(想得更久 / 采样更多 / 搜索)。训练时 scaling 是加参数/数据/步数,一次投入、推理成本不变;TTS 是推理时按需投入、成本随预算增长。直觉:训练决定"潜力上限",TTS 决定"单题兑现潜力的程度"。

> **追问：** 是不是任何题目堆测试时算力都能提分?
> 不是。基模型**根本不会**的题(coverage 极低)堆采样也采不到对的;TTS 主要帮"会做但常做错"的题。

</details>

---

<details>
<summary>Q2: 顺序扩展和并行扩展分别是什么?各自的代表方法?</summary>

**答：** **顺序 (sequential)**:在一条链上想得更久 / 自我修正——long CoT、budget forcing (s1)、self-refine,串行延迟高。**并行 (parallel)**:独立采样多条再聚合——best-of-N、self-consistency、repeated sampling,可并行、吞吐友好。**搜索**是二者混合(宽度上逐步展开)。

> **追问：** 工程上优先上哪个?
> 通常先上**并行**(易实现、可并行)。顺序修正延迟高;搜索最强但依赖可靠 PRM。

</details>

---

<details>
<summary>Q3: 什么是 self-consistency?和 best-of-N 有何不同?</summary>

**答：** **Self-Consistency (Wang et al. 2022)** 对同一题采样多条 CoT,**对最终答案做多数投票**,无需验证器。**best-of-N** 采 N 条后用**验证器/RM 打分取最高的单条**。区别:多数投票靠答案分布的众数,BoN 靠验证器选优;**加权 self-consistency** 是中间形态(按验证器分给票加权)。

> **追问：** 没有验证器时只能用哪个?
> 多数投票(self-consistency),只需答案可归一化比较;BoN/加权投票都需要验证器分数。

</details>

---

<details>
<summary>Q4: 什么是 budget forcing?s1 怎么用它?</summary>

**答：** Budget forcing 是 **s1 (Muennighoff et al. 2025)** 的极简顺序扩展控制:解码时操纵"思考预算"——想提前停但没到下限就注入 `"Wait"` 逼模型继续想(常触发自我纠错);超过上限就强行截断、转入作答。s1 仅用 1000 条 SFT 样本 + budget forcing 就在数学推理上取得有竞争力的结果。

> **追问：** "Wait" 为什么能提分?
> 它抑制了过早结束,给模型机会**复查并纠正**前面的推理;但对简单题强行加长可能 over-thinking 反而掉点。

</details>

---

<details>
<summary>Q5: ORM 和 PRM 有什么区别?为什么搜索类方法偏爱 PRM?</summary>

**答：** **ORM(结果奖励)** 只对完整解/最终答案打分;**PRM(过程奖励)** 对每一推理步打分。搜索(beam/MCTS)需要给**部分路径**打分来剪枝——ORM 给不了,只有 PRM 能指导中途选择。代价:PRM 步级标注更贵(见 Math-Shepherd 自动标注)。

> **追问：** PRM 一定比 ORM 好吗?
> 在需要指导搜索 / 定位错误步时 PRM 更优(Lightman et al. 2023);但 PRM 更难训、噪声大时会把搜索算力浪费在被高估的分支上。

</details>

---

<details>
<summary>Q6: 什么是 coverage / pass@k?为什么它是"上界"而非实际准确率?</summary>

**答：** coverage = **至少有一条样本答对的题目比例**;pass@k 是其无偏估计 $1-\binom{n-c}{k}/\binom{n}{k}$。它假设有 **oracle 验证器**(知道哪条对),所以是**上界**。实际可达准确率受限于真实验证器能否**从 N 条里挑出**那条对的——验证器弱时,coverage 涨而实际准确率几乎不动。

> **追问：** Large Language Monkeys 的主要发现?
> coverage 随采样数在约四个数量级上近似**对数线性**增长;"重复采样 + 验证器"在代码/数学上收益很大——但实际收益受验证器质量限制。

</details>

---

<details>
<summary>Q7: 自我修正 (self-refine) 靠谱吗?</summary>

**答：** 取决于**有没有可靠的外部验证信号**。Self-Refine (Madaan et al. 2023) 在有反馈时能提质;但**纯内在**自我纠正(无 verifier、无 ground-truth)下,模型常"改对为错"或自我说服,净收益有限甚至为负。可靠的顺序修正通常要靠单测 / 计算器 / 检索等外部信号兜底。

> **追问：** 这对 agent 设计有什么启示?
> 给 agent 接**真实可执行的反馈**(跑代码、查工具结果)比让它"凭空反思"更可靠。

</details>

---

<details>
<summary>Q8: 测试时扩展和加大模型参数,该选哪个?</summary>

**答：** 看任务。**TTS 更划算**:基模型已有能力但常出错(coverage 高、单次低)、任务可验证/可聚合。**加参数更划算**:难题超出模型能力(coverage 低)、或无可靠验证器从候选里挑。本质:TTS 兑现的是"潜力",潜力不足时回报迅速饱和(Snell et al. 2024 的难度自适应结论)。

> **追问：** 小模型 + 大量 TTS 能否替代大模型?
> 在部分可验证任务上能媲美甚至超过(Snell);但这有边界,不普适——超出基模型能力的题靠 TTS 补不上。

</details>

---

### L2 — 中级 (Intermediate)

---

<details>
<summary>Q9: 为什么说"验证器是 TTS 的天花板"?给一个具体的失效链条。</summary>

**答：** 并行靠验证器重排、搜索靠验证器剪枝——**最终"挑对的"这一步依赖验证器**。失效链条:验证器有偏(如偏好长解)→ 采样越多,越可能采到一条"长但错、却被验证器高估"的解 → BoN/加权投票把它选出来 → 准确率不升反降。即**验证器的偏差被采样数放大**。所以 coverage 高 ≠ 实际分高,二者之间隔着验证器质量。

> **追问：** 怎么缓解?
> 用规则化验证器(可验证域:exact-match / 单测)绕过神经 RM 的 hack;或用更强 / 生成式验证器、对验证器本身也做评估(见 [reward-modeling-eval §5](cheatsheet-reward-modeling-eval.html))。

</details>

---

<details>
<summary>Q10: step-level beam search 和对完整解做 best-of-N,本质区别和适用场景?</summary>

**答：** BoN 是**先采 N 条完整解、最后用 ORM/PRM 重排**(决策只在末尾);beam search 是**每步剪枝、把算力集中到有希望的分支**(决策贯穿全程)。beam 需要 PRM(给部分路径打分),理论上更省算力(早剪掉坏分支),但**对 PRM 噪声更敏感**——一步误剪可能丢掉唯一正确分支。BoN 更鲁棒(保留完整多样性)但更费算力。任务步数多、PRM 可靠时 beam/MCTS 占优;否则 BoN 更稳。

> **追问：** lookahead 解决什么?
> 打分前先向前 rollout 几步,用"未来"信息修正当前步的短视估计,减少误剪;代价是更贵。

</details>

---

<details>
<summary>Q11: Math-Shepherd 如何在没有人工步级标注的情况下训练 PRM?</summary>

**答：** 对某一中间步,**从它出发做多次 rollout 直到给出最终答案**,用"该步之后能 rollout 到正确答案的比例"作为这一步的**过程标签**(蒙特卡洛估计),再用这些自动标签训练 PRM。该 PRM 可用于 BoN 重排,也可作 PPO 奖励。核心是用"结果可验证"反向蒸馏出"过程信号",绕开昂贵的人工步级标注(对比 PRM800K 的 80 万条人工标注)。

> **追问：** 这种自动标签有什么噪声来源?
> rollout 采样有限带来估计方差;且"歪打正着"(错误步偶然走到正确答案)会给错误步打高分——需足够 rollout 数与一致性检查。

</details>

---

<details>
<summary>Q12: 生成式验证器 (GenRM) 相比判别式 RM 的优势?</summary>

**答：** GenRM (Zhang et al. 2024) 把验证建模为**下一个 token 预测**:让验证器先**生成一段 CoT 判断**对错,再读 yes/no 概率作分数。优势:① 复用 LLM 的生成/推理能力(验证也能"想一想");② **验证器自身可做 TTS**——对验证用 CoT + 多数投票扩展算力;③ 与"LLM-as-judge"统一。代价:比判别式打分更贵(要生成)。

> **追问：** 这会不会把生成器的毛病带进验证器?
> 会——GenRM 也可能被对抗解面诱导出错误判断;且自我验证存在"模型偏好自己风格"的偏置,需独立评估验证器质量。

</details>

---

<details>
<summary>Q13: 什么是 compute-optimal test-time scaling?Snell et al. 的核心结论?</summary>

**答：** 给定测试时算力预算,**按题目难度自适应选策略**(易题少量顺序修正、难题更宽并行采样+搜索),而非固定一种。Snell et al. 2024 结论:① 难度自适应分配显著优于固定策略;② 部分设定下,小模型 + 最优 TTS 可媲美甚至超过大数倍模型的单次推理。但有边界——超出基模型能力的题,TTS 回报迅速饱和。

> **追问：** 实践中"题目难度"怎么估?
> 可用模型自身的不确定性 / 早期少量采样的一致性 / 验证器分布来在线估计,据此分配后续预算。

</details>

---

<details>
<summary>Q14: o1 / R1 与 §2–§4 的解码方法是什么关系?RL 在这里扮演什么角色?</summary>

**答：** §2–§4 是**推理时**的解码/搜索策略;但**模型能否用好这些算力是训练时塑造的**。o1 / R1 用大规模 RL(R1 用 GRPO + RLVR)训练长 CoT,使**自我反思/验算等行为自发涌现**(R1-Zero 表明:在其可验证任务设置下,纯 RL 即可触发)。所以:**RLVR 在训练时让模型习得有效的长推理行为,TTS 在推理时兑现并分配这些算力**。训练侧细节见 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html)。

> **追问：** 为什么 o1/R1 主线选可验证域?
> RLVR 用规则化验证器(exact-match / 单测)≈ ground truth,几乎无神经 RM 被 hack 的问题;代价是只适用可验证域。

</details>

---

### L3 — 高级 (Advanced)

---

<details>
<summary>Q15: 从信息论/选择的角度,为什么"采样数 N → ∞"时实际准确率会被验证器质量卡死,而非趋于 coverage?</summary>

**答：** 设 oracle 准确率(=coverage)为 $C$,但我们只有一个**带噪验证器** $v$。最终准确率 = P(被选中的解恰好正确)。当 $N$ 增大,候选里**正确解的数量**和**"看起来更优但实际错"的解的数量同时增长**;若验证器对"错但高分"的解存在系统性偏好(非零的 false-positive 率),这些干扰项的最大分会随 $N$ 上升而**超过**真正确解的分。于是 $\arg\max_v$ 选中正确解的概率被验证器的判别间隔(margin)与 FP 率上界卡死,**不随 N 趋于 C**。在"固定带噪验证器 + 单次打分 + argmax 选择"这组假设下,只有**无噪 oracle**(规则验证器)才能让实际准确率 → coverage(若改用重复验证 / 校准不确定性 / 一致性验证等更强假设,可部分逼近)。这解释了为何可验证域 (RLVR) 是 TTS 最干净的场景。

> **追问：** 这对"加权投票 vs BoN"的选择有何启示?
> 加权投票对单点验证器噪声更鲁棒(多票平滑),BoN 对验证器尾部错误更敏感(单点 argmax)。验证器噪声大时投票更稳;验证器接近 oracle 时 BoN 上界更高。

</details>

---

<details>
<summary>Q16: 推一下 budget forcing 与 RL-trained long-CoT 的耦合:为什么对没经过长 CoT RL 的模型强行 budget forcing 收效甚微?</summary>

**答：** Budget forcing 只是**解码时不让模型停**,它不创造能力——只能**激发模型已习得的**长推理行为。未经长 CoT RL 的模型,其"更多 token"的边际内容多是**重复 / 离题 / 自我说服**(因为分布里没有"高质量的继续反思"这种模式),注入 `"Wait"` 续写出的是低价值 token,准确率几乎不动甚至因噪声下降。而经 RLVR 训练的模型把"反思/验算/换路"压进了分布,`"Wait"` 才能真正触发有效的额外推理。本质:**TTS 的收益 ∝ 模型在长程推理上的"可激发能力"**,这由训练决定。这也呼应 §6.2——TTS 兑现潜力,潜力由训练塑造。

> **追问：** 那 s1 为何只需 1000 条 SFT 就有效?
> s1 的 1000 条是**精选的高质量长推理轨迹**,把"如何有效地继续思考"的模式蒸进了模型,使 budget forcing 有可激发的对象;它走的是 SFT 蒸馏长 CoT 路线,而非从头 RL。

</details>

---

<details>
<summary>Q17: 为什么 PRM 引导的 beam search 在 PRM 有偏时可能【劣于】对完整解做 BoN?给出偏差传播的机制。</summary>

**答：** beam search **每步**都用 PRM 做 argmax 剪枝,误差会**逐步复合**:若 PRM 在某类中间步上有系统偏差,正确分支可能在**早期**就被误剪,而一旦剪掉**无法恢复**(无回溯的 beam)——最终池里可能根本不含正确解,coverage 在搜索内部就已坍塌。BoN 则保留 N 条**完整**解,PRM 只在**末尾**作用一次,正确解(若已采到)始终留在候选池中,验证器偏差只影响"最后排序"而不影响"是否存在"。所以 PRM 偏差大时:beam 的**早剪不可逆**风险 > BoN 的**末端误排**风险。MCTS 因带回溯/探索项(UCT)缓解了一部分早剪问题,介于两者之间。

> **追问：** 怎么让 beam 更鲁棒?
> 加大 beam 宽度(保留更多分支)、引入随机性/温度避免确定性早剪、用 lookahead 减少短视、或软剪枝(按分数采样而非硬 top-B)。本质是用多样性对冲 PRM 偏差。

</details>

---

<details>
<summary>Q18: over-thinking 现象:为什么"想得更久"在简单题上会掉点?如何设计自适应思考预算?</summary>

**答：** 简单题的正确解通常**短且直接**;强行拉长 CoT 会让模型**引入本不必要的步骤**,每多一步都有非零出错概率(错误累积),还可能"想岔了"推翻本来正确的直觉答案——即**推理的边际收益为负**。自适应预算的思路:① 用**早期少量采样的一致性**估难度——前几条采样若高度一致(低熵)则判为易题、早停;② 用**验证器分布**——若 top 候选分数已远超其余则停;③ 训练一个**元控制器/路由**预测每题的最优预算(把"该想多久"本身学出来)。budget forcing 的固定上下限是粗粒度近似,理想是 per-query 自适应。

> **追问：** 自适应预算与 RL 训练目标如何结合?
> 可在 RL 奖励里**对 token 数加惩罚**(长度正则),让模型学会"够用就停";但惩罚过重会压制必要的长推理——又回到 §6.4 的张力。

</details>

---

<details>
<summary>Q19: 测试时扩展会改变"模型评估"的方法论吗?当两个模型在不同测试时预算下比较,怎样才算公平?</summary>

**答：** 会。单一"pass@1"已不足以刻画一个会用 TTS 的模型——必须报告**准确率-算力曲线** (accuracy vs test-time FLOPs / tokens / samples)。公平比较的关键是**对齐算力轴**:① 固定**总推理 FLOPs** 比准确率(而非固定采样数——大模型单次更贵);② 报告**整条曲线**而非单点(曲线可能交叉:小预算下 A 优、大预算下 B 优);③ 区分**有无 oracle 验证器**(coverage/pass@k 是上界,实际部署用真实验证器的准确率)。否则"小模型 + 大量采样 vs 大模型单次"这类比较会因算力轴不对齐而失真。

> **追问：** 这对 leaderboard 设计意味着什么?
> 需公布评测时的测试时预算与解码协议(采样数 / 是否搜索 / 验证器),否则分数不可比;理想是按 iso-FLOPs 或报告 Pareto 前沿。

</details>

---

## 附录：关键术语速查 (Glossary)

| 英文术语 | 中文 | 简要定义 |
|----------|------|----------|
| Test-Time Scaling (TTS) | 测试时扩展 | 不改权重、推理时投更多算力换准确率 |
| Inference-Time Scaling | 推理时扩展 | TTS 的同义说法 |
| Sequential Scaling | 顺序扩展 | 一条链上想得更久 / 自我修正 |
| Parallel Scaling | 并行扩展 | 独立采样多条再聚合 |
| long CoT | 长思维链 | 更长的逐步推理 token 序列 |
| Budget Forcing | 预算强制 | 解码时控制思考预算 (s1):注 "Wait" / 强截断 |
| Self-Refine | 自我修订 | 同一 LLM 生成→自评→修订循环 |
| Self-Consistency | 自洽 | 多条 CoT 对最终答案多数投票 |
| best-of-N (BoN) | — | 采 N 条用验证器取最高分单条 |
| Weighted Vote | 加权投票 | 按验证器分给每票加权再投 |
| Coverage / pass@k | 覆盖率 | 至少一条样本答对的题目比例(oracle 上界) |
| Repeated Sampling | 重复采样 | 大量采样换 coverage (Large Language Monkeys) |
| Beam Search (step-level) | 步级束搜索 | 每步扩展+PRM 剪枝保留 top-B |
| Tree of Thoughts (ToT) | 思维树 | 树状想法 + 自评估 + 回溯搜索 |
| MCTS | 蒙特卡洛树搜索 | select-expand-simulate-backup 分配搜索预算 |
| ORM | 结果奖励模型 | 只对完整解/最终答案打分 |
| PRM | 过程奖励模型 | 对每一推理步打分,可指导搜索剪枝 |
| Math-Shepherd | — | rollout 自动构造步级标签训 PRM,无需人工 |
| Generative Verifier (GenRM) | 生成式验证器 | 把验证建模为 next-token 预测(生成 CoT 判对错) |
| Compute-Optimal TTS | 计算最优扩展 | 按题目难度自适应分配测试时算力 |
| RLVR | 可验证奖励 RL | 用规则验证器 (exact-match/单测) 当奖励 |
| over-thinking | 过度思考 | 简单题强行拉长 CoT 反而掉点 |

---

*本手册仅供学习参考。涉及的论文结论与数值以原始论文为准;benchmark 分数仅用于说明,不构成横向比较。*

## §A 核心论文时间线 / Key Papers Timeline

- **2022-03 · Self-Consistency Improves Chain of Thought Reasoning in Language Models** — Wang et al., ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) — 用采样多条 CoT + 对最终答案多数投票替代贪心解码,在算术/常识推理上显著提升,是最早的并行测试时扩展之一。

- **2023-03 · Self-Refine: Iterative Refinement with Self-Feedback** — Madaan et al., NeurIPS 2023. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) — 用同一 LLM 做"生成→自我批判→修订"循环,无需额外训练即提质;后续工作指出纯内在自我纠正在无外部信号时收益有限。

- **2023-05 · Tree of Thoughts: Deliberate Problem Solving with Large Language Models** — Yao et al., NeurIPS 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) — 把推理组织成树:节点为连贯"想法",支持自评估与 BFS/DFS 回溯搜索,适合需试错/规划的题(另有 Long et al. 2305.08291 同名异作)。

- **2023-05 · Let's Verify Step by Step** — Lightman et al., ICLR 2024. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) — 构建 PRM800K(80 万条人工步级标注),论证过程监督的验证器在数学解 BoN 重排上优于结果监督,确立 PRM 的价值。

- **2023-12 · Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations** — Wang et al., ACL 2024. [arXiv:2312.08935](https://arxiv.org/abs/2312.08935) — 用 rollout"该步之后走到正确答案的比例"自动构造步级标签训 PRM,无需人工标注,可用于 BoN 重排与 PPO 奖励。

- **2024-07 · Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** — Brown et al., preprint. [arXiv:2407.21787](https://arxiv.org/abs/2407.21787) — 系统研究重复采样:coverage 随采样数在约四个数量级上近似对数线性增长,"采样+验证器"在代码/数学上收益大,但实际收益受验证器质量限制。

- **2024-08 · Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** — Snell et al., ICLR 2025. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314) — 提出 compute-optimal TTS:按题目难度自适应分配测试时算力,部分设定下小模型 + 最优 TTS 可媲美大数倍模型的单次推理。

- **2024-08 · Generative Verifiers: Reward Modeling as Next-Token Prediction** — Zhang et al., ICLR 2025. [arXiv:2408.15240](https://arxiv.org/abs/2408.15240) — 把奖励建模/验证改为下一个 token 预测:验证器生成 CoT 判对错并读出 yes/no 概率,可复用生成能力且验证器自身可做 TTS。

- **2024-09 · Learning to Reason with LLMs** — OpenAI, blog(无 arXiv 论文,o1 系列发布)。[openai.com](https://openai.com/index/learning-to-reason-with-llms/) — 公开 o1 系列:用大规模 RL 训练作答前的长思维链推理,展示准确率随测试时算力提升的曲线,引爆"推理时扩展"范式。

- **2025-01 · rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking** — Guan et al., preprint. [arXiv:2501.04519](https://arxiv.org/abs/2501.04519) — 用 MCTS + 过程奖励模型做自演化,让 1.5B–7B 小模型在数学上达到很强水平,且不依赖从前沿模型蒸馏。

- **2025-01 · s1: Simple test-time scaling** — Muennighoff et al., preprint. [arXiv:2501.19393](https://arxiv.org/abs/2501.19393) — 仅用 1000 条精选推理样本 SFT + budget forcing(注 "Wait" / 强截断控制思考预算),在数学推理上取得有竞争力结果,论证 TTS 可以很简单。

- **2025-01 · DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek-AI, Nature 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — 用 GRPO + 可验证奖励训练长 CoT,长推理行为随 RL 自发涌现(R1-Zero 纯 RL 即触发);训练侧配方见 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html),本页聚焦推理侧。
