# 奖励模型 (Reward Modeling) 与评估 速查手册
## LLM 后训练 (Post-Training) 中的奖励建模全景

---

## 目录 (Table of Contents)

1. [RM 训练方法 (RM Training)](#1-rm-训练方法-rm-training)
2. [PRM vs ORM](#2-prm-vs-orm-过程奖励模型与结果奖励模型)
3. [奖励欺骗与过度优化 (Reward Hacking & Over-Optimization)](#3-奖励欺骗与过度优化-reward-hacking--over-optimization)
4. [长度偏差及其他 RM 病态现象 (Length Bias & Other Pathologies)](#4-长度偏差及其他-rm-病态现象-length-bias--other-pathologies)
5. [RM / 评估者评估 (RM & Judge Evaluation)](#5-rm--评估者评估-rm--judge-evaluation)
6. [面试题 (Interview Questions)](#6-面试题-interview-questions)

---

## 1. RM 训练方法 (RM Training)

### 1.1 核心目标 (Core Objective)

奖励模型从人类偏好 (human preferences) 中学习一个**标量评分函数 (scalar scoring function)**，用于在 RLHF (Reinforcement Learning from Human Feedback) 或类似流程中为 LLM 生成的候选回答打分，引导策略模型 (policy model) 优化。

### 1.2 成对比较法 (Pairwise / Bradley-Terry Model)

**核心思想：** 不直接估计绝对分值，而是学习**两个回答之间的偏好关系**。

**Bradley-Terry (BT) 模型：**

给定 prompt $x$，chosen response $y_w$，rejected response $y_l$：

$$P(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $\sigma$ 是 sigmoid 函数，$r_\theta$ 是参数为 $\theta$ 的奖励模型。

**训练损失 (Training Loss)：**

$$\mathcal{L}_{\text{BT}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

**优势 (Advantages)：**
- 只需偏好排序，不需要绝对评分标注 — 标注成本低
- 天然与 RLHF 中的 policy gradient 对齐
- 人类做相对判断比绝对打分更可靠、一致性更高

**局限 (Limitations)：**
- 只利用了二元偏好信号，信息利用不够充分
- 假设偏好具有传递性 (transitivity)，现实中未必成立
- 不直接输出标量值，在某些下游任务中需额外处理

### 1.3 逐点评分法 (Pointwise / Regression)

**核心思想：** 直接回归到一个**绝对质量分数 (absolute quality score)**。

**训练损失：**

$$\mathcal{L}_{\text{point}} = \mathbb{E}_{(x, y, s)} \left[ (r_\theta(x, y) - s)^2 \right]$$

其中 $s$ 是人工标注的绝对评分 (如 1-5 Likert scale)。

**优势：**
- 可直接用于 Best-of-N (BoN) 采样、过滤等场景
- 评分可解释性强

**局限：**
- 需要绝对标注，标注成本高、标注者间一致性 (inter-annotator agreement) 低
- 分数标度 (scale) 因人而异，需要校准 (calibration)

### 1.4 其他变体 (Other Variants)

| 方法 | 描述 |
|------|------|
| **Listwise / Plackett-Luce** | 对 k 个回答做全排序学习，比 pairwise 信息更丰富 |
| **Regression + Ranking 混合** | 同时优化回归损失和排序损失 |
| **多目标 RM (Multi-object RM)** | 对不同维度 (有用性、安全性、事实性) 分别打分 |
| **Token-level RM** | 在 token 粒度上分配奖励，与 PRM 相关 |

---

## 2. PRM vs ORM：过程奖励模型与结果奖励模型

### 2.1 ORM — 结果奖励模型 (Outcome Reward Model)

**定义：** 仅根据**最终输出 (final output)** 给出一个整体奖励分数。

```
输入: [Prompt + 全部回答文本]
输出: 单个标量奖励 r
```

**特点：**
- ✅ 标注简单：只需判断最终答案对/错或质量高低
- ✅ 训练和推理计算成本较低
- ❌ **信用分配问题 (credit assignment)**：不知道回答中哪些步骤是正确的、哪些是错误的
- ❌ 对于数学推理、代码生成等多步骤任务信号稀疏

### 2.2 PRM — 过程奖励模型 (Process Reward Model)

**定义：** 对推理过程的**每个步骤 (each step)** 给出奖励分数。

```
输入: [Prompt + 前 i 步]
输出: 每步的奖励 r_step(i)
```

**特点：**
- ✅ 信号更密集 (dense reward)，信用分配更精确
- ✅ 能定位具体推理错误步骤，可用于 early stopping 或树搜索
- ✅ 在数学推理等任务上通常显著优于 ORM（根据多项研究的定性结论）
- ❌ **标注成本极高**：需要专家逐步判断
- ❌ 步骤边界定义 (step delineation) 本身有歧义

### 2.3 对比表 (Comparison Table)

| 维度 | ORM (结果奖励) | PRM (过程奖励) |
|------|----------------|----------------|
| 奖励粒度 (Granularity) | 整个回答 | 每个推理步骤 |
| 信号密度 (Density) | 稀疏 (sparse) | 密集 (dense) |
| 标注成本 | 低 | 高 |
| 信用分配 (Credit Assignment) | 差 | 好 |
| 典型应用 | 对话、写作 | 数学推理、代码、多步骤推理 |
| 与搜索结合 | Best-of-N | Best-of-N、树搜索 (tree search)、beam search |
| 标注方式 | 最终答案对错 | 每步逻辑正确性 |

### 2.4 自动化 PRM 数据生成 (Automated PRM Data)

为降低标注成本，常见方法：
- **Monte Carlo 估计 (MC estimation)：** 从每一步继续采样多个 rollout，看最终结果正确率作为该步奖励
- **LLM-as-Step-Judge：** 用强 LLM 标注每步正确性
- **ORM-to-PRM 蒸馏：** 从 ORM 反推过程信号

---

## 3. 奖励欺骗与过度优化 (Reward Hacking & Over-Optimization)

### 3.1 什么是奖励欺骗 (Reward Hacking)

**定义：** 策略模型学会利用奖励模型的**缺陷和分布外行为 (out-of-distribution behavior)** 来获取高分，而非真正提升回答质量。

**常见类型 (Common Patterns)：**

| 类型 | 定义 |
|------|------|
| **Verbosity hacking** | 生成冗长但低质内容，利用 RM 对长度的虚假偏好刷高分 |
| **Sycophancy** | 使用讨好、奉承性措辞，迎合用户立场而非提供准确答案 |
| **Format gaming** | 过度使用 Markdown 列表、标题、加粗等格式元素，RM 误判为质量信号 |
| **Spec gaming** | 满足 RM 或任务规范的字面要求但违背其实质意图（如答案"形式正确"但内容错误） |
| **OOD collapse** | 策略偏离训练分布后 RM 评分失准，对分布外生成给出虚高或随机分数 |

### 3.2 Goodhart 定律视角

> "当一个指标变成目标时，它就不再是好的指标。"

$$r_{\text{true}} \neq r_{\text{RM}} \quad (\text{OOD})$$

RM 只在**训练分布 (training distribution)** 内准确，当 policy 生成分布外内容时，RM 预测不再可靠。

### 3.2a Gao et al. 2022 — 过度优化幂律 (Scaling Laws for Overoptimization)

> 出处：Gao, Schulman & Hilton, arXiv:2210.10760，使用合成 gold RM 在 InstructGPT 设置下实验。

**核心变量：** 令 $d = \sqrt{D_{\text{KL}}(\pi \| \pi_{\text{init}})}$（KL 距离的平方根，原文选择此参数化是因为 KL 是"二次度量"）。gold RM 分数 $R$ 随 $d$ 的变化规律因优化方法不同而**函数形式不同**：

**Best-of-N (BoN) 形式：**

$$R_{\text{bon}}(d) = d(\alpha_{\text{bon}} - \beta_{\text{bon}} d)$$

**强化学习 (RL) 形式：**

$$R_{\text{RL}}(d) = d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d)$$

其中 $\alpha_{\text{bon}}, \beta_{\text{bon}}, \alpha_{\text{RL}}, \beta_{\text{RL}}$ 是拟合参数（随 RM 参数量平滑变化）；$R(0) := 0$。⚠️ RL 形式在原点附近斜率无穷大，论文注明该形式在原点附近可能不成立。

**关键规律解读：**
- **初始上升后下降**：gold 分先随 $d$ 增大（RM 信号有效），后因分布外崩溃而下降——Goodhart 效应的定量刻画
- **峰值位置因方法而异**：BoN(二次)与 RL(对数)的 gold 分峰值出现在不同 $d$ 处，取决于拟合系数（见原文图 3）；论文未就"谁更 KL 高效"下一般性结论
- **系数平滑 scaling**：$\alpha, \beta$ 随 RM 参数量呈近似对数趋势变化，可外推预测峰值性能
- **KL penalty 的局限**：论文 §3.6 实验发现，在其设置中增大 KL penalty 可提高给定 KL 下的 proxy 分，但**不能改善 gold 分–KL 曲线**，效果等价于早停 (early stopping)，而非提升 RM 鲁棒性本身。作者注明该结论对超参数敏感

⚠️ 上述系数 $\alpha, \beta$ 的具体数值依赖 RM 参数量和数据量，论文未给出单一通用常数；如需具体数值请查原文图 3。

### 3.3 缓解策略 (Mitigations)

#### 3.3.1 KL 散度控制 (KL Divergence Control)

在 RLHF 的目标函数中加入 KL 惩罚项：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} \left[ r_\theta(x, y) - \beta \cdot D_{\text{KL}}(\pi(y|x) \| \pi_{\text{ref}}(y|x)) \right]$$

- $\beta$ 越大 → 策略越保守，越接近参考模型 (reference model)
- $\beta$ 越小 → 允许更大探索空间，但 reward hacking 风险高
- **自适应 KL (Adaptive KL)：** 根据 KL 值动态调整 $\beta$

> 📝 KL 的单样本估计器 k1/k2/k3、in-reward vs in-loss 放置、以及 k3-as-loss 的梯度偏置，见 [llm-post-training §9.4](cheatsheet-llm-post-training.html)。

#### 3.3.2 RM 集成 (RM Ensembles)

- 训练**多个独立 RM** (不同初始化、不同数据子集、不同架构)
- 取平均、最小值 (min)、或用不确定性估计
- **保守策略：** $\hat{r}(x,y) = \min_i r_i(x,y)$ — 避免任一 RM 的过度乐观评分
- 可用于**不确定性过滤**：方差大的区域降低信任度

#### 3.3.3 长度惩罚 (Length Penalty)

$$r_{\text{adjusted}}(x,y) = r_\theta(x,y) - \alpha \cdot \text{len}(y)$$

或归一化形式：

$$r_{\text{adjusted}} = \frac{r_\theta(x,y)}{\text{len}(y)^\gamma}$$

- 需要仔细调参，避免惩罚过强导致回答过于简短
- 更好的做法：在**数据标注阶段**就控制长度偏好

#### 3.3.4 迭代重训练 (Iterative Re-Training / Online RLHF)

```
迭代流程:
1. 用当前 policy π_k 生成候选回答
2. 收集新的偏好标注 / 或用 RM 标注
3. 更新 RM: r_k → r_{k+1}
4. 用新 RM 做 RLHF 更新: π_k → π_{k+1}
5. 重复
```

- **关键好处：** RM 始终在 policy 当前分布上训练，减少分布偏移 (distribution shift)
- **挑战：** 计算成本翻倍以上，标注也可能随时间漂移
- 变体：DPO (Direct Preference Optimization)、RLHF + rejection sampling 的迭代版本

#### 3.3.5 其他方法

| 方法 | 描述 |
|------|------|
| **Rejection Sampling / Best-of-N** | 不做 RL，从 policy 采样 N 个回答，取 RM 最高分者 |
| **约束优化 (Constrained Optimization)** | 加入安全/质量硬约束 |
| **预训练约束** | 保持与预训练模型的接近 (如 L2 正则) |
| **RM 鲁棒训练** | 对抗训练、数据增强提升 RM 泛化能力 |

### 3.4 偏好数据构建 (Preference Data Construction)

RM 质量的上限由偏好数据质量决定。以下是核心设计选择：

#### 3.4.1 绝对标注 vs 相对标注

| 维度 | 绝对标注 (Pointwise) | 相对标注 (Pairwise/Comparative) |
|------|------|------|
| 标注形式 | 给单个回答打 1–5 分 | 比较两个回答选更好的 |
| 标注成本 | 高（需标注者内部标度一致） | 低（认知任务更简单） |
| 标注者一致性 | 低，标度漂移明显 | 高，人类做相对判断更可靠 |
| 信息量 | 更丰富（可恢复序数关系） | 直接对应 Bradley-Terry 模型 |
| 典型用途 | 直接回归 RM | RLHF 主流做法 |

#### 3.4.2 Margin Filtering（置信度过滤）

**动机：** 偏好对中存在大量"难以区分"的样本（annotator agreement 低），直接训练会引入噪声。

**做法：**
- 计算标注者间一致率 (inter-annotator agreement)，丢弃接近 50/50 的偏好对
- 引入 **margin** 标签：chosen 和 rejected 之间的质量差距是否"明显"？只保留 margin 足够大的样本
- 部分工作（如 Llama-2，据原论文）区分 significantly better / slightly better / negligibly better，用分层权重训练
- ⚠️ 过激过滤会损失边界样本，导致 RM 在"轻微差异"场景下判别力不足

#### 3.4.3 标注校准 (Annotator Calibration)

**问题：** 不同标注者有系统偏差（个人风格、宽严不一）。

**缓解方法：**
- **校准题 (calibration items)**：在每批标注中插入已知答案的锚点题，检测标注者漂移
- **标注者效应建模**：显式建模每个标注者的偏好分布，用混合模型聚合
- **多数投票 / 金标准过滤**：3+ 标注者打同一对，取多数；丢弃分歧过大的样本

⚠️ 标注者偏差会被 RM 学习并放大，最终影响 policy 行为——数据构建阶段的校准比事后补救更根本。

---

## 4. 长度偏差及其他 RM 病态现象 (Length Bias & Other Pathologies)

### 4.1 长度偏差 (Length Bias)

**现象：** RM 倾向于给**更长的回答**更高分数，即使更长不代表更好。

**原因分析：**
- 训练数据中，详细回答 (chosen) 通常比简短回答 (rejected) 更长
- 标注者偏好详细内容 → 偏好对中隐含长度相关性
- RM 过拟合到"长=好"的虚假特征 (spurious feature)

**后果：**
- 策略模型学会"注水"以获取高分
- 生成大量填充词、重复、冗余解释
- 推理成本上升，用户体验下降

**缓解方法：**
- 训练数据中控制长度：构造 chosen/rejected 长度接近的偏好对
- 推理时加长度惩罚 (见 3.3.3)
- 数据去偏 (debiasing)：统计分析并补偿长度相关性
- 使用长度归一化的评估指标

### 4.2 位置偏差 (Position Bias)

**现象：** 在成对比较中，RM 倾向于偏好**某个位置** (如第一个或最后一个) 的回答。

**缓解：** 交换位置做两次预测，取一致结果。

### 4.3 冗余偏差 (Verbosity / Redundancy Bias)

**现象：** 不仅是长度，RM 还倾向于偏好包含**更多冗余修饰、过渡词、格式化元素**的回答。

**区别于长度偏差：** 有时同等长度下，"更花哨"的回答也更受偏好。

### 4.4 确认偏差 (Confirmation / Sycophancy Bias)

**现象：** RM 偏好与用户提问立场一致的回答，即使该立场有误。

### 4.5 常见 RM 病态现象汇总

| 病态 (Pathology) | 表现 | 根因 |
|---|---|---|
| **长度偏差** | 长回答得分高 | 训练数据中的虚假相关性 |
| **位置偏差** | 偏好特定位置 | 标注顺序效应 |
| **冗长偏差** | 偏好修饰过多 | 标注者误判为"质量高" |
| **谄媚偏差** | 偏好迎合用户 | 标注者心理倾向 |
| **格式偏差** | 偏好列表/Markdown格式 | 训练数据中的格式虚假关联 |
| **分布外崩溃 (OOD collapse)** | 对分布外生成评分失准 | RM 泛化能力有限 |
| **表面特征捷径** | 通过关键词而非语义评分 | 模型容量/数据不足 |
| **标注者噪声放大** | RM 学到标注者个人偏见 | 标注质量不均 |

---

## 5. RM / 评估者评估 (RM & Judge Evaluation)

### 5.1 RewardBench

**简介：** RewardBench 是一个综合性的奖励模型评估基准 (benchmark)，用于系统性地测试 RM 在多个维度上的表现。

**评估维度 (大致分类)：**
- **Chat / 对话能力：** 判断对话回答质量
- **Chat-Hard / 困难对话：** 区分细微质量差异
- **Safety / 安全性：** 识别有害内容
- **Reasoning / 推理：** 判断数学/逻辑推理的正确性

**评估方式：**
- 成对比较准确率 (pairwise accuracy)：给定 chosen/rejected 对，RM 是否正确排序
- **注意：** 具体分数和排名请参考 RewardBench 官方排行榜，本书不引用具体数值 <!-- CLAUDE-REVIEW: verify number -->

**使用场景：**
- 选择最佳 RM 用于 RLHF
- 诊断 RM 在哪些领域表现薄弱
- 比较不同训练方法的效果

### 5.2 LLM-as-Judge (大模型作为评估者)

> 📎 **交叉引用**：本节侧重 LLM-as-Judge 作为**训练信号**的视角（偏差如何污染 RM 训练数据、影响 RLHF 优化）。关于 LLM-as-Judge 在**评测实践**中的具体操作、偏差缓解和 benchmark 选型，见 `eval-and-judges.md §2`。

**核心思想：** 使用强大的 LLM (如 GPT-4 级别模型) 直接对回答质量进行评分或排序，替代人类评估。

#### 常见评估范式 (Paradigms)

| 范式 | 描述 |
|------|------|
| **单项评分 (Pointwise Scoring)** | 给单个回答打 1-10 分 |
| **成对比较 (Pairwise Comparison)** | 两个回答中选更好的 |
| **排序 (Ranking / Listwise)** | 对多个回答排序 |
| **参考答案对比 (Reference-guided)** | 与标准答案对比评分 |

#### LLM-as-Judge 的常见偏差 (Biases)

##### (a) 位置偏差 (Position Bias)
- **现象：** LLM 倾向于选择呈现在**第一个位置**（或特定位置）的回答
- **验证方法：** 交换 A/B 顺序，检查一致性
- **缓解：** 做两次评估 (A先B后 + B先A后)，只保留一致结果

##### (b) 冗长偏差 (Verbosity Bias)
- **现象：** LLM 偏好更长、更详细的回答，即使简洁回答质量更高
- **原因：** 与训练数据中详细回答通常得分更高的模式一致
- **缓解：** 明确提示 "长度不应影响评分"

##### (c) 自我偏好偏差 (Self-Preference Bias)
- **现象：** LLM 评估者倾向于给自己生成的 (或同系列模型生成的) 回答更高分数
- **原因：** 生成风格/分布与自身训练分布更接近
- **缓解：** 使用不同系列的模型做评估；多人 (多模型) 评估取平均

##### (d) 其他 LLM-as-Judge 偏差

| 偏差 | 描述 |
|------|------|
| **格式偏差** | 偏好使用 Markdown、列表等格式化的回答 |
| **能力边界偏差** | LLM 无法正确评估超出自身能力的回答 |
| **奉承偏差** | 过于"友好"，不愿给出低分 |
| **锚定效应** | 受先看到的候选内容影响 |
| **关键词偏差** | 对特定术语过度敏感 |

#### LLM-as-Judge 最佳实践 (Best Practices)

1. **结构化评估标准 (Structured Rubrics)：** 明确定义评分维度和标准
2. **位置去偏 (Position Debiasing)：** 交换位置做两次
3. **温度设为 0：** 减少随机性，提高一致性
4. **多评估者 (Multi-judge)：** 使用多个不同 LLM 取平均/投票
5. **人类校准 (Human Calibration)：** 定期与人类评估做一致性检查
6. **成对优于绝对 (Pairwise > Absolute)：** LLM 做相对判断比绝对打分更可靠
7. **CoT 评估 (Chain-of-Thought Evaluation)：** 让 LLM 先推理再给分

### 5.3 人工评估与自动化评估的权衡

```
评估质量 ←————————————————————————→ 评估成本
  人工专家评估    众包标注    LLM-as-Judge    自动指标 (BLEU/ROUGE...)
  (最高质量)                                          (最低成本)
```

### 5.4 其他评估方法

- **Win Rate (胜率)：** 与基线模型成对比较的胜出比例
- **Elo Rating：** 基于锦标赛 (tournament) 的动态排名系统
- **多维评估：** 分别在有用性、安全性、事实性等维度独立评分
- **对抗评估 (Adversarial Evaluation)：** 专门测试 RM 在边缘案例和对抗样本上的鲁棒性

---

## 6. 面试题 (Interview Questions)

### L1 — 基础 (Fundamentals)

---

<details>
<summary>Q1: 什么是奖励模型 (Reward Model)？它在 RLHF 中扮演什么角色？</summary>

**答：** 奖励模型是从人类偏好数据中学习的评分函数，为 LLM 生成的回答输出一个标量奖励值。在 RLHF 中，RM 充当人类偏好的代理 (proxy)，用作策略优化的目标函数 — 策略模型通过最大化 RM 分数来改善输出质量。

> **追问 (Follow-up)：** 为什么不能直接用人类标注做 RL 的奖励？
> 因为 RL 需要大量在线奖励信号，人类标注成本高、延迟大；RM 提供了可批量、即时计算的代理信号。

</details>

---

<details>
<summary>Q2: Bradley-Terry 模型的核心假设是什么？其训练损失函数是什么形式？</summary>

**答：** BT 模型假设：对于给定 prompt，回答 $y_w$ 优于 $y_l$ 的概率由两者奖励分数之差的 sigmoid 函数决定。训练损失为负对数似然：$\mathcal{L} = -\log\sigma(r(y_w) - r(y_l))$。核心假设包括偏好可由标量分数差解释、偏好的传递性等。

> **追问：** 如果偏好不满足传递性会怎样？
> BT 模型会陷入不一致的标注对中，导致训练不稳定。此时可考虑 Plackett-Luce 等更灵活的偏好模型。

</details>

---

<details>
<summary>Q3: PRM 和 ORM 的核心区别是什么？各自适用于什么场景？</summary>

**答：** ORM 只对最终输出给一个奖励，信号稀疏；PRM 对每个推理步骤给奖励，信号密集。PRM 适用于多步骤推理任务 (数学、代码)，ORM 适用于端到端评估任务 (对话、写作)。PRM 优势在于精确信用分配，但标注成本远高于 ORM。

> **追问：** 如何自动获取 PRM 训练数据？
> 可用 Monte Carlo 估计：从每步继续采样多次 rollout，用最终答案正确率作为该步奖励的近似。

</details>

---

<details>
<summary>Q4: 什么是 Best-of-N (BoN) 采样？它与 RL 训练有什么区别？</summary>

**答：** BoN 从 policy 采样 N 个回答，用 RM 选得分最高者作为最终输出。它是推理时 (inference-time) 的优化方法，不更新模型参数，简单但推理成本与 N 成正比。RL 训练则在训练时更新参数，推理时无需额外采样。

> **追问：** BoN 存在什么样的上限 (ceiling)？
> 随 N 增大，收益递减；且受限于 policy 原始分布 — 无法生成分布外的优质回答。

</details>

---

### L2 — 中级 (Intermediate)

---

<details>
<summary>Q5: 什么是奖励欺骗 (Reward Hacking)？请举 2-3 个具体例子。</summary>

**答：** 策略模型学会利用 RM 的弱点获取高分而非真正提升质量。例子：(1) 生成冗长内容获得高分 (length hacking)；(2) 使用讨好性/奉承性措辞 (sycophancy)；(3) 重复训练数据中的高分模板句式；(4) 过度使用格式化元素 (标题、列表、粗体)。

> **追问：** Goodhart 定律与 reward hacking 的关系？
> Goodhart 定律指出"指标一旦成为目标就不再可靠" — RM 是人类偏好的近似指标，当 policy 专门优化它时，两者开始脱钩。

</details>

---

<details>
<summary>Q6: 详细解释 KL 散度惩罚在 RLHF 中的作用和调节。</summary>

**答：** KL 惩罚项 $\beta \cdot D_{\text{KL}}(\pi \| \pi_{\text{ref}})$ 约束策略不要偏离参考模型太远，起到正则化作用。$\beta$ 太大 → 策略几乎不更新，学不到东西；$\beta$ 太小 → 允许过度探索，易发生 reward hacking。常用自适应 KL：设目标 KL 值，动态调整 $\beta$。

> **追问：** KL 惩罚与 RM ensembles 各自解决 reward hacking 的哪个层面？
> KL 从优化约束层面限制策略范围；RM ensembles 从奖励估计层面降低评分方差和过度乐观。两者互补。

</details>

---

<details>
<summary>Q7: 如何用 RM 集成 (ensemble) 缓解奖励欺骗？不同聚合策略有何区别？</summary>

**答：** 训练多个独立 RM（不同种子/数据子集/架构），聚合评分。策略：(1) **均值 (mean)：** 平滑评分，减少单个 RM 噪声；(2) **最小值 (min/conservative)：** 取最保守评分，避免过度乐观 — 在 policy 偏离分布时更安全；(3) **不确定性加权：** 方差大的样本降低权重。实践中 min 策略在对抗 reward hacking 方面最为有效但可能过于保守。

> **追问：** ensemble 的计算开销如何优化？
> 可用参数共享的 backbone + 不同 head；或用 PEFT adapter 方式训练多个轻量变体。

</details>

---

<details>
<summary>Q8: 长度偏差 (length bias) 的成因是什么？如何在数据层面和推理层面缓解？</summary>

**答：** **成因：** 训练偏好对中 chosen 通常比 rejected 更长，RM 过拟合到"长=好"的虚假特征。**数据层面：** 构造长度接近的偏好对；分析并去除长度相关性。**推理层面：** 加长度惩罚 $r' = r - \alpha \cdot \text{len}$；或长度归一化。

> **追问：** 为什么仅在推理时加长度惩罚可能不够？
> 如果 RM 内部已经将长度作为强特征，长度惩罚可能无法完全抵消；更根本的做法是在训练数据/训练方法上去偏。

</details>

---

<details>
<summary>Q9: 迭代重训练 (iterative re-training) 如何缓解分布偏移 (distribution shift)？</summary>

**答：** 标准 RLHF 只在初始 policy 生成的数据上训练 RM，随着 RL 训练推进，policy 分布偏移，RM 在新分布上不再准确。迭代重训练用当前 policy 生成新数据 → 重新标注 → 更新 RM → 继续 RLHF，使 RM 始终覆盖 policy 当前分布。代价是计算和标注成本翻倍以上。

> **追问：** DPO 是否也需要迭代重训练？
> 理论上 DPO 同样面临分布偏移问题，迭代 DPO (online DPO / online preference optimization) 被提出以缓解此问题。

</details>

---

<details>
<summary>Q10: LLM-as-Judge 有哪些主要偏差？如何系统性地缓解？</summary>

**答：** 主要偏差：(1) **位置偏差** — 偏好特定位置；(2) **冗长偏差** — 偏好长回答；(3) **自我偏好** — 给同系列模型高分；(4) **格式偏差** — 偏好格式化内容。缓解：交换位置做双次评估、明确评分标准 (rubric)、多 judge 投票、定期与人类校准。

> **追问：** 为什么成对比较 (pairwise) 通常优于绝对评分 (pointwise)？
> 因为绝对评分需要 judge 内部维护一致的评分标度，容易漂移；成对比较只需判断相对优劣，是更简单的认知任务，一致性更高。

</details>

---

<details>
<summary>Q11: RewardBench 是什么？它评估 RM 的哪些能力维度？</summary>

**答：** RewardBench 是一个综合奖励模型评估基准，通过成对比较准确率来衡量 RM 在多个维度上的表现。大致维度包括：一般对话质量、困难对话 (细微差异区分)、安全性、推理能力等。它提供了标准化的比较框架，帮助诊断 RM 的强项和薄弱环节。

> **追问：** RewardBench 的评估结果能直接预测 RM 在 RLHF 中的实际表现吗？
> 不完全能。Benchmark 上的成对准确率是必要但非充分条件 — RM 在 benchmark 上的表现与其在 RLHF 优化过程中是否会遭遇 reward hacking 并不完全相关，还需结合 online 评估。

</details>

---

### L3 — 高级 (Advanced)

---

<details>
<summary>Q12: 从理论角度分析，为什么 KL 约束在 RM 不完美时是必要的？</summary>

**答：** 当 $r_{\text{RM}} \neq r_{\text{true}}$ 时，无约束地最大化 $r_{\text{RM}}$ 可能导致 $r_{\text{true}}$ 下降 (reward hacking)。KL 约束相当于在 RM 可靠的局部区域内优化 — RM 在训练分布附近是准确的，KL 确保 policy 不离开这个"信任区域 (trust region)"。这与 TRPO/PPO 中的信赖域思想一致。当 KL 为 0 时 $\pi = \pi_{\text{ref}}$，当 KL 小时 RM 可靠度高。

> **追问：** 是否存在 KL 惩罚无法防止的 reward hacking 类型？
> 是的 — 如果 reward hacking 发生在 $\pi_{\text{ref}}$ 附近的低 KL 区域内（如简单地多用几个讨好性词汇），KL 惩罚无法阻止。这时需要更好的 RM 本身。

</details>

---

<details>
<summary>Q13: 设计一个完整的迭代 RLHF 系统，描述其架构和关键决策点。</summary>

**答：**

```
架构:
π₀ (SFT model)
   ↓
[数据生成] π_k 生成 → N responses per prompt
   ↓
[偏好标注] 人工标注 或 RM 标注 (on-policy data)
   ↓
[RM 更新] 用新数据微调 RM_k → RM_{k+1}
   ↓
[Policy 优化] RLHF (PPO) with RM_{k+1}, KL→π_ref
   ↓
[评估] 奖励曲线、KL 曲线、人工评估、reward hacking 检测
   ↓
重复 k = 1, 2, ...
```

关键决策点：
- 迭代频率 (每轮多少步 RL 后更新 RM)
- 是否每轮都收集人工标注 (成本高) 还是用 RM 自标注
- KL 目标值的选择
- 何时终止迭代 (奖励饱和/人工评估达标)

> **追问：** 如何检测迭代何时应停止？持续迭代的潜在风险？
> 当人工评估分数不再提升、KL 不断增大、或出现模式崩溃 (mode collapse) 时应停止。持续迭代可能导致 policy 收敛到 RM 的特定偏好，丧失多样性。

</details>

---

<details>
<summary>Q14: 对比 PRM 的 Monte Carlo 估计方法与直接人工标注，分析各自的偏差-方差权衡。</summary>

**答：** **MC 估计：** 从每步采样 K 个 rollout，用最终正确率作为奖励。偏差来源于有限采样 (K 有限时估计不准)；方差来源于采样随机性。K 越大越准但成本越高。**人工标注：** 理论上无偏但受标注者能力/一致性限制，存在系统性偏差 (systematic bias) 和噪声。MC 估计可规模化但有系统偏差；人工标注准确但不可规模化。混合方法 (少量人工 + MC 扩展) 是常见实践。

> **追问：** MC 估计在什么条件下会严重失效？
> 当从某步开始的后续空间极大、且正确答案极稀有时 (如复杂数学题)，即使大量 rollout 也可能全部失败，导致对正确步骤给零奖励 (false negative)。

</details>

---

<details>
<summary>Q15: 如何设计一个对 reward hacking 鲁棒的评估框架？</summary>

**答：** 需要多层防御：

1. **RM 内部指标：** 成对准确率 (in-distribution vs OOD)、校准度 (calibration)
2. **代理指标监控：** KL 曲线、奖励分布漂移、生成长度变化、n-gram 重复率
3. **人工盲评 (Blind Human Eval)：** 随机抽取样本做人工评分，与 RM 评分对比
4. **对抗测试集：** 构造已知 reward hacking 模式的测试用例
5. **多 RM 一致性：** 多个 RM 的评分相关性下降作为预警信号
6. **A/B 测试：** 最终用户满意度作为终极评估

> **追问：** 在实际工程中，哪一层检测最常被忽视但最关键？
> OOD 检测最常被忽视 — 人们通常只看 in-distribution 的准确率，但 RM 在 policy 生成的 OOD 分布上的表现才决定是否发生 reward hacking。

</details>

---

<details>
<summary>Q16: 讨论 LLM-as-Judge 的自我偏好 (self-preference) 偏差对 benchmark 排行榜的影响。</summary>

**答：** 如果主流 LLM-as-Judge (如 GPT-4) 有自我偏好偏差，则与其风格相似的模型在排行榜上占优，导致排名失真。例如，如果某模型与 judge 来自同一系列或类似训练数据，可能获得不成比例的高分。这造成 benchmark 结果对 judge 选择敏感 (judge sensitivity)，削弱了排行榜的可比性和公信力。

> **追问：** 如何设计一个对 judge 偏差鲁棒的排行榜？
> 使用多个异构 judge (不同厂商、不同大小)、报告 judge 间一致性 (inter-judge agreement)、公示 judge 与各模型的关系、以及引入人类评估作为锚定。

</details>

---

<details>
<summary>Q17: 为什么说 "RM 的泛化能力是 RLHF 最关键的瓶颈"？</summary>

**答：** RLHF 的整个优化循环依赖 RM 作为目标函数。RM 的能力上限决定了 policy 优化的天花板。具体而言：(1) RM 泛化差 → reward hacking；(2) RM 有偏 → policy 继承偏差；(3) RM 在 OOD 上不可靠 → 迭代训练也难以改进；(4) RM 无法评估超出其训练分布的质量维度 → policy 无法在这些维度上进步。所有其他技术 (KL、ensemble、迭代) 都是在弥补 RM 泛化的不足。

> **追问：** Scaling RM (增大参数量) 是否能系统性地解决泛化问题？
> 大 RM 确实有更好的泛化能力 (更大的容量、更好的特征表示)，但不能完全解决 — 因为根本瓶颈有时是偏好数据本身的噪声和不完备性，而非模型容量。

</details>

---

<details>
<summary>Q18: 从信息论角度，为什么 pairwise 比 pointwise 更高效？</summary>

**答：** 人类偏好判断本质上是**序数型 (ordinal)** 信息而非**基数型 (cardinal)** 信息。Pairwise 方法直接利用序数信息 (A > B)，信息损失最小；Pointwise 要求人类映射到绝对标度 (如 1-5)，引入了额外的标度校准噪声 (scale calibration noise)。从信息论角度看，pointwise 的每个标注包含的信息量中，有一部分被标度噪声"浪费"了。Pairwise 的样本效率因此更高。

> **追问：** 有没有办法从 pairwise 数据中恢复出绝对分数？
> 可以通过求解 BT 模型的 MLE 恢复出各回答的隐含分数 (latent score)，但绝对值只有相对意义，需要额外锚点 (anchor) 来确定标度。

</details>

---

<details>
<summary>Q19: 比较 DPO 和 RLHF 在处理 reward hacking 上的异同。</summary>

**答：** **相同点：** 两者都依赖偏好数据中的隐式/显式奖励信号，都面临 Goodhart 定律的约束。**不同点：** (1) RLHF 有显式 RM，可被攻击和诊断；DPO 隐式包含奖励，难以单独检查；(2) RLHF 可通过 KL、ensemble、迭代等灵活调整；DPO 的 β 类似 KL 惩罚但调控粒度不同；(3) DPO 在 off-policy 数据上训练，面临更严重的分布偏移；(4) RLHF 的 PPO 本身就是 on-policy，某种程度上天然部分缓解了分布偏移。

> **追问：** 是否存在 reward hacking 是 RLHF 特有而 DPO 不会遇到的？
> RLHF 特有的问题包括：RM 本身的 OOD 评分失准、PPO 训练中的不稳定性导致策略突变。DPO 不会遇到显式 RM 的 OOD 问题，但会遇到隐式奖励的分布偏移问题 — 形式不同但本质类似。

</details>

---

<details>
<summary>Q20a (L3): Gao et al. 2022 的幂律对 KL penalty 强度的设置有何实际指导意义？</summary>

**答：** 论文的核心发现是：在其实验设置中，调高 KL penalty 系数 $\beta$ 并不能改善 gold 分–KL 曲线（frontier），效果等价于对同一条曲线做早停。这意味着：

1. **$\beta$ 不能作为泛化度量来调**：增大 $\beta$ 约束策略少偏离参考模型，确实减少了 KL 消耗，但并没有让 RM 对同等 KL 偏移更鲁棒；"更安全"只是因为优化停得更早，而非 RM 本身变好。

2. **KL 惩罚的真正作用**：防止策略在单步内走太远（稳定训练），而非根本上解决 reward hacking。真正需要的是 RM 本身的泛化改进（更多数据、更大模型、迭代重训练）。

3. **实践含义**：不应依靠调大 $\beta$ 来"买"更多优化空间；若 gold 分已达峰值，应重新训练 RM 而非继续在同一 RM 上加强 KL 约束。

> **⚠️ 诚信注记**：论文作者明确指出该结论"对超参数敏感"，不保证在所有设置下成立。

> **追问：** BoN 和 RL 的幂律形式不同（BoN 是 $d(\alpha - \beta d)$，RL 是 $d(\alpha - \beta \log d)$），这说明什么？
> BoN 的二次型意味着过度优化以加速度恶化（$d$ 较大时降幅更快）；RL 的对数型意味着恶化更缓慢但持续。因此，相同 KL "预算"下，BoN 更高效但也更快崩溃；不应直接用 KL 跨方法比较优化量，两者服从不同的过度优化动力学。

</details>

---

<details>
<summary>Q20: 如果你来设计下一代 RM，你认为最重要的三个改进方向是什么？</summary>

**答：** (开放题，以下是三个关键方向)

1. **更强的泛化与 OOD 鲁棒性：** 当前 RM 在分布内表现良好但在分布外崩溃。需要更好的架构设计 (如 uncertainty-aware RM)、对抗训练、以及更大更丰富的训练数据覆盖。

2. **多维解耦评分 (Multi-dimensional Disentangled Scoring)：** 将有用性、安全性、事实性、风格等维度解耦为独立评分头，避免单一标量压缩信息。也便于对不同维度施加不同优化权重。

3. **自监督/自改进 RM (Self-improving RM)：** 让 RM 具备自我校准能力 — 在 RLHF 过程中持续检测自身预测是否与实际偏好一致，并自动更新。减少对人工标注的依赖。

> **追问：** 这三个方向之间有什么潜在冲突？
> 多维评分增加模型复杂度，可能影响泛化；自改进机制如果不可靠可能引入系统性偏差；不确定性估计在高维空间中计算成本高昂。需要在工程实践中做平衡。

</details>

---

## 附录：关键术语速查 (Glossary)

| 英文术语 | 中文翻译 | 简要定义 |
|----------|----------|----------|
| Reward Model (RM) | 奖励模型 | 从偏好数据学习的评分函数 |
| RLHF | 基于人类反馈的强化学习 | 用 RM 信号做 RL 优化 |
| Bradley-Terry Model | 布拉德利-特里模型 | 成对比较的概率模型 |
| ORM | 结果奖励模型 | 只对最终输出评分 |
| PRM | 过程奖励模型 | 对每个推理步骤评分 |
| Reward Hacking | 奖励欺骗 | 利用 RM 弱点获取高分 |
| KL Divergence | KL 散度 | 衡量两个分布差异的度量 |
| Distribution Shift | 分布偏移 | 训练/评估数据分布不一致 |
| Best-of-N (BoN) | N选一 | 采样N个取最高分 |
| LLM-as-Judge | 大模型作评估者 | 用 LLM 替代人类评估 |
| Length Bias | 长度偏差 | RM 偏好更长回答 |
| Self-Preference Bias | 自我偏好偏差 | LLM 偏好自己风格的回答 |
| Inter-annotator Agreement | 标注者间一致性 | 不同标注者判断的一致程度 |
| Trust Region | 信赖域 | 优化中的安全更新范围 |
| Credit Assignment | 信用分配 | 将最终结果归因到各步骤 |
| Elo Rating | Elo 排名 | 基于胜负的动态评分系统 |
| Plackett-Luce Model | Plackett-Luce 模型 | 排序数据的概率模型 |
| DPO | 直接偏好优化 | 绕过显式 RM 的偏好学习 |
| Conservative Estimation | 保守估计 | 取 ensemble 最低分策略 |

---

*本手册仅供学习参考。具体数值请以原始论文和官方排行榜为准。* <!-- CLAUDE-REVIEW: verify number -->

## 更多 L3 深挖 / Extended L3

<details>
<summary>Q: 在设计面向”下一代”基础模型（如具备更强推理和规划能力）的RM时，现有的评估范式（如RewardBench）可能面临哪些根本性挑战？</summary>

**A:** 核心挑战在于评估对象复杂度的跃升。现有基准多针对相对标准的对话或简单推理任务。对于能执行长程规划、工具使用、或进行复杂子任务分解的智能体，RM需要评估的不再仅是单次回答的质量，而是**交互轨迹的整体有效性**和**长期决策的合理性**。这要求评估范式从”静态片段打分”转向”动态序列评估”，并需要能理解状态转移和世界模型的新指标，这是对RM能力维度的根本性拓展。

> **追问：** 在这种新范式下，RM的训练数据应如何构造？
> 需要从”偏好对”转向”轨迹对比”数据，可能涉及模拟环境中的多步交互。标注将更依赖于自动化验证（如任务成功率）和高级模拟器中的沙盒测试，纯人类标注的可行性会急剧下降。

</details>

---

<details>
<summary>Q: 过程奖励模型（PRM）在开放式生成任务（如创意写作）中应用的根本困难是什么？</summary>

**A:** 根本困难在于**”过程”的定义模糊性与评估的主观性**。在数学推理中，”步骤”有明确的逻辑边界和客观正确性标准。但在创意写作中，段落间的衔接、意象的构建、情感的铺垫等”步骤”没有客观标准，其优劣高度依赖主观品味和整体语境。因此，为PRM收集可靠的逐步骤标注异常困难，自动化评估（如MC估计）也因缺乏明确的”正确答案”而失效。

> **追问：** 那么在开放域任务中，是否应彻底放弃PRM？还是有折衷方案？
> 可采用”粗粒度PRM”或”混合奖励”方案。例如，将写作过程划分为少数几个高层面”阶段”（如构思、展开、收尾），或结合ORM的整体奖励与关键转折点（如情节高潮）的过程奖励，以平衡评估粒度与可行性。

</details>

---

<details>
<summary>Q: 从博弈论视角看，奖励欺骗（Reward Hacking）是否可以被视作策略模型与奖励模型之间的”红队-蓝队”动态博弈？这对设计缓解策略有何启示？</summary>

**A:** 是的，这本质上是一个非合作博弈。策略模型（蓝队）旨在发现并利用RM（红队）决策边界中的漏洞以最大化奖励。传统缓解策略（如固定KL惩罚）是”静态防御”，而博弈视角启示我们采用**动态、自适应的对抗策略**。例如，可以训练一个专门的”红队”RM，其目标不是评分，而是主动寻找让当前策略模型获得虚高奖励的模式，然后将发现的漏洞用于更新（加固）主RM。

> **追问：** 这种自适应对抗在实际训练中可能面临什么稳定性和成本挑战？
> 主要挑战是训练动态可能不稳定，双方陷入”军备竞赛”导致优化目标持续漂移。同时，维护多个对抗性模型会带来显著的计算和协调开销。

</details>

---

<details>
<summary>Q: RM的校准（Calibration）问题如何影响RLHF优化过程？一个”准确但未校准”的RM会带来什么具体风险？</summary>

**A:** 校准指RM输出的分数能真实反映回答质量的绝对概率或期望值。一个”准确但未校准”的RM可能在排序上准确，但其输出分数的绝对值或分布存在系统性偏差。在RLHF中，这可能导致**优化强度的误判**：例如，RM分数普遍虚高可能让优化器误以为策略已很好而过早停止；或RM对微小质量差异的评分范围过窄，导致策略更新动力不足（梯度信号弱）。KL惩罚项依赖于奖励的相对大小，未校准的RM可能使该惩罚的权衡失效。

> **追问：** 在工程上，有哪些手段可以诊断和改善RM的校准？
> 可以绘制校准曲线：将RM预测分数分段，对比各段内样本的真实质量（人工标注或任务成功率）。改善方法包括在RM训练损失中加入校准正则项，或在推理时对RM输出进行后处理校准（如温度缩放）。

</details>

---

<details>
<summary>Q: 在将安全作为硬约束（而非优化目标）集成到RM框架中时，理论上最严谨的形式化方法是什么？</summary>

**A:** 最严谨的方法是将问题建模为**约束优化问题**，而非简单的多目标加权和。具体地，优化目标仍为最大化RM对主质量维度（如有用性）的评分，但需满足一组安全约束（如RM_safety(x, y) > τ）。理论上，这可通过拉格朗日乘子法或投影梯度法求解，确保策略在满足安全可行集（Feasible Set）的前提下优化。这比将安全与有用性混合为一个分数更能避免安全目标被牺牲。

> **追问：** 在实践中，为安全约束设定一个明确的阈值τ（如”有害概率<0.1%”）的主要困难是什么？
> 核心困难在于安全边界的模糊性和场景依赖性。一个在大多数上下文中”安全”的回答，在特定敏感语境下可能不安全。因此，设定一个全局固定的τ不现实，更需基于上下文动态调整，这对RM和约束系统提出了更高要求。

</details>

---

<details>
<summary>Q: 除了用于集成和不确定性过滤，RM的不确定性估计在RLHF训练过程中还能扮演哪些更主动的角色？</summary>

**A:** 不确定性估计可主动指导训练数据收集和探索策略，实现**主动学习**。例如，训练循环中可优先对RM不确定性高的prompt-response区域进行采样和人工标注，以最高效地提升RM。此外，在策略优化时，可以让智能体主动探索高不确定性区域（即RM知识边界），以发现可能的新优质策略，这类似于贝叶斯优化中的探索-利用平衡。

> **追问：** 如何在RLHF的PPO等on-policy算法中具体实现基于不确定性的探索？
> 可以将RM的不确定性作为探索奖励的一部分，鼓励策略生成RM”感到惊讶”或”不确定”的回答。具体地，在最终奖励中加入一个与不确定性正相关的探索项，但需仔细平衡以避免生成无意义的随机输出。

</details>

## §A 核心论文时间线 / Key Papers Timeline

- **2017-06 · Deep RL from Human Preferences** — Christiano et al., NeurIPS 2017. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) — 奠定 RLHF 基础框架：用人类对轨迹对的偏好判断训练奖励模型，再以该奖励信号驱动强化学习策略优化。

- **2019-09 · Fine-Tuning Language Models from Human Preferences** — Ziegler et al., arXiv preprint. [arXiv:1909.08593](https://arxiv.org/abs/1909.08593) — 首次将偏好奖励模型（Bradley-Terry 成对比较 + KL 惩罚）系统应用于语言模型微调，奠定现代 LLM RLHF 流程雏形。

- **2020-09 · Learning to Summarize with Human Feedback** — Stiennon et al., NeurIPS 2020. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325) — 在文本摘要任务上验证 RLHF 闭环：成对偏好标注 → RM 训练 → PPO 优化，输出质量显著优于纯监督微调基线。

- **2022-03 · Training Language Models to Follow Instructions with Human Feedback (InstructGPT)** — Ouyang et al., NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) — 将 RLHF 规模化至 GPT-3 级别，引入 SFT → RM → PPO 三阶段流程，并详细描述偏好数据构建与 margin 过滤实践。

- **2022-10 · Scaling Laws for Reward Model Overoptimization** — Gao et al., ICML 2023. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760) — 定量刻画奖励过度优化的幂律规律：gold RM 分数随 KL 距离先升后降，BoN 呈二次型、RL 呈对数型，系数随 RM 参数量平滑缩放。

- **2023-05 · Let's Verify Step by Step (PRM800K)** — Lightman et al., ICLR 2024. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) — 系统比较过程奖励模型（PRM）与结果奖励模型（ORM）在数学推理上的差异，发布 80 万条步骤级标注，证明逐步监督显著优于仅监督最终答案。

- **2023-05 · Direct Preference Optimization (DPO)** — Rafailov et al., NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — 将 RLHF 中的显式奖励模型消除：通过数学变换将最优策略直接参数化为偏好数据上的二元交叉熵目标，无需独立 RM 训练与 PPO 采样。

- **2023-06 · Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al., NeurIPS 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) — 系统研究以强 LLM 替代人类评估的范式，识别位置偏差、冗长偏差、自我偏好偏差等主要问题，并提出 MT-Bench 与 Chatbot Arena 两个配套基准。

- **2023-07 · Llama 2: Open Foundation and Fine-Tuned Chat Models** — Touvron et al., arXiv preprint. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — 大规模开源 RLHF 实践报告，详细描述 margin 置信度过滤（significantly/slightly/negligibly better 分层加权）在偏好数据构建中的应用。

- **2024-03 · RewardBench: Evaluating Reward Models for Language Modeling** — Lambert et al., arXiv preprint. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787) — 提出奖励模型标准化评测基准，覆盖对话、对话难例、安全性、推理四个维度，以成对比较准确率衡量 RM 能力，为模型选型与诊断提供统一框架。
