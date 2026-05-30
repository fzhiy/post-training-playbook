# 评测与 LLM-as-judge / Evaluation & Judges

> 后训练里「**评测**」往往是真正的瓶颈:训练能跑、但"到底变好没有 / 哪里变差"全靠评测说话。本页讲怎么评一个对齐后的模型 + 各种评测的坑。
> ⚠️ 不放具体分数(易过时/易记错);具体数字以各 benchmark/leaderboard 官方为准。

## 0. TL;DR

- 后训练里评测是瓶颈:**能力 benchmark**(有标答、自动)、**偏好/对话评测**(judge 打分)、**RM 评测**三条线各测一面。
- **LLM-as-judge** 省钱但有系统偏置(位置 / 冗长 / 自我偏好 / 格式)——**交换顺序各评一次**是最便宜有效的去偏。
- **数据污染**让分数虚高:用 n-gram 重叠 / Min-k% / canary / 改写掉分来查;用动态私有题 + 时间隔离来防。
- 报告纪律:固定 prompt、多 seed 报方差、查回归(alignment tax)、记住 judge ≠ 真值。

## 1. 三类评测 / Three families

| 类型 | 衡量什么 | 代表 |
|---|---|---|
| **能力 benchmark**(自动、有标准答案) | 知识/推理/代码 | MMLU、GSM8K、MATH、HumanEval/MBPP、BBH、IFEval(指令遵循) |
| **偏好/对话评测**(judge 打分) | "回答好不好"的主观质量 | AlpacaEval(LLM-judge win-rate)、MT-Bench(多轮、judge 评分)、Chatbot Arena(真人成对 → Elo) |
| **奖励模型评测** | RM 是否符合人类偏好 | RewardBench(chat/safety/reasoning 等类)、与人工标注的一致率 |

### 1.1 Benchmark 详细对比 / Benchmark cheat-sheet

> ⚠️ 下表只列**机制与结构**(信号来源、打分方式、坑),不列具体分数。

**偏好 / 对话评测**(judge 或真人):

| 评测 | 信号来源 | 打分机制 | 最突出的偏置 | 去偏 / 控制手段 |
|---|---|---|---|---|
| **AlpacaEval 2.0** | 单一 LLM-judge(model vs 参考答案) | win-rate(对参考的胜率) | 冗长偏置 | length-controlled win-rate(回归扣除长度) |
| **MT-Bench** | LLM-judge | 多轮、1–10 标量评分 或 pairwise | 位置 / 冗长 / 自我偏好 | pairwise 两序交换取平均 |
| **Chatbot Arena** | 真人盲投成对对战 | Bradley-Terry / Elo → 排名 + 置信区间 | 用户分布 / 风格偏好、众包噪声 | 海量投票 + 动态新题抗污染 |

**能力 benchmark**(有标答、自动):

| Benchmark | 测什么 | 题型 | 评分方式 | 已知坑 |
|---|---|---|---|---|
| **MMLU** | 57 学科多选知识 | 4 选 1 选择题 | 选项准确率 | 选项 / 字母顺序偏置;污染严重 |
| **MATH** | 竞赛数学(7 学科 / 5 难度级) | 自由形式解答 | 最终答案匹配(可验证) | 答案解析脆弱;部分被污染 |
| **HumanEval** | Python 代码生成 | 函数补全 + 单测 | pass@k(单测通过率) | 仅 164 题、方差大、易过拟合 |
| **IFEval** | 可验证指令遵循 | 带约束指令(字数 / 格式) | 程序化验证(无需 judge) | 只覆盖可机检约束 |

## 2. LLM-as-judge:怎么用 + 偏置

> 📎 **交叉引用**：本节侧重 LLM-as-Judge 作为**评测实践**的视角（如何选 judge、具体操作、benchmark 应用）。关于 LLM-as-Judge 作为 **RLHF 训练信号**时偏差如何影响 RM 训练和 reward hacking，见 `reward-modeling-eval.md §5.2`。

用一个强模型当裁判给回答打分/two-way 比较。**省钱省时,但有系统偏置**:
- **位置偏置 (position bias)**:倾向选"第一个"答案 → 缓解:**交换顺序各评一次**取平均。
- **冗长偏置 (verbosity / length bias)**:倾向更长的答案 → 缓解:长度去偏、控制长度。
- **自我偏好 (self-preference)**:裁判偏好和自己风格像的输出。
- **格式/风格偏置**:markdown、自信语气更讨喜。
缓解通用招:**reference-guided**(给参考答案)、**rubric/打分量表**、**多裁判投票**、与**人类标注校准**。

**位置去偏的形式化**:记 judge 对有序对的偏好 $J(a,b)\in\{A,B\}$。无位置偏置时 $J(a,b)$ 与 $J(b,a)$ 应选**同一**答案(一致 consistent)。去偏判定:仅当两个顺序都选同一答案才判其胜,否则判 **tie**——把位置偏置显式记成平局,而非让它泄漏进 win-rate。注意:AlpacaEval 默认用**单一顺序**(model 排第一);MT-Bench pairwise 模式用**交换增强**(两序取平均)——并非所有工具都做顺序平均。

### 2.1 位置去偏 judge harness(代码）

```python
# LLM-as-judge：成对比较 + 位置去偏（order-swap）harness。
# 真实里 judge() 调一个强模型并解析其裁决；这里用桩函数演示协议与去偏逻辑。

def judge_debiased(question, ans1, ans2, judge):
    """两个顺序各评一次，消除位置偏置。
    judge(q, A, B) 返回 'A' 或 'B'（它认为哪个位置的答案更好）。
    返回 'ans1' / 'ans2' / 'tie'（两序不一致 → 判平）。"""
    v1 = judge(question, ans1, ans2)            # 顺序 (A=ans1, B=ans2)
    v2 = judge(question, ans2, ans1)            # 交换 (A=ans2, B=ans1)
    pick1 = 'ans1' if v1 == 'A' else 'ans2'     # 第一次真正选中的答案
    pick2 = 'ans2' if v2 == 'A' else 'ans1'     # 交换后 A 位是 ans2，故 v2=='A' 表示选了 ans2
    return pick1 if pick1 == pick2 else 'tie'   # 一致才算数，否则判平

def win_rate(questions, model_answers, ref_answers, judge):
    """model 相对 ref 的去偏 win-rate；tie 记 0.5。"""
    s = 0.0
    for q, m, r in zip(questions, model_answers, ref_answers):
        out = judge_debiased(q, m, r, judge)    # ans1=model, ans2=ref
        s += 1.0 if out == 'ans1' else (0.5 if out == 'tie' else 0.0)
    return s / len(questions)

# --- 演示：一个“永远选第一个”的极端位置偏置 judge，被去偏识破为 tie ---
def biased_judge(q, a, b):
    return 'A'                                  # 极端位置偏置
print(judge_debiased("q", "model", "ref", biased_judge))   # -> 'tie'
print("win_rate:", win_rate(["q"], ["model"], ["ref"], biased_judge))  # -> 0.5
```

## 3. 数据污染 / Contamination

训练集混入了评测集 → 分数虚高、不反映泛化。

### 3.1 污染类型

| 类型 | 描述 |
|------|------|
| **直接重叠** | 训练数据含有评测集的原文题目或答案 |
| **时间污染 (temporal leakage)** | 训练数据截止日期晚于评测集创建时间，模型"见过"测试期内容 |
| **近似重叠** | 改写/翻译版本的测试题出现在训练集中，字符串匹配检测不到 |
| **成员推断污染** | 训练集不含原题，但含有高度相关的分布内样本，导致分数虚高 |

### 3.2 检测方法

- **n-gram / 子串重叠**：计算训练集与评测集之间的 n-gram Jaccard 相似度，设阈值过滤
- **Min-k% Prob (MIA)**：成员推断攻击 (Membership Inference Attack)——计算测试样本在模型下的最低 k% token 概率；成员样本的最低 token 概率通常高于非成员（据 Shi et al. 2024 等工作，具体效果随设置而异）
- **Canary 字符串**：在训练数据中插入随机字符串，若模型能"记住"则说明该数据确实被训练
- **异常低 perplexity**：对测试样本计算 perplexity；若显著低于同类新样本，提示可能见过
- **改写后掉分**：对测试题做语义等价改写，若模型在原题上分数高而改写版上分数低，提示记忆而非泛化

### 3.3 n-gram 去重 (Deduplication)

- 训练数据内部去重（exact + near-dedup）可降低记忆风险，也减少 benchmark 污染传播
- 常用工具：MinHash LSH、suffix array 去重（据 Lee et al. 2022 等工作）
- ⚠️ 去重不等于去污染：去重针对训练集内重复，去污染 (decontamination) 专门针对训练集与评测集的交叉

### 3.4 抗污染 benchmark 设计

- **动态/私有评测集**：每轮评测使用新题（如 Chatbot Arena 的持续收集、LiveBench 的月度更新）
- **程序化生成题**：自动生成可验证答案的题（数学、代码），题目新颖性由生成程序保证
- **时间隔离**：明确报告训练数据截止日期 (data cutoff) 与评测集创建日期，供读者判断时间污染风险
- **污染审计报告**：在论文/报告中公开 decontamination 流程和过滤比例，增加可信度

## 4. 偏好评测的陷阱

- **Goodhart**:一旦某 benchmark 成了优化目标,它就不再是好的衡量(刷榜 ≠ 真变强)。
- **prompt 敏感**:同一模型换 prompt 模板分数能差很多 → 固定评测 prompt、报告方差。
- **judge ≠ 真值**:LLM-judge 的 win-rate 只是"另一个模型觉得好",不等于人类偏好或真实效用。
- **单点准确率 vs 算力预算**:推理模型要看"给定 test-time 算力下的准确率",而非单次。

## 5. 一套实用评测口径(post-training)

1. **能力**:GSM8K/MATH(数学)、HumanEval(代码)、MMLU(知识)、IFEval(指令遵循)。
2. **对齐质量**:AlpacaEval / MT-Bench(judge,带位置去偏)、必要时 Arena。
3. **安全/拒答**:有害提示拒答率、过度拒答(over-refusal)率。
4. **回归**:对比基线,确认没在某些维度变差(alignment tax)。
5. **污染审计** + **多 seed/prompt** 报方差。

---

## 分层面试题 / Stratified follow-ups

### L1 基础
1. 为什么说评测是 post-training 的瓶颈?能力 benchmark 和偏好评测各测什么?
2. LLM-as-judge 是什么?它有哪些已知偏置?
3. 什么是数据污染?为什么它让分数不可信?
4. 能力 benchmark 和偏好评测有什么本质区别?各自适合测哪类东西?
5. AlpacaEval 的 win-rate 是怎么算出来的?它是对着什么参考算胜率?
6. MT-Bench 和 Chatbot Arena 的评测信号有什么不同(谁来打分)?
7. pass@k 是什么意思?为什么代码评测常用它而非单次准确率?
8. 什么是 alignment tax?为什么后训练后要专门查回归?
9. 为什么评测要固定 prompt 模板、多 seed 报方差?
10. IFEval 和 MMLU 在"怎么打分"上有何根本不同(程序化验证 vs 选项准确率)?

### L2 进阶
11. 位置偏置怎么缓解?为什么"交换顺序各评一次"有效?
12. AlpacaEval / MT-Bench / Arena 三者的评测信号有何不同(自动 judge vs 真人 Elo)?
13. 怎么检测训练数据是否污染了某个评测集?
14. 冗长偏置为什么难缠?length-controlled win-rate 是怎么把长度影响扣除的?
15. order-swap 去偏在两序不一致时判 tie,相比"多数投票"在鲁棒性上有何优势?
16. Chatbot Arena 用 Bradley-Terry / Elo 把成对胜负转成排名,这个模型的核心假设是什么?
17. 列举几种数据污染检测手段(n-gram / Min-k% / canary / 改写掉分),各自抓的是什么?
18. 为什么说"去重 (dedup) ≠ 去污染 (decontamination)"?
19. 自我偏好偏置在"用同源模型当 judge"时为什么特别危险?
20. 为什么 judge win-rate 高不等于真人更偏好?二者会在哪里系统性地分叉?

### L3 深挖
21. Goodhart 定律在刷榜上怎么体现?如何设计"抗刷"的评测?
22. reward model 怎么评(RewardBench 的思路)?RM 评测和最终 policy 效果的关系?
23. 对推理模型,为什么评测要从"单次准确率"转向"算力预算下的准确率"?这对评测协议提出什么要求?
24. 如果线上指标(用户留存)和离线评测(judge win-rate)打架,你信哪个、怎么排查?
25. 用单一标量分数概括模型时,怎样会掩盖"能力↑但安全↓"这类帕累托式回归?评测设计上如何把它暴露出来?


## 更多 L3 深挖 / Extended L3

<details>
<summary>Q26. 当用 LLM-as-judge 评估多轮、长上下文对话时，常见的困难是什么？有哪些评估方法上的改进？</summary>

    评估多轮对话的核心挑战是评判模型容易出现**上下文遗忘 (context forgetting)** 或**局部偏好 (local bias)**，即只关注最近一两轮的回答质量而忽略整体对话连贯性和任务完成度。一个关键改进是设计**过程导向的评分量表 (process-oriented rubric)**，明确要求评估每一轮对最终目标的贡献，并引入**分段总结 (segment summarization)** 机制，迫使judge模型先归纳再打分，从而在一定程度上缓解其短视问题。
    **追问**: 除了改进评分量表，是否可以通过改变评测协议来降低评判难度？例如，将其拆解为一系列更简单的子任务评估？

</details>

<details>
<summary>Q27. LLM-as-judge 在评估事实准确性 (factual accuracy) 和逻辑严密性 (logical soundness) 方面的主要局限是什么？如何缓解？</summary>

    其主要局限是评判模型自身的**知识边界 (knowledge boundary)** 和**推理缺陷 (reasoning flaw)** 可能导致误判。它可能无法识别事实错误，或者错误地认为一个有逻辑跳跃的答案是严密的。缓解方法通常采用**混合评估模式 (hybrid evaluation)**：对于事实核查，结合**检索增强验证 (retrieval-augmented verification)**，即先检索权威信息再进行比对；对于逻辑评估，则尝试使用**形式化验证 (formal verification)** 工具或设计专门针对推理链的**逐步验证提示 (step-by-step verification prompts)**。
    **追问**: 在资源有限的情况下，我们应该优先改进judge模型的哪些能力（知识广度、推理能力、工具使用）来最有效地提升其评估准确性？

</details>

<details>
<summary>Q28. 如何评估一个模型的”涌现能力” (emergent capabilities)？这和评估传统能力有何根本不同？</summary>

    评估涌现能力最大的不同在于其**不可预测性 (unpredictability)** 和**非平滑性 (non-smoothness)**。传统能力通常在某个评测集上随着模型规模或训练数据量的增加呈现可预测的提升。而涌现能力表现为在某个阈值点后突然出现，且往往在多个标准 benchmark 上没有直接体现。因此，评估方法必须从**固定测试集 (fixed test sets)** 转向**开放式、程序化生成的探针任务 (open-ended, programmatically generated probe tasks)**，并关注模型在面对全新、复杂任务组合时的**行为模式突变 (behavioral pattern shift)**。
    **追问**: 能否设计一种评测框架，使其不仅能发现涌现能力，还能在一定程度上预测这种能力的出现条件？

</details>

<details>
<summary>Q29. 对 LLM-as-judge 的输出置信度 (confidence) 进行校准有什么意义？实践中如何实现？</summary>

    对 judge 输出的置信度进行校准，是为了让其打分或比较结果具有**可解释的概率意义 (interpretable probabilistic meaning)**。例如，当judge表示”90%确定A比B好”时，这个数字在长期统计上应该接近真实的A优于B的频率。实践中，实现校准需要一个**带有人类标注的校准集 (human-annotated calibration set)**。通过让judge在该集合上反复评估，可以分析其评分与人类共识的偏差分布，进而通过**后处理校准算法 (post-hoc calibration algorithms)**（如Platt Scaling或Isotonic Regression）来调整其原始分数，使其更贴近人类判断的统计规律。
    **追问**: 如果用于校准的人类标注数据本身质量不高或规模很小，会对校准后的judge产生什么影响？有什么替代方案？

</details>

<details>
<summary>Q30. 在评估对话模型的安全性 (safety) 时，为什么”过度拒答” (over-refusal) 是一个重要指标？它和”有害提示拒答率”构成的权衡关系如何分析？</summary>

    过度拒答衡量模型对**良性或边缘性查询 (benign or borderline queries)** 错误拒绝回答的程度，它直接关系到用户体验和模型的**可用性 (utility)**。一个过度拒答率很高的模型虽然安全，但会变得”无用”。分析这一权衡时，不能简单追求两个指标的帕累托最优，而应引入**风险等级分类 (risk tiering)**。将有害提示按严重程度分类，并为每一类设定不同的拒答严格度。评测时，需要分别报告每一类别的拒答率，并通过**代价敏感分析 (cost-sensitive analysis)** 来评估模型在整体风险暴露和用户体验损失之间的平衡。
    **追问**: 如何构建一个能够自动生成覆盖各种风险等级和”灰色地带”提示的、高质量的对抗性安全评测集？

</details>

<details>
<summary>Q31. 如何进行”评估之评估” (meta-evaluation)，即如何判断一个评测基准或一个LLM-as-judge本身是否有效、可靠？</summary>

    对评测基准的元评估主要看其**区分度 (discriminability)**、**鲁棒性 (robustness)** 和**生态效度 (ecological validity)**。区分度指能否有效区分不同能力水平的模型；鲁棒性指对prompt微小改动是否敏感；生态效度指其评估的能力是否与现实世界需求相关。对LLM-as-judge的元评估，则重点考察其与人类判断的**一致性 (agreement)**（如Cohen's Kappa系数）和在不同子群体上的**公平性 (fairness)**。一个关键方法是**交叉验证 (cross-validation)**：用多个不同的、高质量的judge（或人类）去评估同一组数据，看目标judge或基准是否与共识一致。
    **追问**: 在发现某个广泛使用的benchmark可能存在严重偏差或过时后，作为研究者，我们有哪些责任和可行的操作来推动其迭代或警示社区？

</details>

<details>
<summary>Q32. 在领域特定（如医疗、法律）场景下，通用LLM-as-judge的评估会遇到哪些特有挑战？构建领域专家评估流水线时，关键步骤是什么？</summary>

    核心挑战是**领域知识壁垒 (domain knowledge barrier)** 和**评估标准的专业化 (specialized evaluation criteria)**。通用judge可能无法理解领域术语的细微差别或专业逻辑的严谨性。构建领域评估流水线的关键步骤首先是**联合定义评估维度 (co-define evaluation dimensions)**，与领域专家共同确定如”医疗建议的保守性”、”法律引用的准确性”等维度。其次是**构建领域金标准 (domain gold-standard)**，即由专家标注的、具有权威性的参考答案或评判结果集。最后是设计**人机协同评估流程 (human-AI collaborative evaluation)**，让AI judge处理初筛，人类专家负责复核边缘案例和最终裁决。
    **追问**: 当领域专家对同一回答的评价也存在分歧时（例如，不同流派的医生），如何设计一个能包容合理专家分歧、又能进行有效自动化评估的系统？

</details>

## §A 核心论文时间线 / Key Papers Timeline

- **2020-09 · MMLU** — Hendrycks et al., ICLR 2021. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) — 57 学科、四选一的多任务知识评测,确立"广覆盖选择题"范式;选项与字母顺序偏置 + 后期被严重污染是其主要软肋。

- **2021-03 · MATH** — Hendrycks et al., NeurIPS 2021. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) — 竞赛数学(7 学科 / 5 难度级)自由形式解答 + 最终答案可验证匹配,成为数学推理评测的标准件。

- **2021-07 · HumanEval** — Chen et al., arXiv 预印本. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) — 164 道函数补全题 + 隐藏单测,提出 pass@k 作为代码生成评测口径;题量小、方差大。

- **2021-10 · GSM8K** — Cobbe et al., arXiv 预印本. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) — 8.5K 小学数学应用题 + 训练 verifier 重排,奠定链式推理 + 答案验证的评测/训练范式。

- **2023-05 · AlpacaFarm** — Dubois et al., NeurIPS 2023. [arXiv:2305.14387](https://arxiv.org/abs/2305.14387) — 用 LLM-judge 低成本模拟 RLHF 人工偏好标注,使 win-rate 式对齐评测可复现、可迭代。

- **2023-06 · MT-Bench / LLM-as-a-Judge** — Zheng et al., NeurIPS 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) — 多轮对话 judge 评分 + 系统性量化位置/冗长/自我偏好偏置,确立 LLM-as-judge 的方法学基线。

- **2023-10 · Min-K% Prob** — Shi et al., ICLR 2024. [arXiv:2310.16789](https://arxiv.org/abs/2310.16789) — 以样本最低 k% token 概率做预训练数据成员推断,用于检测评测集是否被"见过"(污染探针)。

- **2023-11 · IFEval** — Zhou et al., arXiv 预印本. [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) — 用"可程序化验证"的指令(字数/格式等)做指令遵循评测,无需 judge、客观可复算。

- **2024-03 · Chatbot Arena** — Chiang et al., ICML 2024. [arXiv:2403.04132](https://arxiv.org/abs/2403.04132) — 真人盲投成对对战 + Bradley-Terry/Elo 转排名,海量投票 + 动态新题成为抗污染的真人偏好金标准。

- **2024-03 · RewardBench** — Lambert et al., arXiv 预印本. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787) — 首个系统化 RM 评测基准(chat/safety/reasoning 等类),用配对偏好正确率衡量奖励模型质量。

- **2024-04 · Length-Controlled AlpacaEval** — Dubois et al., COLM 2024. [arXiv:2404.04475](https://arxiv.org/abs/2404.04475) — 用回归把"长度"从 win-rate 中扣除,显著降低冗长偏置、提升与人类排名的相关性。

- **2024-06 · LiveBench** — White et al., ICLR 2025. [arXiv:2406.19314](https://arxiv.org/abs/2406.19314) — 月度滚动更新 + 客观可验证答案的"抗污染"评测,降低数据泄漏导致的虚高。
