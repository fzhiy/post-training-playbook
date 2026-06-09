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
| **GPQA**（Diamond） | 研究生级物理/生物/化学，设计目标：**熟练非专家即使无限制上网也无法 100% 答对**（"Google-proof"） | 4 选 1 选择题 | 选项准确率 | 领域专家准确率约 65%（Diamond 子集更高，具体随学科变化）；选项需 expert-level 知识才能区分，区分度极高 |
| **MMLU-Pro** | MMLU 加强版——排除简单/污染题、10 选 1 加大难度 | 10 选 1 选择题 | 选项准确率 | 随机基线更低(10% vs 25%)；显著缓解 MMLU 的天花板效应 |

**偏好/对话评测的 2024 演进**：

| **Arena-Hard** | 从 Chatbot Arena 海量数据中自动筛选 **500 道"难倒模型"的题**，用 GPT-4 judge 做 pairwise | win-rate(对参考的胜率) | judge 偏置（全盘继承） | 与 Arena Elo 排序 Spearman 约 94.1%，快速近似真人偏好；专挑难题放大区分度 |
| **MixEval** | 将**真实 web 用户查询**匹配到现成 benchmark 题目，用 **ground-truth 答案**（非 judge）评测，动态加权聚合 | 归一化多维度平均分 | 采样权重敏感、子集旧版残留污染 | 多轮权重校准+去污染过滤，力图综合反映能力+偏好；本质是 ground-truth 基准混合物而非 judge benchmark |

### 1.1a pass@k 无偏估计 / Unbiased pass@k

**朴素估计** $1-(1-\hat p)^k$($\hat p=c/n$):对 $k\ge2$ 在任意有限 $n$ 都**系统性低估**真实 pass@k(Jensen 不等式:$(1-\hat p)^k$ 对 $\hat p$ 凸,故 $\mathbb{E}[(1-\hat p)^k]\ge(1-p)^k$),$n$ 越大偏差越小但不为 0;$k=1$ 时 $c/n$ 已无偏。

**无偏估计**(Chen et al., [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) §3):每题采 $n\ge k$ 个候选、$c$ 个过单测,则

$$\widehat{\text{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}$$

直觉:$\binom{n-c}{k}/\binom{n}{k}$ 是随机取 $k$ 个候选**全是错的**的概率,$1$ 减去即「至少一个对」。数值稳定实现(避免大组合数溢出):

```python
import numpy as np
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

> 📝 标准口径(Chen et al.):每题 $n=200$,报告 $k\in\{1,10,100\}$。

### 1.2 Bradley-Terry / Elo:成对胜负 → 排名 / Pairwise wins → ranking

Arena 把海量「A vs B 谁更好」的成对投票转成全序排名,靠的就是 **Bradley-Terry (BT)** 模型(Elo 是它的在线近似)。

**BT 模型**:给每个对象 $i$ 一个隐分数 $s_i$,则

$$P(i \succ j) = \sigma(s_i - s_j) = \frac{1}{1+e^{-(s_i-s_j)}}$$

只有**分差**可辨识(整体平移不改变概率),故需固定一个锚点(如 $s_0=0$)。拟合 = 对成对结果做**逻辑回归 MLE**:对数似然 $\sum_{i,j} w_{ij}\log\sigma(s_i-s_j)$($w_{ij}$ 为 $i$ 胜 $j$ 的次数)是**凹**的(等价地负对数似然是凸损失),凸优化、有全局最优;胜负图连通时解有限且唯一。

**核心假设**(也是其脆弱处):
1. **单维强度**:质量可用一个标量 $s_i$ 概括 → 对象可被全序排列;
2. **(随机)传递性**:若 A 倾向胜 B、B 倾向胜 C,则 A 倾向胜 C——**不允许石头剪刀布式的循环偏好**($A\succ B\succ C\succ A$);
3. **比较独立**:各场对战相互独立,无顺序/学习效应;
4. 基础 BT 不建模平局(Rao-Kupper 等扩展才处理 tie)。

→ 当模型间偏好真的呈**非传递的循环**时(不同维度上各有胜负),单一标量 Elo 会把这种循环压扁成一个序,排名的"客观性"被高估。

**Elo = BT 的在线近似**:每场对战按预测误差做一次定步长更新,可视作在 BT 逻辑损失上做在线梯度更新:

```python
# Elo:Bradley-Terry 的在线/流式版,每场对战做一次定步长的梯度更新
def elo_update(r_a, r_b, score_a, K=32, scale=400):
    e_a = 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))  # BT/logistic 预测的 P(A 胜)
    r_a = r_a + K * (score_a - e_a)                  # score_a ∈ {1, 0.5, 0}
    r_b = r_b + K * ((1 - score_a) - (1 - e_a))
    return r_a, r_b
```

> 📝 批量场景直接对全部成对结果做 BT 的 MLE(逻辑回归)比逐场 Elo 更稳,且能给**置信区间**(Arena 用 BT + bootstrap 报区间)。

### 1.3 SWE-bench:从函数补全到真实软件工程 / SWE-bench

> 📎 **交叉引用**：SWE-bench 是代码评测从"补全函数"到"修真实 bug"的代表性演进。关于 pass@k 的无偏估计见 §1.1a；测试时采样(采样→挑选)的收益上限分析见 [test-time-scaling §3.3](cheatsheet-test-time-scaling.html)。SWE-bench 轨迹也常被用作 agent SFT/RL 的训练数据（见 [long-horizon-agents](cheatsheet-long-horizon-agents.html)）。

**SWE-bench**（Jimenez et al., [arXiv:2310.06770](https://arxiv.org/abs/2310.06770), ICLR 2024）把代码评测从"补全一个函数"升级为**"修复一个真实 GitHub issue"**，是 2024 年代码评测领域最具影响力的新 benchmark 之一。

- **任务**：给模型一个**真实的 GitHub issue 描述** + 对应代码库 **snapshot** → 模型生成 **patch**（`git diff`，跨多文件/多行修改）
- **评测**：在 Docker 容器里 apply patch → 运行 instance 关联的 **FAIL_TO_PASS** 和 **PASS_TO_PASS** 测试 → 全部通过才算 resolved
- **规模**：~2,294 个 issue，来自 12 个流行 Python 仓库（Django, Flask, SymPy, scikit-learn, matplotlib 等）
- **难度**：需理解大型代码库结构、定位 bug 所在文件/函数、写跨文件修改——远超单函数补全（对比 HumanEval 的 164 道独立函数题）

**关键子集**：

| 子集 | 描述 | 实例数 |
|------|------|--------|
| **SWE-bench Lite** | 从原始 SWE-bench 精选的"易入手"子集，用于快速迭代 | ~300 |
| **SWE-bench Verified**（OpenAI 2024-08, [blog](https://openai.com/index/introducing-swe-bench-verified/)） | 人工复核 500 道题：确认 issue 描述质量 + 测试可靠 + 排除不可复现项，修复后的子集 | 500 |

**关键结果轨迹**（% resolved，仅说明相对进展不构成横向比较）：
- 2024-03（原始 SWE-bench）：Devin（Cognition AI）~13.86%（首次突破 10%）
- 2024-08（原始 SWE-bench）：SWE-agent + GPT-4 组合 ~18–20%
- 2024-10（Verified 子集）：Claude 3.5 Sonnet ~49%（当时 SOTA）
- 2025 初（Verified 子集）：多 agent 系统 >50%

> ⚠️ SWE-bench 的核心约束是**评测成本**——每个 instance 需构建完整 Docker 容器 + 跑 repo 测试套件，一次评测可能耗 5–30 min/instance。

**代码评测的三层演进**（HumanEval → BigCodeBench → SWE-bench）：

| 层级 | Benchmark | 文件数 | 评测方式 | 对应能力 |
|------|------|------|------|------|
| 单函数补全 | HumanEval, MBPP | 1 文件 | 单测 pass@k | 基本代码生成 |
| 复杂函数+库调用 | **BigCodeBench**（[arXiv:2406.15877](https://arxiv.org/abs/2406.15877)） | 函数级 + 多样库调用 | 单测 pass@k | 复杂 API 使用 |
| 真实软件工程 | **SWE-bench** | 全 repo（12 个真实仓库） | apply patch → 跑 repo 原测试套件 | 代码理解+定位+编辑 |

**抗污染的补充**——**LiveCodeBench**（Jain et al., [arXiv:2403.07974](https://arxiv.org/abs/2403.07974), ICLR 2025）：从 LeetCode/AtCoder/Codeforces **新题**（时间隔离）自动构建代码评测，涵盖代码生成、执行预测、测试生成等多种任务，按月滚动更新，解决 HumanEval 题量小 + 易过拟合的问题。

### 1.4 Agent 评测 / Agent Evaluation

> 📎 **交叉引用**：本页聚焦 agent **评测**（怎么看 agent 做得好不好）。agent 的训练方法（自演化 RL、工具使用 SFT、computer use 配方）见 [long-horizon-agents](cheatsheet-long-horizon-agents.html)。

后训练正在从"单轮对话"向"多步 agent"演进，评测也必须跟。Agent 评测比传统 benchmark 复杂——需要**模拟真实交互环境**（浏览器、文件系统、API），且成功取决于**多步决策链**而非单次生成质量。

三类代表性 agent 评测：

| Benchmark | 环境 | 典型任务 | 评测方式 | 核心限制 |
|------|------|------|------|------|
| **WebArena**（Zhou et al., [arXiv:2307.13854](https://arxiv.org/abs/2307.13854), ICLR 2024） | 自建网站（电商/论坛/地图/CMS） | "在指定商品下找到评分最高的评论并回复" | 程序化验证（页面状态/URL/元素存在） | 自建环境覆盖面有限，非真实互联网 |
| **TAU-bench**（Yao et al., [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)） | 对话式 tool-agent-user 交互（航班/零售/金融数据库） | "帮用户改签航班到周五晚、靠窗座、用里程升舱" | 数据库状态比对 + pass^k | 需与模拟用户多轮对话，user 行为设定影响评测口径 |
| **OSWorld**（Xie et al., [arXiv:2404.07972](https://arxiv.org/abs/2404.07972), NeurIPS 2024 D&B） | 真实操作系统 VM（Ubuntu/Windows/macOS） | "在 LibreOffice 做表格 → 导出 PDF → 发邮件" | 屏幕状态/文件系统/窗口状态检查 | 最真但也最贵——每个任务需 VM 快照 + VNC |

**agent 评测的三个特殊维度**（区别于传统 benchmark）：

1. **步数依赖**：评测结果是"给定最大步数"下的 success rate——**准确率-步数曲线**比单点数字更有信息量（类比 test-time scaling 的准确率-预算曲线）。同一模型在 5 步 vs 30 步的 success rate 可以差倍数。
2. **不可复现性**：环境状态、工具调用结果可能随网络/时间变化——需 Docker/VM 快照锁环境确保复现。
3. **轨迹质量 ≠ 成功**：一条"绕了 20 步才成功"的轨迹和"3 步直接完成"在 success rate 上等价，但效率和成本差几倍——需报告**步数/ token 效率**。

> ⚠️ agent 评测当前最大的未解决问题：**缺乏跨 benchmark 的一致排名**——某 agent 在 WebArena 上 SOTA 但在 OSWorld 上一般，反之亦然。这反映了 agent 评测的碎片化状态，尚无"agent 界的 Arena"。

### 1.5 长上下文评测 / Long-Context Evaluation

"声称支持 128K/1M"和"实际能用好"是两回事。长上下文评测需回答三个问题：① **能找到吗**（检索）？② **能理解吗**（推理/综合）？③ **能用到吗**（下游任务）？

| Benchmark | 测什么 | 上下文长度 | 核心发现 |
|------|------|------|------|
| **Needle-in-a-Haystack (NIAH)**（Kamradt 2023, [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)） | 在长文档的**随机位置**插入一条事实 → 问模型这条事实 | 可扩到任意长度 | 多数模型在开头的"针"和结尾的"针"接近 100%，但**中间位置显著下降**（U 形曲线）——长上下文也存在位置偏置 |
| **RULER**（Hsieh et al., [arXiv:2404.06654](https://arxiv.org/abs/2404.06654), COLM 2024） | **多针**检索+聚合（多针答案需综合）+ 多跳问答 | 4K–128K+ | 单针 NIAH 的高分**不等于真实能力**；多针+多跳测试下，许多"支持 128K"的模型有效上下文远低于声称值 |
| **LongBench**（Bai et al., [arXiv:2308.14508](https://arxiv.org/abs/2308.14508), ACL 2024） | 21 个数据集、6 大类任务（单/多文档 QA、摘要、代码、少样本学习、合成、对话） | 1K–18K 中英双语 | 不同任务上长上下文能力**不是单一维度**——代码长上下文和摘要长上下文是两类能力，不能用一个标量概括 |

**长上下文评测与后训练的关键联系**：
- **alignment tax 的特殊形式**：长上下文能力常在 RLHF/DPO 对齐训练中**退化**——安全/对话训练数据多为短上下文，导致长上下文召回率隐性下降。评测时需专门检查。
- **judge 自身的长上下文能力**：如果用 LLM-as-judge 评多轮对话或其他长上下文，**judge 本身能不能在长上下文中准确评判**也是一环（§2.2 judge-human agreement 不直接回答这个）。

> ⚠️ "支持 128K"≠"128K 内都能用好"。评测时至少用 RULER（多针）和 LongBench（多任务）两个维度交叉验证，不要只看单针 NIAH 热力图。

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

### 2.2 judge 与人类一致性怎么量 / Judge–human agreement

"与人类校准 / 一致率"反复出现(§1 RM 行、上面的缓解招、Q31),但**只看原始一致率 (raw agreement) 会高估"扣掉偶然后真正的一致程度"**:当某一类标签占多数时,两个标注者光靠瞎猜也能高概率"撞对"。**Cohen's κ** 扣掉这部分偶然一致:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- $p_o$:观测一致率(两者判断相同的比例);
- $p_e$:**偶然**一致率 $=\sum_c p^{(1)}_c\,p^{(2)}_c$(各类别上两标注者边缘比例之积求和)。

$\kappa=1$ 完美,$\kappa=0$ 表示一致程度仅与"按各自边缘分布独立瞎猜"持平,$\kappa<0$ 比瞎猜还差。直觉:若两标注者都按 90/10 边缘**独立**打标,原始一致率约 $0.9^2+0.1^2=0.82$ 看着很高,但 $\kappa\approx0$——**所以报 judge 与人类一致时要报 κ,别只报一致率**。

```python
import numpy as np
def cohens_kappa(labels_a, labels_b):
    cats = sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(cats)}
    n, K = len(labels_a), len(cats)
    conf = np.zeros((K, K))
    for a, b in zip(labels_a, labels_b):
        conf[idx[a], idx[b]] += 1
    p_o = np.trace(conf) / n                            # 观测一致率
    p_e = (conf.sum(0) * conf.sum(1)).sum() / n ** 2    # 偶然一致率
    return (p_o - p_e) / (1 - p_e)
```

> 多于两个标注者(名义类别):用 **Fleiss' κ**;有序评分:用 **weighted κ** 或 **Krippendorff's α**(可设有序距离、容缺失值)。

### 2.3 冗长去偏的回归形式 / Length-controlled win-rate

§1 表里 AlpacaEval 2.0 的去偏写作"回归扣除长度"——具体做法(Dubois et al., [arXiv:2404.04475](https://arxiv.org/abs/2404.04475))是拟合一个**广义线性模型**预测 judge 的偏好,把**长度差**作为显式一项放进去(简化形式):

$$\text{logit}\,P(\text{model}\succ\text{ref}) = \theta_m + \gamma\cdot\Delta_{\text{len}} + \dots$$

其中 $\Delta_{\text{len}}$ 为两答案长度差。报告时令长度项 $\Delta_{\text{len}}=0$、对指令分布取期望,得到 **length-controlled win-rate**——直觉是"两答案等长时模型本该有的胜率",把 GLM 所建模的"长度关联"那部分扣掉。

⚠️ 注意:它扣的是**模型所拟合的长度关联**,与长度相关的真实质量信号可能被一并扣除,未建模的风格效应也扣不掉;实测能显著提升与 Chatbot Arena(真人 Elo)排名的 Spearman 相关性。

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

1. **能力**:GSM8K/MATH(数学)、HumanEval→SWE-bench(代码)、MMLU/MMLU-Pro/GPQA(知识)、IFEval(指令遵循)。
2. **对齐质量**:AlpacaEval / MT-Bench / Arena-Hard(judge,带位置去偏)、必要时 Arena。
3. **安全/拒答**:有害提示拒答率、过度拒答率。
4. **Agent**:WebArena / TAU-bench(多步交互,报准确率-步数曲线)。
5. **长上下文**:RULER(多针检索)+ LongBench(多任务理解),不只看 NIAH 单针。
6. **回归**:对比基线,确认没在某些维度变差(alignment tax)。
7. **污染审计** + **多 seed/prompt** 报方差。

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

## 增补：2024 前沿覆盖 / 2024 Frontier Supplement

> 以下题目覆盖本页 2024 年新增的评测前沿（SWE-bench、Agent 评测、长上下文评测、GPQA/MMLU-Pro）。

<details>
<summary>Qa. GPQA 为什么被称为"Google-proof"？它与 MMLU / MMLU-Pro 在区分度上有何根本不同？</summary>

    **答：** GPQA（Graduate-Level Google-Proof Q&A）的题是领域专家出的**研究生级**物理/生物/化学选择题，设计目标就是"一个熟练的非专家即使无限制上网搜索也无法在 30 分钟内 100% 答对"（因此叫 Google-proof）。与 MMLU 的根本差异：① MMLU 的题人类天花板高（~90%+），区分度低（强模型挤在天花板附近）；② GPQA 的领域专家准确率约 65%（Diamond 子集各学科有所变化），拉开了模型间差距。MMLU-Pro 通过 10 选 1（而非 4 选 1）+ 排除过拟合/污染题来缓解 MMLU 的天花板问题，是 MMLU 到 GPQA 之间的过渡阶梯。

    **追问：** 为什么推理模型（o1/R1）常选择 GPQA Diamond 作为核心评测之一？
    因为 GPQA 要求深度领域知识+推理（单靠检索不能直接答对），天然适合衡量"长推理能否补足知识缺口"——这正是推理模型的卖点。

</details>

<details>
<summary>Qb. SWE-bench 与 HumanEval 评测的代码能力有何本质区别？为什么单靠 pass@k 不够描述 SWE-bench 上的表现？</summary>

    **答：** HumanEval 测的是**单函数补全**（给定签名+docstring → 写函数体），SWE-bench 测的是**真实软件工程**（给定 GitHub issue + 完整代码库 → 定位 bug → 跨文件 patch → 通过该 instance 的 FAIL_TO_PASS 和 PASS_TO_PASS 测试）。本质区别：① 前者测"生成"，后者测"理解+定位+编辑"的复合能力；② 前者每题孤立，后者需理解大型代码库的跨文件依赖。pass@k 不够描述 SWE-bench 表现的原因是：SWE-bench 每题只有一个正确 patch（不像代码生成可以采 k 个候选看是否有一条对）——pass@k 的"k 条至少一条对"逻辑不适用；SWE-bench 用的是 **% resolved**（关联测试全部通过率），是单次提交的 pass/fail。

    **追问：** SWE-bench Verified 为什么重要？
    原始 SWE-bench 的部分 issue 描述不清晰或测试不可靠——模型可能因"描述歧义"或"测试本身有 bug"而失败，而非能力不足。Verified 子集（OpenAI 2024-08 发布）经人工复核 500 道题排除了这类噪声，使评测信号更干净。

</details>

<details>
<summary>Qc. Agent 评测的三个特殊挑战是什么？为什么"success rate"作为唯一指标不够？</summary>

    **答：** ① **步数依赖**：同一模型在 5 步 vs 30 步预算下的 success rate 可能差几倍——必须报告准确率-步数曲线，而非单点数字。② **不可复现性**：网络/时间/环境状态变化导致同一 agent 两次运行结果不同——需容器/VM 快照锁环境。③ **轨迹质量 ≠ 成功**：绕 20 步成功和 3 步成功在 success rate 上等价，但效率和成本天差地别——需同时报告步数/token 效率。"success rate"不够是因为它只回答了"最终有没有成"，没回答"花多大代价成的"和"在什么步数预算下成的"——后两者对真实部署更重要。

    **追问：** 当前 agent 评测最大的未解决问题是什么？
    缺乏跨 benchmark 的一致排名——某 agent 在 WebArena 上 SOTA 但在 OSWorld 上一般（反之亦然），反映了 agent 评测的碎片化状态，尚无"agent 界的 Chatbot Arena"。

</details>

<details>
<summary>Qd. 为什么单针 NIAH 热力图好 ≠ 长上下文能力强？RULER 和 LongBench 各补了什么短板？</summary>

    **答：** 单针 NIAH 只测"能不能在一大堆无关文本里找到一条事实"——这是长上下文能力的**必要但不充分**条件。RULER 补了"多针检索+多跳聚合"（真实使用常需综合多处信息），暴露了很多"单针满分、多针崩盘"的模型。LongBench 补了"多任务多样性"（21 个数据集、6 大类，涵盖摘要、代码、少样本学习等），揭示了长上下文能力**不是单一维度**——一个模型可能在代码长上下文上强但在摘要长上下文上弱。三者关系：NIAH = 基础检索探针，RULER = 压力测试，LongBench = 生态效度（real-world utility）。

    **追问：** 怎么排查后训练是否损害了长上下文能力？
    在后训练前后各跑一次 RULER + LongBench；RLHF/DPO 训练数据多为短上下文，长上下文能力隐性退化是 alignment tax 的一种常被忽略的形式。

</details>

<details>
<summary>Qe. Arena-Hard 和 MixEval 分别是如何解决"benchmark 过拟合/污染"和"单 benchmark 片面"问题的？各自还有什么局限？</summary>

    **答：** **Arena-Hard** 从 Chatbot Arena 的海量真人投票中自动筛选出 500 道"难倒模型"的题做 GPT-4 pairwise judge——它继承了 Arena 的抗污染性（动态题库）同时把评测从"众包投票"压缩成自动 pipeline，与真人 Elo 的 Spearman 约 94.1%。局限：它全盘继承 judge 偏置，且固定 500 题仍有被针对性优化的风险。**MixEval** 将真实 web 用户查询匹配到现成 benchmark 题目，用 ground-truth 答案（非 judge）评测并动态加权聚合——本质是"多源 benchmark 混合物"而非 judge benchmark。局限：采样权重敏感（调权重就能"改变排名"），且子集中旧 benchmark 的污染会渗透进来。

    **追问：** 这与 §4 的 Goodhart 定律有何关系？
    Arena-Hard 和 MixEval 都是在"同一个 benchmark 成为优化目标后就失效"的前提下设计的抗刷机制——前者靠动态+难题筛选，后者靠多源混合稀释单个 benchmark 的权重。

</details>


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

- **2023-07 · WebArena** — Zhou et al., ICLR 2024. [arXiv:2307.13854](https://arxiv.org/abs/2307.13854) — 自建网站环境（电商/论坛/地图/CMS）做 agent 评测，程序化验证页面状态，是最早的系统化 agent benchmark 之一。

- **2023-08 · LongBench** — Bai et al., ACL 2024. [arXiv:2308.14508](https://arxiv.org/abs/2308.14508) — 21 个数据集、6 大类长上下文任务（单/多文档 QA、摘要、代码、少样本、合成、对话）中英双语评测，揭示长上下文能力不是单一维度。

- **2023-10 · Min-K% Prob** — Shi et al., ICLR 2024. [arXiv:2310.16789](https://arxiv.org/abs/2310.16789) — 以样本最低 k% token 概率做预训练数据成员推断,用于检测评测集是否被"见过"(污染探针)。

- **2023-10 · SWE-bench** — Jimenez et al., ICLR 2024. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) — 用真实 GitHub issue → patch → 跑 FAIL_TO_PASS+PASS_TO_PASS 测试评测代码能力，把代码评测从"函数补全"升级到"真实软件工程"，是 2024 年最具影响力的新 benchmark 之一。

- **2023-11 · IFEval** — Zhou et al., arXiv 预印本. [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) — 用"可程序化验证"的指令(字数/格式等)做指令遵循评测,无需 judge、客观可复算。

- **2023-11 · GPQA** — Rein et al., NeurIPS 2024 D&B. [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) — 研究生级"Google-proof"物理/生物/化学选择题，领域专家仅 65–74%，为推理模型提供了高区分度评测靶。

- **2024-03 · Chatbot Arena** — Chiang et al., ICML 2024. [arXiv:2403.04132](https://arxiv.org/abs/2403.04132) — 真人盲投成对对战 + Bradley-Terry/Elo 转排名,海量投票 + 动态新题成为抗污染的真人偏好金标准。

- **2024-03 · RewardBench** — Lambert et al., arXiv 预印本. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787) — 首个系统化 RM 评测基准(chat/safety/reasoning 等类),用配对偏好正确率衡量奖励模型质量。

- **2024-04 · Length-Controlled AlpacaEval** — Dubois et al., COLM 2024. [arXiv:2404.04475](https://arxiv.org/abs/2404.04475) — 用回归把"长度"从 win-rate 中扣除,显著降低冗长偏置、提升与人类排名的相关性。

- **2024-04 · OSWorld** — Xie et al., NeurIPS 2024 D&B. [arXiv:2404.07972](https://arxiv.org/abs/2404.07972) — 在真实操作系统 VM 中评测 agent（LibreOffice/浏览器/文件管理），以屏幕状态+文件系统变化验证任务完成，是目前最真实的 agent 评测但成本极高。

- **2024-06 · LiveBench** — White et al., ICLR 2025. [arXiv:2406.19314](https://arxiv.org/abs/2406.19314) — 月度滚动更新 + 客观可验证答案的"抗污染"评测,降低数据泄漏导致的虚高。

- **2024-06 · MMLU-Pro** — Wang et al., arXiv 预印本. [arXiv:2406.01574](https://arxiv.org/abs/2406.01574) — MMLU 加强版：10 选 1 + 排除过拟合/污染题，解决 MMLU 天花板效应，强模型间区分度显著提升。

- **2024-06 · Arena-Hard** — Li et al., arXiv 预印本. [arXiv:2406.11939](https://arxiv.org/abs/2406.11939) — 从 Chatbot Arena 数据中自动筛选 500 道难题做 GPT-4 pairwise judge，与真人 Elo Spearman ~0.9，快速近似真人偏好。

- **2024-06 · MixEval** — Ni et al., NeurIPS 2024. [arXiv:2406.06565](https://arxiv.org/abs/2406.06565) — 从多 benchmark 混合采样、动态加权聚合，用多源混合稀释单 benchmark 权重对抗片面性+污染。

- **2024-04 · RULER** — Hsieh et al., COLM 2024. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654) — 多针检索+多跳合成的长上下文压力测试，暴露单针 NIAH 热力图掩盖的有效上下文不足问题。

- **2024-06 · BigCodeBench** — Zhuo et al., arXiv 预印本. [arXiv:2406.15877](https://arxiv.org/abs/2406.15877) — 函数级、多样库调用的代码评测，填补 HumanEval（简单函数）到 SWE-bench（全 repo）之间的评测空白。

- **2024-06 · TAU-bench** — Yao et al., arXiv 预印本. [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) — 对话式 tool-agent-user 交互评测，以数据库状态比对验证 agent 任务完成，agent 评测的另一个重要维度。

- **2024-08 · SWE-bench Verified** — OpenAI, [blog](https://openai.com/index/introducing-swe-bench-verified/). — 对 SWE-bench 的 500 道题做人工复核：确认 issue 描述质量+测试可靠性，排除噪声让评测信号更干净。

- **2024-08 · LiveCodeBench** — Jain et al., ICLR 2025. [arXiv:2403.07974](https://arxiv.org/abs/2403.07974) — 从 LeetCode/AtCoder/Codeforces 新题自动构建评测，按月滚动更新做时间隔离，解决 HumanEval 题量小+过拟合问题。
