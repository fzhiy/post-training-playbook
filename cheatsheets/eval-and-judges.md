# 评测与 LLM-as-judge / Evaluation & Judges

> 后训练里「**评测**」往往是真正的瓶颈:训练能跑、但"到底变好没有 / 哪里变差"全靠评测说话。本页讲怎么评一个对齐后的模型 + 各种评测的坑。
> ⚠️ 不放具体分数(易过时/易记错);具体数字以各 benchmark/leaderboard 官方为准。

## 1. 三类评测 / Three families

| 类型 | 衡量什么 | 代表 |
|---|---|---|
| **能力 benchmark**(自动、有标准答案) | 知识/推理/代码 | MMLU、GSM8K、MATH、HumanEval/MBPP、BBH、IFEval(指令遵循) |
| **偏好/对话评测**(judge 打分) | "回答好不好"的主观质量 | AlpacaEval(LLM-judge win-rate)、MT-Bench(多轮、judge 评分)、Chatbot Arena(真人成对 → Elo) |
| **奖励模型评测** | RM 是否符合人类偏好 | RewardBench(chat/safety/reasoning 等类)、与人工标注的一致率 |

## 2. LLM-as-judge:怎么用 + 偏置

用一个强模型当裁判给回答打分/two-way 比较。**省钱省时,但有系统偏置**:
- **位置偏置 (position bias)**:倾向选"第一个"答案 → 缓解:**交换顺序各评一次**取平均。
- **冗长偏置 (verbosity / length bias)**:倾向更长的答案 → 缓解:长度去偏、控制长度。
- **自我偏好 (self-preference)**:裁判偏好和自己风格像的输出。
- **格式/风格偏置**:markdown、自信语气更讨喜。
缓解通用招:**reference-guided**(给参考答案)、**rubric/打分量表**、**多裁判投票**、与**人类标注校准**。

## 3. 数据污染 / Contamination

训练集混入了评测集 → 分数虚高、不反映泛化。
- **检测**:n-gram / 子串重叠、canary 字符串、对测试样本的异常低 perplexity、改写后掉分。
- **缓解**:decontamination(从训练数据剔除与评测重叠的样本)、用**新/私有**评测集、报告污染审计。

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

### L2 进阶
4. 位置偏置怎么缓解?为什么"交换顺序各评一次"有效?
5. AlpacaEval / MT-Bench / Arena 三者的评测信号有何不同(自动 judge vs 真人 Elo)?
6. 怎么检测训练数据是否污染了某个评测集?

### L3 深挖
7. Goodhart 定律在刷榜上怎么体现?如何设计"抗刷"的评测?
8. reward model 怎么评(RewardBench 的思路)?RM 评测和最终 policy 效果的关系?
9. 对推理模型,为什么评测要从"单次准确率"转向"算力预算下的准确率"?这对评测协议提出什么要求?
10. 如果线上指标(用户留存)和离线评测(judge win-rate)打架,你信哪个、怎么排查?
