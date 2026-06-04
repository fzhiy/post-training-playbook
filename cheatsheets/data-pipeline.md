# 后训练数据流水线 (Data Pipeline & Curation) 速查手册
## 从指令合成、拒绝采样、偏好构造到去污染与数据配比

---

## 目录 (Table of Contents)

1. [数据流水线总览 (Pipeline Overview)](#1-数据流水线总览-pipeline-overview)
2. [指令数据合成 (Instruction Data Synthesis)](#2-指令数据合成-instruction-data-synthesis)
3. [拒绝采样与自训练 (Rejection Sampling & Self-Training)](#3-拒绝采样与自训练-rejection-sampling--self-training)
4. [偏好数据构造 (Preference Data Construction)](#4-偏好数据构造-preference-data-construction)
5. [去重与去污染 (Dedup & Decontamination)](#5-去重与去污染-dedup--decontamination)
6. [数据配比与课程 (Data Mixing & Curriculum)](#6-数据配比与课程-data-mixing--curriculum)
7. [面试题 (Interview Questions)](#7-面试题-interview-questions)

---

## 1. 数据流水线总览 (Pipeline Overview)

后训练（SFT / 偏好对齐 / RLVR）的效果上限,**很大程度由数据流水线决定**——同样的算法,数据质量与配比不同,结果天差地别。一条典型后训练数据流水线:

```
来源 (Sources)            合成/采集 (Generate)         清洗 (Clean)              配比/消费 (Mix → Train)
─────────────            ──────────────────          ─────────────            ────────────────────
人工标注                  Self-Instruct / Evol         去重 (dedup)             配比 (mixing)
开源数据集         ─→     拒绝采样 (RFT/STaR)    ─→    去污染 (decontam)  ─→    课程 (curriculum)   ─→  SFT
蒸馏 (teacher)            偏好对 (人类/AI 反馈)         质量过滤 (filter)        质量/难度排序            DPO / RLHF / RLVR
线上日志                  Magpie / 合成教科书          格式归一                 阶段配比
```

### 1.1 三条主线

| 阶段 | 数据形态 | 典型来源 | 消费算法 |
|------|----------|----------|----------|
| **SFT (指令微调)** | (instruction, response) | 人工 / 合成 / 蒸馏 / 拒绝采样 | 监督交叉熵 |
| **偏好对齐** | (prompt, chosen, rejected) | 人类偏好 / AI 反馈 | DPO / RLHF (PPO/GRPO) |
| **RLVR (可验证奖励)** | (prompt, verifier) | 带 ground-truth 的数学/代码 | GRPO + 规则奖励 |

### 1.2 核心原则:质量 > 数量

**LIMA**（Zhou et al., [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)）只用 **1000 条**精心策划的 SFT 样本(无 RLHF)微调 65B LLaMA,效果可与用多个数量级更多数据训练的模型相当。其论点:**对齐主要是在预训练已有知识之上学一种"表面行为/格式",少量高质量样本即可激活**。

> 💡 LIMA 的"Less is More"不是说数据越少越好,而是:**在 SFT 阶段,样本的质量、多样性、与目标分布的对齐程度,比绝对数量更关键**。这直接驱动了下游所有"过滤优先"的流水线设计。

实践含义:流水线的价值不在"堆量",而在**合成多样化高质量候选 → 严格过滤/去重/去污染 → 按目标能力配比**。下面 §2–§6 逐段展开。

---

## 2. 指令数据合成 (Instruction Data Synthesis)

人工写指令数据昂贵且难规模化。主流做法是**用强 LLM 合成**,核心挑战是**多样性**与**质量**。

### 2.1 Self-Instruct — 种子自举

**Self-Instruct**（Wang et al., [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)）:从一小批人工种子任务出发,few-shot 提示模型**生成新指令 + 输入 + 输出**,再过滤掉与已有池过相似或低质的样本,迭代扩充,最后用扩充数据微调**同一个模型**。Stanford **Alpaca**(无 arXiv 论文,blog/repo)即用 Self-Instruct 思路从 `text-davinci-003` 蒸出 52k 指令数据。

**去同质化过滤**是关键:Self-Instruct 用 **ROUGE-L 相似度 < 0.7** 作为门槛,丢弃与现有指令过相似的新指令,防止合成数据塌缩成少数模板。

```python
import random
from difflib import SequenceMatcher

# Self-Instruct 式 bootstrap:few-shot 提示 LLM 生成新【指令】,再用与现有池的
# 相似度过滤去同质化。(原文还会为每条指令生成 input/output 实例;此处只演示
# "指令生成 + 去重"这一步。)
def sim_ratio(a: str, b: str) -> float:
    # 廉价相似度:difflib 序列匹配比率,作为 ROUGE-L 的近似(原文用 ROUGE-L<0.7)
    return SequenceMatcher(None, a, b).ratio()

def self_instruct_round(pool, llm_generate, k=20, sim_thresh=0.7):
    """pool: 已有指令 list;llm_generate(prompt)->新指令文本。"""
    new_items = []
    for _ in range(k):
        demos = random.sample(pool, min(6, len(pool)))           # few-shot 种子
        prompt = "生成一条与下列示例不同的新任务指令:\n" + "\n".join(demos)
        cand = llm_generate(prompt).strip()
        # 过滤 1:与池中任一指令过相似 → 丢弃
        if any(sim_ratio(cand, p) > sim_thresh for p in pool + new_items):
            continue
        # 过滤 2:启发式质量门(按字符长度,兼容中英文)
        if not (5 <= len(cand) <= 300):
            continue
        new_items.append(cand)
    pool.extend(new_items)
    return new_items
```

### 2.2 Evol-Instruct — 指令进化

**WizardLM / Evol-Instruct**（Xu et al., [arXiv:2304.12244](https://arxiv.org/abs/2304.12244), ICLR 2024）:不是平铺生成,而是用 LLM **把种子指令"进化"得更复杂**,两个方向:

- **深度进化 (In-depth)**:加约束、加推理步骤、复杂化输入、加深难度。
- **广度进化 (In-breadth)**:基于已有指令生成全新的、覆盖新领域的指令。

进化后过滤掉"进化失败"(没变难/变空)的样本,得到难度梯度更陡的指令集。

### 2.3 Magpie — 从对齐模型"零提示"抽取

**Magpie**（Xu et al., [arXiv:2406.08464](https://arxiv.org/abs/2406.08464), ICLR 2025）:利用指令模型(如 Llama-3-Instruct)的自回归性质——**只喂"用户轮之前的模板前缀"(不给任何种子指令)**,模型会自己续写出一条用户指令;再让它回答,得到 (instruction, response) 对。无需种子、无需人工,规模化抽取对齐数据。

### 2.4 蒸馏 (Distillation)

用**更强的 teacher 模型**生成数据训练 student,是最常见的合成路线之一。

- **Sequence-Level KD (SeqKD)**（Kim & Rush, [arXiv:1606.07947](https://arxiv.org/abs/1606.07947), EMNLP 2016）:经典做法,在 teacher **解码出的序列**上训练 student(用序列级输出替代 word-level 软标签),student 更快且有竞争力。
- **合成教科书**:**Textbooks Are All You Need**（Gunasekar et al., [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)）用"教科书质量"过滤的网页数据 + GPT-3.5 合成的教科书与习题训练 1.3B 的 phi-1,论文报告 **HumanEval pass@1 = 50.6%**——证明**高质量合成数据可大幅降低规模需求**。

> ⚠️ 蒸馏/合成数据的两个红线:① **能力上限继承**——student 很难超过 teacher 在该分布上的水平(可作判别但难外推);② **license/合规**——用闭源模型(如 GPT-4)输出做训练可能违反其服务条款,公开发布前务必核对许可。

### 2.5 合成数据的风险:同质化与模式坍缩

> ❌ **陷阱**:无过滤的合成数据会**同质化**——少数高频模板、句式、话题反复出现,多样性骤降。递归地"用模型生成的数据再训模型"还可能引发 **model collapse**(分布尾部丢失、方差坍缩)。缓解:相似度去重(§2.1)、多 teacher / 多温度采样、混入真实人类数据、对覆盖度(领域/难度/长度)做显式约束。

---

## 3. 拒绝采样与自训练 (Rejection Sampling & Self-Training)

当任务**有可验证的正确性信号**(数学答案、单测、格式)时,可以让模型**自己生成、自己筛选**出高质量数据,再监督训练——这是 SFT 与 RL 之间的一条高性价比中间路线。

### 3.1 RFT — 拒绝采样微调

**Rejection sampling Fine-Tuning (RFT)**（Yuan et al., [arXiv:2308.01825](https://arxiv.org/abs/2308.01825)）:对每个 prompt 从当前(SFT)模型采样 **N 条带推理的解**,**只保留答案正确的**,并**对推理路径去重**(同一题保留多条"不同"的正确路径增加多样性),把这些自生成的正确样本作为增广数据再做 SFT。

```python
import re

# RFT:对每题采样 N 条 CoT,只留"答案正确"且"推理路径互不重复"的,
# 作为增广 SFT 数据(Yuan et al. 2023)。
def canonical(y: str) -> str:
    # 极粗的 toy 指纹:数字→#、压空白(RFT 原文按 distinct 方程/推理路径去重,远更精细)
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", y)).strip()

def rejection_sampling_ft_data(prompts, policy_sample, verify, gold,
                               N=64, max_keep=4):
    """policy_sample(x)->一条解;verify(y, gold)->bool。"""
    sft_data = []
    for x in prompts:
        seen, kept = set(), []
        for _ in range(N):
            y = policy_sample(x)
            if not verify(y, gold[x]):          # 答案不对 → 拒绝
                continue
            key = canonical(y)
            if key in seen:                      # 同一推理路径 → 去重
                continue
            seen.add(key); kept.append(y)
            if len(kept) >= max_keep:            # 每题最多保留 max_keep 条
                break
        sft_data += [(x, y) for y in kept]
    return sft_data                              # 再在其上做标准 SFT
```

### 3.2 STaR — 用推理自举推理

**STaR**（Zelikman et al., [arXiv:2203.14465](https://arxiv.org/abs/2203.14465), NeurIPS 2022）:让模型生成 CoT 推理,**保留得到正确答案的推理**做微调;对答错的题,给出正确答案让模型**"反推"(rationalize)**出一条能到达该答案的推理,也加入训练集。迭代后推理能力自举上升。

### 3.3 ReST — Grow / Improve 双循环

**ReST**（Gulcehre et al., [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)）:把自训练写成两步交替——

- **Grow**:用当前策略**离线**生成一批样本(扩充数据集);
- **Improve**:在通过**质量阈值**的样本上微调(可逐步提高阈值)。

比在线 RLHF 更省算力:采样与训练解耦,生成的数据可重复使用多轮。

### 3.4 与 RL 的关系

> 💡 RFT/STaR/ReST 可理解为 **reward-weighted behavior cloning / 迭代 Best-of-N 蒸馏**:只在"奖励=1(正确)"的自采样样本上做最大似然。在 on-policy、单步、小更新的条件下,它**近似**一次"二值奖励的策略梯度",但通常**不算完整 RL**——没有在线探索、连续奖励与逐步信用分配。优点是实现简单、稳定、数据可复用。与 RLVR/GRPO 的衔接见 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html) 与 [llm-post-training](cheatsheet-llm-post-training.html)。

---

## 4. 偏好数据构造 (Preference Data Construction)

偏好对 (prompt, chosen, rejected) 是 RLHF/DPO 的燃料,其质量决定 RM/策略的上限。

### 4.1 偏好信号从哪来

| 来源 | 做法 | 代表 |
|------|------|------|
| **人类反馈** | 标注者对成对回答选更优 | InstructGPT / Llama-2 |
| **AI 反馈 (RLAIF)** | 用强 LLM 当裁判生成偏好标签 | Constitutional AI / RLAIF |
| **混合** | AI 初筛 + 人工校验关键子集 | 多数工业流水线 |

- **Constitutional AI**（Bai et al., [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)）:两阶段——先用模型**自我批判+改写**做 SFT,再用一组"宪法原则"驱动 **AI 生成偏好标签**做 RLHF(RLAIF),无害性几乎不靠人工偏好。
- **RLAIF vs RLHF**（Lee et al., [arXiv:2309.00267](https://arxiv.org/abs/2309.00267), ICML 2024）:实验证明用现成 LLM 的偏好标签(RLAIF)在摘要/对话上可与 RLHF 相当;并提出 **direct-RLAIF**(RL 时直接向 LLM 取奖励,免独立 RM)。

### 4.2 UltraFeedback / Zephyr — 规模化 AI 反馈配方

- **UltraFeedback**（Cui et al., [arXiv:2310.01377](https://arxiv.org/abs/2310.01377), ICML 2024）:对约 **6.4 万条指令**用多个模型生成约 **25.6 万个候选回答**,再由 **GPT-4 在多个质量维度上打数值分+写评语**(累计超 100 万条反馈),构造大规模偏好/RM 数据。
- **Zephyr**（Tunstall et al., [arXiv:2310.16944](https://arxiv.org/abs/2310.16944)）:**dDPO**(distilled DPO)直接在 UltraFeedback 的 AI 反馈上做 DPO,无任何人工标注,蒸出对齐能力。

### 4.3 从打分候选构造偏好对

```python
import itertools, random

# 从一组带分数的候选构造偏好对:chosen=高分,rejected=低分,
# 只保留分差(margin)足够大的对,避免"难区分"噪声(UltraFeedback 式)。
def build_preference_pairs(prompt, candidates, scores, margin=1.0, max_pairs=4):
    # candidates: list[str];scores: list[float](GPT-4 / RM 打分)
    assert len(candidates) == len(scores), "候选与分数须等长(zip 会静默截断)"
    ranked = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)
    pairs = []
    for (y_w, s_w), (y_l, s_l) in itertools.combinations(ranked, 2):
        if s_w - s_l < margin:           # 分差太小 → 难区分 → 丢弃
            continue
        pairs.append({"prompt": prompt, "chosen": y_w,
                      "rejected": y_l, "margin": s_w - s_l})
    random.shuffle(pairs)
    return pairs[:max_pairs]             # 限制每 prompt 贡献的对数,防失衡
```

### 4.4 on-policy vs off-policy 偏好

> ⚠️ 偏好对里的回答**是否来自当前策略**会影响效果:**off-policy**(用别的模型或历史数据的对)是**标准且常用**的做法——DPO 本就为离线偏好设计,Zephyr/UltraFeedback 均为离线;**on-policy/在线**(从待优化模型采样)则更贴近当前分布、覆盖更好。风险不在"目标天然无效",而在偏好数据**离当前策略过远**时的覆盖不足与分布偏移(及长度/风格偏置);迭代/在线 DPO 即为缓解此问题。

> 📝 偏好数据的 **margin 置信度过滤、标注者校准、长度去偏**等设计选择,见 [reward-modeling-eval §3.4](cheatsheet-reward-modeling-eval.html);本节侧重"对从哪来"。

---

## 5. 去重与去污染 (Dedup & Decontamination)

### 5.1 为什么去重

**Deduplicating Training Data Makes Language Models Better**（Lee et al., [arXiv:2107.06499](https://arxiv.org/abs/2107.06499), ACL 2022）:移除训练语料中的近重复片段/文档,可使**逐字记忆(verbatim memorization)下降约一个数量级**,同时以更少训练步达到相当或更好的困惑度。后训练同理:重复样本会过度加权、放大记忆与偏置。

### 5.2 exact vs near-dup

- **精确去重 (exact)**:哈希整段/文档,丢弃完全相同者。
- **近似去重 (near-dup)**:用 **MinHash + LSH** 检测高 Jaccard 相似的文本。

**Jaccard 相似度**:$J(A,B)=\dfrac{|A\cap B|}{|A\cup B|}$,其中 $A,B$ 为两文档的 **k-shingle**(k 个连续 token 的集合)。

**MinHash 关键性质**:对随机哈希(置换)$h$,

$$\Pr[\,\min_{x\in A} h(x)=\min_{x\in B} h(x)\,]=J(A,B)$$

即"两文档签名在某一位相同的概率 = 它们的 Jaccard"。用 $m$ 个独立哈希组成签名,签名相同位的比例就是 $J$ 的无偏估计,把 $O(|A||B|)$ 的两两比较降为常数长度签名比对。

```python
import hashlib

# MinHash 近重复检测:k-shingle + 多哈希最小值组成签名,
# 两签名相同位比例 ≈ Jaccard 相似度。
def shingles(text, k=5):
    toks = text.split()
    return {" ".join(toks[i:i+k]) for i in range(max(1, len(toks) - k + 1))}

def minhash_sig(sh_set, num_perm=128):
    sig = []
    for s in range(num_perm):                       # 每个 s 模拟一个独立哈希
        m = min(int(hashlib.md5((str(s) + sh).encode()).hexdigest(), 16)
                for sh in sh_set)                   # 该哈希下的最小值
        sig.append(m)
    return sig

def est_jaccard(sig_a, sig_b):
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)

def is_near_dup(text, corpus_sigs, k=5, num_perm=128, thresh=0.8):
    sig = minhash_sig(shingles(text, k), num_perm)
    # 此处对 corpus_sigs 暴力线性扫描;生产中在其上加 LSH 分桶,只比对同桶候选
    dup = any(est_jaccard(sig, s) >= thresh for s in corpus_sigs)
    return dup, sig                                 # 真重复才丢;否则把 sig 入库
```

### 5.3 去污染 (Decontamination)

**定义**:从训练数据中**移除与评测基准(test set)重叠的样本**,否则模型"见过考题",benchmark 分数虚高、失去意义。

- **n-gram 重叠法**:若训练样本与某 benchmark 测试题共享较长 n-gram(如 GPT-3 用 13-gram),判为污染并剔除。简单但对改写/翻译泄漏不敏感。
- **成员推断 (membership inference)**:**Min-K% Prob**（Shi et al., [arXiv:2310.16789](https://arxiv.org/abs/2310.16789), ICLR 2024）——免训练地判断一段文本是否在预训练中出现过:取该文本**最低 $k\%$ token 的平均对数似然**;未见文本往往含若干极低概率的 outlier token 拉低该均值,而成员(见过)文本即使最低 $k\%$ token 似然也相对更高,故**该均值偏高 → 更可能见过**。可用于**事后审计**模型是否被某 benchmark 污染。

> 🚨 **陷阱**:去污染必须**在训练前、对照所有目标评测集**做;事后发现污染往往无法补救,只能换评测集。报告分数时应声明去污染口径(n-gram 阶数、是否含改写检测),否则"刷分"不可信。

---

## 6. 数据配比与课程 (Data Mixing & Curriculum)

### 6.1 配比 (Data Mixing)

不同来源/领域/任务的数据**按什么比例混合**,显著影响最终能力。**How Far Can Camels Go? (Tülu)**（Wang et al., [arXiv:2306.04751](https://arxiv.org/abs/2306.04751), NeurIPS 2023）系统消融了 12 个指令微调数据集,结论是**没有单一数据集在所有能力上最优——混合互补来源才能全面**。**Tülu 3**（Lambert et al., [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)）进一步给出开放的 SFT→DPO→RLVR 端到端配方与全套数据。

### 6.2 DoReMi — 自动学最优混比

**DoReMi**（Xie et al., [arXiv:2305.10429](https://arxiv.org/abs/2305.10429), NeurIPS 2023）:先训一个**小 proxy 模型**,用 **Group DRO**(分布鲁棒优化)在各域上学习采样权重,再用这组权重重采样训练大模型,论文报告达到相当性能时**快约 2.6×**。

**Group DRO 目标**:在各域权重 $\alpha$ 的单纯形上做极小极大,

$$\min_\theta \max_{\alpha\in\Delta}\ \sum_{i=1}^{D}\alpha_i\, L_i(\theta)$$

DoReMi 用**超额损失**(proxy 相对一个参考模型的 loss 差)驱动指数梯度,上调"更难/更有提升空间"的域的权重。

```python
import numpy as np

# DoReMi:用小 proxy + Group DRO 学各域采样权重(指数梯度上升)。
# 关键:每域"超额损失"须在 *token 级* 先 clamp 再平均——
#   excess_i = mean_token( max(proxy_token_loss − ref_token_loss, 0) );
# 若在域级"先平均再 clamp",正负 token 差会被错误抵消,不忠实于原算法。
def domain_excess(proxy_tok_loss, ref_tok_loss):
    # proxy_tok_loss[i]/ref_tok_loss[i]:第 i 域的 per-token loss 数组
    return np.array([np.maximum(np.asarray(p) - np.asarray(r), 0.0).mean()
                     for p, r in zip(proxy_tok_loss, ref_tok_loss)])

def doremi_update(weights, excess, lr=1.0, smooth=1e-3):
    weights, excess = np.asarray(weights, float), np.asarray(excess, float)
    log_w = np.log(weights) + lr * excess                # 指数梯度上升
    w = np.exp(log_w - log_w.max())                      # 数值稳定 softmax
    w = w / w.sum()
    return (1 - smooth) * w + smooth / len(w)            # 平滑(论文 c≈1e-3),防域饿死

# 训练 proxy 时取各步 weights 的平均作为最终混比,再用它重采样训大模型。
def sample_domain(weights, rng):
    return rng.choice(len(weights), p=weights)
```

### 6.3 课程 (Curriculum)

按**难度/长度/质量**给数据排序,由易到难喂给模型,常能更稳更快收敛。后训练里常见:

- **难度课程**:先简单题后难题(数学/代码尤甚)。
- **长度课程**:先短序列后长序列(配合上下文窗口扩展)。
- **质量课程**:先大规模中等质量、后小规模高质量(quality annealing/退火)。

### 6.4 质量过滤

用**困惑度、奖励模型分、启发式规则、分类器**给样本打分,丢弃低质;但要警惕过滤器自身的偏置(如偏好长文本)被带入数据。

> ⚠️ 配比与课程在很大程度上是**经验性的**,需按目标能力做消融。两个常见反模式:① **来源过单一/过度去重** → 能力坍缩、多样性丢失;② **盲目堆某一高分域** → 其他能力退化(灾难性遗忘),见 [continual-post-training](cheatsheet-continual-post-training.html)。

---

## 7. 面试题 (Interview Questions)

### L1 — 基础 (Fundamentals)

---

<details>
<summary>Q1: 后训练数据流水线包含哪些主要阶段?</summary>

**答：** 大致为:**来源**(人工/开源/蒸馏/日志)→ **合成或采集**(Self-Instruct/Evol-Instruct/拒绝采样/偏好对)→ **清洗**(去重/去污染/质量过滤/格式归一)→ **配比与课程**(领域混比/难度排序)→ **消费**(SFT / DPO·RLHF / RLVR)。核心理念是"合成多样化候选 → 严格过滤 → 按目标能力配比",而非单纯堆量。

> **追问：** SFT 数据与偏好数据在形态上有何不同?
> SFT 是 (instruction, response) 单条监督样本;偏好数据是 (prompt, chosen, rejected) 成对比较样本,供 RM/DPO 学习相对优劣。

</details>

---

<details>
<summary>Q2: Self-Instruct 的核心思想是什么?为什么必须过滤?</summary>

**答：** 从少量人工种子任务出发,few-shot 提示模型自己生成新的"指令+输入+输出",迭代扩充后微调同一模型。必须过滤是因为:不加约束的自生成会**高度同质化**(少数模板反复出现),Self-Instruct 用 ROUGE-L 相似度 < 0.7 丢弃与现有指令过相似者,保住多样性;同时做基本质量门(长度/噪声)。

> **追问：** Alpaca 与 Self-Instruct 是什么关系?
> Alpaca(无 arXiv 论文)用 Self-Instruct 思路,从 `text-davinci-003` 蒸出 52k 指令数据微调 LLaMA-7B,是 Self-Instruct 的一个著名工程化实例。

</details>

---

<details>
<summary>Q3: 什么是拒绝采样微调 (RFT)?和普通 SFT 有何不同?</summary>

**答：** RFT 对每个 prompt 从当前模型采样 N 条带推理的解,**只保留答案正确的、并对推理路径去重**,再用这些自生成的正确样本做 SFT。与普通 SFT 区别:数据来自**模型自身采样+正确性筛选**(self-training),不需要额外人工标注,且天然 on-policy(落在模型当前分布上)。

> **追问：** RFT 需要什么前提条件?
> 需要一个**可自动验证正确性**的信号(数学答案 exact-match、代码单测、格式检查),否则无法"拒绝"错误样本。

</details>

---

<details>
<summary>Q4: 为什么要对训练数据去重?去重带来什么收益?</summary>

**答：** 重复样本会被过度加权,放大**逐字记忆**与数据中的偏置,还浪费算力。Lee et al. 2021 发现去除近重复可使逐字记忆下降约一个数量级,并以更少训练步达到相当/更好的困惑度。后训练里重复指令还会让模型过拟合特定模板。

> **追问：** exact dedup 和 near-dup 有什么区别?
> exact 只去完全相同;near-dup 用 MinHash/LSH 检测高 Jaccard 相似(改写、轻微编辑也能抓),覆盖更全但计算更复杂。

</details>

---

<details>
<summary>Q5: 什么是数据去污染 (decontamination)?不做会怎样?</summary>

**答：** 去污染是从训练数据中移除与**评测基准测试集**重叠的样本。不做的话模型相当于"见过考题",benchmark 分数虚高、无法反映真实泛化能力,横向比较也失去意义。常见做法:n-gram 重叠剔除(如 13-gram)、成员推断审计(Min-K% Prob)。

> **追问：** 去污染应该在流水线哪个环节做?
> **训练前、对照所有目标评测集**做。事后发现污染通常无法补救,只能更换评测集。

</details>

---

<details>
<summary>Q6: 蒸馏数据 (distillation) 是什么?SeqKD 的核心做法?</summary>

**答：** 用更强的 teacher 模型生成数据训练 student。**Sequence-Level KD (SeqKD, Kim & Rush 2016)** 的核心:不在 word-level 软标签上蒸馏,而是在 **teacher 解码出的整条序列**上训练 student,得到更快且有竞争力的模型。LLM 时代的指令蒸馏(如 Alpaca/Zephyr)是其精神延续。

> **追问：** 蒸馏数据有什么根本局限?
> **能力上限继承**——student 很难在该分布上超过 teacher;此外用闭源模型输出训练可能有 license/合规风险。

</details>

---

<details>
<summary>Q7: 偏好数据的 chosen/rejected 对从哪来?人类 vs AI 反馈?</summary>

**答：** 三条路线:① **人类反馈**——标注者对成对回答选更优(InstructGPT/Llama-2);② **AI 反馈 (RLAIF)**——用强 LLM 当裁判生成偏好标签(Constitutional AI、RLAIF);③ **混合**——AI 初筛 + 人工校验关键子集。AI 反馈成本低、可规模化,但继承裁判模型的偏置。

> **追问：** UltraFeedback 是怎么构造偏好数据的?
> 对约 6.4 万条指令用多个模型生成约 25.6 万个候选,再由 GPT-4 多维度打分+写评语,据此构造偏好/RM 数据;Zephyr 用 dDPO 直接在其上做 DPO。

</details>

---

<details>
<summary>Q8: LIMA 的"Less is More"结论说明了什么?</summary>

**答：** LIMA 用 1000 条精心策划的样本(无 RLHF)微调 65B LLaMA,效果可比肩用多得多数据训练的模型。结论:**对齐主要是在预训练知识之上学一种表面行为/格式**,SFT 阶段样本的**质量、多样性、与目标分布对齐度**比绝对数量更关键。它驱动了"过滤优先、质量优先"的流水线设计哲学。

> **追问：** 这是否意味着数据越少越好?
> 不是。是指**高质量少量 > 低质量大量**;数量仍重要,但回报递减,且质量/多样性是前置门槛。

</details>

---

### L2 — 中级 (Intermediate)

---

<details>
<summary>Q9: Self-Instruct / Alpaca / Evol-Instruct / Magpie 的区别?</summary>

**答：** 都生成指令数据,但机制不同:**Self-Instruct** 种子 bootstrap + 相似度过滤;**Alpaca** 是 Self-Instruct 从 davinci 蒸馏的工程实例;**Evol-Instruct (WizardLM)** 用 LLM 把指令"进化"得更难(深度)或覆盖更广(广度);**Magpie** 利用对齐模型的自回归性,只喂用户轮前缀就让模型自吐指令-回答对,无需种子。复杂度梯度:Magpie/Self-Instruct(广度)→ Evol-Instruct(深度难度)。

> **追问：** 为什么 Evol-Instruct 能提升复杂指令遵循?
> 因为它显式制造**难度更高、约束更多**的指令,数据集难度分布上移,模型被迫学习处理多约束/多步骤任务。

</details>

---

<details>
<summary>Q10: RFT、STaR、ReST 三者的异同?</summary>

**答：** 共同点:都是**自训练**——模型生成、按正确性/质量筛选、再监督训练。区别:**RFT** 侧重对每题保留多条去重的正确推理路径做增广 SFT;**STaR** 额外对答错的题用正确答案"反推(rationalize)"出推理再加入;**ReST** 显式写成 Grow(离线生成)/Improve(过阈值微调)双循环,可逐步提阈值、复用数据多轮。

> **追问：** STaR 的 rationalization 解决了什么问题?
> 解决"难题永远采不到正确推理"的覆盖问题——给定答案让模型反推,使难题也能贡献训练信号。

</details>

---

<details>
<summary>Q11: 合成数据的主要风险是什么?如何缓解?</summary>

**答：** ① **同质化**——少数模板/话题/句式反复,多样性骤降;② **model collapse**——递归用模型数据训模型会丢失分布尾部、方差坍缩;③ **错误放大**——teacher 的系统性错误被批量复制;④ **能力上限**——难超 teacher。缓解:相似度去重、多 teacher/多温度采样、混入真实人类数据、对覆盖度(领域/难度/长度)显式约束、保留人工校验子集。

> **追问：** 为什么递归训练会导致 model collapse?
> 每代采样都会偏向高概率区域、丢弃低概率尾部,误差逐代累积,分布逐渐收窄,最终多样性与对真实分布的覆盖崩塌。

</details>

---

<details>
<summary>Q12: on-policy 偏好数据为何重要?off-policy 有什么问题?</summary>

**答：** off-policy(用别的模型或历史数据)是 DPO 的**标准用法**(DPO 本就为离线偏好设计,Zephyr/UltraFeedback 皆如此),并非无效;但当偏好数据**离当前策略较远**时,覆盖不足与分布偏移会让更新效率下降、易沾上数据源的长度/风格偏置。**on-policy/在线**(从待优化模型采样)让信号落在模型真实会生成的区域,覆盖更好、迭代后失配更小。迭代/在线 DPO 即用新策略不断重采偏好对来缓解。

> **追问：** 这和 RM 的"分布偏移"问题是一回事吗?
> 同源——都是"训练信号分布 ≠ 策略当前分布"。RM 表现为 OOD 评分失准,DPO 表现为隐式奖励梯度失真,缓解思路都是迭代地用 on-policy 数据刷新。

</details>

---

<details>
<summary>Q13: MinHash 如何近似 Jaccard?为什么能加速近重复检测?</summary>

**答：** 对随机哈希 $h$,两集合最小哈希相等的概率恰等于 Jaccard:$\Pr[\min h(A)=\min h(B)]=J(A,B)$。用 $m$ 个独立哈希组成长度 $m$ 的签名,**签名相同位的比例**就是 $J$ 的无偏估计。于是把 $O(|A||B|)$ 的逐 shingle 比较降为**常数长度签名比对**;再配合 LSH 分桶,只比较落同桶的候选,避免全量两两比较。

> **追问：** k-shingle 的 k 怎么选?
> k 太小 → 短文档误判为相似(噪声高);k 太大 → 对小改写过敏、召回低。文本去重常用 k=5~10 个 token,按语料粒度调。

</details>

---

<details>
<summary>Q14: 去污染具体怎么做?有哪些坑?</summary>

**答：** 两类:① **n-gram 重叠**——训练样本与 benchmark 测试题共享较长 n-gram(如 GPT-3 用 13-gram)即剔除,简单但对**改写/翻译泄漏不敏感**;② **成员推断**——如 Min-K% Prob,免训练判断文本是否被预训练见过,用于事后审计。坑:漏掉改写/多语泄漏、阈值过松/过严、只查部分 benchmark、未声明去污染口径导致分数不可比。

> **追问：** 为什么改写式泄漏特别危险?
> 因为它绕过精确/n-gram 匹配(换词、改语序、翻译),却让模型实质见过题目语义,n-gram 去污染抓不到,需要语义级检测或更严格的来源管控。

</details>

---

<details>
<summary>Q15: 数据配比为什么重要?DoReMi 怎么自动学混比?</summary>

**答：** 不同领域/任务数据的混合比例显著影响最终能力分布(Tülu 消融显示无单一数据集全能,需互补混合)。**DoReMi** 先训一个小 proxy 模型,用 **Group DRO** 在各域上学采样权重——对每域算相对参考模型的**超额损失**,用指数梯度上调高超额(更难/更有提升空间)的域;再用学到的权重重采样训练大模型,论文报告快约 2.6×。

> **追问：** 为什么用小 proxy 模型而不是直接在大模型上搜混比?
> 在大模型上反复试不同混比成本极高;小 proxy 便宜,学到的域权重可迁移用于大模型的数据重采样,大幅降低搜索成本。

</details>

---

<details>
<summary>Q16: 课程学习 (curriculum) 在后训练里怎么用?</summary>

**答：** 按难度/长度/质量给数据排序、由易到难喂入。常见:**难度课程**(先简单题后难题,数学/代码尤甚)、**长度课程**(先短后长,配合上下文扩展)、**质量退火**(先大规模中等质量,后小规模高质量收尾)。目的是更稳更快收敛、减少早期被难样本带偏。

> **追问：** 课程学习一定有效吗?
> 不一定。效果依任务与难度度量而定;难度度量不准、或模型容量充足时,课程可能与随机打散差别不大,需消融验证。

</details>

---

<details>
<summary>Q17: Constitutional AI / RLAIF 如何用 AI 反馈替代人工?</summary>

**答：** **Constitutional AI** 两阶段:先让模型按一组"宪法原则"**自我批判并改写**自己的有害回答做 SFT,再用模型**按原则生成偏好标签**做 RLHF(即 RLAIF),无害性几乎不依赖人工偏好。**RLAIF vs RLHF** 实验证明现成 LLM 的偏好标签可与人类标签相当,并提出 direct-RLAIF(RL 时直接向 LLM 取奖励、免独立 RM)。

> **追问：** AI 反馈的主要风险?
> 继承裁判模型的偏置(冗长、自我偏好、格式偏好),且可能放大;关键维度仍需人工校准与异构裁判交叉验证(见 reward-modeling-eval)。

</details>

---

### L3 — 高级 (Advanced)

---

<details>
<summary>Q18: RFT 与 RL (policy gradient) 的数学关系?RFT 算不算 RL?</summary>

**答：** RFT 在"奖励=1(正确)"的自采样样本上做最大似然,可看作 **reward-weighted behavior cloning / 迭代 Best-of-N 蒸馏**。它与策略梯度有联系:REINFORCE 梯度 $\mathbb{E}_{y\sim\pi}[r(y)\nabla\log\pi(y)]$ 中,当 $r\in\{0,1\}$ 且只留 $r=1$ 的样本,正是"在正确样本上提升对数似然"——**但这一近似只在 on-policy、单步、小更新时成立**;RFT 通常把数据采一次就固定下来离线训练,缺在线探索、KL 约束与逐步信用分配,因此**一般不称为完整 RL**,优化天花板也弱于在线 GRPO/PPO。

> **追问：** 既然 RFT 更简单,为什么还要 RLVR/GRPO?
> RFT 只用"正确/错误"的硬筛,丢弃了错误样本中的梯度信息,也无在线探索;GRPO 用组内相对优势在线更新、能利用负样本与连续奖励,在难任务上的天花板更高。

</details>

---

<details>
<summary>Q19: 为什么合成数据训练可能导致 model collapse?递归训练的理论隐患?</summary>

**答：** 递归地"用上一代模型生成的数据训练下一代"会逐代**丢失分布尾部**:每次采样偏向高概率模式,低概率(罕见但真实)样本被系统性欠采;有限采样的统计误差 + 函数逼近误差逐代累积,分布方差收窄、众数集中,最终多样性坍塌、偏离真实数据分布。缓解:**始终混入足量真实人类数据**作为锚,控制合成数据占比,显式约束覆盖度。

> **追问：** 这对"用 GPT-4 蒸馏开源模型"这种一次性蒸馏有多大威胁?
> 一次性蒸馏(teacher 固定且强)比"自产自销递归训练"安全得多——主要风险是能力上限继承与 teacher 偏置复制,而非典型的递归坍塌;但若把蒸馏模型的输出再喂回训练形成闭环,坍塌风险上升。

</details>

---

<details>
<summary>Q20: 数据质量 vs 数量的权衡:LIMA 与 scaling 的张力如何理解?</summary>

**答：** scaling law 说"更多数据→更低 loss",LIMA 说"1000 条就够对齐"——二者不矛盾,作用在**不同阶段与目标**:预训练阶段知识获取靠规模(数量主导);SFT 对齐阶段主要是**激活/格式化**预训练已有能力,此时高质量、多样、对齐目标分布的少量样本边际收益最高,数量回报快速递减。所以"质量优先"针对的是后训练对齐,不否定预训练的规模律。

> **追问：** 那 SFT 数据是否完全不看数量?
> 仍看,但有"足够覆盖"的拐点:覆盖目标能力/格式所需的多样性达到后,继续堆同质数据收益很低,甚至因噪声/重复反伤;关键是**多样性覆盖度**而非纯条数。

</details>

---

<details>
<summary>Q21: 设计一个完整的 SFT 数据流水线,关键决策点有哪些?</summary>

**答：**

```
[来源] 人工种子 + 开源数据 + teacher 蒸馏 + 拒绝采样自产
   ↓
[合成] Self-Instruct(广度)+ Evol-Instruct(难度)+ Magpie(规模)
   ↓
[过滤] 相似度去重(MinHash)+ 质量打分(PPL/RM/分类器)+ 启发式规则
   ↓
[去污染] 对照所有目标 benchmark 做 n-gram + 语义级检测
   ↓
[配比/课程] 领域混比(DoReMi/手工消融)+ 难度/长度课程
   ↓
[训练] SFT → 评测 → 诊断薄弱能力 → 回流补数据
```

关键决策点:teacher 选型与 license、合成 vs 真实数据比例、去重阈值(召回 vs 误删)、去污染口径、各能力域配比、是否上课程、回流迭代频率。

> **追问：** 流水线里最容易被忽视但代价最大的环节是哪个?
> **去污染**——一旦污染未清,所有 benchmark 结论都不可信且难追溯;其次是**多样性/覆盖度监控**,同质化常到评测掉点才被发现。

</details>

---

<details>
<summary>Q22: 偏好数据的 margin 过滤与标注校准如何影响 RM/DPO?</summary>

**答：** **margin 过滤**丢弃 chosen/rejected 分差过小(难区分、标注噪声大)的对——好处是降噪、稳化训练;过激则损失边界样本,使 RM/DPO 在"轻微差异"场景判别力不足。**标注校准**(锚点题、标注者效应建模、多数投票)减少系统性标注偏置,否则偏置会被 RM 学到并经 RLHF 放大进策略。二者都属"数据构建阶段的根本性投资",比事后调 KL/ensemble 更治本。

> **追问：** margin 用 AI 裁判分数算时要注意什么?
> AI 分数有自身偏置与刻度漂移,绝对分差不可跨 prompt 直接比;宜在同一 prompt 内比较、必要时温度=0 多次取稳,且与人工校准。详见 reward-modeling-eval §3.4。

</details>

---

<details>
<summary>Q23: 去污染做得不彻底,对 benchmark 评测意味着什么?如何系统性防泄漏?</summary>

**答：** 不彻底 = 模型部分见过考题,分数**系统性虚高**且不可比,甚至误导模型选型与科学结论。系统性防泄漏:① 训练前对照**所有**目标评测集做 n-gram + 语义级检测;② 用 Min-K% 等成员推断**事后审计**;③ 来源管控(避免抓取含 benchmark 的网页/题库);④ 报告时**声明去污染口径**(n-gram 阶数、是否查改写/多语);⑤ 关键结论用**新发布的/私有 holdout** 评测集复核。

> **追问：** 为什么"新发布的评测集"能作为污染的对照?
> 若模型在训练截止后才发布的评测集上表现骤降,强烈提示此前高分部分来自污染;时间上的"不可能见过"提供了天然对照。

</details>

---

<details>
<summary>Q24: DoReMi 的 Group DRO 目标是什么?为什么用 proxy 模型学混比?</summary>

**答：** Group DRO 优化各域加权损失的**最坏情况**:$\min_\theta\max_{\alpha\in\Delta}\sum_i\alpha_i L_i(\theta)$,$\alpha$ 在域权重单纯形上。DoReMi 用一个参考模型定义各域**超额损失**,用指数梯度迭代上调高超额域的权重,得到一组对"难域"鲁棒的混比。用 **proxy(小模型)**是因为直接在大模型上反复搜混比成本极高;小模型上学到的域权重可迁移到大模型的数据重采样,论文报告训练提速约 2.6×。

> **追问：** Group DRO 学到的"上调高损失域"会不会过度偏向噪声域?
> 会有此风险——纯按 loss 高低加权可能放大噪声/不可学域。DoReMi 用相对参考模型的**超额**损失(而非绝对 loss)并配合平滑/裁剪缓解,但对参考模型选择与噪声仍敏感。

</details>

---

<details>
<summary>Q25: 蒸馏/合成数据的 license、版权与"能力上限继承"问题?</summary>

**答：** ① **license/合规**:多数闭源模型(GPT-4 等)的服务条款**禁止用其输出训练竞品模型**,用其蒸馏数据公开发布有合规风险,需核对条款或改用许可宽松的 teacher;② **能力上限继承**:student 难在 teacher 覆盖的分布上超过 teacher,蒸馏适合"拉平/普及"能力而非"突破前沿";③ **偏置/错误复制**:teacher 的系统性错误与风格偏好会被批量复制进 student。实践:多 teacher、混真实数据、保留人工校验、记录数据来源与许可。

> **追问：** 如果想让 student 超过 teacher,数据上能做什么?
> 转向**可验证自产数据**(RFT/RLVR,用环境/单测/答案当 teacher 而非模型)、on-policy 偏好迭代、引入新的真实人类数据或工具反馈——让监督信号来自"客观正确性"而非另一个模型的分布。

</details>

---

## 附录：关键术语速查 (Glossary)

| 英文术语 | 中文 | 简要定义 |
|----------|------|----------|
| Self-Instruct | 自指令 | 从种子任务 bootstrap 合成指令数据 + 相似度过滤 |
| Evol-Instruct | 指令进化 | 用 LLM 把指令进化得更难/更广 (WizardLM) |
| Distillation | 蒸馏 | 用强 teacher 生成数据训练 student |
| SeqKD | 序列级蒸馏 | 在 teacher 解码序列上训练 student |
| Rejection Sampling FT (RFT) | 拒绝采样微调 | 采样 N 条只留正确并去重做 SFT |
| STaR | — | 自举推理:对答错题用答案反推推理 |
| ReST | — | Grow(生成)/Improve(过阈值微调)双循环自训练 |
| RLAIF | AI 反馈强化学习 | 用 LLM 偏好标签替代人类标签 |
| Constitutional AI | 宪法式 AI | 用原则驱动自我批判+AI 偏好对齐 |
| Preference Pair | 偏好对 | (prompt, chosen, rejected) 三元组 |
| on-policy data | 同策略数据 | 来自当前待优化模型的采样 |
| Dedup | 去重 | 移除(近)重复样本 |
| MinHash / LSH | — | 用最小哈希签名近似 Jaccard、加速近重复检测 |
| Jaccard | 杰卡德相似度 | $|A\cap B|/|A\cup B|$ |
| Decontamination | 去污染 | 移除与评测集重叠的训练样本 |
| Min-K% Prob | — | 免训练成员推断,检测预训练是否见过文本 |
| Data Mixing | 数据配比 | 各域/来源的混合比例 |
| DoReMi | — | 用 proxy + Group DRO 自动学域混比 |
| Curriculum | 课程学习 | 按难度/长度/质量排序喂数据 |
| Model Collapse | 模型坍缩 | 递归训练致分布尾部丢失、多样性坍塌 |
| LIMA | — | "Less is More":少量高质量样本即可对齐 |

---

*本手册仅供学习参考。涉及的论文结论与数值以原始论文为准;benchmark 分数仅用于说明,不构成横向比较。*

## §A 核心论文时间线 / Key Papers Timeline

- **2016-06 · Sequence-Level Knowledge Distillation** — Kim & Rush, EMNLP 2016. [arXiv:1606.07947](https://arxiv.org/abs/1606.07947) — 把知识蒸馏提升到序列级:在 teacher 解码出的序列上训练 student,替代 word-level 软标签,得到更快且有竞争力的模型,是 LLM 蒸馏数据的思想源头。

- **2021-07 · Deduplicating Training Data Makes Language Models Better** — Lee et al., ACL 2022. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) — 系统证明去除训练语料近重复可使逐字记忆下降约一个数量级,并以更少训练步达到相当/更优困惑度,奠定"去重是基础动作"的共识。

- **2022-03 · STaR: Bootstrapping Reasoning With Reasoning** — Zelikman et al., NeurIPS 2022. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) — 用模型自生成的正确 CoT 推理迭代自举;对答错题用正确答案"反推"推理也纳入训练,无需大规模人工推理标注。

- **2022-12 · Self-Instruct: Aligning LMs with Self-Generated Instructions** — Wang et al., ACL 2023. [arXiv:2212.10560](https://arxiv.org/abs/2212.10560) — 从少量种子任务 bootstrap 合成指令数据 + ROUGE-L 相似度过滤去同质化,开创低成本规模化指令数据合成范式(Alpaca 即其实例)。

- **2022-12 · Constitutional AI: Harmlessness from AI Feedback** — Bai et al., preprint. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) — 用一组"宪法原则"驱动模型自我批判+改写做 SFT,再以 AI 生成偏好标签做 RLHF(RLAIF),无害性几乎不依赖人工偏好标注。

- **2023-04 · WizardLM: Empowering LLMs to Follow Complex Instructions** — Xu et al., ICLR 2024. [arXiv:2304.12244](https://arxiv.org/abs/2304.12244) — 提出 Evol-Instruct:用 LLM 把种子指令在深度(加约束/难度)与广度(覆盖新领域)上进化,显著提升复杂指令遵循。

- **2023-05 · LIMA: Less Is More for Alignment** — Zhou et al., NeurIPS 2023. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) — 仅 1000 条精选样本(无 RLHF)微调 65B LLaMA 即可对齐,论证对齐主要是激活预训练已有能力,质量/多样性优先于数量。

- **2023-05 · DoReMi: Optimizing Data Mixtures** — Xie et al., NeurIPS 2023. [arXiv:2305.10429](https://arxiv.org/abs/2305.10429) — 用小 proxy 模型 + Group DRO 学习各域采样权重,再重采样训大模型,论文报告达相当性能快约 2.6×,把数据配比从手工调成可优化目标。

- **2023-06 · How Far Can Camels Go? (Tülu)** — Wang et al., NeurIPS 2023. [arXiv:2306.04751](https://arxiv.org/abs/2306.04751) — 系统消融 12 个指令微调数据集,结论是无单一数据集全能、需混合互补来源,并发布 Tülu 开放指令模型套件。

- **2023-06 · Textbooks Are All You Need (phi-1)** — Gunasekar et al., preprint. [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) — 用"教科书质量"过滤数据 + 合成教科书/习题训练 1.3B 的 phi-1,论文报告 HumanEval pass@1 = 50.6%,论证高质量合成数据可大幅降低规模需求。

- **2023-08 · Scaling Relationship on Math Reasoning (RFT)** — Yuan et al., preprint. [arXiv:2308.01825](https://arxiv.org/abs/2308.01825) — 提出拒绝采样微调(RFT):采样多条推理只留正确且去重者做增广 SFT,并研究预训练 loss 与数据量如何共同决定数学推理能力。

- **2023-08 · Reinforced Self-Training (ReST)** — Gulcehre et al., preprint. [arXiv:2308.08998](https://arxiv.org/abs/2308.08998) — 把自训练写成 Grow(离线生成样本)/Improve(在过阈值样本上微调)双循环,采样与训练解耦,比在线 RLHF 更省算力且数据可复用。

- **2023-09 · RLAIF vs. RLHF: Scaling RLHF with AI Feedback** — Lee et al., ICML 2024. [arXiv:2309.00267](https://arxiv.org/abs/2309.00267) — 实验证明用现成 LLM 的偏好标签(RLAIF)在摘要/对话上可与 RLHF 相当,并提出 direct-RLAIF:RL 时直接向 LLM 取奖励、免独立 RM。

- **2023-10 · UltraFeedback: Boosting LMs with Scaled AI Feedback** — Cui et al., ICML 2024. [arXiv:2310.01377](https://arxiv.org/abs/2310.01377) — 对约 6.4 万条指令用多模型生成约 25.6 万个候选、GPT-4 多维度打分+评语,构造大规模 AI 反馈偏好/RM 数据,成为开源对齐的重要燃料。

- **2023-10 · Zephyr: Direct Distillation of LM Alignment** — Tunstall et al., preprint. [arXiv:2310.16944](https://arxiv.org/abs/2310.16944) — 用 dDPO(distilled DPO)直接在 UltraFeedback 的 AI 反馈上做 DPO,无任何人工标注即蒸出强对齐的 7B 模型,验证"AI 反馈 + DPO"配方。

- **2023-10 · Detecting Pretraining Data (Min-K% Prob)** — Shi et al., ICLR 2024. [arXiv:2310.16789](https://arxiv.org/abs/2310.16789) — 提出免训练成员推断 Min-K% Prob:取文本最低 k% token 的平均对数似然——成员文本该均值偏高、未见文本因含低概率 outlier 而偏低,可用于 benchmark 污染审计。

- **2024-06 · Magpie: Alignment Data Synthesis from Scratch** — Xu et al., ICLR 2025. [arXiv:2406.08464](https://arxiv.org/abs/2406.08464) — 利用对齐模型的自回归性,只喂用户轮前模板就让模型自吐指令-回答对,零种子、零人工、规模化合成对齐数据。

- **2024-11 · Tülu 3: Pushing Frontiers in Open Post-Training** — Lambert et al., preprint. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) — 开放端到端后训练配方(SFT→DPO→RLVR)与全套数据/代码,引入 RLVR 作为关键组件,系统呈现数据流水线与配比工程。
