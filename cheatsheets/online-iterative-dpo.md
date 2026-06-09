# 在线 / 迭代 DPO (Online & Iterative DPO) 速查手册
## 从离线 DPO 的分布漂移到 on-policy 采样、自我奖励、自博弈与博弈式偏好优化

---

## 1. 在线 / 迭代 DPO 总览 (Overview)

标准 **DPO**（Rafailov et al., [arXiv:2305.18290](https://arxiv.org/abs/2305.18290), NeurIPS 2023)是**离线 (offline)** 算法:偏好数据集 $\mathcal{D}$ 在训练前一次性收集,来自某个固定的生成策略 $\mu$(通常是 SFT 模型)。训练中 $\pi_\theta$ 不断更新,但 $\mathcal{D}$ 静止不动——这是 **off-policy**。**在线 / 迭代 DPO** 的核心改动只有一句话:**每轮都用当前策略 $\pi_\theta^{(t)}$ 重新采样响应、重建偏好对**,让训练信号始终贴着当前策略的输出分布(**on-policy**)。

```
离线 DPO (off-policy)                      在线 / 迭代 DPO (on-policy)
─────────────────────                      ──────────────────────────
数据一次采自固定 μ (SFT)                    每轮用当前 π_θ^(t) 重采样
π_θ 漂离 μ 后,(y_w,y_l) 不再覆盖其输出       偏好对始终匹配当前策略分布
便宜、可复现、无在线基础设施          ⇄      贵 (每轮采样+打分+训练),需标注器
受旧分布锁死,难超越数据             ⇄      可探索新输出、持续逼近最优策略
```

### 1.1 一张族谱:两个正交的轴

在线偏好优化的方法差异,几乎都落在两个**正交问题**上:**① 响应从哪来?② 偏好谁来标?**

| 轴 | 取值谱 | 代表 |
|------|--------|------|
| **① 响应来源** | 固定 $\mu$(离线)→ 当前 $\pi_\theta$(on-policy)→ 带探索的 $\pi_\theta$ | DPO → Online DPO → XPO |
| **② 偏好标注** | 人类 / 外部 RM / LLM-as-judge / **模型自己** / 博弈均衡 | RLHF·OAIF / Self-Rewarding / Nash-PO·SPPO |

- **§3** 处理"把 DPO 变 on-policy"的最小闭环 + 偏好对怎么构造(RM vs LLM-judge vs 人)。
- **§4** 把标注器**收进模型自身**:自我奖励(自己当裁判)、自博弈(拿自己的输出当负例)。
- **§5** 把"被动 on-policy 采样"升级为**主动探索**与**博弈均衡**(无需 BT 传递性假设)。
- **§6** 落到生产配方(Llama-3 六轮、Tülu-3)与回路里的特有陷阱。

### 1.2 本页边界 (Scope)

> ✅ 本页只讲**"把偏好优化做成 on-policy / 迭代"这一层**。以下内容**不在此重复**,请走交叉链接:
> - **DPO 损失推导、BT 模型、隐式奖励** → [llm-post-training §6](cheatsheet-llm-post-training.html)
> - **离线变体 IPO / KTO / ORPO / SimPO 的损失对比** → [llm-post-training §7](cheatsheet-llm-post-training.html)(§7.5 已给出在线 vs 离线的简表,本页是其深化)
> - **奖励模型怎么训 / 怎么评、过程奖励** → [reward-modeling-eval](cheatsheet-reward-modeling-eval.html)
> - **在线 RLHF (PPO / GRPO) 的策略梯度细节** → [llm-post-training §8](cheatsheet-llm-post-training.html) 与 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html)

---

## 2. 为什么要 on-policy (The Case for On-Policy)

### 2.1 离线 DPO 的三个失效模式

**(a) 分布漂移 (distribution mismatch)。** 训练中 $\pi_\theta$ 离开 $\mu$ 后,$\mathcal{D}$ 里的 $(y_w, y_l)$ 落在 $\pi_\theta$ 已经很少产生的区域。DPO 在这些**过期样本**上拉大 margin,对当前策略真正会生成的输出却没有任何监督——梯度花在了"不会再走的路"上。

**(b) 似然漂移 (likelihood displacement)。** DPO 损失只约束 chosen 与 rejected 的**对数比之差** $\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}$,**不**直接约束 $\log\pi_\theta(y_w)$ 的绝对值。注意这不是单个孤立配对更新的必然——在 §3.1 那种单步 softmax 更新里 $y_w$ 概率其实会上升。但在**真实 LM 的共享参数聚合训练**中(海量相互冲突的偏好对、序列级归一、优化器动力学叠加),会出现一种"达标"路径:把 $y_l$ 压得更狠,而 $y_w$ 的概率**同时也被带低**——只要降得比 $y_l$ 慢,margin 照样变大。被挤出的概率质量往往流向**第三类 OOD 输出**(既非 $y_w$ 也非 $y_l$)。On-policy 数据让 $y_w$ 本就来自当前分布,缓解这种"把好答案也一起压低"的退化。

**(c) OOD 过优化 (over-optimization)。** DPO 的隐式奖励 $\hat r_\theta = \beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ 是当前策略/参考比值的**重参数化**,而非一个会"外推"的独立 RM。问题在于:数据未覆盖的区域**缺少偏好监督来校准**这个比值,离线训练管不到,策略可能朝着"隐式奖励虚高但实际很差"的方向漂。on-policy 重采样相当于不断把"策略现在真实会去的地方"重新拉回标注回路。

> 🚨 三者会在**迭代回路里叠加放大**:这正是 §6 强调"每轮刷新 RM / 加 chosen-NLL 锚 / 控长度"的原因。

**from-scratch 实现**:带 mask、β、reference logps 的 batch 级 DPO loss(面试手撕口径):

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp, logp_ref, mask, beta=0.1):
    """DPO loss(带 mask 的 batch 版,面试手撕标准写法)。
    logp/logp_ref: (B, T) π_θ 与 π_ref 下的逐 token log-prob
    mask: (B, T) 有效 token(排除 padding),B 的前半为 chosen、后半为 rejected
    返回标量 loss。"""
    # seq logp:每条序列的总对数概率
    seq_logp   = (logp * mask).sum(dim=1)      # (B,) π_θ
    seq_logp_r = (logp_ref * mask).sum(dim=1)  # (B,) π_ref
    # 拆回 chosen / rejected(前 B/2 为 chosen)
    B = logp.shape[0] // 2
    logp_c, logp_r_c = seq_logp[:B], seq_logp_r[:B]       # chosen
    logp_j, logp_r_j = seq_logp[B:], seq_logp_r[B:]       # rejected
    # 隐式奖励差 β·[(log π_c/π_ref_c) - (log π_j/π_ref_j)]
    log_ratio_diff = beta * ((logp_c - logp_r_c) - (logp_j - logp_r_j))
    loss = -F.logsigmoid(log_ratio_diff).mean()            # -log σ(Δ)
    return loss
# 关键点:
# ① chosen 与 rejected 的 logp_ref 必须用 π_ref 算,不是 π_θ(旧策略)
# ② β 控制从偏好里学的强度(大=保守/贴 ref,小=激进/易过优化)
# ③ 这是 DPO 的基础写法;迭代版只需每轮用新的(chosen,rejected)对调用此 loss
# ④ 变体:IPO(loss = (log_ratio_diff - 1/(2β))²)、KTO(单样本,不需成对)
```

### 2.2 在线 vs 离线:性能差距从何而来 (Tang et al. 2024)

**Tang et al.**(["Understanding the Performance Gap between Online and Offline Alignment Algorithms"](https://arxiv.org/abs/2405.08448), [arXiv:2405.08448](https://arxiv.org/abs/2405.08448), preprint)做了一组**受控实证 / 机制研究**(实验+消融,非定理证明),系统追问"在线为何稳定优于离线":

- **现象**:在受控设置下,**在线算法稳定优于离线算法**,且差距**不能**仅靠"给离线方法喂更多数据 / 扩大覆盖"补齐。
- **排除项**:他们逐一检验并**否定**了几种朴素解释——差距不是单纯由离线方法的**判别精度**不足、也不是由**损失函数形式**(对比式离线损失 vs 在线 RL)决定的;即便让离线方法用上强对比损失,差距依旧。
- **核心归因**:**on-policy 采样本身**(数据由当前策略生成)是关键驱动力,而非"离线/在线算法"这一标签。离线方法恰恰在"它本应学会区分"的那部分响应上表现不佳。

> 💡 一句话:在他们的设置里,**让数据 on-policy 比换损失函数更重要**。这给了"在线 / 迭代 DPO"一个独立于具体损失的实证支撑——可以**优先**把 DPO 的数据来源改成 on-policy,而未必先换损失;但这不等于"换损失/上 RL 永远没必要"(见下方口径提醒)。

> ⚠ 口径提醒:以上是 Tang et al. 在其特定设置下的结论,**不要外推成"在线在任何任务上都必然更好"**。在线的代价(采样+打分+训练成本、奖励 hacking 风险)是真实的,见 §6。

---

## 3. 在线 DPO 算法 (Online DPO Algorithms)

### 3.1 迭代闭环 (The Iterative Loop)

把离线 DPO 改成 on-policy,最小闭环就三步,逐轮重复:

```
for t in 0..T-1:
   1) 生成:对每个 prompt x,从当前策略 π_θ^(t) 采样 K 条响应      ← on-policy
   2) 标注:用 RM / LLM-judge / 人类把 K 条排序,构造偏好对 (y_w,y_l) → D^(t)
   3) 更新:π_θ^(t+1) ← DPO-update(π_θ^(t), D^(t);  ref = π_ref)
```

形式化即 [llm-post-training §7.5](cheatsheet-llm-post-training.html) 给出的:

$$\pi_\theta^{(t+1)} \leftarrow \text{DPO-update}\!\left(\pi_\theta^{(t)},\;\mathcal{D}^{(t)}\right), \quad \mathcal{D}^{(t)} \sim \pi_\theta^{(t)}$$

下面这段玩具代码用一个**离散响应空间**展示:为什么 on-policy 采到的偏好对能持续把策略推向高奖励区,而一次性离线数据会"用旧了"。DPO 在 softmax 策略上的梯度有个干净的闭式——logsumexp 项相消,margin 对 logits 的梯度就是 $\beta(e_w - e_l)$:

```python
import numpy as np

# ===== DPO on a toy categorical policy =====
# logπ_i = θ_i - logsumexp(θ);  d(logπ_w - logπ_l)/dθ_j = [w==j] - [l==j]
# 所以 margin 对 logits 的梯度与 logsumexp 无关 —— 干净的闭式。

def log_softmax(theta):
    m = theta.max()
    return theta - m - np.log(np.exp(theta - m).sum())

def softmax(theta):
    z = theta - theta.max()
    e = np.exp(z)
    return e / e.sum()

def dpo_step(theta, theta_ref, w, l, beta=0.5, lr=0.3):
    """对一个偏好对 (w 胜, l 负) 做一步 DPO 梯度下降,返回更新后的 logits。"""
    lp, lpr = log_softmax(theta), log_softmax(theta_ref)
    margin = beta * ((lp[w] - lpr[w]) - (lp[l] - lpr[l]))
    sig = 1.0 / (1.0 + np.exp(-margin))      # σ(margin)
    coef = (sig - 1.0) * beta                # dL/dmargin = σ(margin) - 1 < 0
    grad = np.zeros_like(theta)
    grad[w] += coef                          # 抬高 w
    grad[l] -= coef                          # 压低 l
    return theta - lr * grad

def expected_reward(theta, r):
    return float((softmax(theta) * r).sum())

def make_pair(samples, r):
    """从一批采样里按真实奖励取 (最优, 最差) 作偏好对。"""
    s = sorted(samples, key=lambda i: r[i])
    return s[-1], s[0]

# 真实奖励:5 个离散响应,索引 4 最好
r = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
theta0 = np.zeros(5)                          # SFT 起点:均匀
rng = np.random.default_rng(0)

# --- 离线:偏好对一次性采自 π0,之后反复用 ---
off = theta0.copy()
pool = [make_pair(rng.choice(5, size=2, p=softmax(theta0), replace=False), r)
        for _ in range(40)]
for w, l in pool:
    off = dpo_step(off, theta0, w, l)

# --- 在线:每步都从当前策略重新采样偏好对 ---
on = theta0.copy()
for _ in range(40):
    w, l = make_pair(rng.choice(5, size=2, p=softmax(on), replace=False), r)
    on = dpo_step(on, theta0, w, l)           # ref 固定为 SFT

print("offline E[r] =", round(expected_reward(off, r), 3))
print("online  E[r] =", round(expected_reward(on, r), 3))
```

> 💡 玩具直觉:两者都在拉 margin,但**在线**每轮把"策略现在真实会采到的对"喂回去,持续把质量压到高奖励响应上;**离线**池子越用越偏离当前策略,边际收益衰减。真实系统里这一差距还会被似然漂移与过优化进一步拉大。

### 3.2 偏好对怎么构造:RM vs LLM-judge vs 人

on-policy 采到 $K$ 条响应后,**谁来定 $(y_w, y_l)$** 决定了方法的成本与偏置:

| 标注器 | 怎么标 | 优点 | 风险 |
|--------|--------|------|------|
| **外部 RM** | 打分 → 取最高/最低,或按分采样 | 便宜、可批量、可复用已有 RM | RM 被 hack;分数偏置(长度等)随轮次放大 |
| **LLM-as-judge** | 让强模型成对判优劣(**OAIF**) | 无需训 RM、可在线即时标 | 继承裁判模型偏好/风格偏置;judge 也会错 |
| **人类** | 人工标注偏好 | 信号最可信、最贵 | 慢、贵,难逐轮做(Llama-3 用人+RM 混合) |

**OAIF**(Guo et al., ["Direct Language Model Alignment from Online AI Feedback"](https://arxiv.org/abs/2402.04792), [arXiv:2402.04792](https://arxiv.org/abs/2402.04792), preprint)是"LLM-judge 在线标注"的代表:每步对当前策略采的两条响应,用一个 **online annotator LLM** 即时判优劣再做 DPO 更新——把离线 DPO 的"静态偏好集"替换成"在线 AI 反馈",兼顾了 on-policy 与免训 RM。

> 📝 取对不只"最高 vs 最低"一种。**RSO**(Liu et al., ["Statistical Rejection Sampling Improves Preference Optimization"](https://arxiv.org/abs/2309.06657), [arXiv:2309.06657](https://arxiv.org/abs/2309.06657), ICLR 2024)指出:理想的偏好对应采自**最优策略 $\pi^\*$**(注意是目标最优策略,不是当前策略)的分布,于是用**拒绝采样**从 $\pi_\text{ref}$ 里近似采出贴近 $\pi^\*$ 的样本再标注,把"离线数据来源"往**理想的 $\pi^\*$ 分布**纠偏——是连接离线采样与该理想的一步。

### 3.3 参考策略与 β:回路里的两个旋钮

迭代回路里 $\pi_\text{ref}$ 与 $\beta$ 的设定直接决定稳定性:

- **$\pi_\text{ref}$ 每轮重置 vs 固定 SFT。**
  - **每轮重置** $\pi_\text{ref} \leftarrow \pi_\theta^{(t)}$:类信赖域 (trust-region) 步,KL 约束始终是"相对上一轮",更新更稳;代价是**丢掉对 SFT 的锚定**,多轮后可能整体漂走。
  - **固定 SFT** 作 $\pi_\text{ref}$:始终锚定起点;但 $\pi_\theta$ 越漂,$\frac{\pi_\theta}{\pi_\text{ref}}$ 越大,有效约束**越松**,后期容易过优化。
  - 生产实践(Llama-3 / Tülu-3)多在**一轮之内固定 ref**,轮与轮之间再换基准。
- **$\beta$(KL 强度)。** $\beta$ 越大越贴 $\pi_\text{ref}$、越保守;越小越敢漂、越易过优化。$\beta$ 与 ref 的语义对比另见 [llm-post-training §9.3](cheatsheet-llm-post-training.html)。

---

## 4. 自我奖励与自博弈 (Self-Rewarding & Self-Play)

§3 的标注器都是"外部"的。这一节把标注器**收进模型自身**——好处是摆脱外部 RM/人的瓶颈,风险是信号与策略**同源**,容易自我强化偏置。

### 4.1 自我奖励 (Self-Rewarding LMs)

**Self-Rewarding LMs**(Yuan et al., [arXiv:2401.10020](https://arxiv.org/abs/2401.10020), ICML 2024)让**同一个模型同时当策略和裁判**:用 *LLM-as-a-Judge* 提示让模型给自己采的响应打分,据此构造偏好对,再做迭代 DPO。关键叙事是**裁判能力与策略能力一起涨**——论文观察到迭代中指令遵循与"当裁判"的能力同步提升,形成自提升回路(论文做了 $M_1\!\to\!M_2\!\to\!M_3$ 三轮迭代)。

```python
# ===== Self-Rewarding: 模型自己当裁判,构造偏好对 (§4.1) =====
def self_reward_pairs(prompt, gen_fn, judge_fn, k=4):
    """采 k 条,自己打分,取 (最高分, 最低分) 作偏好对;全平则无信号、跳过。"""
    cands = [gen_fn(prompt) for _ in range(k)]
    scored = sorted(((judge_fn(prompt, c), c) for c in cands),
                    key=lambda t: t[0], reverse=True)
    if scored[0][0] == scored[-1][0]:
        return None                          # 无区分度,跳过这条 prompt
    return scored[0][1], scored[-1][1]       # (y_w, y_l)
```

> 🚨 失效模式:裁判与策略同源 → **奖励 hacking / 自我偏好** 会被回路放大(模型偏爱自己的风格、给自己虚高分),多轮后可能**饱和或退化**。实务上靠"固定一部分外部/可验证信号 + 定期人审"兜底。

### 4.2 自博弈:SPIN 与 SPPO

**SPIN**(Self-Play Fine-Tuning, Chen et al., [arXiv:2401.01335](https://arxiv.org/abs/2401.01335), ICML 2024)**完全不需要偏好标签,也不需要外部奖励**:把 **SFT 人类数据当 $y_w$(正例)**、把**当前模型自己生成的响应当 $y_l$(负例)**,用 DPO 式对比目标训练模型去区分"人类数据"与"自己的输出"。这是一种判别器/生成器自博弈——迭代到模型的生成与 SFT 数据**分布不可区分**时收敛($\pi \to p_\text{data}$)。

```python
# ===== SPIN: 自博弈配对 —— 人类数据=胜, 模型自采=负 (§4.2) =====
def spin_pairs(prompts, human_responses, model_gen_fn):
    """y_w 取自 SFT 人类数据,y_l 取自当前模型;相同则无信号、跳过。"""
    pairs = []
    for x, y_human in zip(prompts, human_responses):
        y_self = model_gen_fn(x)
        if y_self != y_human:
            pairs.append((x, y_human, y_self))   # (prompt, y_w, y_l)
    return pairs
```

> ⚠ SPIN 的优化目标(固定点)是 SFT 数据分布:它学的是"逼近人类数据",在**没有外部奖励**时无法把目标推到该分布之外——这是它与"带外部奖励的在线 DPO"的本质区别。

**SPPO**(Self-Play Preference Optimization, Wu et al., [arXiv:2405.00675](https://arxiv.org/abs/2405.00675), preprint / NeurIPS 2024 Workshop)把对齐建模成**两玩家常和博弈**,目标是偏好的 **Nash 均衡**:每轮用偏好模型估计自采样本间的胜率,做一步**乘性权重 / 二次型**更新逼近均衡策略。它不假设偏好可由单一标量奖励(BT)解释——这把我们引向 §5 的博弈式视角。

---

## 5. 探索与博弈式偏好优化 (Exploration & Game-Theoretic PO)

### 5.1 为什么要"博弈式":偏好可能不传递

BT / 奖励式假设**偏好概率由标量奖励之差解释**(每个响应一个标量奖励),因此偏好在期望意义上**可传递**。但真实人类偏好可能**不传递 (intransitive)**:出现 $A\succ B\succ C\succ A$ 的环,任何标量奖励都无法表达。**博弈式偏好优化**绕开这一假设:不找"最大化奖励"的策略,而找偏好两玩家博弈的**Nash 均衡**(最大化对所有对手的最小胜率)。

**Nash-LHF / Nash-MD**(Munos et al., ["Nash Learning from Human Feedback"](https://arxiv.org/abs/2312.00886), [arXiv:2312.00886](https://arxiv.org/abs/2312.00886), ICML 2024):先学一个**偏好模型** $\mathcal{P}(y\succ y'\mid x)$(而非奖励模型),再用 **Nash-MD**(镜像下降)迭代求**正则化博弈的 Nash 均衡**,在 tabular / 正则化设定下可证明收敛。它把 RLHF 从"奖励最大化"推广到"偏好博弈均衡",§4.2 的 SPPO 是同一思想下的自博弈实例。

### 5.2 主动探索:XPO

被动 on-policy 采样只是"从当前策略随机采",并**没有刻意去探索高潜力但不确定的区域**。**XPO**(Exploratory Preference Optimization, Xie et al., [arXiv:2405.21046](https://arxiv.org/abs/2405.21046), ICLR 2025)在 DPO 目标上**只加一项乐观奖励 (optimism bonus)**,鼓励策略去探索"隐式奖励可能很高、但当前不确定"的响应;借助 implicit $Q^\*$-approximation,在其理论假设下可证明 **sample-efficient**。一句话:XPO = 在线 DPO + 一行乐观探索项,把"碰运气采样"升级成"有方向地探索"。

> 💡 谱系:**离线 DPO**(被动用旧数据)→ **在线 DPO**(被动 on-policy 采样)→ **XPO**(主动探索)。设计良好的主动探索更能逃出"当前策略已经会的"那点分布。

### 5.3 主动查询 (Active Querying)

标注预算有限时,**选哪些 prompt / 哪些对去标**也能优化:优先标 **RM 不确定 / 信息增益大**的对,而非均匀标注。与 RM 的不确定性估计结合,详见 [reward-modeling-eval](cheatsheet-reward-modeling-eval.html)。

---

## 6. 实战配方与陷阱 (Practical Recipes & Pitfalls)

### 6.1 生产级迭代配方

- **Llama 3**(Grattafiori et al., ["The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783), [arXiv:2407.21783](https://arxiv.org/abs/2407.21783), preprint):后训练做**六轮**迭代,每轮 = 奖励建模 + 拒绝采样 + SFT + **DPO**;每轮的偏好数据用**上一轮的最好模型**生成、由人类标注——是生产规模的迭代 DPO 范例。
- **Tülu 3**(Lambert et al., ["Tülu 3: Pushing Frontiers in Open Language Model Post-Training"](https://arxiv.org/abs/2411.15124), [arXiv:2411.15124](https://arxiv.org/abs/2411.15124), COLM 2025):**SFT → DPO → RLVR** 的开放配方,其中 DPO 用大规模 **on-policy 偏好混合**(从策略模型采补全、再由 GPT-4 等模型标偏好)。RLVR 细节见 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html)。
- **Iterative Reasoning PO**(Pang et al., ["Iterative Reasoning Preference Optimization"](https://arxiv.org/abs/2404.19733), [arXiv:2404.19733](https://arxiv.org/abs/2404.19733), NeurIPS 2024):面向**推理 / CoT** 的迭代 DPO——以**答案是否正确**(可验证信号)定 $y_w$/$y_l$,并在损失里**额外加一项对 $y_w$ 的 NLL/SFT 项**抑制似然漂移(§2.1b),逐轮在 GSM8K / MATH 等上提升。是"迭代 DPO + 可验证信号 + chosen 锚"的样板。

### 6.2 回路里的特有陷阱

| 陷阱 | 机制 | 缓解 |
|------|------|------|
| **奖励 hacking / 过优化** | 代理 RM 被逐轮利用,Goodhart;离线时只暴露一次,迭代里**复合放大** | 逐轮刷新/重训 RM;KL 锚;留可验证/人审信号 |
| **长度爆炸** | RM / judge 偏好更长答案 → 回路放大长度漂移 | 长度归一(SimPO 式)、报告长度受控胜率 (LC) |
| **似然漂移复合** | 若出现似然漂移,$\log\pi_\theta(y_w)$ 被一起压低的效应会逐轮复合 | 加 chosen-NLL 项(IRPO/RPO);固定 SFT ref 锚 |
| **多样性坍塌** | on-policy 反复强化高分模式,采样多样性下降、信号变弱 | 提温/采更多候选;主动探索(XPO);定期注入新 prompt |
| **计算成本** | 每轮 = 采样 + 打分 + 训练,远贵于一次性离线 DPO | 控制轮数/每轮预算;复用上一轮采样 |

> 🚨 **SimPO**(Meng et al., [arXiv:2405.14734](https://arxiv.org/abs/2405.14734), NeurIPS 2024)的长度归一与去 $\pi_\text{ref}$ 设计,常被借来缓解迭代回路中的长度漂移;但 SimPO 本身是离线损失变体,完整对比见 [llm-post-training §7.4 / §7.6](cheatsheet-llm-post-training.html)。

### 6.3 在线 DPO vs 在线 RLHF (PPO / GRPO)

两者**都 on-policy**,区别在"奖励信号如何进入更新":

| 维度 | 在线 / 迭代 DPO | 在线 RLHF (PPO / GRPO) |
|:---|:---|:---|
| 奖励 | **隐式**(藏在 DPO 损失里),靠成对偏好 | **显式** reward,进策略梯度 |
| 价值网络 | 不需要 | PPO 需 critic;GRPO 用组内基线免 critic |
| 信用分配 | 序列级(整条响应一个对比) | 可做更细的(token/步级)credit + reward shaping |
| 工程复杂度 | 较低(无 RL 基础设施) | 较高(rollout + 优化器 + KL 控制) |
| 定位 | 介于离线 DPO 与完整 RLHF 之间 | 表达力最强、最灵活 |

> 💡 一句话定位:**在线 DPO 拿到了 RLHF 的 on-policy 红利,又保住了 DPO 的简单**;代价是放弃了 RLHF 的细粒度信用分配与奖励塑形灵活性。GRPO/PPO 细节见 [llm-post-training §8](cheatsheet-llm-post-training.html)。

---

## 7. 面试题 (Interview Questions)

### L1 — 基础 (Foundational)

---

<details>
<summary>Q1: 标准 DPO 是 on-policy 还是 off-policy?为什么?</summary>

**答：** **off-policy(离线)**。偏好数据集 $\mathcal{D}$ 在训练前一次性采自固定策略 $\mu$(通常 SFT 模型),训练中 $\pi_\theta$ 不断更新而 $\mathcal{D}$ 静止。一旦 $\pi_\theta$ 漂离 $\mu$,数据里的 $(y_w,y_l)$ 就不再覆盖当前策略真实会生成的输出——这就是 off-policy 的分布漂移。在线 / 迭代 DPO 的改动就是每轮用当前 $\pi_\theta^{(t)}$ 重采样,把它变 on-policy。

> **追问：** "迭代 DPO"和"在线 DPO"是一回事吗?
> 常混用,内核都是"用当前策略重采样偏好对"。细分时,"迭代"强调一轮采一轮训的离散外循环,"在线"可指更细粒度的边采边训;本页统一指 on-policy 偏好优化。

</details>

---

<details>
<summary>Q2: 迭代 DPO 的最小闭环是哪三步?</summary>

**答：** ① **生成**:对每个 prompt 从当前策略 $\pi_\theta^{(t)}$ 采 $K$ 条响应(on-policy);② **标注**:用 RM / LLM-judge / 人把它们排序、构造偏好对 $(y_w,y_l)$ 得到 $\mathcal{D}^{(t)}$;③ **更新**:$\pi_\theta^{(t+1)}\leftarrow\text{DPO-update}(\pi_\theta^{(t)},\mathcal{D}^{(t)})$。逐轮重复。关键就是第①步用**当前**策略采,而非固定数据。

> **追问：** 哪一步最贵?
> 通常是生成+标注:每轮要采样并打分(若用人/强模型标注更贵)。这正是在线相对离线的主要成本。

</details>

---

<details>
<summary>Q3: 在线 DPO 和在线 RLHF (PPO) 都 on-policy,主要区别是什么?</summary>

**答：** 区别在**奖励信号如何进入更新**。在线 DPO 的奖励是**隐式**的(藏在成对 DPO 损失里),不需要显式 reward、不需要 critic,序列级对比;PPO/GRPO 用**显式 reward** 进策略梯度,可做更细的(token/步级)信用分配与 reward shaping,但需要 RL 基础设施(PPO 还要 critic,GRPO 用组内基线免 critic)。在线 DPO 介于离线 DPO 与完整 RLHF 之间:拿到 on-policy 红利、保住 DPO 的简单。

> **追问：** 那为什么不总用 PPO?
> 工程复杂、调参敏感、成本高。在线 DPO 用更小代价拿到 on-policy 的相当一部分收益,是很多开源配方(如 Tülu-3 的 DPO 阶段)的折中。

</details>

---

### L2 — 中级 (Intermediate)

---

<details>
<summary>Q4: 什么是 likelihood displacement?on-policy 数据为什么能缓解?</summary>

**答：** DPO 只优化 chosen 与 rejected 的**对数比之差**,不约束 $\log\pi_\theta(y_w)$ 的绝对值。于是模型可以一边把 $y_l$ 压得更低、一边让 $y_w$ 的概率**也下降**(只要降得慢),margin 照样增大;被挤出的质量常流向**第三类 OOD 输出**——即"好答案也被一起压低"的退化。on-policy 数据让 $y_w$ 本就来自当前分布,且 IRPO 等会**额外加一项对 $y_w$ 的 NLL** 把它的绝对概率锚住,共同缓解漂移。

> **追问：** 只靠加 chosen-NLL 项够吗?
> 能显著缓解似然漂移,但解决不了分布漂移与过优化——后两者还得靠 on-policy 重采样。两类手段正交,常一起用。

</details>

---

<details>
<summary>Q5: Self-Rewarding LM 和 SPIN 都"自给自足",本质区别是什么?</summary>

**答：** **信号来源不同**。Self-Rewarding 让模型**自己当裁判**给自采响应打分(LLM-as-judge),偏好对里**胜负都来自模型生成**,理论上能随模型变强而超越初始数据。SPIN **不打分也不需偏好标签**:固定用 **SFT 人类数据当胜者**、模型自采当负者,做判别式自博弈,收敛到"生成与人类数据不可区分"。因此 SPIN 的固定点是 **SFT 数据分布**(无外部奖励时无法推到该分布之外),Self-Rewarding 则可能突破(但有自我偏好放大的风险)。

> **追问：** 两者各自的主要风险?
> Self-Rewarding:裁判与策略同源 → 奖励 hacking / 自我偏好被回路放大、可能饱和。SPIN:受 SFT 数据质量与覆盖限制,数据差则上限低。

</details>

---

<details>
<summary>Q6: 迭代回路里 π_ref 应该每轮重置还是固定 SFT?各自的代价?</summary>

**答：** **每轮重置** $\pi_\text{ref}\leftarrow\pi_\theta^{(t)}$ 类似信赖域步,KL 约束相对上一轮、更新更稳,但**丢掉对 SFT 的锚定**,多轮后可能整体漂走。**固定 SFT** 始终锚定起点,但 $\pi_\theta$ 越漂,$\pi_\theta/\pi_\text{ref}$ 越大、有效约束越松,后期易过优化。生产实践多在**一轮内固定 ref**,轮间再换基准;$\beta$ 同时调节贴 ref 的强度(大=保守、小=易漂)。

> **追问：** 这和 PPO 的 KL 控制是一回事吗?
> 精神一致(都在限制策略别离参考太远),但 DPO 的 KL 是隐式地编码在 $\beta\log(\pi_\theta/\pi_\text{ref})$ 里,PPO 是显式 KL 罚项/裁剪。详见 [llm-post-training §9.3](cheatsheet-llm-post-training.html)。

</details>

---

<details>
<summary>Q7: Tang et al. 2024 关于"在线为何优于离线"的核心结论是什么?</summary>

**答：** 在其受控研究中:① 在线**稳定优于**离线,且差距**不能**靠给离线喂更多数据/扩覆盖补齐;② 差距**不是**由离线方法的判别精度不足或损失函数形式(对比式离线 vs 在线 RL)决定的——即便用强对比损失,差距仍在;③ 核心归因是 **on-policy 采样本身**(数据由当前策略生成)是关键驱动力,而非"在线/离线算法"这一标签。启示:**让数据 on-policy 比换损失更重要**,你不必弃用 DPO,只需把它的数据来源改成 on-policy。

> **追问：** 能据此说"在线在任何任务都更好"吗?
> 不能。这是其特定设置下的结论,且在线有真实代价(采样/标注/训练成本、奖励 hacking)。要结合任务与预算权衡。

</details>

---

### L3 — 高级 (Advanced)

---

<details>
<summary>Q8: 为什么要"博弈式"偏好优化 (Nash)?它解决了 BT/奖励式的什么前提缺陷?</summary>

**答：** BT/奖励式假设偏好概率由**标量奖励之差**解释,因此偏好(在期望意义上)**可传递**。但真实人类偏好可能**不传递**($A\succ B\succ C\succ A$ 的环),此时不存在能解释偏好的标量奖励。博弈式(NLHF / Nash-MD, Munos et al.)改为学**偏好模型** $\mathcal{P}(y\succ y'\mid x)$,求两玩家博弈的 **Nash 均衡**(最大化对所有对手的最小胜率),无需传递性假设,在 tabular / 正则化设定下可证明收敛到博弈均衡。SPPO 是同一思想下的自博弈实例。

> **追问：** Nash 均衡策略与"奖励最大化策略"在偏好可传递时是否一致?
> 当偏好恰好由 BT 奖励诱导(可传递)时,两者趋于一致;Nash 框架是更一般的超集,在不可传递时仍良定义,这正是它的价值。

</details>

---

<details>
<summary>Q9: 被动 on-policy 采样和 XPO 的"主动探索"差在哪?为什么探索能带来样本效率?</summary>

**答：** 被动 on-policy 只是"从当前策略随机采",采样集中在策略**已经会**的高概率区,对"没见过但可能更好"的响应**不会刻意去试**;迭代久了多样性坍塌、信号变弱。XPO(Xie et al. 2024)在 DPO 目标上加**一项乐观奖励 (optimism bonus)**,主动偏向"隐式奖励可能高、但当前不确定"的响应,借 implicit $Q^\*$-approximation 在其理论假设下可证明**样本高效**。直觉:探索把标注预算花在**信息增益大**的地方(不确定区),而非反复确认已知的好答案——这正是 RL 中"乐观面对不确定性"带来效率增益的经典逻辑。

> **追问：** 不加探索项的在线 DPO 会怎样退化?
> 容易陷入"自我确认":反复强化当前高分模式 → 采样多样性下降 → 新偏好对区分度变低 → 提升停滞。主动探索或定期注入新 prompt/提温可对冲。

</details>

---

<details>
<summary>Q10: 迭代回路里,奖励 hacking 与长度爆炸为何比一次性离线 DPO 更危险?如何系统性缓解?</summary>

**答：** 离线只暴露代理 RM **一次**;迭代回路里,每轮都用 RM/judge 标注再训练,策略会**逐轮**朝"RM 高估方向"漂,误差**复合放大**(Goodhart):RM 偏好长答案 → 回路放大长度漂移;RM 的系统偏置 → 被反复利用。缓解是组合拳:① **逐轮刷新/重训 RM**(别让策略追着一个静态代理跑);② **KL 锚 + 固定 SFT ref** 限制单轮漂移;③ **长度归一 / 报告长度受控胜率 (LC)** 治长度;④ 保留**可验证信号 / 人审**作真值兜底(IRPO 用答案正确性);⑤ 加 **chosen-NLL** 防似然漂移复合。

> **追问：** 为什么"可验证信号"在迭代回路里特别值钱?
> 规则验证器(exact-match/单测)≈ ground truth,在定义良好的任务上**更难被 hack**(但仍需防数据泄漏与规格漏洞),能切断"代理 RM 被逐轮利用"的复合放大链条——这也是 RLVR / IRPO 在推理任务上偏好可验证奖励的原因。

</details>

---

<details>
<summary>Q11: 给定只有一个固定的离线偏好集,你能逼近 on-policy 的收益吗?有哪些手段及其上限?</summary>

**答：** 只能**部分逼近**,无法完全替代。手段:① **RSO**——用拒绝采样从 $\pi_\text{ref}$ 近似采出贴近最优策略 $\pi^\*$ 的样本再标注,把数据来源往 on-policy 理想纠偏;② 去 $\pi_\text{ref}$ / chosen-anchoring(SimPO、加 chosen-NLL)缓解似然漂移;③ 拉大 $\beta$ 限制漂移避免在旧分布外过优化。**上限**:这些都改不了"数据由旧策略生成"这一根本事实——一旦 $\pi_\theta$ 漂到离线集未覆盖的区域,就没有任何监督。Tang et al. 的结论正点明:**on-policy 采样本身**是离线手段补不齐的关键。

> **追问：** 那离线 DPO 还有何不可替代的价值?
> 便宜、可复现、无在线基础设施、适合冷启动/资源受限场景;很多生产配方先离线 DPO 打底,再叠加少量在线/迭代轮次,是性价比折中。

</details>

---

## 附录：关键术语速查 (Glossary)

| 英文术语 | 中文 | 简要定义 |
|----------|------|----------|
| Offline DPO | 离线 DPO | 偏好数据一次采自固定 μ;off-policy |
| Online / Iterative DPO | 在线 / 迭代 DPO | 每轮用当前 π_θ 重采样偏好对;on-policy |
| On-policy / Off-policy | 同策略 / 异策略 | 数据是否来自当前正在优化的策略 |
| Distribution Mismatch | 分布漂移 | π_θ 漂离 μ 后,旧数据不再覆盖其输出 |
| Likelihood Displacement | 似然漂移 | margin 增大但 logπ(y_w) 反而下降 |
| Over-optimization | 过优化 | 朝隐式奖励虚高的 OOD 方向漂 |
| OAIF | 在线 AI 反馈 | LLM-judge 在线即时标注 on-policy 对 |
| LLM-as-judge | 模型当裁判 | 用强模型成对判优劣替代 RM/人 |
| RSO | 统计拒绝采样 | 拒绝采样逼近最优策略分布再标注 |
| Self-Rewarding | 自我奖励 | 同一模型既当策略又当裁判打分 |
| SPIN | 自博弈微调 | 人类数据=胜、模型自采=负的判别式自博弈 |
| SPPO | 自博弈偏好优化 | 求偏好的 Nash 均衡的自博弈更新 |
| NLHF / Nash-MD | Nash 学习 | 学偏好模型、用镜像下降求博弈均衡 |
| Intransitive Preference | 不传递偏好 | A≻B≻C≻A,无标量奖励可表达 |
| XPO | 探索式偏好优化 | DPO + 乐观探索项,样本高效 |
| Optimism Bonus | 乐观奖励项 | 鼓励探索不确定/高潜力响应 |
| IRPO | 迭代推理偏好优化 | 可验证信号定胜负 + chosen-NLL 锚 |
| Reward Hacking | 奖励 hacking | 利用代理 RM 缺陷刷分(回路里复合放大) |
| LC win-rate | 长度受控胜率 | 扣除长度偏置后的胜率口径 |

---

*本手册仅供学习参考。涉及的论文结论与数值以原始论文为准;benchmark 分数仅用于说明,不构成横向比较。*

## §A 核心论文时间线 / Key Papers Timeline

- **2023-05 · Direct Preference Optimization: Your Language Model is Secretly a Reward Model** — Rafailov et al., NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — 把 RLHF 的奖励最大化重参数化为对策略的成对分类损失,免去显式 RM 与 RL;**离线 DPO** 是本页所有在线/迭代方法的基线。

- **2023-09 · Statistical Rejection Sampling Improves Preference Optimization** — Liu et al., ICLR 2024. [arXiv:2309.06657](https://arxiv.org/abs/2309.06657) — 指出理想偏好对应采自最优策略分布,用拒绝采样从 π_ref 近似采出贴近 π* 的样本再标注(RSO),把离线数据来源往 on-policy 理想纠偏。

- **2023-12 · Nash Learning from Human Feedback** — Munos et al., ICML 2024. [arXiv:2312.00886](https://arxiv.org/abs/2312.00886) — 学偏好模型而非奖励模型,用 Nash-MD(镜像下降)求正则化博弈的 Nash 均衡,把 RLHF 从奖励最大化推广到偏好博弈,无需 BT 传递性假设。

- **2024-01 · Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models** — Chen et al., ICML 2024. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) — SPIN:以 SFT 人类数据为胜者、模型自采为负者做判别式自博弈,无需偏好标签或外部奖励;收敛于生成与数据不可区分,固定点为 SFT 数据分布。

- **2024-01 · Self-Rewarding Language Models** — Yuan et al., ICML 2024. [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) — 同一模型既当策略又用 LLM-as-judge 给自采响应打分构造偏好对,迭代 DPO;裁判能力随策略能力同步提升(M1→M2→M3 三轮)。

- **2024-02 · Direct Language Model Alignment from Online AI Feedback** — Guo et al., preprint. [arXiv:2402.04792](https://arxiv.org/abs/2402.04792) — OAIF:每步对当前策略采的两条响应,用一个 online annotator LLM 即时判优劣再做 DPO,把静态偏好集替换成在线 AI 反馈,兼顾 on-policy 与免训 RM。

- **2024-04 · Iterative Reasoning Preference Optimization** — Pang et al., NeurIPS 2024. [arXiv:2404.19733](https://arxiv.org/abs/2404.19733) — 面向 CoT 推理的迭代 DPO:以答案正确性(可验证信号)定胜负,损失里加对 chosen 的 NLL 项抑制似然漂移,逐轮在 GSM8K/MATH 等提升。

- **2024-05 · Self-Play Preference Optimization for Language Model Alignment** — Wu et al., preprint / NeurIPS 2024 Workshop (AFM, Oral). [arXiv:2405.00675](https://arxiv.org/abs/2405.00675) — SPPO:把对齐建模为两玩家常和博弈,用偏好胜率做乘性权重/二次型更新逼近 Nash 均衡,不依赖单一标量奖励 (BT)。

- **2024-05 · Understanding the Performance Gap between Online and Offline Alignment Algorithms** — Tang et al., preprint. [arXiv:2405.08448](https://arxiv.org/abs/2405.08448) — 受控研究:在线稳定优于离线,差距补不齐也不由损失形式决定,核心归因是 **on-policy 采样本身**——让数据 on-policy 比换损失更重要。

- **2024-05 · Exploratory Preference Optimization: Harnessing Implicit Q\*-Approximation for Sample-Efficient RLHF** — Xie et al., ICLR 2025. [arXiv:2405.21046](https://arxiv.org/abs/2405.21046) — XPO:在 DPO 目标上加一项乐观探索奖励,鼓励探索高潜力不确定响应,借 implicit Q*-approximation 在其理论假设下证明样本高效;把被动 on-policy 升级为主动探索。

- **2024-07 · The Llama 3 Herd of Models** — Grattafiori et al., preprint. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) — 后训练做六轮迭代(每轮 RM + 拒绝采样 + SFT + DPO),偏好数据用上一轮最好模型生成、人类标注;生产规模的迭代 DPO 范例。

- **2024-11 · Tülu 3: Pushing Frontiers in Open Language Model Post-Training** — Lambert et al., COLM 2025. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) — 开放端到端配方 SFT → DPO → RLVR,DPO 阶段用大规模 on-policy 偏好混合;全量公开数据/代码/配方。
