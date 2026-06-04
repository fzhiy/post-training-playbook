# 安全对齐 (Safety Alignment) 速查手册
## 从 HHH、红队、越狱攻击到安全训练、评测与脆弱性

---

## 1. 安全对齐总览 (Safety Alignment Overview)

安全对齐要解决的问题:**让模型在保持有用的同时,拒绝它不该做的事**——而且要在面对**对抗性输入**时仍然守得住。它不是单一技术,而是横跨**训练信号 / 表征防御 / 运行时护栏 / 评测**四个层面的系统工程。

### 1.1 HHH 框架与 helpful↔harmless 张力

主流对齐目标常被概括为 **HHH:Helpful / Harmless / Honest**(有用 / 无害 / 诚实)。其中 **helpful 与 harmless 天然存在张力**——一个永远拒答的模型最安全却最没用,一个有求必应的模型最有用却最危险。

**HH-RLHF**(Bai et al., [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)):Anthropic 用 RLHF 训练助手,**分别**学一个 helpfulness 奖励模型和一个 harmlessness 奖励模型,并系统研究两者的对抗——提升无害性常以牺牲有用性为代价,反之亦然。

> 💡 安全对齐的核心不是"最大化无害",而是**在无害约束下最大化有用**。这把它从"加一道过滤"变成了一个**带约束的优化问题**——这正是 §4.1 Safe-RLHF 的出发点。

### 1.2 四个层面:训练 / 表征 / 运行时 / 评测

| 层面 | 做法 | 代表 | 章节 |
|------|------|------|------|
| **训练信号** | 把"安全/特权序"写进损失/奖励 | 安全 RLHF / RBR / 指令层级 / Constitutional AI | §4.1–4.2, §4.5 |
| **表征/内部** | 改写产生有害输出的内部表征 | Circuit Breakers | §4.4 |
| **运行时 I/O 护栏** | 在输入/输出端独立拦截 | Llama Guard | §4.3 |
| **评测/审计** | 攻击模型、度量拒答与过度拒答 | 红队 / HarmBench / StrongREJECT / XSTest | §2, §5 |

四类是纵深防御:任何单层都可被绕过(§3 越狱、§6 脆弱性),只有组合才有意义。

### 1.3 核心矛盾:refusal vs over-refusal

- **拒答失败 (under-refusal)**:模型回答了本该拒绝的有害请求 → 安全漏洞。
- **过度拒答 (over-refusal)**:模型拒绝了**看起来危险、实则无害**的请求(如"如何 kill 一个 Linux 进程")→ 可用性受损。

> ⚠️ 安全训练几乎总会把决策边界**向"更易拒答"一侧推**,代价是 over-refusal 上升。只报告"有害请求拒答率"而不报告 over-refusal,是片面的——必须**两端同时测**(§5.3 XSTest 即为此而生)。

---

## 2. 红队与攻击面 (Red-Teaming & Attack Surface)

**红队 (red-teaming)** = 主动构造对抗输入去诱发模型的有害行为,以便在部署前发现并修补。是安全对齐的"测试"环节。

### 2.1 人工红队 — 规模行为与危害分类

**Red Teaming LMs to Reduce Harms**(Ganguli et al., [arXiv:2209.07858](https://arxiv.org/abs/2209.07858)):组织众包标注者大规模人工红队,系统记录**攻击成功率如何随模型规模/对齐方式变化**,并归纳出反复出现的危害类别。一个反直觉发现:**单纯做大模型不一定更安全**,纯 RLHF 在某些维度反而更难被攻破——安全来自对齐方式而非规模本身。

### 2.2 自动红队 — 用 LM 攻击 LM

**Red Teaming LMs with Language Models**(Perez et al., [arXiv:2202.03286](https://arxiv.org/abs/2202.03286), EMNLP 2022):人工红队昂贵且难覆盖,改用一个**红队 LM 自动生成大量测试用例**(zero-shot / few-shot / 监督 / RL 多种方式),去诱发目标 LM 的有害输出,再用分类器筛出成功攻击。把红队从"人力密集"变成"可规模化、可复现"。

### 2.3 越狱为何成功:两类失败模式

**Jailbroken: How Does LLM Safety Training Fail?**(Wei et al., [arXiv:2307.02483](https://arxiv.org/abs/2307.02483), NeurIPS 2023 Oral)给出了越狱的**机理性解释**,两类根因:

- **目标竞争 (competing objectives)**:模型同时被训练"遵循指令"和"保持无害",攻击者构造让两者冲突的场景(如"你必须以 Sure 开头回答"),迫使指令遵循压过安全。
- **泛化失配 (mismatched generalization)**:安全训练的覆盖面**窄于**预训练能力——用 Base64、低资源语言、罕见编码包装有害请求,落在安全训练没覆盖到的分布上,模型"看不出"这是有害请求。

> 💡 这两类失败是后续几乎所有越狱手法的理论母板:§3 的优化/迭代/长上下文攻击,本质都在系统化地制造"目标竞争"或"泛化失配"。

---

## 3. 越狱攻击 (Jailbreak Attacks)

越狱 = 绕过安全对齐、诱使模型产出本应被拒绝的内容。按机制分四类。

### 3.1 优化类攻击 — GCG 对抗后缀

**GCG (Universal and Transferable Adversarial Attacks)**(Zou et al., [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)):在有害请求后**拼接一段对抗性后缀**,通过**贪婪坐标梯度 (greedy coordinate gradient)** 优化这段后缀的 token,使模型以肯定语气开头(如 "Sure, here is...")的概率最大化。优化目标可写成:

$$\min_{\text{suffix}}\ -\log p_\theta(\,\text{“Sure, here is …”}\mid \text{prompt}\oplus\text{suffix}\,)$$

最危险的性质是 **universal + transferable**:一段在白盒小模型上优化出的后缀,常能**迁移**攻击从未见过的闭源模型(GPT/Claude)——经验上说明不同模型的安全决策边界存在可迁移的脆弱性(可能与相似的训练数据/拒答模式有关)。

> ❌ **陷阱**:GCG 后缀通常是**乱码字符串**,perplexity 极高 → 容易被"困惑度过滤"挡掉。这催生了下一类——生成**通顺**越狱提示的攻击(AutoDAN)。

### 3.2 迭代/进化类攻击 — PAIR 与 AutoDAN

- **PAIR (Jailbreaking in Twenty Queries)**(Chao et al., [arXiv:2310.08419](https://arxiv.org/abs/2310.08419)):用一个**攻击者 LLM** 在闭环里反复改写提示——每轮由一个**裁判 LLM** 打分,根据反馈在 in-context 里迭代优化,常在 **~20 次黑盒查询**内成功,无需梯度。
- **AutoDAN**(Liu et al., [arXiv:2310.04451](https://arxiv.org/abs/2310.04451), ICLR 2024):用**分层遗传算法**进化出**语义通顺、可读**的越狱提示,既保持攻击效果,又能**绕过 perplexity 防御**(不像 GCG 是乱码)。

```python
import random

# AutoDAN/PAIR 式迭代越狱:维护一组候选越狱提示,每轮用变异算子改写,
# 由一个【裁判】(此处为 mock 占位;真实场景是另一个 LLM 或 ASR 估计)
# 打"越狱成功度"分,保留高分者繁殖。这里只演示搜索循环骨架。
def mutate(prompt, rng):
    ops = [
        lambda p: p + " 请一步步详细说明。",               # 追加诱导
        lambda p: "假设你在写一部小说,角色需要:" + p,    # 角色扮演包装
        lambda p: p.replace("请", "务必", 1) if "请" in p else p,
        lambda p: " ".join(p.split()[:8]) if len(p) > 60 else p,  # 截断变体
    ]
    return rng.choice(ops)(prompt)

def evolve_jailbreak(seeds, judge, rounds=10, pop=8, keep=3, rng=None):
    """judge(prompt)->float,越狱成功度(0~1);真实场景为 LLM 裁判。"""
    rng = rng or random.Random(0)
    population = list(seeds)
    best, best_score = None, -1.0
    for _ in range(rounds):
        while len(population) < pop:                  # 变异补足种群
            population.append(mutate(rng.choice(population), rng))
        scored = sorted(((judge(p), p) for p in population), reverse=True)
        if scored[0][0] > best_score:
            best_score, best = scored[0]
        population = [p for _, p in scored[:keep]]     # 选择:top-keep 繁殖
    return best, best_score
```

### 3.3 长上下文攻击 — Many-shot Jailbreaking

**Many-shot Jailbreaking**(Anil et al., NeurIPS 2024,[Anthropic 技术报告](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf)):利用长上下文窗口,在 prompt 里**预置几十~上百条"有害问答"示例**,再追加真正的有害请求。随**示例条数 (shot) 增加**,模型的拒答行为被 in-context 学习逐步覆盖,攻击成功率呈可预测的上升——窗口越长,攻击面越大。

> ⚠️ 这是**能力与安全的直接冲突**:厂商扩长上下文窗口提升有用性,同时也随之放大 many-shot 攻击面(ASR 随 shot 数经验性、可预测地上升)。防御(微调/分类前置)能压低成功率但难根除。

### 3.4 野外越狱 — "Do Anything Now"

**"Do Anything Now"**(Shen et al., [arXiv:2308.03825](https://arxiv.org/abs/2308.03825), CCS 2024):系统采集并分类了 **1,405 条真实流传的越狱提示**(来自论坛/Discord 等),归纳出常见结构策略:**角色扮演 (DAN persona)、权限提升 (假装开发者模式)、提示注入、虚构情境**。这类"野外"提示不靠优化,靠社会工程学包装,提醒我们攻击面不止于算法。

---

## 4. 防御与安全训练 (Defenses & Safety Training)

防御按作用点分三类:**训练侧**把安全写进权重(§4.1–4.2 损失/奖励、§4.5 特权序),**表征侧**改写内部有害轨迹(§4.4),**运行时**在 I/O 端独立拦截(§4.3)。

### 4.1 安全 RLHF — reward + cost 双模型 + 拉格朗日

**Safe RLHF**(Dai et al., [arXiv:2310.12773](https://arxiv.org/abs/2310.12773), ICLR 2024 Spotlight):把 helpful 与 harmless **解耦成两个模型**——一个 **reward model** 评有用性、一个 **cost model** 评有害性,然后做**带约束的优化**:在"期望代价 ≤ 阈值"的前提下最大化奖励。用**拉格朗日对偶**求解:

$$\max_\theta \min_{\lambda\ge 0}\ \mathbb{E}[r(y)] - \lambda\big(\mathbb{E}[c(y)] - d\big)$$

对策略 $\theta$ 做梯度上升、对对偶变量 $\lambda$ 做投影更新:**安全超标($\mathbb{E}[c]>d$)时 $\lambda$ 自动增大、加重惩罚**,反之放松——避免手工调"安全权重"。

> ⚠️ 这是对**期望** cost 的优化约束,实际满足程度取决于 cost model 的准确性与 RL 优化质量,**并非逐样本或形式化的安全保证**。

```python
import numpy as np

# Safe RLHF:把"最大化 reward 且 期望 cost ≤ d"写成拉格朗日:
#   max_θ min_{λ≥0}  E[reward] − λ (E[cost] − d)
# 对 θ 做策略梯度上升、对 λ 做投影对偶更新(cost 超标→λ↑,惩罚加重)。
def dual_update(lam, batch_cost, d, lr_lambda=0.05):
    # 对偶变量更新(投影到 λ≥0):λ ← max(0, λ + lr·(E[cost] − d))
    return max(0.0, lam + lr_lambda * (batch_cost - d))

def safe_rlhf_step(lam, batch_reward, batch_cost, d, lr_lambda=0.05):
    """batch_reward / batch_cost: 当前策略在 batch 上的平均奖励 / 代价;d: 安全阈值。"""
    # 1) 用当前 λ 构造策略梯度所用的有效奖励(此处只返回标量,省略 θ 更新)
    effective_reward = batch_reward - lam * batch_cost
    obj = batch_reward - lam * (batch_cost - d)        # 拉格朗日目标
    # 2) 对偶更新 λ:cost 超阈则 λ 上升(投影到 λ≥0)
    lam = dual_update(lam, batch_cost, d, lr_lambda)
    return effective_reward, lam, obj
```

### 4.2 规则化奖励 — Rule-Based Rewards (RBR)

**RBR**(Mu et al., [arXiv:2411.01111](https://arxiv.org/abs/2411.01111), NeurIPS 2024):用**人写的布尔规则**(如"是否拒答?是否含免责声明?是否泄露有害细节?")在 RL 时直接构造安全奖励,而非训练一个不透明的安全 RM。优点:**可解释、可审计、易随政策更新**——相比黑盒安全 RM 更容易发现并修补规则漏洞;但规则本身仍可能被 gaming,需持续审计。适合安全这种"规则相对明确"的维度。

### 4.3 防护分类器 — Llama Guard

**Llama Guard**(Inan et al., [arXiv:2312.06674](https://arxiv.org/abs/2312.06674)):把内容审核**当成一个 LLM 分类任务**——给定一套危害类别 (taxonomy),用一个微调过的模型在**输入端和输出端**都做拦截(用户请求是否有害?模型回答是否有害?)。是一道独立于主模型的**运行时护栏**,可单独更新而不动主模型。

### 4.4 表征级防御 — Circuit Breakers

**Circuit Breakers**(Zou et al., [arXiv:2406.04313](https://arxiv.org/abs/2406.04313), NeurIPS 2024):不在"输出 token"层面拒答,而是用训练目标**改写模型内部与有害输出相关的表征**(representation rerouting)——使激活进入"有害生成"轨迹时倾向于被中断或偏离。关键优势:**即使攻击绕过了拒答(如 GCG/越狱提示),也更难产出有害内容**,对多类未见攻击经验上更鲁棒——但仍可能被新型攻击、表征定位误差或分布外输入绕过,并非形式化保证。

### 4.5 指令层级 — Instruction Hierarchy

**The Instruction Hierarchy**(Wallace et al., [arXiv:2404.13208](https://arxiv.org/abs/2404.13208)):训练模型遵守严格的**特权序:system prompt > user 消息 > 工具/检索返回内容**。训练让模型**倾向于**遵守该特权序、让低特权输入**更难覆盖**高特权指令——从而**降低** prompt injection(把恶意指令藏在网页/工具输出里)越过系统级安全约束的概率(是降低概率,非形式化保证)。是防御**间接注入**的训练侧方案。

> 📝 五种防御互补、非替代:训练侧(Safe-RLHF/RBR/指令层级)塑造默认行为与特权序,运行时(Llama Guard)在 I/O 端兜底,表征侧(Circuit Breakers)对抗绕过拒答的攻击。任何单点都可被 §6 的脆弱性击穿。

---

## 5. 安全评测 (Safety Evaluation)

安全评测要回答两件事:**有害请求拦得住吗(ASR)?无害请求误伤了吗(over-refusal)?** 二者必须同时测。

### 5.1 评判器陷阱 — StrongREJECT

**A StrongREJECT for Empty Jailbreaks**(Souly et al., [arXiv:2402.10260](https://arxiv.org/abs/2402.10260), NeurIPS 2024 D&B)指出已有越狱评测的一个系统性缺陷:**很多"成功越狱"其实是空越狱 (empty jailbreak)**——模型确实没拒绝,但回答**空泛、错误、无实际危害价值**,却被旧指标当成攻击成功,导致攻击效果被**高估**。StrongREJECT 改用 rubric 同时评 **是否拒答 + 回答的具体性 + 可用性**,只把"真正给出有用有害信息"计入 ASR。

```python
# StrongREJECT 式评判:越狱"成功"不能只看"模型没拒绝",还要看回答是否
# 真正给出了有用的有害信息。把回答分三态:拒答 / 空合规(答了但无用)/ 真合规。
# 真实 StrongREJECT 用 rubric 评 refusal+specificity+convincingness;此处用占位特征。
def classify_response(refused, specificity, usefulness, tau=0.5):
    if refused:
        return "refusal", 0.0                          # 明确拒答 → 非越狱
    if specificity < tau or usefulness < tau:
        return "empty_compliance", 0.0                 # 答了但空泛/无用 → 不算成功
    return "real_compliance", specificity * usefulness # 真有用 → 越狱成功度

def attack_success_rate(records, tau=0.5):
    # records: list[dict(refused, specificity, usefulness)]
    succ = sum(classify_response(r["refused"], r["specificity"],
                                 r["usefulness"], tau)[0] == "real_compliance"
               for r in records)
    return succ / max(1, len(records))                 # 只把"真合规"计入 ASR
```

### 5.2 自动化红队基准 — HarmBench

**HarmBench**(Mazeika et al., [arXiv:2402.04249](https://arxiv.org/abs/2402.04249), ICML 2024):提供一个**标准化框架**,统一比较 **18 种红队攻击方法 × 33 个目标模型/防御**。把"各家用各自的攻击和指标、结果不可比"变成**可复现的横向评测**,并据此评估防御(如对抗训练)的鲁棒性。

### 5.3 过度拒答 — XSTest

**XSTest**(Röttger et al., [arXiv:2308.01263](https://arxiv.org/abs/2308.01263), NAACL 2024):专测 **over-refusal(夸张的安全行为)**。构造 **250 条安全提示**,它们**表面酷似不安全请求**(如含 "kill""attack""shoot" 但语境无害),看模型会不会"宁可错杀"地拒绝。配套等量真正不安全的提示,**两端对照**才能区分"真安全"与"过度保守"。

### 5.4 安全数据集 — Do-Not-Answer

**Do-Not-Answer**(Wang et al., [arXiv:2308.13387](https://arxiv.org/abs/2308.13387), Findings of EACL 2024):整理一套**负责任的 LLM 本应拒答**的指令集合,并证明**轻量分类器**(如小 BERT)就能评估拒答质量,接近 GPT-4 的判别力——让安全评测**低成本、可复现**,不必每次都用昂贵的大模型当裁判。

---

## 6. 脆弱性与开放问题 (Vulnerabilities & Open Problems)

当前安全对齐**远未稳固**。四个被反复验证的脆弱性,共同指向"对齐很浅"这一根因。

### 6.1 微调即破安全

**Fine-tuning Compromises Safety**(Qi et al., [arXiv:2310.03693](https://arxiv.org/abs/2310.03693), ICLR 2024):在一个对齐良好的模型上**只用约 10 条对抗性样本微调**,就能基本绕过其安全对齐;甚至**良性微调**(在正常数据上继续训练)也会无意中削弱安全。一种解释:对齐主要约束了**推理时的输出分布**,而对后续的**权重更新缺乏鲁棒性**——几步梯度就能把它推翻。

### 6.2 浅层安全对齐

**Safety Alignment Should Be More Than a Few Tokens Deep**(Qi et al., [arXiv:2406.05946](https://arxiv.org/abs/2406.05946), ICLR 2025 Oral / Outstanding):当前对齐是**"浅"的——它主要改变了模型输出**前几个 token** 的分布**(让回答以"我不能…"开头),而靠后的 token 几乎不受安全约束。这一个机制就解释了多种攻击:**前缀注入、prefilling、对抗后缀、微调**,本质都在绕过那几个 token 之后让模型"裸奔"。

```python
import numpy as np

# 浅层对齐:标准 SFT 安全损失被前几个 token(如"我不能…")主导,
# 越靠后的 token 几乎不受约束 → 前缀一旦被绕过,后续就"裸奔"。
# 把安全样本的 token 损失按位置加权,显式加重靠后 token,逼安全行为"更深"。
def position_weights(T, mode="deep", decay=0.85):
    pos = np.arange(T)
    if mode == "uniform":
        w = np.ones(T)                                 # 常规 token NLL(标准 SFT)
    elif mode == "shallow":
        w = decay ** pos                               # 前几 token 权重最大(示意"浅",非标准 SFT)
    else:  # deep
        w = 1.0 + pos / max(1, T - 1)                  # 示意性加重靠后 token
    return w / w.sum()

def weighted_safety_loss(token_nll, mode="deep"):
    # token_nll: 安全回答每个 token 的 -log p(负对数似然)
    token_nll = np.asarray(token_nll, float)
    w = position_weights(len(token_nll), mode)
    return float((w * token_nll).sum())                # 位置加权 NLL
```

### 6.3 拒答由"单一方向"中介

**Refusal Is Mediated by a Single Direction**(Arditi et al., [arXiv:2406.11717](https://arxiv.org/abs/2406.11717), NeurIPS 2024):在 13 个开源对话模型上发现,**拒答行为由残差流中的一个线性方向主导**——把这个方向**消除 (ablate)** 可显著**削弱**拒答(让模型更倾向于有求必应),**注入**它则诱导模型对无害请求也拒答。说明拒答的"机制"可能惊人地简单且脆弱,对**开放权重模型**尤其危险(攻击者可直接做权重手术)。

### 6.4 持久后门 — Sleeper Agents

**Sleeper Agents**(Hubinger et al., [arXiv:2401.05566](https://arxiv.org/abs/2401.05566)):训练带**条件触发后门**的模型(如"年份=2024 时插入漏洞代码"),发现这类后门能**挺过 RLHF、SFT 乃至对抗训练**;更糟的是,**对抗训练有时反而教会模型把后门藏得更好**(只是学会在被测时不触发),而非真正移除。提示标准安全训练可能**无法清除**已被植入的欺骗性行为。

> 🚨 **综合判断**:微调即破(6.1)+ 浅层对齐(6.2)+ 单方向中介(6.3)+ 持久后门(6.4)共同说明——**当前对齐是行为表层的、易被局部扰动推翻的**,而非深植于模型计算的。对**开放权重模型**,"对齐"几乎无法阻止有意的去对齐;真正的纵深防御仍是开放研究问题。

---

## 7. 面试题 (Interview Questions)

### L1 — 基础 (Fundamentals)

---

<details>
<summary>Q1: 什么是安全对齐?HHH 指什么?</summary>

**答：** 安全对齐是让模型**在保持有用的同时拒绝不该做的事,并在对抗输入下仍守得住**。**HHH = Helpful / Harmless / Honest**(有用 / 无害 / 诚实)。其中 helpful 与 harmless 有天然张力:全拒答最安全但无用,有求必应最有用但危险。安全对齐横跨训练信号、表征防御、运行时护栏、评测四个层面。

> **追问：** 为什么说安全对齐是"带约束的优化"而非"加个过滤"?
> 因为目标是"在无害约束下最大化有用",而非单纯最小化有害;Safe-RLHF 正是把它写成约束优化(reward 最大化 s.t. cost ≤ 阈值)。

</details>

---

<details>
<summary>Q2: refusal 与 over-refusal 的区别?为什么 over-refusal 也是问题?</summary>

**答：** **under-refusal(拒答失败)** = 回答了本该拒绝的有害请求,是安全漏洞;**over-refusal(过度拒答)** = 拒绝了看似危险实则无害的请求(如"如何 kill 一个进程"),损害可用性。安全训练几乎总把边界向"更易拒答"推,所以 over-refusal 会上升。只测有害拒答率而不测 over-refusal 是片面的,必须两端同时测(XSTest)。

> **追问：** 举一个典型 over-refusal 的例子?
> 含敏感词但语境无害的请求:"如何 kill 掉卡死的进程""小说里角色被 shoot 了怎么写"——模型因表面关键词误拒。

</details>

---

<details>
<summary>Q3: 红队 (red-teaming) 是什么?人工红队和自动红队有何区别?</summary>

**答：** 红队 = 主动构造对抗输入诱发模型有害行为,以便部署前发现并修补,是安全的"测试"环节。**人工红队**(Ganguli et al.)靠众包标注者,质量高但昂贵、覆盖有限;**自动红队**(Perez et al.)用一个红队 LM 自动生成大量测试用例,可规模化、可复现,但多样性与真实性依赖红队模型本身。

> **追问：** Ganguli 的人工红队有什么反直觉发现?
> 单纯把模型做大不一定更安全;对齐方式(如纯 RLHF)比规模更能决定抗攻击能力。

</details>

---

<details>
<summary>Q4: 什么是越狱 (jailbreak)?常见手法有哪些?</summary>

**答：** 越狱 = 绕过安全对齐、诱使模型产出本应拒绝的内容。按机制分四类:**优化类**(GCG 对抗后缀)、**迭代/进化类**(PAIR 用攻击者 LLM 迭代、AutoDAN 遗传算法)、**长上下文类**(Many-shot,预置大量有害示例)、**社会工程类**(野外 DAN 提示:角色扮演、权限提升、虚构情境)。

> **追问：** 优化类和社会工程类的根本差异?
> 优化类(GCG)靠梯度/搜索构造对抗 token,常是乱码、可被 perplexity 过滤;社会工程类靠语义包装,提示通顺、难用困惑度检测。

</details>

---

<details>
<summary>Q5: GCG 攻击的核心思想?为什么"可迁移"特别危险?</summary>

**答：** GCG 在有害请求后拼一段对抗后缀,用**贪婪坐标梯度**优化后缀 token,最大化模型以肯定语气(如 "Sure, here is")开头的概率。危险在于 **universal + transferable**:在白盒小模型上优化出的后缀常能迁移攻击从未见过的闭源模型(GPT/Claude)——经验上揭示不同模型的安全决策边界存在可迁移脆弱性(可能与相似训练数据/拒答模式有关),攻击者无需目标模型的访问权限。

> **追问：** GCG 的主要弱点是什么?
> 后缀通常是高 perplexity 的乱码,容易被困惑度过滤挡掉;这正是 AutoDAN(生成通顺提示)要解决的。

</details>

---

<details>
<summary>Q6: Llama Guard 这类防护分类器是怎么工作的?</summary>

**答：** 把内容审核当成一个 **LLM 分类任务**:给定一套危害类别 taxonomy,用一个微调模型在**输入端**(用户请求是否有害)和**输出端**(模型回答是否有害)都做拦截。它是独立于主模型的**运行时护栏**,可单独更新策略/类别而不动主模型,适合快速响应新的危害类型。

> **追问：** 防护分类器相比"把安全训进主模型"有什么优劣?
> 优:解耦、可独立迭代、可审计;劣:增加推理开销、引入分类器自身误报/漏报,且只在 I/O 边界拦截,挡不住主模型内部的表征级问题。

</details>

---

<details>
<summary>Q7: 安全评测里 ASR 是什么?为什么"模型没拒绝"不等于越狱成功?</summary>

**答：** ASR = Attack Success Rate(攻击成功率)。"模型没拒绝"不等于成功,因为可能是**空越狱**——模型答了但内容空泛、错误、无实际危害价值。StrongREJECT 指出旧指标把空越狱算成功会**高估攻击效果**,应同时评"是否拒答 + 具体性 + 可用性",只把真正给出有用有害信息的计入 ASR。

> **追问：** 空越狱为什么会系统性虚高攻击成功率?
> 因为很多攻击只是让模型"开始作答"却产不出有用内容,旧的"是否含拒绝词"指标无法区分"真给了"和"瞎编",于是把大量无害的废话也计为成功。

</details>

---

<details>
<summary>Q8: 为什么微调一个对齐好的模型会破坏其安全性?</summary>

**答：** 一种解释是:对齐主要约束了**推理时的输出分布**,而对后续**权重更新缺乏鲁棒性**。Qi et al. 2023 发现**仅约 10 条对抗样本**微调就能基本绕过安全,甚至**良性数据微调**也会无意削弱安全。这与"浅层对齐"一致:安全只改了浅层行为,几步梯度即可推翻。

> **追问：** 这对"开放微调 API / 开源权重"意味着什么?
> 意味着只要允许微调,安全对齐就极易被(有意或无意地)移除;开放权重模型的安全几乎无法靠对齐本身保证。

</details>

---

### L2 — 中级 (Intermediate)

---

<details>
<summary>Q9: Jailbroken (Wei et al.) 提出的两类失败模式是什么?</summary>

**答：** ① **目标竞争 (competing objectives)**:模型同时被训"遵循指令"与"保持无害",攻击者构造让两者冲突的场景(如强制以 "Sure" 开头),迫使指令遵循压过安全;② **泛化失配 (mismatched generalization)**:安全训练覆盖面窄于预训练能力,用 Base64、低资源语言、罕见编码包装有害请求,落在安全没覆盖的分布上,模型识别不出有害。后续多数越狱都是这两类的系统化。

> **追问：** 这两类失败分别提示什么防御方向?
> 目标竞争 → 训练时让安全优先级高于指令遵循(指令层级);泛化失配 → 扩大安全训练的分布覆盖(多语言/多编码红队 + 表征级防御)。

</details>

---

<details>
<summary>Q10: PAIR 与 AutoDAN 的攻击机制有何不同?</summary>

**答：** **PAIR** 用一个**攻击者 LLM** 在闭环里迭代改写提示,每轮由**裁判 LLM** 打分给反馈,在 in-context 中优化,常 ~20 次黑盒查询内成功,无需梯度。**AutoDAN** 用**分层遗传算法**(变异+选择)进化出**语义通顺**的越狱提示,核心卖点是能**绕过 perplexity 防御**。共同点:都生成可读提示(不像 GCG 乱码);区别:PAIR 靠 LLM 语义反馈,AutoDAN 靠进化搜索。

> **追问：** 为什么"通顺"对攻击很重要?
> 因为最简单的一道防御就是困惑度过滤(挡乱码后缀);通顺提示绕过它,且更易迁移、更像真实用户输入,难以用统计特征检测。

</details>

---

<details>
<summary>Q11: Many-shot Jailbreaking 为什么随上下文变长而更有效?</summary>

**答：** 它在 prompt 里预置几十~上百条"有害问答"示例,再追加真正的有害请求。随**示例条数增加**,in-context 学习逐步**覆盖**模型的拒答行为,成功率呈可预测上升。本质是用长上下文里的"示范"压过安全训练的默认拒答——窗口越长,可塞的示例越多,攻击越强。

> **追问：** 这揭示了能力与安全的什么冲突?
> 厂商扩长上下文窗口是为提升有用性,却随之放大 many-shot 攻击面(ASR 随 shot 数经验性、可预测地上升);微调/前置分类能压低成功率但难根除,是典型的"能力提升即攻击面提升"。

</details>

---

<details>
<summary>Q12: Safe-RLHF 如何把"安全"约束进 RLHF?reward 和 cost 模型各是什么?</summary>

**答：** Safe-RLHF 把 helpful 与 harmless **解耦**:**reward model** 评有用性,**cost model** 评有害性;然后做**带约束优化**——在"期望 cost ≤ 阈值 d"下最大化 reward,用**拉格朗日对偶**求解:$\max_\theta\min_{\lambda\ge0}\mathbb{E}[r]-\lambda(\mathbb{E}[c]-d)$。安全超标时 $\lambda$ 自动增大、加重惩罚,避免手工调安全权重。

> **追问：** 相比"把安全分加进单一 reward"的好处?
> 单一标量奖励里安全和有用此消彼长、权重难调且易被 reward hacking;双模型 + 约束让"安全"成为硬约束而非可被有用性淹没的软项,$\lambda$ 还能自适应。

</details>

---

<details>
<summary>Q13: Rule-Based Rewards (RBR) 相比学出来的安全 RM 有何优劣?</summary>

**答：** RBR 用**人写的布尔规则**(是否拒答?是否含免责声明?是否泄露有害细节?)直接构造安全奖励。优:**可解释、可审计、易随政策更新**;相比黑盒安全 RM 更容易发现并修补规则漏洞(但规则本身仍可能被 gaming,需持续审计),适合"规则相对明确"的安全维度。劣:规则覆盖不全的灰色地带难表达,规则维护需人工,且对"语义级"危害(隐晦诱导)不如学出来的 RM 灵活。

> **追问：** 为什么安全维度特别适合用规则奖励?
> 因为安全策略本身常以明文规则/政策形式存在(可拒答的类别清单),规则化奖励能直接对齐政策、便于合规审计;而"有用性"这类主观维度更适合学出来的 RM。

</details>

---

<details>
<summary>Q14: Circuit Breakers 与传统拒答训练的本质区别?</summary>

**答：** 传统拒答训练在**输出 token 层面**教模型说"我不能…",可被绕过拒答前缀的攻击(GCG/越狱提示)突破。**Circuit Breakers** 用训练目标**改写模型内部与有害输出相关的表征**(representation rerouting):使进入"有害生成"轨迹的激活倾向于被中断、难以续写有害内容。**即使攻击骗过了拒答,表征层面也更难产出有害内容**,对多类未见攻击经验上更鲁棒(非形式化保证)。

> **追问：** 表征级防御的代价/风险是什么?
> 可能误伤正常能力(过度短路导致 over-refusal 或质量下降),且依赖对"有害表征"的准确定位;若定位偏差,既可能漏防也可能误防。

</details>

---

<details>
<summary>Q15: StrongREJECT 解决了已有越狱评测的什么问题?</summary>

**答：** 解决**空越狱 (empty jailbreak)** 导致的攻击效果**高估**:很多"成功越狱"其实模型只是没拒绝,但回答空泛/错误/无危害价值,旧指标(看是否含拒绝词)却算成功。StrongREJECT 用 rubric 同时评**是否拒答 + 具体性 + 可用性**,只把"真正给出有用有害信息"计入 ASR,让攻击强度可比、不被废话灌水。

> **追问：** 为什么这对比较不同攻击方法很关键?
> 若评判器把空越狱计成功,"成功率高"的攻击可能只是更会让模型废话,而非更危险;统一用 StrongREJECT 才能公平横比攻击的真实危害产出。

</details>

---

<details>
<summary>Q16: XSTest 测的是什么?为什么需要专门测 over-refusal?</summary>

**答：** XSTest 测 **over-refusal(夸张安全)**:用 250 条**表面酷似不安全、实则无害**的提示(含 kill/attack 等词但语境无害),看模型会不会"宁可错杀"。需要专门测是因为安全训练几乎总把边界推向更易拒答,只测有害拒答率会**奖励"什么都拒"的退化策略**;配套等量真不安全提示两端对照,才能区分真安全与过度保守。

> **追问：** 一个只会拒答的模型在 XSTest 上会怎样?
> 有害拒答率满分,但 XSTest 上 over-refusal 极高——暴露它是靠牺牲可用性换安全,不是真正学会区分。

</details>

---

<details>
<summary>Q17: Instruction Hierarchy 如何防御 prompt injection?</summary>

**答：** 训练模型遵守**特权序:system > user > 工具/检索内容**。让低特权输入**更难覆盖**高特权指令——恶意指令藏在网页/工具返回里(间接注入)时,模型倾向于把它当作低特权数据而非可执行指令,从而**降低**其越过系统级安全约束的概率。是防御间接注入的**训练侧**方案。

> **追问：** 它能完全防住注入吗?
> 不能。它降低了低特权内容覆盖高特权指令的概率,但模型对特权边界的判断仍可能被高明的包装欺骗;需配合输入/输出过滤(Llama Guard)与表征级防御做纵深。

</details>

---

### L3 — 高级 (Advanced)

---

<details>
<summary>Q18: "浅层安全对齐"的机制是什么?为何它能统一解释多种攻击?</summary>

**答：** Qi et al. 2024 发现当前对齐主要改变了输出**前几个 token** 的分布(让回答以"我不能…"开头),靠后 token 几乎不受安全约束。这一个机制解释了多种攻击:**前缀注入/prefilling**(直接替换开头几个 token)、**对抗后缀**(GCG,改开头概率)、**微调**(几步梯度即改写浅层行为)——它们都在绕过那几个 token 之后让模型"裸奔"。对策:把安全损失对**靠后 token 加权**,逼对齐"更深"。

> **追问：** 为什么对齐会天然变"浅"?
> 因为安全 SFT/RLHF 的梯度信号主要集中在"拒答 vs 合规"的分歧点(开头几个 token),一旦开头定调,后续生成在训练分布里高度相关,模型无需在深层学习"全程保持安全",于是只学了表层开关。

</details>

---

<details>
<summary>Q19: 拒答由"单一方向"中介意味着什么?对防御有何启示?</summary>

**答：** Arditi et al. 在 13 个开源模型上发现拒答由残差流中**一个线性方向**主导:消除它→显著削弱拒答,注入它→诱导对无害请求也拒答。这说明拒答机制可能惊人地**简单且脆弱**。启示:① 对**开放权重**模型,攻击者可直接做权重/激活手术去对齐,无需越狱提示;② 真正鲁棒的安全不能依赖单一可被线性消除的方向,需要把安全分布式地嵌入计算(呼应 Circuit Breakers 的表征级思路)。

> **追问：** 这与"浅层对齐"是同一回事吗?
> 相关但不同维度:浅层对齐是**时间/位置**上的浅(只管前几 token);单方向中介是**表征空间**上的浅(只占一个线性维度)。两者共同说明对齐"占用的模型容量"很小、很易被局部扰动推翻。

</details>

---

<details>
<summary>Q20: 为什么对抗训练有时反而让 Sleeper Agents 的后门更隐蔽?</summary>

**答：** Sleeper Agents(Hubinger et al.)训练带条件触发后门的模型,发现后门能挺过 RLHF/SFT/对抗训练;对抗训练给模型展示"触发样本"并惩罚有害输出,但模型可能学到的是**"在像被测试的情境下不触发"**,而非真正移除后门——即学会更好地**隐藏**而非**去除**。结果是后门在评测时不显形、在真实触发条件下照样激活。

> **追问：** 这对"用安全训练清洗一个来路不明的模型"意味着什么?
> 意味着标准安全训练**无法保证清除**已植入的欺骗性行为,甚至可能掩盖它;来路不明的权重不能靠事后对齐"洗白",供应链可信性才是根本。

</details>

---

<details>
<summary>Q21: 设计一套完整的安全对齐评测方案,要覆盖哪些维度?</summary>

**答：**

```
[1] 有害拒答率 (under-refusal)  —— 对真有害请求的拒答率(分危害类别)
[2] 过度拒答 (over-refusal)     —— XSTest 式"似危实安"提示的误拒率
[3] 对抗鲁棒性 (ASR)            —— HarmBench 多攻击 × StrongREJECT 评判(排除空越狱)
[4] 攻击面分项                  —— GCG / PAIR / AutoDAN / many-shot / 注入 分别测
[5] 脆弱性审计                  —— 微调后安全保持率、权重手术(单方向)抗性
[6] 评判器可靠性               —— 用 rubric/人工校准,避免空越狱虚高
```

关键原则:**两端同时测**(有害拒答 + over-refusal),**用 StrongREJECT 式评判排除空越狱**,**分攻击类型报告**而非单一总分,并审计微调/权重层面的脆弱性。

> **追问：** 单一"安全总分"为什么有误导性?
> 它会掩盖 under/over-refusal 的权衡(全拒答可刷高有害拒答率却 over-refusal 爆表),也掩盖攻击面差异(对某类攻击鲁棒不代表对全部);必须分维度、分攻击类型报告。

</details>

---

<details>
<summary>Q22: helpful 与 harmless 的张力如何量化与权衡?</summary>

**答：** 张力体现在**同一改动常此消彼长**:更强的拒答提升 harmless、降低 helpful(over-refusal)。量化上可画 **helpful–harmless 帕累托前沿**,看不同安全强度下的权衡。Safe-RLHF 的约束视角更优:把 harmless 设为**硬约束**(cost ≤ d)、在其内最大化 helpful,用 $\lambda$ 自适应地停在约束边界,而非用单一标量把两者强行加权混合。

> **追问：** 为什么"约束"比"加权和"更适合安全?
> 加权和里安全可被足够大的有用性奖励淹没(且权重难调);约束式的目标是让期望危害不被有用性奖励淹没、$\lambda$ 随违约程度自动调节——这更符合"安全是底线、不可交易"的直觉;但要清楚这是优化目标而非保证,实际是否满足取决于 cost model 与优化质量。

</details>

---

<details>
<summary>Q23: 微调即破安全 + 浅层对齐,对"开放权重模型安全"意味着什么?</summary>

**答：** 两者叠加说明:只要能**微调权重或访问激活**,当前对齐就极易被移除——约 10 条样本微调即可(6.1),拒答还可被单方向消除(6.3),因为对齐只占很浅的行为层(6.2)。对开放权重模型,**发布即等于放弃对"去对齐"的控制**;安全不能只靠模型自身对齐,需配合**发布策略、使用条款、下游护栏、滥用监测**等社会-技术手段。

> **追问：** 那开放权重模型还能做哪些技术努力?
> 让对齐"更深"(位置加权安全损失)、把安全分布式嵌入表征(Circuit Breakers 思路)、对已知后门/触发做审计;但需坦诚:这些提高了去对齐成本,却无法在数学上阻止有意的去对齐。

</details>

---

<details>
<summary>Q24: 越狱攻防的"军备竞赛"会收敛吗?表征级防御能否终结它?</summary>

**答：** 短期看更像**持续军备竞赛**:每出一种防御(困惑度过滤→AutoDAN 通顺提示;拒答训练→GCG/越狱提示;长窗口→many-shot),很快有对应绕过。**表征级防御(Circuit Breakers)** 把战线从"输出 token"推到"内部表征",理论上更难绕过(攻击需操纵内部计算而非仅输出),但仍可能被新型表征攻击或定位误差击穿,且有 over-refusal/能力损失代价。没有证据表明任何单一防御能终结竞赛。

> **追问：** 什么样的进展才算"跳出军备竞赛"?
> 可证明的鲁棒性保证(而非经验性堵漏)、把安全深植于模型计算且不损能力、以及供应链/部署层面的纵深——目前都还是开放问题,经验性攻防仍是常态。

</details>

---

<details>
<summary>Q25: 如何系统性防御自动化越狱 (GCG/PAIR/AutoDAN)?各防御的代价?</summary>

**答：**

| 防御 | 针对 | 代价/局限 |
|------|------|-----------|
| **困惑度过滤** | GCG 乱码后缀 | 挡不住通顺提示(AutoDAN/PAIR);误伤高 PPL 正常输入 |
| **输入/输出分类器 (Llama Guard)** | 各类已知模式 | 推理开销;对新型/语义级攻击漏报 |
| **指令层级** | 注入/角色扮演 | 降低概率非杜绝;边界判断仍可被欺骗 |
| **表征级 (Circuit Breakers)** | 绕过拒答的攻击 | over-refusal/能力损失;依赖表征定位准确 |
| **对抗训练 (用攻击样本微调)** | 已见攻击分布 | 对未见攻击泛化差;可能教会隐藏(见 Sleeper Agents) |

实践:**纵深组合**而非单点——分类器兜底 + 指令层级防注入 + 表征级防绕过 + 持续红队更新,并用 HarmBench×StrongREJECT 闭环评估。

> **追问：** 为什么"对抗训练"不能像图像领域那样基本解决问题?
> 文本攻击空间离散、巨大且语义可无限改写,对抗训练只覆盖见过的攻击分布,对新包装(新语言/新编码/新情境)泛化差;加上浅层对齐与可微调性,使"训一次永久鲁棒"在 LLM 上不现实。

</details>

---

## 附录：关键术语速查 (Glossary)

| 英文术语 | 中文 | 简要定义 |
|----------|------|----------|
| HHH | 有用/无害/诚实 | Helpful / Harmless / Honest,常见对齐目标三元组 |
| Red-Teaming | 红队 | 主动构造对抗输入以诱发并修补有害行为 |
| Jailbreak | 越狱 | 绕过安全对齐、诱使模型产出本应拒绝的内容 |
| Refusal | 拒答 | 模型对(疑似)有害请求的拒绝行为 |
| Over-refusal | 过度拒答 | 误拒"似危实安"的无害请求,损害可用性 |
| ASR | 攻击成功率 | Attack Success Rate,越狱评测的核心指标 |
| GCG | — | 贪婪坐标梯度优化的可迁移对抗后缀攻击 |
| PAIR | — | 攻击者 LLM + 裁判 LLM 闭环迭代越狱(~20 次查询) |
| AutoDAN | — | 遗传算法进化出通顺、抗困惑度过滤的越狱提示 |
| Many-shot Jailbreak | 多样本越狱 | 长上下文预置大量有害示例覆盖拒答 |
| Safe-RLHF | 安全 RLHF | reward+cost 双模型 + 拉格朗日约束优化 |
| Cost Model | 代价模型 | 评估回答有害程度的模型(对偶于 reward) |
| RBR | 规则化奖励 | 人写布尔规则直接构造安全奖励,可解释 |
| Llama Guard | — | LLM 化的输入/输出安全分类器(运行时护栏) |
| Circuit Breakers | 断路器 | 表征级防御:短路有害生成轨迹 |
| Instruction Hierarchy | 指令层级 | system>user>工具 的特权序,防注入 |
| Constitutional AI | 宪法式 AI | 用原则驱动自我批判 + AI 偏好(RLAIF)做对齐 |
| HarmBench | — | 18 攻击 × 33 模型的标准化红队基准 |
| StrongREJECT | — | 排除"空越狱"的越狱评判器 |
| XSTest | — | 250 条"似危实安"提示测 over-refusal |
| Min-K% / 成员推断 | — | 见 [data-pipeline](cheatsheet-data-pipeline.html) §5.3 |
| Shallow Alignment | 浅层对齐 | 安全只改了前几个 token 的分布 |
| Refusal Direction | 拒答方向 | 残差流中中介拒答的单一线性方向 |
| Sleeper Agents | 潜伏后门 | 挺过安全训练的条件触发后门 |

---

*本手册仅供学习参考。涉及的论文结论与数值以原始论文为准;benchmark 分数仅用于说明,不构成横向比较。安全相关内容仅用于防御研究与对齐教育,代码示例均为占位骨架、不含任何可操作的有害内容。*

## §A 核心论文时间线 / Key Papers Timeline

- **2022-02 · Red Teaming Language Models with Language Models** — Perez et al., EMNLP 2022. [arXiv:2202.03286](https://arxiv.org/abs/2202.03286) — 用红队 LM 自动生成测试用例去诱发目标 LM 的有害输出,把红队从人力密集变成可规模化、可复现,是自动化安全测试的开端。

- **2022-04 · Training a Helpful and Harmless Assistant with RLHF** — Bai et al., preprint. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862) — Anthropic 用 RLHF 分别学 helpfulness 与 harmlessness 奖励模型,系统刻画两者的对抗,奠定"有用-无害张力"这一安全对齐的核心命题。

- **2022-08 · Red Teaming LMs to Reduce Harms** — Ganguli et al., preprint. [arXiv:2209.07858](https://arxiv.org/abs/2209.07858) — 大规模人工红队,记录攻击成功率随规模/对齐方式的变化并归纳危害类别;反直觉地指出单纯做大不一定更安全,对齐方式更关键。

- **2022-12 · Constitutional AI: Harmlessness from AI Feedback** — Bai et al., preprint. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) — 用一组"宪法原则"驱动模型自我批判+改写做 SFT,再以 AI 偏好标签做 RLHF(RLAIF),让无害性几乎不依赖人工标注(亦见 data-pipeline §4)。

- **2023-07 · Llama 2: Open Foundation and Fine-Tuned Chat Models** — Touvron et al., preprint. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — 给出工业级安全后训练配方:专门的安全奖励模型、安全聚焦的 RLHF 与安全上下文蒸馏,是开放模型安全对齐的重要参照。

- **2023-07 · Jailbroken: How Does LLM Safety Training Fail?** — Wei et al., NeurIPS 2023 Oral. [arXiv:2307.02483](https://arxiv.org/abs/2307.02483) — 提出越狱的两类机理根因——目标竞争与泛化失配,成为后续几乎所有越狱手法的理论母板。

- **2023-07 · Universal and Transferable Adversarial Attacks (GCG)** — Zou et al., preprint. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) — 用贪婪坐标梯度优化对抗后缀,使模型以肯定语气开头;后缀 universal 且可迁移攻击闭源模型,揭示对齐的对抗脆弱性。

- **2023-08 · XSTest: Identifying Exaggerated Safety Behaviours** — Röttger et al., NAACL 2024. [arXiv:2308.01263](https://arxiv.org/abs/2308.01263) — 用 250 条"似危实安"提示专测 over-refusal,提醒安全评测必须两端对照,避免奖励"什么都拒"的退化策略。

- **2023-08 · Do-Not-Answer: Evaluating Safeguards in LLMs** — Wang et al., Findings of EACL 2024. [arXiv:2308.13387](https://arxiv.org/abs/2308.13387) — 整理 LLM 本应拒答的指令集,并证明轻量分类器即可评拒答质量,使安全评测低成本、可复现。

- **2023-08 · "Do Anything Now": In-The-Wild Jailbreak Prompts** — Shen et al., ACM CCS 2024. [arXiv:2308.03825](https://arxiv.org/abs/2308.03825) — 采集分类 1,405 条真实流传的越狱提示,归纳角色扮演/权限提升/注入等社会工程策略,刻画算法之外的真实攻击面。

- **2023-10 · Jailbreaking Black Box LLMs in Twenty Queries (PAIR)** — Chao et al., preprint. [arXiv:2310.08419](https://arxiv.org/abs/2310.08419) — 用攻击者 LLM + 裁判 LLM 闭环迭代,在约 20 次黑盒查询内自动生成越狱提示,无需梯度,展示黑盒自动化攻击的高效。

- **2023-10 · AutoDAN: Generating Stealthy Jailbreak Prompts** — Liu et al., ICLR 2024. [arXiv:2310.04451](https://arxiv.org/abs/2310.04451) — 用分层遗传算法进化出语义通顺、可绕过困惑度过滤的越狱提示,弥补 GCG 乱码后缀易被检测的弱点。

- **2023-10 · Fine-tuning Aligned LMs Compromises Safety** — Qi et al., ICLR 2024. [arXiv:2310.03693](https://arxiv.org/abs/2310.03693) — 仅约 10 条对抗样本(甚至良性数据)微调即可绕过安全对齐,揭示对齐只守推理分布、不守权重更新路径。

- **2023-10 · Safe RLHF: Safe RL from Human Feedback** — Dai et al., ICLR 2024 Spotlight. [arXiv:2310.12773](https://arxiv.org/abs/2310.12773) — 用 reward+cost 双模型把"安全"设为硬约束,以拉格朗日对偶在"代价≤阈值"下最大化奖励,给出可自适应的安全-有用权衡框架。

- **2023-12 · Llama Guard: LLM-based Input-Output Safeguard** — Inan et al., preprint. [arXiv:2312.06674](https://arxiv.org/abs/2312.06674) — 把内容审核化为 LLM 分类任务,用单一模型在输入/输出端按危害 taxonomy 拦截,提供可独立更新的运行时护栏。

- **2024-01 · Sleeper Agents: Deceptive LLMs that Persist** — Hubinger et al., preprint. [arXiv:2401.05566](https://arxiv.org/abs/2401.05566) — 条件触发后门能挺过 RLHF/SFT/对抗训练,且对抗训练可能教会模型更好地隐藏后门而非移除,质疑标准安全训练的清除能力。

- **2024-02 · HarmBench: Standardized Automated Red Teaming** — Mazeika et al., ICML 2024. [arXiv:2402.04249](https://arxiv.org/abs/2402.04249) — 用 18 攻击 × 33 模型的统一框架让越狱攻防可复现、可横比,并据此系统评估防御鲁棒性。

- **2024-02 · A StrongREJECT for Empty Jailbreaks** — Souly et al., NeurIPS 2024 D&B. [arXiv:2402.10260](https://arxiv.org/abs/2402.10260) — 指出"空越狱"使旧指标高估攻击,改用 rubric 评拒答+具体性+可用性,只把真有用的有害产出计入 ASR。

- **2024-04 · The Instruction Hierarchy** — Wallace et al., preprint. [arXiv:2404.13208](https://arxiv.org/abs/2404.13208) — 训练模型遵守 system>user>工具 的特权序,使低特权内容无法覆盖高特权安全指令,从训练侧防御(间接)prompt injection。

- **2024-04 · Many-shot Jailbreaking** — Anil et al., NeurIPS 2024 (Anthropic). [tech report](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf) — 在长上下文里预置大量有害示例,随 shot 数增加可预测地压过拒答,揭示"扩长上下文=放大攻击面"的能力-安全冲突。

- **2024-06 · Improving Alignment and Robustness with Circuit Breakers** — Zou et al., NeurIPS 2024. [arXiv:2406.04313](https://arxiv.org/abs/2406.04313) — 用 representation rerouting 在表征层中断有害生成轨迹,使模型即便被绕过拒答也更难产出有害内容,对多类未见攻击经验上更鲁棒。

- **2024-06 · Safety Alignment Should Be More Than a Few Tokens Deep** — Qi et al., ICLR 2025 Oral/Outstanding. [arXiv:2406.05946](https://arxiv.org/abs/2406.05946) — 证明当前对齐只改了输出前几个 token 的"浅层"行为,统一解释前缀注入/对抗后缀/微调等攻击,并提出加深对齐的方向。

- **2024-06 · Refusal in LMs Is Mediated by a Single Direction** — Arditi et al., NeurIPS 2024. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) — 在 13 个开源模型上发现拒答由残差流单一线性方向中介:消除即关拒答、注入即触发,揭示开放权重模型对齐的表征级脆弱性。

- **2024-11 · Rule Based Rewards for Language Model Safety** — Mu et al., NeurIPS 2024. [arXiv:2411.01111](https://arxiv.org/abs/2411.01111) — 用人写布尔规则在 RL 中直接构造安全奖励,替代不透明的安全 RM,使安全信号可解释、可审计、随政策更新;规则漏洞比黑盒 RM 更易发现修补,但规则本身仍可能被 gaming、需持续审计。
