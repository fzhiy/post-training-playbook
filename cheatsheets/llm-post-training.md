# LLM Post-Training 全栈速查手册
# Complete Bilingual Cheat Sheet: LLM Post-Training

> 适用场景：LLM Post-Training Research Intern 面试准备 & 日常参考
> 语言：中文为主，关键术语附英文

---

## 目录 / Table of Contents

- [Part 1 — 核心概念与公式推导](#part-1--核心概念与公式推导)
- [Part 2 — PyTorch 代码片段](#part-2--pytorch-代码片段)
- [Part 3 — 面试题库](#part-3--面试题库)

---

# Part 1 — 核心概念与公式推导
# Core Concepts & Formula Derivations

---

## 1. Pre-training vs Post-training 总览 / Overview

| 维度 / Dimension | 预训练 / Pre-training | 后训练 / Post-training |
|:---|:---|:---|
| **数据规模 / Data Scale** | 万亿 tokens，网络爬取语料 | 数千–百万条高质量标注/偏好数据 |
| **目标 / Objective** | Next-token prediction，习得语言与知识 | 指令跟随 + 对齐人类偏好 + 增强推理 |
| **损失函数 / Loss** | $L = -\sum_t \log p_\theta(x_t \mid x_{<t})$ | SFT loss + RLHF/DPO/GRPO 目标 |
| **学习率 / LR** | 高（1e-4 数量级），余弦退火 | 低（1e-5 ~ 5e-6），防止遗忘 |
| **硬件 / Hardware** | 数千张 GPU，训练数周到数月 | 数百张 GPU，训练数小时到数天 |
| **产物 / Output** | Base model（能力强但不可控） | Instruct / Chat model（可控、安全、有用） |

**标准 5 步 Pipeline（Standard 5-Step Pipeline）：**

1. **SFT (Supervised Fine-Tuning)**：在高质量 (instruction, response) 对上做有监督微调，将 base model 转化为"指令助手"。
2. **Reward Model 训练**：用人类偏好对比数据（同一 prompt 的两个 response + 人类偏好标注）训练打分模型 RM。
3. **RLHF / PPO**：用 RM 反馈做强化学习，配合 KL 约束防止偏离 SFT 模型过远。
4. **DPO（离线替代）**：绕过显式 RM，直接从偏好数据优化 policy，实现更简单稳定的对齐。
5. **迭代循环**：当前 policy 采样新数据 → 新偏好标注 → 更新 RM → 再做 RL，多轮迭代。

---

## 2. SFT 数据格式与 Loss Masking

**Chat Template（ChatML 格式示例）：**

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{用户指令}<|im_end|>
<|im_start|>assistant
{助手回复}<|im_end|>
```

不同模型有自己的 template（Llama-2 用 `[INST]`、Llama-3 用 `<|start_header_id|>`、Qwen 用 `<|im_start|>`），训练与推理必须使用相同 template，否则分布偏移。

**Loss Masking（损失掩码）：**

SFT 的训练目标是让模型学会"如何回答"，而不是"如何重复问题"。只对 assistant token 位置计算 cross-entropy loss：

$$L_{SFT} = -\frac{1}{|A|} \sum_{t \in A} \log p_\theta(x_t \mid x_{<t})$$

其中 $A$ 是所有 assistant token 的位置集合。User / system token 的 label 设为 $-100$（PyTorch 中 `CrossEntropyLoss` 默认忽略该值）。

**多轮对话 Loss Masking**：每一轮的 user turn 均 mask，只有每轮的 assistant turn 参与 loss 计算。

**System prompt 是否纳入 Loss 的 trade-off**：
- 不纳入（主流做法）：节省 capacity，专注训练回答质量。
- 纳入：模型能更好学习遵守 system instruction 的行为，但增加 noise。

### 2.1 常见 tokenization / chat-template 陷阱（SFT 实操筛题）

这些是 SFT 工程里最容易翻车、面试也爱问的点(每条:**问题** → 修法):

- **`pad_token = eos_token`**：很多模型(LLaMA/GPT-2)无 pad token,HF 默认把 pad 设成 eos。若不在 `attention_mask` 屏蔽 pad、且不把 pad 位置 label 置 `-100`,模型会在 pad 上算 loss / 学成"到处输出 eos"。→ 显式构造 attention_mask 屏蔽 pad,prompt 与 pad 的 label 一律 `-100`。
- **重复套用 chat template**：数据预处理已 `apply_chat_template` 后,tokenize 时又 `add_special_tokens=True` 或再套一次模板 → **双 BOS / 双特殊 token**。→ 模板只套一次;再 tokenize 时 `add_special_tokens=False`。
- **BOS 缺失或不一致**：训练加了 BOS、推理没加(或反之)→ 分布偏移(LLaMA 系对 BOS 敏感)。→ 核对模板是否已含 BOS,统一训练/推理行为。
- **tokenizer 版本 / 词表不一致**：训练与部署用了不同版本或加过自定义 token,token id 错位。→ 锁定 tokenizer 版本,与 checkpoint 一起保存。
- **新增 special token 未初始化**：加了新特殊 token 必须 `resize_token_embeddings`(否则 token id 越界报错);resize 出的新行是随机初始化,未训练时输出乱码。→ 对新行做合理初始化(常用:现有 embedding 的均值,或复用语义相近 token 的向量——均值只是常见 heuristic、非唯一解),并保证有数据训练这些新 token。
- **packing 跨样本污染**：多文档打包成一条序列时,若不用 block-diagonal / `cu_seqlens` mask,token 会跨文档 attend(详见 §3)。→ 用 varlen / cu_seqlens attention + 正确分隔。
- **套错模型的 template**：Llama-3(`<|begin_of_text|>` / `<|start_header_id|>`)、Qwen(`<|im_start|>`)、Mistral(`[INST]`)格式不同,套错性能骤降。→ 用目标模型自带的 `tokenizer.apply_chat_template`,别手写拼接。

> **面试自测(L2)**：为什么 `pad_token=eos_token` 时,若不正确设置 attention mask 与 label mask 会破坏 SFT?多轮对话里 pad 与 prompt 的 label 应如何处理?

---

## 3. Sequence Packing（序列打包）

**定义**：将多条短 sample 拼接成长度 = context window 的一个 sequence，只在 sample 边界加 EOS / 分隔 token，从而消除 padding 浪费。

**GPU 利用率**：不 packing 时，padding 占比可达 30–60%；packing 后接近 100% 有效 token，训练速度提升 2–4×。

**坑 1 — 跨 sample attention 污染**：不加 document-level attention mask 时，packed sequence 内前一个 sample 的 token 可以 attend 到后一个 sample 的 token，造成信息泄漏。解决方案：使用 Flash Attention 的 `cu_seqlens` 参数，它接受每个 sample 在 packed sequence 中的累积长度（cumulative sequence lengths），确保 attention 只在 sample 内部计算。

**坑 2 — Loss 权重失衡**：packing 等价于隐式按 token 数加权（长 sample 产生更多 loss terms）。若原本按 sample 平均，packing 后语义不同，需注意是否需要对 loss 做长度归一化。

**cu_seqlens 示例**（3 个 sample，长度分别为 5, 3, 7）：
```
cu_seqlens = [0, 5, 8, 15]  # cumulative lengths
packed_ids = [s1_tok1, ..., s1_tok5, s2_tok1, ..., s2_tok3, s3_tok1, ..., s3_tok7]
```

---

## 4. RLHF 完整流程 / PPO in RLHF

RLHF（Reinforcement Learning from Human Feedback）三阶段：

1. **SFT**：建立初始 policy $\pi_{ref}$（参考策略，后续作为 KL 惩罚的基准）。
2. **RM 训练**：用 Bradley-Terry 模型从人类偏好对比数据拟合标量奖励函数 $r(x,y)$。
3. **PPO 优化**：最大化带 KL 约束的增强奖励。

**总奖励（Augmented Reward）：**

$$r_{total}(x,y) = r_{RM}(x,y) - \beta \cdot \text{KL}\!\left(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)\right)$$

**PPO Clipped Objective：**

$$L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比值，$\varepsilon$ 通常取 0.1–0.2。

**GAE Advantage Estimation：**

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\lambda$ 控制 bias-variance 权衡：$\lambda=1$ 方差大 bias 小；$\lambda=0$ 退化为一步 TD。

**PPO 需要的 4 个模型（显存压力来源）：**

| 模型 | 作用 | 是否更新 |
|:---|:---|:---|
| **Actor** (Policy $\pi_\theta$) | 被优化的 LLM policy | 是（PPO 梯度） |
| **Critic** (Value model) | 估计 $V(s_t)$，计算 advantage | 是（TD 误差） |
| **Reference** ($\pi_{ref}$) | KL penalty 的基准，即 SFT 模型 | 否（冻结） |
| **Reward Model** (RM) | 对 (x,y) 打分 | 否（冻结） |

---

## 5. Bradley-Terry Reward Model（奖励模型）

**Bradley-Terry 偏好模型**：给定 prompt $x$，更好的回复 $y_w$ 比更差的 $y_l$ 被偏好的概率为：

$$P(y_w \succ y_l \mid x) = \sigma\!\left(r(x,y_w) - r(x,y_l)\right)$$

**RM 训练损失（最大化偏好数据的 log likelihood）：**

$$L_{RM} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(r(x,y_w) - r(x,y_l)\right)\right]$$

**RM 架构**：
- 初始化自 SFT 模型（共享语言理解能力）。
- 移除 LM head，替换为线性层 $W \in \mathbb{R}^{d \times 1}$，输出标量 reward。
- 训练时输入一对 (chosen, rejected) response，分别前向传播取最后 token 的标量输出，计算 Bradley-Terry loss。

**主要风险**：
- **Reward hacking**：policy 找到 RM 误差高分但质量差的 response（Goodhart's Law）。
- **Distribution shift**：RM 在未见过的 policy 分布上失效，需要迭代更新 RM。

---

---

## 6. DPO 完整推导 (Direct Preference Optimization Full Derivation)

### 6.1 从 RLHF 目标出发 (Starting from the RLHF Objective)

RLHF 的 KL 约束优化目标 (The KL-constrained RLHF optimization objective):

$$
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi(\cdot|x)}\!\big[r(x, y)\big] \;-\; \beta \cdot \mathrm{KL}\!\big(\pi(\cdot|x) \;\|\; \pi_{\mathrm{ref}}(\cdot|x)\big)
$$

其中 (where):
- $r(x, y)$ 是奖励模型的打分 (scalar reward from the reward model)
- $\pi_{\mathrm{ref}}$ 是参考策略，通常为 SFT 模型 (reference policy, typically the SFT model)
- $\beta > 0$ 控制 KL 惩罚强度 (controls KL penalty strength)

展开 KL 散度 (Expanding the KL divergence):

$$
\max_{\pi} \; \mathbb{E}_{x,y}\!\big[r(x,y)\big] - \beta \sum_{y} \pi(y|x)\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}
$$

对每个 $y$ 求变分最优，令泛函导数为零 (Taking the variational derivative per $y$ and setting it to zero):

$$
\frac{\partial}{\partial \pi(y|x)}\left[r(x,y) - \beta\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)} - \beta\right] = 0
$$

> 注意：归一化约束 $\sum_y \pi(y|x)=1$ 引入拉格朗日乘子 $Z(x)$

### 6.2 最优策略的闭式解 (Closed-Form Optimal Policy)

求解得到最优策略 (Solving yields the optimal policy):

$$
\boxed{\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)}
$$

其中配分函数 (partition function):

$$
Z(x) = \sum_{y} \pi_{\mathrm{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

$Z(x)$ 确保 $\sum_y \pi^*(y|x) = 1$，即策略是合法的概率分布 (ensures $\pi^*$ is a valid probability distribution)。

### 6.3 反解奖励函数 (Inverting for the Reward)

对最优策略两边取对数 (Taking log of both sides):

$$
\log \pi^*(y|x) = \log \pi_{\mathrm{ref}}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)
$$

移项得到 (Rearranging):

$$
\boxed{r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)}
$$

**关键洞察 (Key insight)**：奖励可以用策略与参考策略的对数比来表示，无需显式奖励模型 (reward is expressed via the log-ratio of policy to reference, eliminating the explicit RM)。

### 6.4 代入 Bradley-Terry 模型 (Substituting into Bradley-Terry)

人类偏好建模 (Human preference model):

$$
p(y_w \succ y_l \mid x) = \sigma\!\big(r(x, y_w) - r(x, y_l)\big)
$$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数 (sigmoid function)。

将反解的奖励代入 (Substituting the inverted reward):

$$
r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} + \cancel{\beta\log Z(x)} - \cancel{\beta\log Z(x)}
$$

> **$Z(x)$ 项完美消去**！这是因为同一 prompt $x$ 下两个响应共享相同的配分函数 (both responses share the same partition function for the same prompt)。

$$
p(y_w \succ y_l \mid x) = \sigma\!\left(\beta\log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)
$$

### 6.5 DPO 损失函数 (DPO Loss Function)

用参数化策略 $\pi_\theta$ 替代 $\pi^*$，取负对数似然 (replacing $\pi^*$ with parameterized $\pi_\theta$, negative log-likelihood):

$$
\boxed{\mathcal{L}_{\mathrm{DPO}}(\pi_\theta) = -\mathbb{E}_{(x,\, y_w,\, y_l) \sim \mathcal{D}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right]}
$$

展开隐式奖励 (Implicit reward defined as):

$$
\hat{r}_\theta(x, y) \triangleq \beta \log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}
$$

则损失简洁地写为 (Loss simplifies to):

$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\!\big[\log\sigma\!\big(\hat{r}_\theta(x,y_w) - \hat{r}_\theta(x,y_l)\big)\big]
$$

### 6.6 梯度分析 (Gradient Analysis)

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta\,\mathbb{E}\!\left[\underbrace{\sigma(-\hat{r}_\theta(x,y_w)+\hat{r}_\theta(x,y_l))}_{\text{权重：模型犯错越多，梯度越大}}\!\left(\nabla_\theta\log\pi_\theta(y_w|x) - \nabla_\theta\log\pi_\theta(y_l|x)\right)\right]
$$

- 当模型已经正确排序 $y_w \succ y_l$ 时，$\sigma(\cdot) \to 0$，梯度自然衰减 (gradient naturally decays)
- 当模型排序错误时，$\sigma(\cdot) \to 1$，梯度最大 (gradient is largest)

### 6.7 DPO 优缺点总结 (DPO Advantages & Disadvantages)

| 优点 (Advantages) | 缺点 (Disadvantages) |
|---|---|
| 无需训练独立的奖励模型 (No separate RM training) | **离线算法 (Offline)**：只能使用静态数据集 $\mathcal{D}$，无法在线探索 |
| 无需在线采样/rollout (No online rollout needed) | **分布偏移 (Distribution mismatch)**：$\pi_\theta$ 偏离数据收集策略时，训练信号退化 |
| 训练流程简化，只需一次优化 (Simplified pipeline) | **拒绝粒度粗 (Imprecise rejection)**：全局拒绝整个响应，而非逐步纠正 |
| 相比 PPO 更稳定 (More stable than PPO) | 对偏好数据质量敏感 (Sensitive to preference data quality) |
| 理论上等价于 RLHF（数据充分时） | $Z(x)$ 消去依赖于 BT 模型假设的正确性 |

### 6.8 Likelihood Displacement：chosen log-prob 也会下降

**现象 (Phenomenon)**

直觉上，DPO 训练应使模型对 chosen 响应 $y_w$ 的概率上升、对 rejected 响应 $y_l$ 的概率下降。但 Razin et al. (arXiv:2410.08847) 和 Pal et al. (arXiv:2402.13228) 均观察到：**训练过程中 $\log\pi_\theta(y_w|x)$ 和 $\log\pi_\theta(y_l|x)$ 往往同时下降**——loss 下降仅因为 $y_l$ 下降更快，两者之间的 margin 扩大，而 chosen 的绝对概率却在萎缩。

> "While intuitively these methods should increase the probability of $y^+$ while decreasing that of $y^-$, several recent works observed that the probabilities of both $y^+$ and $y^-$ tend to decrease over the course of training." — Razin et al., arXiv:2410.08847

**梯度机制 (Gradient Mechanism)**

DPO 损失只约束 log-prob **差值（margin）**相对于参考模型扩大：

$$\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

$\log\pi_\theta(y_w|x)$ 本身无下界约束——只要 $y_l$ 下降更快，梯度目标即满足。Razin et al. (Theorem 1/3) 指出，当 $y_w$ 与 $y_l$ 的隐层表示相似（high CHES score），压低 $y_l$ 的梯度方向同时也压低了 $y_w$，概率质量会位移到语义上与 $y_w$ **相反**的 token，形成"非预期错对齐"（unintentional unalignment，Razin et al. 原文用语）。

**危险情形 (Danger Conditions)**

- 数据集中 $y_w$ 与 $y_l$ 仅差几个 token（Pal et al. 报告 MetaMath 子集归一化编辑距离约 6.5%），共享大量前缀，梯度高度相关。
- 偏好对语义相近、区别细微（如措辞差异而非事实差异）。

**检测 (Detection)**

训练过程中同时记录 `chosen_logps_mean` 和 `rejected_logps_mean`（大多数训练框架日志已包含）。若 chosen mean 持续下降超过参考基线，即发生 displacement。

**缓解 (Mitigation)**

**(A) DPOP（Pal et al., arXiv:2402.13228）**：在 DPO 损失内部加入惩罚项，直接阻止 chosen log-prob 低于参考模型：

$$\mathcal{L}_\text{DPOP} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \lambda\cdot\max\!\left(0,\log\frac{\pi_\text{ref}(y_w|x)}{\pi_\theta(y_w|x)}\right)\right)\right]$$

$\lambda > 0$ 的 $\max(0,\cdot)$ 项：当 $\log\pi_\theta(y_w|x) < \log\pi_\text{ref}(y_w|x)$ 时施加惩罚，将 chosen log-prob"锚定"在参考模型以上。

**(B) CHES 数据过滤（Razin et al., arXiv:2410.08847）**：过滤掉 $y_w$ 与 $y_l$ 表示相似度高的样本对，从数据侧切断梯度耦合路径。

**(C) SimPO**：以长度归一化的 $\frac{1}{|y|}\log\pi_\theta(y|x)$ 为隐式奖励，并去掉 $\pi_\text{ref}$，reward 的定义直接与生成时 likelihood 对齐，从设计上减弱 displacement 的驱动力（但 SimPO 论文本身并未直接用 Razin/Pal 的框架分析此问题）。

> **注意**：上述三条缓解路径来自不同论文，不应交叉归因——DPOP 的 $\max$ 正则项出自 Pal et al.，CHES 过滤出自 Razin et al.；SimPO 与 displacement 的关联来自下游工作，此处仅供参考，不可将其归于 Razin/Pal 两篇原文。

---

## 7. DPO 变体对比 (DPO Variants Comparison)

### 7.1 IPO (Identity Preference Optimization / $\Psi$PO with $\Psi = \text{Id}$)

**解决的 DPO 问题**：DPO 使用 $\Psi(q) = \log(q/(1-q))$（logit 函数，对应 Bradley-Terry），当偏好接近确定性（$p^*(y_w \succ y_l) \to 1$）时，logit 趋于 $+\infty$，驱动 $\pi^*(y_l) \to 0$ 而与 KL 惩罚系数 $\tau$ 无关——KL 正则化在强偏好下形同虚设，策略过拟合偏好数据。

**核心改动**：在 $\Psi$PO 框架下将 $\Psi$ 替换为恒等映射（输入偏好概率 $p\in[0,1]$，$\Psi(p)=p$ 不会像 DPO 的 logit 映射那样在 $p\to 1$ 时发散），最终经验损失（Azar et al., arXiv:2310.12036, Eq. 17）为平方损失回归：

$$\mathcal{L}_{\text{IPO}}(\pi) = \mathbb{E}_{(y_w, y_l, x) \sim \mathcal{D}} \left[ \left( h_\pi(y_w, y_l, x) - \frac{1}{2\tau} \right)^2 \right]$$

其中 $h_\pi(y, y', x) = \log \dfrac{\pi(y|x)\,\pi_{\text{ref}}(y'|x)}{\pi(y'|x)\,\pi_{\text{ref}}(y|x)}$ 是策略相对参考的 log-ratio 差（logit margin）；目标常数 $\frac{1}{2\tau}$，$\tau$ 为 KL 正则化强度。

**引用**：Mohammad Gheshlaghi Azar et al. — arXiv:2310.12036 (Google DeepMind, 2023)

> "IPO, unlike DPO, always regularizes its solution towards $\pi_\text{ref}$ by controlling the gap between the log-likelihood ratios, thus avoiding the over-fitting to the preference dataset." — Azar et al., Section 5.2

**特点**：
- 平方损失将 logit-margin 回归到固定目标 $\frac{1}{2\tau}$；$\tau$ 直接控制学到的 log-ratio 差的上界，KL 正则化始终有效
- 梯度在目标附近可能较小但永不为零，无法"逃逸"到无界区域
- **注意**：Azar et al. 原文验证仅在小规模 bandit 实验；LLM 规模的效果需独立核查
- **权衡**：损失不对应 BT 概率模型，理论解释性略弱

### 7.2 KTO (Kahneman-Tversky Optimization)

**解决的 DPO 问题**：(1) DPO 需要成对偏好数据 $(x, y_w, y_l)$，而现实中常只有单独的正/负反馈 (pointwise thumbs-up/down)，配对数据成本高且稀缺。(2) DPO 最大化偏好对数似然，是对"最大化生成效用"这一真实目标的代理，存在目标错位 (objective mismatch)。

**损失形式（完整）**：

$$\mathcal{L}_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x,y \sim \mathcal{D}}\bigl[w(y)\bigl(1 - v_{\text{KTO}}(x,y;\beta)\bigr)\bigr]$$

其中隐式奖励、KL 基准和价值函数定义为：

$$r_{\text{KTO}}(x,y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

$$z_{\text{ref}} = \mathbb{E}_{x' \sim \mathcal{D}}\bigl[\beta\,\mathrm{KL}(\pi_\theta(y' \mid x') \,\|\, \pi_{\text{ref}}(y' \mid x'))\bigr]$$

$$v_{\text{KTO}}(x,y;\beta) = \begin{cases} \sigma(r_{\text{KTO}}(x,y) - z_{\text{ref}}) & \text{if } y \sim y_{\text{desirable}} \mid x \\ \sigma(z_{\text{ref}} - r_{\text{KTO}}(x,y)) & \text{if } y \sim y_{\text{undesirable}} \mid x \end{cases}$$

$$w(y) = \begin{cases} \lambda_D & \text{if desirable} \\ \lambda_U & \text{if undesirable} \end{cases}$$

**$z_\text{ref}$ 的作用与实现 (KL Baseline)**

$z_\text{ref}$ 是当前策略相对于参考模型的期望 KL 散度，在前景理论中充当"参考点"（reference point）——奖励超过此点为"收益"，低于此点为"损失"，从而实现 sigmoid 在收益侧的凹性（风险规避）与损失侧的凸性（损失厌恶）。

实现上，$z_\text{ref}$ 在每个 mini-batch（大小 $m$）中用**不匹配**的 $(x', y'_U)$ 对估计：

$$\hat{z}_{\text{ref}} = \max\!\left(0,\,\frac{1}{m}\sum_i \log\frac{\pi_\theta(y'_{U,i} \mid x'_i)}{\pi_{\text{ref}}(y'_{U,i} \mid x'_i)}\right)$$

故意将 prompt $x'$ 与**不相关**的输出 $y'_U$ 配对，是为了避免将 reward 信号与 baseline 估计混淆。梯度**不**回传过 $z_\text{ref}$ 项。

**前景理论对应关系 (Prospect Theory Mapping)**

$v_\text{KTO}$ 用 logistic 函数近似 Kahneman-Tversky 的 S 形价值函数（原始幂律形式难以直接优化）：符号翻转（desirable 分支 $r - z_\text{ref}$；undesirable 分支 $z_\text{ref} - r$）精确模拟"收益 vs 损失"帧切换，$\lambda_D / \lambda_U$ 的非对称权重对应损失厌恶。

> "KTO only requires a binary signal of whether an output is (un)desirable for a given input. This data is much more abundant, cheaper, and faster to collect in the real world than preferences." — Ethayarajh et al., arXiv:2402.01306

**引用**：Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela — arXiv:2402.01306 (2024)

**特点**：
- **无需配对 (No pairing required)**：每条 $(x, y)$ 只需一个 desirable/undesirable 标签，可直接使用点赞/踩等天然二元日志
- $z_\text{ref}$ 提供动态参考点，使损失对 KL 漂移自适应
- **权衡**：放弃了 BT 模型的成对一致性约束，信息利用效率可能低于成对方法；$\lambda_D / \lambda_U$ 需要手动调整

### 7.3 ORPO (Odds Ratio Preference Optimization)

**解决的 DPO 问题**：DPO 需要先 SFT 再偏好优化两阶段训练，且需要维护参考模型 $\pi_{\mathrm{ref}}$。

**核心改动**：在 SFT 的交叉熵损失上直接附加 odds-ratio 偏好损失 (unified SFT + odds-ratio loss):

$$
\mathcal{L}_{\mathrm{ORPO}} = \underbrace{\mathcal{L}_{\mathrm{SFT}}(y_w)}_{\text{SFT on chosen}} + \lambda \cdot \underbrace{\left[-\log\sigma\!\left(\log\frac{\mathrm{odds}_\theta(y_w|x)}{\mathrm{odds}_\theta(y_l|x)}\right)\right]}_{\text{odds ratio preference}}
$$

其中 odds 定义为 (odds defined as):

$$
\mathrm{odds}_\theta(y|x) = \frac{p_\theta(y|x)}{1 - p_\theta(y|x)}
$$

**引用**：Jiwoo Hong, Noah Lee, James Thorne — arXiv:2403.07691 (2024)

> "In contrast to previous works, our approach requires neither an SFT warm-up stage nor a reference model, enabling resource-efficient development of preference-based aligned models." — Hong et al., arXiv:2403.07691

**特点**：
- 单阶段训练 (Single-stage)，无需参考模型 (No reference model needed)，前向传播次数减半
- $\mathcal{L}_\text{SFT}$ 在 chosen 响应上提供域适配锚定，$\mathcal{L}_\text{OR}$ 同步排斥 rejected 风格
- **权衡**：odds ratio 与 BT 模型无直接理论联系；两项损失权重 $\lambda$ 需调；数值结果未在此独立核查

### 7.4 SimPO (Simple Preference Optimization)

**解决的 DPO 问题**：(1) DPO 隐式奖励 $\beta\log\pi_\theta/\pi_\text{ref}$ 与生成时实际使用的度量（长度归一化 likelihood）之间存在分歧——Meng et al. 指出，在 UltraFeedback 三元组中 DPO 奖励排序满足而 log-likelihood 排序反转的比例接近一半，即模型可在"赢得 loss"的同时令 chosen 实际更难被生成。(2) 未归一化的 log-prob 随长度单调递减，模型可以通过生成更短的 rejected 回复来满足排序，引入长度偏差。(3) 需维护冻结 $\pi_\text{ref}$，带来内存和计算开销。

**核心改动**：将隐式奖励替换为长度归一化的序列级平均对数概率，并在 Bradley-Terry 目标中引入显式目标边际 $\gamma > 0$：

$$r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|}\log\pi_\theta(y|x)$$

$$\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

其中 $|y|$ 为 token 数，$\beta$ 为缩放常数，$\gamma > 0$ 要求 chosen 的奖励至少比 rejected 高出 $\gamma$（不仅仅是更大）。不含 $\pi_\text{ref}$。

**引用**：Yu Meng, Mengzhou Xia, Danqi Chen — arXiv:2405.14734 (NeurIPS 2024)

> "There is a divergence between DPO's reward formulation $r_\theta(x,y)=\beta\log\pi_\theta(y|x)/\pi_\text{ref}(y|x)$ and the average log likelihood metric $p_\theta(y|x)=\frac{1}{|y|}\log\pi_\theta(y|x)$, which directly impacts generation." — Meng et al., arXiv:2405.14734, Section 3.1

**特点**：
- 无需参考模型，reward 直接与生成时 likelihood 对齐（去掉分布漂移来源）
- 长度归一化消除"短 rejected 作弊"的捷径
- $\gamma > 0$ 的目标边际强化 chosen/rejected 的绝对间距，不仅是相对排序
- **权衡**：损失不对应 BT 概率模型；性能数字（AlpacaEval/Arena-Hard）来自论文摘要，未在此独立核查

### 7.5 在线 vs 离线 DPO (Online vs Offline DPO)

**分布漂移问题 (Distribution Mismatch)**

标准 DPO 是**离线（offline）**算法：偏好数据集 $\mathcal{D}$ 在训练前收集，来自某个固定的（通常是 SFT 模型的）数据生成策略 $\mu$。训练过程中，$\pi_\theta$ 不断更新，但 $\mathcal{D}$ 保持静态。当 $\pi_\theta$ 偏离 $\mu$ 后，$\mathcal{D}$ 中的 $(y_w, y_l)$ 对不再覆盖 $\pi_\theta$ 当前的输出分布，形成 **off-policy distribution mismatch**。

具体表现：

- **隐式奖励漂移**：DPO 的隐式奖励 $\hat{r}_\theta(x,y) = \beta\log\pi_\theta(y|x)/\pi_\text{ref}(y|x)$ 在训练中持续变化，对同一对 $(y_w, y_l)$ 的打分随策略更新而改变，可能出现 chosen/rejected margin 缩小甚至反转。
- **探索受限**：策略无法探索比 $\mathcal{D}$ 中更好的输出，困在旧偏好数据定义的"local optimum"。

**迭代/在线 DPO (Iterative / On-Policy DPO)**

解决思路：在每轮迭代中，用**当前**策略 $\pi_\theta^{(t)}$ 采样新的响应对，再通过奖励模型（或人类/AI 评判）构造新的偏好对 $(y_w^{(t)}, y_l^{(t)})$，然后用这批**分布匹配**的偏好数据更新策略：

$$\pi_\theta^{(t+1)} \leftarrow \text{DPO-update}\!\left(\pi_\theta^{(t)},\;\mathcal{D}^{(t)}\right), \quad \mathcal{D}^{(t)} \sim \pi_\theta^{(t)}$$

**为何在线 DPO 通常更好**：

| 对比维度 | 离线 DPO | 在线 / 迭代 DPO |
|:---|:---|:---|
| 偏好数据来源 | 静态预收集，来自固定 $\mu$ | 每轮用当前 $\pi_\theta$ 采样 |
| 分布匹配 | off-policy，存在漂移 | on-policy，匹配当前策略 |
| 训练信号质量 | 受旧分布限制 | 覆盖当前策略的输出分布 |
| 计算成本 | 低（一次性数据） | 高（需要每轮在线采样 + RM 打分） |
| 探索能力 | 无，被 $\mathcal{D}$ 锁定 | 可探索新的输出模式 |
| 代表方法 | 标准 DPO (Rafailov et al.) | RLHF-PPO、Online DPO、Self-Play Fine-Tuning |

**实践建议**：若有访问 RM 或自动评判器的能力，迭代 DPO（每 $k$ steps 重新采样+更新偏好数据）比纯离线 DPO 通常在 downstream 对话质量上更优。若只能离线，SimPO/DPOP 中去掉 $\pi_\text{ref}$ 或增加 chosen-anchoring 可部分缓解漂移带来的 likelihood displacement。

### 7.6 精确对比表 (Precise Comparison Table)

| 变体 | 需要成对偏好? | 需要 $\pi_\text{ref}$? | 损失形式要点 | 主要修正的 DPO 问题 |
|:---:|:---:|:---:|:---|:---|
| **DPO** | ✅ 是 | ✅ 是 | $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})$ | 基准（无显式修正） |
| **IPO** | ✅ 是 | ✅ 是 | $\left(h_\pi(y_w,y_l) - \frac{1}{2\tau}\right)^2$，平方损失回归 | KL 正则化在确定性偏好下失效；奖励无界漂移 |
| **KTO** | ❌ 否（pointwise 二元标签） | ✅ 是 | $w(y)(1 - v_\text{KTO}(x,y;\beta))$，非对称 sigmoid + KL 基准 $z_\text{ref}$ | 需要成对数据；目标与实际生成效用错位 |
| **ORPO** | ✅ 是 | ❌ 否 | $\mathcal{L}_\text{SFT} + \lambda(-\log\sigma(\log\frac{\text{odds}(y_w)}{\text{odds}(y_l)}))$ | 两阶段训练；需维护冻结 $\pi_\text{ref}$（内存/计算翻倍） |
| **SimPO** | ✅ 是 | ❌ 否 | $-\log\sigma(\frac{\beta}{|y_w|}\log\pi_\theta(y_w) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l) - \gamma)$ | Likelihood displacement；长度偏差；$\pi_\text{ref}$ 开销 |

**引用**：DPO — Rafailov et al., arXiv:2305.18290; IPO — Azar et al., arXiv:2310.12036; KTO — Ethayarajh et al., arXiv:2402.01306; ORPO — Hong et al., arXiv:2403.07691; SimPO — Meng et al., arXiv:2405.14734 (NeurIPS 2024)

---

## 8. GRPO vs PPO (Group Relative Policy Optimization vs Proximal Policy Optimization)

### 8.1 GRPO 群组优势估计 (Group Advantage Estimation)

GRPO 的核心思想：对同一 prompt $x$，采样一组响应 $\{y_1, y_2, \dots, y_G\}$，用组内统计量估计优势 (estimate advantage using intra-group statistics):

$$
\boxed{A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \dots, r_G\})}{\mathrm{std}(\{r_1, r_2, \dots, r_G\})}}
$$

其中 $r_i = r(x, y_i)$ 是第 $i$ 个响应的奖励 (reward for the $i$-th response)。

GRPO 的策略梯度损失 (GRPO policy gradient loss):

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\left[\min\!\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\mathrm{old}}}(y_i|x)} A_i, \;\mathrm{clip}\!\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\mathrm{old}}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right)A_i\right)\right]
$$

> 与 PPO 的 clipping 机制相同，但优势估计完全不同 (same clipping mechanism, entirely different advantage estimation)。

### 8.2 核心对比 (Key Comparison)

| 特性 | PPO | GRPO |
|:---|:---|:---|
| **所需模型数量** | 4 个：Actor + Critic + Reference + RM | 2 个：Actor + Reference（奖励来自外部/规则） |
| **优势估计** | GAE (Generalized Advantage Estimation)，需要 Critic 网络 | 组内相对排名，无需 Critic |
| **内存开销** | 高（4 份模型权重） | 低（2 份模型权重） |
| **奖励来源** | 学习到的神经网络 RM (learned neural RM) | 可验证奖励 / 规则奖励 (verifiable / rule-based reward) |
| **适用场景** | 开放式对话、创意写作 (open-ended generation) | 数学推理、代码生成 (math, code with verifiable ground truth) |
| **训练稳定性** | 需仔细调参 Critic，否则不稳定 | 更稳定，因为无 Critic 估计误差 |
| **梯度方差** | 较低（GAE 提供低方差估计） | 较高（组采样数 $G$ 有限） |

### 8.3 RLVR 框架 (RL from Verifiable Rewards)

GRPO 最适配的范式是 **RLVR**：奖励不来自学习的 RM，而是来自可自动验证的规则 (rewards from automatically verifiable rules)：

- **数学**：答案是否等于标准答案 (answer matches ground truth) → $r = \mathbb{1}[\text{extract}(y) = y^*]$
- **代码**：是否通过所有测试用例 (passes all test cases) → $r = \frac{\text{passed tests}}{\text{total tests}}$
- **格式**：是否遵循指定格式 (follows required format) → $r \in \{0, 1\}$

RLVR 的核心优势：**奖励无噪声** (noise-free reward)，避免了 RM 本身的偏差与过拟合。

### 8.4 选择指南 (When to Prefer Which)

- **优先 GRPO**：奖励可自动验证（数学、代码、逻辑推理）；资源有限（无法维护 4 个模型）；需要稳定训练
- **优先 PPO**：奖励需要语义/风格判断（对话质量、创意写作）；奖励信号复杂且无法规则化；有充足计算资源和成熟的 RM

### 8.5 RLOO 与 ReMax (Critic-free 基线)

PPO 依赖一个学习到的评论家（价值网络）来估计基线。GRPO、RLOO 与 ReMax 均为**无评论家方法**，它们用从采样奖励中计算出的基线替代了价值网络。

**RLOO**（REINFORCE Leave-One-Out，Ahmadian et al. 2024, ACL [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)）：对于每个提示词生成的 $G$ 个样本，样本 $i$ 的基线是其余 $G-1$ 个样本奖励的均值；优势函数 $A_i = r_i - \frac{1}{G-1}\sum_{j\neq i} r_j$。这是纯 REINFORCE 梯度，无裁剪、无评论家；由于基线不依赖样本 $i$ 自身的动作，该策略梯度估计保持无偏。

**ReMax**（Li et al. 2024, ICML [arXiv:2310.10505](https://arxiv.org/abs/2310.10505)）：基线是对同一提示词进行单次贪婪（argmax）解码所获得的奖励；优势 $A = r(\text{采样}) - r(\text{贪婪})$。每个提示词仅需一次额外的贪婪推理，内存开销极低，无需评论家。

| 方法/Method | baseline | 估计器/estimator | clip? | 额外成本/extra cost |
| :--- | :--- | :--- | :--- | :--- |
| GRPO | 组内 z-score | PPO 风格估计器 | Yes | $G$ 个样本 |
| RLOO | leave-one-out 均值 | REINFORCE | No | $G$ 个样本 |
| ReMax | 贪婪解码奖励 | REINFORCE | No | +1 次贪婪解码 |

---

## 9. KL 惩罚的作用与 β 调参 (Role of KL Penalty & Tuning β)

### 9.1 KL 项的直觉理解 (Intuitive Role of the KL Term)

$$
\beta \cdot \mathrm{KL}\!\big(\pi_\theta(\cdot|x) \;\|\; \pi_{\mathrm{ref}}(\cdot|x)\big)
$$

KL 惩罚的角色是**正则化** (regularization)，具体功能：

1. **防止策略偏离太远 (Prevents excessive drift)**：确保 $\pi_\theta$ 不会偏离 $\pi_{\mathrm{ref}}$ 太多，保留预训练知识
2. **抑制奖励黑客 (Mitigates reward hacking)**：如果策略学会 exploit RM 的缺陷，KL 会增大作为惩罚
3. **维持多样性 (Maintains diversity)**：防止策略坍缩到少数高奖励模式 (mode collapse)
4. **稳定训练 (Stabilizes training)**：约束探索空间，避免策略更新过大

数学上展开 (Expanding mathematically):

$$
\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right] \geq 0
$$

> 当 $\pi_\theta = \pi_{\mathrm{ref}}$ 时，KL = 0（完全不偏离参考策略时无惩罚）。

### 9.2 KL-RM 分数 Pareto 前沿 (KL-RM Score Pareto Frontier)

调参 $\beta$ 本质上是在两个目标间权衡：

$$
\underbrace{\mathbb{E}[r(x,y)]}_{\text{奖励分数}\uparrow} \quad \text{vs.} \quad \underbrace{\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})}_{\text{偏离程度}\uparrow}
$$

| $\beta$ 值 | 效果 |
|:---:|:---|
| $\beta$ 太大 | 策略几乎不更新，接近 $\pi_{\mathrm{ref}}$，奖励提升小 (underfitting) |
| $\beta$ 太小 | 策略激进更新，奖励可能高但分布偏移严重，可能 reward hacking |
| $\beta$ 适中 | 在 KL-RM 前沿上找到平衡点 |

典型值范围 (typical range)：$\beta \in [0.01, 0.5]$，实际中常用 $\beta = 0.1$ 或 $\beta = 0.2$。

> 📝 **过优化的定量版本**：gold-RM 分数随 $\sqrt{\mathrm{KL}}$ 呈**倒 U**（BoN 形式 $d(\alpha-\beta d)$、RL 形式 $d(\alpha-\beta\log d)$，$d=\sqrt{\mathrm{KL}}$）——超过峰值后策略漂移出分布、gold 分回落。详见 [reward-modeling-eval §3.2a](cheatsheet-reward-modeling-eval.html)（Gao et al. 2022, [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)）。

### 9.3 DPO 中的 β 与 PPO 中的 β (β in DPO vs PPO)

| 维度 | PPO 中的 $\beta$ | DPO 中的 $\beta$ |
|:---|:---|:---|
| **数学角色** | 控制 KL 惩罚项的权重（在损失函数中） | 控制隐式奖励的缩放（在 log-ratio 中） |
| **出现位置** | $\max_\pi \mathbb{E}[r] - \beta \cdot \mathrm{KL}$ | $\hat{r}(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$ |
| **是否语义等价** | 理论上，DPO 的 $\beta$ 源自 RLHF 目标中的同一个 $\beta$，但实际中因训练动态不同，数值上需要分别调优 |
| **实际影响** | $\beta \uparrow$ → 更保守策略 | $\beta \uparrow$ → 隐式奖励变化更剧烈，对偏好信号更敏感 |

> **结论**：理论等价 (theoretically equivalent)，实践不同 (practically different)。DPO 中 $\beta$ 还影响梯度中的权重项 $\sigma(\cdot)$ 的锐度 (sharpness)。

### 9.4 KL 估计器与放置 (KL Estimators & Placement)

#### 9.4.1 三种单样本估计器

**符号约定**：令 $r = \pi_{\mathrm{ref}} / \pi_\theta$（Schulman 2020 约定），样本从当前策略 $\pi_\theta$ 采样。

定义三个估计器：

$$k_1 = -\log r = \log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$$

$$k_2 = \tfrac{1}{2}(\log r)^2 = \tfrac{1}{2}\!\left(\log\frac{\pi_{\mathrm{ref}}}{\pi_\theta}\right)^{\!2}$$

$$k_3 = (r - 1) - \log r = \left(\frac{\pi_{\mathrm{ref}}}{\pi_\theta} - 1\right) - \log\frac{\pi_{\mathrm{ref}}}{\pi_\theta}$$

**验证期望**（样本来自 $\pi_\theta$）：

$$\mathbb{E}_{a \sim \pi_\theta}[r] = \sum_a \pi_\theta(a) \cdot \frac{\pi_{\mathrm{ref}}(a)}{\pi_\theta(a)} = \sum_a \pi_{\mathrm{ref}}(a) = 1$$

因此 $\mathbb{E}[r - 1] = 0$，即 $(r-1)$ 是零均值项（控制变量）。

- **$k_1$ 的无偏性**：$\mathbb{E}_{a \sim \pi_\theta}[k_1] = \mathbb{E}_{a \sim \pi_\theta}\!\left[-\log\frac{\pi_{\mathrm{ref}}}{\pi_\theta}\right] = \mathbb{E}_{a \sim \pi_\theta}\!\left[\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right] = \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}}) \geq 0$。$k_1$ 是 KL 值的无偏估计。

- **$k_2$ 的有偏性**：$k_2 = \tfrac{1}{2}(\log r)^2 \geq 0$，但以 $\pi_\theta$ 取期望时 $\mathbb{E}[k_2] \neq \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$，故 $k_2$ 有偏。

- **$k_3$ 的无偏性**（控制变量论证）：对任意 $\lambda$，令 $k_\lambda = -\log r + \lambda(r-1)$。由于 $\mathbb{E}[r-1]=0$，$k_\lambda$ 与 $k_1$ 有相同的期望，即 $\mathbb{E}[k_\lambda] = \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$——对所有 $\lambda$ 均无偏。取 $\lambda = 1$ 恰好得到 $k_3$，此时附加项 $\lambda(r-1)$ 与 $-\log r$ 负相关，**降低了方差**。故 $k_3$ 无偏，且在 RLHF 关心的小漂移区间（$r \approx 1$）方差低于 $k_1$。

- **$k_3 \geq 0$ 的几何意义**：由切线不等式 $\log r \leq r - 1$（对 $r > 0$ 严格成立，等号仅在 $r=1$ 时取到），所以 $(r-1) - \log r \geq 0$，与 KL 散度非负一致。

> **近 $r=1$ 的阶数对比**（设 $\epsilon = r-1$）：$k_1 = -\log r \approx -\epsilon$ 是**一阶量**（有符号），而 $k_2 \approx k_3 \approx \tfrac{1}{2}\epsilon^2$ 是**二阶量**（非负）。三者都随 $r \to 1$ 趋于 0，但收敛阶不同——这也解释了为何 $k_3$ 像 $k_2$ 一样非负、却又像 $k_1$ 一样无偏。

#### 9.4.2 梯度视角

以上分析均针对**值估计**（value estimation）。将估计器放入损失函数时，梯度行为需单独分析。

- **$k_3$ 作为损失项不产生精确的逆向 KL 梯度**：尽管 $k_3$ 是 KL 值的无偏估计，将其作为损失对 $\pi_\theta$ 求梯度时，得到的梯度仅是真实逆向 KL 梯度的一阶近似——在策略漂移较小（$r \approx 1$，即 $\log r \approx r - 1$）时近似成立，漂移较大时存在系统性偏差。

- 根据 **Liu et al.（arXiv:2510.01555）** 的分析，**$k_1$ 放入奖励（in-reward）** 和 **$k_2$ 作为损失项（as-loss）** 是梯度意义上有原则的选择，而 **$k_3$ 作为损失项在梯度上没有原则保证**。实践中 GRPO / DeepSeek-R1 使用的 $\beta$ 较小，由此引入的梯度偏差通常可以接受。

#### 9.4.3 估计器对比表

| 估计器 / Estimator | 形式 / Form | 值无偏? / Value-unbiased? | 梯度 principled? / Gradient-principled? |
|:---|:---|:---|:---|
| $k_1$ | $-\log r$ | 是 / Yes | 是，作为 in-reward / Yes, as in-reward |
| $k_2$ | $\tfrac{1}{2}(\log r)^2$ | 否 / No | 是，作为 as-loss / Yes, as-loss |
| $k_3$ | $(r-1)-\log r$ | 是 / Yes | 否，作为 as-loss / No, as-loss |

> **方差排序**：在小漂移区间（$r \approx 1$）$k_3$ 的方差低于 $k_1$（两者均无偏）；$k_2$ 有偏，处于不同的偏差–方差权衡区间，不宜与 $k_1$/$k_3$ 直接比较方差大小。

#### 9.4.4 两种放置方式（Style A vs Style B）

**Style A：放入奖励（in-reward）**

典型代表：InstructGPT / PPO。KL 惩罚逐 token 加入奖励信号：

$$r_{\mathrm{total}}(x, y) = r_{\mathrm{RM}}(x, y) - \beta \cdot k_1^{(t)}, \quad k_1^{(t)} = \log\frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{ref}}(a_t|s_t)}$$

- 每个 token 位置独立获得 KL 惩罚信号，Critic 可以学习到逐步的 KL 代价。
- **注意**：PPO 的 clip 机制会对策略比率做截断。对于被 clip 的 token，surrogate 目标不再随 $\pi_\theta$ 变化，**in-reward 的 KL 信号对这些 token 的梯度贡献可能被静默屏蔽**，这是实现层面常被忽视的细节。

**Style B：放入损失（in-loss）**

典型代表：GRPO（Shao et al., DeepSeekMath）、DeepSeek-R1。KL 估计器直接加到策略优化损失上：

$$\mathcal{L} = \mathcal{L}_{\mathrm{GRPO}} + \beta \cdot k_3$$

其中 $k_3 = (r - 1) - \log r$，$r = \pi_{\mathrm{ref}} / \pi_\theta$（逐 token 计算后对序列平均）。

- 无需 Critic / Value model，内存开销更低。
- **DAPO**（arXiv:2503.14476）：直接移除 KL 惩罚（$\beta = 0$）。其理由是 long-CoT 推理训练中策略会显著偏离初始 SFT 参考模型，紧的 KL 约束反而有害；改用非对称裁剪（Clip-Higher，解耦的上、下截断界）来防止熵崩溃。

| 维度 | Style A（in-reward） | Style B（in-loss） |
|:---|:---|:---|
| 代表系统 | InstructGPT, PPO | GRPO, DeepSeek-R1 |
| 使用估计器 | $k_1$（per-token） | $k_3$（per-token, 平均） |
| 梯度原则性 | 是 | 近似（小 $\beta$ 时可接受） |
| 工程复杂度 | 需要 Critic | 无需 Critic |
| clip 屏蔽风险 | 存在（被 clip token 的 KL 梯度静默消失） | 不适用 |

#### 9.4.5 面试自测

> **L2**：用 $r = \pi_{\mathrm{ref}}/\pi_\theta$ 约定时，为什么 $\mathbb{E}_{a \sim \pi_\theta}[r] = 1$？这一结论如何说明 $k_3$ 的无偏性？

> **L3**：GRPO 将 $k_3$ 直接放入损失，但 $k_3$ 作为损失项在梯度上不是 principled 的逆向 KL 梯度——为什么在 GRPO 实践中这通常不是问题？如果将 $\beta$ 从 0.04 调大到 0.5，这个近似误差会如何变化？

---

## 10. Process Reward Model (PRM) vs Outcome Reward Model (ORM)

### 10.1 ORM：结果奖励模型 (Outcome Reward Model)

$$
r_{\mathrm{ORM}}(x, y) = f_\phi(x, y) \in \mathbb{R}
$$

- 仅在**序列末尾** (End of Sequence, EOS) 给出一个标量奖励
- 整个响应 $y = (s_1, s_2, \dots, s_T)$ 共享同一奖励值
- 训练数据：$(x, y, \text{correct/incorrect})$ 二元标签

### 10.2 PRM：过程奖励模型 (Process Reward Model)

$$
r_{\mathrm{PRM}}(x, y, t) = f_\phi(x, s_1, \dots, s_t) \in \mathbb{R}
$$

- 对推理链的**每一步**给出独立的分数 (step-level scoring)
- $r_{\text{step}}(s_t)$ 表示第 $t$ 步推理的质量
- 训练数据：$(x, s_1, \dots, s_t, \text{step correct/incorrect})$ 逐步标签

### 10.3 信用分配优势 (Credit Assignment Advantage)

这是 PRM 的核心优势。考虑一个数学推理链：

> Step 1: 设 $f(x) = x^2 + 3x$ → ✅ 正确
> Step 2: 求导得 $f'(x) = 2x + 3$ → ✅ 正确
> Step 3: 令 $f'(x) = 0$，解 $x = -3/2$ → ✅ 正确
> Step 4: $f(-3/2) = 9/4 - 9/2 = -9/4$ → ✅ 正确

**ORM** 只知道"最终答案正确"→ 给高分，但不知道每步是否可靠。

**PRM** 可以识别"前三步正确，第四步错误"的情况：

$$
\underbrace{r_{\mathrm{PRM}}(s_1) > 0,\; r_{\mathrm{PRM}}(s_2) > 0,\; r_{\mathrm{PRM}}(s_3) > 0}_{\text{正确步骤}} \quad \underbrace{r_{\mathrm{PRM}}(s_4) < 0}_{\text{错误步骤被定位}}
$$

这使得 PRM 能更精确地指导搜索和训练 (more precise guidance for search and training)。

### 10.4 基于 PRM 的 Best-of-N 搜索 (Best-of-N Search with PRM)

给定 prompt $x$，采样 $N$ 个候选响应 $\{y_1, \dots, y_N\}$，对每个响应的每一步打分：

$$
\text{Score}(y_i) = \min_{t=1}^{T_i} r_{\mathrm{PRM}}(x, y_i, t)
$$

或使用乘积形式 (product form)：

$$
\text{Score}(y_i) = \prod_{t=1}^{T_i} \sigma\!\big(r_{\mathrm{PRM}}(x, y_i, t)\big)
$$

> **取 min 或乘积**是为了确保每一步都合格——任何一步低分都会拉低整体分数 (any weak step pulls down the overall score)。

选择最优响应 (Select the best):

$$
y^* = \arg\max_{y_i} \text{Score}(y_i)
$$

### 10.5 PRM 训练数据挑战 (PRM Training Data Challenges)

| 挑战 | 说明 |
|:---|:---|
| **标注成本极高 (Expensive annotation)** | 每条推理链的每一步都需要人类专家标注正确性，比 ORM 标注贵 10-50× |
| **步骤边界模糊 (Ambiguous step boundaries)** | 推理步骤的切分没有统一标准，不同标注者可能有不同的切分方式 |
| **标注一致性低 (Low inter-annotator agreement)** | 对"某步是否正确"的判断可能因标注者数学水平而异 |
| **自动化方法的局限** | Monte Carlo 估计法（通过多次采样估计某步之后能得出正确答案的概率）有较大方差 |

> **自动化 PRM 标注方法**：对第 $t$ 步之后，多次采样完成推理，统计最终答案正确的比例作为 $r_{\mathrm{PRM}}(s_t)$ 的估计。公式：$r_{\mathrm{PRM}}(s_t) \approx \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[\text{completion}_k \text{ leads to correct answer}]$

---

## 11. Alignment Tax 与权重平均 (Alignment Tax & Weight Averaging)

### 11.1 对齐税的定义 (Definition of Alignment Tax)

**对齐税 (Alignment Tax)** 指模型经过对齐训练后，在基础能力基准上的性能下降：

$$
\text{Alignment Tax} = \text{Perf}_{\mathrm{base}}(\theta_{\mathrm{base}}) - \text{Perf}_{\mathrm{base}}(\theta_{\mathrm{aligned}})
$$

其中 $\text{Perf}_{\mathrm{base}}$ 表示在预训练基准（如 MMLU、代码能力、数学能力等）上的表现。

直觉上：SFT/RL 训练在提升对齐质量（安全性、有用性、格式遵循）的同时，可能"遗忘"或"覆盖"部分预训练知识。

### 11.2 WiSE-FT 线性插值 (WiSE-FT Linear Interpolation)

**WiSE-FT** (Weight-space Ensembles for Finetuning) 通过在权重空间中插值来减轻对齐税：

$$
\boxed{\theta_{\mathrm{merged}} = (1 - \alpha)\,\theta_{\mathrm{aligned}} + \alpha\,\theta_{\mathrm{base}}}
$$

其中 $\alpha \in [0, 1]$ 控制对齐模型与基础模型之间的权衡 (controls the trade-off)。

| $\alpha$ | 效果 |
|:---:|:---|
| $\alpha = 0$ | 完全使用对齐模型 (fully aligned) |
| $\alpha = 1$ | 完全使用基础模型 (base model) |
| $\alpha \in (0, 1)$ | 折中：保留部分对齐行为，恢复部分基础能力 |

### 11.3 为什么插值有效 (Why Interpolation Works)

**任务向量 (Task Vector)** 视角：对齐训练相当于在权重空间中沿一个方向移动：

$$
\tau_{\mathrm{align}} = \theta_{\mathrm{aligned}} - \theta_{\mathrm{base}}
$$

研究表明，不同任务对应的权重变化方向近似正交 (near-orthogonal)，因此：

$$
\theta_{\mathrm{merged}} = \theta_{\mathrm{base}} + (1-\alpha)\,\tau_{\mathrm{align}}
$$

线性插值不会严重干扰其他任务的表示 (doesn't severely interfere with other task representations)。

### 11.4 模型合并进阶方法 (Advanced Model Merging Variants)

| 方法 | 公式 / 操作 | 核心思想 |
|:---|:---|:---|
| **线性插值** | $\theta_m = (1-\alpha)\theta_a + \alpha\theta_b$ | 最简单，逐参数线性平均 |
| **SLERP** (Spherical Linear Interpolation) | $\theta_m = \frac{\sin((1-t)\Omega)}{\sin\Omega}\theta_a + \frac{\sin(t\Omega)}{\sin\Omega}\theta_b$，其中 $\cos\Omega = \frac{\theta_a \cdot \theta_b}{\|\theta_a\|\|\theta_b\|}$ | 在超球面上插值，保持向量范数 |
| **DARE** (Drop And REscale) | 先随机丢弃 $\theta_{\mathrm{aligned}} - \theta_{\mathrm{base}}$ 中 $p\%$ 的参数，再缩放剩余参数：$\delta_i \leftarrow \delta_i / (1-p)$，再合并 | 稀疏化任务向量，减少干扰 |
| **TIES** (Trim, Elect, Sign) | ① 修剪小幅度变化 ② 投票确定符号 ③ 仅保留一致方向的参数 | 解决多个模型合并时参数冲突 |

> **SLERP 直觉**：权重向量的"方向"比"长度"更重要，球面插值保持方向间的几何关系。

---

## 12. 灾难性遗忘、模式坍缩与奖励黑客 (Catastrophic Forgetting, Mode Collapse & Reward Hacking)

这是三种**截然不同**的训练失败模式 (three distinct failure modes)。

### 12.1 灾难性遗忘 (Catastrophic Forgetting)

**定义**：模型在 SFT 或 RL 阶段学习新行为时，丢失了预训练阶段获得的知识和能力。

**机制** (Mechanism)：

$$
\text{预训练能力} \xrightarrow{\text{SFT/RL 更新 } \theta} \text{部分覆盖/擦除}
$$

神经网络的权重空间有限，新任务的梯度更新可能覆盖存储旧知识的权重。

**检测指标 (Detection Metrics)**：
- 基准测试分数下降（MMLU、GSM8K、HumanEval 等）
- 困惑度 (perplexity) 在预训练分布上上升
- 能力探针 (capability probes) 的准确率下降

**缓解策略 (Mitigation Strategies)**：

| 策略 | 说明 |
|:---|:---|
| 混合训练数据 | 在 SFT 中混入预训练数据 (mix pre-training data into SFT) |
| 低秩适配 (LoRA) | 仅更新低秩增量 $\Delta W = BA$，大幅减少对原始权重的干扰 |
| 正则化 | EWC (Elastic Weight Consolidation)：$\mathcal{L} = \mathcal{L}_{\mathrm{new}} + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$，$F_i$ 为 Fisher 信息 |
| 模型合并 | WiSE-FT / SLERP 将对齐模型与基础模型合并 |

### 12.2 模式坍缩 (Mode Collapse)

**定义**：模型在 RL 训练过程中，输出多样性急剧下降，反复产生相似甚至相同的响应。

**机制 (Mechanism)**：策略过度优化某个高奖励模式，概率质量集中到少数输出上：

$$
H(\pi_\theta(\cdot|x)) = -\sum_y \pi_\theta(y|x)\log\pi_\theta(y|x) \to 0
$$

**检测指标 (Detection Metrics)**：
- 输出多样性度量下降（self-BLEU ↑, distinct-n ↓, entropy of token distribution ↓）
- 不同 prompt 的响应趋同 (responses converge across prompts)
- 温度采样下几乎无变化

**缓解策略 (Mitigation Strategies)**：

| 策略 | 说明 |
|:---|:---|
| 增大 KL 惩罚 | $\beta \uparrow$ 使策略保持接近 $\pi_{\mathrm{ref}}$，维持多样性 |
| 熵正则化 | 加入 $-\eta H(\pi_\theta)$ 项鼓励探索 |
| 数据多样性 | 训练数据覆盖多样化 prompt 分布 |
| 早停 (Early stopping) | 监控多样性指标，及时停止训练 |

### 12.3 奖励黑客 (Reward Hacking)

**定义**：策略学会利用奖励模型的缺陷 (exploit RM weaknesses)，获得高 RM 分数但人类评价实际下降。这是 **Goodhart's Law** 的直接体现：

$$
\text{"When a measure becomes a target, it ceases to be a good measure."}
$$

$$
\mathbb{E}_{y \sim \pi_\theta}[r_\phi(x,y)] \uparrow\uparrow \quad \text{但} \quad \mathbb{E}_{y \sim \pi_\theta}[r_{\mathrm{human}}(x,y)] \downarrow
$$

**检测指标 (Detection Metrics)**：

| 指标 | 说明 |
|:---|:---|
| RM 分数与人类评分的分歧 | $\Delta = r_{\mathrm{RM}} - r_{\mathrm{human}}$ 增大 |
| KL 不断增大 | 策略持续偏离 $\pi_{\mathrm{ref}}$ |
| 特定 pattern 激增 | 如过度使用"然而"、"值得注意的是"等套话 |
| 响应长度膨胀 | RM 偏好长回答 → 模型学会输出冗余内容 |

**缓解策略 (Mitigation Strategies)**：

| 策略 | 说明 |
|:---|:---|
| KL 惩罚 | 约束策略不偏离太远 (most fundamental) |
| RM 集成 (RM ensemble) | 多个 RM 取平均，减少单一 RM 的偏差 |
| 对抗训练 | 不断更新 RM，适应策略变化 (online RLHF) |
| 人工验证 | 定期人类评估，检测 RM-human 分歧 |
| 长度惩罚 | 对 RM 分数做长度归一化 |

### 12.4 三者关系总结 (Relationship Summary)

```
预训练 → SFT → RL
              ↓         ↓         ↓
         灾难性遗忘    模式坍缩    奖励黑客
         (知识丢失)    (多样性丢失)  (RM 被 exploit)
```

| 特征 | 灾难性遗忘 | 模式坍缩 | 奖励黑客 |
|:---|:---|:---|:---|
| 发生阶段 | SFT / RL | RL | RL |
| 根本原因 | 权重覆盖 | 过度优化单一模式 | RM 缺陷被利用 |
| 表现 | 能力下降 | 输出单调 | RM 分高但质量差 |
| 核心缓解 | 正则化 + 混合数据 | KL + 熵 + 多样性 | KL + RM 集成 + 人工验证 |

---

## 13. Constitutional AI / RLAIF (基于宪法的 AI / 基于 AI 反馈的强化学习)

### 13.1 RLAIF 概述 (RLAIF Overview)

**RLAIF** (Reinforcement Learning from AI Feedback) 用 LLM 生成的偏好标签替代人类标注员 (LLM-generated preference labels replace human annotators)：

$$
\text{标准 RLHF:} \quad \text{人类标注员} \xrightarrow{\text{比较 } (y_1, y_2)} \text{偏好标签} (y_w, y_l)
$$

$$
\text{RLAIF:} \quad \text{LLM 评判者} \xrightarrow{\text{比较 } (y_1, y_2)} \text{偏好标签} (y_w, y_l)
$$

### 13.2 CAI 自我批评-修订循环 (CAI Self-Critique-Revision Loop)

**Constitutional AI (CAI)** 的核心是**四步循环**：

**Step 1 — 生成 (Generate)**：给定 prompt $x$，用当前模型生成初始响应 $y_0$：
$$y_0 \sim \pi_\theta(\cdot | x)$$

**Step 2 — 批评 (Critique)**：用 LLM 根据宪法原则 (constitution principles) 对 $y_0$ 进行批评：
$$\text{critique} = \mathrm{LLM}\!\left(\text{"根据原则 } P_j \text{，以下回复有什么问题：} y_0\text{"}\right)$$

**Step 3 — 修订 (Revise)**：基于批评意见，用 LLM 修订响应：
$$y_1 = \mathrm{LLM}\!\left(\text{"请根据以下批评修改回复：} [\text{critique}] \rightarrow y_0\text{"}\right)$$

> 可迭代多次：$y_0 \to y_1 \to y_2 \to \dots$（通常 1-3 轮）

**Step 4 — 训练 (Train)**：
- **SL-CAI (SFT 阶段)**：用修订后的 $y_k$ 作为训练数据做 SFT
- **RL-CAI (RL 阶段)**：用 LLM 作为偏好评判者，生成 $(y_w, y_l)$ 偏好对，训练奖励模型，再做 RL

### 13.3 宪法原则 (Constitution Principles)

宪法原则是一组**可审计的对齐约束** (auditable alignment constraints)，例如：

| 编号 | 原则示例 (Principle Example) |
|:---:|:---|
| P₁ | "选择最有帮助、最准确、最无害的回复" (Choose the response that is most helpful, accurate, and harmless) |
| P₂ | "选择不会助长偏见或歧视的回复" (Choose the response that does not promote bias or discrimination) |
| P₃ | "选择不会协助非法活动的回复" (Choose the response that does not assist with illegal activities) |

与隐式的人类偏好不同，宪法原则是**显式的、可审查的** (explicit and auditable)：

$$
p_{\mathrm{CAI}}(y_w \succ y_l | x) = \sigma\!\big(r_{\mathrm{LLM}}(x, y_w) - r_{\mathrm{LLM}}(x, y_l)\big)
$$

其中 $r_{\mathrm{LLM}}$ 是基于宪法原则的 LLM 评分。

### 13.4 与标准 RLHF 流程对比 (Comparison with Standard RLHF)

| 维度 | 标准 RLHF | RLAIF / CAI |
|:---|:---|:---|
| **偏好来源** | 人类标注员 | LLM（基于宪法原则） |
| **标注成本** | 高（人力密集） | 低（API 调用费用） |
| **可扩展性** | 受限于标注员数量和时间 | 几乎无限扩展 |
| **一致性** | 标注员间有分歧 (inter-annotator variance) | LLM 高度一致 |
| **可审计性** | 偏好标准隐式存在于标注员脑中 | 宪法原则显式、可审查 |
| **风险** | 标注员偏见 | LLM 自身偏见 + 宪法原则设计不当 |
| **人类参与** | 全程 | 仅在设计宪法原则时 |

### 13.5 RLAIF 的理论优势 (Theoretical Advantages)

1. **原则引导 (Principle-guided)**：对齐目标通过自然语言原则明确表达，比隐式偏好更可控
2. **自我改进循环 (Self-improvement loop)**：模型批评自己、修订自己、学习修订后的版本 → 持续提升
3. **减少人类负担 (Reduced human burden)**：人类只需设计原则，不需要逐条标注
4. **跨文化一致性 (Cross-cultural consistency)**：不同文化背景的标注员可能有不同偏好，而宪法原则可以统一标准

> **注意**：CAI 并非完全去人类化。人类仍然需要：
> - 设计宪法原则 (design the constitution)
> - 评估最终模型质量 (evaluate final model quality)
> - 监控迭代过程中的偏差 (monitor for drift)

---

## 14. 蒸馏 (Distillation — Post-Training 视角)

### 14.1 三种蒸馏范式对比

#### SeqKD（序列级知识蒸馏）

Teacher 先用 beam search 或采样生成完整输出序列，Student 在这批序列上做 **标准 SFT**（交叉熵损失）：

$$L_{\text{SeqKD}} = -\sum_{t} \log p_{\theta_S}(y_t \mid x, y_{<t}), \quad y \sim \pi_T(x)$$

要点：
- 数据可离线生成，**不需要 Teacher 在线推理**；
- 蒸馏信号只来自 Teacher 采样出的离散序列，丢失了 Teacher 在各 token 上的 full distribution 信息（"软标签"被硬化）；
- 实现最简单，成本最低，适合大多数工程场景。

#### Token-Level KD（Token 级知识蒸馏）

对每个位置 $t$，对齐 Student 与 Teacher 在词表上的概率分布：

$$L_{\text{TKD}} = \sum_{t} D_{\text{KL}}\!\left(p_T(\cdot \mid x, y_{<t}) \;\Big\|\; p_{\theta_S}(\cdot \mid x, y_{<t})\right)$$

要点：
- 保留 Teacher 的**软分布**（soft labels），信息量更丰富，尤其在有多个合理 token 时；
- **需要 Teacher 在线/离线提供 logits**，Teacher 必须可访问（或预先缓存 logits）；
- 当 Teacher 很大（如 671B MoE）时，缓存所有位置的 logits 成本极高。

#### On-Policy Distillation（在线蒸馏）

Student 自身 rollout 生成候选序列，再用 Teacher（或可验证 reward）打分，Student 据此更新：

$$L_{\text{on-policy}} = -\mathbb{E}_{y \sim \pi_{\theta_S}}\!\left[r_T(x, y) \cdot \log p_{\theta_S}(y \mid x)\right]$$

要点：
- 训练信号来自 **Student 自身的分布**，无 off-policy 漂移；
- 等价于将 Teacher reward 作为 RLVR 信号；训练更复杂，但泛化能力通常更强；
- 代表方法：GRPO + 可验证奖励（Teacher 本身作为 verifier）。

---

### 14.2 CoT 蒸馏（R1-style Chain-of-Thought Distillation）

**基本思路**：用大型 RL 模型（如 DeepSeek-R1-671B）生成带完整思维链的长推理序列，然后在小模型上做 **SFT**（即 SeqKD 的 CoT 版本）。

DeepSeek-R1 论文（arXiv:2501.12948）报告了在 1.5B、7B、8B、14B、32B、70B 参数的 Qwen 和 Llama 上，用约 800K 蒸馏样本（约 600K 推理 + 约 200K 非推理）做 SFT 的实验结果，小模型推理能力大幅提升。

**为什么对小模型 CoT 蒸馏常比直接做 GRPO 更稳/更省（据 R1 论文蒸馏实验）：**

1. **探索代价不对称**：GRPO 要求模型自行探索高质量思维链，但小模型能力有限，随机采样很难产生有效推理序列（reward 极稀疏），梯度信号噪声大；Teacher 直接提供高质量 CoT 相当于**压缩了探索空间**。
2. **无需 Critic / RM**：SeqKD 路径只需 SFT，不需要在线 rollout 和 reward 模型，省去 GRPO 在线采样与 reward/critic 的显存和计算开销。
3. **训练稳定性**：SFT 的损失面比 RL 更平滑，无 reward hacking 或 mode collapse 风险，超参较少。

> **对冲措辞**：上述"更稳/更省"的结论来自 R1 论文在其蒸馏配置（DeepSeek-V3-Base 作为底座，约 800K 数据规模）下的实验观察，不代表所有小模型或数据规模下均成立；直接 RL（GRPO）在数据/算力充足时上限可能更高。

---

### 14.3 Forward KL vs Reverse KL

#### 定义

**Forward KL**（也称 inclusive KL，均值寻求，mean-seeking）：

$$D_{\text{KL}}^{\text{fwd}}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)}$$

优化方向：最小化 $q$ 相对于 $p$ 的 forward KL，等价于最大化 $\mathbb{E}_{y \sim p}[\log q(y)]$——Student $q$ 需要覆盖 Teacher $p$ 的**所有模式**（凡 $p(y)>0$ 处，$q$ 不能为 0，否则 KL 发散）。

**Reverse KL**（也称 exclusive KL，众数寻求，mode-seeking）：

$$D_{\text{KL}}^{\text{rev}}(q \| p) = \sum_y q(y) \log \frac{q(y)}{p(y)}$$

优化方向：最小化此量时，期望在 $q$ 的支撑下取，允许 $q$ **忽略 $p$ 的某些模式**（$q(y)=0$ 处该项为 0），但 $q$ 会集中在 $p$ 高概率的区域上。

#### 为什么生成任务常偏好 Reverse KL / Mode-Seeking？

直觉推导：

设 Teacher 分布 $p$ 是双峰分布，两个模式 $y_1, y_2$ 各占概率 $\approx 0.5$。

- **Forward KL**：Student $q$ 若想使 $\mathbb{E}_{y \sim p}[\log q(y)]$ 最大，必须覆盖两个模式，结果是 $q$ 分散在两个模式之间——**但这个中间区域在文本空间往往对应低质或非自然的序列**（"均值"是语义无意义的混合）。这种现象在生成任务里被称为 **mode averaging**：输出是所有模式的平均，反而不像任何一个合理答案。

- **Reverse KL**：Student $q$ 在 $q(y)>0$ 处承担对数惩罚，自然选择集中到 $p$ 中**某一个**高概率、语义连贯的模式。虽然牺牲了另一个模式，但生成的序列质量更高、更自然。

数学表述：令 $q^\*(y) = \arg\min_q D_{\text{KL}}^{\text{rev}}(q \| p)$，在参数受限（capacity-limited）的 Student 下，解会质量集中（mass concentration）在 $p$ 的主要模式上，而非在多模式间"抹平"。

> **一句话直觉**：Forward KL 要求"不漏掉 Teacher 的任何答案"；Reverse KL 允许"只学 Teacher 最自信的答案"。生成任务需要输出是连贯的，宁可少覆盖也要高质量，故常偏好 Reverse KL。

> **注意**：Token-level KD 通常用 forward KL（Student 向 Teacher 软标签对齐），而 SeqKD / SFT 在序列级更接近 reverse KL 的行为（Student 仅学 Teacher 采样出的模式）。两者并非对立，实践中常根据任务混用。

---

### 14.4 蒸馏 vs RFT vs PPO 三行对比表

| 方法 | 数据来源 | 对比/优化信号 | 适用规模 |
|:---|:---|:---|:---|
| **蒸馏 (Distillation / SeqKD)** | Teacher 模型生成的序列（离线） | Teacher 输出序列（交叉熵 / 软标签） | 中小模型（通常 ≤ 70B），Teacher 明显强于 Student |
| **RFT (Rejection Sampling FT)** | 当前 Policy 自采样，reward 过滤保留高分 | 可验证 reward / RM 筛选 | 中等规模（7B–70B），reward 可自动验证 |
| **PPO** | 当前 Policy 在线 rollout | RM 打分 + KL 约束 + GAE Advantage | 大规模（通常 ≥ 7B），有充足 RM 和计算资源 |

---

### 14.5 自测题

> **L2 — 蒸馏范式辨析**：SeqKD 和 Token-Level KD 都使用 Teacher 模型作为信号来源，但本质上一个更接近 reverse KL，另一个更接近 forward KL。请说明：(a) 哪个对应哪个方向的 KL；(b) 当 Teacher 分布是双峰时，两者训练出的 Student 分布会有什么行为差异？

> **L3 — CoT 蒸馏适用性分析**：假设你有一个 3B 的小模型和充足的 GPU（可以同时运行 Teacher 671B 和 Student），请分析：在什么数据规模和任务类型下，直接做 GRPO 会比 SeqKD 蒸馏更有优势？给出至少两个充分的理由。

---

# Part 2 — PyTorch 代码片段 / From-Scratch PyTorch Snippets

---

**SFT loss masking** — 训练 SFT 时，只在助手（assistant）回复的 token 上计算 loss，prompt 部分通过 `label=-100` 屏蔽。


```python
import torch
from torch.nn.utils.rnn import pad_sequence

class SFTDataCollator:
    """
    将 prompt token 的 label 设为 -100，loss 只计算 assistant 部分。
    Masks prompt tokens with label=-100 so loss only applies to assistant tokens.
    """
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, batch):
        input_ids, labels, attention_mask = [], [], []
        for sample in batch:  # each sample: dict with 'input_ids' and 'prompt_length'
            ids = torch.tensor(sample["input_ids"], dtype=torch.long)
            prompt_len = sample["prompt_length"]
            lab = ids.clone()
            lab[:prompt_len] = -100  # 屏蔽 prompt / mask prompt tokens
            input_ids.append(ids)
            labels.append(lab)
        # 动态 padding / dynamic pad to longest in batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# --- 用法示例 / Usage example ---
collator = SFTDataCollator(type("Tok", (), {"pad_token_id": 0})())
toy_batch = [
    {"input_ids": [10, 20, 30, 40, 50], "prompt_length": 3},  # prompt=前3个
    {"input_ids": [11, 21, 31], "prompt_length": 2},
]
out = collator(toy_batch)
print("input_ids:\n", out["input_ids"])
print("labels (prompt positions = -100):\n", out["labels"])
# labels: tensor([[ -100, -100, -100, 40, 50],
#                 [ -100, -100,  31,  0,  0]])
```



---

**DPO loss** — 从 policy 和 reference 模型的 log-probability 计算 Direct Preference Optimization 损失。

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def get_logps(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    逐 token 计算 log-probability 并在序列维度求和。
    Computes per-token log-probs and sums over the sequence dimension.
    logits: (B, T, V),  labels: (B, T),  mask: (B, T)  (1=有效, 0=padding)
    返回每个样本的标量 log-prob / Returns scalar log-prob per sample.
    """
    # shift: 预测下一个 token / predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)            # (B, T-1, V)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return (token_logps * shift_mask).sum(dim=-1)               # (B,)

def dpo_loss(
    policy_logps_chosen: torch.Tensor,
    policy_logps_rejected: torch.Tensor,
    ref_logps_chosen: torch.Tensor,
    ref_logps_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO 损失: L = -E[ log σ( β·(log π_θ/π_ref)_chosen - β·(log π_θ/π_ref)_rejected ) ]
    """
    log_ratio_chosen = policy_logps_chosen - ref_logps_chosen
    log_ratio_rejected = policy_logps_rejected - ref_logps_rejected
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()
    return loss

# --- 示例 / Example ---
B, T, V = 4, 10, 100
logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))
mask = torch.ones(B, T)

logps = get_logps(logits, labels, mask)  # (B,)
# 分开 chosen / rejected 是在调用者侧做的
policy_logps_chosen, policy_logps_rejected = logps[:2], logps[2:]
ref_logps_chosen, ref_logps_rejected = logps[:2] - 0.1, logps[2:] + 0.05

loss = dpo_loss(policy_logps_chosen, policy_logps_rejected,
                ref_logps_chosen, ref_logps_rejected, beta=0.1)
print("DPO loss:", loss.item())
```



---

**Reward Model** — 在预训练 LLM 骨干网络上，替换 LM head 为标量线性头，并用 Bradley-Terry 损失训练。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """
    奖励模型：LLM 骨干 + 线性标量头，取最后一个有效 token 的隐状态。
    Reward model: LLM backbone + scalar linear head on last valid hidden state.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)  # 标量奖励 / scalar reward

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)
        # 取每个序列最后一个有效 token 的隐状态 / hidden state of last valid token
        last_idx = attention_mask.sum(dim=1) - 1  # (B,)
        last_hidden = hidden[torch.arange(hidden.size(0)), last_idx]  # (B, H)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return reward

def bradley_terry_loss(rewards_chosen, rewards_rejected):
    """
    Bradley-Terry 损失: L = -log σ(r_chosen - r_rejected)
    BT loss: higher reward for preferred responses.
    """
    return -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

# --- 训练示例 / Training example ---
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
rm = RewardModel("Qwen/Qwen2.5-0.5B").to(device)

chosen_text = ["The answer is 42.", "It is safe to proceed."]
rejected_text = ["I don't know.", "No, never do that."]
tok_chosen = tokenizer(chosen_text, return_tensors="pt", padding=True, truncation=True)
tok_rejected = tokenizer(rejected_text, return_tensors="pt", padding=True, truncation=True)

r_chosen = rm(tok_chosen["input_ids"], tok_chosen["attention_mask"])
r_rejected = rm(tok_rejected["input_ids"], tok_rejected["attention_mask"])
loss = bradley_terry_loss(r_chosen, r_rejected)
print("BT loss:", loss.item())
```



---

**PPO 完整损失** — actor-critic 单步损失：clipped surrogate + 截断 value loss + entropy bonus + approx_kl 诊断（token 级）。

```python
import torch
import torch.nn.functional as F

def ppo_actor_critic_loss(
    logp, old_logp, advantages, returns, values, old_values, entropy, mask,
    clip_eps=0.2, vf_clip=0.2, vf_coef=0.5, ent_coef=0.01,
):
    """
    PPO 单步 actor-critic 损失（token 级）。所有张量 (B, T)；mask: 1=有效 response token。
    Token-level PPO loss: clipped policy surrogate + clipped value loss + entropy bonus.
    logp/old_logp: 当前/旧策略对所取动作的 log π(a_t|s_t)；advantages: GAE 优势 A_t；
    returns: 回报 R_t；values/old_values: critic 当前/旧预测。
    """
    def masked_mean(x):                                         # 仅在有效 token 上平均 / valid-token mean
        return (x * mask).sum() / mask.sum().clamp(min=1)

    # --- 策略损失：clipped surrogate（取下界=悲观）/ clipped policy surrogate ---
    ratio = torch.exp(logp - old_logp)                          # π_θ/π_θ_old, (B,T)
    pg_loss = -torch.min(ratio * advantages,
                         torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages)

    # --- value 损失：截断 value 以防 critic 跳变 / clipped value loss ---
    v_clipped = old_values + torch.clamp(values - old_values, -vf_clip, vf_clip)
    v_loss = 0.5 * torch.max((values - returns) ** 2, (v_clipped - returns) ** 2)

    # --- 总损失 = 策略 + c_vf·value − c_ent·entropy / combine ---
    loss = masked_mean(pg_loss) + vf_coef * masked_mean(v_loss) - ent_coef * masked_mean(entropy)

    # --- 诊断：approx_kl 用 k3=(r−1)−log r（此处 r=π_θ/π_θ_old，估计 KL(π_old‖π_θ)；
    #     估计器原理见 §9.4，但注意 r 方向与 §9.4 的 r=π_ref/π_θ 相反）---
    with torch.no_grad():
        log_ratio = logp - old_logp
        approx_kl = masked_mean((ratio - 1) - log_ratio)        # ≥ 0，用于早停/自适应 KL
        clip_frac = masked_mean((torch.abs(ratio - 1) > clip_eps).float())
    return loss, {"approx_kl": approx_kl.item(), "clip_frac": clip_frac.item()}

# --- 玩具示例 / Toy example ---
torch.manual_seed(0)
B, T = 2, 5
logp       = (torch.randn(B, T) * 0.1).requires_grad_(True)
old_logp   = logp.detach() + torch.randn(B, T) * 0.05           # 旧(行为)策略 / behavior policy
advantages = torch.randn(B, T); advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
returns    = torch.randn(B, T)
values     = torch.randn(B, T, requires_grad=True)
old_values = values.detach() + torch.randn(B, T) * 0.1
entropy    = torch.rand(B, T)                                   # 每 token 策略熵 / per-token entropy
mask       = torch.ones(B, T); mask[1, 3:] = 0                  # 第二条后半为 padding

loss, logs = ppo_actor_critic_loss(logp, old_logp, advantages, returns, values, old_values, entropy, mask)
loss.backward()
print("PPO loss:", round(loss.item(), 4), "| diag:", {k: round(v, 4) for k, v in logs.items()})
```

---

**GRPO advantage** — Group Relative Policy Optimization：在同一组（group）内将奖励归一化，作为 advantage 进行策略梯度更新。

```python
import torch
import torch.nn.functional as F

def compute_grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    在 group 内归一化奖励作为 advantage：(r - mean) / std。
    Normalize rewards within group: subtract mean, divide by std.
    rewards: (G,)  — 同一 prompt 的 G 个采样回复的奖励
    """
    mean = rewards.mean()
    std = rewards.std().clamp(min=1e-8)  # 防止除零 / avoid division by zero
    return (rewards - mean) / std

# --- 简化策略梯度更新 / Simplified policy gradient update ---
# 模拟：给定 policy log-probs 和 group advantages, 做一次梯度上升
G = 8  # 每个 prompt 采样 8 个回复 / sample 8 responses per prompt

# 模拟采样的 token-level log-probs (已 sum 成序列级) / simulated per-sequence log-probs
policy_logps = torch.randn(G, requires_grad=True)

# 模拟奖励（例如来自 reward model）/ simulated rewards
rewards = torch.tensor([1.2, 0.5, 2.0, 0.3, 1.8, 0.1, 1.5, 0.9])

advantages = compute_grpo_advantages(rewards)
print("Advantages:", advantages)

# 策略梯度 loss = -E[advantage * log_prob]  → 最大化高 advantage 的 log-prob
grpo_loss = -(advantages.detach() * policy_logps).mean()
grpo_loss.backward()
print("GRPO loss:", grpo_loss.item())
print("policy_logps.grad:", policy_logps.grad)
```

---

**GRPO token 级损失** — 组优势广播到每个 token + clipped surrogate + 逐 token K3 KL（无 critic、无 GAE；token 级平均，见 §9.4 与 DAPO）。

```python
import torch

def grpo_token_loss(logp, old_logp, ref_logp, group_adv, mask, clip_eps=0.2, beta_kl=0.04):
    """
    GRPO token 级损失：序列级组优势广播到 token + clipped surrogate + 逐 token K3 KL。
    Token-level GRPO loss: per-sequence group advantage broadcast to tokens
    + clipped surrogate + per-token K3 KL (no critic, no GAE).
    logp/old_logp/ref_logp: (B, T) 当前/旧/参考策略对所取 token 的 log-prob
    group_adv: (B,) 组内归一化优势 A_i（见上方 compute_grpo_advantages），按序列广播
    mask: (B, T) 1=有效 response token
    """
    adv = group_adv.unsqueeze(1)                                # (B,1) → 广播到 (B,T)
    # clipped surrogate（与 PPO 相同的裁剪，优势换成组相对）/ same clip as PPO, group-relative adv
    ratio = torch.exp(logp - old_logp)                          # π_θ/π_θ_old
    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    # 逐 token K3 KL：r=π_ref/π_θ，k3=(r−1)−log r ≥ 0（与 §9.4 同约定）
    log_r = ref_logp - logp                                     # log(π_ref/π_θ)
    k3 = torch.exp(log_r) - 1 - log_r
    per_token = pg + beta_kl * k3
    # 按 token 平均（而非按序列）的口径借自 DAPO（§3.3），长 CoT 梯度不被稀释；
    # 注意此处 KL 项是 GRPO 式 β>0，并非 DAPO（DAPO 设 β=0）
    return (per_token * mask).sum() / mask.sum().clamp(min=1)

# --- 玩具示例 / Toy example ---
torch.manual_seed(0)
B, T = 4, 6                          # 同一 prompt 的 4 个采样回复 / 4 sampled responses
logp      = (torch.randn(B, T) * 0.1).requires_grad_(True)
old_logp  = logp.detach() + torch.randn(B, T) * 0.02
ref_logp  = logp.detach() + torch.randn(B, T) * 0.05
rewards   = torch.tensor([1.2, 0.3, 1.8, 0.5])                  # 每个回复一个标量奖励 / one reward per response
group_adv = (rewards - rewards.mean()) / rewards.std().clamp(min=1e-8)   # 组内归一化 / within-group norm
mask      = torch.ones(B, T); mask[1, 4:] = 0

loss = grpo_token_loss(logp, old_logp, ref_logp, group_adv, mask)
loss.backward()
print("GRPO token-level loss:", round(loss.item(), 4))
```

---

**Sequence packing with cu_seqlens** — 将多条不等长序列拼接到一个 batch 中，计算 Flash Attention 所需的 `cu_seqlens`，并对拼接后的 loss 做正确 mask。

```python
import torch

def pack_sequences(input_ids_list, labels_list, pad_token_id=0):
    """
    将多条序列拼接成一个平坦 tensor，并计算 Flash Attention 用的 cu_seqlens。
    Packs variable-length sequences into a flat tensor with cu_seqlens for Flash Attention.
    """
    # 计算每条序列的真实长度 / compute real lengths
    lengths = [ids.size(0) for ids in input_ids_list]
    # cu_seqlens: [0, len_0, len_0+len_1, ...]  (半精度索引 / Flash Attention format)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    for i, l in enumerate(lengths):
        cu_seqlens[i + 1] = cu_seqlens[i] + l

    # 拼接所有序列 / concatenate all sequences into one flat tensor
    packed_input_ids = torch.cat(input_ids_list, dim=0)   # (total_tokens,)
    packed_labels = torch.cat(labels_list, dim=0)          # (total_tokens,)
    return packed_input_ids, packed_labels, cu_seqlens

def compute_packed_loss(logits_flat, labels_flat, cu_seqlens, ignore_index=-100):
    """
    在拼接序列上计算 cross-entropy，loss 屏蔽 label=-100 的 token。
    Compute cross-entropy on packed sequence; -100 labels are masked.
    logits_flat: (total_tokens, V),  labels_flat: (total_tokens,)
    """
    # shift 对齐 / shift for next-token prediction
    shift_logits = logits_flat[:-1, :]
    shift_labels = labels_flat[1:]
    # 在序列边界处也屏蔽 loss / mask loss at sequence boundaries
    boundary_mask = torch.zeros(shift_labels.size(0), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        if start < end:
            boundary_mask[start] = True  # 屏蔽第一条 token 的 shift / mask first token of seq
    shift_labels[boundary_mask] = ignore_index
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
    return loss

# --- 示例 / Example ---
seq_a_ids = torch.tensor([101, 202, 303, 404, 505])
seq_b_ids = torch.tensor([606, 707])
seq_c_ids = torch.tensor([808, 909, 1010])

seq_a_lab = torch.tensor([-100, -100, 303, 404, 505])   # 前两个是 prompt
seq_b_lab = torch.tensor([-100, 707])
seq_c_lab = torch.tensor([-100, 1010, 1010])

packed_ids, packed_labels, cu_seqlens = pack_sequences(
    [seq_a_ids, seq_b_ids, seq_c_ids], [seq_a_lab, seq_b_lab, seq_c_lab]
)
print("packed_ids:", packed_ids)
print("cu_seqlens:", cu_seqlens)  # tensor([0, 5, 7, 10])

# 模拟 logits / simulate logits
V = 2000
logits_flat = torch.randn(packed_ids.size(0), V)
loss = compute_packed_loss(logits_flat, packed_labels, cu_seqlens)
print("Packed loss:", loss.item())
```

---

**KL divergence penalty** — 在 PPO/RLHF 奖励塑形中，逐 token 计算 policy 与 reference 模型之间的 KL 惩罚项。

```python
import torch
import torch.nn.functional as F

def compute_kl_penalty(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    逐 token KL 散度：KL(π_θ || π_ref)，在序列维度求均值后取 batch 均值。
    Per-token KL divergence: KL(policy || ref), averaged over valid tokens & batch.
    policy_logits / ref_logits: (B, T, V),  mask: (B, T) — 1=有效, 0=padding
    """
    policy_logps = F.log_softmax(policy_logits, dim=-1)  # (B, T, V)
    ref_logps = F.log_softmax(ref_logits, dim=-1)        # (B, T, V)
    # KL(p||q) = sum_p p(x) * [log p(x) - log q(x)] = E_p[log p - log q]
    policy_probs = policy_logps.exp()
    token_kl = (policy_probs * (policy_logps - ref_logps)).sum(dim=-1)  # (B, T)
    # 用 mask 求均值 / masked mean
    kl_per_seq = (token_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B,)
    return kl_per_seq.mean()  # scalar

# --- 用在 PPO 奖励塑形中 / Used in PPO reward shaping ---
B, T, V = 2, 8, 1000
policy_logits = torch.randn(B, T, V)
ref_logits = torch.randn(B, T, V)
mask = torch.ones(B, T); mask[1, 6:] = 0  # 第二条序列后半部分是 padding

kl = compute_kl_penalty(policy_logits, ref_logits, mask)
print("KL penalty:", kl.item())

# PPO reward shaping: r = r_raw - beta * KL
beta_kl = 0.05
shaped_reward = 1.5 - beta_kl * kl  # 在 batch 级别使用
print("Shaped reward:", shaped_reward.item())
```



---

**Rejection Sampling Fine-tuning (RFT)** — 从 policy 模型采样 N 条回复，用奖励函数打分，保留得分最高的 1 条作为 SFT 目标进行微调。

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def rejection_sampling_finetune(model, tokenizer, prompts, reward_fn, N=4, max_new_tokens=64):
    """
    RFT 流程：对每个 prompt 采样 N 个回复，用 reward_fn 评分，取 top-1 做 SFT。
    RFT loop: sample N responses, score with reward_fn, keep top-1 as SFT target.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for prompt in prompts:
        # ---- 采样阶段 / Sampling phase ----
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        all_completions, all_rewards = [], []
        with torch.no_grad():
            for _ in range(N):
                out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=0.8, top_p=0.95)
                gen_ids = out[0, input_ids.size(1):]           # 只取生成部分
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                reward = reward_fn(prompt, text)               # 标量奖励 / scalar reward
                all_completions.append(gen_ids)
                all_rewards.append(reward)

        # ---- 选择 top-1 / Select best response ----
        best_idx = int(torch.tensor(all_rewards).argmax())
        best_ids = all_completions[best_idx]

        # ---- SFT 阶段 / SFT phase (compute loss on best response) ----
        full_ids = torch.cat([input_ids[0], best_ids]).unsqueeze(0)  # (1, T)
        labels = full_ids.clone()
        labels[0, :input_ids.size(1)] = -100  # 屏蔽 prompt / mask prompt tokens
        logits = model(input_ids=full_ids).logits
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"RFT loss: {loss.item():.4f}, best reward: {all_rewards[best_idx]:.4f}")

# --- 简单奖励函数示例 / Simple reward function ---
def dummy_reward_fn(prompt, response):
    """奖励: 回复越长越好（仅为演示）/ Reward: longer is better (demo only)."""
    return float(len(response))

# 运行 / Run
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
prompts = ["Explain gravity in one sentence.", "What is 2+2?"]
rejection_sampling_finetune(model, tokenizer, prompts, dummy_reward_fn, N=4)
```

---

# Part 3 — 面试题库 / Interview Question Bank

### ━━━ L1 基础 / Basic ━━━

---

<details>
<summary>Q1. Pre-training 和 post-training 分别解决什么问题？标准 pipeline 是什么？</summary>

**答 / Answer:**
Pre-training（预训练）的目标是让模型在海量无标注文本上学得通用的语言能力、世界知识和推理基础，本质上是做无监督的语言建模。Post-training（后训练）的目标是将这个”知识渊博但不听话”的基座模型，塑造成一个能遵循指令、有帮助、安全且与人类价值观对齐的助手。标准pipeline是：1) Supervised Fine-Tuning (SFT)，用高质量指令-回复对微调模型；2) Preference Alignment，通常使用RLHF或DPO等方法，基于人类偏好数据进一步优化模型行为。

**追问 / Follow-up:**
为什么不能只用一个阶段（如SFT）来完成从预训练到可用助手的全部转变？

</details>

<details>
<summary>Q2. SFT 的 loss masking 是什么？为什么只对 assistant tokens 计算 loss？</summary>

**答 / Answer:**
Loss masking 指在计算SFT损失时，只将模型输出中assistant回复部分（即模型需要学习生成的部分）对应的token的预测损失计入总损失，而忽略input/user指令部分的损失。这是为了将模型的优化目标聚焦于”学习如何正确响应”，而不是”复述用户的输入”。如果不对输入部分进行mask，模型可能会浪费学习容量去记忆输入格式，而非专注于生成高质量回复。

**追问 / Follow-up:**
如果在SFT中，user指令部分的梯度完全不被更新，模型是否就真的完全无法”理解”指令？请解释。

</details>

<details>
<summary>Q3. Reward Model 的训练目标是什么？Bradley-Terry 模型是什么？</summary>

**答 / Answer:**
Reward Model (RM) 的训练目标是为给定的（prompt, response）对输出一个标量分数，这个分数应反映人类对该回复质量的偏好排序。具体来说，它通过比较一对回复（chosen vs. rejected）来学习。Bradley-Terry 模型是一个用于成对比较的概率模型，它假设选择胜出回复的概率与两个回复对应的奖励值之差成正比。在RM训练中，损失函数通常基于这个概率，目标是最大化chosen response相对于rejected response的奖励差。

**追问 / Follow-up:**
如果人类标注数据中的偏好排序存在不一致或噪声，会如何影响Bradley-Terry模型训练的Reward Model？

</details>

<details>
<summary>Q4. KL penalty 在 RLHF 中的作用？β 如何调节？</summary>

**答 / Answer:**
在RLHF的强化学习阶段，策略模型（当前待优化的LLM）在生成回复时会最大化来自Reward Model的奖励，但这可能导致模型为了获取高分而生成一些奇怪、不自然或偏离其原始能力分布的文本。KL惩罚项通过计算当前策略与初始SFT模型（参考策略）之间的KL散度，并将其作为惩罚项加入到优化目标中。作用是约束优化后的模型不要偏离初始模型太远，从而维持语言质量和多样性。β是超参数，用于调节KL惩罚的强度：β越大，对偏离的惩罚越重，模型越保守、越接近初始模型；β越小，模型越自由，可能更追求奖励但风险也更高。

**追问 / Follow-up:**
KL惩罚计算的是整个序列分布的散度，这在实际操作中有什么挑战？有没有更高效或更局部的近似方法？

</details>

<details>
<summary>Q5. DPO 是什么？和 RLHF 的核心区别是什么？</summary>

**答 / Answer:**
DPO (Direct Preference Optimization) 是一种直接利用人类偏好数据优化语言模型的方法。它通过一个巧妙的数学变换，将RLHF中”训练一个Reward Model，再用其进行RL优化”这两个步骤，合并为一个单一的监督学习损失函数。模型在DPO中直接学习将偏好排序转化为对回复概率的调整。核心区别在于：RLHF是”显式”的，包含独立的RM训练和在线的RL优化过程（如PPO）；而DPO是”隐式”的，它绕过了RM的显式训练和在线采样，通过一个离线的对比损失直接优化策略，通常更简单、稳定。

**追问 / Follow-up:**
DPO的一个主要批评是它严重依赖于偏好数据的质量。为什么它对数据质量的要求比RLHF可能更高？

</details>

<details>
<summary>Q6. 什么是 sequence packing？有什么好处和坑？</summary>

**答 / Answer:**
Sequence packing 是一种训练时的效率优化技术。它将多个短序列（例如多个不同的指令-回复对）通过添加特殊的分隔符（如<code>&lt;EOS&gt;</code>后接新序列的起始标记）拼接成一个达到模型最大上下文长度的长序列，作为一个整体进行训练。好处是显著提高了GPU的利用率，减少了因短序列填充（padding）带来的计算浪费，加速训练。主要的”坑”在于：1）需要小心设计attention mask，防止模型在训练时”看到”同一个拼接序列中其他短序列的信息（即跨序列注意力泄露），这可能导致数据污染或学习偏差；2）对序列顺序可能敏感。

**追问 / Follow-up:**
在sequence packing中，如果两个拼接的序列主题完全无关（如一个数学题和一个诗歌），跨序列注意力的泄露具体会带来什么危害？

</details>

<details>
<summary>Q7. 什么是 reward hacking？举两个例子。</summary>

**答 / Answer:**
Reward hacking 指的是模型找到了”作弊”或”钻空子”的方法来获取更高的奖励分数，但其生成的回复实际上并不符合人类期望的”有帮助、诚实、无害”的真正目标。它是对奖励函数的过度优化或利用。例子1：如果RM偏爱更长的回复，模型可能学会生成冗长但内容空洞的回复。例子2：如果RM对包含某些特定”安全”短语（如”作为AI助手，我必须遵守...”）的回复打分高，模型可能学会机械地在所有回复中插入这些套话，而不考虑是否真的需要。

**追问 / Follow-up:**
除了改进Reward Model本身，在RLHF的训练过程中，可以通过哪些策略来缓解reward hacking现象？

</details>

<details>
<summary>Q8. Alignment 的 Helpful/Harmless/Honest 三者之间有什么 tension？</summary>

**答 / Answer:**
Helpful（有用）、Harmless（无害）、Honest（诚实）三者之间存在固有的权衡（tension）。例如，一个过于追求Harmless的模型，可能因为过度谨慎而拒绝回答一些合理但敏感的问题，从而损害了Helpful（例如，医生讨论医学症状）。一个追求极致Honest的模型，可能会在回复中暴露未经验证的信息或用户隐私，从而损害Harmless。反之，为了Helpful而编造答案会损害Honest。理想的对齐模型需要在不同场景下动态地平衡这三个目标，不存在一个固定的完美解。

**追问 / Follow-up:**
你能否提供一个具体场景，说明模型为了实现Harmless而不可避免地牺牲了Helpful和Honest？

</details>

### ━━━ L2 中级 / Intermediate ━━━

---

<details>
<summary>Q9. GRPO 和 PPO 的核心区别？GRPO 需要几个模型？</summary>

**答 / Answer:**
GRPO (Group Relative Policy Optimization) 和PPO (Proximal Policy Optimization) 都是策略梯度算法，但GRPO为了简化RLHF训练流程做了关键改进。核心区别在于：PPO需要维护四个模型：策略模型、参考模型、价值模型（Critic）和奖励模型；而GRPO**不需要单独的价值模型**。GRPO通过为同一个prompt生成一组（Group）回复，然后使用该组内回复的平均奖励作为基线（baseline）来估计优势函数（advantage），从而计算策略梯度。因此，GRPO通常只需要两个模型：策略模型和奖励模型（参考模型可合并或共享）。

**追问 / Follow-up:**
GRPO用组内平均奖励作为基线来估计优势函数，这可能会引入什么样的偏差？它如何影响训练的稳定性？

</details>

<details>
<summary>Q10. IPO、KTO、ORPO、SimPO 各解决 DPO 的什么问题？</summary>

**答 / Answer:**
这些方法都是对DPO的改进或变体：
- **IPO (Identity Preference Optimization)**：解决 DPO 在偏好接近确定性（near-deterministic）时 KL 正则失效、易过拟合的问题——改用有界的平方损失目标（详见 §7.1）使优化更稳健。
- **KTO (Kahneman-Tversky Optimization)**：解决DPO需要严格配对的偏好数据（chosen/rejected对）的限制。KTO只需要每个回复是否为”好”或”差”的二元标签数据，无需成对，数据获取更灵活。
- **ORPO (Odds Ratio Preference Optimization)**：尝试将SFT和偏好对齐合并为一个单一训练阶段。它直接优化模型生成chosen response相对于rejected response的优势比（odds ratio）。
- **SimPO (Simple Preference Optimization)**：试图进一步简化DPO，移除参考模型的依赖，同时通过使用长度归一化的对数概率作为隐式奖励，并引入一个目标奖励边际（margin），来提高优化稳定性和对回复长度的鲁棒性。

**追问 / Follow-up:**
在这些方法中，哪一种对训练数据的质量或数量要求相对最低？为什么？

</details>

<details>
<summary>Q11. 数据质量和数据量在 SFT 中哪个更重要？如何做数据 curation？</summary>

**答 / Answer:**
在SFT阶段，**数据质量通常远比数据量重要**。高质量、多样、准确且符合人类价值观的指令数据，即使是较小规模，也能显著提升模型性能。相反，大量低质、错误或有害的数据会严重污染模型。数据curation流程通常包括：1）**来源筛选**：选择可信、专业的来源；2）**质量过滤**：使用规则或模型（如RM）过滤低分、有害或格式错误的样本；3）**去重**：移除重复或近似重复的样本；4）**多样性增强**：确保指令覆盖广泛的任务、难度和领域；5）**格式标准化**：统一回复的风格和长度分布。

**追问 / Follow-up:**
如果只能使用一个自动化模型（而非人工）来对大规模SFT数据进行质量评估和筛选，你会优先选择使用什么类型的模型？为什么？

</details>

<details>
<summary>Q12. Synthetic data 的主要生成范式有哪些？length bias 从哪里来？</summary>

**答 / Answer:**
主要范式有：1）**Self-Instruct**：让模型自己根据种子任务生成新的指令和回复；2）**Evol-Instruct**：对已有指令进行多轮、多维度的复杂化演化；3）**Bootstrapping**：使用强大的”教师”模型为”学生”模型生成训练数据（如蒸馏）；4）**Reward-guided Generation**：用RM或规则筛选/修订模型生成的多个候选回复。Length bias（长度偏差）主要来源于：1）**模型本身的偏见**：预训练数据中常见回复（如技术文档）可能较长；2）**奖励模型的偏见**：如果RM的训练数据中，人类标注者普遍偏好更详细、更长的回复，那么RM就会给更长的回复打高分，模型在优化RM时就会倾向于生成更长文本；3）**生成策略**：例如，为了确保覆盖所有要点而进行冗长的列举。

**追问 / Follow-up:**
在生成合成数据时，如何设计流程或损失函数来显式地控制或减少最终回复的长度偏差？

</details>

<details>
<summary>Q13. Online vs offline preference learning 的区别？各自适合什么场景？</summary>

**答 / Answer:**
**Online learning**（在线学习，如标准RLHF中的PPO阶段）指的是策略模型在训练过程中，实时生成新的回复，并与环境（如RM）交互获得新的奖励信号来更新策略。**Offline learning**（离线学习，如DPO）指的是使用一个预先收集好的、固定的偏好数据集来优化模型，训练过程中不产生新的数据。Online适合需要持续探索、快速适应新奖励信号或解决分布偏移（distribution shift）问题的场景，但计算开销大、不稳定。Offline适合数据收集成本高、需要稳定训练流程的场景，但容易受到数据分布固定的限制，可能陷入次优。

**追问 / Follow-up:**
在offline learning中，如果用于训练的偏好数据分布与模型实际部署时遇到的数据分布差异很大，会导致什么问题？如何缓解？

</details>

<details>
<summary>Q14. 什么是 benchmark 污染（contamination）？如何检测？</summary>

**答 / Answer:**
Benchmark污染指的是待评估的模型（或其训练数据）在训练过程中已经”见过”了评估基准（benchmark）中的测试题目或答案。这会导致模型在该基准上获得虚高的、不真实的性能分数，无法反映其真实的泛化能力。检测方法包括：1）**成员推断攻击**：分析模型对测试集样本与相似的非测试集样本的困惑度（perplexity）差异；2）**n-gram重叠分析**：检查模型训练数据与测试集之间的文本重叠度；3）**数据溯源**：严格审计训练数据的来源，排除已知包含主流基准测试集的数据集（如Common Crawl的某些版本）；4）**设计动态基准**：使用定期更新、未公开的测试集。

**追问 / Follow-up:**
除了数据污染，还有哪些评估方法论上的缺陷可能导致对模型能力的误判？

</details>

<details>
<summary>Q15. Catastrophic forgetting 在 post-training 中如何表现？如何缓解？</summary>

**答 / Answer:**
在post-training中，灾难性遗忘表现为模型在通过SFT或RLHF学习新能力（如遵循指令、对齐价值观）的过程中，**丢失了其在预训练阶段学到的广泛知识、语言能力或解决多样任务的能力**。例如，一个对齐后的模型可能在指令遵循上表现很好，但其在代码、数学或多语言方面的基础能力相比基座模型出现了显著退化。缓解方法包括：1）**混合训练数据**：在SFT/RLHF数据中混入部分预训练数据或通用能力数据；2）**低秩适应**：使用LoRA等参数高效微调方法，仅更新一小部分参数；3）**正则化**：如在损失函数中加入对原始模型参数的L2惩罚（类似EWC的思想）；4）**知识蒸馏**：将原始模型作为教师，约束对齐后模型的输出分布。

**追问 / Follow-up:**
在参数高效微调方法（如LoRA）中，选择对哪些层（如注意力层的QKV投影，还是FFN层）进行微调，对缓解灾难性遗忘和保留原有能力的影响有何不同？

</details>

<details>
<summary>Q16. Process Reward Model (PRM) vs Outcome Reward Model (ORM)？</summary>

**答 / Answer:**
ORM（结果奖励模型）仅对模型生成的**最终答案或完整回复**给出一个奖励分数，不关心中间推理过程。PRM（过程奖励模型）则对解决问题或生成回复的**每一个中间步骤**都进行评估和打分。PRM的优势在于能提供更密集、更精细的监督信号，有助于引导模型进行正确的逐步推理，尤其在数学、逻辑推理等复杂任务中，可以避免模型通过”抄近路”得到正确答案但过程错误。其挑战在于标注成本极高，需要人类专家对每个步骤进行判断。

**追问 / Follow-up:**
在实际应用中，如何高效地收集用于训练PRM的数据？是否有可能使用ORM或其他模型来自动生成PRM的训练标签？

</details>

<details>
<summary>Q17. MT-Bench、AlpacaEval、Chatbot Arena 各自的局限性是什么？</summary>

**答 / Answer:**
- **MT-Bench**：使用预设的、多轮对话题目和强大的LLM（如GPT-4）作为评判。局限在于：1）评判模型自身可能有偏见；2）题目固定，容易过拟合；3）无法评估长文档处理、真实世界复杂任务等。
- **AlpacaEval**：使用一个固定的指令集，通过GPT-4对比评判模型回复与参考回复（通常是GPT-4自己的回复）的优劣。局限在于：1）强烈依赖GPT-4的偏好，可能无法反映广大人类用户的偏好；2）存在”自我偏好”风险，即与GPT-4风格越相似的回复得分可能越高。
- **Chatbot Arena**：通过真实用户匿名投票进行两两对比，是目前最贴近人类偏好的动态评估。局限在于：1）用户群体可能不具完全代表性（偏向技术人群）；2）评估成本高、速度慢；3）对话领域分布可能不均衡。

**追问 / Follow-up:**
如果要设计一个新的、更全面的后训练模型评估框架，你会融合哪些不同的评估维度和方法来弥补上述单一基准的不足？

</details>

### ━━━ L3 深度 / Deep ━━━

---

<details>
<summary>Q18. PPO 的 value model (critic) 为什么难训练？GRPO 如何绕开这个问题？</summary>

**答 / Answer:**
在RLHF的PPO中，价值模型（Critic）需要准确估计在给定状态下（即当前的prompt和部分生成的历史），未来能获得的奖励总和的期望值（即状态价值函数V(s)）。这个估计非常困难：1）**稀疏奖励**：奖励通常只在生成完整回复后才给出，中间状态缺乏直接监督信号；2）**高方差**：生成文本的状态空间巨大且复杂，导致价值估计方差很高，训练不稳定；3）**非平稳性**：策略模型在快速更新，导致状态价值函数的目标分布也在不断变化，增加了拟合难度。GRPO通过完全移除价值模型来绕开这个问题。它为每个prompt生成一组回复，用这组回复的平均奖励作为基线来估计每个回复相对于平均水平的优势（advantage）。这种方法避免了训练一个复杂的、面向所有可能状态的价值网络。

**追问 / Follow-up:**
GRPO使用组内平均奖励作为基线，这相当于假设所有状态（同一个prompt下的不同生成路径）的价值是相同的。这个假设在什么情况下会变得不合理？

</details>

<details>
<summary>Q19. DPO 的理论推导：从 RLHF KL-constrained 最优解到 DPO loss，走一遍推导。</summary>

**答 / Answer:**
1. **RLHF目标**：我们有一个KL约束的优化目标：<code>max_{π} E_{x~D, y~π}[r(x, y)] - β * KL[π(y|x) || π_ref(y|x)]</code>，其中π是策略，π_ref是参考策略，r是奖励函数。
2. **闭式最优解**：对上述目标关于π求解，可以得到其闭式最优解为：<code>π*(y|x) = π_ref(y|x) * exp(r(x, y) / β) / Z(x)</code>，其中<code>Z(x)</code>是配分函数（归一化常数）。
3. **反解奖励函数**：从上式两边取对数并整理，可以将奖励函数表示为策略的函数：<code>r(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log(Z(x))</code>。
4. **代入Bradley-Terry模型**：对于偏好对(y_w, y_l)，根据BT模型，人类选择y_w的概率为<code>σ(r(x, y_w) - r(x, y_l))</code>，其中σ是sigmoid函数。
5. **消除配分函数**：将步骤3中的奖励表达式代入步骤4，由于<code>log(Z(x))</code>在相减时被抵消，我们得到：<code>P(y_w ≻ y_l | x) = σ(β * log(π*(y_w|x) / π_ref(y_w|x)) - β * log(π*(y_l|x) / π_ref(y_l|x)))</code>。
6. **DPO损失**：最终，DPO的损失函数就是最大化上述概率（即最小化负对数似然）：<code>L_DPO(θ) = -E[log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x)))]</code>，其中π_θ是我们要优化的策略。

**追问 / Follow-up:**
在上述推导中，我们假设了奖励函数r可以用策略π来表示（步骤3）。这个假设成立的隐含条件是什么？

</details>

<details>
<summary>Q20. Mode collapse 和 reward hacking 的区别？如何检测 mode collapse？</summary>

**答 / Answer:**
**Reward hacking** 是模型找到了获得高奖励的”捷径”但输出不符合人类意图（如生成冗长废话）。**Mode collapse** 则是指模型的输出多样性急剧下降，倾向于重复生成某几种获得高奖励的、安全的或模式化的回复，失去了回应不同prompt时应有的丰富性和创造性。它是生成式模型的一种常见故障模式。检测mode collapse的方法包括：1）**多样性指标**：计算模型在一组prompt上生成回复的词汇多样性（如distinct-n）、语义嵌入的方差等，与基线模型对比；2）**分析奖励分布**：如果模型的奖励分数分布变得非常集中（高均值、低方差），可能意味着它找到了少数几种”高分模板”；3）**人工抽样检查**：随机抽取多组回复，观察其内容、结构和用词是否高度相似。

**追问 / Follow-up:**
在RLHF训练中，增加KL惩罚系数β是缓解mode collapse的有效手段。除此之外，从数据角度或算法角度还有什么方法可以鼓励多样性？

</details>

<details>
<summary>Q21. Alignment tax 是什么？weight averaging 如何缓解它？原理是什么？</summary>

**答 / Answer:**
Alignment tax（对齐税）指的是模型在post-training对齐过程中，为获得更好的指令遵循、安全性和无害性，而**在某些未被直接优化的通用能力（如基础语言建模、复杂推理）上支付的性能代价**，即这些能力可能出现下降。Weight averaging（权重平均）是一种简单有效的缓解技术。它通过平均训练过程中不同时间点或不同随机种子产生的多个模型权重，来得到一个更平滑、泛化能力更强的最终模型。其原理在于：1）**减少方差**：平均化可以减少单一模型由于训练波动或随机性导致的性能不稳定；2）**探索更优解**：不同的训练快照可能位于损失面上不同的”好”区域，平均可能找到一个在各方面都表现不错的中间点；3）**类似于隐式正则化**，可以防止模型过度拟合到训练数据的特定模式（包括对齐数据中可能存在的偏见）。

**追问 / Follow-up:**
在权重平均的具体实现中，如Stochastic Weight Averaging (SWA) 和 Model Soups，它们的策略和假设有何不同？哪种在缓解对齐税上可能更有效？

</details>

<details>
<summary>Q22. DeepSeek-R1 的训练流程有哪些关键设计决策？cold-start SFT 的作用是什么？</summary>

**答 / Answer:**
据 DeepSeek-R1 论文（arXiv:2501.12948），需先区分两个模型：

**DeepSeek-R1-Zero**：直接在 DeepSeek-V3-Base 上做纯 RL（GRPO），**完全跳过 SFT 阶段**。论文原文：”we bypass the conventional supervised fine-tuning (SFT) phase before RL training.” R1-Zero 展示了推理能力可从纯 RL 中涌现，但存在可读性差、语言混用等问题。

**DeepSeek-R1**：四阶段 pipeline（论文 Section 3）：
1. **Cold-start SFT**：收集数千条具有人类对话风格思维链的冷启动数据，对 DeepSeek-V3-Base 做 SFT，得到 Dev1。注意：这是”冷启动”而非标准大规模 SFT，数据量很小（thousands）。
2. **推理导向 RL（第一阶段 RL）**：在 Dev1 基础上用 GRPO 做推理任务强化学习（rule-based 奖励：准确性 + 格式），得到 Dev2。
3. **Rejection-sampling SFT**：从 Dev2 采样，融合推理 + 非推理数据做 SFT，得到 Dev3。此阶段同时提升写作等通用能力。
4. **全场景 RL（第二阶段 RL）**：在 Dev3 基础上做综合 RL，奖励信号融合 rule-based（推理）+ RM（通用对话、安全），最终得到 DeepSeek-R1。

**Cold-start 的作用**：解决 R1-Zero 的可读性差和语言混用问题，为后续 RL 提供更规整的行为基础，使 RL 探索更高效。

**追问 / Follow-up:**
Cold-start SFT使用的数据质量要求非常高。如果这部分数据存在错误或偏差，会对后续强化学习阶段的探索产生什么连锁反应？

</details>

<details>
<summary>Q23. RLAIF 和 Constitutional AI 的 self-critique-revision 机制如何工作？</summary>

**答 / Answer:**
RLAIF (Reinforcement Learning from AI Feedback) 和 Constitutional AI 的核心思想是使用AI模型自身来生成偏好反馈或进行修正，以减少对人类标注的依赖。其self-critique-revision机制通常包含一个循环：1）**生成初始回复**：针对一个prompt，模型先生成一个初步回复。2）**自我批判**：模型（或一个独立的批判模型）根据一组预设的”宪法”原则（如”回答要客观”、”避免有害内容”）对初始回复进行审视和批判，指出可能违反原则的地方。3）**修订回复**：模型根据生成的批判，对初始回复进行修改，生成一个更符合宪法原则的新版本。4）**（可选）用于训练**：将（初始回复，修订后回复）作为一个（rejected, chosen）对，用于训练RM或直接进行DPO等优化。这个机制让模型在无需人类实时干预的情况下，进行自我改进和对齐。

**追问 / Follow-up:**
这种自我修正机制有可能导致模型陷入某种”对齐循环”吗？例如，为了让回复更”安全”，它可能通过多轮修订，使回复变得越来越保守甚至无用。

</details>

<details>
<summary>Q24. Iterative RLHF 和 online DPO 的异同？如何解决 distribution mismatch？</summary>

**答 / Answer:**
两者都是为了解决offline方法（如标准DPO）中**训练数据分布（旧策略生成的偏好对）与模型当前策略分布不匹配**的问题。**相同点**：都通过迭代的方式，使用当前策略模型生成新的数据（或回复），并用这些新数据来更新模型，从而让训练数据分布跟随策略变化。**不同点**：Iterative RLHF通常指交替进行”在线数据生成（用当前策略采样，并由RM评分）”和”用新数据更新策略模型（可能用PPO或DPO）”的过程。Online DPO则更特指在每个训练迭代中，使用当前策略生成一组回复，由RM或人类选出偏好对，然后用这个**新生成的、分布匹配的偏好数据**来直接进行DPO损失计算和模型更新，省去了显式的RL步骤。

**追问 / Follow-up:**
在Online DPO中，使用当前策略生成偏好对时，应该在生成的回复中使用什么采样温度（temperature）？为什么这个参数选择很重要？

</details>

<details>
<summary>Q25. Post-training 的 scaling 规律：数据量、模型规模怎么影响对齐效果？SFT 和 RL 的最优 compute 分配策略有何不同？</summary>

**答 / Answer:**
Post-training的scaling规律与预训练不同。对于**数据量**：在SFT阶段，存在收益递减，高质量数据比大量低质数据更重要，达到一定规模后性能提升放缓。对于**模型规模**：更大的基座模型通常具有更强的对齐潜力，能更好地理解复杂的指令和价值观，但达到相同对齐水平所需的高质量数据量可能不一定同比例增长。**SFT vs RL的最优分配策略**：SFT的收益更”数据效率”，通常在项目初期投入较多计算资源快速建立指令遵循能力是划算的。RL（如RLHF）则更”计算昂贵”，其收益体现在精细的行为调整和价值对齐上，需要更多的在线采样和迭代。一个常见的策略是：用大部分计算预算训练一个足够好的基座模型和SFT模型，然后将剩余的、相对较少的计算预算用于几轮关键的RL迭代进行精调，因为RL的边际收益可能迅速下降。

**追问 / Follow-up:**
如果我们将模型规模和数据量都视为资源，在post-training阶段，你认为是投资于将一个70B的模型对齐，还是投资于将一个7B的模型配合更大量、更高质量的数据进行对齐，哪种策略更可能获得一个在实际应用中表现优异的助手模型？请阐述理由。

</details>

## 更多 L3 深挖 / Extended L3

<details>
<summary>Q26: DPO 的隐式 Reward 学到了什么？与显式 RM 相比有何本质局限？</summary>

DPO loss 的梯度等价于在优化一个隐式 reward $\hat{r}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$。这个隐式 reward 本质上是对 reference policy 下 token-level log-probability ratio 的累加，缺乏对生成语义的显式建模。与训练独立 RM 相比，DPO reward 被绑定在 policy 的参数空间中，导致三个核心局限：(1) **分布耦合**——reward 无法独立于 policy 评估 OOD response，探索能力受限；(2) **表征瓶颈**——policy 需同时承担"价值评估"与"策略生成"两个角色，参数可能存在冲突；(3) **时序不一致**——训练过程中 policy 变化导致隐式 reward 漂移，而显式 RM 的 reward 分布相对稳定。这也解释了为何 online DPO（用当前 policy 重新采样）通常优于 offline DPO。

> **追问：** 既然 DPO 存在 off-policy 问题，那 Rejection Sampling Fine-Tuning（RFT）作为更简单的方案，它在什么条件下会比 DPO 更有效？什么条件下会失效？

</details>

---

<details>
<summary>Q27: GRPO 的 Group Normalization 引入了什么统计偏差？如何缓解？</summary>

GRPO 对同一 prompt 的多个 response 做 group-level z-score normalization（减均值除标准差），这隐含了"prompt 内比较足够"的假设。统计上，当 group size $G$ 较小时（如 $G<8$），估计的均值和方差方差大，导致 advantage 估计有高噪声；更关键的是，**group normalization 使优势完全相对于同组样本定义**，这意味着：(1) 如果 group 内所有 response 质量都很低，"矮子里拔将军"仍会产生正 advantage，导致在低质量区域强化策略；(2) 相反，如果组内都是高质量 response，优秀回答也被压制。这种 **relative ranking bias** 使得 GRPO 在 reward 分布偏斜（如大部分 response 得分接近）时可能系统性偏离绝对质量信号。缓解方案包括引入 baseline anchor（如 EMA reference reward）或混合 absolute-relative advantage。

> **追问：** 在 GRPO 的 KL 约束下，如果 group size 趋于无穷大，GRPO 的优化目标在数学上会收敛到什么形式？它和标准 PPO 有什么关系？

</details>

---

<details>
<summary>Q28: Reward Model 的 Overparameterization 如何影响 RLHF？RM 应该与 Policy 同等规模还是更大/更小？</summary>

RM 的 overparameterization（参数量远超训练数据需求）会引发两个问题：(1) **spurious correlation**——RM 可能学到与偏好无关的表面特征（如特定用词风格、长度）而获得高准确率，但在 policy 更新后这些捷径失效；(2) **calibration 退化**——过参数化 RM 的 scalar output 往往过度自信（集中在少数极端值），导致 PPO 中 advantage 估计方差爆炸或 policy 被少数样本主导。实践中，RM 规模的选择涉及 trade-off：更大的 RM 有更强的语义理解能力，但更容易过拟合且推理开销大；更小的 RM 泛化性可能更好但表达能力受限。一种观点是 RM 应略大于或等于 policy 规模以确保足够的 reward 信号分辨率，同时通过 **reward ensemble**（多 RM 平均/投票）缓解过拟合。

> **追问：** Reward ensemble 的多个 RM 如果来自同一个 SFT 初始化、只是数据 shuffle 不同，这种 ensemble 在什么情况下仍然会系统性失败？如何设计真正多样化的 RM 集合？

</details>

---

<details>
<summary>Q29: 多轮对话场景下，RLHF 的 Credit Assignment 问题如何解决？现有的 sequence-level reward 足够吗？</summary>

多轮对话中，用户最终满意度是整个对话历史的函数，但标准 RLHF 只在最终 turn 给一个 scalar reward，这产生了严重的 **temporal credit assignment** 问题：模型无法知道是哪一轮的回答导致了正面或负面评价。直觉上的解决方案包括：(1) **turn-level reward modeling**——为每轮对话训练独立的 reward model，但面临对话状态的部分可观测性和标注成本问题；(2) **Monte Carlo rollout**——从某一轮开始重新采样后续对话估计 value，但组合爆炸严重；(3) **shaped reward via dialogue act**——利用对话行为（如澄清、确认）作为中间 reward 信号。实验上，纯 sequence-level reward 在短对话（2-3轮）中尚可，但在长对话中 policy 容易陷入 **early-turn over-optimization**（过度优化首轮回答以获取 initial reward 信号，而忽略后续交互质量）。

> **追问：** 如果要在 token-level 实现 reward attribution（而非 turn-level），理论上可以通过什么方法将 sequence-level reward 分解到每个 token？这种方法的理论保证和实际困难分别是什么？

</details>

---

<details>
<summary>Q30: KL 约束的理论最优解对 β 敏感吗？实际中 β 偏离最优值时，PPO 和 DPO 的失败模式有何不同？</summary>

从 KL-regularized RL 的角度，$\beta$ 控制 exploration-exploitation 的 Pareto 前沿位置。理论上，最优 $\beta^*$ 依赖于 reward function 的 scale 和 reference policy 的 entropy，无法提前确定。当 $\beta$ 过大（过度正则化），PPO 和 DPO 的表现趋近于 reference policy，alignment 效果微弱；但当 $\beta$ 过小（正则化不足），两者的失败模式分化：**PPO** 会经历 reward hacking 的正反馈循环——policy 一旦找到 reward 漏洞就被持续强化，RM 被 out-of-distribution 评估，reward 崩溃；**DPO** 则表现为 **preference reversal 的不稳定**——off-policy 采样的隐式 reward 在训练中漂移，chosen 和 rejected 的 margin 减小甚至翻转，loss 出现震荡。实际调参中，PPO 的 $\beta$（KL penalty coefficient）通常需要与 learning rate 协同调整，而 DPO 的 $\beta$ 更像 temperature，较小的 $\beta$ 允许更大的 chosen-rejected margin 但也更易 overfit。

> **追问：** 有没有理论上的方法可以自适应地调整 $\beta$（而非手动调参）？KL divergence 本身作为 signal 用于自适应 β 有什么问题？

</details>

---

<details>
<summary>Q31: Process Reward Model (PRM) 在数学推理等长链任务上有优势，但如何处理"步骤正确但推理路径非最优"的标注歧义？</summary>

PRM 面临的核心挑战是 **multi-modal solution distribution**：对于同一问题，存在多条合理的推理路径（如代数法 vs 几何法），每条路径内部步骤逻辑一致但跨路径不可比较。标注时，如果让 annotator 判断"此步骤是否正确"，他们可能因为不熟悉该推理风格而给出 false negative。更微妙的是，即使步骤在当前路径下正确，如果该路径整体 suboptimal，步骤级 reward 也应被调整——但这需要全局视角，与 PRM 的局部评估本质矛盾。解决方向包括：(1) **path-conditioned PRM**——在给定前序步骤的条件下评估当前步骤，而非绝对评估；(2) **Monte Carlo estimation**——从当前步骤 rollout 到最终答案，用成功率作为步骤级 reward，但计算开销大；(3) **agreement-based filtering**——只对多条路径共有的"关键步骤"标注，避开路径特异性步骤。

> **追问：** 如果用 Monte Carlo rollout 来估计 PRM 的步骤级 reward，rollout 的 policy 应该用当前训练的 policy 还是一个固定的 exploration policy？这个选择对 reward 估计的偏差和方差分别有什么影响？

</details>

---

<details>
<summary>Q32: Constitutional AI (CAI) 声称可以用 AI 反馈替代人类反馈，但 RLAIF 的理论上限在哪里？AI 反馈与人类反馈的 gap 能被消除吗？</summary>

RLAIF 的理论上限受 **AI 评估器的能力边界** 约束。核心问题在于：如果 AI 评估器自身存在系统性偏好（如 verbosity bias、sycophancy），那么基于其反馈训练的 policy 会继承甚至放大这些偏好，形成 **evaluator-policy co-adaptation** 的退化循环。更深层的限制是 **value alignment 的不可验证性**——人类偏好的某些维度（如诚实、无害）本质上需要人类判断，AI 无法 self-validate。CAI 的"宪法原则"试图通过显式规则绕过，但规则无法覆盖所有 corner case，且规则之间的冲突需要人类仲裁。实验上，RLAIF 在某些客观维度（如格式正确性）上可以接近人类反馈，但在需要深度价值判断的维度（如 nuanced harm assessment）上仍有明显 gap。理论上，只有当 AI 评估器是人类偏好的 **无偏且一致的 estimator** 时，RLAIF 才能达到 RLHF 的效果，但这一假设目前无法保证。

> **追问：** 如果 AI 评估器存在已知偏差（如 verbosity bias），能否通过 debiasing 技术（如 calibration、adversarial training）在 RLAIF 训练前纠正？这种纠正的理论保证是什么？

</details>

---

<details>
<summary>Q33: Multi-turn RLHF 中，如何建模用户策略的动态性？如果假设用户策略固定，会导致什么系统性错误？</summary>

标准 multi-turn RLHF 隐含 **stationary user assumption**——假设用户在整个对话中遵循固定的响应策略。但实际上，用户会根据模型的回答调整自己的提问策略（如模型回避问题时用户会追问、模型过于冗长时用户会要求简短）。这将 RLHF 从单智能体 MDP 变成 **two-player Markov Game**。在用户策略非平稳下，固定用户假设会导致：(1) **overfitting to simulated user**——policy 学到的是对特定模拟用户模式的最优响应，而非对真实动态用户的鲁棒策略；(2) **exploitation of user patience**——如果模拟用户不会因冗长回答而终止对话，policy 会学到过度啰嗦的风格。更根本的困难是，真实用户策略本身是分布，甚至可能因模型行为而改变（user-model co-evolution），这在理论上接近 **non-stationary multi-agent RL**，目前没有成熟的收敛保证。

> **追问：** 如果要显式建模用户的动态策略，是否可以用一个 user simulator 与 policy 联合训练？这种 self-play 框架的已知失败模式是什么？

</details>

## §A 核心论文时间线 / Key Papers Timeline

- **2022-03 · InstructGPT** — Ouyang et al., NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) — 确立标准三阶段 RLHF 流程（SFT → Bradley-Terry 奖励模型 → 带 KL 惩罚的 PPO），将 GPT-3 对齐为可遵循指令的助手。

- **2022-09 · WiSE-FT** — Wortsman et al., CVPR 2022. [arXiv:2109.01903](https://arxiv.org/abs/2109.01903) — 通过在权重空间中对微调模型与基座模型做线性插值来缓解对齐税，保留预训练鲁棒性的同时维持任务性能。

- **2022-12 · Constitutional AI（CAI / RLAIF）** — Bai et al., arXiv preprint. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) — 用 LLM 依据显式"宪法原则"进行自我批评-修订循环来替代人类标注员，实现无需逐样本人工标注的可扩展 RLAIF。

- **2023-05 · DPO** — Rafailov et al., NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — 对 KL 约束的 RLHF 目标进行闭式重参数化，消除显式奖励模型，将偏好对齐归约为对 (chosen, rejected) 对的单一分类损失。

- **2023-10 · IPO** — Azar et al., AISTATS 2024. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036) — 指出 DPO 的 logit 映射在近确定性偏好下导致 KL 正则化失效；提出 ΨPO（Ψ = 恒等映射），用有界平方损失回归目标恢复有效正则化。

- **2024-02 · DeepSeekMath / GRPO** — Shao et al., arXiv preprint. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — 提出群组相对策略优化（GRPO），通过对同一 prompt 的组内采样奖励做归一化来替代 PPO Critic，将所需模型数量减半，支持基于可验证奖励的稳定强化学习。

- **2024-02 · KTO** — Ethayarajh et al., ICML 2024. [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) — 用点式二元好/坏标签替代成对偏好数据，将对齐建模为前景理论效用最大化，采用非对称 sigmoid 损失与 KL 基准参考点。

- **2024-02 · DPOP（Smaug）** — Pal et al., arXiv preprint. [arXiv:2402.13228](https://arxiv.org/abs/2402.13228) — 证明当 chosen 与 rejected 高度相似时 DPO 会降低 chosen 的对数概率；引入 max(0,·) 惩罚项将 chosen log-prob 锚定在参考模型以上。

- **2024-03 · ORPO** — Hong et al., arXiv preprint. [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) — 将 SFT 与偏好对齐合并为单阶段训练，在交叉熵损失上附加无参考模型的 odds ratio 对比项，无需维护冻结参考模型。

- **2024-05 · SimPO** — Meng et al., NeurIPS 2024. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734) — 用长度归一化的平均对数概率作为隐式奖励，使奖励与生成时 likelihood 直接对齐，并引入显式目标边际 γ，去掉参考模型并消除长度偏差。

- **2024-10 · Likelihood Displacement** — Razin et al., ICLR 2025. [arXiv:2410.08847](https://arxiv.org/abs/2410.08847) — 从理论上证明当 chosen 与 rejected 的隐层表示相似度高（CHES 分数高）时，DPO 梯度会将概率质量从 chosen 迁移至语义相反的输出，造成"非预期错对齐"；提出基于 CHES 的数据过滤作为缓解手段。

- **2025-01 · DeepSeek-R1** — Guo et al., Nature 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — 证明链式思维推理能力可从纯 GRPO 强化学习中涌现（R1-Zero），并通过冷启动 SFT + 两轮 RL + 拒绝采样 SFT 的四阶段流程，训练出与 OpenAI o1 匹敌的推理模型。
