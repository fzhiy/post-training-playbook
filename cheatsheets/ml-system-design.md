# ML / LLM 系统设计速查表
# ML / LLM System Design — Cheat Sheet

> 面向 LLM Research Intern 岗位准备 | For LLM research intern preparation
> 公开发布版 · 无特定论文内部结果 | Public edition · No proprietary results included

---

## 目录 / Table of Contents

1. [概念与公式推导 / Concepts & Key Formulas](#一概念与公式推导--concepts--key-formulas)
2. [PyTorch 代码片段 / From-Scratch Snippets](#二pytorch-代码片段--from-scratch-snippets)
3. [面试题 / Interview Questions](#三面试题--interview-questions)

---

## 一、概念与公式推导 / Concepts & Key Formulas

### 1.1 因果语言模型 / Causal Language Modeling (CLM)

**核心思想 / Core Idea：** 自回归地预测下一个 token，训练时使用 causal mask 阻止未来信息泄漏。

**损失函数 / Loss Function：**

$$
\mathcal{L}_{\text{CLM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

**推导要点 / Derivation：**
- 由链式法则（chain rule）：$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$
- 取对数并取负号即得交叉熵损失（cross-entropy loss）
- 实现时，logits 形状为 `(batch, seq_len, vocab_size)`，target 为左移一位的 token ids

---

### 1.2 Softmax 与 Attention / Softmax & Attention

**Scaled Dot-Product Attention：**

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**为什么除以 $\sqrt{d_k}$？ / Why scale by $\sqrt{d_k}$？**
- 假设 $Q, K$ 各元素独立同分布，均值为 0、方差为 1
- 则 $QK^\top$ 中每个元素的方差为 $d_k$
- 若 $d_k$ 很大，softmax 输入数值大 → 梯度消失（softmax 饱和）
- 除以 $\sqrt{d_k}$ 将方差归一化到 1，保持梯度稳定

**Multi-Head Attention (MHA)：**

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attn}(XW_i^Q,\; XW_i^K,\; XW_i^V)
$$

其中 $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$，$d_k = d_v = d_{\text{model}} / h$。

**GQA / MQA 变体 / Variants：**
- **Multi-Query Attention (MQA)：** 所有 head 共享同一组 $K, V$，仅 $Q$ 不同 → KV cache 显著缩小
- **Grouped-Query Attention (GQA)：** 将 $h$ 个 query head 分成 $g$ 组，每组共享 $K, V$，是 MHA 与 MQA 的折中

---

### 1.3 Position Encoding / 位置编码

**Rotary Position Embedding (RoPE)：**

$$
\tilde{q}_m = q_m e^{im\theta}, \quad \tilde{k}_n = k_n e^{in\theta}
$$

其中 $\theta_j = 10000^{-2j/d}$。

$$
\langle \tilde{q}_m, \tilde{k}_n \rangle = \text{Re}[q_m^* k_n \, e^{i(m-n)\theta}]
$$

**性质 / Properties：**
- 内积仅依赖相对位置 $(m-n)$ → 自然编码相对位置
- 无需学习参数（deterministic）
- 外推性优于 learned positional embedding（配合 NTK-aware scaling 可扩展长度）

**RoPE 的实际实现 / Practical Implementation：**

$$
\text{RoPE}(x) = x \odot \cos(m\theta) + \text{rotate\_half}(x) \odot \sin(m\theta)
$$

对 $x \in \mathbb{R}^{d_k}$，每对相邻维度 $(x_{2i}, x_{2i+1})$ 做 2D 旋转。

---

### 1.4 LoRA — 低秩适配 / Low-Rank Adaptation

**动机 / Motivation：** 全参数微调大型模型显存开销大（需存储参数、梯度、优化器状态各一份）。LoRA 冻结预训练权重，仅训练低秩增量。

**核心公式 / Key Formula：**

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

其中 $W_0 \in \mathbb{R}^{d \times k}$ 冻结，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**缩放因子 / Scaling：**

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

$\alpha$ 为缩放超参数，典型设为 $\alpha = 2r$ 或 $\alpha = r$。

**参数量分析 / Parameter Count：**
- 原始参数：$d \times k$
- LoRA 参数：$d \times r + r \times k = r(d + k)$
- 例：$d = 4096, k = 4096, r = 16$ → LoRA 参数 = $16 \times 8192 = 131072$，占原始的 $131072 / (4096^2) \approx 0.78\%$

**初始化 / Initialization：**
- $A$：使用 Kaiming 均匀分布初始化（或高斯）
- $B$：零初始化 → 训练开始时 $\Delta W = BA = 0$，不改变预训练输出

**合并推理 / Merge for Inference：**

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA
$$

合并后推理无额外开销。

---

### 1.5 RLHF 与 DPO / Reinforcement Learning from Human Feedback

**奖励模型训练 / Reward Model Training (Bradley-Terry)：**

$$
\mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
$$

其中 $y_w \succ y_l$ 为人工标注的偏好对（preferred vs rejected）。

**PPO 目标 / PPO Objective：**

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)} \!\Big[ r_\phi(x, y) - \beta \, \text{KL}\big(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\big) \Big]
$$

**KL 散度的作用 / Role of KL：**
- $\beta$ 过小 → reward hacking（策略钻 reward model 的漏洞）
- $\beta$ 过大 → 策略几乎不动（退化为 SFT 模型）

**DPO（Direct Preference Optimization）/ 直接偏好优化：**

绕过显式 reward model，从 Bradley-Terry 模型出发推导：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma \!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

**DPO 优势 / Advantages：**
- 无需 RL 采样循环（不需要在训练时生成 response）
- 无需显式 reward model
- 训练更稳定，超参更少

**DPO 局限 / Limitations：**
- 隐式 reward 可能不如显式 RM 的泛化能力
- 对偏好数据质量更敏感（没有 RM 的"缓冲"）
- 不易做 online RL（需要 on-policy 采样来改进）


---

### 1.5b RLHF 分布式架构 / Distributed RLHF Architecture

#### Naive Co-located PPO 的 GPU 利用率问题

最简单的实现方式是把 actor、reference model、critic、reward model 全部跑在同一批 GPU 上（co-located）。瓶颈在 **rollout 阶段**：

```
┌─────────────────────────────────────────────────────┐
│  Co-located PPO（简化时间线）                         │
│                                                     │
│  ──[rollout: actor 自回归生成]──►  ──[train: PPO 更新]──► │
│         GPU 忙于推理              trainer 忙，actor 闲   │
└─────────────────────────────────────────────────────┘
```

- **rollout 时**：actor 逐 token 自回归，计算不密集，GPU MFU（Model FLOP Utilization）往往偏低；trainer（ZeRO/FSDP）处于空闲。
- **train 时**：前向 + 反向计算密集，actor 又没有推理任务；rollout worker 闲置。
- 结果：两个阶段相互空等，整体 GPU 利用率是两段利用率的加权平均，远低于纯训练或纯推理时的峰值。

⚠️ 这不是精确测量值，具体 MFU 因模型规模、batch size、硬件而异——以上描述的是**定性问题**，实际数字请参考对应框架（OpenRLHF、veRL 等）的技术报告。

---

#### Disaggregated Rollout + Train 拓扑

为解决上述问题，**分离（disaggregated）** rollout worker 和 train worker：

```
┌──────────────────────────────────────────────────────────────────┐
│  Disaggregated PPO 拓扑                                          │
│                                                                  │
│  ┌─────────────────────────┐      ┌──────────────────────────┐  │
│  │   Rollout Workers        │      │   Train Workers           │  │
│  │   (vLLM / SGLang 引擎)   │      │   (ZeRO-3 / FSDP)        │  │
│  │                         │      │                          │  │
│  │  actor (inference mode) │─────►│  actor (grad update)     │  │
│  │  ref model (frozen)     │      │  critic (grad update)    │  │
│  │  reward model (frozen)  │      │                          │  │
│  └─────────────────────────┘      └──────────────────────────┘  │
│           │  生成 responses + 奖励              ▲                 │
│           │  (rollout buffer)                  │ 权重同步         │
│           └────────────────────────────────────┘                 │
│              每 N 步（或每轮 rollout）同步一次 actor 权重           │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计点：**

- **Rollout workers** 加载 actor 的推理权重（FP16/BF16），用 vLLM 或 SGLang 做 continuous batching 自回归生成，效率高。
- **Train workers** 用 ZeRO-3 或 FSDP 持有完整的可训练参数（含优化器状态），执行 PPO/GRPO 梯度更新。
- **权重同步（weight sync）**：train workers 更新完一批后，将最新 actor 权重广播给 rollout workers。同步频率通常是每个 PPO iteration 同步一次（即每 rollout + train 完整循环）；也有实现支持更精细的分步同步。
- **Ref model / RM**：一般以推理模式常驻在 rollout 侧（冻结权重，无需梯度），节省 train 侧显存。

---

#### 4 模型显存拆解 + LoRA-in-RL 如何省显存

标准 RLHF 涉及四个模型：

| 模型 | 参数 | 梯度 | 优化器状态（AdamW） | 典型位置 |
|------|------|------|-------------------|---------|
| **Actor** | ✅（训练） | ✅ | ✅（$m, v$，FP32 约 8 bytes/参数） | Train workers |
| **Ref model** | ✅（冻结） | ✗ | ✗ | Rollout workers 或独立节点 |
| **Critic** | ✅（训练） | ✅ | ✅ | Train workers（可与 actor 共 GPU） |
| **Reward Model** | ✅（冻结） | ✗ | ✗ | Rollout workers |

**单模型（以 7B 参数为例）显存估算（仅量级，非精确值）：**

$$
M_{\text{param}} \approx 7 \times 10^9 \times 2\,\text{bytes/param (BF16)} \approx 14\,\text{GB}
$$

$$
M_{\text{opt}} \approx 7 \times 10^9 \times 8\,\text{bytes} \approx 56\,\text{GB}
$$

其中 $M_{\text{param}}$ 为参数显存（BF16），$M_{\text{opt}}$ 为 AdamW 优化器状态显存（FP32 $m$ + $v$ 各一份，共 8 bytes/参数）。4 个模型 naive co-located，显存需求量级在数百 GB——7B 尚可塞进单机 8×80G，但 naive co-located 的 GPU 利用率很低（见下文）；更大模型（如 70B）显存则远超单机。

**LoRA-in-RL 的节省：**

- 仅训练 actor 和 critic 的 LoRA 旁路（$r \ll d$），冻结预训练权重。
- 梯度和优化器状态只与 LoRA 参数量成比例，参数量减少约 $99\%$ 时（例如 rank=16），优化器状态从 56 GB 量级降到约 1 GB 量级（数量级估算）。
- 代价：LoRA 本身的表达能力受秩限制，RL 阶段的策略更新幅度可能受约束。实践中 PPO + LoRA 已在多个公开工作中验证可行（具体效果视任务和 rank 而定，需参考原始论文数据）。

---

#### Async vs Sync Rollout 的 Staleness

| 模式 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **Sync rollout** | rollout 完成后才开始 train，train 完成后才开始下一轮 rollout | 无 staleness，on-policy | GPU 利用率低（两阶段轮流空闲） |
| **Async rollout** | rollout worker 持续生成，train worker 持续更新；权重同步有延迟 | GPU 利用率高，吞吐高 | **Staleness**：rollout 用的是 $k$ 步之前的旧权重，数据是 off-policy 的 |

**Staleness 的影响：**

- 生成数据时用的策略 $\pi_{\theta_{\text{old}}}$ 与更新目标 $\pi_\theta$ 的分布偏差增大。
- PPO 的 clip objective 对小幅度 off-policy 有一定容忍度（通过 $r_t(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$ 的 importance ratio 矫正），但 staleness 过大时 importance ratio 方差急剧增大。
- 实践中，许多框架选择 **近似同步**（每 $k$ 步同步一次权重），在吞吐和 staleness 之间折中。

---

#### 参考实现：OpenRLHF vs veRL

| 维度 | **OpenRLHF** | **veRL** |
|------|-------------|---------|
| 定位 | 研究友好、简洁、快速上手 | 面向大规模生产，性能优化更激进 |
| Rollout 引擎 | vLLM（深度集成） | vLLM / SGLang 均支持 |
| 训练并行 | DeepSpeed ZeRO-3 | FSDP + Megatron-LM TP/PP 均支持 |
| 4 模型调度 | 支持 co-located 和 disaggregated 模式 | Hybrid Engine（rollout/train 共享 GPU，动态切换） |
| LoRA-in-RL | ✅ 支持 | ✅ 支持 |
| 代码量 | 较少，架构清晰，适合二次开发 | 较多，但生产特性完备（checkpoint、fault tolerance） |
| 典型引用场景 | 学术实验、快速验证算法思路 | 大规模 post-training 流水线 |

✅ **两者都是公开实现，可作为"系统设计题"的参考答案骨架。** 具体性能数字请参阅各自的官方技术报告和 GitHub，不同版本、硬件间数字差异较大，面试中说"量级"而非精确值更安全。

---

#### 吞吐量估算：Rollout vs Train GPU-hours 比例

⚠️ 以下为**定性量级分析**，具体数字因模型规模、response 长度、硬件配置高度敏感，面试中应明确"举例估算"而非引用精确 benchmark。

**思路框架（以 7B actor 为例）：**

- **Rollout cost**：自回归生成是 memory-bound，每生成 token 仍需过所有层的前向（KV cache 后每步只处理 1 个新 token、而非整个序列，但层数不变），吞吐受 HBM 带宽限制。若平均 response 长度 $L_r$，则 rollout 的计算量约正比于 $B \times L_r \times \text{param\_size}$（内存访问量）。
- **Train cost**：前向 + 反向约为 $6 \times B \times s \times P$ FLOPs（$s$ 为序列长度，$P$ 为参数量；其中前向 $\approx 2P$、反向 $\approx 4P$，合计 $6P$ 每 token）。
- **典型结论**（量级）：在 response 较长（数百 token）时，rollout GPU-hours 往往与 train GPU-hours **同量级，甚至更高**——这是 disaggregated 架构的核心动机之一。若 rollout 远快于 train，那 disaggregation 的增益有限；若 rollout 是瓶颈，多分配 rollout worker 是自然的扩展方式。

---

### 1.6 分布式训练 — 并行策略 / Distributed Training Parallelism

#### Data Parallelism (DP) / 数据并行

每张卡持有完整模型副本，梯度通过 All-Reduce 同步。

**通信量 / Communication：** 每步 All-Reduce 参数梯度 = $2 \times |\theta|$（ring all-reduce）。

#### ZeRO（Zero Redundancy Optimizer）/ 零冗余优化

| 阶段 Stage | 分片内容 Sharded | 显存占比 Memory per GPU |
|---|---|---|
| ZeRO-1 | Optimizer states（Adam: $m, v$） | ~参数量的 4×（与 DP 相同参数显存）|
| ZeRO-2 | + Gradients | ~参数量的 2× |
| ZeRO-3 | + Parameters | ~参数量的 $1/P$（P = GPU 数）|

**代价 / Overhead：** ZeRO-3 需要前向时 All-Gather 参数，通信量增加。

**$16\Phi$ 显存分解（混合精度 Adam，$\Phi$ = 参数量）/ The $16\Phi$ memory breakdown:**

| 组成 Component | 精度 | 字节/参数 | 显存 |
|---|---|---|---|
| 模型参数 (fp16) | fp16 | 2 | $2\Phi$ |
| 梯度 (fp16) | fp16 | 2 | $2\Phi$ |
| Adam 优化器状态 | fp32 | 12 | $12\Phi$ |

其中优化器状态 $12\Phi$ = fp32 主权重副本 $4\Phi$ + 一阶动量 $m$（$4\Phi$）+ 二阶动量 $v$（$4\Phi$），合计 **$16\Phi$**（如 7.5B 模型 → 120 GB，单卡放不下）。各 ZeRO 阶段在 $P$ 卡上的单卡显存：

| 阶段 | 分片内容 | 单卡显存 | $P\to\infty$ |
|---|---|---|---|
| baseline (DP) | 无 | $16\Phi$ | $16\Phi$ |
| ZeRO-1 | 优化器状态 | $2\Phi + 2\Phi + \tfrac{12\Phi}{P}$ | $4\Phi$ |
| ZeRO-2 | + 梯度 | $2\Phi + \tfrac{14\Phi}{P}$ | $2\Phi$ |
| ZeRO-3 | + 参数 | $\tfrac{16\Phi}{P}$ | $\to 0$ |

> ZeRO-3 三者全分片，通信量约为纯 DP 的 1.5×（前向 all-gather 参数、反向 all-gather 参数 + reduce-scatter 梯度）——用通信换显存。来源：Rajbhandari et al. 2020, arXiv:1910.02054。

#### Tensor Parallelism (TP) / 张量并行

将每一层的权重矩阵按列或行切分到多张卡。

- **Column-parallel：** $Y = XA$，$A$ 按列切分为 $[A_1, A_2]$，各卡计算 $XA_i$，无需通信即得部分结果。后续若需行切分层，可融合一次 AllReduce。
- **Row-parallel：** $Y = A_1 X_1 + A_2 X_2$，各卡独立计算后做一次 AllReduce。

**Megatron-LM 设计：** Column-parallel Linear → GeLU（本地）→ Row-parallel Linear → AllReduce。整个 MLP 块只需 **一次** AllReduce（+ 反向一次）。

#### Pipeline Parallelism (PP) / 流水线并行

将模型按层切段分配到不同机器。

- **GPipe 策略：** 将 mini-batch 拆成 $M$ 个 micro-batch，顺序前向 + 逆序反向。
- **1F1B 调度：** 交替执行 1 次前向和 1 次反向，减少 pipeline bubble 和峰值显存。
- **Bubble 率：** $\text{Bubble} \approx (P-1) / (M + P - 1)$，$P$ = pipeline stages，$M$ = micro-batches。

#### Sequence Parallelism (SP) / 序列并行

对 LayerNorm、Dropout 等不含参数但占激活显存的操作，沿序列维度切分。

- Ring Attention：将长序列切为 $P$ 段分到 $P$ 张卡，通过环形通信传递 KV，激活显存从 $O(N)$ 降为 $O(N/P)$。

**实践选型 / Practical Guidance：**
- 单机 8 卡：DP/ZeRO-2 + TP（NVLink 快）
- 多机：DP/ZeRO-3 + PP（跨节点带宽低）+ TP（节点内）
- 超长上下文：加入 SP（Ring Attention）

---

### 1.7 KV Cache 显存分析 / KV Cache Memory Analysis

每层每个 token 需缓存 $K$ 和 $V$：

$$
\text{KV cache (bytes)} = 2 \times L \times n_{\text{heads}} \times d_{\text{head}} \times s \times \text{bytes\_per\_param}
$$

- $L$ = 层数，$n_{\text{heads}}$ = KV head 数（GQA 时少于 Q head 数），$d_{\text{head}}$ = 每个 head 维度，$s$ = 序列长度
- FP16 下 bytes_per_param = 2

**PagedAttention（vLLM）：** 将 KV cache 分为固定大小的 page（如 16 tokens/page），按需分配，消除显存碎片，支持更多并发。

---

### 1.8 量化基础 / Quantization Fundamentals

**对称量化 / Symmetric Quantization：**

$$
x_q = \text{round}\!\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1} - 1}
$$

**非对称量化 / Asymmetric Quantization：**

$$
x_q = \text{round}\!\left(\frac{x - z}{s}\right), \quad s = \frac{x_{\max} - x_{\min}}{2^b - 1}, \quad z = x_{\min}
$$

**GPTQ — 基于 OBS 的逐层后训练量化（Frantar et al., ICLR 2023, arXiv:2210.17323）：**
- 逐层最小化重建误差 $\|WX - \hat{W}X\|_2^2$；沿用 OBS/OBQ，用 Hessian $H = 2XX^\top$ 的逆来补偿。
- 量化第 $q$ 个权重后，把误差按 $\delta = -\dfrac{w_q - \mathrm{quant}(w_q)}{[H^{-1}]_{qq}}\,(H^{-1})_{:,q}$ 分摊到**尚未量化**的权重上，抵消量化造成的输出偏移。
- GPTQ 的工程化：固定列顺序（免去 OBQ 的逐权重贪心选择）+ Cholesky 分解保数值稳定 + 分块更新，可在数小时内把 175B 量化到 3–4 bit。

**AWQ — 激活感知权重量化（Lin et al. 2023, arXiv:2306.00978）：**
- 观察：权重并非同等重要，约 0.1–1% 的"显著权重"由**激活幅度**（而非权重幅度）识别。
- 做法：对显著通道做 per-channel 缩放——权重乘 $s>1$、对应激活除以 $s$（$\hat{W}=W\,\mathrm{diag}(s),\ \hat{X}=X\,\mathrm{diag}(s)^{-1}$，乘积 $\hat{X}\hat{W}^\top=XW^\top$ 不变），使显著权重的相对量化误差变小；逐层网格搜索最优 $s$。纯前向、无需反传。

**SmoothQuant — 把量化难度从激活迁移到权重（Xiao et al., ICML 2023, arXiv:2211.10438）：**
- 问题：激活存在 per-channel 离群值（outlier）极难量化，而权重平滑好量化。
- 做法：per-channel 平滑 $\hat{X}=X\,\mathrm{diag}(s)^{-1},\ \hat{W}=\mathrm{diag}(s)\,W$，缩放因子 $s_j=\dfrac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$（$\alpha\approx0.5$），把激活的动态范围"匀"一部分给权重，实现 W8A8。

**FP8（Hopper/H100）：** E4M3（4 指数 3 尾数，范围 ±448）用于前向的权重/激活；E5M2（5 指数 2 尾数，动态范围更大 ±57344）用于梯度。相比 INT8 免去 scale 校准、对离群值更鲁棒。

**KV-cache 量化：** 长上下文下 KV cache 主导显存。K 沿 channel 维有离群值 → 宜 per-channel 量化；V 较平滑 → per-token 量化（如 KIVI, arXiv:2402.02750）。常用 int8/int4/fp8，可把 KV 显存降 2–4×；int8/fp8 多数任务精度损失可忽略，int4 则依任务而定（长上下文检索更敏感）。


---

### 1.9 Speculative Decoding / 投机解码

**核心思想：** 用小型 draft model 并行预测 $k$ 个 token，再用 target model 一次前向验证。

**接受-拒绝采样 / Accept-Reject：**
- 对位置 $t$，若 target model 概率 $p(x_t) \geq$ draft model 概率 $q(x_t)$ → 接受
- 若 $p(x_t) < q(x_t)$，以概率 $p(x_t)/q(x_t)$ 接受，否则从 $\max(0, p(x_t) - q(x_t))$ 重新采样
- 保证输出分布与直接用 target model 采样**完全一致**（无损）

**加速比 / Speedup：** 取决于 draft model 与 target model 的 token 接受率。典型场景下可获得 $1.5\times$–$2.5\times$ 加速。


---

### 1.10 模型设计通用框架 / 7-Step ML System Design Framework

| 步骤 | 英文 | 要点 |
|------|------|------|
| 1 | Clarify | 数据量、模型规模、QPS、延迟 SLA、显存预算、成功指标 |
| 2 | Data | 来源、清洗策略、标注方式（人工/弱监督/模型生成）、数据飞轮 |
| 3 | Model | 架构选择、参数量、Pre-train vs Fine-tune vs RAG、PEFT vs 全参 |
| 4 | Training Infra | 并行策略（DP/TP/PP/SP）、显存优化、batch size、LR schedule |
| 5 | Evaluation | 离线 benchmark + 人工评估 + Safety eval |
| 6 | Serving | 量化、dynamic batching、KV cache 管理、延迟 vs 吞吐 |
| 7 | Monitoring | 质量漂移（PPL、accuracy）、数据分布偏移、safety incidents |

---

## 二、PyTorch 代码片段 / From-Scratch Snippets

> 以下为教学用途的最小实现，突出核心逻辑，省略生产级的错误处理和优化。

### 2.1 Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    q: torch.Tensor,   # (batch, n_heads, seq_q, d_k)
    k: torch.Tensor,   # (batch, n_heads, seq_k, d_k)
    v: torch.Tensor,   # (batch, n_heads, seq_k, d_v)
    mask: torch.Tensor | None = None,  # (batch, 1, seq_q, seq_k) or broadcastable
) -> torch.Tensor:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v), attn_weights
```

### 2.2 Causal Self-Attention Layer

```python
import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)
```

### 2.3 LoRA Layer

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank delta."""

    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))  # B init to 0
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Return a new Linear with merged weights (for deployment)."""
        merged_weight = self.base.weight.data + (self.lora_b @ self.lora_a) * self.scaling
        new_linear = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        new_linear.weight.data.copy_(merged_weight)
        if self.base.bias is not None:
            new_linear.bias.data.copy_(self.base.bias.data)
        return new_linear
```

### 2.4 Grouped-Query Attention (GQA)

```python
import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads  # repeat factor
        self.d_k = d_model // n_q_heads

        self.wq = nn.Linear(d_model, n_q_heads * self.d_k, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match Q heads: (B, n_kv, T, d_k) -> (B, n_q, T, d_k)."""
        if n_rep == 1:
            return x
        B, N, T, D = x.shape
        return x[:, :, None, :, :].expand(B, N, n_rep, T, D).reshape(B, N * n_rep, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_q_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)
```

### 2.5 RoPE (Rotary Position Embedding)

```python
import torch

def precompute_rope_freqs(dim: int, max_len: int = 4096, base: float = 10000.0):
    """Precompute sin/cos tables for RoPE."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
    t = torch.arange(max_len).float()           # (max_len,)
    freqs = torch.outer(t, freqs)                # (max_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)    # each (max_len, dim/2)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor.
    x: (batch, n_heads, seq_len, d_k)
    cos, sin: (seq_len, d_k/2)
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    cos = cos.unsqueeze(0).unsqueeze(0)  # broadcast
    sin = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)
```

### 2.6 DPO Loss

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_logps_w: torch.Tensor,   # log pi_theta(y_w | x)
    policy_logps_l: torch.Tensor,   # log pi_theta(y_l | x)
    ref_logps_w: torch.Tensor,      # log pi_ref(y_w | x)
    ref_logps_l: torch.Tensor,      # log pi_ref(y_l | x)
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization loss."""
    log_ratio_w = policy_logps_w - ref_logps_w
    log_ratio_l = policy_logps_l - ref_logps_l
    logits = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(logits).mean()
```

### 2.7 KV Cache Wrapper (Minimal)

```python
import torch

class KVCache:
    """Minimal KV cache for autoregressive generation."""

    def __init__(self, max_len: int, n_heads: int, d_k: int, device: torch.device):
        self.max_len = max_len
        self.k = torch.zeros(1, n_heads, max_len, d_k, device=device)
        self.v = torch.zeros(1, n_heads, max_len, d_k, device=device)
        self.cur_len = 0

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor):
        """Append new KV from one decoding step."""
        seq_len = new_k.shape[2]
        self.k[:, :, self.cur_len:self.cur_len + seq_len] = new_k
        self.v[:, :, self.cur_len:self.cur_len + seq_len] = new_v
        self.cur_len += seq_len

    def get(self):
        """Return the current cached KV (trimmed to cur_len)."""
        return self.k[:, :, :self.cur_len], self.v[:, :, :self.cur_len]
```

### 2.8 Symmetric INT8 Quantize / Dequantize

```python
import torch

def symmetric_quantize_int8(weight: torch.Tensor):
    """Per-tensor symmetric INT8 quantization."""
    scale = weight.abs().max() / 127.0
    w_q = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return w_q, scale

def symmetric_dequantize_int8(w_q: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize INT8 back to float."""
    return w_q.float() * scale
```

---

### 2.9 Tensor-Parallel Linear (Column / Row)

```python
import torch
import torch.nn as nn

# Megatron 张量并行 Linear 的核心是一对共轭通信算子 f / g：
#   f：前向 identity，反向 all-reduce；  g：前向 all-reduce，反向 identity。
# 下面用单进程模拟 2-way 切分（all-reduce 退化为对分片求和 / all-gather 退化为 cat），
# 并验证 TP 结果与未切分 Linear 完全一致。Single-process simulation of 2-way TP.

def column_parallel(X, W, b, n_shards=2):
    """列并行：按 out_features 切 W=[W_1..W_n]，各卡本地算 X·W_iᵀ，输出沿特征维分片。
    Column-parallel: split W along output dim; no comm needed to get sharded output."""
    Ws, bs = torch.chunk(W, n_shards, dim=0), torch.chunk(b, n_shards, dim=0)
    outs = [X @ Wi.T + bi for Wi, bi in zip(Ws, bs)]   # 每张卡独立计算 / local matmul
    return torch.cat(outs, dim=-1)                      # g：gather（此处 cat 模拟）

def row_parallel(X, W, b, n_shards=2):
    """行并行：输入 X 已沿特征维分片，按 in_features 切 W，各卡算部分积后 all-reduce 求和。
    Row-parallel: input is feature-sharded; partial products summed via all-reduce."""
    Xs, Ws = torch.chunk(X, n_shards, dim=-1), torch.chunk(W, n_shards, dim=1)
    partial = [Xi @ Wi.T for Xi, Wi in zip(Xs, Ws)]
    return sum(partial) + b                             # f 的共轭：all-reduce（此处 sum 模拟），bias 只加一次

# --- 验证：TP 等价于普通 Linear / TP equals a plain Linear ---
torch.manual_seed(0)
B, d_in, d_out = 4, 8, 6
X = torch.randn(B, d_in)
ref = nn.Linear(d_in, d_out)
W, b = ref.weight.data, ref.bias.data                  # W: (d_out, d_in), b: (d_out,)
Y_ref = ref(X)
print("column-parallel max err:", (column_parallel(X, W, b) - Y_ref).abs().max().item())  # ~0
print("row-parallel    max err:", (row_parallel(X, W, b) - Y_ref).abs().max().item())     # ~0
```

> Megatron MLP 把 **column-parallel → GeLU（本地）→ row-parallel** 串联，整个块前向只需 **一次** all-reduce（反向一次），把通信摊薄到最少。

---

## 三、面试题 / Interview Questions

### L1 — 基础 / Basic

<details>
<summary>Q1: Transformer 中 self-attention 的时间复杂度是多少？如何降低？</summary>

**答 / Answer：** 标准 self-attention 时间复杂度为 $O(n^2 d)$（$n$ 为序列长度，$d$ 为维度），因为需要计算 $n \times n$ 的注意力矩阵。降低方法包括：
- **FlashAttention：** 不改变数学结果，通过 tiling 和 recomputation 减少 HBM 访问，实际墙钟时间降低
- **稀疏 Attention：** Longformer/BigBird 使用局部窗口 + 全局 token，复杂度降至 $O(n \cdot w)$
- **Linear Attention：** 用核函数近似 softmax，复杂度 $O(n d^2)$，但精度通常有损失

**追问 / Follow-up：** FlashAttention 为什么不算"近似"attention？它做了哪些底层优化？

> FlashAttention 将 Q、K、V 分块（tiling）载入 SRAM，在 SRAM 中计算 softmax 的 online normalization（通过维护 running max 和 running sum），然后将结果写回 HBM。数学上与标准 attention 完全等价，只是减少了 HBM 读写次数。

</details>

---

<details>
<summary>Q2: 什么是 Layer Normalization？它和 Batch Normalization 有什么区别？</summary>

**答 / Answer：**
- **BatchNorm：** 对同一特征维度，跨 batch 维度计算均值和方差。训练时需维护 running mean/var，推理时用固定统计量。对 batch size 敏感，不适合变长序列。
- **LayerNorm：** 对同一样本，跨特征维度计算均值和方差（每个 token 独立归一化），不依赖 batch 统计。Transformer 中的标准选择。

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

**追问 / Follow-up：** RMSNorm 相比 LayerNorm 有什么优势？

> RMSNorm 去掉了 mean-centering 步骤，仅做 variance normalization：$\text{RMSNorm}(x) = \gamma \odot x / \sqrt{\text{mean}(x^2) + \epsilon}$。计算量略少，实践效果相近，被 LLaMA 系列采用。

</details>

---

<details>
<summary>Q3: 什么是梯度裁剪（gradient clipping）？为什么 LLM 训练中几乎必用？</summary>

**答 / Answer：** 梯度裁剪将梯度的范数限制在一个阈值以内：
$$
\text{if } \|g\| > c: \quad g \leftarrow g \cdot \frac{c}{\|g\|}
$$
LLM 训练中，少数异常样本可能产生极大梯度（gradient spike），导致参数突变甚至 NaN。梯度裁剪（典型 $c = 1.0$）是防止训练崩溃的标准手段。

**追问 / Follow-up：** 如何判断 gradient clipping 的阈值设得是否合适？

> 观察训练日志中 clipping 的触发频率。偶尔触发（< 5% 的步数）是正常的；若频繁触发说明 learning rate 可能过大；若从未触发且训练稳定，阈值可能偏大。

</details>

---

<details>
<summary>Q4: 什么是 warmup 和 cosine decay？为什么 LLM 预训练常用这个 LR schedule？</summary>

**答 / Answer：**
- **Warmup：** 训练初期线性增加 learning rate（通常前 1%–3% steps），因为初始时模型参数随机，梯度方向不稳定，大学习率容易发散。
- **Cosine decay：** warmup 后 LR 按余弦曲线从峰值衰减到接近零：$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$

**追问 / Follow-up：** WSD（Warmup-Stable-Decay）schedule 和 cosine schedule 有什么区别？

> WSD 在 warmup 后保持恒定 LR（stable phase），最后再快速 decay。优势是中间 checkpoint 质量较好，适合需要在训练中间取 checkpoint 做 downstream 评估的场景。

</details>

---

<details>
<summary>Q5: 解释 flash attention 的基本原理，为什么它能加速？</summary>

**答 / Answer：** FlashAttention 的核心是 **IO-aware** 算法设计：
1. 将 Q、K、V 切成小块（block），每块足够小以放入 GPU SRAM（on-chip memory）
2. 在 SRAM 中完成 softmax 和矩阵乘法
3. 使用 **online softmax**（通过维护 row-wise max 和 sum 的 running statistics）避免需要全局信息才能计算 softmax
4. 不需要将 $n \times n$ 的注意力矩阵写入 HBM（显存），从而减少 HBM 读写量

加速来源：标准 attention 需要将注意力矩阵写入/读出 HBM，HBM 带宽是瓶颈；FlashAttention 将计算集中在 SRAM，HBM 读写量从 $O(n^2)$ 降到 $O(n^2 d^2 / M)$（$M$ 为 SRAM 大小）。

**追问 / Follow-up：** FlashAttention 对训练和推理的收益分别有多大？

> 训练中主要节省反向传播时的 HBM 访问（正向不存注意力矩阵，反向需要时重新计算）；推理中主要在 prefill 阶段受益（长 prompt），decode 阶段（单 token）收益较小。

</details>

---

<details>
<summary>Q6: 什么是 PEFT（Parameter-Efficient Fine-Tuning）？列举至少三种方法并简述。</summary>

**答 / Answer：**
- **LoRA / QLoRA：** 在权重矩阵旁插入低秩旁路（$BA$），仅训练旁路参数。QLoRA 进一步将基础权重量化为 4-bit。
- **Prefix Tuning：** 在每层 attention 的 key 和 value 前拼接可训练的"虚拟 token"向量。
- **Adapter：** 在 Transformer 子层之间插入小型 MLP bottleneck（down-projection → 非线性 → up-projection），仅训练 adapter 参数。
- **Prompt Tuning：** 在输入 embedding 前拼接少量可训练的 soft prompt 向量（仅在输入层）。

**追问 / Follow-up：** 这些方法的参数效率和表达能力之间有什么 trade-off？

> 参数越少越省显存，但表达能力上限越低。LoRA 因直接作用于权重矩阵，在参数量相近时通常表现优于 adapter 和 prefix tuning。极端场景（如仅有几十条数据）下，参数少反而能防止过拟合。

</details>

---

<details>
<summary>Q7: Continuous batching 和 static batching 有什么区别？</summary>

**答 / Answer：**
- **Static batching：** 收集一批请求，等所有请求都生成完毕才释放 batch。如果一个请求很短而其他很长，短请求完成后 GPU 资源被浪费（padding 等待）。
- **Continuous batching（iteration-level scheduling）：** 每生成一步（一个 token），就检查是否有请求完成，完成的请求立即被新请求替换。GPU 利用率显著提升。

**追问 / Follow-up：** PagedAttention 和 continuous batching 是配合使用的吗？

> 是的。Continuous batching 解决了"什么时候调度请求"的问题，PagedAttention 解决了"KV cache 如何分配显存"的问题——将 KV cache 分成固定大小的 page，按需分配，避免因请求长度不一导致的显存碎片。

</details>

---

### L2 — 中级 / Intermediate

<details>
<summary>Q8: 解释 ZeRO 的三个阶段分别做了什么，各自的通信开销如何？</summary>

**答 / Answer：**
- **ZeRO-1：** 将 optimizer states（Adam 的 $m$ 和 $v$，占 8 bytes/参数 in FP32）分片到各卡。每卡只维护 $1/P$ 的 optimizer state，更新后 AllGather 参数。
- **ZeRO-2：** 在 ZeRO-1 基础上，将梯度也分片。每卡只保存 $1/P$ 的梯度（其余 Reduce-Scatter 后丢弃）。
- **ZeRO-3：** 参数也分片。前向和反向时，按需 AllGather 所需参数，用完释放。

通信量：ZeRO-1 和 ZeRO-2 与标准 DP 相同（每步 $2|\theta|$）。ZeRO-3 额外增加前向的 AllGather 通信（约 $1.5\times$ 总通信量）。

**追问 / Follow-up：** 在什么情况下 ZeRO-2 比 ZeRO-3 更好？

> 当模型能 fit 到单卡的参数显存中（但 optimizer states 放不下）时，ZeRO-2 通信量更小。典型场景是用 gradient checkpointing + ZeRO-2 微调中等规模模型（如 7B–13B）。

</details>

---

<details>
<summary>Q9: PPO 在 RLHF 中的具体实现流程是什么？为什么需要 KL 惩罚？</summary>

**答 / Answer：** RLHF-PPO 每步流程：
1. 采样一批 prompt，用当前 policy $\pi_\theta$ 生成 response
2. 用 reward model 对每个 (prompt, response) 打分
3. 用 reference policy $\pi_{\text{ref}}$ 计算 KL 惩罚
4. 计算 advantage（通常用 GAE）
5. 用 PPO clip 目标更新 policy（多轮 mini-batch 更新）

**需要 KL 惩罚的原因：** 没有 KL 约束，policy 会快速偏向 reward model 的 OOD（out-of-distribution）盲区——生成 reward model 评分高但人类实际不喜欢的回复（reward hacking）。KL 惩罚让 policy 不偏离 $\pi_{\text{ref}}$（即 SFT 模型）太远。

**追问 / Follow-up：** Reward hacking 能给一个具体例子吗？

> 比如 reward model 偏好长回答（因为训练数据中好答案通常较长），policy 可能学到无论什么问题都生成很长的、充满重复内容的回答来获得高分，但人类评估者会觉得冗长无用。

</details>

---

<details>
<summary>Q10: 如何防止指令微调（instruction tuning）导致的灾难性遗忘？</summary>

**答 / Answer：** 常见方法：
- **Replay / 混合训练：** 在 SFT 数据中混入一部分通用指令数据或预训练数据
- **LoRA / PEFT：** 只更新少量参数，预训练知识保留在冻结的主权重中
- **正则化：** EWC（Elastic Weight Consolidation）等方法对重要参数施加惩罚，防止大幅偏离
- **低学习率：** 全参微调时用比预训练低 1–2 个数量级的 LR

**追问 / Follow-up：** 如何量化"灾难性遗忘"的程度？

> 在微调前后分别在通用 benchmark（如 MMLU、HellaSwag）和目标任务 benchmark 上评估。若通用 benchmark 性能下降超过几个百分点，说明存在显著遗忘。

</details>

---

<details>
<summary>Q11: GPTQ 和 AWQ 的核心思路有什么不同？</summary>

**答 / Answer：**
- **GPTQ（Optimal Brain Quantization 系列）：** 逐层量化，利用二阶信息（Hessian 逆）最小化量化前后的层输出重建误差。按列顺序量化，每量化一列就更新剩余列的补偿。
- **AWQ（Activation-Aware Weight Quantization）：** 核心观察是少数"显著通道"（salient channels，激活值大的通道）对输出质量至关重要。AWQ 对这些通道的权重进行保护（如使用 per-channel scaling 提升有效精度），而非均匀量化所有权重。


**追问 / Follow-up：** 量化到 INT4 时，为什么 smooth quant 对激活值很重要？

> 激活值中常有 outlier（异常大的值），导致量化范围被拉大，有效精度降低。SmoothQuant 通过将激活的 outlier "迁移"到权重中（数学上等价的 per-channel scaling），使激活分布更均匀，从而使权重和激活都能量化到较低位宽。

</details>

---

<details>
<summary>Q12: Sequence Parallelism 和 Tensor Parallelism 如何配合工作？</summary>

**答 / Answer：** 在 Megatron-LM 的设计中：
- TP 切分线性层（attention 和 MLP 的权重矩阵）
- SP（Sequence Parallelism）切分 **非线性操作**（LayerNorm、Dropout）的激活值——沿序列维度切分
- 连接点：TP 层结束时需要 AllReduce（或 ReduceScatter），SP 层结束时也需通信。Megatron-LM 将这两个通信融合（fuse），实际没有增加通信量。

好处：TP 的 AllReduce 后激活在每卡上是完整序列（冗余），SP 去掉这个冗余，每卡只保存 $1/P$ 的序列激活，显著降低激活内存。

**追问 / Follow-up：** SP 对梯度 checkpointing 有帮助吗？

> 有。SP 减少了每卡保存的激活量，如果不用 gradient checkpointing，激活显存从 $O(L \cdot n \cdot d)$ 降到 $O(L \cdot n/P \cdot d)$。即使用了 gradient checkpointing，recompute 时的临时显存也相应减少。

</details>

---

<details>
<summary>Q13: 解释 RLHF 中 reward model 的训练方法，以及如何评估 reward model 的质量。</summary>

**答 / Answer：**
- **训练：** 使用 Bradley-Terry 偏好模型。给定 prompt $x$ 和一对 response $(y_w, y_l)$（$y_w$ 被标注为更好），reward model 的 loss 为 $-\log \sigma(r(x, y_w) - r(x, y_l))$。模型通常从 SFT 模型初始化，去掉语言模型 head，换上一个输出标量的 head。
- **评估指标：**
  - **偏好预测准确率：** 在 held-out 偏好对上预测哪个更好的准确率
  - **Reward 分布区分度：** chosen 和 rejected 的 reward 分布是否充分分离
  - **Reward hack 鲁棒性：** 在 policy 生成的 OOD response 上，reward 是否仍能合理排序

**追问 / Follow-up：** 为什么 reward model 需要定期更新？

> 因为 policy 在 RL 训练中不断变化，生成的 response 分布会逐渐偏离 reward model 训练时的数据分布（即 SFT 模型的输出分布）。在分布外数据上，reward model 可能给出不准确的评分，导致 reward hacking。

</details>

---

<details>
<summary>Q14: vLLM 的 PagedAttention 解决了什么问题？具体机制是什么？</summary>

**答 / Answer：** 
- **问题：** 传统 KV cache 为每个请求预分配一块连续显存（按最大序列长度）。但实际生成长度不一，导致大量显存浪费（内部碎片）且无法在请求间共享（外部碎片）。
- **PagedAttention 机制：** 借鉴操作系统的虚拟内存分页思想：
  1. 将 KV cache 分成固定大小的 **block**（如每 block 存 16 个 token 的 KV）
  2. 用 **block table** 记录每个请求的逻辑 block 到物理 block 的映射
  3. 生成新 token 时动态分配新 block，请求结束后释放
  4. 支持 **copy-on-write**：对 beam search 中共享同一 prefix 的多个 beam，KV block 可以共享

**追问 / Follow-up：** PagedAttention 对 latency 有负面影响吗？

> block table 的地址间接寻址引入了少量开销（相对于连续内存直接访问），但在实际推理中这个开销非常小（通常 < 5%），因为 attention 计算本身是 compute-bound 或 memory-bound 的，寻址开销不是瓶颈。

</details>

---

<details>
<summary>Q15: 如何设计一个 LLM 的离线评估套件（eval harness）？需要考虑哪些方面？</summary>

**答 / Answer：**
- **任务抽象：** 每个 task 定义 dataset、prompt template（few-shot 格式）、metric、output type（generation / loglikelihood）
- **评估模式：**
  - Likelihood-based（如 MMLU）：计算各选项 log-prob，选最大者
  - Generation-based（如 GSM8K）：生成后用规则/code exec 判断
  - LLM-as-judge（如 MT-Bench）：用更强的模型打分
- **可复现性：** 固定 seed、记录 prompt template 和 few-shot 示例、temperature=0（或固定）
- **效率：** likelihood 题适合大 batch；generation 题按长度排序减少 padding
- **防污染：** 检测训练数据与 test set 的 n-gram 重叠

**追问 / Follow-up：** 为什么要区分 "knowledge" 和 "reasoning" 评估？

> 因为模型可能在 knowledge-heavy 任务（如 MMLU 中的事实题）上表现好，但在 reasoning-heavy 任务（如数学、代码）上表现差，反之亦然。分开评估有助于定位模型能力短板。

</details>

---

<details>
<summary>Q16: 如何为 LLM 微调选择合适的 LoRA rank？</summary>

**答 / Answer：** 需要考虑的因素：
- **任务复杂度：** 简单的分类/抽取任务 r=4–16 通常足够；复杂的推理/生成任务可能需要 r=32–64
- **数据量：** 数据少时用小 rank 防止过拟合；数据充足时可以增大 rank 提升容量
- **target modules：** 仅对 q_proj, v_proj 应用 LoRA（参数最少）→ 对所有 linear 层应用（q/k/v/o + MLP 的 gate/up/down）参数更多但效果通常更好
- **常见做法：** 从 r=16 开始，α=2r，在验证集上比较 r=8/16/32/64 的效果

**追问 / Follow-up：** LoRA 可以和 QLoRA 结合使用吗？4-bit 量化基础权重 + LoRA 低秩更新的精度损失大吗？

> 可以，QLoRA 就是这个思路。实践表明，4-bit NF4 量化基础权重 + LoRA 微调，在多数任务上与 FP16 全参微调的差距在可接受范围内（通常 1–3 个百分点内），但显存节省巨大。

</details>

---

<details>
<summary>Q-RLHF-A（L2）：为什么 naive co-located PPO 的 GPU 利用率低？Disaggregated 架构如何解决这个问题？</summary>

**答 / Answer：**

Naive co-located PPO 将 rollout 和 train 串行在同一批 GPU 上：

- **Rollout 阶段**：actor 做自回归推理（memory-bound，吞吐受 HBM 带宽限制），trainer 空等。
- **Train 阶段**：PPO 反向传播计算密集，rollout worker 空等。

两段交替，整体 GPU 利用率等于两段分别利用率的加权平均，远低于纯训练峰值。

**Disaggregated 架构的解法：**
1. 独立的 **rollout workers**（vLLM/SGLang 引擎）持续生成 response，产出 rollout buffer。
2. 独立的 **train workers**（ZeRO-3/FSDP）从 buffer 取数据，持续执行 PPO/GRPO 更新。
3. 两组 worker **并发运行**，权重以某一频率（通常每 iteration）同步。

这样 rollout 和 train 各自针对自身负载优化（推理引擎 vs. 训练框架），不再相互阻塞。

**追问 1 / Follow-up 1：** Disaggregated 架构下，rollout worker 和 train worker 需要多大的权重同步带宽？

> 以 7B 参数 BF16 为例，一次完整权重同步约 14 GB 数据。若每分钟同步一次，约 14 GB ÷ 60 s ≈ 0.23 GB/s，远低于 NVLink/RDMA 带宽上限（同步开销可忽略）。若用 LoRA-in-RL，只需同步 LoRA 参数（量级 ~100 MB），同步开销大幅降低。

**追问 2 / Follow-up 2：** Async rollout 引入的 staleness 对 PPO 有什么影响？如何缓解？

> Staleness 导致 rollout 使用旧参数 $\pi_{\theta_{\text{old}}}$ 生成数据，形成 off-policy 偏差。PPO 的 importance ratio clip（$\epsilon \approx 0.1\text{–}0.2$）对小幅 staleness 有容忍度，但 staleness 过大时梯度估计方差增大，训练不稳定。缓解方式：控制权重同步频率（不超过几个 mini-batch 更新），或使用更激进的 importance sampling 校正。

</details>

---

### L3 — 深度 / Deep

<details>
<summary>Q17: Megatron-LM 的 Column-Parallel 和 Row-Parallel Linear 是如何减少 AllReduce 次数的？</summary>

**答 / Answer：**

考虑两层连续的线性变换 $Y = GELU(XA)B$（MLP block），$A \in \mathbb{R}^{h \times 4h}$，$B \in \mathbb{R}^{4h \times h}$：

1. **Column-Parallel $A$：** 将 $A$ 按列切为 $[A_1, A_2]$，每卡计算 $GELU(X A_i)$——独立完成，**无需通信**。GELU 是逐元素操作，天然可分。
2. **Row-Parallel $B$：** 将 $B$ 按行切为 $\begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$，每卡计算 $Y_i = GELU(XA_i) B_i$。
3. **最后 AllReduce：** $Y = Y_1 + Y_2$（一次 AllReduce）。

关键洞察：Column-Parallel 输出正好是 Row-Parallel 的输入，中间的非线性函数（GELU）是逐元素的，不需要通信。因此 **整个 MLP block 只需一次 AllReduce**（前向），反向时也只需一次。

若不做这个设计，每层都需 AllReduce，通信量翻倍。

**追问 / Follow-up：** Attention block 的 QKV 投影和 output 投影也能用同样的技巧吗？

> 是的。QKV 投影用 Column-Parallel（输出分给各 head，自然按列切分），output 投影用 Row-Parallel，然后 AllReduce。整个 attention block 也只需一次 AllReduce。

</details>

---

<details>
<summary>Q18: Speculative Decoding 为什么是无损的？推导接受概率。</summary>

**答 / Answer：**

设 target model 分布为 $p(x)$，draft model 分布为 $q(x)$。

**接受-拒绝采样：**
1. 从 $q(x)$ 采样 token $x$
2. 若 $p(x) \geq q(x)$：接受（概率 1）
3. 若 $p(x) < q(x)$：以概率 $p(x)/q(x)$ 接受

**接受 token 为 $x$ 的总概率：**
- 从 $q$ 采样到 $x$ 且接受：$q(x) \cdot \min(1, p(x)/q(x)) = \min(p(x), q(x))$
- 从 $q$ 采样到 $x$ 且拒绝后重新采样到 $x$：更复杂但可推导

**最终有效概率：**

$$
P(\text{output}=x) = \min(p(x), q(x)) + \frac{\max(0, p(x) - q(x))}{1 - \sum_i \min(p(i), q(i))} \cdot \delta
$$

实际上，可以证明通过上述拒绝采样 + 修正采样，最终输出分布 **精确等于** $p(x)$。

核心直觉：当 $p(x) > q(x)$ 时，draft model "欠采样"了 $x$，需要从 rejection 的剩余概率中补偿；当 $p(x) < q(x)$ 时，通过拒绝来"减掉"多余概率。

**追问 / Follow-up：** Speculative decoding 的效率瓶颈在哪里？

> 瓶颈在于 draft model 的接受率。如果 draft model 和 target model 分布差距大，接受率低，大部分 draft token 被拒绝，加速效果差。改善方法包括：用 medusa-style 多头预测、或选择与 target model 分布更接近的 draft model。

</details>

---

<details>
<summary>Q19: DPO 从 Bradley-Terry 偏好模型是如何推导出来的？</summary>

**答 / Answer：**

**Step 1：** Bradley-Terry 模型假设最优 policy $\pi^*$ 满足：

$$
p(y_w \succ y_l | x) = \sigma(r^*(x, y_w) - r^*(x, y_l))
$$

**Step 2：** 在 KL 约束下，最优 policy 的封闭解为：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

其中 $Z(x)$ 是配分函数。

**Step 3：** 从中解出 reward：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

**Step 4：** 将 $r$ 代入 Bradley-Terry 模型，$Z(x)$ 在差值中抵消：

$$
p(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

**Step 5：** 将 $\pi^*$ 替换为可训练的 $\pi_\theta$，取负对数似然即得 DPO loss。

**追问 / Follow-up：** DPO 推导假设偏好数据来自最优策略，这个假设在实践中会带来什么问题？

> 实践中偏好数据通常来自 SFT 模型（非最优策略），这导致 DPO 隐式学习的 reward 可能不够准确。这也是 online DPO（iterative DPO，每轮用最新 policy 生成数据）效果通常优于 offline DPO 的原因。

</details>


---

<details>
<summary>Q20: 评估 LLM 时，benchmark 饱和（saturation）是什么问题？如何应对？</summary>

**答 / Answer：**
- **问题：** 当主流模型在某个 benchmark（如 MMLU）上得分接近天花板（如 >90%），区分度下降。可能的原因包括：
  - 训练数据污染（test set 数据被混入训练集）
  - 任务本身难度不足（主要是知识检索，非深度推理）
  - 评测格式被优化（模型针对 benchmark 的 prompt 格式做了优化）
- **应对方法：**
  - 使用更难的 benchmark（如 MMLU-Pro、GPQA、MATH）
  - 使用动态生成的评测题目
  - 关注人类评估（如 Chatbot Arena 的 Elo 排名）
  - 检测和报告数据污染情况

**追问 / Follow-up：** HELM 和 lm-evaluation-harness 的设计哲学有什么不同？

> HELM（Stanford）强调"全面性"——覆盖多维度（accuracy、calibration、robustness、fairness、efficiency），每个 scenario 都有详细的文档和标准化评测流程，但扩展新任务较重。lm-evaluation-harness（EleutherAI）强调"灵活性和社区贡献"——任务定义简洁（config-driven），社区可快速添加新任务，400+ 任务覆盖广泛，但标准化程度相对较低。

</details>

---

<details>
<summary>Q21: 解释 Disaggregated Serving（prefill/decode 分离）的动机和设计。</summary>

**答 / Answer：**

**动机：** Prefill（处理 prompt）和 Decode（逐 token 生成）的计算特征完全不同：

| 特征 | Prefill | Decode |
|------|---------|--------|
| 计算类型 | Compute-bound（大矩阵乘法） | Memory-bound（小 batch，大量 KV cache 访问） |
| GPU 利用率 | 高（计算密集） | 低（内存带宽瓶颈） |
| 最优配置 | 高算力 GPU | 高显存带宽 |

**Disaggregated Serving 设计：**
- Prefill 节点：高算力配置，大 batch 处理 prompt → 生成 KV cache
- Decode 节点：高带宽配置，接收 KV cache → 逐 token 生成
- KV cache 通过高速网络（RDMA/NCCL）在节点间传输

**收益：** 两阶段可以独立扩缩容，避免 decode 阶段的 memory-bound 特性拖累 prefill 的 compute utilization。

**追问 / Follow-up：** KV cache 传输的带宽需求有多大？

> 对于一个 70B 模型、序列长度 4K、FP16 KV cache，每个请求的 KV cache 大约在几百 MB 量级。若 decode 节点需每秒处理数十个请求的 KV cache 接入，则需要数十 GB/s 的网络带宽，这在现代数据中心的 RDMA 网络下是可行的。

</details>


---

<details>
<summary>Q22: 如何在分布式训练中处理梯度检查点（gradient checkpointing）的显存-计算 trade-off？</summary>

**答 / Answer：**
- **原理：** 正向传播时不保存中间激活值，仅保存部分"检查点"（通常每层边界保存一次）。反向传播时从最近的检查点重新计算所需的激活。
- **显存：** 从 $O(L \cdot a)$（$a$ 为每层激活大小）降到 $O(\sqrt{L} \cdot a)$ 或 $O(L')$（$L'$ = 检查点数量）
- **计算：** 额外约 33% 的正向计算（每个检查点段需重新前向一次）

**实践选择：**
- 显存充足时不用（节省时间）
- 显存不够但能承受 33% 训练变慢时开启
- 可选择性开启（如只对某些大层做 checkpoint）

**追问 / Follow-up：** Selective gradient checkpointing 如何选择哪些层做检查点？

> 通常选择激活值最大的层（如 attention 层的注意力矩阵是 $O(n^2)$ 的显存大户），而激活值较小的层（如 LayerNorm、embedding）不做 checkpoint，从而在显存节省和计算开销间取得更好的平衡。

</details>

---

<details>
<summary>Q23: 解释 PPO 中的 clipping 机制，以及在 RLHF 中为何可能需要调整。</summary>

**答 / Answer：**

PPO 的 clipped surrogate objective：

$$
L^{CLIP} = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$，$\epsilon$ 通常为 0.1–0.2。

**作用：** 当 $r_t$ 偏离 1 太远时，clip 限制了目标函数的变化幅度，防止单步更新过大。

**在 RLHF 中的特殊考虑：**
- 标准 RL（游戏等）中 state-action 空间大，$r_t$ 偏离不多
- 在 RLHF 中，language model 的生成空间是指数级的，policy 可能快速变化
- 因此 $\epsilon$ 可能需要调小，或者增加 PPO 更新的 epoch 数来充分利用每批采样

**追问 / Follow-up：** PPO 中的 value function loss 和 policy loss 如何平衡？

> 通常用加权求和：$L = L^{CLIP} + c_1 L^{VF} - c_2 H(\pi)$，其中 $L^{VF}$ 是 value function 的 MSE loss，$H(\pi)$ 是 entropy bonus 防止过早坍缩。在 RLHF 中 $c_1$ 和 $c_2$ 的调优对训练稳定性很关键。

</details>

---

<details>
<summary>Q24: 如何设计一个能检测 benchmark 数据污染（contamination）的系统？</summary>

**答 / Answer：**
- **N-gram 重叠检测：** 将 test set 的 n-gram（如 8-gram、13-gram）与训练数据做集合交集。若重叠率超过阈值，标记为可能被污染。
- **Membership inference：** 检查模型对 test set 样本的困惑度是否异常低（与 held-out 数据相比），低困惑度可能暗示该样本曾出现在训练集中。
- **Canonical order test：** 打乱选项顺序，若正确率大幅下降，可能模型记忆了特定位置的答案（暗示污染而非真正理解）。
- **Canary test：** 在 test set 中插入独特的"金丝雀"句子，训练后检查模型能否完美复述。

**追问 / Follow-up：** 为什么 n-gram 重叠检测可能产生假阳性？

> 因为一些公共知识（如"太阳从东边升起"）在训练集和测试集中都会出现，n-gram 重叠不代表真正的"记忆"。需要区分"事实性公共知识"和"特定测试样本的逐字复制"。

</details>

---

<details>
<summary>Q-RLHF-B（L3）：设计一个支持 70B actor 的 RLHF 训练系统。描述 4 模型的显存拆解方案、rollout/train 拓扑，以及你在 LoRA-in-RL vs 全参数更新之间如何选择。</summary>

**答 / Answer：**

**第一步：Clarify**
- 70B actor（约 140 GB BF16 参数）+ critic（同量级或小一号）+ ref model + RM
- 4 模型 naive co-located 显存需求：参数 + optimizer states 约在 1 TB 量级（不可行，需分离）
- 目标：在 8–64 张 80G A100/H100 上跑起来，吞吐满足合理的训练周期

**第二步：4 模型显存拆解（量级估算）**

| 模型 | 参数（BF16） | 梯度 | 优化器（FP32 AdamW） | 部署策略 |
|------|------------|------|-------------------|---------|
| Actor（训练） | ~140 GB | ~140 GB | ~560 GB | Train workers，ZeRO-3 分片 |
| Critic（训练） | ~140 GB（可用小模型） | ~140 GB | ~560 GB | 同上，或独立 ZeRO 组 |
| Ref model（冻结） | ~140 GB | 无 | 无 | Rollout workers，推理模式 |
| Reward model（冻结） | 数 GB–~140 GB | 无 | 无 | Rollout workers |

- 全参数训练时，actor + critic 的完整训练状态（参数 + 梯度 + 优化器）约 1.5–2 TB 量级，ZeRO-3 分片到 train workers 需**数十张** 80G GPU（具体取决于是否含 FP32 master copy、激活与框架 overhead）。
- 使用 **LoRA-in-RL**（rank=16–32）时，actor 可训练参数下降到总参数的 $\lesssim 1\%$，optimizer states 从 ~560 GB 降到数 GB 量级，大幅降低 train workers 显存需求。

**第三步：拓扑设计**

```
Rollout cluster（推理优化）          Train cluster（训练优化）
┌──────────────────────────┐         ┌─────────────────────────┐
│ vLLM / SGLang            │         │ ZeRO-3 / FSDP           │
│  - actor (FP16 weights)  │◄──权重──│  - actor (trainable)    │
│  - ref model (frozen)    │  同步   │  - critic (trainable)   │
│  - RM (frozen)           │         │                         │
│                          │──data──►│  rollout buffer         │
│  连续 rollout，输出       │         │  PPO / GRPO 更新         │
│  (prompt, resp, reward,  │         │                         │
│   log_prob, value)       │         │                         │
└──────────────────────────┘         └─────────────────────────┘
```

- rollout 与 train **并发**（异步）或**交替**（同步），权重每 iteration 同步一次。
- Ref model 和 RM 只需推理，放 rollout 侧节省 train 侧显存。

**第四步：LoRA-in-RL vs 全参数更新的选择**

| 考量 | 倾向 LoRA-in-RL | 倾向全参更新 |
|------|----------------|------------|
| 显存预算 | 严格（少卡） | 充裕（多卡） |
| 策略需要改变的幅度 | 小（对话风格对齐） | 大（复杂推理能力提升） |
| 训练稳定性 | 更稳定（小秩约束） | 需更仔细调 $\beta$, clip |
| 参考 | OpenRLHF LoRA 模式 | veRL / Megatron-LM 全参 |

⚠️ 以上显存数字为**数量级估算**（基于参数量 × bytes/参数的公式推算），实际值因激活、KV cache、框架 overhead 而有较大差异，面试中请说明"估算"。

**追问 / Follow-up：** 在 disaggregated 架构中，rollout 和 train 资源比例如何决定？

> 取决于 rollout throughput 与 train throughput 的比值。若 rollout 是瓶颈（response 很长、batch 很大），增加 rollout worker 数；若 train 是瓶颈（critic 计算量大、PPO mini-batch 多），增加 train worker 数。实践中先 profile 两侧的 GPU-hours / iteration，按比例分配，再根据实际队列 utilization 调整。

</details>

---

<details>
<summary>Q25: 综合设计题：为一个日活千万的 AI 客服系统设计完整的 LLM 系统，从数据到部署。</summary>

**答 / Answer（高层概要）：**

**1. Clarify：**
- 日活千万 → QPS 估计约 100–1000（考虑每个用户日均 1–3 轮对话）
- 延迟 SLA：P95 < 2s（首 token），P99 < 5s
- 需要领域适配（客服话术、产品知识）

**2. Data：**
- 历史客服对话日志 → 清洗脱敏 → 构建 SFT 数据
- 定期从线上 bad case（低评分、转人工）中采样 → 人工标注 → 回流训练
- RAG：将产品文档、FAQ 构建为向量知识库

**3. Model：**
- Base model：选 7B–13B 量级（平衡效果和推理成本）
- SFT（LoRA）在客服数据上微调
- RAG 检索增强：用户 query → 检索相关文档 → 拼入 prompt context

**4. Serving：**
- 量化：INT8 或 INT4（GPTQ/AWQ）→ 降低单卡推理成本
- vLLM / TensorRT-LLM 部署，continuous batching + PagedAttention
- 多副本 + 负载均衡，按流量自动扩缩容

**5. Monitoring：**
- 线上指标：转人工率、用户满意度评分、平均对话轮次
- 质量漂移：定期在标准测试集上跑 eval，监控分数变化
- Safety：对输出做敏感词和有害内容过滤

**追问 / Follow-up：** 这个系统中，RAG 和微调各解决什么问题？它们可以互相替代吗？

> 微调解决"风格和格式"——让模型以客服的语气和流程回答；RAG 解决"知识和事实"——提供最新的产品信息和公司政策。它们互补而非替代：只微调会"幻觉"产品细节；只 RAG 会让模型语气像通用助手而非专业客服。理想方案是两者结合。

</details>

---

## 附录：关键术语对照表 / Appendix: Key Term Glossary

| 中文 | English | 缩写 |
|------|---------|------|
| 因果语言模型 | Causal Language Model | CLM |
| 低秩适配 | Low-Rank Adaptation | LoRA |
| 参数高效微调 | Parameter-Efficient Fine-Tuning | PEFT |
| 人类反馈强化学习 | Reinforcement Learning from Human Feedback | RLHF |
| 直接偏好优化 | Direct Preference Optimization | DPO |
| 奖励模型 | Reward Model | RM |
| 数据并行 | Data Parallelism | DP |
| 张量并行 | Tensor Parallelism | TP |
| 流水线并行 | Pipeline Parallelism | PP |
| 序列并行 | Sequence Parallelism | SP |
| 零冗余优化器 | Zero Redundancy Optimizer | ZeRO |
| 完全分片数据并行 | Fully Sharded Data Parallel | FSDP |
| 键值缓存 | Key-Value Cache | KV Cache |
| 训练后量化 | Post-Training Quantization | PTQ |
| 基于激活感知的权重量化 | Activation-Aware Weight Quantization | AWQ |
| 投机解码 | Speculative Decoding | — |
| 分页注意力 | PagedAttention | — |
| 检索增强生成 | Retrieval-Augmented Generation | RAG |
| 指令微调 | Instruction Tuning / SFT | SFT |
| 灾难性遗忘 | Catastrophic Forgetting | — |
| 知识蒸馏 | Knowledge Distillation | KD |
| 领域自适应预训练 | Domain-Adaptive Pretraining | DAP |

---



## 更多 L3 深挖 / Extended L3

<details>
<summary>Q26: Explain the IO-aware tiling strategy in FlashAttention. Why does standard attention have a memory access bottleneck, and how does the online softmax trick enable block-wise computation without materializing the full N×N attention matrix?</summary>

标准 attention 需要将完整的 $N \times N$ attention matrix 写入 HBM（High Bandwidth Memory），IO 成为瓶颈。FlashAttention 利用 GPU SRAM（速度快但容量小）做 tiling：

1. 将 $Q, K, V$ 分成大小为 $B_r \times d$ 和 $B_c \times d$ 的块，每次只将一个块载入 SRAM
2. 对每个 Q 块，遍历所有 K/V 块，在 SRAM 中计算局部 attention
3. 利用 **online softmax** 维护 running max $m$ 和 running sum $\ell$：处理第 $j$ 个 KV 块后，用修正因子 $e^{m_{j-1} - m_j}$ 更新之前累积的输出 $O_j$，避免需要全局归一化

$$O_j = \text{diag}(\ell_j)^{-1}\Big(e^{m_{j-1}-m_j}\,\ell_{j-1}\,O_{j-1} + \tilde{P}_j V_j\Big)$$

IO 复杂度从 $O(N^2 d)$ 次 HBM 访问降至 $O(N^2 d^2 / M)$（$M$ 为 SRAM 大小），显存从 $O(N^2)$ 降为 $O(N)$（无需物化完整 attention matrix）。

> **追问：** FlashAttention 反向传播需要重新计算 attention matrix（recomputation），这与 gradient checkpointing 的异同是什么？在超长序列场景下，FlashAttention v2 引入了哪些进一步的并行化优化？

</details>

---

<details>
<summary>Q27: RoPE 的 NTK-aware interpolation 如何解决长序列外推问题？为什么简单的 position interpolation 会损失高频信息？</summary>

简单的 position interpolation（PI）将位置 $m$ 统一缩放为 $m \cdot L_{\text{train}} / L_{\text{target}}$，问题在于 RoPE 频率 $\theta_j = 10000^{-2j/d}$ 跨越多个数量级：
- **低维度**（$j$ 小）→ 高频，编码近距离精细位置关系
- **高维度**（$j$ 大）→ 低频，编码远距离粗略位置关系

统一缩放后，高频维度的旋转角度变化过于密集，模型无法区分相邻 token（高频信息被"挤在一起"），相当于对图像做低通滤波后丢失边缘细节。

**NTK-aware interpolation** 将基频从 $b$ 重新缩放为 $b' = b \cdot \alpha^{d/(d-2)}$（$\alpha = L_{\text{target}}/L_{\text{train}}$）：
- 低维度高频部分几乎不变 → 保持局部分辨率
- 高维度低频部分被拉伸 → 编码更长距离

类比 NTK 理论中高频 vs 低频特征的学习难度差异：高频特征需要更高分辨率，低频特征可以安全外推。

> **追问：** YaRN 在 NTK-aware 基础上进一步对 attention score 施加 temperature scaling，其动机是什么？为什么仅修改位置编码不足以完全恢复长上下文任务的性能？

</details>

---

<details>
<summary>Q28: 在 Mixture of Experts (MoE) 架构中，如何设计 auxiliary load balancing loss 来防止 expert collapse？capacity factor 的作用是什么？</summary>

MoE 中的 **expert collapse**（路由坍缩）：少数 expert 被高频选中，其余几乎闲置，模型有效容量浪费。

**Auxiliary load balancing loss：**

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

- $N$ = expert 数，$f_i$ = 被路由到 expert $i$ 的 token 比例（离散统计），$P_i$ = router 对 expert $i$ 的平均概率（连续可微）
- $f_i \cdot P_i$ 项鼓励二者均匀分布：当某 expert 被频繁选中且 router 对其信心也高时，惩罚最大
- $\alpha$ 设较小值，防止主导主训练 loss

**Capacity factor (CF)：** 限制每个 expert 单次处理 token 上限 = $\text{CF} \times T/N$。CF 过小 → token 被丢弃（overflow）→ 信息损失；CF 过大 → 计算浪费（padding）。CF 需根据负载不均匀程度动态调整。

> **追问：** DeepSeek-MoE 提出 fine-grained expert segmentation（将大 expert 拆为多个小 expert）与 shared expert 机制。这种设计如何从根本上缓解负载均衡（要求均匀）与模型能力（要求专精）之间的张力？

</details>

---

<details>
<summary>Q29: ZeRO-3 的 All-Gather 通信如何与前向/反向计算重叠（overlap）？为什么 naive 实现会导致显著的通信瓶颈？</summary>

ZeRO-3 每层前向需 All-Gather 完整参数才能计算。**Naive 实现**：All-Gather → 等待 → 计算 → 释放，通信与计算串行，GPU 空闲等待时间长。

**Overlap 策略（以反向为例的依赖图分析）：**

```
前向：compute(L) ← All-Gather(L)          compute(L+1) ← All-Gather(L+1)
       ↓ 可重叠：compute(L) 执行时，异步 prefetch All-Gather(L+1)
```

- **前向：** 计算第 $l$ 层时，异步启动第 $l+1$ 层参数的 All-Gather（prefetch）。要求：第 $l$ 层计算时间 ≥ 第 $l+1$ 层通信时间。
- **反向：** 类似地，计算第 $l$ 层梯度时 prefetch 第 $l-1$ 层参数，同时 Reduce-Scatter 第 $l$ 层梯度也可与下一层计算重叠。

**代价：** 同时持有的参数副本增加（当前层 + prefetch 层），显存压力上升。总通信量约 $3 \times |\theta|$ per step（高于 DP 的 $2 \times |\theta|$），在跨节点带宽有限时可能成为瓶颈。

> **追问：** 在什么模型规模和硬件条件下，ZeRO-3 的通信开销会变得不可接受，使得 TP（节点内 NVLink）+ ZeRO-2 成为更优选择？请从通信量与计算量的比值角度分析。

</details>

---

<details>
<summary>Q30: DPO 的训练数据是 off-policy 的（由 $\pi_{\text{ref}}$ 生成），这会导致什么理论偏差？Iterative DPO 如何缓解这个问题？</summary>

DPO loss 中的 $\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 本质上是 importance-weighted reward 估计。

**Off-policy 偏差来源：**
- 当 $\pi_\theta$ 与 $\pi_{\text{ref}}$ 分布差距增大时，importance weight 方差增大，梯度估计不稳定
- 训练数据覆盖的 $y$ 空间固定在 $\pi_{\text{ref}}$ 的支撑集上。$\pi_\theta$ 可能已学会生成训练数据中未见过的 response，但这些 response 无法被 DPO loss 评估 → 优化信号存在盲区
- 类似 off-policy RL 中的 distribution shift：策略越偏离数据收集策略，估计越不可靠

**Iterative DPO 的缓解：**
1. 用当前 $\pi_\theta$ 采样新的 response
2. 用 reward model 或人工标注偏好
3. 以新 $\pi_\theta$ 作为新 $\pi_{\text{ref}}$，重新训练 DPO
4. 重复 → 训练数据逐步 on-policy

**Online DPO** 更进一步：在训练循环中实时采样 $\pi_\theta$ 的 output，用 RM 打分后立即更新。

> **追问：** Online DPO 中，如果 reward model 本身存在系统性偏差（如偏好冗长回答），online 迭代会如何放大这个问题？这与 PPO 中的 reward hacking 在机制上有何异同？

</details>

---

<details>
<summary>Q31: RLHF 中 reward model 过优化（overoptimization）的现象如何用理论解释？proxy reward 与真实质量的分歧如何随 KL 增大而变化？</summary>

这是 **Goodhart's Law** 的体现：当优化一个 proxy 指标到极致时，该指标与真实目标脱钩。

**理论直觉：**
- 设真实 reward $r^*(x,y)$，proxy RM $r_\phi(x,y)$，二者之差为 $\delta(x,y) = r_\phi - r^*$
- $\pi_\theta$ 沿 $\nabla_\theta \mathbb{E}[r_\phi]$ 方向优化时，不仅提升了 $r^*$，也同时在"钻 $\delta$ 的空子"——进入 $r_\phi$ 高估的区域
- 随 $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ 增大，策略偏离训练分布越远，$r_\phi$ 的泛化误差（$|\delta|$）单调增大
- 定性观察：proxy reward 持续上升，真实质量先升后降，两条曲线的交叉点即为"过优化拐点"

**影响分歧速率的因素：**
- RM 容量越大、偏好数据越多样 → 拐点出现越晚
- 策略探索空间越大（生成越长、越多样）→ 越容易找到 reward hacking 的路径

**缓解策略：** KL 惩罚、RM ensemble（取多个 RM 的 min 或 variance penalty）、定期更新 RM。

> **追问：** Reward model ensemble 在实践中如何利用多个 RM 之间的一致性与不一致性？取 min、取均值、还是用 disagreement 作为 uncertainty signal 各自的优劣是什么？计算开销如何影响其可行性？

</details>

---

<details>
<summary>Q32: Multi-head Latent Attention (MLA) 如何通过低秩压缩减少 KV cache 显存？与 GQA 在压缩机制上有何本质区别？</summary>

MLA 不再存储完整的 $K, V$，而是存储低维 **latent vector** $c_t^{KV}$，推理时再解压：

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll n_h \cdot d_h$$

KV cache 仅保存 $c_t^{KV}$（维度 $d_c$），计算 attention 时投影回：

$$k_t = W^{UK} c_t^{KV}, \quad v_t = W^{UV} c_t^{KV}$$

KV cache 大小从 $2 \times L \times n_h \times d_h \times s$ 降为 $L \times d_c \times s$（$d_c$ 可远小于 $2 n_h d_h$）。

**与 GQA 的本质区别：**

| 维度 | GQA | MLA |
|------|-----|-----|
| 压缩对象 | head 维度（减少 KV head 数） | feature 维度（低秩投影） |
| 压缩性质 | 离散的、结构化的（head 分组） | 连续的、灵活的（可学习子空间） |
| cache 内容 | 真实 K, V 值（只是 head 少了） | 压缩后的 latent vector（需解压） |
| 多样性保持 | 直接保留独立 head | 依赖低秩子空间的表达能力 |

MLA 的优势：可以在保持较多 Q head 数的同时大幅压缩 cache（head 数不再直接决定 cache 大小）。代价：推理时需额外投影计算，且低秩约束可能限制不同 head 的 pattern 多样性。

> **追问：** MLA 的低秩压缩是否会导致不同 head 的 attention pattern 趋同（loss of head diversity）？投影矩阵 $W^{UK}$ 的高秩性是否能完全缓解这种风险？实践中有什么信号可以检测 head 多样性的退化？

</details>
