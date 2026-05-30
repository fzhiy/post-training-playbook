# 推理-RL 前沿 / Reasoning-RL Frontier

> 2024–2026 post-training 最热的一条线:从 PPO 到 critic-free 的 GRPO / RLOO,再到 long-CoT 推理 RL(DAPO / Dr.GRPO)与 RLVR。面前沿大厂(Seed / DeepSeek / Qwen / Moonshot)高频。
> ⚠️ 具体论文数字(某 benchmark 分数等)以**原论文**为准;本页重**机制与取舍**,刻意不堆数字。

## 0. 一句话演化 / The evolution

`PPO`(actor+critic+ref+RM,GAE 优势)→ **`GRPO`**(去掉 critic,用「组内相对」当 baseline)→ **`DAPO` / `Dr.GRPO`**(修 GRPO 在 long-CoT 下的偏置与熵塌缩);旁支 **`RLOO`**(leave-one-out baseline)。奖励来源:学习的 RM →(在可验证域)**`RLVR`**(规则/验证器给奖励)。

## 1. PPO 回顾(基线)/ PPO recap
- 四个模型:policy(actor)、value(critic)、reference、reward model。
- 优势用 **GAE**;目标是 clipped surrogate $L^{CLIP}=\mathbb{E}\big[\min(\rho A,\ \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)A)\big]$,$\rho=\pi_\theta/\pi_{\theta_{old}}$;外加 $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})$。
- 痛点:显存(4 个模型)、value 网络难训、长序列奖励稀疏。

## 2. GRPO — 去 critic / Group Relative Policy Optimization
- 对每个 prompt 采样**一组** $G$ 个回答,奖励 $r_1..r_G$;用**组内统计**当 baseline 替代 value 网络:
$$A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+\varepsilon}$$
- 目标同 PPO 的 clipped surrogate,但优势用 $A_i$、**无 critic、无 GAE**;保留对 ref 的 KL（估计器 k1/k2/k3 与 in-reward/in-loss 放置见 [llm-post-training §9.4](cheatsheet-llm-post-training.html)）。
- 收益:省一个 value 模型且不用学 value;对**可验证奖励**(数学/代码)特别稳。DeepSeek 系用它。

**from-scratch 实现**(组内 z-score 优势 + 逐 token clip + K3 KL,in-loss):

```python
import torch

def grpo_loss(logp, logp_old, logp_ref, rewards, mask, group_size,
              clip_eps=0.2, beta=0.04):
    # logp/logp_old/logp_ref: (B, T) 逐 token logprob；B = n_prompts * group_size
    r = rewards.view(-1, group_size)                       # (n_prompts, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    adv = adv.reshape(-1, 1)                               # (B,1) 组内 z-score 优势
    ratio = torch.exp(logp - logp_old)                     # importance ratio ρ
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy = torch.min(surr1, surr2)                       # clipped surrogate
    logr = logp_ref - logp                                 # log(π_ref/π_θ)
    kl = torch.exp(logr) - logr - 1                        # K3 估计器，恒 ≥ 0
    per_tok = policy - beta * kl                           # KL 放在 loss 里
    seq = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)  # 1/|o_i| 长度归一化
    return -seq.mean()
# Dr.GRPO 去偏：优势去掉 /std；seq 改用常数归一化（如最大长度）而非 1/|o_i|。
```

- 关键点:① 优势在**组内**标准化(z-score),取代 value baseline;② `min(surr1, surr2)` 是 PPO 同款裁剪,但 ratio 用**逐 token** logprob;③ K3 = $e^{\log r}-\log r-1\ge0$ 是无偏且非负的 KL 估计器(`logr` 方向要对:$\log(\pi_{ref}/\pi_\theta)$);④ 末行 `1/|o_i|` 长度归一化是原始 GRPO 的写法——Dr.GRPO 指出它偏好长错误回答,去偏时改常数。

## 3. RLOO — REINFORCE leave-one-out
- 也 critic-free:样本 $i$ 的 baseline = **其余 $G-1$ 个样本奖励的均值**,REINFORCE 式梯度。
- 比 PPO 简单(无 clip/critic),RLHF 上与 PPO 竞争。与 GRPO 的差异在 baseline 构造(留一 vs 组内标准化)与是否 clip。

## 4. DAPO — 让 GRPO 在 long-CoT 下不崩 / Decoupled-clip & Dynamic-sAmpling PO
ByteDance 2025 开源配方,针对长链推理 RL 的四个改动:
1. **Clip-Higher**:上、下裁剪 $\epsilon$ **解耦**、抬高上界 → 给低概率 token 上升空间,**防熵塌缩**(policy 过早变确定、停止探索)。
2. **Dynamic Sampling**:丢掉「一组全对 / 全错」的 prompt(组内优势恒 0、无梯度),保证每个 batch 都有效。
3. **Token-level loss**:按 **token** 而非 **序列** 平均,避免长回答的梯度被稀释(长 CoT 关键)。
4. **Overlong reward shaping**:对超长回答软惩罚,稳住训练。

## 5. Dr.GRPO — 修 GRPO 的优化偏置 / GRPO Done Right
- 指出 GRPO 两处**偏置**:优势里的 **std 归一化**(放大题目难易不平衡)+ loss 里的 **1/回答长度** 归一化(偏好「更长的错误回答」)。
- 解法:**去掉 std 除法 + 去掉长度归一化**(改用常数)→ 更**无偏**的估计,同性能下 token 更省、回答不虚长。

## 5.5 GSPO — 序列级重要性比 / Group Sequence Policy Optimization

> 💡 GSPO(Qwen 团队,Zheng et al., [arXiv:2507.18071](https://arxiv.org/abs/2507.18071),2025-07)把重要性采样(IS)校正的粒度从「每个 token」升到「整条序列」,缓解 GRPO 在大规模 MoE 训练中的不稳定。

**GRPO 的 token 级比率为何不稳。** GRPO 沿用 PPO,对每个 token 单独算比率 $w_{i,t}=\pi_\theta(y_{i,t}\mid x,y_{i,<t})/\pi_{\theta_\text{old}}(\cdots)$:

- 单 token 比率是单样本估计,方差天然高,长 CoT 中噪声沿序列累积。
- 单个 $w_{i,t}$ 偶尔越过 $[1-\epsilon,1+\epsilon]$,该 token 梯度即被 clip 置零 —— 长序列里频繁发生,即便整体策略偏移很小。
- **MoE 尤甚**:一次更新后路由器可能把同一 token 发给不同专家,分子/分母走了不同计算路径;路由漂移直接表现为比率尖峰,触发 clip,论文称之为「灾难性且不可逆的模型崩溃」(原文)。

**GSPO 的解法:单元匹配(unit matching)。** 奖励赋予整条序列,IS 校正的单元也应是序列。序列级比率取**长度归一化的几何平均**:

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_\text{old}}(y_i\mid x)}\right)^{1/|y_i|}=\exp\!\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_\text{old}}(\cdots)}\right)$$

目标函数形同 PPO clip,但比率换成 $s_i$、优势 $\hat A_i$ 为序列级组内 z-score(同 GRPO):

$$J_\text{GSPO}(\theta)=\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G\min\!\big(s_i\hat A_i,\ \mathrm{clip}(s_i,1{-}\varepsilon_l,1{+}\varepsilon_r)\,\hat A_i\big)\right]$$

整条序列要么采用、要么整体被 clip —— 单个 token 的路由跳变不再能独立触发梯度截断。

| 维度 | GRPO(token 级) | GSPO(序列级) |
|---|---|---|
| IS 比率 | $w_{i,t}=\pi_\theta(y_{i,t})/\pi_{\theta_\text{old}}(y_{i,t})$ | $s_i=(\pi_\theta(y_i)/\pi_{\theta_\text{old}}(y_i))^{1/|y_i|}$ |
| clip 范围(论文设置) | $\varepsilon_l{=}0.2,\ \varepsilon_r{=}0.27$ | $\varepsilon_l{=}3{\times}10^{-4},\ \varepsilon_r{=}4{\times}10^{-4}$ |
| 截断粒度 | 每 token 独立 | 整条序列 |
| MoE 路由漂移 | 比率尖峰 → 误触发 clip | 几何平均平滑大部分抖动 |

> 📝 **别误读 clip 数量级差异。** GSPO 的 $\varepsilon\sim10^{-4}$ 远小于 GRPO 的 $\sim0.2$,这是两种比率**定义不同**带来的设计选择,**不是**几何平均把偏移「压缩到 1」的数学必然 —— 若所有 token 同向偏移,$s_i$ 与 token 比率同阶、并不收缩。几何平均只抹平序列内正负抖动(降方差);GSPO 用极小 $\varepsilon$ 在序列级施加更紧的 proximal 约束,实践中 clip 几乎每步都激活。

**稳定性与工程收益(论文结果,无独立复现):**
- MoE 稳定:序列似然不随单 token 路由漂移剧烈波动,无需此前的 Routing Replay(内部临时方案,本文首次披露)。
- 精度容忍度(precision robustness):序列级聚合对单 token 数值精度不敏感,可直接用推理引擎(如 vLLM)的 log-prob,省去训练引擎重算。
- 在 Qwen3-30B-A3B-Base 上,GSPO 训练曲线(AIME'24 / LiveCodeBench / CodeForces Elo)优于 GRPO;论文称其促成了 Qwen3 模型的性能提升(关联声明,无受控消融)。

> ⚠️ GMPO([arXiv:2507.20673](https://arxiv.org/abs/2507.20673))认为序列级 clip「过于激进」、丢失梯度信息,主张 token 级 clip + 几何平均加权;两者各有取舍,尚无定论。

**CISPO**(MiniMax,[arXiv:2506.13585](https://arxiv.org/abs/2506.13585),2025-06)从另一角度修 clip 截断梯度:不 clip 概率比率(那会让越界 token 梯度归零),而是 clip **标量 IS 权重**本身、保留所有 token 的梯度。论文报告在 Qwen2.5-32B 上对比 DAPO 约 2× 训练加速。GSPO 在序列级做单元匹配,CISPO 在 token 级保梯度完整 —— 是修复 GRPO clip 的两条互补路线。

```python
import torch
# 玩具:G=3 条回答,长度 6/5/4;新旧策略下的逐 token logprob
logp_new = torch.tensor([[-1.2,-0.8,-1.5,-0.4,-2.1,-1.0],
                         [-0.9,-1.3,-0.7,-1.8,-0.6, 0.0],
                         [-1.1,-0.5,-1.4,-0.9, 0.0, 0.0]])
logp_old = torch.tensor([[-1.3,-0.9,-1.4,-0.5,-2.0,-1.1],
                         [-1.0,-1.2,-0.8,-1.7,-0.7, 0.0],
                         [-1.0,-0.6,-1.3,-1.0, 0.0, 0.0]])
lengths = torch.tensor([6., 5., 4.])
mask = torch.arange(6)[None, :] < lengths[:, None].long()   # (G,T) 真实 token 掩码

log_ratio = logp_new - logp_old                             # 逐 token 对数比率
w_token = torch.exp(log_ratio)                              # GRPO:token 级比率 w_{i,t}
# GSPO:序列级比率 = 各 token 比率的长度归一化几何平均
mean_log_ratio = (log_ratio * mask.float()).sum(1) / lengths
s_seq = torch.exp(mean_log_ratio)                           # s_i = (π_θ/π_old)^(1/|y_i|)

eps = 0.2                      # GRPO token 级 clip
eps_l, eps_r = 3e-4, 4e-4      # GSPO 序列级 clip(非对称)
grpo_clip = (((w_token < 1-eps) | (w_token > 1+eps)) & mask).sum().item()
gspo_clip = ((s_seq < 1-eps_l) | (s_seq > 1+eps_r)).sum().item()

for i in range(3):
    r = mask[i]
    print(f"resp{i} len={int(lengths[i])}  token比率[{w_token[i][r].min():.3f},{w_token[i][r].max():.3f}]  s_i={s_seq[i]:.4f}")
print(f"GRPO 截断 {grpo_clip}/{int(mask.sum())} 个 token (eps={eps})")
print(f"GSPO 截断 {gspo_clip}/3 条序列 (eps_l={eps_l}, eps_r={eps_r})")
# 注:此处 s_i(~1.02–1.03)已超出 GSPO 的极小 eps → 实践中几乎每条序列都被 clip。
# 极小 eps 是有意施加的紧 proximal 约束,而非「GSPO 比 GRPO 截断更少」的证据。
```

## 6. RLVR — 可验证奖励 / RL from Verifiable Rewards
- 奖励来自**规则/验证器**(数学 exact-match、代码跑单测),而非学习的神经 RM。
- 利:几乎无「神经 RM 被 hack」(验证器 ≈ ground truth);弊:**只适用可验证域**。是 o1 / R1 式推理 RL 的奖励基座。

## 6.5 DeepSeek-R1 四阶段配方 / DeepSeek-R1 recipe
把上面的 GRPO + RLVR 串成一条完整产线。R1 不是「一把 RL 到底」,而是 **SFT 与 RL 交替** 四阶段:

| 阶段 | 名称 | 做什么 | 奖励 / 数据 |
|---|---|---|---|
| 1 | 冷启动 SFT | 用少量高质量 long-CoT 样本微调 base | 监督数据(修可读性 / 格式 / 语言混杂) |
| 2 | 推理 RL | GRPO + RLVR 在数学/代码上拉推理 | 规则奖励(答案 exact-match + 格式 + 语言一致性) |
| 3 | 拒绝采样 SFT | 用阶段 2 的策略大量采样、筛对的,再 SFT | 自蒸馏数据(论文约 80 万条:推理 + 通用混合) |
| 4 | 全场景 RL | 在全部 prompt 上再 RL,对齐通用偏好 | 可验证域用规则奖励 + 通用域用 helpful/harmless RM |

- **R1-Zero**:**纯 RL、无 SFT**(直接从 base 跑 GRPO + 规则奖励)。证明推理能力可由 RL **自发涌现**(自我反思 / 验算),但有**可读性差 / 中英混杂**问题 → 正是阶段 1 冷启动 SFT 的动机。
- **R1-Distill**:把 R1 产出的推理数据**只做 SFT**(不跑 RL)蒸馏进小稠密模型(Qwen / Llama 1.5B–70B)。论文结论:**蒸馏 > 在小模型上直接 RL**——小模型自身 RL 探索不动,不如直接吃大模型的推理轨迹。
- 过程奖励(PRM)与结果奖励(ORM)的取舍、Math-Shepherd 式 rollout 自动标注,见 [reward-modeling-eval §2](cheatsheet-reward-modeling-eval.html);R1 主线用的是**规则化 ORM**(RLVR),而非神经 PRM。

## 7. long-CoT 与测试时扩展 / long chain-of-thought & test-time scaling
- RLVR 在**长 CoT** 上训练 → 模型学会「想得更久」(更多推理 token),准确率随**测试时计算**上升(inference-time scaling)。
- 现象:自我反思 / 回溯 /「aha moment」自发涌现;评测从「单次准确率」转向「给定算力预算下的准确率」。

## 8. self-rewarding / self-play
- **Self-Rewarding LM**:模型当自己的 judge 产偏好数据、迭代偏好优化,减少人工标注依赖。
- **SPIN(self-play)**:用模型自己旧输出当「负样本」做对抗式微调。风险:自我偏好被放大。

---

## 分层面试题 / Stratified follow-ups

### L1 基础
- GRPO 相比 PPO 省了什么?「组内相对优势」具体怎么算?
- RLVR 的奖励从哪来?为什么能缓解 reward hacking?它的局限是什么?

### L2 进阶
- long-CoT RL 里「token-level vs sequence-level」loss 为什么有区别?(长回答梯度被稀释)
- GRPO 用 std 归一化优势会引入什么偏置?Dr.GRPO 怎么修?
- DAPO 的 clip-higher 解决什么问题?熵塌缩是什么、为什么坏?

### L3 深挖
- 推一遍:GRPO 的优势为何可视为「用组内均值近似 value baseline」?偏差/方差怎么权衡?
- 为什么 dynamic sampling(丢全对/全错组)能提效?和 curriculum / 难度采样什么关系?
- 测试时扩展对「能力是否训练即得」的范式意味着什么?推理 RL 与「纯 SFT 蒸馏长 CoT」的取舍?
- critic-free(GRPO/RLOO)在什么情况下反而不如带 value 的 PPO?


## 更多 L3 深挖 / Extended L3

<details>
<summary>Q: GRPO 保留 KL 惩罚 $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})$，但在 long-CoT 训练中模型需要探索远超 ref 分布的长推理路径。如何理解这个张力？去掉 KL 会怎样？</summary>

**A:** KL 惩罚的作用是**策略锚定**——防止 reward hacking 下策略飞掉(collapse 到某个 reward shortcut)。但在 long-CoT 场景下，模型要学到的正是 ref 模型**不会做**的长链反思行为，KL 本质上在惩罚"创新推理路径"。实践中的取舍：$\beta$ 过大→模型学不到长 CoT，推理能力上限被 ref 锁死；$\beta$ 过小→策略可能退化为 reward hacking（如重复 token 骗 verifier）。DAPO 原始配方实际上**去掉了 KL**，转而靠 clip-higher + dynamic sampling 来防崩；GRPO 保留 KL 但通常设较低值。本质上是在 **"不崩"** 和 **"能探索"** 之间走钢丝。
  **追问:** 如果既想去掉 KL 以放开探索、又想防止策略崩溃，除了 clip-higher 还有哪些可行的替代锚定机制？（如 EMA reference、正则化到 SFT checkpoint 等）

</details>

---

<details>
<summary>Q: 从方差缩减(variance reduction)角度，GRPO 的组内标准化 baseline 和 RLOO 的 leave-one-out baseline 理论上各有什么优劣？</summary>

**A:** 两种方法都是 REINFORCE 的 variance reduction 变体，区别在 baseline 构造：GRPO 用 $\mathrm{mean}(r)$ 并除以 $\mathrm{std}(r)$（即 z-score）；RLOO 对样本 $i$ 用 $\frac{1}{G-1}\sum_{j\neq i}r_j$。RLOO 的 leave-one-out baseline 无偏（因为 $r_i$ 不参与自己的 baseline），而 GRPO 的 $\mathrm{mean}(r)$ 包含 $r_i$ 自身，引入轻微自相关偏置（但在 $G$ 较大时可忽略）。然而 GRPO 的 **std 归一化**同时做了**方差缩放**，在奖励尺度不确定时更鲁棒，但也引入 Dr.GRPO 指出的题目难度偏置。RLOO 不做 std 归一化，奖励尺度敏感但更无偏。选择取决于任务奖励分布的稳定性。
  **追问:** 如果把 RLOO 的 leave-one-out baseline 和 GRPO 的 std 归一化结合起来，会怎样？这种混合有没有已知问题？

</details>

---

<details>
<summary>Q: DAPO 的 token-level loss 把序列梯度除以总 token 数 $T$，即 $L_{\text{token}}=\frac{1}{T}\sum_t \ell_t$。这是否会反过来引入对"短正确回答"的隐性偏好？</summary>

**A:** 会有一定影响，但方向比直觉复杂。Token-level loss 保证每个 token 贡献等权梯度，确实意味着**短正确回答的每个 token 拿到的梯度更大**（因为总梯度被 $T$ 个 token 均分，$T$ 小则单 token 梯度大）。但关键在于梯度的**符号**：正确回答的梯度是正向强化，错误回答是负向惩罚。因此 token-level loss 实际上让**短错误回答**受到的惩罚更集中、力度更大——对训练效率来说未必是坏事。真正的风险出现在**奖励是 0/1 二值**时：长正确回答和短正确回答获得相同的总奖励，但 token-level loss 下长回答的单 token 强化信号更弱，可能导致模型逐渐压缩正确推理路径的长度。
  **追问:** 如果用 DAPO token-level loss 同时搭配 outcome reward（只看最终答案对错），模型压缩推理链长度和压缩到正确答案之间如何竞争？

</details>

---

<details>
<summary>Q: 当前 RLVR 只能用在可验证域(math/code)，能否把过程奖励模型(PRM)当作"软验证器"融入 GRPO/DAPO 框架？技术上有哪些障碍？</summary>

**A:** 思路上可行：用 PRM 对 CoT 的每一步打分，将 step-level score 聚合成 sequence-level reward，再接入 GRPO 的组内优势计算。但障碍有三：①**PRM 本身的标注与训练**——需要 step-level 人工标注或自动标注(如 Monte Carlo rollout 估计)，成本高且准确率有限；②**奖励对齐问题**——PRM 给的是"这一步推理质量"的分数，和最终答案正确性可能不一致（步骤好但答案错 vs 步骤糙但答案对），导致 RL 信号矛盾；③**时序信用分配**——把 step-level score 聚合成 sequence reward 时，如何加权（均值？最终步？最差步？）直接影响学习动态。简单均值会模糊关键步骤的贡献，最终步又退化为 outcome reward。
  **追问:** 在 GRPO 框架内，能否对组内不同样本用不同的 reward aggregation 策略（自适应加权），而不是统一方案？

</details>

---

<details>
<summary>Q: 推理 RL 中常见 0/1 二值奖励（答案对=1，错=0），当 prompt 难度差异很大时，一组样本可能全对或全错。除了 dynamic sampling 丢弃这类 prompt，还有什么方法可以从"全错组"中提取有效训练信号？</summary>

**A:** 全错组的核心问题是组内所有奖励相同 → 优势恒为零 → 零梯度。几种思路：①**引入过程奖励**——即使最终答案都错，中间推理步骤的质量可能有差异，用 PRM 或推理长度/格式合规性等辅助信号制造组内差异；②**混合奖励设计**——在 outcome reward 上叠加格式奖励、推理完整性奖励等软信号，使"全错"组仍能区分好坏回答；③**跨 prompt baseline**——不局限于组内相对，而用 batch 级或 moving average 的全局 baseline 来给即使是全错组的样本提供梯度方向；④**难度分桶后重采样**——把全错 prompt 标记为"过难"，降采样频率但不完全丢弃，避免训练集偏向简单题。每种方法的代价不同：过程奖励需要额外标注/模型，跨 prompt baseline 可能引入高方差。
  **追问:** 跨 prompt baseline（例如用 EMA 全局均值作 baseline）在 GRPO 框架中具体如何实现？和组内 baseline 混合使用时权重如何调？

</details>

---

<details>
<summary>Q: GRPO 中组大小 $G$ 的选择对训练效果有什么理论和实践影响？$G=1$ 和 $G\to\infty$ 分别退化成什么？</summary>

**A:** $G=1$ 时只有一个样本，$\mathrm{mean}(r)=r_1$，优势恒为零——完全没有梯度，GRPO 完全失效。$G=2$ 退化为两两对比，本质上是 pairwise preference learning 的在线版。$G\to\infty$ 时组内均值和标准差收敛到**总体期望和标准差**，baseline 近似于全局 value function 的蒙特卡洛估计，此时 GRPO 在理论上接近带全局 baseline 的 REINFORCE。实践中的权衡：$G$ 太小→baseline 方差高、优势估计噪声大；$G$ 太大→每个 prompt 的采样成本高（推理开销线性增长），且多样性可能反而降低（重复采样中大量相似回答）。DeepSeek 的实践用中等 $G$ 值。此外，$G$ 和 clip range、learning rate 之间存在耦合——$G$ 大时优势估计更准，可以容忍更激进的更新步长。
  **追问:** 是否存在自适应 $G$ 的策略——对"难"prompt 采更多样本、"易"prompt 采更少？这和 dynamic sampling 如何协调？

</details>

---

<details>
<summary>Q: Self-Rewarding 范式中模型用自己的判断生成偏好数据并迭代训练。从在线学习理论角度，这种自我博弈在什么条件下会收敛，什么条件下会模式崩塌(mode collapse)？</summary>

**A:** 收敛的核心条件是**奖励信号必须持续提供有效的区分度**——即模型必须能区分自己的输出的好坏。当模型能力远低于任务难度时，自我判断噪声大但不会系统性偏移，训练可能缓慢但不崩。模式崩塌的典型触发条件：①**锚定效应放大**——模型偏好与自己风格接近的回答，正反馈循环导致输出多样性持续降低，最终坍缩到某个"自我偏好"的狭窄模式；②**判断饱和**——当模型输出质量趋同时，奖励信号退化为噪声，训练失去方向；③**reward hacking 自身 judge**——模型学会"说服"自己的 judge 而非真正改进，类似 Goodhart 定律作用于自身。缓解手段包括：保留 SFT anchor、定期引入外部验证信号、限制迭代轮数。
  **追问:** 如果在 self-rewarding 循环中引入一个固定的外部 verifier（如代码单测）作为"锚点"来校准自评分数，这能在多大程度上防止模式崩塌？

</details>

---

<details>
<summary>Q: DAPO 的 clip-higher 通过解耦上下 clip bound 来给低概率 token 上升空间。从信息几何(information geometry)角度，为什么 standard PPO 的对称 clip 在长 CoT 中会系统性地压制有价值的低概率推理 token？</summary>

**A:** Standard PPO 的对称 clip $[1-\epsilon, 1+\epsilon]$ 作用于 importance ratio $\rho=\pi_\theta/\pi_{\theta_{old}}$。关键洞察：在长 CoT 中，模型已经习得的高概率 token（如常见推理连接词）的 $\rho$ 接近 1，对称 clip 对它们影响小；但**新学到的低概率推理模式**（如回溯、自我纠正的特定 token 序列）初始概率极低，当正优势 $A>0$ 时 $\rho$ 从小于 1 往上走，上界 $1+\epsilon$ 很快就 clip 住了——这些 token 的提升空间被硬性封顶。与此同时，当负优势 $A<0$ 时，下界 $1-\epsilon$ 同样限制了高概率 token 的下降，但高概率 token 的空间本身就大，下降余量充足。因此对称 clip 在信息几何上造成了**不对称的学习动态**：高概率 token 的"下降通道"比低概率 token 的"上升通道"更宽。Clip-higher 通过抬高上界打破了这个不对称。
  **追问:** 除了解耦 clip bound，是否可以从 trust region 的角度出发（如用 KL 约束代替 hard clip），来更优雅地解决这个问题？这在长 CoT 场景下的计算代价如何？

</details>

## §A 核心论文时间线 / Key Papers Timeline

- **2017-07 · PPO** — Schulman et al., arXiv 预印本. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) — clipped surrogate objective + GAE 优势估计，奠定 LLM RL 的基线框架（actor + critic + ref + RM 四模型）。
- **2024-01 · Self-Rewarding LM** — Yuan et al., 预印本 (Meta). [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) — 模型自当 judge，用 LLM-as-a-Judge 产偏好数据迭代，减少人工标注；风险是自我偏好放大。
- **2024-01 · SPIN** — Chen et al., ICML 2024. [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) — Self-Play 微调，用模型旧输出做负样本对抗式优化。
- **2024-02 · GRPO / DeepSeekMath** — Shao et al., 预印本. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — 去掉 critic，用组内相对奖励（z-score 标准化）替代 value baseline，保留对 ref 的 KL 惩罚；DeepSeek 系核心算法。
- **2024-02 · RLOO** — Ahmadian et al., ACL 2024. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740) — critic-free，样本 i 的 baseline = 其余 G-1 个样本奖励均值（leave-one-out）；纯 REINFORCE、无 clip，RLHF 上与 PPO 竞争。
- **2025-01 · DeepSeek-R1 / RLVR** — Guo et al., Nature 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) — 用规则/验证器（数学 exact-match、代码单测）替代神经 RM，几乎消除 reward hacking；GRPO + 长 CoT RL 涌现自我反思，开启 inference-time scaling。
- **2025-03 · DAPO** — Yu et al., 预印本 (ByteDance Seed / 清华 AIR). [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) — 长 CoT RL 四项改动：Clip-Higher（防熵塌缩）、Dynamic Sampling、Token-level loss、Overlong reward shaping。
- **2025-03 · Dr.GRPO** — Liu et al., 预印本. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) — 修 GRPO 两处偏置（std 归一化、1/长度归一化），去掉后估计更无偏、token 更省、回答不虚长。
- **2025-06 · CISPO** — MiniMax team, 预印本 (MiniMax-M1 技术报告). [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) — 对标量 IS 权重本身做 clip 而非对概率比率做 clip，保留所有 token 的梯度信号；论文报告在 Qwen2.5-32B 上对比 DAPO 约 2× 训练加速。
- **2025-07 · GSPO** — Zheng et al., 预印本 (阿里 Qwen). [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) — 将 IS 校正粒度从 token 级提升到序列级（长度归一化几何平均），缓解 GRPO 在大规模 MoE 训练中的崩溃，并用于 Qwen3 训练。
