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
- 目标同 PPO 的 clipped surrogate,但优势用 $A_i$、**无 critic、无 GAE**;保留对 ref 的 KL。
- 收益:省一个 value 模型且不用学 value;对**可验证奖励**(数学/代码)特别稳。DeepSeek 系用它。

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

## 6. RLVR — 可验证奖励 / RL from Verifiable Rewards
- 奖励来自**规则/验证器**(数学 exact-match、代码跑单测),而非学习的神经 RM。
- 利:几乎无「神经 RM 被 hack」(验证器 ≈ ground truth);弊:**只适用可验证域**。是 o1 / R1 式推理 RL 的奖励基座。

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
