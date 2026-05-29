# SimPO 偏好优化损失 — 从零实现学习演练

**SimPO: Simple Preference Optimization with a Reference-Free Reward**
Meng et al., arXiv:2405.14734, NeurIPS 2024.

---

## 1. 数学原理 / Mathematical Formulation

### 隐式奖励 / Implicit Reward

SimPO 用响应的**长度归一化平均 log-prob** 作为隐式奖励，完全不需要参考模型：

$$
r(y \mid x) \;=\; \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid y_{<t},\, x)
\;=\; \frac{\beta}{|y|} \log \pi_\theta(y \mid x)
$$

其中 $|y|$ 是响应的生成 token 数，$\beta$ 是温度缩放系数。

### SimPO 损失 / Loss (per sample)

给定偏好对 $(x, y_w, y_l)$（$y_w$ 为优选，$y_l$ 为拒绝）：

$$
\mathcal{L}_{\text{SimPO}} \;=\; -\log\sigma\!\left(
  \frac{\beta}{|y_w|}\log\pi_\theta(y_w \mid x)
  \;-\;
  \frac{\beta}{|y_l|}\log\pi_\theta(y_l \mid x)
  \;-\; \gamma
\right)
$$

- $\gamma > 0$：**目标 margin**，要求两个奖励之差至少达到 $\gamma$，而非"稍微好一点点"就满足。
- $\beta$：论文默认约 2.0–2.5（比 DPO 的 0.1 大，因为没有 KL 约束项归一化）。

### 逐序列 log-prob 辅助函数 / Per-Sequence Log-Prob

$$
\log \pi_\theta(y \mid x) = \sum_{t} \mathbf{1}[y_{t+1} \neq \texttt{-100}] \cdot \log \pi_\theta(y_{t+1} \mid y_{\le t},\, x)
$$

标准 next-token 预测：logits 左移一位，labels 右移一位，mask 掉 `ignore_index=-100` 的 padding 位置，再对 token 维度求和。

---

## 2. SimPO 与 DPO 的关键差异 / Key Differences from DPO

| 维度 | DPO | SimPO |
|---|---|---|
| **参考模型** | 必须保持 π_ref 在线（前向传播，通常冻结） | **不需要**，节省一次前向 + 显存 |
| **奖励定义** | β · (log π_θ − log π_ref)，序列总 log-prob 之差 | β/\|y\| · log π_θ，**长度归一化**的平均 log-prob |
| **长度偏好** | 偏向长序列（总 log-prob 越长越低，负数更大） | 中性：每 token 平均奖励，不因长度而偏置 |
| **Margin** | 无显式 margin，只需"chosen > rejected"即满足 | 硬 margin γ：chosen 必须比 rejected 至少高 γ |
| **超参数默认值** | β ≈ 0.1 | β ≈ 2.0–2.5，γ ≈ 0.5–1.0 |
| **训练复杂度** | 需要双倍前向（policy + reference）| 单次前向（仅 policy） |

**直觉 / Intuition**

DPO 中，一个长但质量平均的回答和一个短而精炼的回答，前者的总 log-prob 绝对值更小（负数），造成奖励虚高 —— 这是 DPO 已知的**长度偏差（length bias）**问题。SimPO 用 $1/|y|$ 归一化后，奖励变成每 token 的平均 log-prob，长短响应在同一尺度下比较。

---

## 3. 文件清单 / Files

| 文件 | 说明 |
|---|---|
| `from_scratch.py` | 核心实现：`sequence_logp`（辅助）+ `simpo_loss`，以及 `__main__` 自测入口 |
| `test_simpo_loss.py` | 单元测试：形状、数值正确性、γ 单调性、长度归一化退化、梯度方向 |
| `README.md` | 本文件 |

---

## 4. 运行 / Run

```bash
# 演示与自测 / Demo & smoke-test
python from_scratch.py

# 运行单元测试 / Run tests
python test_simpo_loss.py
```

---

## 5. 追问分层 / Stratified Follow-ups

### L1 — 基础 / Basic

1. **为什么 SimPO 不需要参考模型？** 它用什么代替了 DPO 中的 log π_ref 项？去掉参考模型的训练代价有什么变化？
2. **长度归一化的作用：** 如果不做归一化（直接用总 log-prob），模型会倾向生成更长还是更短的响应？为什么？
3. **margin γ 的直觉：** γ=0 和 γ=1 在优化行为上有何不同？如果 γ 设得过大会发生什么？
4. **β 为什么比 DPO 大这么多？** DPO 的 β≈0.1，SimPO 的 β≈2.0；这背后的原因是什么（提示：缺少了哪个归一化项）？

### L2 — 进阶 / Intermediate

5. **与 DPO 的等价条件：** 如果 |y_w| == |y_l| 且 γ=0，SimPO 损失和 DPO 损失在什么条件下形式相似（注意：SimPO 仍然没有 π_ref）？两者能完全等价吗？
6. **梯度方向验证：** 对 logps_chosen 求 ∂L/∂logps_chosen，推导其符号，并解释为什么它必须为负才能使模型"提升优选响应的质量"。
7. **长度定义的边界情况：** 如果某个响应的有效长度 |y|=0（全是 padding），会发生什么数值问题？在真实训练代码中应如何防护？
8. **γ 的调参策略：** 假设你在 LLaMA-3-8B 上微调，验证集上发现 chosen_reward > rejected_reward 但 margin < γ 的比例很高，应该如何调整超参数？

### L3 — 深入 / Deep

9. **无参考模型的理论代价：** DPO 推导依赖 KL(π_θ || π_ref) 惩罚来防止策略退化。SimPO 没有这个约束，论文中用什么机制部分替代了它的作用？这在长训练周期下有什么风险？
10. **与 IPO / KTO 的比较：** SimPO 解决的是 DPO 的长度偏差问题，IPO 解决的是过拟合到 Bradley-Terry 假设的问题，KTO 解决的是配对偏好数据稀缺的问题。如果你的数据集同时存在长度偏差和标注噪声，你会如何设计损失函数？
11. **从在线到离线的差距：** SimPO 是离线（offline）方法，使用静态偏好数据集。论文中的 on-policy 实验（用 SimPO 策略采样再标注）和离线实验性能差距有多大？这说明了 SimPO 的哪个局限性？
