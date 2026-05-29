# PPO Clipped Surrogate Objective 学习演练

本仓库通过一个独立的实现文件，从零开始研究 PPO（Proximal Policy Optimization）的裁剪代理目标函数。这是一个聚焦于理解 PPO 核心数学与实现细节的专项练习。

---

## 1. 数学基础 / Mathematical Foundation

PPO 的裁剪代理目标旨在限制策略更新幅度，其核心公式如下。

**概率比 $r_t$**:
$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $$
代码实现: `ratio = torch.exp(action_logprobs - old_action_logprobs)`

**裁剪代理目标 $L^{CLIP}$**:
$$ L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \ \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right] $$
其中 $\hat{A}_t$ 为优势函数估计，$\epsilon$ 为裁剪超参数。

**代码实现对应**:
```python
surrogate_unclipped = ratio * advantages
ratio_clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
surrogate_clipped = ratio_clipped * advantages
surrogate_per_sample = torch.min(surrogate_unclipped, surrogate_clipped)
loss = -surrogate_per_sample.mean()  # 因为优化器最小化损失，故取负
```

**诊断指标**:
- **近似 KL 散度**: $\approx \mathbb{E}[r - 1 - \log r]$ (Schulman blog)
- **裁剪比例**: $|\ r - 1\ | > \epsilon$ 的样本比例

---

## 2. 直觉与复杂度 / Intuition & Complexity

**直觉**:
- PPO 通过 `min` 操作选择一个保守的目标，防止策略在每次更新中发生过大变化。
- 当优势 $A > 0$（好的动作），$r$ 会被限制不超过 $1+\epsilon$，避免过度提升该动作概率。
- 当优势 $A < 0$（坏的动作），$r$ 会被限制不低于 $1-\epsilon$，避免过度降低该动作概率。
- 这种对称约束使得策略更新保持在“信任域”内。

**实现复杂度**:
- **时间复杂度**: $O(B)$，其中 $B$ 为 batch size。所有操作均为逐元素。
- **空间复杂度**: $O(B)$，存储中间张量 `ratio`, `surrogate_unclipped` 等。
- **数值稳定性**: 代码使用 `log_ratio` 避免直接计算概率比，防止数值溢出。

---

## 3. 文件 / Files

- `from_scratch.py`: 包含 PPO 裁剪代理目标和 GAE 的从零实现，以及一个简单的演示。
- `test_ppo_clip.py`: 包含针对裁剪代理目标函数的单元测试。
- `README.md`: 本说明文件。

---

## 4. 运行 / Run

运行演示与自检：
```bash
python from_scratch.py
```

运行测试套件：
```bash
python test_ppo_clip.py
```

---

## 5. 追问分层 / Stratified Follow-ups

### L1 基础
1. 为什么 PPO 要对策略更新进行裁剪（clipping）？如果不裁剪，可能会发生什么？
2. 解释 `ratio` 为什么在代码中通过 `exp(log_ratio)` 计算，而不是直接计算概率的比值。
3. 在 `min` 操作中，为什么要对 `surrogate_unclipped` 和 `surrogate_clipped` 取较小值？

### L2 中级
1. 代码中 `approx_kl` 的计算公式 `((ratio - 1) - log_ratio).mean()` 的来源是什么？它近似了什么量？
2. 解释超参数 `clip_epsilon` 的作用。如果将其设为 0 或非常大（如 100），优化行为会如何变化？
3. GAE（Generalized Advantage Estimation）函数中的 `lam`（$\lambda$）参数有什么作用？它如何权衡偏差（bias）与方差（variance）？

### L3 深入
1. 讨论 PPO 裁剪目标与 TRPO（Trust Region Policy Optimization）中 KL 散度约束在数学形式和实际行为上的异同。
2. 在反向传播时，梯度如何通过 `min` 操作流回 `cur_logprobs`？从自动微分的角度分析其行为。
3. 代码中 `advantages` 在演示部分进行了白化（whitening）。为什么需要对优势函数进行归一化？如果不这样做，对训练稳定性可能产生什么影响？