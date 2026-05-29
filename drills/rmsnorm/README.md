# RMSNorm 从零实现学习练习 / From-Scratch RMSNorm Study Drill



> 从零手写 **RMSNorm (Root-Mean-Square Layer Normalization)**，与 PyTorch `nn.RMSNorm` 对比验证，加深对归一化层的理解。

---

## 1. 数学原理 / The Math

给定输入向量 $\mathbf{x} \in \mathbb{R}^d$，RMSNorm 的计算过程如下：

**Step 1 — 计算均方根 (Root Mean Square)**：

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

**Step 2 — 归一化并缩放 (Normalize & Scale)**：

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}$$

其中：
- $\boldsymbol{\gamma} \in \mathbb{R}^d$ 是可学习的增益向量 (learnable gain vector)，初始化为全 $\mathbf{1}$
- $\epsilon$ 是防止除零的极小常数 (代码中默认 $10^{-6}$)
- $\odot$ 表示逐元素乘法 (element-wise multiplication)

**与 LayerNorm 的对比**：LayerNorm 会先减均值 (re-centering) 再除以标准差；RMSNorm 省去了减均值这一步，也不含 bias 项，仅通过 RMS 缩放。计算更轻量，实践中效果相当。

---

## 2. 直觉与复杂度 / Intuition & Complexity

**直觉**：RMSNorm 的核心思想是——"不需要把数据居中，只要把向量的长度 (scale) 统一就行"。每个 token 在 hidden dimension 上的向量被缩放到相似的幅度，使后续层的权重能够更稳定地工作。可以把 $\boldsymbol{\gamma}$ 想象为归一化后给每个特征重新"调音量"的旋钮。

**复杂度**：
- **时间复杂度**：$O(d)$，对最后一维做一次逐元素平方、一次求和、一次开方、一次除法，加上逐元素乘 $\boldsymbol{\gamma}$
- **空间复杂度**：$O(1)$ 额外空间（`keepdim=True` 产生的中间张量形状为 $(B, S, 1)$，不随 $d$ 增长）
- 对比 LayerNorm，省去了均值计算与减法，理论 FLOPs 约减少 20–30%

---

## 3. 文件说明 / Files

本练习目录下**仅含**以下三个文件：

| 文件 | 说明 |
|---|---|
| `from_scratch.py` | RMSNorm 手写实现 (`RMSNorm` 类) 及简易自测脚本 |
| `test_rmsnorm.py` | 完整测试套件：形状、数值、与 `nn.RMSNorm` 对比、梯度等 |
| `README.md` | 本说明文档 |

---

## 4. 运行方式 / Run

```bash
# 运行演示与自测 / Demo & self-test
python from_scratch.py

# 运行完整测试 / Full test suite
python test_rmsnorm.py
```

---

## 5. 追问分层 / Stratified Follow-ups

### L1 — 基础 / Basic

1. **RMSNorm 和 LayerNorm 最核心的区别是什么？** 为什么去掉 mean subtraction 在实践中仍然有效？
2. **$\epsilon$ 的作用是什么？** 如果设为 0 会出现什么问题？
3. **$\boldsymbol{\gamma}$ 初始化为全 1 的意义是什么？** 这使得网络在训练初期的行为等价于什么操作？
4. 代码中 `mean(dim=-1, keepdim=True)` 的 `keepdim=True` 起什么作用？去掉会怎样？

### L2 — 进阶 / Intermediate

5. **RMSNorm 的梯度公式是什么？** 请手动推导 $\frac{\partial \text{RMSNorm}(\mathbf{x})_j}{\partial x_i}$，并解释其物理含义。
6. 为什么 RMSNorm 在大语言模型（如 LLaMA、Gemma）中几乎取代了 LayerNorm？从 **计算效率** 和 **训练稳定性** 两个角度分析。
7. 如果输入 $\mathbf{x}$ 的数值范围差异极大（某维度数值为 $10^6$ 级，另一维度为 $10^{-6}$ 级），RMSNorm 会如何处理？这与 z-score normalization 有何不同？
8. **能否把 RMSNorm 用在第一维 (batch dimension) 而非最后一维？** 为什么实际中从不这样做？

### L3 — 深入 / Deep

9. **RMSNorm 与 $\ell_2$ 归一化 (unit normalization) 的关系**：当 $\boldsymbol{\gamma} = \mathbf{1}$ 时，$\text{RMSNorm}(\mathbf{x})$ 的范数是多少？它和 $\frac{\mathbf{x}}{\|\mathbf{x}\|_2}$ 差一个什么常数因子？这个因子随 $d$ 如何变化？
10. **对 Softmax 的影响**：如果把 Transformer 中的 RMSNorm 换成 LayerNorm，在 attention score 的分布上会产生什么可观察的差异？请从理论上推导或给出直觉解释。
11. **RMSNorm 对梯度缩放的影响**：在深层 Transformer 中，RMSNorm 对反向传播中的梯度范数起到什么调节作用？能否从 Jacobian 矩阵的谱范数角度分析？
12. **并行化视角**：RMSNorm 的计算是否适合和 attention / FFN 做 **fused parallel**（如 Parallel Transformer）？为什么？在硬件层面（GPU warp-level），RMSNorm 的 reduction 操作是瓶颈吗？