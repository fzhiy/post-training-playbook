# Softmax + Cross-Entropy + Label Smoothing 从零实现学习笔记

一个用于学习数值稳定实现的 PyTorch 练习，包含 softmax、带 label smoothing 的交叉熵损失函数。

## 1. 数学原理

**Softmax 函数**:
给定 logits 向量 $\mathbf{z} = [z_1, z_2, \dots, z_K]$, softmax 输出概率分布 $\mathbf{p}$:
$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$
为了数值稳定性，计算时会减去最大值 $m = \max(\mathbf{z})$:
$$
p_i = \frac{e^{z_i - m}}{\sum_{j=1}^K e^{z_j - m}}
$$

**Label Smoothing**:
给定真实类别 one-hot 向量 $\mathbf{y}$ (仅第 $t$ 维为1) 和 smoothing 因子 $\epsilon$，平滑后的分布 $\mathbf{y}'$ 为:
$$
\mathbf{y}' = (1 - \epsilon)\mathbf{y} + \frac{\epsilon}{K} \mathbf{1}
$$

**交叉熵损失**:
对于单个样本，计算预测分布 $\mathbf{p}$ 与目标分布 $\mathbf{y}'$ 的交叉熵 $H(\mathbf{y}', \mathbf{p})$:
$$
H(\mathbf{y}', \mathbf{p}) = -\sum_{i=1}^K y'_i \log p_i
$$
最终，对一个 batch 的 $N$ 个样本取平均得到总损失 $\mathcal{L}$:
$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N H(\mathbf{y}'_n, \mathbf{p}_n)
$$

## 2. 直觉与复杂度

*   **数值稳定性**: 核心技巧是在 `exp` 前减去 `logits` 的最大值，防止浮点数上溢。
*   **Label Smoothing**: 一种正则化技术，防止模型对预测过于自信。它将真实标签的 one-hot 分布（尖锐）变为更平滑的分布，鼓励模型输出更低的熵（即更不确定）。
*   **时间复杂度**: 两个函数的核心操作（`max`, `sum`, `exp`）都在最后一个维度（类别维度）进行。对于形状为 `(N, K)` 的输入，单次操作的时间复杂度为 $O(N \times K)$。
*   **空间复杂度**: 需要存储与输入同形状的中间结果（如 `shifted`, `exp_shifted`），空间复杂度为 $O(N \times K)$。

## 3. 文件

本学习目录仅包含以下三个文件：

*   `from_scratch.py`: 核心实现，包含 `stable_softmax` 和 `label_smoothing_cross_entropy` 两个函数。
*   `test_cross_entropy.py`: 单元测试文件，用于验证实现的正确性和数值稳定性。
*   `README.md`: 本说明文档。

## 4. 运行

执行内置的自检演示：
```bash
python from_scratch.py
```
运行完整的单元测试套件：
```bash
python test_cross_entropy.py
```

## 5. 追问分层 / Stratified follow-ups

**L1 基础**
1.  `stable_softmax` 函数中减去 `max_logits` 的作用是什么？
2.  `label_smoothing_cross_entropy` 函数中，`smoothed` 变量是如何从 `one_hot` 和 `epsilon` 计算得到的？
3.  最终损失值是如何从 `loss_per_sample` 计算得出的？

**L2 中级**
1.  如果 `epsilon = 0`，`label_smoothing_cross_entropy` 的计算结果会等价于标准的交叉熵损失吗？为什么？
2.  代码中计算 `log_softmax` 的方式 (`shifted - log_sum_exp`) 与直接使用 `torch.log(torch.softmax(logits, dim=-1))` 相比，为何在数值上更稳定？
3.  `scatter_` 函数在此处的作用是什么？它完成了一项什么关键的数据转换？

**L3 深度**
1.  假设某个样本的 `logits` 值非常大（例如 `[1000, 1000, 0]`），请追踪代码执行过程，解释为什么 `stable_softmax` 和 `label_smoothing_cross_entropy` 能够避免产生 `NaN` 或 `Inf`。
2.  从梯度的角度，分析 label smoothing ($\epsilon > 0$) 如何影响模型对正确类别的参数更新。
3.  此实现与 PyTorch 内置的 `torch.nn.CrossEntropyLoss` 在功能和计算路径上有何主要异同？（提示：考虑内置函数是否集成了 softmax、是否支持 label smoothing 参数）。