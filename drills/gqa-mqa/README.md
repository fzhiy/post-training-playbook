# GQA/MQA 从零实现学习练习

这是一个基于纯 PyTorch 从零实现的 Grouped-Query Attention (GQA) 和 Multi-Query Attention (MQA) 的学习练习。

## 1. 数学原理

核心是 **缩放点积注意力**，其输入为 Query $Q$, Key $K$, Value $V$，输出为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

在 **Grouped-Query Attention** 中，假设总查询头数为 $H$，KV头数为 $H_{kv}$，则每组查询头数 $G = H / H_{kv}$。投影过程为：

$$Q = XW_Q \in \mathbb{R}^{B \times H \times S \times d_{head}}, \quad K = XW_K \in \mathbb{R}^{B \times H_{kv} \times S \times d_{head}}, \quad V = XW_V \in \mathbb{R}^{B \times H_{kv} \times S \times d_{head}}$$

**GQA 核心操作**：将 $K$ 和 $V$ 沿头维度重复 $G$ 次，以匹配查询头数：

$$K' = \text{repeat\_interleave}(K, G, \text{dim}=1) \in \mathbb{R}^{B \times H \times S \times d_{head}}$$
$$V' = \text{repeat\_interleave}(V, G, \text{dim}=1) \in \mathbb{R}^{B \times H \times S \times d_{head}}$$

之后计算标准注意力（带因果掩码）：

$$\text{scores} = \frac{Q (K')^\top}{\sqrt{d_{head}}} \in \mathbb{R}^{B \times H \times S \times S}$$
$$\text{causal\_mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$
$$\text{weights} = \text{softmax}(\text{scores} + \text{causal\_mask})$$
$$\text{output} = \text{weights} \cdot V' \in \mathbb{R}^{B \times H \times S \times d_{head}}$$

最终投影回模型维度：

$$\text{Out} = \text{Concat}(\text{head}_1, ..., \text{head}_H) W_O$$

**特例**：
*   **Multi-Head Attention (MHA)**: $H_{kv} = H$，即 $G=1$。
*   **Multi-Query Attention (MQA)**: $H_{kv} = 1$，即 $G=H$。

## 2. 直觉与复杂度

**直觉**：
GQA 是 MHA 和 MQA 之间的折中方案。它通过让多个查询头共享一组 KV 头（即一个“组”），在保持较高模型表达能力的同时，显著减少了 KV 缓存的内存占用和计算量。MQA 是其极限情况（所有查询头共享同一组 KV）。

**计算复杂度**（以 FLOPs 计，忽略激活函数等）：
对于序列长度 $S$，头维度 $d_{head}$：
1.  **Q/K/V 投影**: $O(S \cdot d_{model} \cdot (H + 2H_{kv}) \cdot d_{head})$
2.  **注意力分数计算** $QK^\top$: $O(B \cdot H \cdot S^2 \cdot d_{head})$
3.  **注意力加权求和** $\text{weights} \cdot V'$: $O(B \cdot H \cdot S^2 \cdot d_{head})$
4.  **输出投影**: $O(S \cdot H \cdot d_{head} \cdot d_{model})$

主要优势在于推理时的 **KV 缓存** 大小从 MHA 的 $O(H \cdot S \cdot d_{head})$ 降低为 GQA 的 $O(H_{kv} \cdot S \cdot d_{head})$。

## 3. 文件说明

本练习目录包含 **EXACTLY** 三个文件：

*   `from_scratch.py`: GQA 模块的从零实现代码，包含一个简短的自我测试。
*   `test_gqa_mqa.py`: 对 GQA 和 MQA 实现的单元测试。
*   `README.md`: 本说明文件。

## 4. 运行命令

1.  **运行演示/自我测试**：
    ```bash
    python from_scratch.py
    ```
    这将实例化 GQA、MQA 和 MHA 模块，进行前向传播并检查输出形状和梯度流。

2.  **运行单元测试**：
    ```bash
    python test_gqa_mqa.py
    ```

## 5. 追问分层 / Stratified follow-ups

### L1 基础
1.  Grouped-Query Attention 与标准的 Multi-Head Attention 在架构上的主要区别是什么？
2.  什么是 Multi-Query Attention？它与 Grouped-Query Attention 有什么关系？
3.  为什么说 Grouped-Query Attention 能降低推理时的内存消耗？具体影响了哪一部分的内存？

### L2 中间
4.  在 GQA 的实现中，`n_heads` 和 `n_kv_heads` 参数需要满足什么约束条件？为什么？
5.  代码中如何将 KV 头的数量“扩展”以匹配查询头的数量？请描述具体操作（`repeat_interleave` 的作用）。
6.  除了降低内存，GQA 对模型训练的计算量有直接影响吗？与 MHA 相比是增加、减少还是基本不变？

### L3 深入
7.  在 GQA 中，一个组内的多个查询头共享同一组 KV 投影（$W_K$, $W_V$）。从表示学习的角度看，你认为这种共享可能会带来什么优势或潜在问题？
8.  如果将 GQA 应用于超长序列（例如上下文长度超过 $2^{16}$），除了 KV 缓存外，还有哪些性能或计算瓶颈可能会变得更加突出？
9.  在本实现的因果注意力掩码中，我们使用了一个固定的上三角布尔矩阵。在分布式训练或序列并行中，这个掩码可能需要如何调整？