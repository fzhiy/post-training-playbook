# SwiGLU Feed-Forward Block — 从零实现研究练习

本练习基于纯 PyTorch 从零开始实现 SwiGLU 前馈块。旨在深入理解其数学原理、内部结构与计算流程。

## 1. 数学定义 (Math)

SwiGLU 前馈块将输入 `x` 通过门控线性单元（GLU）与 Swish 激活函数相结合。其计算过程如下：

设输入张量 $x$ 的形状为 $(B, T, d_{model})$。

1.  **门控路径 (Gate Path)**:
    $$gate = \text{Swish}(x @ W_1)$$
    其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，结果 $gate$ 的形状为 $(B, T, d_{ff})$。

2.  **值路径 (Value Path)**:
    $$value = x @ W_3$$
    其中 $W_3 \in \mathbb{R}^{d_{model} \times d_{ff}}$，结果 $value$ 的形状为 $(B, T, d_{ff})$。

3.  **门控组合 (Gated Combination)**:
    $$gated = gate \odot value$$
    其中 $\odot$ 表示逐元素乘法（Hadamard积），$gated$ 的形状保持 $(B, T, d_{ff})$。

4.  **输出投影 (Output Projection)**:
    $$out = (gated @ W_2) + b_{2}$$
    其中 $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$，最终输出 $out$ 的形状恢复为 $(B, T, d_{model})$。

**Swish 激活函数**的数学定义为：
$$\text{Swish}(z) = z \cdot \sigma(z)$$
其中 $\sigma$ 是 Sigmoid 函数。

## 2. 直觉与复杂度 (Intuition & Complexity)

**核心思想**：SwiGLU 不是简单地对输入进行非线性变换，而是通过一个“门” (`gate`) 来控制另一个“值” (`value`) 信号的通过量。这种门控机制允许网络更灵活地学习复杂的函数映射。

- **与标准 FFN 的区别**：标准 FFN 使用 `ReLU(x @ W1) @ W2`，包含**两个**权重矩阵。SwiGLU 因其门控设计，需要**三个**权重矩阵 (`W1`, `W3`, `W2`)，参数量更大，但通常能带来更好的性能。
- **维度约定**：本实现遵循 LLaMA 的实践，将隐藏层维度 $d_{ff}$ 默认设为 $d_{model}$ 的 $8/3$ 倍，并向上取整到 256 的倍数（`d_ff = ((8/3 * d_model + 255) // 256) * 256`），以优化硬件计算效率。
- **计算复杂度**：给定序列长度 $T$、模型维度 $d_{model}$ 和中间维度 $d_{ff}$，前向传播的复杂度主要来源于三次矩阵乘法，为 $O(T \cdot d_{model} \cdot d_{ff})$。

## 3. 文件 (Files)

本练习目录**仅包含**以下三个文件：
- `from_scratch.py`：SwiGLU 前馈块的完整 PyTorch 实现及快速自检脚本。
- `test_swiglu_ffn.py`：针对实现的单元测试文件。
- `README.md`：本说明文档。

## 4. 运行 (Run)

1.  **查看演示与自检**：运行主脚本，它会实例化一个 SwiGLU 块，执行一次前向传播，并验证输出形状与 Swish 函数的正确性。
    ```bash
    python from_scratch.py
    ```

2.  **运行测试**：执行测试文件，对实现进行更全面的正确性验证。
    ```bash
    python test_swiglu_ffn.py
    ```

## 5. 追问分层 / Stratified follow-ups

**L1 基础 (Basic)**
1.  在本实现的 SwiGLU 块中，哪三个线性层分别对应数学公式中的 $W_1$, $W_3$, 和 $W_2$？
2.  代码中定义的 `Swish` 函数，其数学表达式是什么？

**L2 中等 (Intermediate)**
3.  为什么说 SwiGLU 的参数量比一个标准两层的 FFN（如使用 ReLU 激活）更多？请大致计算两者的参数量对比。
4.  代码中 `d_ff` 的默认计算公式 `int(8/3 * d_model)` 是怎么来的？为什么要将其对齐到 256 的倍数？

**L3 深入 (Deep)**
5.  SwiGLU 结合了 Gated Linear Unit (GLU) 和 Swish 激活。请解释，为什么这种“门控”机制可能比直接使用 Swish 或 ReLU 等单一激活函数的 FFN 更强大？
6.  在 `forward` 方法中，`gate` 和 `value` 分别经过 `W1` 和 `W3` 投影后，形状相同。它们在数学和功能上是对称的吗？为什么？
7.  从梯度反向传播的角度，Swish 函数 $z \cdot \sigma(z)$ 相对于传统的 ReLU 有什么潜在优势或劣势？