# 从头实现温度 / top-k / top-p (nucleus) 采样学习演练 README

本演练基于纯 PyTorch 从头实现温度缩放、top-k 和 top-p（nucleus）采样，用于从 logits 中采样 token。目录中仅包含三个文件，无其他依赖。

## 数学基础 / Mathematical Foundations

所有操作基于模型输出的 logits 向量 $z \in \mathbb{R}^{V}$，其中 $V$ 为词表大小。采样过程如下：

1. **温度缩放 / Temperature Scaling**  
   给定温度 $\tau > 0$，缩放后的 logits 为 $z_i / \tau$，然后通过 softmax 转换为概率：  
   $$
   p_i = \frac{e^{z_i / \tau}}{\sum_{j=1}^V e^{z_j / \tau}}
   $$  
   - $\tau \to 0$：分布趋近贪心解码（argmax）。  
   - $\tau \to \infty$：分布趋近均匀分布。  
   - $\tau = 1.0$：无变化。

2. **Top-k 过滤 / Top-k Filtering**  
   对每个样本，保留 logits 中最大的 $k$ 个值，其余设为 $-\infty$：  
   $$
   \tilde{z}_i = 
   \begin{cases} 
   z_i, & \text{if } z_i \in \text{top-}k \text{ values} \\
   -\infty, & \text{otherwise}
   \end{cases}
   $$  
   等价于 mask 掉非 top-k 的 token。

3. **Top-p（Nucleus）过滤 / Top-p Filtering**  
   先将 logits 转换为概率 $p_i = \text{softmax}(z_i)$，并按降序排列得到 $p_{(1)} \geq p_{(2)} \geq \dots \geq p_{(V)}$。计算累积概率：  
   $$
   \text{cumulative}_m = \sum_{j=1}^m p_{(j)}
   $$  
   找到最小的 $m$ 使得 $\text{cumulative}_m \geq p$（$p$ 为 top_p 阈值），则保留前 $m$ 个 token（至少保留 `min_tokens_to_keep` 个），其余设为 $-\infty$。在代码中，通过移位累积和实现 mask。

4. **采样 / Sampling**  
   应用上述过滤后，对 filtered logits 计算 softmax 概率，并从多项分布中采样一个 token：  
   $$
   \text{next\_token} \sim \text{Multinomial}(\text{softmax}(\tilde{z}))
   $$  
   处理顺序为：温度缩放 → top-k → top-p → 多项采样。

## 直觉与复杂度 / Intuition and Complexity

- **直觉 / Intuition**：  
  - **温度**控制随机性：低温使输出更确定（偏向高概率 token），高温增加多样性。  
  - **Top-k** 直接限制选择范围到最可能的 $k$ 个 token，简单但固定。  
  - **Top-p (nucleus)** 动态调整候选集，基于概率质量累积，确保多样性同时保持连贯性。

- **复杂度 / Complexity**：  
  - 温度缩放：$O(1)$ 逐元素操作，向量化高效。  
  - Top-k：使用 `torch.topk`，时间复杂度约为 $O(V \log k)$，但实践中常为线性。  
  - Top-p：需排序和累积和，时间复杂度 $O(V \log V)$，但批量处理时被并行化。  
  整体，对于批量大小 $B$ 和词表大小 $V$，主要操作为 $O(BV)$ 或 $O(BV \log V)$，适合 GPU 加速。

## 文件 / Files

演练目录包含 EXACTLY 以下三个文件：
- `from_scratch.py`：核心实现，包含 `apply_temperature`、`apply_top_k`、`apply_top_p` 和 `sample` 函数。
- `test_sampling.py`：测试文件，验证采样功能的正确性。
- `README.md`：本说明文件。

## 运行 / Run

仅支持以下两个命令：
- **演示 / 自测试 / Demo / Self-test**：运行 `python from_scratch.py`，执行快速 smoke test 并输出示例结果。
- **测试 / Tests**：运行 `python test_sampling.py`，运行单元测试以确保功能正确。

## 追问分层 / Stratified follow-ups

### L1 基础 / Basic
1. 温度缩放（temperature scaling）如何影响概率分布？当温度接近 0 时会发生什么？  
2. Top-k 采样中，参数 $k$ 的作用是什么？为什么需要限制 $k \leq \text{vocab\_size}$？  
3. 在 top-p 采样中，什么是“nucleus”？如何通过阈值 $p$ 控制候选 token 数量？

### L2 中级 / Intermediate
1. 比较 top-k 和 top-p 采样：在什么场景下 top-p 可能比 top-k 更优？  
2. 解释为什么在采样管道中，温度缩放通常先于 top-k 和 top-p 应用？  
3. 如何调整温度、top-k 和 top-p 参数来平衡文本生成的多样性和连贯性？

### L3 深入 / Deep
1. 分析温度缩放与 softmax 的数学关系：为什么缩放 logits 等价于调整分布熵？  
2. 在 top-p 过滤中，移位累积和（shifted cumulative sum）的算法设计如何确保正确 mask？讨论边界条件。  
3. 从信息论角度，讨论温度、top-k 和 top-p 如何影响生成文本的多样性和困惑度（perplexity）。在实际应用中，如何联合调参以优化模型性能？