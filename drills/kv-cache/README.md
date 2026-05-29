# From-Scratch Autoregressive Decoding with KV Cache

纯 PyTorch 手写 Transformer 自回归解码 + KV Cache 实现，无任何外部推理框架依赖。

---

## 1. 数学原理 / Math

**Scaled Dot-Product Causal Attention（带因果遮罩的缩放点积注意力）：**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

其中因果遮罩 $M$ 为上三角（对角线除外）：

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

**KV Cache 核心操作：** Prefill 阶段处理全部 prompt 建立缓存，Decode 阶段每步仅输入一个 token，将新投影的 $K_{\text{cur}}, V_{\text{cur}}$ 与缓存拼接：

$$K_{\text{new}} = \text{concat}(K_{\text{cache}},\; K_{\text{cur}}), \quad V_{\text{new}} = \text{concat}(V_{\text{cache}},\; V_{\text{cur}})$$

注意分数维度为 $(B, H, T_{\text{cur}}, T_{\text{kv}})$，其中 $T_{\text{kv}} = T_{\text{cache}} + T_{\text{cur}}$。因果遮罩取全长矩阵的最后 $T_{\text{cur}}$ 行：

$$\hat{M} = M[T_{\text{kv}} - T_{\text{cur}} : T_{\text{kv}},\; :]$$

**Pre-Norm Decoder Block：**

$$x' = x + \text{CausalMHA}(\text{LN}(x)), \qquad \hat{x} = x' + \text{FFN}(\text{LN}(x'))$$

**FFN：** $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$，其中 $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}},\; W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$。

**Position Encoding：** 可学习的绝对位置嵌入，解码时通过 `position_offset` 保持位置索引连续：

$$x = \text{TokEmb}(t) + \text{PosEmb}(\text{offset} + t)$$

**采样策略：** temperature scaling + top-k filtering

$$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}, \qquad z'_i = \begin{cases} z_i / \tau & \text{if } z_i / \tau \in \text{top-k} \\ -\infty & \text{otherwise} \end{cases}$$

当 $\tau = 0$ 时退化为 greedy（argmax）。

---

## 2. 直觉与复杂度 / Intuition & Complexity

**无 Cache vs 有 Cache 解码对比：**

| | 无 Cache | 有 KV Cache |
|---|---|---|
| 第 $t$ 步 Attention 计算 | $O(t \cdot d)$ | $O(1 \cdot d)$（仅新 token 作 query） |
| 生成 $n$ token 总量 | $O(n^2 \cdot d)$ | $O(n \cdot d)$（缓存线性增长） |

**直觉：** KV Cache 本质是用 **内存换计算**——将之前所有 step 的 Key/Value 向量缓存起来，避免重复计算。Prefill 一次处理全部 prompt 填充缓存，之后每步 decode 只需处理一个 token，Attention 的 query 维度恒为 1。

**因果遮罩的精妙之处：** 当 decode 阶段 $T_{\text{cur}} = 1$ 时，遮罩从 $T_{\text{kv}} \times T_{\text{kv}}$ 矩阵中取最后 1 行，确保新 token 只能看到它自己及之前所有位置。

---

## 3. Files

| 文件 | 说明 |
|---|---|
| `from_scratch.py` | 核心实现：`CausalMultiHeadAttention`、`TransformerDecoderBlock`、`MiniGPT`、`generate` 解码循环、`_sample` 采样函数，以及 `__main__` 自测试 |
| `test_kv_cache.py` | 单元测试：验证 KV Cache 的正确性（缓存拼接、形状、与无缓存结果的一致性等） |
| `README.md` | 本文件 |

---

## 4. Run

```bash
# 运行演示 / 自测试（随机权重 + 贪心/采样生成）
python from_scratch.py

# 运行单元测试
python test_kv_cache.py
```

---

## 5. 追问分层 / Stratified Follow-ups

### L1 — 基础 / Basic

1. Prefill 阶段和 Decode 阶段分别输入模型的 token 数是多少？为什么要区分这两个阶段？
2. 因果遮罩（causal mask）的作用是什么？如果没有它会怎样？
3. `temperature` 参数如何影响生成的多样性？`temperature=0` 意味着什么？

### L2 — 中级 / Intermediate

4. 为什么 KV Cache 只缓存 Key 和 Value 而不缓存 Query？从计算图角度解释。
5. 代码中 `position_offset` 的作用是什么？如果去掉它会导致什么问题？
6. 解码阶段因果遮罩的切片 `causal[T_kv - T_cur : T_kv, :]` 为什么不能直接用完整的 `T_kv × T_kv` 矩阵？
7. Pre-norm（先 LayerNorm 再 Attention/FFN）相比 Post-norm 有什么训练稳定性上的优势？

### L3 — 深度 / Deep

8. 如果将 `cached_k` 的 `torch.cat` 操作替换为预分配 buffer + 原地写入，具体会减少哪些开销？在什么场景下收益最大？
9. 本实现中 KV Cache 的内存占用为 $O(L \cdot n_{\text{heads}} \cdot T \cdot d_k)$，当序列长度 $T$ 极大时有哪些经典的压缩策略（如 GQA、MQA、Sliding Window）？它们各自牺牲了什么？
10. 当前采样使用的是独立的 top-k + temperature；如果引入 nucleus sampling（top-p），概率分布的截断逻辑有何本质区别？在什么分布特征下 top-p 优于 top-k？