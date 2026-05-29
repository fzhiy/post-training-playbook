# Drill: Sequence Packing from scratch

> 可运行的 from-scratch 实现 + 测试。目标:每一行都能在面试里推导和辩护。
> Runnable from-scratch implementation with tests — derive and defend every line.

## 背景 / Background

大模型训练时一个 batch 通常含多条不等长文本。**Padding** 的做法是把所有序列补齐到最长者,浪费 $O(L_{max} - L_i)$ 计算;更严重的是在同一个 padded batch 里做 self-attention 时,**普通 padding mask 无法阻止不同文档之间的 attention 泄漏**——需要额外 per-sample causal mask 或牺牲批次效率。

**Sequence Packing** 把多条序列直接拼接成一条长序列,用 `cu_seqlens`(cumulative sequence lengths)记录边界,然后通过 block-diagonal attention mask(或 flash-attn varlen 内核)保证跨文档 attention 权重严格为 0。零填充、零泄漏。

## 为什么不能用普通 padding mask 替代 cu_seqlens？

| 方面 | Padding mask | cu_seqlens + block-diagonal |
|---|---|---|
| 空间 | $(B, L_{max})$ 标记哪些是 pad | $(N+1,)$ 整数,记文档边界 |
| 计算浪费 | $O(L_{max} - L_i)$ 个无效 token 仍参与 QK 矩阵乘法 | $T = \sum L_i$,零浪费 |
| 跨文档隔离 | 无法阻止同 batch 内不同文档间的 attention(padding mask 只标记 pad 位,不标记文档边界) | block-diagonal mask 把跨文档位置填 $-\infty$,softmax 后严格为 0 |
| 内存 | 需要 $(B, L_{max}, L_{max})$ 的注意力矩阵 | 同理 $(T, T)$,但 $T \ll B \cdot L_{max}$(当序列长短不均时) |
| 生产实践 | 适合序列等长 / 小 batch | flash-attn `varlen` 接口直接消费 `cu_seqlens`,O(T) 索引避免构造完整 mask |

一句话:**padding mask 标记的是"哪里是 pad",而 cu_seqlens 标记的是"哪里是文档边界"——这是两个不同的概念**,前者无法推导后者。

## 数学 / The math

记 $N$ 条文档长度为 $l_1, \dots, l_N$,总长 $T = \sum_{i=1}^{N} l_i$。

**cu_seqlens:**
$$\text{cu}[0] = 0,\quad \text{cu}[i] = \sum_{j=1}^{i} l_j$$

**Block-diagonal mask:**
$$M[s, t] = \mathbf{1}[\text{doc}(s) = \text{doc}(t)] \;\wedge\; (\text{causal} \Rightarrow s \geq t)$$

其中 $\text{doc}(s) = i$ 当且仅当 $\text{cu}[i] \leq s < \text{cu}[i+1]$。

**Attention over packed sequence:**
$$\mathrm{Attn}(Q, K, V)_s = \frac{\sum_{t: M[s,t]} \exp\!\left(\frac{q_s^\top k_t}{\sqrt{d_k}}\right) v_t}{\sum_{t: M[s,t]} \exp\!\left(\frac{q_s^\top k_t}{\sqrt{d_k}}\right)}$$

跨文档位置 $M[s,t]=0$ → scores 填 $-\infty$ → $\exp(-\infty)=0$ → 权重严格为 0。

**Position IDs** 在每条文档内从 0 重置,确保 RoPE 等位置编码在文档级别正确。

## 文件

- `from_scratch.py` — `pack_sequences` + `build_block_diagonal_mask` + `packed_attention_forward`(不用 flash-attn、不用 `nn.MultiheadAttention`)。
- `test_sequence_packing.py` — 11 个测试:cu_seqlens 边界、position_ids 重置、block-diagonal 隔离跨文档、注意力权重求和为 1。

```bash
python test_sequence_packing.py        # 或 python -m pytest test_sequence_packing.py
```

## 追问分层 / Stratified follow-ups

- **L1**:packed_ids 和 padding 的区别是什么?cu_seqlens 里存的是什么?为什么 position_ids 要在每个文档内重置?loss_mask 有什么用?
- **L2**:block-diagonal mask 是怎么用 cu_seqlens 构造的?跨文档 attention 为什么在 softmax 后严格为 0 而不是"很小"?packed sequence 的 causal mask 和普通 causal mask 有什么区别?为什么普通 padding mask 无法替代 cu_seqlens(见上表)?
- **L3**:flash-attn `flash_attn_varlen_func` 如何用 cu_seqlens 避免构造 $(T, T)$ mask(IO-aware 分块 + 索引算术)?在 RLHF/DPO 训练中同一个 batch 混合 chosen 和 rejected 序列时如何正确设置 cu_seqlens 和 loss_mask?多文档 packing 对梯度的影响:若两条文档共享一个 packed tensor,梯度是否会跨文档累积?
