# Drill: Attention from scratch

> 可运行的 from-scratch 实现 + 测试。目标:每一行都能在面试里推导和辩护。
> Runnable from-scratch implementation with tests — derive and defend every line.

## 数学 / The math

注意力对一组 query 在一组 key-value 上做加权聚合:

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $QK^\top$:每个 query 与每个 key 的点积相似度,形状 $(L_q, L_k)$。
- $\sqrt{d_k}$ 缩放:设各维独立、均值 0 方差 1,则 $\mathrm{Var}(q\cdot k)=d_k$。不缩放会把 softmax 推向饱和、梯度消失;除以 $\sqrt{d_k}$ 把方差拉回 ~1。
- softmax 沿 **key 维**归一化 → 每个 query 的注意力分布。
- 乘 $V$:用该分布对 value 加权求和。
- 多头:把 $d_{model}$ 切成 $h$ 个 $d_{head}=d_{model}/h$ 子空间,各自独立注意力 → 拼接 → 输出投影 $W_O$,让不同头关注不同关系。

## 复杂度 / Complexity

- 时间 / 显存 $O(L^2 d)$ —— 序列长度的平方项是长上下文的核心瓶颈。
- 自回归解码时缓存 K、V(**KV cache**),把每步从重算 $O(L^2)$ 摊销到 $O(L)$。

## 文件

- `from_scratch.py` —— `scaled_dot_product_attention` + `MultiHeadAttention`(不用 `nn.MultiheadAttention` / `F.sdpa`)。
- `test_attention.py` —— 与 PyTorch 参考实现对拍:数值 `allclose` + causal mask 正确性 + 反向传播有限。

```bash
python test_attention.py        # 或 python -m pytest test_attention.py
```

## 追问分层 / Stratified follow-ups

- **L1**:为什么除以 $\sqrt{d_k}$ 而不是 $d_k$?softmax 为什么沿 key 维而不是 query 维?mask 为什么填 $-\infty$ 而不是 0?
- **L2**:causal mask 怎么实现?自注意力的时间/显存复杂度是多少?KV cache 省的是哪一部分计算,为什么训练时不需要?
- **L3**:FlashAttention 在**不改变数学结果**的前提下怎么把显存从 $O(L^2)$ 降到 $O(L)$(IO-aware 分块 + online softmax)?MQA / GQA 如何用 KV 头共享换显存与带宽?多头相比单头大注意力的本质收益是什么?
