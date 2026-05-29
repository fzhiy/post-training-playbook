# Drill: SFT loss masking from scratch

> 可运行的 from-scratch 实现 + 测试。目标:每一行都能在面试里推导和辩护。
> Runnable from-scratch implementation with tests — derive and defend every line.

## 做什么 / What this covers

监督微调（SFT）的损失函数有两个分开的问题需要解决：

Supervised fine-tuning (SFT) involves two distinct sub-problems:

1. **Label masking** — 只在 assistant 回答的 token 上计算损失；prompt / user turn 被设成 `ignore_index=-100`，对梯度没有贡献。
   Only compute loss on assistant-response tokens; prompt / user tokens are set to `ignore_index=-100` and contribute no gradient.

2. **Masked cross-entropy** — 在含有 `ignore_index` 的 labels 上安全地算交叉熵；索引越界问题用 clamp-then-mask 模式规避。
   Safely compute cross-entropy over a labels tensor containing `ignore_index`; index-out-of-bounds is avoided with the clamp-then-mask pattern.

## 数学 / The math

### Label masking

给定序列 $x_{1:L}$ 和若干 assistant spans $\{[s_i, e_i)\}$，构造标签：

Given sequence $x_{1:L}$ and assistant spans $\{[s_i, e_i)\}$, construct:

$$
y_t = \begin{cases} x_t & \text{if } t \in \bigcup_i [s_i, e_i) \\ -100 & \text{otherwise} \end{cases}
$$

### Masked cross-entropy loss

Per-token 负对数似然（只对 active 位置求和）：

Per-token negative log-likelihood summed only over active positions:

$$
\mathcal{L} = \frac{1}{|\mathcal{A}|} \sum_{t \in \mathcal{A}} -\log p_\theta(y_t \mid x_{<t})
$$

其中 $\mathcal{A} = \{t : y_t \neq -100\}$，$p_\theta$ 是模型的 softmax 输出。

where $\mathcal{A} = \{t : y_t \neq -100\}$ and $p_\theta$ is the model's softmax output.

**两种归一化 / Two normalisation conventions:**

| mode | denominator | 适用场景 |
|------|-------------|---------|
| `"token"` | $\|\mathcal{A}\|$（非 mask token 数） | HuggingFace Trainer / TRL 默认；每个 token 等权 |
| `"sample"` | $L$（序列总长度） | 部分 RL trainer；loss scale 随 batch size 稳定 |

### Clamp-then-mask 模式 / The clamp-then-mask pattern

直接用 `-100` 做 `torch.gather` 的索引会触发越界错误（CUDA backend 尤甚）。正确做法：

Directly using `-100` as an index into `torch.gather` raises an out-of-bounds error on most backends. The safe pattern:

```python
safe_labels = labels.clamp(min=0)         # 1. make every index legal
nll = -log_probs.gather(1, safe_labels)   # 2. gather — all indices in [0, V-1]
nll = nll * (labels != ignore_index)      # 3. zero out the masked positions
```

步骤 3 的零乘保证 -100 位置的"错误"gather 结果对 loss 贡献为 0，数学上等价于跳过这些位置。

The multiply-by-zero in step 3 ensures the "wrong" gather result at `-100` positions contributes 0 to the loss — mathematically identical to skipping those positions entirely.

## 为什么重要 / Why it matters

- 没有 label mask，模型会在 prompt token 上也算梯度 → 训成一个不分语境乱答的模型（格式崩坏、prompt leaking）。
  Without label masking, the model trains on prompt tokens too → learns to regurgitate prompts, format collapses.
- 错误的归一化（per-sample vs per-token）在 mixed-length batch 里会造成长短样本梯度不均衡，影响收敛。
  Wrong normalisation in mixed-length batches causes unequal gradient weight for long vs short samples, hurting convergence.
- clamp-then-mask 是生产代码里常见的 defensive pattern；面试时能主动提出说明工程意识。
  The clamp-then-mask pattern is a common production defensive technique — proactively naming it signals engineering awareness.

## 文件 / Files

- `from_scratch.py` — `mask_labels_for_sft` + `masked_ce_loss`（两种归一化）
- `test_sft_loss_mask.py` — 11 个 assert 自测，覆盖单轮/多轮 masking、两种归一化、全 mask 边界、clamp 无越界、端到端管道

```bash
python test_sft_loss_mask.py        # 或 python -m pytest test_sft_loss_mask.py
```

## 追问分层 / Stratified follow-ups

- **L1**: 为什么要 mask prompt token？不 mask 会怎样？
  Why mask prompt tokens? What goes wrong if you don't?
  > 模型会对 prompt 中的每个 token 分配梯度，学到"复读"prompt 的行为而不是真正的 assistant 行为；chat 格式崩坏。

- **L1**: `ignore_index=-100` 的默认值从哪里来？为什么选 -100 而不是 0？
  Where does `ignore_index=-100` come from? Why not 0?
  > PyTorch 约定（`F.cross_entropy` 默认值）；0 是合法 vocab id，-100 在正常词表范围外、不歧义。

- **L2**: token 归一化 vs sample 归一化在 mixed-length batch 时各有什么问题？
  What are the failure modes of token vs sample normalisation in a mixed-length batch?
  > Token 归一化给长样本更大的 batch-level 梯度（更多 active token）。Sample 归一化给有大量 padding 的短样本很小的 loss，但对 batch 内 loss scale 更稳定。具体选哪种取决于是否做 padding 截断和下游 RL loss 的 scale 需求。

- **L2**: 如果 gather index 不 clamp 直接传 -100 会发生什么？各平台差异？
  What happens if you pass -100 directly to gather without clamping? Platform differences?
  > CPU：undefined behavior（可能静默返回垃圾值）；CUDA：assertion fail / illegal memory access；MPS：报错。Clamp 是防御性写法，不改变数学结果。

- **L3**: 多轮对话时，如何确保 chat template 的 `[INST]`/`<|user|>` token 边界与 span 索引完全对齐？TRL 是怎么做的？
  In multi-turn chat, how do you guarantee chat-template token boundaries align exactly with span indices? How does TRL handle this?
  > TRL `DataCollatorForCompletionOnlyLM` 通过搜索特殊 token（如 `<|assistant|>`）的 token id 序列来找 span 边界，而不是字符级偏移；tokenizer 的 `add_special_tokens` 设置会影响边界，需要一致性。

- **L3**: 在 pack-sequence（多个样本拼入同一个 context window）的场景下，label mask 需要注意什么？
  When packing multiple samples into one context window, what extra care does label masking require?
  > 需要额外的 attention mask（或 document mask）防止样本之间的跨文档注意力；同时每个样本的 assistant span 必须在 pack 后的全局坐标系内重新计算。FlashAttention 的 `cu_seqlens` 接口支持这个 use-case。
