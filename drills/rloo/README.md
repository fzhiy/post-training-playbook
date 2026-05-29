# Drill: RLOO (REINFORCE Leave-One-Out)

> 可运行的 from-scratch 实现 + 测试。critic-free 的策略梯度基线。

## 数学 / The math

对每个 prompt 采 $K$ 个回答,奖励 $r_1..r_K$。RLOO 用**留一法**作基线(无 value 网络):

$$A_i = r_i - \frac{1}{K-1}\sum_{j\neq i} r_j = r_i - \frac{(\sum_k r_k) - r_i}{K-1}$$

策略梯度损失(最小化):$\;L = -\frac{1}{BK}\sum_{i}\mathrm{sg}(A_i)\,\log\pi_\theta(o_i)$,其中 $\mathrm{sg}$ 是 stop-gradient(基线不回传)。

性质:每个 prompt 内 $\sum_i A_i = 0$(无偏、低方差基线)。

## 与 PPO / GRPO 的对比

| | baseline | critic? | clip? |
|---|---|---|---|
| PPO | value 网络 $V(s)$ + GAE | 需要 | 有 |
| GRPO | 组内标准化 $(r-\mu)/\sigma$ | 不需要 | 有 |
| **RLOO** | **组内留一均值** $\frac{\sum_{j\neq i}r_j}{K-1}$ | 不需要 | 无(纯 REINFORCE) |

## 文件 / Files

- `from_scratch.py` — `rloo_advantages` + `rloo_loss`(+ `python from_scratch.py` 演示)。
- `test_rloo.py` — 留一基线解析对拍、组内和为 0、loss 有限 + 反传。

```bash
python from_scratch.py     # 演示
python test_rloo.py        # 测试(或 pytest)
```

## 追问分层 / Stratified follow-ups

- **L1**:为什么要减基线?留一均值相比"全组均值"有什么好处?
- **L2**:为什么 $\sum_i A_i = 0$?基线为什么要 stop-gradient?$K$ 太小/太大各有什么问题?
- **L3**:RLOO vs GRPO 的方差与偏差权衡?RLOO 为什么可以不 clip 而 PPO 需要?在线 vs 离线采样对 RLOO 的影响?
