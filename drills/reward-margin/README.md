# Drill: Reward Margin / Bradley-Terry Loss

> 可运行的 from-scratch 实现 + 测试。reward model 训练的核心损失。

## 数学 / The math

Bradley-Terry 模型:给定 chosen / rejected 的标量奖励 $r_w, r_l$,

$$P(w \succ l) = \sigma(r_w - r_l)$$

训练 reward model = 最大化该似然 = 最小化**成对 logistic 损失**:

$$L = -\,\mathbb{E}\big[\log\sigma(r_w - r_l)\big]$$

直觉:把 chosen 的分数推高于 rejected,差距(margin)$r_w-r_l$ 越大,损失越小。
- margin $=0$ → $L=\log 2\approx 0.693$;
- margin $\to+\infty$ → $L\to 0$。

诊断量:**margin**(平均奖励差)、**accuracy**(排序正确的比例 $r_w>r_l$)。

## 与 DPO 的关系

DPO 把策略本身当成隐式 reward($r=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$),它的损失就是把这里的 $r_w,r_l$ 换成隐式奖励的同一个 BT 损失——所以 DPO「不需要单独的 reward model」。

## 文件 / Files

- `from_scratch.py` — `bt_loss` + `reward_metrics`(margin / accuracy)(+ `python from_scratch.py` 演示)。
- `test_reward_margin.py` — margin=0 → $\log 2$、大 margin → 0、metrics + 反传。

```bash
python from_scratch.py            # 演示
python test_reward_margin.py      # 测试(或 pytest)
```

## 追问分层 / Stratified follow-ups

- **L1**:为什么用 $\log\sigma(r_w-r_l)$ 而不是直接最大化 $r_w-r_l$?accuracy 和 loss 有什么区别?
- **L2**:margin=0 时损失为什么是 $\log 2$?reward model 的**长度偏置**怎么来、怎么缓解?
- **L3**:DPO 的损失和这个 BT 损失什么关系?pointwise RM 与 pairwise RM 的取舍?reward hacking 在这个框架下如何发生?
