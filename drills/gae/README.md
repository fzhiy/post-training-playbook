# Drill: GAE (Generalized Advantage Estimation)

> 可运行的 from-scratch 实现 + 测试。PPO 里估计优势 $A_t$ 的标准做法。

## 数学 / The math

TD 残差:$\;\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

GAE 是 $\delta$ 的指数加权和:

$$A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}$$

实现上用**反向递推**(从后往前):$\;A_t = \delta_t + \gamma\lambda\,A_{t+1}$。回报 $R_t = A_t + V(s_t)$。

两个极端:
- $\lambda=0$:$A_t=\delta_t$ —— 单步 TD,**低方差、高偏差**。
- $\lambda=1$:$A_t=\sum_l\gamma^l r_{t+l} - V(s_t)$ —— 蒙特卡洛,**高方差、低偏差**。
- $\lambda\in(0,1)$:在偏差/方差间插值。

## 文件 / Files

- `from_scratch.py` — `compute_gae(rewards, values, last_value, gamma, lam, dones)`(+ `python from_scratch.py` 演示)。
- `test_gae.py` — $\lambda{=}0$ 退化为单步 TD、$\lambda{=}1$ 退化为 MC$-V$、done 重置 bootstrap。

```bash
python from_scratch.py     # 演示
python test_gae.py         # 测试(或 pytest)
```

## 追问分层 / Stratified follow-ups

- **L1**:$\delta_t$ 是什么?GAE 里 $\lambda$ 调大调小分别意味着什么?
- **L2**:为什么用反向递推而不直接求和?$\lambda=0$ 和 $\lambda=1$ 各对应什么经典估计?`done` 为什么要切断 bootstrap?
- **L3**:GAE 的偏差/方差如何随 $\gamma,\lambda$ 变化?为什么 GRPO/RLOO 能不用 GAE?优势归一化(减均值除标准差)在 PPO 里的作用?
