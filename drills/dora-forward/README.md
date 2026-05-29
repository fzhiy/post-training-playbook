# DoRA 从零实现 · Weight-Decomposed LoRA (per-output-row)

> **学习用单文件 drill**：从零复现 DoRA (Liu et al., 2024)，以 per-row（逐输出通道）归一化为唯一模式。

---

## 1 · 数学推导 / Math

**标准 LoRA：**

$$W' = W_0 + s \cdot BA$$

其中 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 是冻结预训练权重，$B \in \mathbb{R}^{d_{\text{out}} \times r}$，$A \in \mathbb{R}^{r \times d_{\text{in}}}$，缩放因子 $s = \frac{\alpha}{r}$。

**DoRA 的核心思想——将权重分解为幅值 (magnitude) 与方向 (direction)：**

$$W' = m \odot \frac{W_0 + s \cdot BA}{\left\| W_0 + s \cdot BA \right\|_c + \epsilon}$$

其中：

- $m \in \mathbb{R}^{d_{\text{out}}}$：可学习的 per-output-channel 幅值向量；
- $\|\cdot\|_c$：**per-row L2 范数**（沿 `dim=1`），对权重矩阵的每一行（即每个输出神经元）独立归一化，得到形状为 $(d_{\text{out}}, 1)$ 的范数向量；
- $\epsilon = 10^{-8}$：数值稳定项；
- $\odot$：此处为广播乘法——$m$ 通过 `unsqueeze(1)` 扩展为 $(d_{\text{out}}, 1)$，再与方向矩阵 $(d_{\text{out}}, d_{\text{in}})$ 逐元素相乘。

**初始化保证输出不变：**

$$B_0 = \mathbf{0} \implies \Delta W = 0 \implies W' = m_0 \odot \frac{W_0}{\|W_0\|_c}$$

令 $m_0 = \|W_0\|_c$（per-row L2 范数），则：

$$W' = \|W_0\|_c \odot \frac{W_0}{\|W_0\|_c} = W_0$$

完美还原预训练权重，**零扰动**。

**合并推理 (merge)：**

训练完成后，可将 DoRA 权重烘焙进单一矩阵，推理时退化为普通 `F.linear`，无额外开销。

---

## 2 · 直觉与复杂度 / Intuition & Complexity

### 直觉

| 概念 | 类比 |
|------|------|
| **方向 (direction)** | 预训练权重矩阵每一行的"方向"，即单位向量。LoRA 低秩更新在此方向上做微调。 |
| **幅值 (magnitude)** | 每个输出通道独立的标量，控制该通道的"强度"。可学习，解耦于方向。 |
| **per-row 归一化** | 将每行（每个输出神经元的全部输入权重）视为一个向量，除以其 L2 范数。 |

与标准 LoRA 直接加 $\Delta W$ 不同，DoRA 先归一化再乘幅值，使得 **方向更新更稳定、幅值可独立调节**，在下游任务中往往收敛更快、泛化更好。

### 可训练参数量

$$\underbrace{r \cdot d_{\text{in}}}_{\texttt{lora\_A}} + \underbrace{d_{\text{out}} \cdot r}_{\texttt{lora\_B}} + \underbrace{d_{\text{out}}}_{\texttt{magnitude}}$$

例如：$d_{\text{in}}=16, d_{\text{out}}=8, r=4 \Rightarrow 4 \times 16 + 8 \times 4 + 8 = 104$ 个可训练参数。

### 计算开销

与标准 LoRA 相比，DoRA 额外引入一次 per-row 范数计算，复杂度 $O(d_{\text{out}} \cdot d_{\text{in}})$，在实际大模型中通常可忽略。merge 后推理开销为零。

---

## 3 · 文件 / Files

| 文件 | 说明 |
|------|------|
| `from_scratch.py` | DoRA 核心实现 (`DoRALinear` 类) + 内置 smoke-test |
| `test_dora_forward.py` | 额外的前向传播/梯度/merge 测试 |
| `README.md` | 本文档 |

---

## 4 · 运行 / Run

```bash
# 快速验证：init 一致性、梯度流、训练效果、merge/unmerge 往返
python from_scratch.py

# 独立测试脚本
python test_dora_forward.py
```

两个命令均无需额外依赖（仅 `torch`），无需下载数据集或模型权重。

---

## 5 · 追问分层 / Stratified follow-ups

### L1 — 基础 / Basic

1. **初始化原理**：为什么 $B$ 初始化为零？如果改为随机初始化会怎样？
2. **per-row 归一化**：代码中 `torch.norm(adapted_w, p=2, dim=1, keepdim=True)` 里的 `dim=1` 代表什么物理含义？如果改成 `dim=0`（per-column）会归一化什么？
3. **参数计数**：对于 $d_{\text{in}}=4096, d_{\text{out}}=4096, r=16$，DoRA 比标准 LoRA 多了多少可训练参数？占比多少？

### L2 — 进阶 / Intermediate

4. **方向 vs. 幅值的解耦**：论文声称将方向和幅值解耦能带来更好的学习动力学。请从梯度流的角度解释：归一化操作如何改变 $\frac{\partial \mathcal{L}}{\partial m}$ 和 $\frac{\partial \mathcal{L}}{\partial (BA)}$ 的性质？
5. **scaling 因子**：`scaling = lora_alpha / r` 的作用是什么？如果去掉它（令 scaling=1），训练行为会发生什么变化？
6. **merge/unmerge 精度**：代码中 merge 和 unmerge 使用了 `torch.no_grad()` 和 `.detach()`。为什么？如果不这样做，merge 操作会被纳入计算图导致什么问题？

### L3 — 深入 / Deep

7. **per-row vs. per-column 归一化**：本实现采用 per-row（per-output-channel）归一化。如果改为 per-column（per-input-dimension）归一化，数学公式和物理含义会如何变化？在什么场景下哪种更合理？代码改动最小的方案是什么？
8. **数值稳定性**：代码中使用 `norm + 1e-8` 防止除零。在混合精度训练 (FP16/BF16) 下，这个 epsilon 是否足够？请讨论可能出现的数值问题及缓解方案。
9. **与 QDoRA 的联系**：DoRA 的 magnitude-direction 分解思想可以和量化 (Quantization) 结合（如 QLoRA → QDoRA）。请分析：如果 $W_0$ 被量化为 INT4 NF4 格式，per-row 归一化步骤应该在量化前还是量化后执行？对精度和效率有何影响？

---

> **引用**：Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. arXiv:2402.09353.