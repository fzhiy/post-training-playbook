# 持续 / 终身 Post-training / Continual & Lifelong(只收生产验证方法)

> 模型是**迭代更新**的(加新数据/新能力/新对齐轮)→ 灾难性遗忘与能力回归是规模化时的真问题。
> ⚠️ 本页**只收在大规模生产中验证过的方法**;经典学术 CL 算法单列、明确标注「未经生产验证」,面试别当工业标准答。

## 1. 生产里「持续」长什么样

不是教科书式的在线流式持续学习,而是:**周期性从 base / checkpoint 重训 + 调数据配比**。目标 = 加新能力/新对齐,同时不退化已有能力(避免 alignment tax / 回归)。

## 2. ✅ 生产验证过的工具箱

### 2.1 数据回放 / 混合(replay / rehearsal)—— 最主力
持续微调时混入一定比例的**旧/通用数据**(指令数据配比)。最朴素也最有效的防遗忘手段;工程重点是配比、去重、质量过滤。

### 2.2 低学习率 + 少 epoch + PEFT
小步微调限制权重漂移;LoRA / adapter 做**廉价增量适配 + 改动隔离**(改坏了可丢弃 adapter)。

### 2.3 对 base 的 KL 正则
RLHF 的 $\beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})$ 本质就是把策略**锚在 base 附近**、防漂移与遗忘。

### 2.4 模型合并 / 权重平均
- **model soups**(平均多个微调 checkpoint)、**task arithmetic**(任务向量加减)、**EMA**、**WiSE-FT**。
- 平均 / soup 验证充分;**TIES / DARE** 较新,生产用需自行验证。

### 2.5 蒸馏 consolidation
把多个专家 / 更新后的 teacher **蒸馏**成一个模型,巩固能力、压缩多轮迭代。

## 3. ❌ 未经生产验证(学术——别当工业标准)

- 正则化系:**EWC、SI、MAS**;梯度投影系:**GEM / A-GEM**;结构系:**PackNet、progressive networks**。
- 这些在 LLM 规模生产里**基本不用**——成本/复杂度高,效果不及朴素的 replay + 合并 + PEFT + KL。
- 面试口径:可提「学术上有 EWC 等」,但要诚实补一句「**生产主流是 replay / 合并 / PEFT / KL**」。

## 4. 把你的 CL 背景诚实地用上

Fed-TaLoRA(联邦持续微调)、Continual Agent → **可迁移的洞察**(遗忘度量、保持率视角、聚合一致性)。
- ✅ 诚实框架:「我研究持续学习,所以理解生产里为什么更朴素的 replay / 合并就够用、以及它们的边界」。
- ❌ 别声称「我做过生产级 continual post-training」。

## 分层面试题 / Stratified follow-ups

### L1 基础
- 持续微调为什么会遗忘?最朴素有效的防遗忘方法是什么(replay)?为什么 LoRA 有助于减遗忘?

### L2 进阶
- replay 的数据配比怎么定?KL 正则为什么能防遗忘?model soup / 权重平均为什么 work、前提是什么(同一初始化/模式连通性)?

### L3 深挖
- 为什么经典 CL 算法(EWC 等)在 LLM 生产里不流行?task arithmetic 的假设与失效情形?
- 连续 SFT → DPO → RL 多轮里,遗忘最严重在哪一步、怎么缓解?
- 「持续对齐(continual alignment)」与「重训」在成本/效果上怎么权衡?什么时候值得真正做增量而非重训?
