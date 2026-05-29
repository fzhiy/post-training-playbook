# Post-Training Playbook

> 面向 **LLM 后训练(post-training / alignment)** 研究·算法实习面试的双语学习手册:每个主题 = 公式推导 + from-scratch PyTorch + 分层面试题(L1/L2/L3),外加可运行的「手撕」drill。交付格式借鉴 [ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)。
>
> A bilingual (中文 / EN) study & interview-prep playbook for **LLM post-training / alignment** roles: each topic = formula derivation + from-scratch PyTorch + stratified (L1/L2/L3) interview questions, plus runnable from-scratch drills.

## 📖 在线浏览 / Read online

👉 **https://fzhiy.github.io/post-training-playbook/** — 可搜索、手机/电脑响应式。

## 📂 内容 / Contents

- **`cheatsheets/`** — 双语题解:ML/DL 基础、LLM 架构、post-training 流程、PEFT、数学与统计、ML 系统设计、**推理-RL 前沿(GRPO / DAPO / Dr.GRPO / RLVR / long-CoT)**、**奖励建模与评测**。
- **`drills/`** — 可运行 from-scratch 实现 + 测试:attention、LoRA / DoRA、RoPE、KV cache、采样(temperature / top-k / top-p)、RMSNorm、GQA/MQA、SwiGLU、**DPO / PPO-clip / GRPO loss**、AdamW、cross-entropy。
- **`docs/`** — 渲染好的静态站点(GitHub Pages 源)。`node build.js` 重建。

## ⚠️ 说明 / Disclaimer

AI 辅助整理(生成 + 跨模型复审)的**学习笔记**,**WIP,欢迎 issue / PR 纠错**。
- 手撕 drill 的测试请在本地有 PyTorch 的环境跑过验证。
- 涉及具体论文的数字/结论以**原论文**为准。

## 🔧 构建 / Build

```bash
node build.js   # 读 cheatsheets/ + drills/ → 生成 docs/
```

## License

[MIT](LICENSE) © 2026 Feng Yu
