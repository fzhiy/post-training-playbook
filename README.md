# 📚 Post-Training Playbook

> **LLM 后训练(post-training / alignment / reasoning-RL)研究·算法实习面试的双语自学手册。**
> 每个主题 = 公式推导 + from-scratch PyTorch + 分层面试题(L1/L2/L3),外加可运行的「手撕」drill。交付格式借鉴 [ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)。
>
> A bilingual (中文 / EN) self-study & interview-prep playbook for **LLM post-training / alignment** roles.

🔗 **在线阅读 / Read online:** <https://ac.fzhiy.net/post-training-playbook/> · MIT · AI-assisted · WIP

## 🚀 从这里开始 / Start here

打开 **<https://ac.fzhiy.net/post-training-playbook/>**(可搜索、响应式、手机可读),先看 **[学习路径 / Roadmap](https://ac.fzhiy.net/post-training-playbook/cheatsheet-00-roadmap.html)** 按序刷。

> 公式(KaTeX)+ 代码(highlight.js)**构建时静态渲染**、**零运行时 CDN**(国内直连、可离线)、每页目录 TOC、首页关键词搜索。

## 📂 内容 / Contents

- **`cheatsheets/`(10)** — ML/DL 基础 · 数学统计 · LLM 架构 · PEFT · post-training 流程 · **奖励建模与评测** · **推理-RL 前沿(GRPO / DAPO / Dr.GRPO / RLVR / long-CoT)** · ML 系统设计 · 算法 · 学习路径。
- **`drills/`(14)** — 可运行 from-scratch + 测试:attention · LoRA / DoRA · RoPE · KV cache · 采样 · RMSNorm · GQA/MQA · SwiGLU · **DPO / PPO-clip / GRPO loss** · AdamW · cross-entropy。
- **`docs/`** — 渲染好的静态站点(GitHub Pages 源)。

## 🔧 构建 / Build

```bash
npm install      # marked + katex + highlight.js(仅构建时用)
node build.js    # 读 cheatsheets/ + drills/ → 生成 docs/
```

## ⚠️ 说明 / Disclaimer

AI 辅助整理(生成 + 跨模型复审)的**学习笔记**,**WIP**。
- ✅ 全部 **14 个 drill 的测试已在 PyTorch 2.4 跑通**(`pytest` 或 `python test_*.py`)。
- 涉及具体论文的数字 / 结论以**原论文**为准。
- 发现问题欢迎 issue / PR —— 见 [CONTRIBUTING](CONTRIBUTING.md)。

## License

[MIT](LICENSE) © 2026 Feng Yu
