# 长程 / 自进化 Agent:生产现状 vs 前沿(面试向)

> **long-horizon(长程)** = 多步、长流程、需持续自主执行的任务;**self-evolving(自进化)** = 让 agent 用自生成数据 / 自我反馈不断变强。
> ⚠️ 本页**严格分栏**:【生产】= 已发布产品 / 官方工程指南;【前沿】= 论文 / 技术报告,**未成工业标准**。面试别把前沿当生产标准答。
> 诚信声明:本页"面试考点"是据**公开论文 / JD 推断的高频问题簇**,**非可查证的真实原题**;不放未经核实的 benchmark 数字。深度前沿(自进化全自动化等)不在本 playbook 范围,只给信号。

## 1. 【生产】长程 agentic 现在长什么样(已发布产品)

国际两家已把"长程 agentic"做进产品 —— 聊"agent 落地"时的硬通货:

- **Anthropic computer use**(2024-10-22 public beta;Anthropic API / Amazon Bedrock / Google Vertex AI):Claude 看屏幕截图 → 移动光标 / 点击 / 输入,把指令翻成一连串电脑操作;任务"需要几十、有时几百步"。官方明说**实验性、笨拙易错**,建议从低风险任务起步。<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">把 Claude 接上屏幕:截图→光标/点击/输入,几十至几百步的电脑操作(官方称实验性)。<a href="https://www.anthropic.com/news/3-5-models-and-computer-use">Anthropic 2024 ↗</a></span></span>
- **OpenAI Operator**(2025-01 研究预览,ChatGPT Pro)→ **ChatGPT agent**(2025-07-17,合并 Operator + deep research):底层 **CUA**(Computer-Using Agent)= GPT-4o 视觉 + RL 推理;循环 = 看截图 → CoT 推理下一步 → 点击/滚动/输入,直到完成或需人介入。安全:输密码要用户接管、拒高风险任务(如银行转账)。<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">OpenAI 的 computer-use agent:GPT-4o 视觉 + RL,截图→推理→操作循环。<a href="https://openai.com/index/introducing-operator/">OpenAI 2025 ↗</a></span></span> ChatGPT agent 在一台**虚拟电脑**上用 visual browser + text browser + terminal + API。<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">把 Operator(操作)与 deep research(综述)合到一台虚拟电脑上。<a href="https://openai.com/index/introducing-chatgpt-agent/">OpenAI 2025 ↗</a></span></span>

## 2. 【生产】长程 agent 的工程支柱(官方指南口径,面试高频)

**Anthropic《Building Effective Agents》(2024-12-19)**<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Anthropic agent 工程经典:workflow vs agent、ACI、停止条件、每步环境 ground truth。<a href="https://www.anthropic.com/research/building-effective-agents">Anthropic 2024 ↗</a></span></span> ≈ 这一行的工程"圣经":

- **Workflow vs Agent**(必答):workflow = LLM / 工具走**预定义代码路径**;agent = LLM**自己动态决定**流程与工具用法(控制流归谁是关键区别)。
- 常见模式:prompt chaining、routing、parallelization(sectioning / voting)、orchestrator-workers、evaluator-optimizer;以及 **autonomous agents**(靠环境反馈循环、自主规划到完成或触发停止条件)。
- **何时才上 agent**:任务开放、步数不可预测、无法硬编码路径,且愿用更高延迟/成本换表现 —— **否则先用简单方案**(单次调用 + 检索 + few-shot)。
- 工程要点:把 **ACI(agent-computer interface)** 当 HCI 一样精雕;设**停止条件**(如最大迭代数);每步要**环境 ground truth**(工具结果 / 代码执行)评估进度;sandbox + guardrails 防误差累积。

**Claude Agent SDK**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">长程 agent 循环 gather→act→verify→repeat + 上下文管理(compaction / 文件当记忆 / subagent)。<a href="https://claude.com/blog/building-agents-with-the-claude-agent-sdk">Anthropic ↗</a></span></span> 的核心循环(可直接背):

> **gather context → take action → verify work → repeat**

- **上下文管理**(长程不爆 context):compaction(自动摘要旧消息)、**文件系统当记忆**(grep / tail 按需取)、subagents(隔离上下文、并行、只回传摘要)。
- **自我验证**:rules-based(linter,精确)、visual(截图,验 UI)、LLM-as-judge(模糊标准,代价是延迟 / 稳定性)。
- 可靠性 = 合适的工具 + 清晰反馈 + 代表性场景测试 + 按失败模式迭代。

## 3. 【前沿】长程 / agentic 的训练范式(论文口径,未成工业标准)

> ⚠️ 以下是**研究**口径,非某产品的生产部署声明。面试可讲"我跟踪到 X 方向",别说"这是工业标准"。

### 3.1 稀疏 / 长程奖励 + 难度带设计(高频系统设计题)
长程任务奖励稀疏(常只有最终成败)。反复出现的工程原则:**有效 RL 信号只在中间难度带**,要显式防止训练数据退化到两端。

**Self-Play SWE-RL**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">同一 LLM 注入并修 bug 的自对弈 RL;难度带分段奖励(本页只取奖励设计)。<a href="https://arxiv.org/abs/2512.18552">Wei 2025 ↗</a></span></span>:同一 LLM 既注入 bug 又修 bug,用测试套件当奖励。bug 注入奖励是分段函数($s$ = 修复者解出该 bug 的比例,即 solve rate):

$$
r_{\text{inject}} = \begin{cases} -\alpha, & s \in \{0, 1\} \\ 1-(1+\alpha)\,s, & 0 < s < 1 \end{cases}, \quad \alpha = 0.8
$$

对"太难(没人解出,$s{=}0$)"和"太易(全解出,$s{=}1$)"都给负分,只奖中间难度。*(本页只取奖励设计;其性能数字未经核实,不引用。)*

**MiMo**<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">小米已发布模型的 RL 配方:去 KL loss、Clip-Higher、动态采样过滤 pass-rate 0/1。<a href="https://arxiv.org/abs/2505.07608">Xiaomi 2025 ↗</a></span></span>(小米已发布模型的 RL 配方):动态采样**过滤 pass-rate$=0/1$** 的 prompt,并维护 10% 简单题池防后期策略更新不稳定。

**两者动机一致** = 难度自适应课程:把信号集中到模型"够得着但还没掌握"的题上。

### 3.2 Web / agent RL 的三大挑战(WebRL 框架)
答"为什么 web/long-horizon agent 难训"的标准结构:① **训练任务稀缺**;② **反馈信号稀疏**;③ **策略分布漂移**。<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">web agent 训练三大挑战:任务稀缺 / 稀疏反馈 / 策略漂移(本页只取此框架)。<a href="https://arxiv.org/abs/2411.02337">Qi 2024 ↗</a></span></span> *(WebRL 核心是"失败轨迹 → 自进化课程";本页只取其三大挑战框架,不展开该机制。)*

### 3.3 与现有页的接点
GRPO 改进(Clip-Higher、去 KL loss)源自字节 **DAPO**<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">字节的 GRPO 改进:Clip-Higher、去 KL loss。<a href="https://arxiv.org/abs/2503.14476">ByteDance 2025 ↗</a></span></span>、被 MiMo 采用 —— 详见 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html),本页不重复。

## 4. 【前沿·最不成熟】自进化 / self-evolving(最需谨慎)

> ⚠️ 这块**绝大多数是研究**,生产证据弱。面试别声称工业标准,更别报未核实数字。

- 思路:让 agent 用**自生成数据 / self-play / 从失败造新任务 / reflection-self-correction** 自我提升,绕开人工标注。
- 现状口径:有论文探索(自动课程、self-play 搜索等),但"无监督全自动扩展"的强表述常**经不起核实**;当**开放问题**讲,而非成熟方案。深度处理留给独立的 [agent-post-training-playbook](https://github.com/fzhiy/agent-post-training-playbook)。
- **诚实接点(你的研究)**:你的 **Continual Agent(进行中)+ Fed-TaLoRA 抗遗忘** 视角 → 可说"我关注 agent 的持续学习 / 抗遗忘,所以理解自进化为什么在生产里还不成熟、边界在哪";**别**说"我做过自进化 agent 生产系统"。见 [continual-post-training](cheatsheet-continual-post-training.html)。

## 5. 面试考点 / Stratified follow-ups

> 据公开论文 / JD 推断的高频簇,**非真实原题**。

### L1 基础
- workflow 和 agent 的区别?什么时候**不该**用 agent(workflow / 单次调用就够)?
- computer use / Operator 怎么工作(截图 → 推理 → 操作的循环)?

### L2 进阶
- 长程 agent 怎么不爆 context(compaction / 文件当记忆 / subagent)?
- 长程任务奖励稀疏怎么办?为什么要难度带 / 动态采样(过滤 pass-rate $0/1$)?
- agent 怎么自我验证(rules / visual / LLM-as-judge),各自代价?

### L3 深挖
- 设计一个长程 coding agent 的奖励:怎么防 reward hacking 与退化(太难 / 太易都没信号)?
- web / long-horizon agent 训练的三大挑战(任务稀缺 / 稀疏反馈 / 策略漂移)各怎么缓解?
- 自进化 / self-play 训练的失败模式是什么?为什么生产里还不敢全自动?

## 参考文献 / References

> 点上标 `[N]` 跳到此处、点 `↩` 返回原文;宽屏时摘要(gist)直接浮现在右页边。

<ol class="refs">
<li id="ref-1">Anthropic — <em>Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku</em>(2024-10-22). <a href="https://www.anthropic.com/news/3-5-models-and-computer-use">anthropic.com</a> <a href="#fnref-1">↩</a></li>
<li id="ref-2">Anthropic — <em>Building Effective Agents</em>(2024-12-19). <a href="https://www.anthropic.com/research/building-effective-agents">anthropic.com</a> <a href="#fnref-2">↩</a></li>
<li id="ref-3">Anthropic — <em>Building agents with the Claude Agent SDK</em>. <a href="https://claude.com/blog/building-agents-with-the-claude-agent-sdk">claude.com</a> <a href="#fnref-3">↩</a></li>
<li id="ref-4">OpenAI — <em>Introducing Operator</em> / <em>Computer-Using Agent (CUA)</em>(2025-01). <a href="https://openai.com/index/introducing-operator/">openai.com</a> <a href="#fnref-4">↩</a></li>
<li id="ref-5">OpenAI — <em>Introducing ChatGPT agent</em>(2025-07-17). <a href="https://openai.com/index/introducing-chatgpt-agent/">openai.com</a> <a href="#fnref-5">↩</a></li>
<li id="ref-6">Wei et al.(Meta / FAIR) — <em>Toward Training Superintelligent Software Agents through Self-Play SWE-RL</em>. <a href="https://arxiv.org/abs/2512.18552">arXiv:2512.18552</a> — 本页只取奖励设计,不含未核实性能数字. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Xiaomi LLM-Core — <em>MiMo Technical Report</em>. <a href="https://arxiv.org/abs/2505.07608">arXiv:2505.07608</a>. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Qi et al. — <em>WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum RL</em>. <a href="https://arxiv.org/abs/2411.02337">arXiv:2411.02337</a> — 本页只取三大挑战框架,不展开其自进化课程机制. <a href="#fnref-8">↩</a></li>
<li id="ref-9">ByteDance Seed — <em>DAPO</em>. <a href="https://arxiv.org/abs/2503.14476">arXiv:2503.14476</a>. <a href="#fnref-9">↩</a></li>
</ol>
