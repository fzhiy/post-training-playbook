# 长程 / 自进化 Agent:生产现状 vs 前沿(面试向)

> **long-horizon(长程)** = 多步、长流程、需持续自主执行的任务;**self-evolving(自进化)** = 让 agent 用自生成数据 / 自我反馈不断变强。
> ⚠️ 本页**严格分栏**:【生产】= 已发布产品 / 官方工程指南;【前沿】= 论文 / 技术报告,**未成工业标准**。面试别把前沿当生产标准答。
> 诚信声明:本页"面试考点"是据**公开论文 / JD 推断的高频问题簇**,**非可查证的真实原题**;不放未经核实的 benchmark 数字。深度前沿(自进化全自动化等)不在本 playbook 范围,只给信号。

## 1. 【生产】长程 agentic 现在长什么样(已发布产品)

国际两家已把"长程 agentic"做进产品 —— 聊"agent 落地"时的硬通货:

- **Anthropic computer use**(2024-10-22 public beta;Anthropic API / Amazon Bedrock / Google Vertex AI):Claude 看屏幕截图 → 移动光标 / 点击 / 输入,把指令翻成一连串电脑操作;任务"需要几十、有时甚至成百步"。官方明说**实验性、笨拙易错**,建议从低风险任务起步。<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">把 Claude 接上屏幕:截图→光标/点击/输入,几十至几百步的电脑操作(官方称实验性)。<a href="https://www.anthropic.com/news/3-5-models-and-computer-use">Anthropic 2024 ↗</a></span></span>
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

**agent 循环骨架**(精简版,展示控制流;实际需加 error handling / compaction / sandbox):

```python
def agent_loop(task, tools, max_steps=50, token_limit=128_000):
    """长程 agent 核心循环: gather context → take action → verify → repeat."""
    context = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    context.append({"role": "user", "content": task})

    for step in range(max_steps):
        # 1. gather: LLM 推理下一步做什么
        response = llm.chat(context, tools=tools)

        if response.has_tool_calls():
            for tc in response.tool_calls:
                result = execute_tool(tc)                # 真实工具执行(沙箱内)
                context.append(tool_result_msg(tc, result))
        else:
            break                                        # agent 判断任务完成

        # 2. verify: 用环境 ground truth 检查当前步
        if not verify_step(response, tools):             # rules/visual/LLM-as-judge
            context.append({"role": "user",
                           "content": "Step seems wrong; review the result and retry."})

        # 3. context 管理: 防 token 溢出
        if estimate_tokens(context) > token_limit:
            context = compact_context(context)           # 自动摘要旧消息

    return response.content
# 关键设计:
# - execute_tool 必须沙箱化(code 在隔离环境跑、browser 在虚拟浏览器)
# - verify_step 从廉价到昂贵梯次: rules → visual → LLM-as-judge
# - compact_context 保留关键决策点标记、只压缩中间对话
```

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

```python
def difficulty_band_reward(solve_rate, alpha=0.8):
    """Self-Play SWE-RL 分段奖励:太难/太易都罚,只奖中间难度。
    solve_rate: s ∈ [0,1] = 修复者中解出该 bug 的比例。"""
    if solve_rate == 0.0 or solve_rate == 1.0:
        return -alpha                       # 太难(没人解出)或太易(全解出) → 负分
    return 1.0 - (1.0 + alpha) * solve_rate # 中间 s ∈ (0,1):s 越小(越难)奖励越高
# 直觉: s=0→-0.8; s≈0.1(极难)≈0.82 最高; s=0.5→0.1; s=1→-0.8。
# 鼓励生成"极难但非无解"的 bug——多数修复者做不出、但偶尔有人能解的真难 bug。
# 注:此函数在 s→0+ 时奖励最接近 1；实际训练中还需过一致性验证,不合格则额外扣分。
```

### 3.2 Web / agent RL 的三大挑战(WebRL 框架)
答"为什么 web/long-horizon agent 难训"的标准结构:① **训练任务稀缺**;② **反馈信号稀疏**;③ **策略分布漂移**。<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">web agent 训练三大挑战:任务稀缺 / 稀疏反馈 / 策略漂移(本页只取此框架)。<a href="https://arxiv.org/abs/2411.02337">Qi 2024 ↗</a></span></span> *(WebRL 核心是"失败轨迹 → 自进化课程";本页只取其三大挑战框架,不展开该机制。)*

### 3.3 与现有页的接点
GRPO 改进(Clip-Higher、去 KL loss)源自字节 **DAPO**<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">字节的 GRPO 改进:Clip-Higher、去 KL loss。<a href="https://arxiv.org/abs/2503.14476">ByteDance 2025 ↗</a></span></span>、被 MiMo 采用 —— 详见 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html),本页不重复。

## 4. 【前沿·最不成熟】自进化 / self-evolving(最需谨慎)

> ⚠️ 这块**绝大多数是研究**,生产证据弱。面试别声称工业标准,更别报未核实数字。

- 思路:让 agent 用**自生成数据 / self-play / 从失败造新任务 / reflection-self-correction** 自我提升,绕开人工标注。
- 现状口径:有论文探索(自动课程、self-play 搜索等),但"无监督全自动扩展"的强表述常**经不起核实**;当**开放问题**讲,而非成熟方案。深度处理留给独立的 [agent-post-training-playbook](https://github.com/fzhiy/agent-post-training-playbook)。

**现有研究尝试(框架层面,非性能声明)**:

| 方向 | 代表工作 | 核心机制 | 已知的脆弱点 |
|------|------|------|------|
| 失败→新任务 | **WebRL**（[arXiv:2411.02337](https://arxiv.org/abs/2411.02337)） | 从失败轨迹反向构造新训练任务（"在这步错了→创造一个专项练习"） | 自动生成的任务可能不是"有效困难"而是"噪声" |
| Self-play RL | **Self-Play SWE-RL**（[arXiv:2512.18552](https://arxiv.org/abs/2512.18552)） | 同一 LLM 注入 bug → 修复 bug,用测试套件做 ground truth 奖励 | bug 注入可能退化到"注入太简单/太离谱的 bug" |
| 反思自纠正 | **Self-Refine**（[arXiv:2303.17651](https://arxiv.org/abs/2303.17651)） | 模型生成→自我批判→据反馈修订 | 纯内在自我纠正常改对为错（见 [test-time-scaling §2.3](cheatsheet-test-time-scaling.html)） |
| 多 agent 互评 | **MULTI-AGENT Debate** 系列 | agent A 生成、agent B 批判、迭代改进 | 多个 agent 共享相同的基础模型偏置→可能 groupthink |

**四种核心失败模式**(面试 L3 标准答案框架):① **模式崩塌**——模型偏好自己风格→正反馈→多样性丧失；② **奖励崩塌**——self-judgment 质量退化(越来越"宽容"自己)；③ **难度退化**——自生成任务趋向简单(模型造自己会做的题)；④ **分布外漂移**——自生成数据偏离真实任务分布。根本原因:**缺乏外部 ground truth 锚定**——没有独立于模型的客观验证信号来校准方向。可验证域(code/math)能提供锚定,开放域几乎没办法。对面试:坦率分析边界比宣称"能做"更有说服力。

- **诚实接点(你的研究)**:你的 **Continual Agent(进行中)+ Fed-TaLoRA 抗遗忘** 视角 → 可说"我关注 agent 的持续学习 / 抗遗忘,所以理解自进化为什么在生产里还不成熟、边界在哪";**别**说"我做过自进化 agent 生产系统"。见 [continual-post-training](cheatsheet-continual-post-training.html)。

## §A 核心论文 / 产品时间线 / Key Papers & Products Timeline

- **2023-03 · Self-Refine** — Madaan et al., NeurIPS 2023. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) — 同一 LLM "生成→自我批判→修订"循环,自进化思想萌芽；后续工作指出纯内在自我纠正收益有限。

- **2023-07 · WebArena** — Zhou et al., ICLR 2024. [arXiv:2307.13854](https://arxiv.org/abs/2307.13854) — 最早的系统化 web agent 评测之一,自建网站环境(电商/论坛/CMS)做程序化验证。

- **2024-05 · SWE-agent** — Yang et al., NeurIPS 2024. [arXiv:2405.15793](https://arxiv.org/abs/2405.15793) — 提出 ACI(Agent-Computer Interface)设计原则,用"精心设计的命令行工具"而非"原始 shell"提高 agent 效率。

- **2024-10 · Anthropic Computer Use** — Anthropic, 产品发布(public beta). [blog](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 看截图→操作电脑,任务需"几十到几百步",首个提供 computer use 公开 beta 的前沿 AI 模型。

- **2024-11 · WebRL** — Qi et al., ICLR 2025. [arXiv:2411.02337](https://arxiv.org/abs/2411.02337) — 提出 web agent 训练三大挑战(任务稀缺/稀疏反馈/策略漂移)+ 自进化课程框架;本页取其框架,未展开其机制。

- **2024-12 · Building Effective Agents** — Anthropic, 工程指南. [blog](https://www.anthropic.com/research/building-effective-agents) — workflow vs agent、常见模式、何时该上 agent、ACI/Sandbox 原则,agent 工程"圣经"。

- **2025-01 · OpenAI Operator / CUA** — OpenAI, 研究预览. [blog](https://openai.com/index/introducing-operator/) — GPT-4o 视觉 + RL 推理的 computer-use agent,截图→CoT→操作循环;2025-07 整合为 ChatGPT agent。

- **2025-03 · DAPO** — Yu et al.(ByteDance Seed), preprint. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) — Clip-Higher + 去 KL loss,被 MiMo 等 agent RL 配方采用(详见 [reasoning-rl-frontier](cheatsheet-reasoning-rl-frontier.html))。

- **2025-05 · MiMo** — Xiaomi LLM-Core, 技术报告. [arXiv:2505.07608](https://arxiv.org/abs/2505.07608) — 已发布模型的完整 RL 配方:去 KL loss + Clip-Higher + 动态采样过滤 + 难度带课程。

- **2025-12 · Self-Play SWE-RL** — Wei et al.(Meta/FAIR), accepted to ICML 2026. [arXiv:2512.18552](https://arxiv.org/abs/2512.18552) — 同一 LLM 注入并修复 bug 的 self-play RL,分段难度带奖励;本页只取奖励设计,不含未核实性能数字。

## 5. 面试考点 / Stratified follow-ups

> 据公开论文 / JD 推断的高频簇,**非真实原题**。

### L1 基础

<details>
<summary>Q1: workflow 和 agent 的区别？什么时候**不该**用 agent？</summary>

**答：** **Workflow** = LLM/工具走**预定义代码路径**（控制流在代码里——开发者事先写好"先调 A→如果 X 则调 B→否则调 C"）；**Agent** = LLM **自己动态决定**流程和工具用法（控制流在模型里——模型根据当前状态选择下一步做什么）。Anthropic 的判定标准：agent 适用场景=**任务开放、步数不可预测、无法硬编码路径**，且愿意用更高延迟/成本换表现。不该用 agent 的场景：① 任务步骤数可预测（如"查天气→给建议"固定两段）→ prompt chaining + routing 就够了；② 单次 LLM 调用 + 检索能搞定（如 FAQ 问答）；③ 延迟/成本敏感且 agent 的额外收益不足以覆盖代价。黄金法则：**"先用最简单的方案，只在必要时升级到 agent。"**

> **追问：** orchestrator-workers 和 autonomous agent 的区别在哪？
> orchestrator-workers 的"调度者"是 LLM 动态决定的（不像 prompt chaining 硬编码），但"执行者"是固定的子 LLM 调用——控制流在调度者手里、不在代码里，所以更接近 agent 侧，属于"受控的 agent"。

</details>

<details>
<summary>Q2: computer use / Operator 怎么工作（截图 → 推理 → 操作的循环）？</summary>

**答：** 核心循环：**截图 → 视觉理解当前屏幕 → CoT 推理下一步操作 → 执行操作（点击/滚动/输入/快捷键）→ 等待环境变化 → 重复**。两家实现：① **Anthropic computer use**（2024-10）：Claude 看屏幕截图，输出结构化 tool-use 指令（`computer` tool type），任务"需要几十、有时甚至成百步"，官方称实验性、笨拙易错；② **OpenAI Operator / CUA**（2025-01→07）：GPT-4o 视觉 + RL 推理，底层同样走 screenshot→CoT→action 循环，ChatGPT agent 整合进虚拟电脑（visual browser + terminal + API）。安全设计的共同点：输密码/支付需用户接管（非全自动）、拒绝高风险任务（银行转账）、sandbox 隔离。面试时强调两个点：① 视觉理解是关键瓶颈（截图分辨率/延迟/元素定位不准）；② 步数多→误差累积→需要每步环境 ground truth 验证。

> **追问：** 两者的关键工程区别是什么？
> Anthropic 给的是 API tool（开发者集成到自己的系统），OpenAI 给的是产品（ChatGPT 内直接操作虚拟电脑）——前者灵活但需自己搭环境，后者开箱即用但定制受限。

</details>

### L2 进阶

<details>
<summary>Q3: 长程 agent 怎么不爆 context（compaction / 文件当记忆 / subagent）？</summary>

**答：** Claude Agent SDK 提供了三种组合策略：① **compaction**：自动摘要旧消息——prompt 过长时把早期对话压缩成简短摘要释放 token，关键信息不丢；② **文件系统当记忆**：不把所有上下文放 prompt，而是把长输出/中间结果/历史写入文件系统→需要时 grep/tail 按需取——本质上用磁盘当"外部记忆"；③ **subagents**：把子任务分给独立 agent（隔离上下文、并行执行），只回传摘要给主 agent——防止子任务细节污染主上下文。实践中三者梯次组合：compaction 处理对话历史、文件系统处理长持久记忆、subagents 处理并行独立子任务。额外手段：结构化输出（限制 token 浪费）、step counter（强制截断）。

> **追问：** compaction 会不会丢失关键信息？
> 有风险——早期对话里被压缩掉的细节可能后来变得关键。缓解：保留引用的原始消息指针（"需要细节时回查原文"）、对关键决策点设置不可压缩标记。

</details>

<details>
<summary>Q4: 长程任务奖励稀疏怎么办？为什么要难度带 / 动态采样？</summary>

**答：** 长程 agent 任务通常只有**最终成功/失败**的二元信号——中间 100 步没有任何奖励反馈。难度带设计的核心洞察：**有效 RL 信号只存在于"模型刚好会但不确定"的中间难度带**。两端都没梯度：太难→总是失败→全错组优势为零；太易→总是成功→全对组优势为零。解决方案：**动态采样过滤 pass-rate=0/1 的 prompt**（MiMo 做法：评估每个 prompt 上的当前 pass-rate→丢弃全对/全错→保留"刚好会"的难度带 + 10% 简单题池防后期不稳定）。Self-Play SWE-RL 更进一步：用**分段奖励函数**显式惩罚难度两端（$s=0$ 太难没人解出、$s=1$ 太易全解出）→ 只奖 $0<s<1$ 的中间难度。两者动机一致：**难度自适应课程——把训练信号集中到模型"够得着但还没掌握"的区域**。

> **追问：** 训练后期难度收敛（大部分 prompt 要么太易要么太难）怎么办？
> 扩充 prompt 库、注入新题；同时保留少量全对/全错 prompt 作为"锚点"——不过滤干净，维持极小比例的简单/困难样本防分布偏斜。

</details>

<details>
<summary>Q5: agent 怎么自我验证（rules / visual / LLM-as-judge），各自代价？</summary>

**答：** Claude Agent SDK 提供了三种验证方式，从廉价到昂贵梯次使用：① **rules-based**（linter/单测/正则/状态检查）：最精确、零延迟、可自动化——但只适用于可规则化属性（代码对错、格式合规、页面 URL 正确）；② **visual**（截图验证）：适合 UI agent——用视觉模型/像素比对判断"界面是否达到预期状态"——比 rules 灵活但需要额外的视觉理解能力，且截图/处理开销不小；③ **LLM-as-judge**：最灵活——可评估模糊标准（"回答是否 helpful""交互是否自然"）——但代价是：额外 LLM 调用的延迟和成本 + judge 自身偏差和错误（位置/冗长/自我偏好→见 [eval-and-judges §2](cheatsheet-eval-and-judges.html)）。实践中从廉价到昂贵梯次使用：先 rules→不够再看 visual→最后才上 LLM-as-judge。

> **追问：** 什么场景下必须上 LLM-as-judge？
> 开放性任务的标准本身就是模糊的——如"对话是否 helpful""生成的邮件是否得体"——这些无法规则化，只能靠人类偏好或 LLM judge 近似。代价高但无替代方案。

</details>

### L3 深挖

<details>
<summary>Q6: 设计一个长程 coding agent 的奖励：怎么防 reward hacking 与退化？</summary>

**答：** 核心挑战：① 只有最终 test pass/fail 二元信号→中间 N 步无反馈；② agent 可能 reward hack（重复执行空操作消耗步数、写死答案绕过真实推理）；③ 太难/太易 prompt 梯度为零。设计方案：**主奖励** = 最终测试通过率（0/1 二元），叠加**步数效率惩罚**（每多一步扣 λ，防无限循环）；**过程奖励** = 对关键中间产物（如"定位到 bug 所在文件"）给小奖励，缓解中间步无信号问题——但需注意"过程奖励被 hack"的风险（模型学会做"看起来像在找 bug"的动作但不真正找）；**难度带过滤** = 丢弃全对/全错 prompt + 保留 10% 简单题锚点。防 reward hacking 的关键：coding agent 选**可验证域**——代码单测 ≈ ground truth，几乎不可被神经 RM 骗——这就是为什么 coding agent RL 主线和 GRPO/RLVR 走同一条路（规则化验证器）。

> **追问：** 过程奖励（intermediate reward）会不会被 hack？
> 会——这就是为什么 R1 主线选 outcome reward（只看最终答案对错）而非过程奖励。coding agent 场景下，中间验证信号（如跑 linter、检查文件是否存在）比 AI 模型的 PRM 更可靠——优先用 rule-based 中间信号。

</details>

<details>
<summary>Q7: web / long-horizon agent 训练的三大挑战各怎么缓解？</summary>

**答：** 「三大挑战」框架来自 WebRL（Qi et al. 2024）：① **训练任务稀缺**：真实 web 任务多样但标注贵。缓解：自动生成任务模板（从网站结构衍生新任务）+ 从失败轨迹反向构造新训练任务（"agent 在这步走错了→创造一个专项训练题"）；② **反馈信号稀疏**：网页任务通常只有最终目标完成度的二元信号。缓解：对中间状态变化（页面 URL 跳转/DOM 变化/表单填写进度）给 shaping reward + 对关键里程碑设 checkpoint 奖励；③ **策略分布漂移**：agent 的行为改变了其后续观测的分布——新策略产生不同的浏览路径，同一 prompt 在不同策略下的"好操作"定义不同。缓解：online RL（每轮更新策略后立即用新策略采样，不用旧数据）+ 先 SFT 固化基础浏览行为再 RL 微调。这三个挑战相互耦合：任务稀缺限制了训练规模→限制了应对分布漂移的能力→又反过来加剧了稀疏反馈的难度。

> **追问：** 为什么 web agent 比 math agent 难这么多？
> Math agent 的环境是静态的（一个数学题不会因为模型尝试次数改变），web agent 的环境是交互式的——agent 的行为改变了页面状态，导致分布漂移。加上 web 缺乏干净的"正确答案"——成功模棱两可（"订到机票了但价格不最优"算成功吗？），奖励更难定义。

</details>

<details>
<summary>Q8: 自进化 / self-play 训练的失败模式是什么？为什么生产里还不敢全自动？</summary>

**答：** 四种核心失败模式：① **模式崩塌（mode collapse）**：模型偏好自己风格→正反馈循环→输出多样性持续降低→退化为狭窄的"自我偏好"策略；② **奖励崩塌（reward collapse）**：self-judgment 质量逐轮退化（模型越来越"宽容"自己的输出，打分越来越高但实际质量不变甚至下降）→越训练越差；③ **难度退化（difficulty collapse）**：自生成任务趋向简单——模型倾向造自己会做的题（因为这样做有正奖励）→训练集难度持续下降→训练失去意义；④ **分布外漂移（distributional drift）**：自生成数据逐渐偏离真实任务分布→在真实任务上效果反降（"在自己的世界里越练越好，在真实世界里越练越差"）。根本原因：**缺乏外部 ground truth 锚定**——没有客观的、独立于模型的验证信号来校准自进化方向。可验证域（code 单测/math exact-match）可以提供这样的锚定，但开放域几乎没有。生产里不敢全自动就是因为：没有外部锚定的自进化系统，短期看起来好，长期必然崩。

> **追问：** 有什么正在探索的"半自动"折中方案？
> ① 周期性引入人类评估校准方向（少量人工标注 re-anchor 自进化方向）；② 用可验证域做锚点任务、混合开放域任务训练（代码/数学的 ground truth 信号间接约束开放域行为）；③ 多 agent 互相 critique（一个 agent 生成、另一个 agent 批判——不依赖单一 self-judgment）。三种都有人试，但无公认成熟方案——都是研究阶段。

</details>

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
