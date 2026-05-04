---
type: concept
status: active
created: '2026-05-03'
updated: '2026-05-03'
aliases:
  - AHE
  - Agentic Harness Engineering
tags:
  - agent
  - coding-agent
  - harness
  - self-evolution
  - observability
sources:
  - '[[Wiki/Sources/AHE Agentic Harness Engineering]]'
confidence: high
---
# Agentic Harness Engineering (AHE)

## 定义

Agentic Harness Engineering (AHE) 是一种**可观测性驱动的闭环系统**，用另一个 agent（Evolve Agent）自动演化编码 agent 的 harness（线束/脚手架），保持基座模型固定不变，仅编辑模型外部的工程组件。

核心主张：harness 优化的瓶颈是**可观测性**（observability），而非 agent 能力——一旦演化 agent 获得结构化上下文和清晰的动作空间，就能可靠收敛到更好的 harness 设计。

## AHE 的三大可观测性支柱

### 1. 组件可观测性（Component Observability）

通过 **NexAU** 解耦 harness 基板实现：
- 7 种可编辑组件暴露为**独立文件**：system prompt、tool description、tool implementation、middleware、skill、sub-agent configuration、long-term memory
- 每个故障模式仅映射到**单一组件类**，编辑即 git commit
- 种子 harness 故意**最小化**（仅 bash），迫使每个新增组件通过实测 rollouts 证明价值
- 每次 pass-rate 变化可定位到**一个文件**，而非散落在数百行非结构化 prompt 中

### 2. 经验可观测性（Experience Observability）

通过 **Agent Debugger** 实现：
- 将数百万 token 的原始 rollout 轨迹蒸馏为**分层证据语料**
- 轨迹框架化为可导航文件环境，debugger agent 用 shell/grep 等通用工具分析
- 输出两层报告：
  - **Per-task analysis**：每个任务的 pass/fail 状态 + 根因分析
  - **Benchmark-level overview**：聚合所有报告为单文档入口
- 原始轨迹保留供验证，支持**渐进式信息披露**（progressive disclosure）

### 3. 决策可观测性（Decision Observability）

通过 **Evolve Agent + Change Manifest** 实现：
- 每条编辑附**自声明预测**：明确列出预期修复的任务和风险回归的任务
- 下一轮将预测与观察到的 task-level delta 交叉比对 → **per-edit verdict**
- 验证失败的编辑在**文件粒度回滚**
- 两个约束保证可控性：
  - **Controllability**：Evolve Agent 只能写 harness workspace，其余只读
  - **Evidence-driven**：每条编辑必须引用证据、根因、修复方案和预测影响

## AHE 外循环

```
每轮迭代 = Rollout → Clean → Attribute(验证+回滚) → AgentDebugger(蒸馏) → Evolve(编辑+manifest) → Commit
```

关键设计：
- k ≥ 2 rollouts/任务，使每个任务携带 pass-rate 信号
- Attribution 在蒸馏**之前**运行，将上轮 manifest 作为契约验证而非理由陈述
- 首轮并行运行 explore agent 从 NexAU 源码和公开参考中 seed 少量可复用 skills

## 核心发现

### 实验数据

- **Terminal-Bench 2**：10 轮提升 69.7% → 77.0%，超越人工 Codex-CLI (71.9%)
- **SWE-bench-verified 迁移**：综合成功率最高 (75.6%)，比种子省 12% token
- **跨模型迁移**：全部 5 个替代模型正增益 (+2.3 ~ +10.1 pp)，跨家族 > 家族内

### 增益来源（组件消融）

| 单独替换 | Δ pass@1 |
|---------|----------|
| + AHE memory only | **+5.6 pp** |
| + AHE tool only | +3.3 pp |
| + AHE middleware only | +2.2 pp |
| + AHE system_prompt only | **-2.3 pp** |

→ **增益存在于 tools、middleware、long-term memory 中，而非 system prompt**
→ 事实性 harness 结构可迁移，散文级策略不可

### 关键局限

1. **组件非加性交互**：三个正组件单独增益之和 (+11.1 pp) > 全量增益 (+7.3 pp)，因为 memory/middleware/prompt 都趋向同一种 closure-style verification，堆叠产生冗余
2. **回归盲区（Regression Blindness）**：agent 修复预测远好于随机（~5x），但回归预测仅略好于随机（~2x）——能预测什么会变好，无法预测什么会变差
3. **操作点耦合**：超时和步数预算在 GPT-5.4 high 上拟合，跨推理层级增益非单调

## 与其他概念的关系

- **vs 模型内 RL 训练**：AHE 是"模型外优化"的互补轴——RL 训练改变模型内部策略，AHE 改变模型外部工程组件。两者可协同
- **vs Prompt-only 自演化**（ACE, TF-GRPO）：prompt-only 方法无法触及 AHE 增益所在的组件层（tools/middleware/memory），且 prompt 中的策略文本在跨任务时增加成本而不重塑底层策略
- **vs 人工 harness 工程**：AHE 将 harness 设计从"手工工艺"升级为"可自动化的学习问题"，关键创新在于通过可观测性设计解决了 attribution 瓶颈

## 深度分析

### AHE 的范式意义

AHE 提出了一个被低估的洞察：**在 coding agent 系统中，模型能力和 harness 质量是两个独立且互补的杠杆**。随着基座模型快速迭代（DeepSeek、Kimi、Qwen 等每月更新），手工调整 harness 无法跟上节奏。AHE 提供的自动化方案，让 harness 成为一个"可学习的适应表面"（learnable adaptation surface）。

### 为什么可观测性是瓶颈而非 agent 能力？

AHE 的设计哲学是"给 agent 一个清晰的地图，它就能导航"。这与其他 agent 工作的一个关键区别：大多数 agent 研究关注"让 agent 更强"（更好的推理、更长的上下文、更多的工具），而 AHE 关注"让 agent 看到自己在做什么"。三个支柱各自解决一种"看不见"的问题——看不见组件边界、看不见轨迹信号、看不见编辑后果。

### 回归盲区的深层含义

回归盲区不是工程疏忽，而是**agent 自我改进的根本性不对称**：证明一个改动有帮助相对容易（找到它修复的 case），但预测它会破坏什么需要全局推理（枚举所有可能受影响的 case）。这种不对称可能不是 AHE 特有的，而是所有自我改进系统的固有挑战。关闭这个 gap 是整个 self-evolution 方向最清晰的下一步。

### Harness 演化的"知识折旧"

论文暗示了一个未充分探讨的问题：随着基座模型升级，AHE 演化的 harness 是否会"贬值"？跨模型迁移的正增益给出了部分积极信号，但增益大小与模型饱和度的反比关系（越弱的模型获益越多）暗示：当模型变得足够强时，它可以通过 prompt 中的推理自行补偿 harness 的不足。这意味着 harness 演化的价值可能存在一个"能力窗口"——模型太弱时 harness 也救不了，模型太强时 harness 可有可无，中间地带是 harness 工程的主战场。

## 来源

- [[Wiki/Sources/AHE Agentic Harness Engineering]]
