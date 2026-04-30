---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - AgentBanana
tags:
  - agent
  - image-editing
  - model
sources:
  - Wiki/Sources/Agent Banana
---
# Agent Banana

## 基本信息

- 全称：Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling
- 来源：[[Wiki/Sources/Agent Banana]]
- 作者：Ruijie Ye, Jiayi Zhang et al. (TAMU, Brown, UW-Madison, UCSD, USC, Adobe Research, Meta AI, Princeton)
- 项目网站：https://agent-banana.github.io/

## 核心设计

Agent Banana 是首个将 Agentic 框架系统性地引入高保真图像编辑的工作。采用分层 Planner-Executor 架构：

- **Planner**：全局意图理解、复杂指令分解为原子子任务、进度监控和回滚决策。
- **Executor**：执行原子编辑操作、调用工具（MCP Server）、自检纠错（Quality Test）。

两大核心机制：

1. **Context Folding**：三级记忆抽象（Asset Level → Execution Level → Planning Level），压缩长交互历史避免 context overflow。
2. **Image Layer Decomposition (ILD)**：基于物体感知 mask 局部编辑 + 高斯融合，支持原生 4K 分辨率，避免全局重采样导致的细节退化。

五种原子操作：replace、remove、add、adjust、undo。

## 评测

- **HDD-Bench**：自建多轮 4K 分辨率 benchmark，11.8M 像素，stepwise 可验证目标，graph-based 状态追踪评估。
- 在 HDD-Bench 上多轮一致性（IC 0.871）和背景保真度（SSIM 0.84）最优。
- 同时兼容标准单轮编辑 benchmark。

## 与其他工作的关系

- 属于 [[Wiki/Topics/扩散模型图像编辑与生成]] 的 Agentic 范式分支。
- 与 [[Wiki/Entities/Mind-Brush]]、[[Wiki/Entities/VisionCreator]] 同为 Agent 驱动的视觉生成/编辑模型。
- 受 ReAct 范式启发，将 Thought-Action-Observation 循环引入图像编辑。
- 属于 [[Wiki/Topics/LLM Agent 与工具使用]] 在视觉编辑领域的应用实例。

## 技术定位

Agent Banana 代表了一种趋势：从"端到端黑盒编辑模型"向"可解释、可纠错、可人机协作的 Agentic 编辑系统"的转变。其 Context Folding 和 ILD 机制为长期多轮编辑任务提供了系统级的解决方案。
