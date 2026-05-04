---
type: concept
status: active
created: '2026-05-03'
updated: '2026-05-03'
aliases:
  - 编码 Agent 线束
  - harness engineering
tags:
  - agent
  - coding-agent
  - harness
  - engineering
sources:
  - '[[Wiki/Sources/AHE Agentic Harness Engineering]]'
  - '[[Wiki/Sources/LLM Agent 综述 2024]]'
confidence: high
---
# Harness（编码 Agent 线束）

## 定义

Harness（线束/脚手架）是**编码 agent 中所有模型外部可编辑工程组件的统称**，包括：system prompt（定义工作风格）、tools（暴露文件系统和 shell）、middleware（控制上下文、执行和恢复）、skills（可复用能力模块）、sub-agents（子任务委派）、long-term memory（经验积累）和 tool descriptions（工具接口描述）。

Harness 位于基座模型与任务环境之间，**调解模型如何感知环境并对其采取行动**，是决定长时域编码任务完成率的**一等杠杆**。

## 为什么 Harness 重要？

### 1. 独立于模型的性能杠杆

即使基座模型固定不变，harness 设计对长时域编码 benchmark 的性能有**实质性影响**。多个研究已证明：相同的基座模型 + 不同的 harness → 显著不同的 pass rate。

### 2. 模型特异性

最优 harness 是**模型特定的**（model-specific）：为一个基座模型调优的 harness 在另一个模型上通常表现不佳，需要重新适配。在基座模型快速迭代的当下，这形成了一个"适配缺口"——手工调整跟不上模型更新速度。

### 3. Harness 是可学习的外部适应表面

AHE（Agentic Harness Engineering）证明：harness 可以被视为一个**可学习的适应表面**（learnable adaptation surface）——基座模型提供推理能力，harness 编码工程经验和协调模式。两者形成互补：模型内部优化（RL 训练）vs 模型外部优化（harness 演化）。

## Harness 的 7 种可编辑组件（按 NexAU 框架）

| 组件 | 作用 | 演化难度 | AHE 增益 |
|------|------|---------|---------|
| **System Prompt** | 定义工作风格、行为规则 | 易编辑 | **负增益** (-2.3pp) |
| **Tool Implementation** | 实现 shell、文件系统等工具 | 中等 | +3.3 pp |
| **Middleware** | 控制上下文、执行流程、恢复策略 | 较难 | +2.2 pp |
| **Long-term Memory** | 存储边界案例经验、失败模式 | 难 | **+5.6 pp** |
| **Skill** | 可复用的子任务能力模块 | 中等 | - |
| **Sub-agent Config** | 子任务委派配置 | 中等 | - |
| **Tool Description** | 工具接口描述和文档 | 易编辑 | - |

## 关键洞察：为什么 system prompt 单独不工作？

AHE 的组件消融实验揭示了一个反直觉发现：**单独替换 AHE 演化的 system prompt 导致性能倒退**，尽管 system prompt 包含了经过 10 轮演化的 79 行"通用纪律"。

原因：system prompt 中的规则的**可执行性依赖于其他三个组件**（tools/middleware/memory）。例如，prompt 可能说"在结束前验证结果"，但如果没有 middleware 的 finish-hook 强制执行这个检查、或 memory 中存储的具体验证模式，这条规则只是散文，无法转化为一致的行为。

→ **事实性 harness 结构（tools/middleware/memory）可迁移，散文级策略（prompt）不可。**

## 组件间的非加性交互

AHE 的另一个核心发现：三个正组件单独增益之和 (+11.1 pp) > 全量组合增益 (+7.3 pp)。原因是组件间存在**冗余行为**：

- Memory、middleware 和 system prompt 都趋向同一种 closure-style verification（闭合验证）
- 堆叠它们导致在长时域任务中**花费多余的轮次做重复检查**
- 演化 agent 优化的是聚合指标（55 个 Medium 任务主导），收敛到 Medium-heavy 的权衡

→ **组件不是独立的乐高积木，它们之间存在竞争和冗余。**

## 与 Agent 架构的关系

Harness 是 [[Wiki/Concepts/LLM Agent 架构]] 中 Profile-Memory-Planning-Action 框架的**工程实现层**：
- Profile → System Prompt
- Memory → Long-term Memory
- Planning → （内化在模型推理中，不由 harness 直接控制）
- Action → Tools + Middleware + Skills + Sub-agents

Harness 将 Agent 架构的概念模块**实例化为可编辑的文件**，使得自动演化成为可能。

## 深度分析

### Harness 的"能力窗口"假说

跨模型迁移实验揭示了一个微妙模式：

- **弱模型获益最大**：deepseek-v4-flash +10.1 pp, qwen-3.6-plus +6.3 pp
- **强模型获益较小**：GPT-5.4 medium/xhigh 仅 +2.3 pp
- **推理层级非单调**：medium +2.3, high +7.3, xhigh +2.3（超时耦合）

可能的解释：存在一个**harness 价值的能力窗口**：
1. 模型太弱 → harness 再优化也救不了根本性的推理不足
2. 模型适中 → harness 的结构化支持弥补推理短板，增益最大
3. 模型太强 → 可以从 prompt 中自行推理出等效的协调模式，harness 边际价值递减

如果这个假说成立，它意味着 harness 工程的最大价值阶段是**模型能力快速增长但尚未饱和的"中间地带"**——恰好是当前 2025-2026 年的状态。

### Harness vs RL 训练：两个正交的优化轴

| 维度 | RL 训练（模型内） | Harness 演化（模型外） |
|------|------------------|---------------------|
| 优化目标 | 模型权重 | 工程组件（文件） |
| 可审计性 | 低（权重是黑箱） | 高（每个编辑是 git commit） |
| 可回滚性 | 困难 | 简单（文件级 revert） |
| 跨模型迁移 | 不适用 | 部分可迁移 |
| 优化成本 | 极高（GPU 集群） | 中等（API 调用 + 沙箱） |

两者是互补的：RL 训练提升模型的"原始推理能力"，harness 演化优化"如何将推理能力转化为任务完成"。

## 来源

- [[Wiki/Sources/AHE Agentic Harness Engineering]]
- [[Wiki/Sources/LLM Agent 综述 2024]]
