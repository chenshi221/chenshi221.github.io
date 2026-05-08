---
type: topic
status: active
created: '2026-04-30'
updated: '2026-05-07'
aliases:
  - Agent 主题
tags:
  - agent
  - LLM
  - tool-use
sources:
  - Wiki/Sources/LLM Agent 综述 2023
  - Wiki/Sources/LLM Agent 综述 2024
  - Wiki/Sources/Agent AI Survey 2024
  - Wiki/Sources/Agent Banana
  - Wiki/Sources/AHE Agentic Harness Engineering
  - Wiki/Sources/Self-Evolving Agents 综述
---
# LLM Agent 与工具使用

## 概述

LLM Agent 是指以大语言模型为核心控制器，通过感知环境、规划行动、使用工具来完成复杂任务的自主系统。本主题涵盖 Agent 架构理论、Agent 在图像编辑中的应用，以及与多模态视觉生成的交叉。

## 核心来源

- [[Wiki/Sources/LLM Agent 综述 2023]]：LLM Agent 首篇全面综述（复旦 NLP），Brain-Perception-Action 三模块框架 + Agent 社会仿真与人格涌现，2023。
- [[Wiki/Sources/LLM Agent 综述 2024]]：LLM Agent 统一框架（Profile-Memory-Planning-Action），中国人民大学 2024 综述。
- [[Wiki/Sources/Agent AI Survey 2024]]：多模态 Agent AI 综述，强调跨现实训练和 embodied AI，Stanford & Microsoft 2024。
- [[Wiki/Sources/Agent Banana]]：Agentic Planner-Executor 框架用于高保真图像编辑，Context Folding + Image Layer Decomposition。
- [[Wiki/Sources/AHE Agentic Harness Engineering]]：**可观测性驱动的编码 Agent 线束自动演化**，三大支柱实现闭环自改进，10 轮提升 pass@1 7.3pp，跨 benchmark 和模型家族可迁移。
- [[Wiki/Sources/Self-Evolving Agents 综述]]：**自演化智能体系统性综述**，提出模型中心→环境中心→模型-环境协同演化三大范式分类体系，强调环境应是与 Agent 共同演化的可优化伙伴。

## 关键概念

- [[Wiki/Concepts/LLM Agent 架构]]：Agent 的四大核心模块与设计空间。
- [[Wiki/Concepts/Self-Evolving Agents 自演化智能体]]：**以最小人类监督自主协调改进循环的 Agent 范式**，强自主性+主动探索两大性质，三大演化方向（模型中心/环境中心/协同演化）。
- [[Wiki/Concepts/Harness 编码 Agent 线束]]：**编码 Agent 中所有模型外部可编辑工程组件的统称**，7 种组件类型的解耦与演化。
- [[Wiki/Concepts/Agentic Harness Engineering (AHE)]]：**可观测性驱动的 harness 自动演化闭环**，三大支柱（组件/经验/决策可观测性）+ 自归因契约。
- Agent 能力获取：微调（Fine-tuning）vs 非微调（Prompt Engineering、Mechanism Engineering）。
- 多 Agent 协作：MetaGPT、ChatDev 等多角色分工模式。
- Agentic 图像编辑：将 Agent 的规划-执行-自检循环引入图像编辑任务。

## 与现有 Wiki 的关系

- 与 [[Wiki/Topics/扩散模型图像编辑与生成]] 交叉于 Agent Banana：扩散模型提供编辑能力，Agent 框架提供规划和工具使用能力。
- 与 [[Wiki/Entities/Mind-Brush]]、[[Wiki/Entities/VisionCreator]] 同属 Agentic 生成/编辑方向。

## 开放问题

- 如何量化 Agent 规划的质量和可靠性？
- Agent 工具调用（Tool Use）的泛化性：从 API 到视觉编辑工具的迁移？
- 长期任务中的累积误差如何检测和纠正？
- **Harness 自动化演化**：AHE 解决了 harness 可观测性瓶颈，但回归盲区（regression blindness）和组件非加性交互仍是开放挑战。
- **Harness + 模型联合优化**：AHE 的"模型外优化"与 RL 训练的"模型内优化"能否协同？能否形成联合闭环？
- **Harness 的知识折旧**：随着基座模型升级，已演化的 harness 如何"保鲜"？何时需要重新演化？
