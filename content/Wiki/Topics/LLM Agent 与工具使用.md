---
type: topic
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Agent 主题
tags:
  - agent
  - LLM
  - tool-use
sources:
  - Wiki/Sources/LLM Agent 综述 2024
  - Wiki/Sources/Agent AI Survey 2024
  - Wiki/Sources/Agent Banana
---
# LLM Agent 与工具使用

## 概述

LLM Agent 是指以大语言模型为核心控制器，通过感知环境、规划行动、使用工具来完成复杂任务的自主系统。本主题涵盖 Agent 架构理论、Agent 在图像编辑中的应用，以及与多模态视觉生成的交叉。

## 核心来源

- [[Wiki/Sources/LLM Agent 综述 2024]]：LLM Agent 统一框架（Profile-Memory-Planning-Action），中国人民大学 2024 综述。
- [[Wiki/Sources/Agent AI Survey 2024]]：多模态 Agent AI 综述，强调跨现实训练和 embodied AI，Stanford & Microsoft 2024。
- [[Wiki/Sources/Agent Banana]]：Agentic Planner-Executor 框架用于高保真图像编辑，Context Folding + Image Layer Decomposition。

## 关键概念

- [[Wiki/Concepts/LLM Agent 架构]]：Agent 的四大核心模块与设计空间。
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
