---
type: source
status: processed
source_file: >-
  [[Clippings/Agent Banana High-Fidelity Image Editing with Agentic Thinking and
  Tooling]]
title: 'Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling'
site: arXiv 2602.09084
author: 'Ruijie Ye, Jiayi Zhang et al. (TAMU, Adobe, Meta)'
published: '2024'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - agent
  - image-editing
  - tool-use
  - high-resolution
aliases:
  - Agent Banana 论文
---
# Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling

## 核心结论

- 提出 Agent Banana，一个分层 Agentic Planner-Executor 框架，用于高保真、目标感知的图像编辑。
- 引入两个核心机制：**Context Folding**（压缩长交互历史为结构化记忆）和 **Image Layer Decomposition**（局部图层编辑，保持非目标区域不变）。
- 支持原生 4K 分辨率编辑，避免下采样带来的细节损失。

## 关键事实

- 来源：TAMU、Brown、UW-Madison、UCSD、USC、Adobe Research、Meta AI、Princeton 等联合，2024。
- 架构：Planner（全局意图理解、任务分解）+ Executor（原子编辑操作、工具调用、自检纠错）。
- Context Folding 三级抽象：Asset Level（ImageContext）→ Execution Level（ToolContext）→ Planning Level（ActionContext）。
- Image Layer Decomposition：基于动态物体感知 mask，在局部 patch 上编辑后高斯融合回原图。
- 五种原子操作：replace、remove、add、adjust、undo。
- 配套 benchmark HDD-Bench：多轮、4K 原生分辨率、stepwise 可验证目标。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/扩散模型图像编辑与生成]]
- 关联：[[Wiki/Entities/Agent Banana]]
- 关联：[[Wiki/Topics/LLM Agent 与工具使用]]
- 编辑能力上代表 Agentic 范式对扩散模型编辑的增强。

## 后续问题

- Context Folding 的压缩有没有信息损失？
- ILD 机制对极小物体的编辑精度如何？
