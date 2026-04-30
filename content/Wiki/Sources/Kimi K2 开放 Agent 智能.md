---
type: source
status: processed
source_file: '[[Clippings/Kimi K2 Open Agentic Intelligence.md]]'
title: 'Kimi K2: Open Agentic Intelligence'
site: Kimi Blog
author: Moonshot AI
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - MoE
  - Agent
  - Kimi
  - RLVR
  - sparsity
aliases:
  - Kimi K2
---
# Kimi K2 开放 Agent 智能

## 核心结论

- [[Kimi K2]] 是 1.04T 参数的 MoE 模型（32B 激活），核心目标是打造开放的 Agentic 智能。
- 提出 MuonClip 优化器（Muon + QK-Clip），在 MoE 大规模训练中稳定且高效。
- 提出 sparsity scaling law：在给定 FLOPs 预算下，稀疏度（激活参数/总参数比）存在最优值，K2 的最优稀疏度约为 4.8%。
- 大规模 Agent 数据合成管线：tool specs -> agents -> tasks -> trajectories，自动化生成多样化的 Agent 训练数据。
- 奖励信号采用 RLVR（可验证奖励）+ self-critique rubric reward，无需人工标注即可进行大规模 RL 训练。
- 预训练数据 15.5T tokens，在多个 Agent、推理、coding benchmarks 上达到 SOTA。

## 关键事实

- MoE 架构：1.04T 总参数，32B 激活参数，稀疏度约 3.1%。
- MuonClip = Muon 优化器 + QK-Clip 梯度裁剪，专为 MoE 大规模训练设计。
- Sparsity scaling law：稀疏度 48（即 4.8% 激活比例）为最优。
- Agent 合成管线四步：设计 tool specs -> 生成 agent 定义 -> 生成任务 -> 生成交互轨迹。
- RLVR + self-critique 奖励：基于可验证结果（代码执行、数学结果）和模型自评。
- 训练数据 15.5T tokens。

## 方法或论证路径

- MuonClip 解决了 MoE 训练中常见的 loss spike 问题，通过 QK-Clip 约束注意力矩阵梯度。
- Sparsity scaling law 为 MoE 架构设计提供了理论指导：并非越稀疏越好。
- Agent 数据合成摆脱了对人工标注的依赖，可实现规模化 Agent 训练。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/国产大模型演进]]、[[Wiki/Concepts/MoE 混合专家模型]]、[[Wiki/Concepts/多模态 Agent]]
- 与 [[Wiki/Sources/DeepSeek-V3.2 开源大模型前沿]] 对照：两者均聚焦 Agent 能力，DeepSeek 用 GRPO + 环境合成，Kimi 用 RLVR + agent 数据合成。
- 支持：MoE 是大规模模型的共同选择，Agent 数据合成是前沿方向。

## 可能的矛盾或待核实点

- Sparsity scaling law 的结论（4.8%）是否普适于其他 MoE 架构？
- Agent 合成数据的质量是否足以替代人工标注？有无系统性偏见？

## 后续问题

- MuonClip 与 DeepSeek 的 auxiliary-loss-free load balancing 如何比较？
- 多 Agent 协作（Agent Swarm）在 K2 上是否有预留接口？
