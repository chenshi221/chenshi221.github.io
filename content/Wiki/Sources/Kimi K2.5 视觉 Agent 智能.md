---
type: source
status: processed
source_file: '[[Clippings/Kimi K2.5 Visual Agentic Intelligence.md]]'
title: 'Kimi K2.5: Visual Agentic Intelligence'
site: Kimi Blog
author: Moonshot AI
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - 多模态
  - Agent
  - Kimi
  - 视觉
  - RL
aliases:
  - Kimi K2.5
---
# Kimi K2.5 视觉 Agent 智能

## 核心结论

- [[Kimi K2.5]] 是联合文本-视觉优化的多模态 Agent 模型，实现文本和视觉的端到端联合 RL 训练。
- 提出 zero-vision SFT 策略：在 SFT 阶段不提供视觉输入，让模型先学会纯文本推理，再引入视觉——防止模型在视觉训练中退化。
- Agent Swarm/PARL 架构：多 Agent 并行执行 + 反思循环，支持复杂多步 agentic 任务。
- MoonViT-3D：原生分辨率视觉编码器，支持 NaViT 动态打包，提升多分辨率图像处理效率。
- Toggle token-efficient RL：选择性激活视觉 tokens 进行 RL 训练，大幅降低训练开销。
- Decoupled Encoder Process (DEP)：编码器与解码器分离处理，提升推理效率。

## 关键事实

- 联合文本-视觉优化（early fusion, low vision ratio）。
- Zero-vision SFT 防止视觉训练导致文本能力退化。
- Agent Swarm/PARL：多 Agent 并行 + 反思机制。
- MoonViT-3D + NaViT packing，原生分辨率处理。
- Toggle RL：选择关键视觉 tokens 进行 RL 训练，降低计算开销。
- DEP 架构解耦编码和解码过程。

## 方法或论证路径

- 联合 RL 训练（文本+视觉）需要特殊处理以避免跨模态干扰，zero-vision SFT 是关键步骤。
- Agent Swarm 通过多 Agent 并行和反思循环解决复杂多步任务。
- Toggle 方法在视觉 RL 中实现 token 级别的效率优化。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/国产大模型演进]]、[[Wiki/Concepts/多模态 Agent]]
- 与 [[Wiki/Sources/Kimi-VL 技术报告]] 形成完整的多模态能力线：Kimi-VL 负责基础 VLM，K2.5 负责 Agent 化。
- 与 [[Wiki/Sources/Kimi K2 开放 Agent 智能]] 一脉相承：K2 是纯文本 Agent，K2.5 扩展到多模态 Agent。

## 可能的矛盾或待核实点

- Zero-vision SFT 的效果是否被后续 ablation 充分验证？
- Agent Swarm 的计算开销是否在实际部署中可控？

## 后续问题

- K2.5 的 Agent Swarm 与 DeepSeek-V3.2 的 agent 合成管线如何对比？
- MoonViT-3D 与 MoonViT（Kimi-VL）的具体架构差异是什么？
