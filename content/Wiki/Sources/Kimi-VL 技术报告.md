---
type: source
status: processed
source_file: '[[Clippings/Kimi-VL Technical Report.md]]'
title: Kimi-VL Technical Report
site: Kimi Blog
author: Moonshot AI
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - VLM
  - 多模态
  - Kimi
  - MoE
  - MoonViT
aliases:
  - Kimi-VL
---
# Kimi-VL 技术报告

## 核心结论

- [[Kimi-VL]] 是 2.8B 激活参数的 MoE 视觉语言模型（VLM），以极小参数量在多项 VLM benchmarks 上达到 SOTA。
- 提出 MoonViT 原生分辨率编码器，结合 NaViT packing 技术，支持原生分辨率输入而无需统一缩放。
- 支持 128K 上下文长度，可处理长文档、多页 PDF 等多帧多页视觉输入。
- 推出 Kimi-VL-Thinking 变体：在 VL 基础上叠加长链 CoT SFT + RL，实现视觉推理能力。
- 证明了小参数 MoE VLM 可以通过架构创新和训练策略达到甚至超越大模型的性能。

## 关键事实

- 架构：2.8B 激活参数的 MoE VLM。
- 视觉编码器：MoonViT 原生分辨率 + NaViT packing。
- 上下文：128K tokens。
- Thinking 变体：long-CoT SFT + RL，带有思维链推理能力。
- 在多模态 benchmarks（MMBench、MME、DocVQA 等）达到 SOTA。

## 方法或论证路径

- MoonViT 原生分辨率避免了图像缩放带来的信息损失，NaViT 将不同尺寸的图像打包为统一 batch。
- 小 MoE 设计在计算效率和性能之间取得平衡：2.8B 激活参数即可匹配更大模型。
- Thinking 变体验证了推理能力训练范式（SFT + RL）在多模态场景下同样有效。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/国产大模型演进]]、[[Wiki/Concepts/MoE 混合专家模型]]、[[Wiki/Concepts/多模态 Agent]]
- 与 [[Wiki/Sources/Kimi K2.5 视觉 Agent 智能]] 形成多模态能力递进。
- 与 [[Wiki/Sources/DeepSeek-VL技术报告]] 对照：均探索小参数高性能 VLM 路线。

## 可能的矛盾或待核实点

- 2.8B 激活参数是否在所有场景下都能超过大模型，特别是在复杂视觉推理任务中？
- NaViT packing 的效率在实践中是否受 GPU 内存布局影响？

## 后续问题

- Kimi-VL-Thinking 与 K2.5 的关系：是否共享视觉 backbone？
- MoonViT 与 ViT 标准版本的具体架构差异？
