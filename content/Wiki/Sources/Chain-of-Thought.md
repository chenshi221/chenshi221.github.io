---
type: source
status: processed
source_file: >-
  [[Clippings/Chain-of-Thought Prompting Elicits Reasoning in Large Language
  Models.md]]
title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
site: arXiv
author: Jason Wei et al. (Google Research)
published: 2022-01
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - chain-of-thought
  - prompting
  - reasoning
  - emergent-ability
aliases:
  - CoT
---
# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

## 核心结论

- 提出了 **Chain-of-Thought (CoT) 思维链提示**：通过在小样本提示（few-shot prompting）中提供中间推理步骤示例，引导大语言模型生成逐步推理过程。
- CoT 推理能力在**足够大的模型**（约 100B 参数以上）中自然涌现，小模型上 CoT 效果甚微甚至有害。
- 仅 8 个 CoT 示例提示 PaLM 540B，即在数学推理 benchmark GSM8K 上达到 SOTA，超越当时微调过的 GPT-3+验证器。

## 关键事实

- 作者：Jason Wei, Xuezhi Wang, Dale Schuurmans 等（Google Research, Brain Team）。
- 发表于 NeurIPS 2022（arXiv:2201.11903）。
- 验证了三类推理任务：算术推理（GSM8K 等）、常识推理（CSQA、StrategyQA）、符号推理（last letter concatenation、coin flip）。
- 关键发现：CoT 是一种**涌现能力**（emergent ability），只在参数量突破某个阈值后才显著提升。
- PaLM 540B + CoT 在 GSM8K 上达 58.0%（此前最佳为 55% 的微调 GPT-3 + 验证器）。

## 方法亮点

- 方法极其简单：无需微调、无需模型修改、无需额外训练，仅靠少量 prompt 示例即可显著提升复杂推理性能。
- CoT 的本质是让模型的"内部思考过程"外显化，通过逐步推理降低单步推理的问题复杂度。
- 开创了 prompting-based reasoning 范式，成为后续 ToT、GoT、self-consistency 等方法的基础。

## 与现有 Wiki 的关系

- 是 [[Wiki/Topics/推理增强方法]] 的启源节点。
- 关联概念：[[Wiki/Concepts/Chain-of-Thought 思维链]]。
- 直接启发了 [[Wiki/Sources/Tree of Thoughts]] 和 [[Wiki/Sources/Graph of Thoughts]]。

## 后续问题

- CoT 在视觉推理、多模态推理中的适用性如何？
- CoT 是否会导致幻觉推理（reasoning to wrong answer with convincing steps）？
