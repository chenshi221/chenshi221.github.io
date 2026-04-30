---
type: source
status: processed
source_file: >-
  [[Clippings/DeepSeek-Coder-V2 Breaking the Barrier of Closed-Source Models in
  Code Intelligence]]
title: >-
  DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code
  Intelligence
site: arxiv
author: DeepSeek-AI
published: 2024-06
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - DeepSeek-Coder-V2
  - 代码生成
  - MoE
  - 338语言
  - GPT-4级
aliases:
  - DeepSeek-Coder-V2
---
# DeepSeek-Coder-V2 代码智能

## 核心结论

DeepSeek-Coder-V2 是在 DeepSeek-V2 checkpoint 基础上额外预训练 6T tokens 的 MoE 代码语言模型，在代码和数学任务上达到 GPT4-Turbo 级别性能。开源模型中首次对标闭源代码模型（Claude 3 Opus、Gemini 1.5 Pro）。编程语言从 86 扩展到 338 种，上下文从 16K 扩展到 128K。

## 关键事实

- 作者：DeepSeek-AI（Qihao Zhu、Daya Guo 等核心贡献者），2024 年 6 月
- 基础：DeepSeek-V2 中间 checkpoint + 6T tokens 继续预训练
- 架构：与 V2 相同的 MoE 架构（MLA + DeepSeekMoE），236B 总/21B 激活
- 编程语言：从 DeepSeek-Coder-33B 的 86 种扩展到 338 种
- 增强能力：编码 + 数学推理 + 通用语言（同时保持 V2 基础能力）
- 开源权重完整发布

## 方法或论证路径

- 数据混合：高质量代码语料（多语言源码） + 数学推理数据 + 通用数据
- 继续预训练：在 V2 基础上进行额外的代码/数学密集型预训练
- 评估：HumanEval、MBPP、LiveCodeBench（代码）、GSM8K、MATH（数学）上对标 GPT4-Turbo
- 消融：展示代码数据比例对通用能力的 trade-off

## 与现有 Wiki 的关系

- 关联：[[Wiki/Entities/DeepSeek 系列模型]]——补充了代码特化分支
- 关联：[[Wiki/Concepts/MoE 混合专家模型]]
- 继承：V2 的技术栈在代码领域应用

## 后续问题

- 338 种编程语言的覆盖面有多广？低频语言的训练充分性如何？
- Coder-V2 与通用 V2 之间的蒸馏或知识共享机制
