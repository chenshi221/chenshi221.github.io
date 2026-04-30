---
type: source
status: processed
source_file: >-
  [[Clippings/DeepSeek LLM Scaling Open-Source Language Models with
  Longtermism]]
title: 'DeepSeek LLM: Scaling Open-Source Language Models with Longtermism'
site: arXiv
author: DeepSeek-AI
published: 2024-01
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - deepseek
  - scaling-laws
  - pretraining
  - open-source
aliases: []
---

# DeepSeek LLM: 开源语言模型与长期主义

## 核心结论

- 这是 DeepSeek 系列的第一篇公开论文，介绍了 DeepSeek LLM 7B 和 67B 模型，标志着 DeepSeek 正式进入开源 LLM 领域。
- 系统性研究了 scaling laws，提出使用非嵌入 FLOPs/token（M）替代传统的参数量 N 来表示模型规模，从而获得更精确的扩展规律。
- 发现数据质量显著影响最优模型/数据分配策略：数据质量越高，越应该将更多计算预算分配给模型规模扩展。
- DeepSeek LLM 67B 在代码、数学和推理方面超越 LLaMA-2 70B，Chat 版本在开放式评估中优于 GPT-3.5。

## 关键事实

- 预训练数据：2 万亿 token，中英文为主。
- 模型架构：大体沿用 LLaMA 设计（Pre-Norm RMSNorm、SwiGLU、RoPE），70B 模型使用 GQA。
- 优化器：AdamW，采用多步学习率调度器（替代余弦调度器）以便于持续训练。
- SFT 使用超过 100 万实例，DPO 用于提升对话性能。
- 训练框架：HAI-LLM，集成数据并行、张量并行、序列并行、流水线并行和 FlashAttention。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Entities/DeepSeek 系列模型]]、[[Wiki/Topics/国产大模型演进]]
- 这是 DeepSeek 系列的开端，后续 V2、V3、V3.2、R1 均在此基础上演进。
- 其中的 scaling laws 研究为后续模型的参数/数据分配提供了理论基础。

## 可能的矛盾或待核实点

- 论文中不同数据集上拟合的 scaling laws 存在显著差异，scaling laws 的跨数据集泛化性仍需验证。

## 后续问题

- scaling laws 在多模态和 MoE 架构下是否仍然适用？
