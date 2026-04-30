---
type: source
status: processed
site: arXiv
source_file: '[[Clippings/The Llama 3 Herd of Models.md]]'
title: The Llama 3 Herd of Models
author: Llama Team (Meta AI)
published: '2024'
processed: '2026-04-30'
tags:
  - Llama-3
  - open-source
  - scaling
  - DPO
  - Meta
---
# The Llama 3 Herd of Models

## 核心结论

- 发布 **Llama 3** 系列（8B、70B、405B），是 Meta 迄今为止规模最大、能力最强的基础语言模型，原生支持多语言、代码、推理和工具使用。
- 最大模型是 405B 的 dense Transformer，128K token 上下文窗口，在与 GPT-4 等领先模型的竞争中表现相当。
- 提出三个关键杠杆：数据（预训练数据 15T token vs Llama 2 的 1.8T）、规模（计算量是 Llama 2 最大的 50 倍）、管理复杂度（选择简单架构而非 MoE）。
- 小模型（8B、70B）被系统性过度训练（train beyond compute-optimal），以在推理成本范围内最大化性能。

## 关键方法或创新点

- **预训练**：15.6T 多语言 token，使用 dense Transformer + RoPE + Grouped Query Attention (GQA)。
- **后训练策略**：SFT + Rejection Sampling + **DPO**（Direct Preference Optimization），故意避开更复杂的 RLHF 算法以保持稳定性和可扩展性。
- **多阶段训练**：依次训练代码、推理、长上下文等能力。
- **工具使用**：内置支持搜索、代码执行等工具调用。
- 通过组合方式（compositional approach）将图像、视频、语音能力集成到 Llama 3 中。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/大语言模型基础]]、[[Wiki/Concepts/Scaling Laws]]、[[Wiki/Concepts/RoPE 旋转位置编码]]
- Llama 3 是 [[Wiki/Sources/LLaMA|LLaMA]] 和 Llama 2 的直接进化，将开源 LLM 能力推至 GPT-4 级别。
- 小模型过度训练策略与 [[Wiki/Sources/Chinchilla 缩放定律|Chinchilla]] 形成有趣对照：计算最优并非总是目的，推理效率同样重要。
- DPO 替代 RLHF 是重要的方法论创新。

## 局限或注意事项

- 405B Dense 模型的推理成本仍然很高。
- 多模态能力通过组合方式接入（而非端到端统一训练），可能不如 GPT-4o 的全模态方法高效。
- 论文未充分讨论训练数据的版权和隐私问题。
