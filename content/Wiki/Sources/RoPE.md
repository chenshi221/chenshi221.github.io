---
type: source
status: processed
site: arXiv
source_file: '[[Clippings/RoFormer Enhanced Transformer with Rotary Position Embedding.md]]'
title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding'
author: Jianlin Su et al. (Zhuiyi Technology)
published: '2021'
processed: '2026-04-30'
tags:
  - RoPE
  - position-encoding
  - transformer
  - NLP
---
# RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)

## 核心结论

- 提出了 **RoPE (Rotary Position Embedding)**，一种新的位置编码方法，通过旋转矩阵编码绝对位置信息，同时在自注意力计算中自然地融入相对位置依赖。
- RoPE 具有多项优良性质：序列长度灵活性、随相对距离增大而衰减的 token 间依赖性、以及支持线性自注意力配备相对位置编码。
- 在多项 NLP 基准上，使用 RoPE 的 RoFormer 模型优于使用其他位置编码（正弦编码、可学习位置嵌入等）的 Transformer。
- RoPE 被后续几乎所有主流 LLM 采用（LLaMA、Mistral、Qwen、DeepSeek 等）。

## 关键方法或创新点

- **核心公式**：对 query 和 key 向量按维度配对进行二维旋转，旋转角度随位置递增。q_m = R_m · q, k_n = R_n · k，注意力分数 q_m^T · k_n 自然包含了相对位置信息 (m - n)。
- **远程衰减**：随着相对距离增大，注意力分数自然衰减（由旋转频率控制），无需手动 mask。
- **兼容线性注意力**：可以直接集成到线性自注意力机制中保留相对位置信息。
- 与 Transformer 原始的正弦位置编码[[Wiki/Sources/Transformer]]相比，RoPE 在自注意力计算中显式地编码了相对位置关系。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/RoPE 旋转位置编码]]、[[Wiki/Entities/Transformer]]、[[Wiki/Topics/大语言模型基础]]
- RoPE 是 Transformer 位置编码演进线的关键节点：正弦编码 → 可学习嵌入 → 相对位置偏差 → RoPE。
- 几乎所有现代开源 LLM 都使用 RoPE 或其变体（YaRN 扩展等）。

## 局限或注意事项

- RoPE 的直接外推能力有限（超出训练时见过的最大长度时性能下降），需要 NTK-aware scaling 或 YaRN 等方法帮助外推。
- 旋转频率设置影响模型对不同距离范围的注意力分配——频率选择是一个超参数调优点。
- 二维旋转要求偶数维度，需按维度对分组设计。
