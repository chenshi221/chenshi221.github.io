---
type: source
status: processed
site: arXiv
source_file: '[[Clippings/Attention Is All You Need.md]]'
title: Attention Is All You Need
author: Ashish Vaswani et al. (Google Brain)
published: '2017'
processed: '2026-04-30'
tags:
  - transformer
  - attention
  - NLP
  - fundamentals
---
# Attention Is All You Need

## 核心结论

- 提出了 **Transformer** 架构，完全基于注意力机制，摒弃了 RNN 和 CNN，是序列到序列建模的范式转变。
- 在 WMT 2014 英德翻译上达到 28.4 BLEU（超越所有已有模型包括集成模型），英法翻译上达到 41.8 BLEU，训练成本仅为之前最优模型的一小部分。
- 自注意力层以常数次操作连接任意位置，解决了 RNN 的顺序计算瓶颈和长距离依赖问题，训练高度可并行。
- 多头注意力允许模型在不同表示子空间中联合关注信息，且单头注意力与多头注意力质量差距达 0.9 BLEU。

## 关键方法或创新点

- **Scaled Dot-Product Attention**：对点积除以 sqrt(d_k) 防止 softmax 进入小梯度区域。
- **Multi-Head Attention**：h=8 个头并行投影，每个头 d_k=d_v=64，拼接后线性投影。
- **位置编码（Sinusoidal PE）**：使用正弦/余弦函数注入位置信息，波长形成几何级数，允许外推到更长的序列长度。
- **残差连接 + Layer Normalization**：每个子层前后都用 LN(x + Sublayer(x))。
- **学习率 warmup**：前 4000 步线性增加，之后反向 sqrt 衰减。
- 模型也成功泛化到英语成分句法分析（WSJ 23 F1 达到 92.7）。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Entities/Transformer]]、[[Wiki/Topics/大语言模型基础]]
- 后续发展：BERT、GPT 系列、LLaMA、ViT 等都基于 Transformer 架构。
- 位置编码的演进：本文的正弦位置编码 → 后[[Wiki/Concepts/RoPE 旋转位置编码]]等改进。

## 局限或注意事项

- 自注意力复杂度 O(n^2 · d)，无法直接处理长序列；文中提到研究受限注意力作为未来方向。
- 当时仅在翻译和解析任务上评估（2017），尚未探索在语言模型预训练 + 微调范式中的应用。
- 模型规模较小（base 65M, big 213M 参数），与现代 LLM 的规模差距巨大。
