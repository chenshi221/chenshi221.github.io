---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Transformer架构
sources:
  - '[[Wiki/Sources/Transformer]]'
tags:
  - transformer
  - architecture
  - attention
  - NLP
confidence: high
---
# Transformer

## 基本信息

- 论文：[[Wiki/Sources/Transformer|Attention Is All You Need]] (2017)
- 作者：Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin（Google Brain / Google Research）
- 地位：现代 NLP 和 AI 的架构基石

## 核心架构

Transformer 是一种基于纯注意力机制的序列到序列模型，由 Encoder 和 Decoder 两部分组成：

### Encoder

- N=6 层，每层包含：Multi-Head Self-Attention + Position-wise FFN
- 残差连接 + Layer Normalization（Post-LN）
- d_model = 512, d_ff = 2048

### Decoder

- N=6 层，每层包含：Masked Multi-Head Self-Attention + Cross-Attention（对 encoder 输出）+ FFN
- Casual masking 确保自回归生成
- 自回归解码，逐 token 生成

### 关键组件

1. **Scaled Dot-Product Attention**：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
2. **Multi-Head Attention**：h=8 个头并行，每个头 d_k=d_v=64
3. **Positional Encoding**：正弦/余弦编码，注入位置信息
4. **Position-wise FFN**：两个线性变换 + ReLU，对每个位置独立应用

## 三种注意力模式

| 模式 | Query | Key/Value | 用途 |
|------|-------|-----------|------|
| Encoder Self-Attention | 当前位置 | 所有位置 | 编码双向上下文 |
| Decoder Self-Attention (Masked) | 当前位置 | 当前位置及之前 | 自回归生成 |
| Cross-Attention | Decoder 层输出 | Encoder 输出 | 连接源语言和目标语言 |

## Transformer 的影响与衍生

### 编码器路线（Encoder-only）
- [[Wiki/Sources/BERT|BERT]]：双向 Transformer 编码器，预训练 + 微调范式

### 解码器路线（Decoder-only）
- [[Wiki/Concepts/GPT 系列模型|GPT 系列]]：自回归 Transformer 解码器，从 GPT-1 到 GPT-4o
- [[Wiki/Sources/LLaMA|LLaMA]]、[[Wiki/Sources/Llama 3|Llama 3]]：开源解码器模型

### 编码器-解码器路线
- T5：统一的 text-to-text Transformer

### 视觉 Transformer
- ViT：将 Transformer 应用于图像分类
- Swin Transformer：层次化视觉 Transformer
- DiT：扩散 Transformer，用于图像生成

### 位置编码演进
- 原始：正弦位置编码（Sinusoidal PE）
- 改进：可学习位置嵌入（Learned PE）
- [[Wiki/Concepts/RoPE 旋转位置编码|RoPE]]：旋转位置编码，现代 LLM 标配

## 为何 Transformer 成功？

1. **可并行训练**：自注意力全部 token 同时计算，不像 RNN 必须串行
2. **短路径长度**：任意两个位置之间的最大路径长度是 O(1)（vs RNN 的 O(n)）
3. **表达能力**：多头注意力允许模型关注不同表示子空间
4. **通用性**：同一架构适用于文本、图像、音频、视频等多种模态

## 相关概念

- [[Wiki/Concepts/RoPE 旋转位置编码|RoPE]]
- [[Wiki/Concepts/Scaling Laws|Scaling Laws]]
- [[Wiki/Concepts/GPT 系列模型|GPT 系列模型]]
- [[Wiki/Topics/大语言模型基础|大语言模型基础]]
