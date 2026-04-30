---
title: Attention Is All You Need
authors:
  - Ashish Vaswani
  - Noam Shazeer
  - Niki Parmar
  - Jakob Uszkoreit
  - Llion Jones
  - Aidan N. Gomez
  - Łukasz Kaiser
  - Illia Polosukhin
institutions: Google Brain / Google Research / University of Toronto
aliases:
  - Transformer
  - Transformer论文
date: '2026-04-30'
publish_date: '2017-06'
venue: NeurIPS 2017
tags:
  - 论文
  - 深度学习
  - NLP
  - 注意力机制
  - 机器翻译
  - 基础架构
url: 'https://arxiv.org/abs/1706.03762'
code: 已开源（Tensor2Tensor / 众多第三方实现）
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：本文提出了 Transformer，一种完全基于注意力机制、摒弃循环和卷积的序列转换模型，在机器翻译任务上以更少的训练时间和计算量取得了 SOTA 成绩，并成为后来几乎所有大语言模型和多模态模型的基础架构。

![](https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png)

*Figure 1: Transformer 模型架构。左半边为编码器，右半边为解码器。*

---

## Intro

### Motivation

2017 年之前，序列建模和机器翻译的主流方法是基于 RNN / LSTM / GRU 的编码器-解码器架构。这些循环模型的核心问题是**天然的串行性**：每个时间步的计算依赖前一个时间步的隐藏状态，导致训练无法并行化，在处理长序列时尤为严重。注意力机制虽然已被用于连接编码器和解码器，但始终是作为循环网络的辅助模块存在。

### 贡献

1. **提出 Transformer 架构**：完全基于注意力机制，摒弃了循环和卷积，极大提升了训练并行度
2. **Scaled Dot-Product Attention + Multi-Head Attention**：将注意力计算转化为高效的矩阵乘法，多头机制让模型同时关注不同表示子空间
3. **正弦位置编码**：用三角函数注入序列位置信息，支持外推到训练时未见过的序列长度
4. **在机器翻译上达到 SOTA**：EN-DE 28.4 BLEU，EN-FR 41.8 BLEU，训练成本仅为当时最佳模型的几十分之一

---

## Method 核心方法

### 1. 架构总览：编码器-解码器结构

Transformer 保留了经典的编码器-解码器框架：
- **编码器**：N=6 层堆叠，每层包含 Multi-Head Self-Attention + Position-wise FFN，各子层均采用残差连接 + Layer Normalization
- **解码器**：N=6 层堆叠，每层包含 Masked Multi-Head Self-Attention + Cross-Attention（Q 来自解码器，K/V 来自编码器输出）+ Position-wise FFN

所有子层输出维度 d_model = 512（base）/ 1024（big）。

### 2. Scaled Dot-Product Attention

核心公式：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d_{k}}})V$$

除以 sqrt(d_k) 的原因：当 d_k 较大时，点积的数值增大，会将 softmax 推到梯度极小的饱和区。缩放后避免了这一问题。

![](https://arxiv.org/html/1706.03762/x2.png)

*Figure 2: （左）Scaled Dot-Product Attention。（右）Multi-Head Attention，由多个并行的注意力层组成。*

### 3. Multi-Head Attention

将 Q、K、V 分别用 h 组不同的线性投影矩阵映射到 d_k、d_k、d_v 维度，并行计算注意力，然后拼接再投影。

$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head_{1}},...,\mathrm{head_{h}})W^{O}$$

$$\mathrm{head_{i}}=\mathrm{Attention}(QW^{Q}_{i},KW^{K}_{i},VW^{V}_{i})$$

Base: h=8, d_k=d_v=64（d_model/h）。Big: h=16, d_k=d_v=64。

### 4. Position-wise Feed-Forward Network

$$\mathrm{FFN}(x)=\max(0,xW_{1}+b_{1})W_{2}+b_{2}$$

两个线性变换中间夹 ReLU，d_ff = 2048（base）/ 4096（big）。对每个位置独立且相同地应用。

### 5. 位置编码

Transformer 没有循环和卷积，自身无法感知序列顺序。本文使用正弦位置编码加到 embedding 上：

$$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{\text{model}}})$$

$$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{\text{model}}})$$

实验表明，可学习的位置 embedding 效果几乎相同，但正弦版本允许外推到更长序列。

### 6. 自注意力的优势分析

与循环层和卷积层相比，自注意力有三方面优势：

| 层类型 | 每层复杂度 | 最少串行操作数 | 最大路径长度 |
| --- | --- | --- | --- |
| Self-Attention | O(n^2·d) | O(1) | O(1) |
| Recurrent | O(n·d^2) | O(n) | O(n) |
| Convolutional | O(k·n·d^2) | O(1) | O(log_k(n)) |

自注意力的信息传递路径最短（O(1)），长距离依赖学习能力最强。

---

## 实验/评估/结果

### 机器翻译

在 WMT 2014 EN-DE 和 EN-FR 上评估：

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练成本 (FLOPs) |
| --- | --- | --- | --- |
| Transformer (base) | 27.3 | 38.1 | 3.3×10^18 |
| Transformer (big) | **28.4** | **41.8** | 2.3×10^19 |
| GNMT+RL Ensemble | 26.30 | 41.16 | 1.1×10^21 |
| ConvS2S Ensemble | 26.36 | 41.29 | 1.2×10^21 |

- Transformer big 在 EN-DE 上超越所有已有模型（包括集成模型）超过 2 BLEU
- 在 EN-FR 上超越所有单模型，训练成本仅 1/4
- Base 模型训练只需 12 小时（8 P100），Big 模型 3.5 天

### 消融实验

- **头数**：单头（24.9）< 4头（25.5）< 8头（25.8）< 16头（25.8），超过 16 头后饱和
- **d_k 减小**：头数增加但 d_k 减小（保持总计算量不变）时，质量下降（25.4 at d_k=16）
- **模型大小**：增大 d_model 和 d_ff 持续提升效果
- **Dropout**：P_drop=0.1 最佳（25.8），去掉 dropout（P_drop=0.0）降至 24.6
- **位置编码**：正弦编码 vs 可学习 embedding 几乎无差别（25.7 vs 25.8）

### 训练细节

- 数据集：WMT 2014 EN-DE（4.5M 句对）、EN-FR（36M 句对）
- Tokenization：BPE（EN-DE，~37K vocab）/ word-piece（EN-FR，~32K vocab）
- 优化器：Adam，b1=0.9，b2=0.98，warmup_steps=4000
- 正则化：Residual Dropout（P_drop=0.1）+ Label Smoothing（0.1）
- 硬件：8 块 NVIDIA P100 GPU

### 扩展到其他任务：English Constituency Parsing

在 WSJ 数据集上，Transformer 在 40K 句子的半监督设置下取得 91.3 F1，大幅领先其他模型；在完整训练集上取得 92.7 F1，证明 Transformer 不仅适用于翻译，也具备良好的泛化性。

---

## 结论

本文提出了 Transformer，一种完全基于注意力机制的序列转换架构。它在机器翻译上以更少的训练时间达到了 SOTA，且对英语成分句法分析等任务具有良好泛化性。论文明确提出了自注意力在计算复杂度、并行性和长距离依赖学习三个维度上的理论优势。这一架构成为了随后 BERT、GPT、LLaMA 等几乎所有主流预训练大模型的基石。

---

## 思考

### 优点

1. **简洁而优雅的架构创新**：用单一的注意力机制统一了序列建模，去掉了 RNN 的串行瓶颈。Scaled Dot-Product Attention 和 Multi-Head Attention 的设计至今仍是 LLM 的核心构件。

2. **工程与理论并重**：不仅给出了实验 SOTA，还提供了自注意力相对 RNN/CNN 在计算复杂度、并行性、路径长度三个维度上的理论分析（Table 1），这是极具说服力的论证方式。

3. **泛化前瞻性**：论文中提到的 "restricted self-attention"（将注意力限制在局部窗口）预示了后来的 Sparse Attention、FlashAttention 等研究方向。正弦位置编码的外推特性预示了后续的长度泛化研究。

4. **影响深远**：这篇论文的引用量超过 15 万次，其架构设计（Multi-Head Attention、残差连接 + LayerNorm、Position-wise FFN）几乎被所有后续 LLM 继承。

### 缺点与局限

1. **训练数据规模小**：仅在百万级的机器翻译数据上训练。2017 年时不可能预见大规模预训练（Transformer 的架构优势在数十亿 token 级预训练中才被充分释放，详见 BERT 和 GPT 系列）。

2. **未涉及文本生成的训练**：仅做了编码器-解码器的翻译训练，没有探索仅解码器（decoder-only）架构用于语言建模的可能性。

3. **位置编码的后续演进**：正弦位置编码后来被可学习 embedding（BERT）、相对位置编码（Transformer-XL）、RoPE（LLaMA 系列）等方案取代。

4. **没有涉及注意力复杂度的改进**：O(n^2) 的复杂度在长序列上是瓶颈，本文只是简单提到 "restricted self-attention" 的方向，但未展开。

### 与已有 Wiki 的连接

- 关联概念：[[Transformer]]、[[Scaled Dot-Product Attention]]、[[Multi-Head Attention]]、[[位置编码]]（Positional Encoding）
- 关联论文：[[AI阅读笔记/BERT]]、[[AI阅读笔记/GPT-3]]、[[AI阅读笔记/RoPE]]（RoPE 是对位置编码的重大改进）
- 关联比较：编码器-解码器 vs 仅编码器 vs 仅解码器架构的演变
