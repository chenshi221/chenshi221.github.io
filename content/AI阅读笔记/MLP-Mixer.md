---
title: 'MLP-Mixer: An all-MLP Architecture for Vision'
authors:
  - Ilya Tolstikhin
  - Neil Houlsby
  - Alexander Kolesnikov
  - Lucas Beyer
  - Xiaohua Zhai
  - Thomas Unterthiner
  - Jessica Yung
  - Andreas Steiner
  - Daniel Keysers
  - Jakob Uszkoreit
  - Mario Lucic
  - Alexey Dosovitskiy
institutions: Google Research (Google Brain)
aliases:
  - MLP-Mixer
  - Mixer
date: '2026-04-30'
publish_date: 2021-05
venue: 'NeurIPS 2021'
tags:
  - 论文
  - MLP
  - 视觉架构
  - 图像分类
  - 无需卷积
  - 无需注意力
url: 'https://arxiv.org/abs/2105.01601'
code: 'https://github.com/google-research/vision_transformer'
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：MLP-Mixer 提出了一个极简的视觉架构——完全基于多层感知机（MLP），不使用卷积、不使用自注意力——通过在 patch 维度（token-mixing）和通道维度（channel-mixing）上交替应用 MLP 实现空间和特征混合，在大规模预训练下性能接近 ViT 和 CNN，挑战了"卷积或注意力是视觉模型必需品"的固有观念。

---

## Intro

### Motivation

在计算机视觉的发展史中，**卷积（CNN）**和**自注意力（Transformer）**是两种主导性的架构选择。人们普遍认为，视觉模型需要某种形式的"空间混合"操作来捕捉图像中像素或 patch 之间的关系。

MLP-Mixer 提出了一个 provocatively 简单的问题：**如果卷积和注意力都不需要呢？** 纯粹的多层感知机（MLP）是否足以构建一个有竞争力的视觉模型？

### 核心主张

1. MLP 独立应用于图像 patch 时，交替混合两种信息就足够了：**token 之间的信息（空间结构）**和**通道之间的信息（特征）**
2. 这种极简设计在大规模预训练下可以达到接近 ViT 和 CNN 的性能
3. 这证明了"数据规模和现代训练策略"比"精巧的归纳偏置"更重要

### 贡献

1. 提出 MLP-Mixer，第一个纯 MLP 的视觉架构（无卷积、无注意力、无循环结构）
2. 设计了 token-mixing MLP + channel-mixing MLP 的交替架构
3. 在 ImageNet 分类上达到有竞争力的精度（87.9% Top-1 with Mixer-H/14 + ImageNet-21K）
4. 哲学贡献：挑战了视觉架构设计的"必需品"假设，证明简单的架构加足够的数据也能取得好效果

---

## Method 核心方法

![](https://ar5iv.labs.arxiv.org/html/2105.01601/assets/x1.png)

*Figure 1: MLP-Mixer 架构总览——图像切分为 patch 并线性投影后得到 tokens×channels 矩阵，反复经过 Mixer Layer（Token-mixing MLP + Channel-mixing MLP）实现跨空间和跨通道的信息混合，最终通过全局平均池化和分类头输出。完全不使用卷积或自注意力。*

### 1. 架构总览

MLP-Mixer 的输入与 ViT 类似：

1. 图像切分为 S 个不重叠的 patch（如 14×14 = 196 个 patch）
2. 每个 patch 通过线性投影映射到 C 维隐藏空间
3. 得到输入矩阵 X ∈ R^{S × C}（patches × channels）

核心组件是 Mixer Layer，重复 N 层：

**Mixer Layer = Token-mixing MLP + Channel-mixing MLP**

```
U = X + W₂ σ(W₁ LN(X)ᵀ)ᵀ  ← Token-mixing: 跨 patch 混合，对每一列(channel)独立
Y = U + W₄ σ(W₃ LN(U))    ← Channel-mixing: 跨 channel 混合，对每一行(patch)独立
```

其中 σ 是 GELU 激活函数，LN 是 Layer Normalization。

### 2. Token-Mixing MLP

- 操作在矩阵的列维度上（即跨所有 patch 的相同 channel）
- 数学上：对 X 转置，应用 MLP（作用于 patch 维度），再转置回来
- 这个 MLP 的参数是共享的——对所有 channel 使用相同的 W₁, W₂
- 功能：混合从不同空间位置来的信息（"这个位置和那个位置的特征之间有什么关系"）

### 3. Channel-Mixing MLP

- 操作在矩阵的行维度上（即单个 patch 的所有 channel）
- 对每个 patch 独立应用 MLP（作用于 channel 维度）
- 这个 MLP 的参数是共享的——对所有 patch 使用相同的 W₃, W₄
- 功能：混合单个位置内不同 channel 的信息（"这个特征和那个特征之间有什么关系"）

### 4. 与 CNN 和 Transformer 的比较

| 操作 | CNN | ViT | MLP-Mixer |
|------|-----|-----|-----------|
| 空间混合 | 卷积核（局部、权重共享） | 自注意力（全局、权重动态） | Token-mixing MLP（全局、权重静态） |
| 特征混合 | 1×1 卷积 / 全连接 | FFN（MLP） | Channel-mixing MLP |
| 归纳偏置 | 强（局部性、平移等变性） | 弱 | 极弱（仅输入格式暗含空间结构） |

关键差异：**Token-mixing MLP 的权重是静态的（与输入内容无关）**，这与自注意力形成对比（注意力权重依赖 content-based query/key 相似度）。

### 5. 模型变体

| 模型 | Patch Size | Token-Mixing 隐层 | Channel-Mixing 隐层 | 层数 | 参数量 |
|------|-----------|-------------------|--------------------|------|--------|
| Mixer-S/32 | 32×32 | 256 | 2048 | 8 | 19M |
| Mixer-S/16 | 16×16 | 256 | 2048 | 8 | 19M |
| Mixer-B/16 | 16×16 | 384 | 3072 | 12 | 59M |
| Mixer-L/16 | 16×16 | 512 | 4096 | 24 | 207M |
| Mixer-H/14 | 14×14 | 512 | 4096 | 32 | 431M |

---

## 实验/评估/结果

### ImageNet 分类（无额外数据）

| 模型 | ImageNet Top-1 | 参数量 |
|------|---------------|--------|
| Mixer-B/16 | 76.4% | 59M |
| ViT-B/16 | 77.9% | 86M |
| BiT-R152x2 (CNN) | 77.3% | 236M |

纯 ImageNet 训练下，Mixer 的性能比 ViT 和 CNN 略低，验证了"弱归纳偏置需要大数据"的假设。

### ImageNet-21K 预训练 + 微调

| 模型 | ImageNet Top-1 | 参数量 |
|------|---------------|--------|
| Mixer-B/16 | 82.3% | 59M |
| Mixer-L/16 | 84.2% | 207M |
| Mixer-H/14 | 87.9% | 431M |
| ViT-B/16 | 83.9% | 86M |
| ViT-L/16 | 85.2% | 307M |
| BiT-L (CNN) | 85.4% | 438M |

在大规模预训练加持下，Mixer-H/14 的性能(87.9%)大幅超越 ViT 和同级别的 CNN。这证明：**在大数据下，归纳偏置的价值下降，模型容量和数据规模的价值上升**。

### 关键分析

1. **Token-mixing MLP 学到的权重模式**：可视化后发现，不同层的 token-mixing MLP 权重呈现出不同的空间频率模式——类似 CNN 不同层的卷积核学到不同的频率响应。

2. **分辨率泛化**：与 ViT 一样，Mixer 的 token-mixing MLP 权重与输入 patch 数量耦合。可以通过插值或微调来适配不同分辨率。

3. **与现代 CNN 的对比**：Mixer-B/16 的性能低于同时期的 EfficientNet-B7，但差距在大模型上缩小。

---

## 结论

MLP-Mixer 证明了纯粹基于 MLP 的架构可以在图像分类上达到有竞争力的性能。它的核心价值不在于精度超越 SOTA（它没有），而在于**哲学层面的启示**：当数据规模足够大时，架构中可以去除专门为视觉设计的归纳偏置，简单通用的操作就能取得好结果。这呼应了同时期"追求更通用、更简单架构"的研究潮流。

---

## 思考

### 优点

1. **哲学立意高**："没有卷积、没有注意力，MLP 就够了"——论文标题本身就是一种宣言。它是"数据为王的架构极简主义"潮流的旗舰作品之一。

2. **设计优雅**：Token-mixing + Channel-mixing 的分离设计非常自然，映射到化学和数学中的"交替优化"概念。这种简洁性在教学和概念传播上有巨大价值。

3. **极端对比实验**：用最弱的归纳偏置（MLP）做最强的对比——只要大数据和现代训练策略，什么都能工作。这个实验本身比任何具体的精度数字更有价值。

4. **推动了后续工作**：Mixer 的成功启发了 ResMLP、gMLP、ConvMixer 等更加极简的架构尝试，推动了社区对"归纳偏置 v.s. 数据规模"的深入讨论。

### 缺点与局限

1. **实际部署价值有限**：Mixer 在推理速度和精度上均不如同等规模的 CNN 和 ViT。它的价值是学术性的（证明可能性），而不是工程性的（实用工具）。

2. **未扩展到检测和分割**：与 Swin Transformer 不同，Mixer 主要验证了分类任务。其固定分辨率的输出格式不适合密集预测任务（虽然可以通过改造扩展）。

3. **数据需求不现实**：Mixer-H/14 需要 ImageNet-21K 预训练才能达到 87.9%。对于大多数实际场景（数据稀缺领域），CNN 仍然是更好的选择。

4. **Token-mixing MLP 的权重刚性**：自注意力的核心优势在于其 content-dependent 的权重——模型可以根据输入内容动态调整"关注哪里"。MLP 的静态权重限制了在需要灵活空间推理的任务上的表现。

### 与已有 Wiki 的连接

- 关联概念：[[MLP]]、[[Token Mixing]]、[[Channel Mixing]]、[[归纳偏置]]、[[架构极简主义]]
- 关联实体：[[ViT]]、[[ResNet]]、[[ConvNeXt]]
- 关联论文：[[AI阅读笔记/ViT]]、[[AI阅读笔记/Swin Transformer]]
- 关联比较：[[Wiki/Comparisons/CNN vs Transformer 在视觉任务上的对比]]
