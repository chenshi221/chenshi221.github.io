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

MLP-Mixer 的 provocatively 简单设计：完全不使用卷积、不使用自注意力——仅用两层 MLP（分别沿 patch 维度和 channel 维度交替）实现空间和特征混合。

### 1. 架构总览

输入与 ViT 一致：图像切分为 S 个不重叠 patch（如 14×14=196）、线性投影到 C 维 → 得到矩阵 X ∈ R^{S×C}。经 N 层 Mixer Layer 后，全局平均池化 + 线性分类头输出。

**Mixer Layer = Token-mixing MLP + Channel-mixing MLP**（均含残差连接 + LayerNorm）：

$$\mathbf{U}_{*,i} = \mathbf{X}_{*,i} + \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \cdot \text{LN}(\mathbf{X})_{*,i}) \quad \text{(Token-mixing)}$$
$$\mathbf{Y}_{j,*} = \mathbf{U}_{j,*} + \mathbf{W}_4 \cdot \sigma(\mathbf{W}_3 \cdot \text{LN}(\mathbf{U})_{j,*}) \quad \text{(Channel-mixing)}$$

其中 $\sigma$ = GELU。两 MLP 结构相同（单隐藏层升维→降维），但作用于不同轴。

### 2. Token-Mixing MLP——跨空间混合

作用于 X 的**列**（所有 patch 的同一 channel）：
- 实现：转置 X → 对每列（即每个 channel 的所有 patch 值）独立应用同一 MLP → 转置回
- 输入维度 S 个 token，隐藏维度 $D_S$（>S，如 S=196, $D_S$=256 或 384）
- 参数 **共享**——所有 channel 使用相同的 W₁, W₂
- 功能：混合不同空间位置的信息（"左上角 patch 和右下角 patch 的同一 channel 有什么关系"）

**关键特性**：权重是**静态的**——训练后固定，不依赖输入内容。这与自注意力形成根本对比（注意力权重是 content-based 的动态值）。

### 3. Channel-Mixing MLP——跨特征混合

作用于 X 的**行**（单个 patch 的所有 channel）：
- 对每行（单个 patch）独立应用同一 MLP
- 输入维度 C，隐藏维度 $D_C$（>C，如 C=768, $D_C$=3072）
- 参数**共享**——所有 patch 使用相同的 W₃, W₄
- 功能：混合单个位置内的不同 feature channel

本质上等同于 Transformer 的 Position-wise FFN——对每个位置独立做特征变换。

### 4. 架构对比与设计哲学

| 维度 | CNN | ViT | **MLP-Mixer** |
|------|-----|-----|---------------|
| 空间混合方式 | 卷积核（局部） | 自注意力（全局） | Token-mixing MLP（全局） |
| 空间混合权重 | 静态（共享参数） | **动态**（content-based） | 静态（共享参数） |
| 特征混合方式 | 1×1 Conv | FFN | Channel-mixing MLP |
| 归纳偏置 | 强 | 弱 | **极弱** |

**核心哲学问题**：为什么静态权重的 Token-mixing MLP 能 work？Mixer 的每一层 Token-mixing MLP 对每个 channel **独立**做相同的空间混合——"对 channel 1 的所有位置做一种全局混合，对 channel 2 也做同样的混合"。信息不跨 channel 流动（跨 channel 由 Channel-mixing 负责）。这种解耦使架构极简——但代价是极度依赖数据量来学习空间结构。

### 5. 模型变体

| 模型 | Patch | S | C | $D_S$ | $D_C$ | N | 参数量 |
|------|-------|---|----|-------|-------|---|--------|
| Mixer-S/32 | 32×32 | 49 | 512 | 256 | 2048 | 8 | 19M |
| Mixer-S/16 | 16×16 | 196 | 512 | 256 | 2048 | 8 | 19M |
| Mixer-B/16 | 16×16 | 196 | 768 | 384 | 3072 | 12 | 59M |
| Mixer-L/16 | 16×16 | 196 | 1024 | 512 | 4096 | 24 | 207M |
| Mixer-H/14 | 14×14 | 256 | 1280 | 512 | 4096 | 32 | 431M |

*S/32 vs S/16：相同参数量，仅 patch 大小不同→序列长度不同。*

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
