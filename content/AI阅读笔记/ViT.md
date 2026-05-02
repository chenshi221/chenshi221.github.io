---
title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
authors:
  - Alexey Dosovitskiy
  - Lucas Beyer
  - Alexander Kolesnikov
  - Dirk Weissenborn
  - Xiaohua Zhai
  - Thomas Unterthiner
  - Mostafa Dehghani
  - Matthias Minderer
  - Georg Heigold
  - Sylvain Gelly
  - Jakob Uszkoreit
  - Neil Houlsby
institutions: Google Research
aliases:
  - ViT
  - Vision Transformer
date: '2026-04-30'
publish_date: 2021-06
venue: 'ICLR 2021'
tags:
  - 论文
  - 视觉Transformer
  - 图像分类
  - 自注意力
  - 预训练
url: 'https://arxiv.org/abs/2010.11929'
code: 'https://github.com/google-research/vision_transformer'
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：Vision Transformer (ViT) 首次将标准 Transformer 架构直接应用于图像分类——将图像切分为固定大小的 patch，作为 token 序列输入 Transformer 编码器，在大规模预训练（JFT-300M）下，ViT 以更少的计算资源和更简洁的结构超越了当时 SOTA 的 CNN 模型。

---

## Intro

### Motivation

Transformer 架构在 NLP 领域取得了巨大成功（BERT、GPT），但计算机视觉的主流范式仍然是卷积神经网络（CNN）。虽然已有工作尝试将自注意力机制引入视觉（如 non-local networks、stand-alone attention），但它们要么是 CNN 的补充组件，要么高度定制化导致难以规模化。

ViT 的核心问题是：**能否尽可能不修改标准的 Transformer，直接将其应用于图像？**

### 核心主张

1. 将图像切分为固定大小的 patch（如 16x16），每个 patch 线性投影为一个 token
2. 使用标准 Transformer 编码器处理这些 patch token + 可学习的 class token
3. 在大规模数据集上预训练后，ViT 可以在中小型下游任务上达到或超越 CNN SOTA
4. 对于中等规模数据集（如 ImageNet），ViT 如果不做预训练则效果不如 CNN——Transformer 的归纳偏置更弱，需要更大数据来补偿

### 贡献

1. 提出了 Vision Transformer (ViT)：仅使用标准 Transformer 架构，不做视觉特定的修改
2. 系统的数据规模实验：证明 ViT 的性能高度依赖预训练数据规模
3. 在 JFT-300M 上预训练后，ViT 在多个图像识别 benchmark 上达到 SOTA 或接近 SOTA
4. 训练效率更高：相比同等性能的 CNN，ViT 需要更少的计算资源（TPU 训练时间更短）

---

## Method 核心方法

ViT 的设计哲学是"尽可能不修改标准 Transformer"——图像→patch→token→Transformer，仅 patch embedding 是视觉特有的。

### 1. 图像 Patch 化（Image to Patches）

输入图像 $x \in \mathbb{R}^{H\times W\times C}$ 被切分为 $N = HW/P^2$ 个不重叠的 2D patch（P×P 大小）。每个 patch 展平为一个长度为 $P^2\cdot C$ 的向量，通过可训练的线性投影 $E \in \mathbb{R}^{(P^2\cdot C)\times D}$ 映射到 D 维嵌入空间：

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}^1_p E; \dots; \mathbf{x}^N_p E] + \mathbf{E}_{\text{pos}}$$

- $\mathbf{x}_{\text{class}} \in \mathbb{R}^D$：可学习的分类 token（类似 BERT [CLS]），最终隐藏状态用作分类
- $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1)\times D}$：1D 可学习位置嵌入（而非 2D——论文也尝试了 2D 位置嵌入但无显著提升）

### 2. 模型规格（原文 Table 1）

| 模型 | 层数 | Hidden | MLP | 头数 | 参数量 | Patch | 输入 |
|------|------|--------|-----|------|--------|-------|------|
| ViT-B/16 | 12 | 768 | 3072 | 12 | 86M | 16×16 | 224² |
| ViT-L/16 | 24 | 1024 | 4096 | 16 | 307M | 16×16 | 224² |
| ViT-H/14 | 32 | 1280 | 5120 | 16 | 632M | 14×14 | 224² |

ViT-B/16 即 "Base" + "16×16 patch"。更小的 patch（14×14）产生更长的序列（更多 token），细粒度更高但计算量更大。

### 3. Transformer 编码器

嵌入序列经 L 层标准 Transformer 编码器（Pre-Norm 形式）：

$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$
$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

其中 MSA = Multi-head Self-Attention（所有 patch token + class token 之间全互注意），MLP = GELU 激活的两层线性变换。

最后一层 class token 输出 $\mathbf{y} = \text{LN}(\mathbf{z}^0_L)$ 经单层线性分类头得到预测。论文也探索了全局平均池化（GAP）替代 class token（效果相近）。

**Hybrid 变体**：用 CNN 特征图替代原始图像 patch 作为输入。CNN 输出的特征图（如 ResNet-50 的最后一层）被展平为 patch 序列。Hybrid 版本在中小数据集上优于纯 ViT——因为 CNN 提供了更强的局部归纳偏置。

### 4. 预训练与微调策略

**预训练**：
- 大规模数据集（ImageNet-21K：14M 图像；JFT-300M：300M 图像）上预训练
- 优化器：Adam（β₁=0.9, β₂=0.999），batch 4096，weight decay 0.1
- 学习率 warmup（10K steps）+ cosine decay
- 所有实验在 TPUv3 上完成（ViT-H/14 约 2500 TPUv3-core-days）

**微调**：
- 移除预训练分类头，添加新分类头（零初始化）
- 高分辨率微调时（如 384²），patch 大小不变→序列变长→1D 位置嵌入通过 2D 插值适配
- 可微调全部参数（效果最佳）或仅分类头（有限提升）

**关键实验发现**（Figure 2 from paper）：
- 纯 ImageNet 训练（无预训练）：ViT 显著不如 ResNet——弱归纳偏置需要大数据
- ImageNet-21K 预训练：ViT 与 BiT（ResNet）持平
- JFT-300M 预训练：ViT 全面超越 BiT

### 5. 归纳偏置分析与架构选择

| 属性 | CNN | ViT |
|------|-----|-----|
| 局部性 | 内建（卷积核） | 仅 MLP 局部（门控激活）+ 需要从数据中学 |
| 平移等变性 | 内建（权重共享） | 需要从位置嵌入和数据中学 |
| 尺度不变性 | 内建（池化/多尺度） | 无——ViT 输出固定分辨率 |
| 全局关系 | 深层可学 | **自注意力天然全局**——每层直接访问全图 |

ViT 的核心 trade-off：弱归纳偏置在数据不足时泛化差，但在大数据量下允许更灵活的模式学习——最终超越强偏置的 CNN。

---

## 实验/评估/结果

### 图像分类

**JFT-300M 预训练**（300M 张图片）：
| 模型 | ImageNet Top-1 | ImageNet-RealL | CIFAR-10/100 | Oxford Pets | VTAB |
|------|---------------|----------------|--------------|-------------|------|
| ViT-H/14 | **88.55%** | 90.72% | 99.50%/94.55% | 94.9% | 77.1% |
| BiT-L (ResNet152x4) | 87.54% | 90.54% | 99.42%/94.16% | 94.5% | 76.3% |
| Noisy Student (EfficientNet) | 88.4% | 90.2% | - | - | - |

ViT-H/14 在使用更少计算资源的前提下超越了当时 SOTA CNN。

**ImageNet-21K 预训练**：ViT 在该中等规模数据集上的结果与 BiT 持平，但不如 JFT-300M。

**纯 ImageNet 训练**（无预训练）：ViT 显著不如 ResNet——证明了大规模预训练对 ViT 的不可或缺性。

### 数据规模 v.s. 性能

实验系统性地展示了 ViT 性能随预训练数据规模的增长趋势：
- 1M → 3M → 10M → 30M → 100M → 300M 图像
- 随着数据量增加，ViT 的性能持续提升，在大数据量下超越 CNN
- CNN（如 BiT）在数据量较小时保持优势，但在大数据量下被 ViT 追上或超越

### 训练效率

| 模型 | TPUv3 训练天数 | ImageNet Top-1 |
|------|---------------|----------------|
| ViT-H/14 | 0.47K | 88.55% |
| BiT-L | 1.04K | 87.54% |
| Noisy Student | 6.7K | 88.4% |

ViT 的预训练效率显著高于同等精度的 CNN。

### 内部表征分析

- 位置嵌入学习到了 2D 空间结构（相邻 patch 的嵌入相似）
- 浅层自注意力已能关注到图像中的语义相关区域
- 类 CLS token 表征可泛化到多种下游任务

![](https://arxiv.org/html/2010.11929v2/x5.png)

*Figure 2: ViT 的自注意力可视化——不同注意力头在不同层关注不同的图像区域。浅层关注局部空间邻域，深层逐渐关注语义相关的全局区域（如物体边界和前景），证明 ViT 无需卷积即可学到类似 CNN 的分层特征提取模式。*

---

## 结论

Vision Transformer 证明了标准 Transformer 可以作为一种通用视觉主干，在大规模预训练支持下超越精心设计的 CNN。其核心发现是：Transformer 对 CNN 的替代不是架构优越性的结果，而是"更弱的归纳偏置 + 更大数据"这一范式的胜利。ViT 为后续的 Swin Transformer、DETR、MAE 等工作铺平了道路。

---

## 思考

### 优点

1. **开创性的范式转换**：ViT 是 NLP 和 CV 架构统一的重要里程碑。它证明了"架构的统一是可能的，前提是数据规模足够大"。

2. **实验设计经典**：数据规模从 1M 到 300M 的系统性实验，清晰地揭示了 ViT 的数据饥渴属性。这个实验设计成为后续工作的标准模板。

3. **极简美学**：完全使用标准 Transformer，不做任何视觉特定的改动——这种设计哲学呼应了"追求通用性"的 AI 研究传统。

4. **效率优势真实存在**：ViT 在 TPU 上的训练效率优于 CNN，这不仅仅是一个学术声明，而是实际部署中的竞争优势。

### 缺点与局限

1. **数据依赖性强**：ViT 在小数据上的表现不如 CNN，限制了其在数据稀缺领域的应用。后续 MAE、DINO 等工作正是为了解决这个问题。

2. **位置编码的局限性**：1D 可学习位置编码不能很好地泛化到不同分辨率。需要插值或重新训练。Swin Transformer 等后续工作用相对位置偏差解决了这个问题。

3. **计算复杂度与分辨率平方增长**：自注意力的 O(N²) 复杂度使得 ViT 在高分辨率图像上的应用受限。虽然 patch 大小可以调整，但这改变了有效分辨率。

4. **检测/分割任务上的表现**：原始 ViT 主要验证了分类任务。在目标检测和语义分割等密集预测任务上，需要特征金字塔多尺度结构，而 ViT 输出的是固定分辨率。

5. **JFT-300M 的不可公开性**：JFT-300M 是 Google 内部数据集，社区无法完全复现 ViT 的最佳结果。后续的 MAE 等工作正是为了在公开数据集上达到类似效果。

### 与已有 Wiki 的连接

- 关联概念：[[Transformer]]、[[自注意力]]、[[Patch Embedding]]、[[归纳偏置]]
- 关联实体：[[BERT]]、[[ResNet]]、[[ViT]]
- 关联论文：[[AI阅读笔记/Swin Transformer]]、[[AI阅读笔记/MAE]]、[[AI阅读笔记/MLP-Mixer]]
- 关联比较：[[Wiki/Comparisons/CNN vs Transformer 在视觉任务上的对比]]
