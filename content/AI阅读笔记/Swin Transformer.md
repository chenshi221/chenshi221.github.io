---
title: 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows'
authors:
  - Ze Liu
  - Yutong Lin
  - Yue Cao
  - Han Hu
  - Yixuan Wei
  - Zheng Zhang
  - Stephen Lin
  - Baining Guo
institutions: Microsoft Research
aliases:
  - Swin Transformer
  - Swin
date: '2026-04-30'
publish_date: 2021-08
venue: 'ICCV 2021 (Best Paper Award)'
tags:
  - 论文
  - 视觉Transformer
  - 分层架构
  - Shifted Window
  - 目标检测
  - 语义分割
url: 'https://arxiv.org/abs/2103.14030'
code: 'https://github.com/microsoft/Swin-Transformer'
rating: ⭐⭐⭐⭐⭐
status: Read
---
**一句话总结**：Swin Transformer 提出了分层视觉 Transformer 架构，通过 Shifted Window 自注意力机制在保持线性计算复杂度的同时实现了跨窗口信息交互，在图像分类、目标检测和语义分割三大任务上全面达到 SOTA，并获得 ICCV 2021 最佳论文奖。

---

## Intro

### Motivation

ViT 证明了 Transformer 在视觉领域的潜力，但它存在两个关键局限：
1. **固定尺度的特征图**：ViT 输出固定分辨率的特征，不适合需要多尺度特征金字塔的密集预测任务（如检测、分割）
2. **全局自注意力的 O(N²) 复杂度**：无法高效处理高分辨率图像

CNN 天然具有层次化结构（通过池化逐步降低分辨率、增加通道数），这种设计非常适合视觉任务。能否设计一个 Transformer，既具有 CNN 的层次化特征金字塔，又有 Transformer 的全局建模能力？

### 核心主张

1. **层次化设计**：像 CNN 一样通过 patch merging 逐步降低分辨率、增加维度，产生多尺度特征
2. **Shifted Window 自注意力**：将自注意力限制在局部窗口内（线性复杂度），通过交替移动窗口实现跨窗口建模
3. **通用视觉主干**：Swin Transformer 可在分类、检测、分割上统一使用

### 贡献

1. 提出层次化视觉 Transformer 架构，产生多尺度特征金字塔
2. 提出 Shifted Window 自注意力，O(N) 复杂度和跨窗口建模兼得
3. 在 COCO 目标检测（+2.7 AP）和 ADE20K 语义分割（+3.2 mIoU）上大幅超越此前 SOTA
4. 获得 ICCV 2021 最佳论文奖（Marr Prize）

---

## Method 核心方法

Swin Transformer 解决了 ViT 的两大核心局限——(1) 全局自注意力的 O(N²) 复杂度无法高效处理高分辨率图像；(2) 固定分辨率输出无法适配检测/分割的 FPN 多尺度需求。解决方案：层次化架构 + Shifted Window 自注意力。

### 1. 层次化架构

Swin 仿照 CNN 的多阶段设计，通过 Patch Merging 逐步降低分辨率、增加通道数，产生 4 个尺度的特征金字塔：

| Stage | 操作 | 输出分辨率 | 通道数 |
|-------|------|-----------|--------|
| **1** | Patch Partition (4×4) + Linear Embedding | H/4 × W/4 | C |
| **2** | Patch Merging (2×2→4C→2C) + Swin Blocks | H/8 × W/8 | 2C |
| **3** | Patch Merging (2×2→8C→4C) + Swin Blocks | H/16 × W/16 | 4C |
| **4** | Patch Merging (2×2→16C→8C) + Swin Blocks | H/32 × W/32 | 8C |

Patch Merging 操作：将 2×2 相邻 patch 的特征拼接（4C 维）→ 线性层降维到 2C。这使得 Swin 可作为 FPN 的通用主干直接接入检测和分割任务——这是 ViT 做不到的。

### 2. Shifted Window 自注意力（核心创新）

**动机**：全局自注意力 O(H²W²) 复杂度。将注意力限制在局部窗口 → O(HW·M²)（M=7 为窗口大小，线性于分辨率）。但纯窗口注意力缺乏**跨窗口连接**。

**解决方案**：连续两个 block 交替使用不同的窗口划分模式：

**Block l（W-MSA）——普通窗口**：
- 将特征图均匀划分为 M×M 窗口
- 每个窗口内独立计算多头自注意力（8 head, 32-dim per head）

**Block l+1（SW-MSA）——偏移窗口**：
- 窗口划分偏移 (⌊M/2⌋, ⌊M/2⌋)
- 偏移后窗口跨越原窗口边界，实现跨窗口信息交互

连续两个 block 的数学表达：

$$\hat{\mathbf{z}}^l = \text{W-MSA}(\text{LN}(\mathbf{z}^{l-1})) + \mathbf{z}^{l-1}$$
$$\mathbf{z}^l = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^l)) + \hat{\mathbf{z}}^l$$
$$\hat{\mathbf{z}}^{l+1} = \text{SW-MSA}(\text{LN}(\mathbf{z}^l)) + \mathbf{z}^l$$
$$\mathbf{z}^{l+1} = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^{l+1})) + \hat{\mathbf{z}}^{l+1}$$

**高效实现：循环移位 + 掩码注意力**。偏移后会产生 4 类大小不一的窗口（角、边），直接实现计算不均衡。Swin 的 trick：将特征图循环移位使窗口重新对齐→在标准窗口中计算注意力时用掩码阻止非相邻区域交互→计算后反向移位恢复。比 naive padding 方案计算量低得多。

**复杂度对比**：

| 注意力类型 | 每窗口复杂度 | 总复杂度 |
|-----------|------------|---------|
| 全局 MSA | O((HW)²·d) | O(H²W²·d) |
| **Window MSA** | O(M²·M²·d) | **O(HW·M²·d)** (线性) |
| Shifted Window | O(M²·M²·d) | O(HW·M²·d) (线性) |

### 3. 相对位置偏置

Swin 在自注意力中引入可学习的相对位置偏置 B（替代 ViT 的绝对位置嵌入）：

$$\text{Attention}(Q,K,V) = \text{SoftMax}(QK^T/\sqrt{d} + B)V$$

B ∈ R^{M²×M²} 来自参数化矩阵 $\hat{B} \in \mathbb{R}^{(2M-1)\times(2M-1)}$——每个相对位置 (Δx, Δy) 对应 B̂ 中一个可学习标量。相比 ViT 的绝对位置嵌入，相对偏置对不同分辨率和窗口大小有天然泛化性（直接外推，无需插值）。

### 4. 模型变体

| 模型 | C | Stage 1 | Stage 2 | Stage 3 | Stage 4 | 参数量 | FLOPs |
|------|---|---------|---------|---------|---------|--------|-------|
| Swin-T | 96 | 2 | 2 | 6 | 2 | 28M | 4.5G |
| Swin-S | 96 | 2 | 2 | 18 | 2 | 50M | 8.7G |
| Swin-B | 128 | 2 | 2 | 18 | 2 | 88M | 15.4G |
| Swin-L | 192 | 2 | 2 | 18 | 2 | 197M | 34.5G |

*C=第一阶段通道数，后续阶段通过 Patch Merging 翻倍。各阶段层数如 {2,2,6,2} 表示 Stage 1-4 的 Swin Block 数量。*

---

## 实验/评估/结果

### ImageNet-1K 图像分类

| 模型 | Image Size | Params | FLOPs | Top-1 Acc |
|------|-----------|--------|-------|-----------|
| Swin-T | 224² | 28M | 4.5G | 81.3% |
| Swin-S | 224² | 50M | 8.7G | 83.0% |
| Swin-B | 224² | 88M | 15.4G | 83.5% |
| Swin-B | 384² | 88M | 47.0G | 84.5% |
| DeiT-B | 384² | 86M | 55.4G | 83.1% |
| ViT-B/16 | 384² | 86M | 55.4G | 77.9% |

Swin-B/384 以更少的 FLOPs 超越了 DeiT-B。

### COCO 目标检测

（使用 Mask R-CNN + Swin-T 和 Swin-S 作为主干）

| 主干 | AP^box | AP^mask |
|------|--------|---------|
| ResNet-50 | 38.2 | 34.7 |
| Swin-T | **43.7** (+5.5) | **39.8** (+5.1) |
| ResNeXt101-64x4d | 41.9 | 38.4 |
| Swin-S | **46.5** (+4.6) | **42.1** (+3.7) |

### ADE20K 语义分割

（使用 UperNet + Swin 主干）

| 主干 | mIoU |
|------|------|
| ResNet-101 | 44.9 |
| Swin-S | **49.5** (+4.6) |
| Swin-L | **53.5** |

### 消融实验

- **Shifted Window 的作用**：移除 W-MSA/SW-MSA 交替 → 分类精度下降 1.0-1.5 个百分点
- **相对位置偏置的作用**：去掉位置偏置 → 分类精度下降 1.2-1.3 个百分点
- **窗口大小 M**：M=7 最佳，更大的窗口精度增益递减而计算量线性增长
- **层次化设计**：直接对最后一张特征图做全局自注意力不如层次化 + 窗口自注意力

---

## 结论

Swin Transformer 通过层次化设计和 Shifted Window 自注意力，成功将 Transformer 适配为通用视觉主干网络。它在分类、检测、分割三大任务上均达到 SOTA，证明了精心设计的 Transformer 架构可以全面替代 CNN。其线性复杂度和多尺度输出使其成为后续大量工作的基础（如 Swin V2、SwinUNETR、Video Swin Transformer 等）。

---

## 思考

### 优点

1. **Shifted Window 是优雅的工程创新**：用极简的窗口偏移 + 循环移位掩码注意力，解决了 transformer 在视觉任务上的两个核心难题——计算复杂度和多尺度建模。ICCV 最佳论文奖实至名归。

2. **通用主干设计**：Swin 不只为分类而设计，而是从一开始就考虑了检测和分割的需求（多尺度输出、线性复杂度）。这种"通用主干"的设计哲学影响了后续所有视觉 Transformer 工作。

3. **扎实的工程实现**：循环移位 + 掩码注意力的实现虽然增加了一些复杂性，但保证了计算效率和可部署性。

4. **相对位置偏置的推广**：比 ViT 的绝对位置嵌入更灵活、泛化更好，成为后续视觉 Transformer 的标配。

### 缺点与局限

1. **窗口大小是固定的 M=7**：这限制了有效感受野。虽然通过层次化逐步增加了视野，但在同一尺度上的感受野始终受限于窗口大小。后续 CSWin 等工作使用十字形窗口解决了这个问题。

2. **无全局注意力层**：虽然 SW-MSA 实现了跨窗口交互，但对语义级别的全局关系（如"图像中有几个人"）可能建模不足。当任务需要全局推理时可能吃亏。

3. **预训练数据规模**：ImageNet-1K 和 ImageNet-22K 的预训练规模限制了 Swin 的上限。Swin V2 通过扩大到 3B 参数和更大数据解决了部分问题。

4. **计算效率不如 CNN**：尽管声称线性复杂度，但实际推理速度仍不如同等精度的 CNN（如 ConvNeXt），尤其是在 mobile 设备上。

### 与已有 Wiki 的连接

- 关联概念：[[Transformer]]、[[Shifted Window]]、[[层次化架构]]、[[相对位置编码]]、[[FPN]]
- 关联实体：[[ViT]]、[[DETR]]、[[ResNet]]、[[ConvNeXt]]
- 关联论文：[[AI阅读笔记/ViT]]、[[AI阅读笔记/MLP-Mixer]]、[[AI阅读笔记/MAE]]
- 关联比较：[[Wiki/Comparisons/CNN vs Transformer 在视觉任务上的对比]]
