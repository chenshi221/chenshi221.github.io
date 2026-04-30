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

![](https://ar5iv.labs.arxiv.org/html/2103.14030/assets/x1.png)

*Figure 1: Swin Transformer 架构总览——整体采用类似 CNN 的层次化设计：(a) 通过 Patch Merging 逐步降低分辨率、增加通道数，产生 4 个尺度的特征图(H/4, H/8, H/16, H/32)；(b) 每个 Swin Transformer Block 使用基于 Shifted Window 的自注意力替代标准多头自注意力。这种设计使 Swin 可适配 FPN 进行检测和分割。*

### 1. 层次化架构（Hierarchical Architecture）

Swin Transformer 的整体架构类似 CNN 的多阶段设计：

**Stage 1: Patch Partition + Linear Embedding**
- 输入：H×W×3 的 RGB 图像
- 划分为 4×4 patch，每个 patch 展开为 48 维向量
- 经过 Linear Embedding 投影到 C 维

**Stage 2-4: Patch Merging + Swin Transformer Blocks**
- 每个 Stage 开始时会进行 Patch Merging：将 2×2 相邻 patch 的 feature 拼接后降维（2× 下采样）
- 每个 Stage 内包含多个 Swin Transformer Block
- 最终产生 4 个尺度的特征图：H/4-W/4, H/8-W/8, H/16-W/16, H/32-W/32

这种设计使得 Swin Transformer 可以作为通用主干直接接入 FPN（Feature Pyramid Network）进行检测和分割。

### 2. Shifted Window 自注意力（核心创新）

这是 Swin Transformer 最重要的创新。

**普通 Window 自注意力（W-MSA）**：
- 将特征图划分为 M×M 的窗口（默认 M=7）
- 每个窗口内独立计算自注意力
- 计算复杂度从 O(H²W²) 降至 O(HW·M²) ——与图像分辨率呈线性关系
- 但窗口之间没有信息交互！

**Shifted Window 自注意力（SW-MSA）**：
- 连续的两个 Swin Transformer Block 中：
  - Block l：使用普通窗口划分
  - Block l+1：使用偏移（shift）后的窗口划分（偏移量为 ⌊M/2⌋, ⌊M/2⌋）
- 交替使用两种窗口划分方式，实现跨窗口的信息交互

数学表达：

ẑ^l = W-MSA(LN(z^{l-1})) + z^{l-1}
z^l = MLP(LN(ẑ^l)) + ẑ^l
ẑ^{l+1} = SW-MSA(LN(z^l)) + z^l
z^{l+1} = MLP(LN(ẑ^{l+1})) + ẑ^{l+1}

**高效移位实现**：
- 由于窗口偏移后会产生更多、更小的窗口（大小不一），原始实现会导致计算不均衡
- Swin 使用**循环移位（cyclic shift）+ 掩码注意力**策略
- 将特征图循环移位使窗口重新对齐，在不同区域使用不同的注意力掩码防止跨区域交互
- 计算后再反向循环移位恢复原状

![](https://ar5iv.labs.arxiv.org/html/2103.14030/assets/x2.png)

*Figure 2: Shifted Window 自注意力机制——左侧为规则的 W-MSA（窗口内自注意力），右侧为偏移后的 SW-MSA。通过在连续两个 block 中交替使用两种窗口划分方式，实现了跨窗口信息交互，同时保持了 O(N) 的线性计算复杂度。*

### 3. 相对位置编码（Relative Position Bias）

Swin Transformer 在自注意力计算中加入了相对位置偏置：

Attention(Q, K, V) = SoftMax(QK^T/√d + B) V

其中 B ∈ R^{M²×M²} 是可学习的相对位置偏置矩阵，取自参数化矩阵 B̂ ∈ R^{(2M-1)×(2M-1)}。

相比 ViT 的绝对位置嵌入，相对位置偏置有更好的泛化性（不同分辨率直接外推）。

### 4. 模型变体

| 模型 | C | Layer Numbers | 参数量 |
|------|---|---------------|--------|
| Swin-T | 96 | {2, 2, 6, 2} | 28M |
| Swin-S | 96 | {2, 2, 18, 2} | 50M |
| Swin-B | 128 | {2, 2, 18, 2} | 88M |
| Swin-L | 192 | {2, 2, 18, 2} | 197M |

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
