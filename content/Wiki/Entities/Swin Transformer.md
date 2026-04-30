---
type: entity
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Swin
tags:
  - vision-transformer
  - backbone
  - hierarchical
  - microsoft
sources:
  - '[[Wiki/Sources/Swin Transformer]]'
confidence: high
---
# Swin Transformer

## 基本信息

- 全称：Swin Transformer
- 论文：[[Wiki/Sources/Swin Transformer]]
- 作者：Ze Liu, Yutong Lin, Yue Cao, Han Hu 等（Microsoft Research Asia）
- 发表：ICCV 2021（马尔奖 / 最佳论文奖）
- 类型：通用视觉 Backbone

## 核心架构

Swin Transformer 的核心创新是层级化 + shifted window 自注意力：

1. **层级化结构**：4 个 Stage，每个 Stage 通过 Patch Merging 将空间分辨率减半、通道数加倍（类似 CNN 的金字塔特征层次）。
2. **Shifted Window Self-Attention**：每个 Stage 内使用局部窗口自注意力（W-MSA），相邻层之间通过窗口偏移（SW-MSA）实现跨窗口信息交互。
3. **线性复杂度**：自注意力限制在固定大小窗口（如 7x7）内，计算量与图像分辨率成线性关系 O(N)。

## 关键特性

- **通用 Backbone**：在分类（ImageNet 87.3% top-1）、检测（COCO 58.7 box AP）、分割（ADE20K 53.5 mIoU）上全面超越此前 SOTA。
- **层级化**：天然适配 FPN 等多尺度特征架构，使其可无缝替代 ResNet 作为下游任务的 backbone。
- **Shifted Window**：以极小的额外计算代价（窗口偏移的开销可忽略）实现跨窗口连接。
- **对 MLP 架构的启示**：Swin 的层级 + 窗口设计也被证明对 all-MLP 架构（如 MLP-Mixer 的后续变体）有益。

## 变体与演进

- Swin-T (Tiny)、Swin-S (Small)、Swin-B (Base)、Swin-L (Large)：不同规模的模型。
- SwinV2：改进在大模型上的训练稳定性（引入 residual-post-norm 和 cosine attention）。
- 启发了 CSwin（十字形窗口）、Video Swin Transformer（3D 窗口）等后续工作。

## 影响力

- 获得 ICCV 2021 最佳论文奖，是继 ResNet 之后最具影响力的通用视觉 backbone 之一。
- "层级化 Transformer"成为后续 ViT 研究的主流范式。
- 被广泛用于检测（Mask R-CNN、Cascade R-CNN）、分割（UperNet）等任务的迁移学习中。

## 关联

- 前身：[[Wiki/Entities/Vision Transformer (ViT)]]
- 主题：[[Wiki/Topics/Vision Transformer 演进]]
- 架构对比：与 ViT（固定分辨率、全局注意力）、MLP-Mixer（无注意力）形成对比。
