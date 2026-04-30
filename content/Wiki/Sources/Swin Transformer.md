---
type: source
status: processed
source_file: >-
  [[Clippings/Swin Transformer Hierarchical Vision Transformer using Shifted
  Windows.md]]
title: 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows'
site: arXiv
author: Ze Liu et al. (Microsoft Research Asia)
published: 2021-03
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - vision-transformer
  - hierarchical
  - shifted-window
  - backbone
aliases:
  - Swin Transformer
  - Swin
---
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 核心结论

- Swin Transformer 提出了一种**层级化 Vision Transformer**，解决了 ViT 的两个关键局限：固定分辨率 token 和平方级自注意力计算复杂度。
- 通过 shifted window 自注意力机制，在局部窗口内计算注意力，同时通过窗口偏移实现跨窗口信息交互，计算复杂度从 O(N²) 降至 O(N)。
- 层级化设计使其天然适配 dense prediction 任务（检测、分割），成为通用视觉 backbone。

## 关键事实

- 作者：Ze Liu, Yutong Lin, Yue Cao, Han Hu 等（Microsoft Research Asia）。
- 发表于 ICCV 2021（最佳论文奖，Marr Prize），arXiv:2103.14030。
- 核心结构：Patch Partition → 4 个 Stage（每阶段做 patch merging 降低分辨率、增加通道数）→ Swin Transformer Block（W-MSA + SW-MSA 交替）。
- 代表性结果：ImageNet-1K top-1 87.3%；COCO 检测 box AP 58.7 / mask AP 51.1；ADE20K 分割 mIoU 53.5，全面超越此前 SOTA。
- Shifted window 设计也被证明对 all-MLP 架构有益（见 MLP-Mixer 后续工作）。

## 方法亮点

- **层级结构**：逐 stage 降低空间分辨率、增加通道维度，类似 CNN 的 pyramidal 特征层次。
- **Shifted window**：W-MSA（规则窗口）与 SW-MSA（偏移窗口）交替，以极小代价实现跨窗口连接。
- **线性复杂度**：自注意力限制在固定大小窗口内，使计算量与图像分辨率呈线性关系。

## 与现有 Wiki 的关系

- 从 [[Wiki/Sources/ViT]] 的"等距 patch"演进到层级结构，见 [[Wiki/Topics/Vision Transformer 演进]]。
- 关联实体：[[Wiki/Entities/Swin Transformer]]。
- 启发了后续大量层级 ViT 变体（PVT、CSwin 等）。

## 后续问题

- Shifted window 是否是最优的跨窗口交互方案？与其他方法（如 deformable attention）相比如何？
- Swin 在视频和 3D 任务上的扩展（Video Swin Transformer）。
