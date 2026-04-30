---
type: source
status: processed
source_file: >-
  [[Clippings/An Image is Worth 16x16 Words Transformers for Image Recognition
  at Scale.md]]
title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
site: arXiv
author: Alexey Dosovitskiy et al. (Google Research)
published: 2020-10
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - vision-transformer
  - image-classification
  - self-attention
aliases:
  - ViT
---
# ViT: An Image is Worth 16x16 Words

## 核心结论

- ViT 首次证明了 **纯 Transformer 架构可以直接应用于图像分类**，不需要依赖 CNN 的归纳偏置（inductive bias）。
- 将图像切分为固定大小的 patch（如 16x16），作为 token 序列输入标准 Transformer encoder，与 NLP 中处理词元的方式完全一致。
- 在大规模预训练（JFT-300M）条件下，ViT 在多个图像识别 benchmark 上达到或超越当时 SOTA CNN，且训练计算资源显著更少。

## 关键事实

- 作者：Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov 等（Google Research, Brain Team）。
- 发表于 ICLR 2021（arXiv:2010.11929）。
- 核心设计：图像 → patch embedding + position embedding → Transformer encoder → MLP head 分类。
- 关键发现：ViT 在小数据集上不如 CNN（缺乏归纳偏置），但数据量足够大时（ImageNet-21k 或 JFT-300M 预训练）可超越 CNN。
- 代表性结果：ViT-H/14 在 ImageNet 上达 88.55% top-1 accuracy。

## 方法亮点

- 将 NLP 的 Transformer 标准架构几乎零修改用于视觉。
- 使用 2D position embedding 保留空间结构信息。
- 大规模预训练 + 下游微调的范式直接平移自 NLP（BERT/GPT 路线）。

## 与现有 Wiki 的关系

- 为 [[Wiki/Topics/Vision Transformer 演进]] 的奠基节点。
- 关联实体：[[Wiki/Entities/Vision Transformer (ViT)]]。
- 开启了"CNN-free"视觉架构的研究方向，后续引出 [[Wiki/Sources/Swin Transformer]]、[[Wiki/Sources/MLP-Mixer]] 等工作。

## 后续问题

- ViT 在 dense prediction 任务（检测、分割）上表现如何？Swin 对此做了专门改进。
- patch-based 方法是否丢失了细粒度空间信息？
