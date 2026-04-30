---
type: source
status: processed
source_file: '[[Clippings/MLP-Mixer An all-MLP Architecture for Vision.md]]'
title: 'MLP-Mixer: An all-MLP Architecture for Vision'
site: arXiv
author: Ilya Tolstikhin et al. (Google Research)
published: 2021-05
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - mlp-mixer
  - vision-architecture
  - inductive-bias
aliases:
  - MLP-Mixer
---
# MLP-Mixer: An all-MLP Architecture for Vision

## 核心结论

- MLP-Mixer 提出了一个**仅使用多层感知机（MLP）的视觉架构**，证明了在足够数据规模下，卷积和注意力都**不是必需的**。
- 核心思想：将图像 patch 通过两类 MLP 层交替处理——channel-mixing MLP（跨通道信息交互）和 token-mixing MLP（跨空间位置信息交互）。
- 在大规模数据集上训练时，MLP-Mixer 能达到与 CNN 和 ViT 可比的分类性能，且预训练和推理成本相近。

## 关键事实

- 作者：Ilya Tolstikhin, Neil Houlsby 等（Google Research, Brain Team）。
- 发表于 NeurIPS 2021（arXiv:2105.01601）。
- 架构：Per-patch Fully-connected → 多个 Mixer Layer（Token-mixing MLP + Channel-mixing MLP，各带 skip-connection 和 LayerNorm）→ Global Average Pooling → 分类头。
- Token-mixing MLP 在 patch 维度上操作（跨空间），Channel-mixing MLP 在通道维度上操作（逐 patch 独立），两者都是对矩阵转置后应用相同的 MLP 操作。
- 代表性结果：在 ImageNet-21k 预训练 + ImageNet 微调下，Mixer-H/14 达 87.94% top-1（与 ViT-H/14 的 88.55% 接近）。

## 方法亮点

- 极简架构：没有卷积、没有自注意力，仅由 MLP + 残差连接 + 归一化层组成。
- 计算复杂度与 ViT 同阶（O(N²) 来自 token-mixing MLP 的矩阵乘法）。
- 这一工作具有概念上的挑衅性："如果 MLP 就够了，那视觉归纳偏置（局部性、平移等变性）到底有多重要？"

## 与现有 Wiki 的关系

- 与 [[Wiki/Sources/ViT]]、[[Wiki/Sources/Swin Transformer]] 共同构成 [[Wiki/Topics/Vision Transformer 演进]]。
- 被视为 "ViT → Swin → MLP-Mixer" 演进路线中的概念极限测试。
- 推动了关于"视觉归纳偏置是否必要"的深层讨论。

## 后续问题

- MLP-Mixer 在 dense prediction 任务上的表现远不如 Swin，说明纯 MLP 缺乏处理多尺度特征的能力。
- gMLP、ResMLP 等后续变体如何改进 token-mixing 机制？
