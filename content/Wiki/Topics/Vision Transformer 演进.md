---
type: topic
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Vision 基础模型演进
tags:
  - vision-transformer
  - architecture-evolution
  - foundation-model
sources:
  - '[[Wiki/Sources/ViT]]'
  - '[[Wiki/Sources/Swin Transformer]]'
  - '[[Wiki/Sources/MLP-Mixer]]'
confidence: high
---
# Vision Transformer 演进

## 概述

从 CNN 主导的视觉识别到 Transformer 全面介入，Vision 基础模型的架构演进经历了三个标志性节点：**ViT 打破范式 → Swin 实现通用 backbone → MLP-Mixer 质问归纳偏置的必要性**。

## 演进而貌

### 1. ViT（2020）：Transformer 进入视觉

- [[Wiki/Sources/ViT]] 首次证明：纯 Transformer（无需 CNN）可直接处理图像 patch 序列并取得有竞争力的分类性能。
- 核心思想：图像 = 16x16 patch 序列，与 NLP 中 token 序列的处理方式完全一致。
- 局限：需要大规模预训练（JFT-300M）才能超越 CNN；在小数据上不如 CNN（因为缺乏平移等变性和局部性等视觉归纳偏置）；固定的 16x16 patch 分辨率不适应多尺度视觉任务。
- 更大意义：证明了"CNN 不是必需的"，打开了视觉架构设计的全新空间。

### 2. Swin Transformer（2021）：层级化与通用 Backbone

- [[Wiki/Sources/Swin Transformer]] 解决了 ViT 的两个核心局限：(1) 引入层级化结构（金字塔特征层次），适配检测和分割等 dense prediction 任务；(2) shifted window 自注意力将计算复杂度从 O(N²) 降至 O(N)。
- 成为首个广泛使用、可与 ResNet 竞争通用 backbone 地位的 Transformer 模型。
- 获 ICCV 2021 最佳论文奖（Marr Prize）。
- 启发了大量层级 ViT 变体（PVT、CSwin、CrossViT 等）。

### 3. MLP-Mixer（2021）：归纳偏置的极限测试

- [[Wiki/Sources/MLP-Mixer]] 提出一个"挑衅性"问题：如果纯 MLP 也够用，那视觉归纳偏置（局部性、平移等变性、自注意力）到底多重要？
- 仅使用两类 MLP（token-mixing + channel-mixing），不包含任何卷积或注意力，在大数据上达到与 ViT 可比的性能。
- 意义不在于作为实用 backbone（它在 dense prediction 上表现不佳），而在于概念上的极限测试——它揭示了"大规模数据 + 简单架构"可以部分替代精心设计的归纳偏置。

## 关键洞察

1. **从"必需"到"可选"**：CNN → ViT → MLP-Mixer 的递进是不断剥离归纳偏置的过程。每剥离一层，数据规模的要求就更高一层。
2. **通用性 vs 专业性**：ViT 在分类上强但在检测/分割上弱；Swin 的层级设计弥补了这一点；MLP-Mixer 则再次偏向分类任务。这说明架构的"通用性"并非免费午餐。
3. **架构收敛趋势**：2023 年后，Vision Transformer 内部架构趋于收敛（分层 + 窗口注意力），但新的讨论转向了与 LLM 的统一（如 DiT、原生多模态模型）。

## 关联

- 概念：[[Wiki/Concepts/多模态对比学习]] — ViT 的视觉编码器是 CLIP/SigLIP 的基础组件。
- 实体：[[Wiki/Entities/Vision Transformer (ViT)]]、[[Wiki/Entities/Swin Transformer]]。
- 当前前沿：ViT 作为 VLM 视觉编码器、DiT 用于扩散模型、原生多模态统一架构。
