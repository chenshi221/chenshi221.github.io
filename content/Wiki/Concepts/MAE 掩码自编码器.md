---
type: concept
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - MAE
  - Masked Autoencoder
  - 掩码自编码器
  - MIM
tags:
  - MAE
  - 自监督学习
  - ViT
  - CV
  - 何恺明
sources:
  - '[[Wiki/Sources/MAE 掩码自编码器]]'
confidence: high
---
# MAE 掩码自编码器

## 定义

MAE（Masked Autoencoder，掩码自编码器）是一种基于重建的视觉自监督学习方法。灵感来自 NLP 的 BERT，MAE 随机掩码输入图像的大部分 patch（75%），使用编码器-解码器架构重建被掩码的像素。核心创新是非对称设计：编码器仅处理可见 patch（提升效率），轻量解码器重建全图。

## 为什么 MAE 对 CV 很重要

在 MAE 之前，CV 自监督学习的主流是**对比学习**（MoCo、SimCLR）。MAE 证明基于重建的掩码建模（masked image modeling, MIM）可以比对比学习更有效——且更简单。

关键洞察来自图像与文本的信息密度差异：
- 文本词是高信息密度的语义单元，一个词可能改变整句意思
- 图像像素是低信息密度的冗余信号，掩掉一部分像素后仍可从邻近区域推断
- 因此掩码率需要比 BERT（15%）高得多（75%），才能产生有意义的自监督任务

## 非对称架构设计

```
输入图像 (224x224)
    ↓ patchify
编码器 (ViT, 24层) → 仅处理 ~25% 可见 patch
    ↓
解码器 (轻量, 8层) → 处理 可见 latent + [MASK] token
    ↓
重建全图（像素级损失，仅在 masked patch 上计算）
```

- 编码器不处理 mask token = 训练加速 3 倍以上
- 解码器仅预训练时使用，预训练后丢弃
- 下游任务只使用编码器提取特征

## 与 ViT 的关系

MAE 是 ViT 自监督预训练的自然配对：
- ViT 证明 Transformer 可以直接处理图像（分词为 patch）
- MAE 证明 Transformer 可以通过掩码重建学习高质量视觉表示
- 结合 ViT + MAE = 不需要 CNN inductive bias 的端到端视觉学习

参见 [[Wiki/Entities/Vision Transformer (ViT)]] 和 [[Wiki/Topics/Vision Transformer 演进]]

## 扩展和影响

- **VideoMAE**：将掩码建模扩展到视频领域
- **MultiMAE**：扩展到多模态（RGB、深度、语义分割）
- **CAE**（Context Autoencoder）：在 MAE 基础上增加可见 patch 的表示学习
- **DINOv2**：Meta 的自监督视觉基础模型，在一定程度上继承了 MAE 的思路

## 来源

- [[Wiki/Sources/MAE 掩码自编码器]] — 原始论文（He et al., 2021）
- 关联：[[Wiki/Entities/Vision Transformer (ViT)]]、[[Wiki/Topics/Vision Transformer 演进]]
