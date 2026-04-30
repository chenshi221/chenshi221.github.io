---
type: source
status: processed
source_file: '[[Clippings/Masked Autoencoders Are Scalable Vision Learners]]'
title: Masked Autoencoders Are Scalable Vision Learners
site: arxiv
author: 'Kaiming He, Xinlei Chen et al. (FAIR)'
published: '2021'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - MAE
  - 自监督学习
  - ViT
  - 何恺明
  - CV
aliases:
  - MAE
---
# MAE 掩码自编码器

## 核心结论

何恺明等提出的 MAE（Masked Autoencoder）证明了掩码自编码是 CV 中的可扩展自监督学习器。核心设计：(1) 非对称编码器-解码器架构——编码器仅作用于可见 patch（不含 mask token），轻量解码器重建全图；(2) 高掩码率（75%）产生非平凡自监督任务。ViT-Huge 仅用 ImageNet-1K 即达到 87.8% top-1 准确率，迁移性能超越监督预训练。

## 关键事实

- 作者：Kaiming He、Xinlei Chen 等（FAIR），2021
- 非对称设计：编码器只处理 ~25% 可见 patch，解码器处理所有 patch（可见 latent + mask token）
- 75% 掩码率下训练加速 3 倍以上
- 轻量解码器（可用浅层小模型）在预训练后直接丢弃，仅编码器用于下游
- ViT-L 在 COCO 检测和分割上超越监督预训练

## 方法或论证路径

- 灵感来自 NLP 中的 BERT 掩码语言建模，但图像和文本本质不同（图像信息密度低、冗余高）
- 高掩码率 + 编码器不处理 mask token = 训练高效且学到有意义表示
- 解码器仅在预训练时使用，设计为轻量级（8 层 vs 编码器 24 层）
- 像素重建目标 + 仅对 masked patch 计算损失
- 在 ImageNet 分类、COCO 检测/分割等下游任务上验证迁移能力

## 与现有 Wiki 的关系

- 关联：[[Wiki/Concepts/MAE 掩码自编码器]]、[[Wiki/Topics/Vision Transformer 演进]]、[[Wiki/Entities/Vision Transformer (ViT)]]
- 补充：ViT 架构演进的关键一环——从监督 ViT 到自监督 MAE

## 可能的矛盾或待核实点

- MAE 与对比学习（MoCo、SimCLR）在不同数据规模下的优劣需结合 Chinchilla 缩放定律重新审视

## 后续问题

- MAE 预训练在视频（VideoMAE）和多模态中的扩展
- 掩码率与数据规模的关系：大数据集下是否仍需 75%？
