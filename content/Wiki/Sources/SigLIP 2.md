---
type: source
status: processed
source_file: >-
  [[Clippings/SigLIP 2 Multilingual Vision-Language Encoders with Improved
  Semantic Understanding, Localization, and Dense Features.md]]
title: >-
  SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic
  Understanding, Localization, and Dense Features
site: arXiv
author: Google DeepMind
published: 2025-02
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - siglip
  - contrastive-learning
  - vision-language
  - multilingual
aliases:
  - SigLIP 2
---
# SigLIP 2: Multilingual Vision-Language Encoders

## 核心结论

- SigLIP 2 是 SigLIP 的升级版多语言视觉-语言编码器家族，通过**统一训练配方**整合了多项独立发展的技术（captioning 预训练、自监督损失、在线数据筛选），在全部模型规模上显著超越前代。
- 关键改进包括：更强的语义理解、更好的定位能力（localization）、更稠密的视觉特征。
- 支持多分辨率变体和原始宽高比保持，并在多语言公平性上做了 debiasing 努力。

## 关键事实

- 作者来自 Google DeepMind，2025 年 2 月（arXiv:2502.14786）。
- 核心升级：
  - 训练目标：原始 SigLIP 的 sigmoid 对比损失 + captioning 预训练 + 自蒸馏 + masked prediction。
  - 在线数据筛选：动态过滤低质量图文对。
  - 多分辨率：支持多种输入分辨率和原始宽高比。
  - 多语言 debiasing：更丰富的数据混合和去偏技术。
- 模型规模：ViT-B (86M) / L (303M) / So400m (400M) / g (1B)，全部开源。
- 在零样本分类、图文检索、VLM 视觉特征提取、定位和 dense prediction 任务上全面优于 SigLIP。

## 方法亮点

- 统一训练配方（unified recipe）：将之前分离的技术整合到一起产生协同效应。
- 定位和 dense feature 的显著提升是此前 SigLIP 的明显短板。
- 多分辨率支持让模型更灵活地适配不同下游任务。

## 与现有 Wiki 的关系

- 是 [[Wiki/Concepts/多模态对比学习]] 的标志性工作之一，与 [[Wiki/Sources/CLAP]] 共享对比学习核心思想。
- 对比 CLIP 使用 softmax 损失，SigLIP 家族使用 sigmoid 损失，训练更稳定且可扩展到更大 batch。
- SigLIP 2 的视觉编码器是许多现代 VLM 的基础组件。

## 后续问题

- sigmoid vs softmax 对比损失的深度对比：是否在所有场景下 sigmoid 都更优？
- captioning 预训练的加入是否会改变模型从"对齐"到"生成"的倾向？
