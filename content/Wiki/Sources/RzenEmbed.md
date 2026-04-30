---
type: source
status: processed
source_file: '[[Clippings/RzenEmbed Towards Comprehensive Multimodal Retrieval]]'
title: 'RzenEmbed: Towards Comprehensive Multimodal Retrieval'
site: arXiv 2510.27350
author: Weijian Jian et al. (360 AI Research)
published: '2024'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - embedding
  - multimodal
  - video-retrieval
  - document-retrieval
aliases:
  - RzenEmbed 论文
---
# RzenEmbed: Towards Comprehensive Multimodal Retrieval

## 核心结论

- 提出 RzenEmbed，支持文本、图像、视频、视觉文档（visual documents）四种模态的统一 Embedding 框架。
- 两阶段训练策略，第二阶段引入改进的 InfoNCE loss：hardness-weighted 机制 + 假负样本（false negative）消除。

## 关键事实

- 来源：360 AI Research（奇虎 360），2024。
- 基座模型：Qwen2-VL，利用其原生动态分辨率（Native Dynamic Resolution）和 M-RoPE 处理视频时序。
- 关键创新：(1) Hardness-weighted 机制：为困难样本分配更高权重；(2) 假负样本消除：识别并排除语义相似但被误标为负的样本；(3) 可学习温度参数：不同任务自动调整相似度分布锐度；(4) Model Souping 提高稳定性。
- 在 MMEB 和 MMEB-V2 上取得 SOTA，尤其在视频和视觉文档检索任务大幅领先。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/多模态 Embedding 与检索]]
- 关联：[[Wiki/Concepts/多模态 Embedding 模型]]
- 关联：[[Wiki/Comparisons/多模态 Embedding 模型比较]]

## 后续问题

- 假负样本消除策略的泛化性如何？不同领域的语义相似度阈值可能不同。
