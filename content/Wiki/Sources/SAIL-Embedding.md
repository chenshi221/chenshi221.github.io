---
type: source
status: processed
source_file: >-
  [[Clippings/SAIL-Embedding Technical Report Omni-modal Embedding Foundation
  Model]]
title: 'SAIL-Embedding: Omni-modal Embedding Foundation Model'
site: arXiv 2510.12709
author: ByteDance
published: 2025-11
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - embedding
  - omni-modal
  - recommendation
  - industry
aliases:
  - SAIL-Embedding 论文
---
# SAIL-Embedding: Omni-modal Embedding Foundation Model

## 核心结论

- 提出 SAIL-Embedding，全模态（omni-modal）Embedding 基础模型，支持视觉、文本、音频的任意组合输入。
- 面向字节跳动抖音真实业务场景，在冷启动推荐、召回、粗排、精排全链路取得显著线上收益。

## 关键事实

- 来源：字节跳动（ByteDance），2024。
- 训练数据规模：超过 100 亿样本，包括 Item-to-Item（Copair/Search/Live 共 7 类）和 Query-to-Item 检索对。
- 核心创新：(1) 动态难负样本挖掘（自适应阈值）；(2) 自适应多源数据平衡（从数据分布学习权重替代手工调参）；(3) 多阶段训练：Content-aware 渐进式训练 + Collaboration-aware 推荐增强训练。
- 推荐增强训练：通过 Sequence-to-Item 和 ID-to-Item 蒸馏融入用户行为信号。
- 线上收益：抖音精选 7 天 LT +0.5%，抖音 Feed 排序模型 AUC +0.1%。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/多模态 Embedding 与检索]]
- 关联：[[Wiki/Concepts/多模态 Embedding 模型]]
- 关联：[[Wiki/Comparisons/多模态 Embedding 模型比较]]

## 后续问题

- 全模态 Embedding 中音频模态的实际贡献度如何量化？
- 线上系统的实时更新机制如何设计？
