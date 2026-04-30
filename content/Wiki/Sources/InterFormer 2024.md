---
type: source
status: processed
source_file: "[[Clippings/InterFormer Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction]]"
title: "InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction"
site: "arxiv.org"
author: "Zhichen Zeng et al. (UIUC, Meta AI)"
published: "2024"
processed: "2026-04-30"
updated: "2026-04-30"
tags:
  - "CTR-prediction"
  - "feature-interaction"
  - "sequence-modeling"
  - "heterogeneous-interaction"
  - "bidirectional"
aliases:
  - "InterFormer 异构交互学习"
---

# InterFormer: Effective Heterogeneous Interaction Learning

## 核心结论

InterFormer 是 Meta 提出的异构信息交互学习框架，针对 CTR 预估中序列特征与非序列特征的融合问题，通过双向信息流和交错式交互设计，在 Meta Ads 多平台部署实现 0.15% NE 增益和 24% QPS 提升。

## 关键事实

- **两大瓶颈**：(1) 单向信息流：传统方法仅用非序列信息指导序列建模，反向的序列到非序列信息流被忽略；(2) 过早信息聚合：早期汇总（求和、池化、拼接）导致过多信息丢失。
- **三层架构**：
  - Interaction Arch：对非序列特征和序列汇总进行特征交互，学习 behavior-aware 非序列表示。
  - Sequence Arch：通过 Personalized FFN (PFFN) 用非序列汇总指导序列变换，再经 MHA 建模序列依赖，学习 context-aware 序列表示。
  - Cross Arch：连接 Interaction 和 Sequence Arch，实现双向信息交换和选择性信息聚合。
- **PFFN**：用 non-sequence summarization 作为 query，通过 MLP 学习序列的线性投影参数，实现个性化序列变换。
- **兼容性**：Interaction Arch 可灵活替换为 DCNv2、DHEN 等不同的交互模块。
- **生产部署**：在 Meta Ads 多个平台部署，相比 SOTA 模型实现 0.15% NE 增益和 24% QPS 增益。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/生成式推荐系统]]，[[Wiki/Concepts/CTR 预估]]
- InterFormer 是 CTR 预估领域首次明确解决双向异构交互问题的工作

## 后续问题

- 双向交互在更多异构模态（如图像、视频）下的扩展性？
- PFFN 的个性化投影与标准 attention 相比的计算开销？
