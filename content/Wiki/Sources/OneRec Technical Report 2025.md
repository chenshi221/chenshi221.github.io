---
type: source
status: processed
source_file: "[[Clippings/OneRec Technical Report]]"
title: "OneRec Technical Report"
site: "arxiv.org"
author: "OneRec Team (Kuaishou)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags:
  - "generative-recommendation"
  - "end-to-end"
  - "scaling-laws"
  - "reinforcement-learning"
  - "industrial-deployment"
aliases:
  - "OneRec 技术报告"
---

# OneRec Technical Report

## 核心结论

OneRec 是快手提出的端到端生成式推荐系统，将传统多级级联架构（召回-粗排-精排）统一为单一的 encoder-decoder 生成框架。该系统在快手主 APP 部署，处理约 25% 的 QPS，App Stay Time 提升 0.54%（快手）和 1.24%（快手极速版）。

## 关键事实

- **计算效率**：相比传统级联架构，端到端设计将 OPEX 降至仅 10.6%。训练和推理的 MFU 分别达到 23.7% 和 28.8%，接近 LLM 社区水平（传统推荐模型仅 4.6%/11.2%）。
- **Scaling Laws**：将推荐模型 FLOPs 提升 10 倍后，发现了推荐系统的 scaling laws，为资源分配提供了理论指导。
- **强化学习**：在传统级联架构中效果有限的 RL 技术，在 OneRec 框架下展现出巨大潜力，通过 reward model 引导的 RL 策略优化可实现显著提升。
- **Tokenizer**：使用 RQ-Kmeans 对多模态协同表示进行层次化语义 ID 量化，优于 RQ-VAE，得到更好的重建质量和码本利用率。
- **Encoder**：设计四条专门化嵌入通路（用户静态特征、短期行为、正向反馈、终身行为），处理不同时间尺度的用户交互模式。

## 方法与论证路径

采用 encoder-decoder 架构，encoder 压缩用户行为序列，decoder 自回归生成语义 ID。预训练阶段进行 next token prediction，后训练阶段通过 reward model 进行 RL 对齐。推理时先从语义 ID 映射回视频，可选 reward-based 筛选进一步优化结果。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/生成式推荐系统]]，[[Wiki/Concepts/生成式推荐]]，[[Wiki/Entities/OneRec 系列模型]]
- 支持：端到端生成式推荐在工业级场景中可超越传统级联架构
- 该论文是 OneRec 系列的核心技术报告，包含最全面的系统架构描述

## 后续问题

- Scaling laws 在更大规模下的外推可靠性如何？
- 与传统级联架构的最优混合策略是什么？
