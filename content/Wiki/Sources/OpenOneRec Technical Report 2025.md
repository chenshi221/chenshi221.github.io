---
type: source
status: processed
source_file: "[[Clippings/OpenOneRec Technical Report An Open Foundation Model and Benchmark to Accelerate Generative Recommendation]]"
title: "OpenOneRec Technical Report: An Open Foundation Model and Benchmark to Accelerate Generative Recommendation"
site: "arxiv.org"
author: "OneRec Team (Kuaishou)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags:
  - "generative-recommendation"
  - "foundation-model"
  - "benchmark"
  - "scaling-laws"
  - "open-source"
aliases:
  - "OpenOneRec 开源基础模型"
---

# OpenOneRec Technical Report

## 核心结论

OpenOneRec 是首个开源的推荐基础模型（1.7B/8B），同时发布 RecIF-Bench 评估基准和完整训练管线。在 Amazon 10 个数据集上平均 Recall@10 提升 26.8%。

## 关键事实

- **RecIF-Bench**：覆盖 8 个任务的 holistic 评估基准，分为四层能力层级：L0 语义对齐（Item Understanding）、L1 基础推荐（短视频/广告/商品推荐 + 标签预测）、L2 指令跟随（交互式推荐 + 标签条件推荐）、L3 推理（推荐解释）。
- **Scaling Laws**：推荐领域的参数最优值 N_opt 与计算预算 C 的 0.44 次方成正比，数据最优值 D_opt 与 C 的 0.56 次方成正比。指数关系与 NLP 领域不同（NLP: a≈0.5, b≈0.5），推荐领域 b > a，表明推荐更"数据饥渴"。
- **训练策略**：Co-Pretraining 将推荐语料和通用文本混合训练，有效缓解灾难性遗忘。通过交替 General Distillation 和 Rec-RL 的后训练策略平衡通用推理和推荐性能。
- **数据规模**：训练数据集包含 16 万用户、约 1.3 万 item captions 和对应交互。Pro 版本扩展至约 2000 万用户。
- **开源生态**：开放完整训练管线（数据处理、co-pretraining、post-training）和模型权重。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/生成式推荐系统]]，[[Wiki/Concepts/生成式推荐]]，[[Wiki/Entities/OneRec 系列模型]]
- 关联 AI 知识：Scaling Laws（与 Chinchilla 的关系）、灾难性遗忘

## 后续问题

- 推荐 scaling laws 中 b > a（数据饥渴）的根本原因是什么？
- 跨域迁移能力的理论上限在哪里？
