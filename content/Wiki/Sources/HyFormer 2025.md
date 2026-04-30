---
type: source
status: processed
source_file: "[[Clippings/HyFormer Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction]]"
title: "HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction"
site: "arxiv.org"
author: "Yunwen Huang et al. (ByteDance)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags:
  - "CTR-prediction"
  - "feature-interaction"
  - "sequence-modeling"
  - "unified-architecture"
  - "hybrid-transformer"
aliases:
  - "HyFormer 混合 Transformer"
---

# HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction

## 核心结论

HyFormer 是字节跳动提出的统一混合 Transformer 架构，通过交替执行 Query Decoding 和 Query Boosting，将长序列建模和异构特征交互紧密集成到单一 backbone 中。已在字节跳动全量部署，服务数十亿用户。

## 关键事实

- **核心洞察**：传统"长序列建模、再做特征交互"的分离式 pipeline 存在单向、滞后融合的限制。HyFormer 重新思考两个模块的角色：Query Decoding 负责"序列建模"，Query Boosting 负责"特征交互"，两者交替迭代。
- **Query Decoding**：将 non-sequential 特征扩展为多个 Global Token（语义 query），通过 cross-attention 对长行为序列的逐层 K/V 表示进行解码，使全局信息直接塑造序列表示。
- **Query Boosting**：在 Query Decoding 之后，通过 MLP-Mixer 风格的 token mixing 增强 decoded queries 和 non-sequence tokens 之间的交互，逐步丰富语义表示。
- **三种序列编码策略**：Full Transformer Encoding（最高容量）、LONGER-style Efficient Encoding（平衡效率）、Decoder-style Lightweight Encoding（最低延迟），支持灵活部署。
- **Scaling**：在十亿级工业数据集上，同参数和 FLOPs 预算下全面超越 LONGER 和 RankMixer，且展现场景扩展行为。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/生成式推荐系统]]，[[Wiki/Concepts/CTR 预估]]
- HyFormer 在架构思路上是 OneTrans 的进一步演进，强调交替而非同时交互

## 后续问题

- Query Decoding 和 Query Boosting 的最优交替比例如何确定？
- 与 OneTrans 的 Unified Causal Attention 相比，交替设计的优劣势？
