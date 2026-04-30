---
type: source
status: processed
source_file: "[[Clippings/OneTrans Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender]]"
title: "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender"
site: "arxiv.org"
author: "Zhaoqi Zhang et al. (ByteDance, NTU)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags:
  - "CTR-prediction"
  - "ranking-model"
  - "feature-interaction"
  - "sequence-modeling"
  - "transformer"
  - "unified-architecture"
aliases:
  - "OneTrans 统一 Transformer"
---

# OneTrans: Unified Feature Interaction and Sequence Modeling

## 核心结论

OneTrans 是字节跳动提出的统一 Transformer 排序模型，用单一 backbone 同时完成用户行为序列建模和特征交互，替代传统的"先编码序列、再交互特征"分离式 pipeline。在线 A/B 测试实现 5.68% GMV/u 提升。

## 关键事实

- **Unified Tokenizer**：将 sequential 特征（用户行为序列）转为 S-tokens，non-sequential 特征（用户/物品/上下文特征）转为 NS-tokens，拼接为统一 token 序列输入 Transformer。
- **Mixed Parameterization**：S-tokens 共享一套 Q/K/V 和 FFN 参数；NS-tokens 各自拥有 token-specific 参数，适应异构特征的语义差异。
- **Pyramid Stack**：利用 causal attention 的信息向尾部集中的特性，逐层剪枝 S-tokens 的 query 范围，减少序列长度以降低计算量。
- **Cross-Request KV Caching**：将用户侧计算在不同候选 item 间复用，推理复杂度从 O(C) 降至 O(1)。
- **Scaling**：在工业数据上展示了 near log-linear 的性能增长与模型规模的关系，验证了排序模型的 scaling law。

## 方法与论证路径

用 causal Transformer 替代传统 encode-then-interaction pipeline。NS-tokens 可通过 Group-wise 或 Auto-Split 两种 tokenizer 生成。统一 causal mask 使 NS-tokens 可以 attend 全部 S-tokens，实现序列-特征双向交互。继承 FlashAttention、混合精度训练等 LLM 优化技术。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/生成式推荐系统]]，[[Wiki/Concepts/CTR 预估]]
- 与生成式推荐路线不同，OneTrans 仍是排序模型，属于传统推荐范式的演进

## 后续问题

- S-tokens 共享参数 vs 全 token-specific 的 trade-off 边界在哪里？
- Pyramid 策略的剪枝率与性能损失关系？
