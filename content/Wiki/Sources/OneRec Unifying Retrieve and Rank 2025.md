---
type: source
status: processed
source_file: "[[Clippings/OneRec Unifying Retrieve and Rank with Generative Recommender and Preference Alignment]]"
title: "OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment"
site: "arxiv.org"
author: "Jiaxin Deng et al. (Kuaishou)"
published: "2025"
processed: "2026-04-30"
updated: "2026-04-30"
tags:
  - "generative-recommendation"
  - "session-wise-generation"
  - "DPO"
  - "preference-alignment"
  - "MoE"
aliases:
  - "OneRec 统一召回排序"
---

# OneRec: Unifying Retrieve and Rank

## 核心结论

该论文提出 OneRec 的早期版本，是首个在真实工业场景中显著超越传统级联推荐系统的端到端生成式模型。在快手部署后实现 1.6% 的 watch-time 提升。

## 关键事实

- **Session-wise 生成**：区别于传统 point-wise 的 next item 预测，OneRec 一次性生成整个 session（5-10 个视频），让模型自主学习 session 内部结构和多样性。
- **MoE 扩展**：在 decoder 的 FFN 层采用稀疏 Mixture-of-Experts，在不按比例增加 FLOPs 的前提下扩展模型容量。
- **Iterative Preference Alignment (IPA)**：设计 reward model 模拟用户偏好，从 beam search 结果中选择 best-chosen 和 worst-rejected 样本构建 preference pair，通过 DPO 进行迭代偏好对齐。
- **Self-hard Negatives**：不使用随机负采样，而是从模型自己的 beam search 结果中选取低分样本作为 rejected，提高了 DPO 训练效率。
- **Balanced K-Means**：采用平衡 K-means 聚类构建语义 ID 码本，解决 RQ-VAE 的"沙漏现象"（码本不平衡）。

## 方法与论证路径

模型采用 T5 风格的 encoder-decoder 架构。Encoder 处理用户正向行为序列，Decoder 自回归生成 session 中的视频语义 ID 列表。训练分为两阶段：(1) session-wise 生成预训练；(2) IPA 阶段利用 DPO 进行偏好对齐。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/生成式推荐系统]]，[[Wiki/Concepts/生成式推荐]]，[[Wiki/Entities/OneRec 系列模型]]
- 这是 OneRec 系列的奠基性论文，首次实现了召回和排序的统一

## 可能的矛盾或待核实点

- Session-wise 生成 vs 后续版本的优化方向（V2 转向 lazy decoder-only）
- DPO 训练样本量极有限的场景下是否仍有效
