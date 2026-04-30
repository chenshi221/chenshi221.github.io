---
type: source
status: processed
source_file: >-
  [[Clippings/Magic-MM-Embedding Towards Visual-Token-Efficient Universal
  Multimodal Embedding with MLLMs]]
title: Magic-MM-Embedding
site: arXiv 2602.05275
author: Qi Li et al. (Honor)
published: '2024'
processed: '2026-04-30'
updated: '2026-04-30'
tags:
  - embedding
  - multimodal
  - MLLM
  - token-compression
aliases:
  - Magic-MM-Embedding 论文
---
# Magic-MM-Embedding: Visual-Token-Efficient Universal Multimodal Embedding

## 核心结论

- 提出 Magic-MM-Embedding，解决 MLLM Embedding 模型因大量视觉 token 导致推理成本高的问题。
- 通过参数无关的视觉 token 压缩（双线性插值下采样 75%）大幅降低延迟和显存，同时保持 SOTA 性能。

## 关键事实

- 来源：Honor Device Co., Ltd，2024。
- 核心挑战：标准 MLLM（如 LLaVA-1.5 的 576 个 visual tokens）在检索任务中存在严重冗余，计算成本与 token 数成二次关系。
- 三阶段渐进训练：(1) 多模态基础能力恢复（通用指令微调）；(2) 多模态对比预训练（16M 样本 + 自精炼难负样本挖掘）；(3) 任务感知微调（MLLM-as-Judge 构造高质量难负样本）。
- 基座模型：InternVL3-VTC。
- 在 MMEB 35 任务上以 1/4 视觉 token 取得 SOTA。

## 与现有 Wiki 的关系

- 关联：[[Wiki/Topics/多模态 Embedding 与检索]]
- 关联：[[Wiki/Concepts/多模态 Embedding 模型]]
- 关联：[[Wiki/Comparisons/多模态 Embedding 模型比较]]

## 后续问题

- 75% 压缩率是否对所有类型的图像（如密集文字文档）都适用？
