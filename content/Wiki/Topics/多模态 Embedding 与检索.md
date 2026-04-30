---
type: topic
status: active
created: '2026-04-30'
updated: '2026-04-30'
aliases:
  - Embedding 主题
tags:
  - embedding
  - multimodal
  - retrieval
sources:
  - Wiki/Sources/Magic-MM-Embedding
  - Wiki/Sources/ObjEmbed
  - Wiki/Sources/RzenEmbed
  - Wiki/Sources/SAIL-Embedding
---
# 多模态 Embedding 与检索

## 概述

多模态 Embedding 指将文本、图像、视频、音频等异构数据映射到统一语义空间的表示学习技术。近年来从 CLIP 双塔架构向 MLLM 基的统一 Embedding 范式演进，支持指令引导的跨模态检索。本主题涵盖 4 个代表性工作，覆盖效率优化、物体级表示、多模态扩展和工业级落地。

## 核心来源

- [[Wiki/Sources/Magic-MM-Embedding]]：视觉 token 压缩实现高效 MLLM Embedding，75% token 削减 + SOTA 性能。
- [[Wiki/Sources/ObjEmbed]]：物体级 Embedding，语义+IoU 双 embedding，支持检测/指代/局部检索。
- [[Wiki/Sources/RzenEmbed]]：文本+图像+视频+文档四模态统一 Embedding，hardness-weighted loss + false negative 消除。
- [[Wiki/Sources/SAIL-Embedding]]：全模态（视觉+文本+音频）工业级 Embedding，字节跳动抖音线上部署。

## 关键概念

- [[Wiki/Concepts/多模态 Embedding 模型]]：CLIP 双塔 vs MLLM 统一架构的范式对比。
- 对比学习核心技术：InfoNCE loss、难负样本挖掘（hard negative mining）、假负样本消除。
- MMEB benchmark：多模态 Embedding 的标准评测基准。
- Token 效率：MLLM Embedding 中视觉 token 压缩的必要性与方法。

## 技术趋势

1. 从 CLIP 双塔到 MLLM 统一编码：更强的跨模态交互和指令遵循能力。
2. 从图像级到物体级：ObjEmbed 代表的细粒度表示。
3. 从双模态到全模态：SAIL-Embedding 的视觉+文本+音频。
4. 从学术 benchmark 到工业落地：SAIL-Embedding 的线上推荐系统部署。

## 开放问题

- 如何平衡 MLLM Embedding 的表达力与推理成本？
- 全模态 Embedding 的模态缺失（modality-missing）问题？
- 动态更新的 Embedding 如何保持一致性？
